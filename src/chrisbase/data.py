import itertools
import json
import logging
import math
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from io import IOBase
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Optional, ClassVar
from typing import Iterable, List, Tuple, Mapping

import datasets
import pandas as pd
import pymongo.collection
import pymongo.database
import pymongo.errors
import typer
from chrisbase.io import get_hostname, get_hostaddr, current_file, first_or, cwd, hr, flush_or, make_parent_dir, setup_unit_logger, setup_dual_logger, open_file, file_lines, new_path, get_http_clients, log_table, LoggingFormat, to_yaml
from chrisbase.time import now, str_delta
from chrisbase.util import tupled, SP, NO, to_dataframe
from dataclasses_json import DataClassJsonMixin
from elasticsearch import Elasticsearch
from more_itertools import ichunked
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from pymongo import MongoClient
from transformers import set_seed
from typing_extensions import Self

logger = logging.getLogger(__name__)


class AppTyper(typer.Typer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            add_completion=False,
            pretty_exceptions_enable=False,
            **kwargs)

    @staticmethod
    def run(*functions: Callable[..., Any], **kwargs) -> None:
        app = AppTyper()
        for function in functions:
            app.command(**kwargs)(function)
        app()


@contextmanager
def temporary_mutable_conf(*cfgs):
    """
    여러 OmegaConf 객체를 with 블록 동안만 수정 가능하게 만들고,
    블록이 끝나면 원래 readonly 상태를 복원한다.
    """
    original_states = [OmegaConf.is_readonly(c) for c in cfgs]

    # 모두 mutable 로 설정
    for c in cfgs:
        OmegaConf.set_readonly(c, False)

    try:
        yield  # ‼️ 단 한 번만 yield
    finally:
        # 원래 상태로 복원
        for c, state in zip(cfgs, original_states):
            OmegaConf.set_readonly(c, state)


@contextmanager
def disable_datasets_progress():
    """
    Context manager to temporarily disable the datasets progress bar.
    On entering, it calls datasets.disable_progress_bar(),
    and on exiting, it restores the progress bar with datasets.enable_progress_bar().
    """
    # Turn off the progress bar
    datasets.disable_progress_bar()
    try:
        yield
    finally:
        # Always re-enable the progress bar, even if an error occurs
        datasets.enable_progress_bar()


class NewProjectEnv(BaseModel):
    hostname: str = get_hostname()
    hostaddr: str = get_hostaddr()
    global_rank: int = Field(default=-1)
    local_rank: int = Field(default=-1)
    node_rank: int = Field(default=-1)
    world_size: int = Field(default=-1)
    time_stamp: str = Field(default=now('%m%d-%H%M%S'))
    python_path: Path = Path(sys.executable).absolute()
    current_dir: Path = Path().absolute()
    current_file: Path = Path(sys.argv[0])
    command_args: list[str] = sys.argv[1:]
    output_home: str | Path = Field(default="output")
    output_name: str | Path | None = Field(default=None)
    run_version: str | int | Path | None = Field(default=None)
    output_file: str | Path = Field(default=None)
    logging_file: str | Path = Field(default=None)
    logging_level: int = Field(default=logging.INFO)
    logging_format: LoggingFormat = Field(default=LoggingFormat.BRIEF_00)
    datetime_format: str = Field(default="[%m.%d %H:%M:%S]")
    argument_file: str | Path = Field(default=None)
    random_seed: int = Field(default=None)
    max_workers: int = Field(default=1)
    debugging: bool = Field(default=False)
    output_dir: Path | None = Field(default=None, init=False)

    @field_validator('logging_format', mode='before')
    def validate_logging_format(cls, v):
        # 만약 입력값이 문자열이라면, 해당 문자열이 LoggingFormat의 멤버 이름과 일치하는지 확인
        if isinstance(v, str):
            try:
                # 간단하게 Enum 멤버 이름으로 변환
                return LoggingFormat[v]
            except KeyError:
                # 만약 Enum 멤버 이름이 아니라면, 실제 값과 일치하는지 체크
                for member in LoggingFormat:
                    if v == member.value:
                        return member
                raise ValueError(f"Invalid logging_format: {v}.")
        return v

    @model_validator(mode='after')
    def after(self) -> Self:
        if self.output_home:
            self.output_home = Path(self.output_home)
            self.output_dir = self.output_home
            if self.output_name:
                self.output_dir = self.output_dir / self.output_name
            if self.run_version:
                self.output_dir = self.output_dir / str(self.run_version)
        self.setup_logger(self.logging_level)
        if self.random_seed:
            set_seed(self.random_seed)
            logger.info(f"Set random seed to {self.random_seed}")
        return self

    def setup_logger(self, logging_level: int = logging.INFO):
        if self.output_dir and self.logging_file:
            setup_dual_logger(
                level=logging_level,
                fmt=self.logging_format.value,
                datefmt=self.datetime_format,
                stream=sys.stdout,
                filename=self.output_dir / self.logging_file,
            )
        else:
            setup_unit_logger(
                level=logging_level,
                fmt=self.logging_format.value,
                datefmt=self.datetime_format,
                stream=sys.stdout,
            )
        return self


class TimeChecker(BaseModel):
    t1: datetime = datetime.now()
    t2: datetime = datetime.now()
    started: str | None = Field(default=None)
    settled: str | None = Field(default=None)
    elapsed: str | None = Field(default=None)

    def set_started(self):
        self.started = now()
        self.settled = None
        self.elapsed = None
        self.t1 = datetime.now()
        return self

    def set_settled(self):
        self.t2 = datetime.now()
        self.settled = now()
        self.elapsed = str_delta(self.t2 - self.t1)
        return self


class NewCommonArguments(BaseModel):
    env: NewProjectEnv = Field(default=None)
    time: TimeChecker = Field(default_factory=TimeChecker)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.__class__.__name__, "value"]
        df = pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
        ]).reset_index(drop=True)
        return df

    def info_args(self, c="-", w=137):
        log_table(logger, self.dataframe(), c=c, w=w, level=logging.INFO, tablefmt="tsv", bordered=True)
        return self

    def save_args(self, to: Path | str = None) -> Path | None:
        if self.env.output_dir and self.env.argument_file:
            args_file = to if to else self.env.output_dir / self.env.argument_file
            args_json = self.model_dump_json(indent=2)
            make_parent_dir(args_file).write_text(args_json, encoding="utf-8")
            return args_file
        else:
            return None


class NewIOArguments(NewCommonArguments):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input: "InputOption" = Field(default=None)
    output: "OutputOption" = Field(default=None)
    option: BaseModel | None = None

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.__class__.__name__, "value"]
        return pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.input, data_prefix="input", data_exclude=["file", "table", "index"]),
            to_dataframe(columns=columns, raw=self.input.file, data_prefix="input.file") if self.input.file else None,
            to_dataframe(columns=columns, raw=self.input.table, data_prefix="input.table") if self.input.table else None,
            to_dataframe(columns=columns, raw=self.input.index, data_prefix="input.index") if self.input.index else None,
            to_dataframe(columns=columns, raw=self.output, data_prefix="output", data_exclude=["file", "table", "index"]),
            to_dataframe(columns=columns, raw=self.output.file, data_prefix="output.file") if self.output.file else None,
            to_dataframe(columns=columns, raw=self.output.table, data_prefix="output.table") if self.output.table else None,
            to_dataframe(columns=columns, raw=self.output.index, data_prefix="output.index") if self.output.index else None,
            to_dataframe(columns=columns, raw=self.option, data_prefix="option") if self.option else None,
        ]).reset_index(drop=True)


@dataclass
class TypedData(DataClassJsonMixin):
    data_type: ClassVar[str] = None

    def __post_init__(self):
        TypedData.data_type = self.__class__.__name__


@dataclass
class OptionData(TypedData):
    def __post_init__(self):
        super().__post_init__()


@dataclass
class ResultData(TypedData):
    def __post_init__(self):
        super().__post_init__()


@dataclass
class ArgumentGroupData(TypedData):
    tag = None

    def __post_init__(self):
        super().__post_init__()


@dataclass
class StreamOption(OptionData):
    name: str | Path = field()
    home: str | Path = field(default=Path("."))
    user: str | None = field(default=None)
    pswd: str | None = field(default=None)
    reset: bool = field(default=False)
    required: bool = field(default=False)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)

    def __str__(self):
        if self.user:
            return f"{self.user}@{self.home}/{self.name}"
        else:
            return f"{self.home}/{self.name}"


@dataclass
class FileOption(StreamOption):
    mode: str = field(default="rb")
    encoding: str = field(default="utf-8")

    @staticmethod
    def from_path(path: str | Path, name: str | Path | None = None, mode: str = "rb", encoding: str = "utf-8", reset: bool = False, required: bool = False) -> "FileOption":
        path = Path(path)
        return FileOption(
            home=path.parent,
            name=name if name else path.name,
            mode=mode,
            encoding=encoding,
            reset=reset,
            required=required,
        )


@dataclass
class TableOption(StreamOption):
    sort: str | List[Tuple[str, int] | str] = field(default="_id")
    find: dict = field(default_factory=dict)
    timeout: int = field(default=30 * 1000)

    @staticmethod
    def from_path(path: str | Path, user: str | None = None, pswd: str | None = None,
                  sort: str | List[Tuple[str, int] | str] = "_id", find: dict | None = None, timeout: int = 30 * 1000,
                  reset: bool = False, required: bool = False) -> "TableOption":
        path = Path(path)
        return TableOption(
            home=path.parent,
            name=path.name,
            user=user,
            pswd=pswd,
            sort=sort,
            find=find if find else {},
            timeout=timeout,
            reset=reset,
            required=required,
        )


@dataclass
class IndexOption(StreamOption):
    window: int = field(default=1000)
    scroll: str = field(default="3m")
    sort: str | None = field(default=None)
    timeout: int = field(default=10)
    retrial: int = field(default=3)
    create: str | Path | None = field(default=None)
    create_args = None

    def __post_init__(self):
        super().__post_init__()
        self.create_args = {}
        if self.create:
            self.create = Path(self.create)
            if self.create.exists() and self.create.is_file():
                content = self.create.read_text()
                if content:
                    self.create_args = json.loads(content)


class Streamer:
    def __init__(self, opt: StreamOption | None):
        self.opt: StreamOption = opt

    def __enter__(self):
        if not self.opt:
            return None
        if self.open():
            if self.opt.reset:
                self.reset()
        elif self.opt.required:
            assert self.usable(), f"Could not open source: {self.opt}"
        return self

    def __exit__(self, *exc_info):
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def usable(self) -> bool:
        return False

    def open(self) -> bool:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def first_usable(*streamers: "Streamer") -> "Streamer":
        for x in streamers:
            if x is not None and x.usable():
                return x
        assert False, (f"No usable streamers among {len(streamers)}: "
                       f"{', '.join([type(x).__name__ for x in streamers])}")

    def is_file_streamer(self):
        return isinstance(self, FileStreamer)

    def is_mongo_streamer(self):
        return isinstance(self, MongoStreamer)

    def is_elastic_streamer(self):
        return isinstance(self, ElasticStreamer)

    def as_file_streamer(self) -> "FileStreamer":
        if isinstance(self, FileStreamer):
            return self
        else:
            assert False, f"Invalid type: {type(self).__name__}"

    def as_mongo_streamer(self) -> "MongoStreamer":
        if isinstance(self, MongoStreamer):
            return self
        else:
            assert False, f"Invalid type: {type(self).__name__}"

    def as_elastic_streamer(self) -> "ElasticStreamer":
        if isinstance(self, ElasticStreamer):
            return self
        else:
            assert False, f"Invalid type: {type(self).__name__}"


class FileStreamer(Streamer):
    def __init__(self, opt: FileOption | None):
        super().__init__(opt=opt)
        self.opt: FileOption = opt
        self.path: Path = self.opt.home / self.opt.name
        self.fp: IOBase | None = None

    def __exit__(self, *exc_info):
        if self.fp:
            self.fp.close()

    def __len__(self) -> int:
        if self.usable():
            self.fp.flush()
            return file_lines(self.path, encoding=self.opt.encoding)
        else:
            return -1

    def __iter__(self):
        if self.fp is not None:
            for line in self.fp:
                if "b" in self.opt.mode:
                    line = line.decode(self.opt.encoding)
                yield line.strip()

    def usable(self) -> bool:
        return self.readable() or self.writable()

    def readable(self) -> bool:
        return (
                self.fp is not None
                and "r" in self.opt.mode
                and self.fp.readable()
        )

    def writable(self) -> bool:
        return (
                self.fp is not None
                and ("w" in self.opt.mode or "a" in self.opt.mode)
                and self.fp.writable()
        )

    def open(self) -> bool:
        self.path = self.opt.home / self.opt.name
        if "w" in self.opt.mode or "a" in self.opt.mode:
            make_parent_dir(self.path).touch()
        if self.path.exists() and self.path.is_file():
            self.fp = open_file(
                self.path,
                mode=self.opt.mode,
                encoding=None if "b" in self.opt.mode else self.opt.encoding
            )
        return self.usable()

    def reset(self):
        if self.usable():
            logger.info(f"Truncate the file content: {self.opt}")
            self.fp.truncate(0)


class MongoStreamer(Streamer):
    def __init__(self, opt: TableOption | None):
        super().__init__(opt=opt)
        self.opt: TableOption = opt
        self.cli: MongoClient | None = None
        self.db: pymongo.database.Database | None = None
        self.table: pymongo.collection.Collection | None = None
        self.server_info: dict | None = None

    def __exit__(self, *exc_info):
        if self.cli:
            self.cli.close()

    def __len__(self) -> int:
        if self.usable():
            return self.count(self.opt.find)
        else:
            return -1

    def __iter__(self):
        if self.usable():
            for row in self.table.find(self.opt.find).sort(self.opt.sort):
                yield row

    def usable(self) -> bool:
        try:
            res = self.db.command("ping")
        except pymongo.errors.ServerSelectionTimeoutError:
            res = {"ok": 0, "exception": "ServerSelectionTimeoutError"}
        return res.get("ok", 0) > 0 and self.table is not None

    def open(self) -> bool:
        assert len(self.opt.home.parts) >= 2, f"Invalid MongoDB host: {self.opt.home}"
        db_addr, db_name = self.opt.home.parts[:2]
        self.cli = MongoClient(f"mongodb://{db_addr}/?timeoutMS={self.opt.timeout}")
        try:
            self.server_info = self.cli.server_info()
        except pymongo.errors.ServerSelectionTimeoutError:
            raise AssertionError(f"Could not connect to MongoDB: {self.opt.home}")
        self.db = self.cli.get_database(db_name)
        self.table = self.db.get_collection(f"{self.opt.name}")
        return self.usable()

    def reset(self):
        if self.usable():
            logger.info(f"Drop an existing table: {self.opt}")
            self.db.drop_collection(f"{self.opt.name}")

    def count(self, query: Mapping[str, Any] = None) -> int:
        if self.usable():
            if not query:
                return self.table.estimated_document_count()
            else:
                return self.table.count_documents(query)
        else:
            return -1


class ElasticStreamer(Streamer):
    def __init__(self, opt: IndexOption | None):
        super().__init__(opt=opt)
        self.opt: IndexOption = opt
        self.cli: Elasticsearch | None = None

    def __exit__(self, *exc_info):
        if self.cli:
            self.cli.close()

    def __len__(self) -> int:
        if self.usable():
            self.refresh()
            res = self.cli.cat.count(index=self.opt.name, format="json")
            if res.meta.status == 200 and len(res.body) > 0 and "count" in res.body[0]:
                return int(res.body[0]["count"])
        return -1

    def __iter__(self):
        if self.usable():
            self.refresh()
            res = self.cli.search(
                index=self.opt.name,
                size=self.opt.window,
                scroll=self.opt.scroll,
                sort=self.opt.sort
            )
            while res.meta.status == 200 and len(res.body['hits']['hits']):
                for item in res.body['hits']['hits']:
                    yield item["_source"]
                res = self.cli.scroll(scroll_id=res['_scroll_id'], scroll=self.opt.scroll)

    def open(self) -> bool:
        self.cli = Elasticsearch(
            hosts=f"http://{self.opt.home}",
            basic_auth=(self.opt.user, self.opt.pswd),
            request_timeout=self.opt.timeout,
            retry_on_timeout=self.opt.retrial > 0,
            max_retries=self.opt.retrial,
        )
        return self.usable()

    def usable(self) -> bool:
        return self.cli and self.cli.ping()

    def reset(self):
        if self.cli.indices.exists(index=self.opt.name):
            logger.info(f"Drop an existing index: {self.opt}")
            self.cli.indices.delete(index=self.opt.name)
        self.cli.indices.create(index=self.opt.name, **self.opt.create_args)
        logger.info(f"Created a new index: {self.opt}")
        logger.info(f"- option: keys={list(self.opt.create_args.keys())}")

    def refresh(self, only_opt: bool = True):
        self.cli.indices.refresh(index=self.opt.name if only_opt else None)

    def status(self, only_opt: bool = True):
        if self.usable():
            self.refresh()
            res = self.cli.cat.indices(index=self.opt.name if only_opt else None, v=True)
            if res.meta.status == 200:
                logger.info(hr('-'))
                for line in res.body.strip().splitlines():
                    logger.info(line)
                logger.info(hr('-'))


@dataclass
class OutputOption(OptionData):
    file: FileOption | None = field(default=None)
    table: TableOption | None = field(default=None)
    index: IndexOption | None = field(default=None)


@dataclass
class InputOption(OutputOption):
    start: int = field(default=0)
    limit: int = field(default=-1)
    batch: int = field(default=1)
    inter: int = field(default=10000)
    # total: int = field(default=-1)
    data: Iterable | None = field(default=None)
    file: FileOption | None = field(default=None)
    table: TableOption | None = field(default=None)
    index: IndexOption | None = field(default=None)

    @dataclass
    class InputItems:
        num_item: int

        @property
        def items(self):
            if isinstance(self, InputOption.SingleItems):
                return self.singles
            elif isinstance(self, InputOption.BatchItems):
                return self.batches
            else:
                assert False, f"Invalid type: {type(self).__name__}"

        def has_single_items(self):
            return isinstance(self, InputOption.SingleItems)

        def has_batch_items(self):
            return isinstance(self, InputOption.BatchItems)

        def as_single_items(self) -> "InputOption.SingleItems":
            if isinstance(self, InputOption.SingleItems):
                return self
            else:
                assert False, f"Invalid type: {type(self).__name__}"

        def as_batch_items(self) -> "InputOption.BatchItems":
            if isinstance(self, InputOption.BatchItems):
                return self
            else:
                assert False, f"Invalid type: {type(self).__name__}"

    @dataclass
    class SingleItems(InputItems):
        singles: Iterable[Any]

    @dataclass
    class BatchItems(InputItems):
        batches: Iterable[Iterable[Any]]

    def ready_inputs(self, inputs: Iterable, total: int = None, str_to_dict: bool = False) -> "InputOption.SingleItems | InputOption.BatchItems":
        if str_to_dict:
            inputs = map(self.safe_dict, inputs)
        if total and total > 0:
            self.total = total
        num_item = max(0, self.total)
        assert num_item > 0, f"Invalid total: num_item={num_item}, total={total}, self.total={self.total}"
        if self.start > 0:
            inputs = islice(inputs, self.start, self.total)
            num_item = max(0, min(num_item, num_item - self.start))
        if self.limit > 0:
            inputs = islice(inputs, self.limit)
            num_item = min(num_item, self.limit)
        if self.batch <= 1:
            return InputOption.SingleItems(
                num_item=num_item,
                singles=inputs,
            )
        else:
            return InputOption.BatchItems(
                num_item=math.ceil(num_item / self.batch),
                batches=ichunked(inputs, self.batch),
            )

    @staticmethod
    def safe_dict(x: str | dict) -> dict:
        if isinstance(x, dict):
            return x
        else:
            return json.loads(x) if x.strip().startswith('{') else {}


@dataclass
class ProjectEnv(TypedData):
    project: str = field()
    job_name: str = field(default=None)
    job_version: int = field(default=None)
    hostname: str = field(init=False)
    hostaddr: str = field(init=False)
    time_stamp: str = now('%m%d-%H%M%S')
    python_path: Path = field(init=False)
    current_dir: Path = field(init=False)
    current_file: Path = field(init=False)
    working_dir: Path = field(init=False)
    command_args: List[str] = field(init=False)
    max_workers: int = field(default=1)
    calling_sec: float = field(default=0.001)
    waiting_sec: float = field(default=300.0)
    debugging: bool = field(default=False)
    logging_home: str | Path | None = field(default=None)
    logging_file: str | Path | None = field(default=None)
    argument_file: str | Path | None = field(default=None)
    date_format: str = field(default="[%m.%d %H:%M:%S]")
    message_level: int = field(default=logging.INFO)
    message_format: str = field(default=logging.BASIC_FORMAT)
    http_clients = get_http_clients()

    def __post_init__(self):
        self.hostname = get_hostname()
        self.hostaddr = get_hostaddr()
        self.python_path = Path(sys.executable).absolute()
        self.current_dir = Path().absolute()
        self.current_file = current_file().absolute()
        project_path_candidates = [self.current_dir] + list(self.current_dir.parents) + list(self.current_file.parents)
        self.project_path = first_or([x for x in project_path_candidates if x.name.startswith(self.project)]) if self.project else None
        self.working_dir = cwd(self.project_path)
        self.command_args = sys.argv[1:]
        self.logging_home = Path(self.logging_home).absolute() if self.logging_home else None
        self.logging_file = new_path(self.logging_file, post=self.time_stamp) if self.logging_file else None
        self.argument_file = new_path(self.argument_file, post=self.time_stamp) if self.argument_file else None
        self._setup_logger()

    def info_env(self, c="-", w=137):
        log_table(logger, to_dataframe(self), c=c, w=w, level=logging.INFO, tablefmt="tsv", bordered=True)
        return self

    def _setup_logger(self):
        if self.logging_home and self.logging_file:
            setup_dual_logger(level=self.message_level, fmt=self.message_format, datefmt=self.date_format, stream=sys.stdout, filename=self.logging_home / self.logging_file)
        else:
            setup_unit_logger(level=self.message_level, fmt=self.message_format, datefmt=self.date_format, stream=sys.stdout)
        return self

    def set_job_name(self, name: str = None):
        self.job_name = name
        return self

    def set_logging_home(self, output_home: str | Path | None, absolute: bool = False):
        if absolute:
            self.logging_home = Path(output_home).absolute() if output_home else None
        else:
            self.logging_home = Path(output_home) if output_home else None
        self._setup_logger()

    def set_logging_file(self, logging_file: str | Path | None):
        self.logging_file = Path(logging_file) if logging_file else None
        self._setup_logger()

    def set_argument_file(self, argument_file: str | Path | None):
        self.argument_file = Path(argument_file) if argument_file else None


@dataclass
class CommonArguments(ArgumentGroupData):
    tag = None
    env: ProjectEnv = field()
    time = TimeChecker()

    def __post_init__(self):
        super().__post_init__()

    def save_args(self, to: Path | str = None) -> Path | None:
        if self.env.logging_home and self.env.argument_file:
            args_file = to if to else self.env.logging_home / self.env.argument_file
            args_json = self.to_json(default=str, ensure_ascii=False, indent=2)
            make_parent_dir(args_file).write_text(args_json, encoding="utf-8")
            return args_file
        else:
            return None

    def info_args(self, c="-", w=137):
        log_table(logger, self.dataframe(), c=c, w=w, level=logging.INFO, tablefmt="tsv", bordered=True)
        return self

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        df = pd.concat([
            to_dataframe(columns=columns, raw={"tag": self.tag}),
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
        ]).reset_index(drop=True)
        return df


@dataclass
class IOArguments(CommonArguments):
    input: InputOption = field()
    output: OutputOption = field()
    option: BaseModel | None = None

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.input, data_prefix="input", data_exclude=["file", "table", "index"]),
            to_dataframe(columns=columns, raw=self.input.file, data_prefix="input.file") if self.input.file else None,
            to_dataframe(columns=columns, raw=self.input.table, data_prefix="input.table") if self.input.table else None,
            to_dataframe(columns=columns, raw=self.input.index, data_prefix="input.index") if self.input.index else None,
            to_dataframe(columns=columns, raw=self.output, data_prefix="output", data_exclude=["file", "table", "index"]),
            to_dataframe(columns=columns, raw=self.output.file, data_prefix="output.file") if self.output.file else None,
            to_dataframe(columns=columns, raw=self.output.table, data_prefix="output.table") if self.output.table else None,
            to_dataframe(columns=columns, raw=self.output.index, data_prefix="output.index") if self.output.index else None,
            to_dataframe(columns=columns, raw=self.option, data_prefix="option") if self.option else None,
        ]).reset_index(drop=True)


@dataclass
class Counter:
    step: int = 1
    _incs = itertools.count()
    _base = itertools.count()

    def __str__(self):
        return f"Counter(val={self.val()}, step={self.step})"

    def __repr__(self):
        return f"Counter(val={self.val()}, step={self.step})"

    def inc(self) -> int:
        for _ in range(self.step):
            next(self._incs)
        return self.val()

    def val(self) -> int:
        return next(self._incs) - next(self._base)


class JobTimer:
    def __init__(self, name=None, args: CommonArguments | NewCommonArguments | DictConfig = None, prefix=None, postfix=None,
                 verbose=1, mt=0, mb=0, pt=0, pb=0, rt=0, rb=0, rc='-', rw=137,
                 flush_sec=0.1, mute_loggers=None, mute_warning=None):
        self.name = name
        self.args = args
        self.prefix = prefix if prefix and len(prefix) > 0 else None
        self.postfix = postfix if postfix and len(postfix) > 0 else None
        self.flush_sec = flush_sec
        self.mt: int = mt
        self.mb: int = mb
        self.pt: int = pt
        self.pb: int = pb
        self.rt: int = rt
        self.rb: int = rb
        self.rc: str = rc
        self.rw: int = rw
        self.verbose: int = verbose
        assert isinstance(mute_loggers, (type(None), str, list, tuple, set))
        assert isinstance(mute_warning, (type(None), str, list, tuple, set))
        self.mute_loggers = tupled(mute_loggers)
        self.mute_warning = tupled(mute_warning)
        self.t1: Optional[datetime] = datetime.now()
        self.t2: Optional[datetime] = datetime.now()
        self.td: Optional[timedelta] = self.t2 - self.t1

    def __enter__(self):
        try:
            self.mute_loggers = [logging.getLogger(x) for x in self.mute_loggers] if self.mute_loggers else None
            if self.mute_loggers:
                for x in self.mute_loggers:
                    x.disabled = True
                    x.propagate = False
            if self.mute_warning:
                for x in self.mute_warning:
                    warnings.filterwarnings('ignore', category=UserWarning, module=x)
            flush_or(sys.stdout, sys.stderr, sec=self.flush_sec if self.flush_sec else None)
            if self.verbose > 0:
                if self.mt > 0:
                    for _ in range(self.mt):
                        logger.info('')
                if self.rt > 0:
                    for _ in range(self.rt):
                        logger.info(hr(c=self.rc, w=self.rw))
                if self.name:
                    logger.info(f'{self.prefix + SP if self.prefix else NO}[INIT] {self.name}{SP + self.postfix if self.postfix else NO}')
                    if self.rt > 0:
                        for _ in range(self.rt):
                            logger.info(hr(c=self.rc, w=self.rw))
                if self.pt > 0:
                    for _ in range(self.pt):
                        logger.info('')
                flush_or(sys.stdout, sys.stderr, sec=self.flush_sec if self.flush_sec else None)
            if self.args:
                if hasattr(self.args, "time"):
                    self.args.time.set_started()
                if self.verbose >= 1:
                    if hasattr(self.args, "info_args"):
                        self.args.info_args(c="-", w=self.rw)
                    else:
                        yaml_str = to_yaml(self.args, resolve=True, width=4096).rstrip()
                        logger.info("[args]")
                        sum(logger.info(f"  {l}") or 1 for l in yaml_str.splitlines())
                        logger.info(hr(c=self.rc, w=self.rw))
                if self.verbose >= 2:
                    if hasattr(self.args, "save_args"):
                        self.args.save_args()
            self.t1 = datetime.now()
            return self
        except Exception as e:
            raise e
            # logger.error(f"[JobTimer.__enter__()] [{type(e)}] {e}")
            # exit(11)

    def __exit__(self, *exc_info):
        try:
            if self.args:
                if hasattr(self.args, "time"):
                    self.args.time.set_settled()
                if self.verbose >= 2:
                    self.args.save_args()
            self.t2 = datetime.now()
            self.td = self.t2 - self.t1
            flush_or(sys.stdout, sys.stderr, sec=self.flush_sec if self.flush_sec else None)
            if self.verbose > 0:
                if self.pb > 0:
                    for _ in range(self.pb):
                        logger.info('')
                if self.rb > 0:
                    for _ in range(self.rb):
                        logger.info(hr(c=self.rc, w=self.rw))
                if self.name:
                    logger.info(f'{self.prefix + SP if self.prefix else NO}[EXIT] {self.name}{SP + self.postfix if self.postfix else NO} ($={str_delta(self.td)})')
                    if self.rb > 0:
                        for _ in range(self.rb):
                            logger.info(hr(c=self.rc, w=self.rw))
                if self.mb > 0:
                    for _ in range(self.mb):
                        logger.info('')
                flush_or(sys.stdout, sys.stderr, sec=self.flush_sec if self.flush_sec else None)
            if self.mute_loggers:
                for x in self.mute_loggers:
                    x.disabled = False
                    x.propagate = True
            if self.mute_warning:
                for x in self.mute_warning:
                    warnings.filterwarnings('default', category=UserWarning, module=x)
        except Exception as e:
            logger.error(f"[JobTimer.__exit__()] [{type(e)}] {e}")
            exit(22)


def find_sublist_range(haystack: List[Any], sublist: List[Any], case_sensitive: bool = True) -> List[int]:
    if not case_sensitive:
        haystack = [x.lower() if isinstance(x, str) else x for x in haystack]
        sublist = [x.lower() if isinstance(x, str) else x for x in sublist]

    sub_len = len(sublist)
    for i in range(len(haystack) - sub_len + 1):
        if haystack[i:i + sub_len] == sublist:
            return list(range(i, i + sub_len))
    return list()
