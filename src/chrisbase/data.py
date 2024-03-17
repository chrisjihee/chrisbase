import json
import logging
import math
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from io import IOBase
from itertools import islice
from pathlib import Path
from typing import List, Optional, Mapping, Any, Iterable, Tuple

import pandas as pd
import pymongo.collection
import pymongo.database
import pymongo.errors
import typer
from dataclasses_json import DataClassJsonMixin
from elasticsearch import Elasticsearch
from more_itertools import ichunked
from pymongo import MongoClient

from chrisbase.io import get_hostname, get_hostaddr, current_file, first_or, cwd, hr, flush_or, make_parent_dir, get_ip_addrs, setup_unit_logger, setup_dual_logger, open_file, file_lines, to_table_lines
from chrisbase.time import now, str_delta
from chrisbase.util import tupled, SP, NO, to_dataframe

logger = logging.getLogger(__name__)


class AppTyper(typer.Typer):
    def __init__(self):
        super().__init__(
            add_completion=False,
            pretty_exceptions_enable=False,
        )


@dataclass
class TypedData(DataClassJsonMixin):
    data_type = None

    def __post_init__(self):
        self.data_type = self.__class__.__name__


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
    strict: bool = field(default=False)

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


@dataclass
class TableOption(StreamOption):
    sort: str | List[Tuple[str, int] | str] = field(default="_id")
    find: dict = field(default_factory=dict)
    timeout: int = field(default=5000)


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
        elif self.opt.strict:
            assert self.usable(), f"Could not open source: opt={self.opt}"
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
        self.path: Path | None = None
        self.fp: IOBase | None = None

    def __exit__(self, *exc_info):
        if self.fp:
            self.fp.close()

    def __len__(self) -> int:
        if self.usable():
            self.fp.flush()
            return file_lines(self.path)
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
            self.path.touch()
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
        self.db = self.cli.get_database(db_name)
        self.table = self.db.get_collection(f"{self.opt.name}")
        return self.usable()

    def reset(self):
        if self.usable():
            logger.info(f"Drop an existing table: {self.opt}")
            self.db.drop_collection(f"{self.opt.name}")

    def count(self, query: Mapping[str, Any], exact: bool = False) -> int:
        if self.usable():
            if exact:
                return self.table.count_documents(query)
            else:
                return self.table.estimated_document_count()
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
    file: FileOption | None = field(default=None)
    table: TableOption | None = field(default=None)
    index: IndexOption | None = field(default=None)

    @dataclass
    class InputItems:
        total: int

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

    def ready_inputs(self, inputs: Iterable, total: int) -> "InputOption.SingleItems | InputOption.BatchItems":
        inputs = map(self.safe_dict, inputs)
        if self.start > 0:
            inputs = islice(inputs, self.start, total)
            total = max(0, min(total, total - self.start))
        if self.limit > 0:
            inputs = islice(inputs, self.limit)
            total = min(total, self.limit)
        if self.batch <= 1:
            return InputOption.SingleItems(
                total=total,
                singles=inputs,
            )
        else:
            return InputOption.BatchItems(
                total=math.ceil(total / self.batch),
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
    hostname: str = field(init=False)
    hostaddr: str = field(init=False)
    time_stamp: str = now('%m%d.%H%M%S')
    python_path: Path = field(init=False)
    current_dir: Path = field(init=False)
    current_file: Path = field(init=False)
    working_dir: Path = field(init=False)
    command_args: List[str] = field(init=False)
    num_ip_addrs: int = field(init=False)
    max_workers: int = field(default=1)
    debugging: bool = field(default=False)
    msg_level: int = field(default=logging.INFO)
    msg_format: str = field(default=logging.BASIC_FORMAT)
    date_format: str = field(default="[%m.%d %H:%M:%S]")
    output_home: str | Path | None = field(default=None)
    logging_file: str | Path | None = field(default=None)
    argument_file: str | Path | None = field(default=None)

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
        self.ip_addrs, self.num_ip_addrs = get_ip_addrs()
        self.output_home = Path(self.output_home).absolute() if self.output_home else None
        self.logging_file = Path(self.logging_file) if self.logging_file else None
        self.argument_file = Path(self.argument_file) if self.argument_file else None
        self._setup_logger()

    def info_env(self):
        for line in to_table_lines(to_dataframe(self)):
            logger.info(line)
        return self

    def _setup_logger(self):
        if self.output_home and self.logging_file:
            setup_dual_logger(level=self.msg_level, fmt=self.msg_format, datefmt=self.date_format, stream=sys.stdout, filename=self.output_home / self.logging_file)
        else:
            setup_unit_logger(level=self.msg_level, fmt=self.msg_format, datefmt=self.date_format, stream=sys.stdout)
        return self

    def set_job_name(self, name: str = None):
        self.job_name = name
        return self

    def set_output_home(self, output_home: str | Path | None):
        self.output_home = Path(output_home).absolute() if output_home else None
        self._setup_logger()

    def set_logging_file(self, logging_file: str | Path | None):
        self.logging_file = Path(logging_file) if logging_file else None
        self._setup_logger()

    def set_argument_file(self, argument_file: str | Path | None):
        self.logging_file = Path(argument_file) if argument_file else None


@dataclass
class TimeChecker(ResultData):
    t1 = datetime.now()
    t2 = datetime.now()
    started: str | None = field(default=None)
    settled: str | None = field(default=None)
    elapsed: str | None = field(default=None)

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


@dataclass
class CommonArguments(ArgumentGroupData):
    tag = None
    env: ProjectEnv = field()
    time = TimeChecker()

    def __post_init__(self):
        super().__post_init__()

    def save_args(self, to: Path | str = None) -> Path | None:
        if self.env.output_home and self.env.argument_file:
            args_file = to if to else self.env.output_home / self.env.argument_file
            args_json = self.to_json(default=str, ensure_ascii=False, indent=2)
            make_parent_dir(args_file).write_text(args_json, encoding="utf-8")
            return args_file
        else:
            return None

    def info_args(self):
        for line in to_table_lines(self.dataframe()):
            logger.info(line)
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
        ]).reset_index(drop=True)


class RuntimeChecking:
    def __init__(self, args: CommonArguments):
        self.args: CommonArguments = args

    def __enter__(self):
        self.args.time.set_started()
        self.args.save_args()

    def __exit__(self, *exc_info):
        self.args.time.set_settled()
        self.args.save_args()


class JobTimer:
    def __init__(self, name=None, args: CommonArguments = None, prefix=None, postfix=None,
                 verbose=True, mt=0, mb=0, pt=0, pb=0, rt=0, rb=0, rc='-',
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
        self.verbose: bool = verbose
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
            if self.verbose:
                if self.mt > 0:
                    for _ in range(self.mt):
                        logger.info('')
                if self.rt > 0:
                    for _ in range(self.rt):
                        logger.info(hr(c=self.rc))
                if self.name:
                    logger.info(f'{self.prefix + SP if self.prefix else NO}[INIT] {self.name}{SP + self.postfix if self.postfix else NO}')
                    if self.rt > 0:
                        for _ in range(self.rt):
                            logger.info(hr(c=self.rc))
                if self.pt > 0:
                    for _ in range(self.pt):
                        logger.info('')
                flush_or(sys.stdout, sys.stderr, sec=self.flush_sec if self.flush_sec else None)
            if self.args:
                self.args.time.set_started()
                if self.verbose:
                    self.args.save_args()
                    self.args.info_args()
            self.t1 = datetime.now()
            return self
        except Exception as e:
            raise e
            # logger.error(f"[JobTimer.__enter__()] [{type(e)}] {e}")
            # exit(11)

    def __exit__(self, *exc_info):
        try:
            if self.args:
                self.args.time.set_settled()
                self.args.save_args()
            self.t2 = datetime.now()
            self.td = self.t2 - self.t1
            flush_or(sys.stdout, sys.stderr, sec=self.flush_sec if self.flush_sec else None)
            if self.verbose:
                if self.pb > 0:
                    for _ in range(self.pb):
                        logger.info('')
                if self.rb > 0:
                    for _ in range(self.rb):
                        logger.info(hr(c=self.rc))
                if self.name:
                    logger.info(f'{self.prefix + SP if self.prefix else NO}[EXIT] {self.name}{SP + self.postfix if self.postfix else NO} ($={str_delta(self.td)})')
                    if self.rb > 0:
                        for _ in range(self.rb):
                            logger.info(hr(c=self.rc))
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
