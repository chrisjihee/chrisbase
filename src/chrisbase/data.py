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
from typing import List, Optional, Mapping, Any

import pandas as pd
import pymongo.collection
import pymongo.database
import pymongo.errors
import typer
from dataclasses_json import DataClassJsonMixin
from elasticsearch import Elasticsearch
from more_itertools import ichunked
from pymongo import MongoClient

from chrisbase.io import get_hostname, get_hostaddr, running_file, first_or, cwd, hr, str_table, flush_or, make_parent_dir, get_ip_addrs, configure_unit_logger, configure_dual_logger, open_file
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
class FileOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    mode: str = field(default="rb")
    strict: bool = field(default=False)
    encoding: str = field(default="utf-8")

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)

    def __repr__(self):
        return f"{self.home}/{self.name}"


@dataclass
class TableOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    sort: str = field(default="_id")
    find: dict = field(default_factory=dict)
    reset: bool = field(default=False)
    strict: bool = field(default=False)
    timeout: int = field(default=5000)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)

    def __repr__(self):
        return f"{self.home}/{self.name}"


@dataclass
class IndexOption(OptionData):
    home: str = field()
    name: str = field()
    user: str = field()
    pswd: str = field()
    reset: bool = field(default=False)
    strict: bool = field(default=False)
    timeout: int = field(default=10)
    retrial: int = field(default=3)
    create: str | Path | None = field(default=None)
    create_args = None

    def __post_init__(self):
        self.create_args = {}
        if self.create:
            self.create = Path(self.create)
            if self.create.exists() and self.create.is_file():
                content = self.create.read_text()
                if content:
                    self.create_args = json.loads(content)

    def __repr__(self):
        return f"{self.user}@{self.home}/{self.name}"


class LineFileWrapper:
    def __init__(self, opt: FileOption | None):
        self.opt: FileOption = opt
        self.path: Path | None = None
        self.fp: IOBase | None = None

    def __exit__(self, *exc_info):
        if self.fp:
            self.fp.close()

    def __enter__(self):
        if not self.opt:
            return None
        self.open(strict=self.opt.strict)
        return self

    def open(self, strict: bool = False):
        self.path = self.opt.home / self.opt.name
        if self.path.exists() and self.path.is_file():
            self.fp = open_file(
                self.path,
                mode=self.opt.mode,
                encoding=None if "b" in self.opt.mode else self.opt.encoding
            )
        if strict:
            assert self.usable(), f"Could not open file: opt={self.opt}"

    def usable(self) -> bool:
        return self.fp is not None and self.fp.readable()

    def __iter__(self):
        if self.fp is not None:
            for line in self.fp:
                if "b" in self.opt.mode:
                    line = line.decode(self.opt.encoding)
                yield line.strip()


class MongoDBWrapper:
    def __init__(self, opt: TableOption | None):
        self.opt: TableOption = opt
        self.cli: MongoClient | None = None
        self.db: pymongo.database.Database | None = None
        self.table: pymongo.collection.Collection | None = None

    def __exit__(self, *exc_info):
        if self.cli:
            self.cli.close()

    def __enter__(self):
        if not self.opt:
            return None
        self.open(strict=self.opt.strict)
        if self.opt.reset:
            self.reset()
        return self

    def __iter__(self):
        if self.table is not None:
            return self.table.find(self.opt.find).sort(self.opt.sort)

    def __len__(self) -> int:
        if self.table is not None:
            return self.table.count_documents(self.opt.find)
        else:
            return 0

    def open(self, strict: bool = False):
        assert len(self.opt.home.parts) >= 2, f"Invalid MongoDB host: {self.opt.home}"
        db_addr, db_name = self.opt.home.parts[:2]
        self.cli = MongoClient(f"mongodb://{db_addr}/?timeoutMS={self.opt.timeout}")
        self.db = self.cli.get_database(db_name)
        self.table = self.db.get_collection(f"{self.opt.name}")
        if strict:
            assert self.usable(), f"Could not connect to MongoDB: opt={self.opt}"

    def usable(self) -> bool:
        try:
            res = self.db.command("ping")
        except pymongo.errors.ServerSelectionTimeoutError:
            res = {"ok": 0, "exception": "ServerSelectionTimeoutError"}
        return res.get("ok", 0) > 0

    def reset(self):
        logger.info(f"Drop an existing table: {self.opt}")
        self.db.drop_collection(f"{self.opt.name}")

    def count(self, query: Mapping[str, Any]) -> int:
        return self.table.count_documents(query, limit=1)


class ElasticSearchWrapper:
    def __init__(self, opt: IndexOption | None):
        self.opt: IndexOption = opt
        self.cli: Elasticsearch | None = None

    def __exit__(self, *exc_info):
        if self.cli:
            self.cli.close()

    def __enter__(self):
        if not self.opt:
            return None
        self.open(strict=self.opt.strict)
        if self.opt.reset:
            self.reset()
        return self

    def open(self, strict: bool = False):
        self.cli = Elasticsearch(
            hosts=f"http://{self.opt.home}",
            basic_auth=(self.opt.user, self.opt.pswd),
            request_timeout=self.opt.timeout,
            retry_on_timeout=self.opt.retrial > 0,
            max_retries=self.opt.retrial,
        )
        if strict:
            assert self.usable(), f"Could not connect to ElasticSearch: opt={self.opt}"

    def usable(self) -> bool:
        return self.cli and self.cli.ping()

    def reset(self):
        if self.cli.indices.exists(index=self.opt.name):
            logger.info(f"Drop an existing index: {self.opt}")
            self.cli.indices.delete(index=self.opt.name)
        self.cli.indices.create(index=self.opt.name, **self.opt.create_args)
        logger.info(f"Created a new index: {self.opt}")
        logger.info(f"- option: keys={list(self.opt.create_args.keys())}")

    def refresh(self, verbose: bool = False):
        self.cli.indices.refresh(index=self.opt.name)
        if verbose:
            res = self.cli.cat.indices(index=self.opt.name, v=True)
            if res.meta.status == 200:
                logger.info(hr('-'))
                for line in res.body.strip().splitlines():
                    logger.info(line)
                logger.info(hr('-'))

    def __len__(self) -> int:
        self.cli.indices.refresh(index=self.opt.name)
        if self.cli is not None:
            res = self.cli.cat.count(index=self.opt.name, format="json")
            if res.meta.status == 200 and len(res.body) > 0 and "count" in res.body[0]:
                return int(res.body[0]["count"])
        return 0


@dataclass
class DataOption(OptionData):
    start: int = field(default=0)
    limit: int = field(default=-1)
    batch: int = field(default=1)
    inter: int = field(default=10000)
    total: int = field(default=-1)
    file: FileOption | None = field(default=None)
    table: TableOption | None = field(default=None)
    index: IndexOption | None = field(default=None)

    @staticmethod
    def safe_dict(x: str | dict) -> dict:
        if isinstance(x, dict):
            return x
        else:
            return json.loads(x) if x.strip().startswith('{') else {}

    def input_batches(self, inputs, num_input: int):
        inputs = map(DataOption.safe_dict, inputs)
        if self.start > 0:
            inputs = islice(inputs, self.start, num_input)
            num_input = max(0, min(num_input, num_input - self.start))
        if self.limit > 0:
            inputs = islice(inputs, self.limit)
            num_input = min(num_input, self.limit)
        batch = ichunked(inputs, self.batch)
        num_batch = math.ceil(num_input / self.batch)
        return batch, num_batch, num_input


@dataclass
class ProjectEnv(TypedData):
    project: str = field()
    job_name: str = field(default=None)
    hostname: str = field(init=False)
    hostaddr: str = field(init=False)
    python_path: Path = field(init=False)
    working_path: Path = field(init=False)
    running_file: Path = field(init=False)
    command_args: List[str] = field(init=False)
    num_ip_addrs: int = field(init=False)
    max_workers: int = field(default=1)
    output_home: str | Path | None = field(default=None)
    logging_file: str | Path | None = field(default=None)
    argument_file: str | Path = field(default="arguments.json")
    debugging: bool = field(default=False)
    msg_level: int = field(default=logging.INFO)
    msg_format: str = field(default=logging.BASIC_FORMAT)
    date_format: str = field(default="[%m.%d %H:%M:%S]")
    time_stamp: str = now('%m%d.%H%M%S')

    def set(self, name: str = None):
        self.job_name = name
        return self

    def info_args(self):
        table = str_table(to_dataframe(self), tablefmt="presto")  # "plain", "presto"
        for line in table.splitlines() + [hr(c='-')]:
            logger.info(line)
        return self

    def __post_init__(self):
        assert self.project, "Project name must be provided"
        self.hostname = get_hostname()
        self.hostaddr = get_hostaddr()
        self.python_path = Path(sys.executable)
        self.running_file = running_file()
        self.project_path = first_or([x for x in self.running_file.parents if x.name.startswith(self.project)])
        assert self.project_path, f"Could not find project path for {self.project} in {', '.join([str(x) for x in self.running_file.parents])}"
        self.working_path = cwd(self.project_path)
        self.running_file = self.running_file.relative_to(self.working_path)
        self.command_args = sys.argv[1:]
        self.ip_addrs, self.num_ip_addrs = get_ip_addrs()
        self.output_home = Path(self.output_home) if self.output_home else None
        self.logging_file = Path(self.logging_file) if self.logging_file else None
        self.argument_file = Path(self.argument_file)
        configure_unit_logger(level=self.msg_level, fmt=self.msg_format, datefmt=self.date_format, stream=sys.stdout)


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
    time = TimeChecker()
    env: ProjectEnv = field()

    def __post_init__(self):
        super().__post_init__()
        if self.tag:
            if not self.env.argument_file.stem.startswith(self.tag):
                self.env.argument_file = self.env.argument_file.with_stem(f"{self.tag}-{self.env.argument_file.stem}")
            if self.env.logging_file:
                if not self.env.logging_file.stem.startswith(self.tag):
                    self.env.logging_file = self.env.logging_file.with_stem(f"{self.tag}-{self.env.logging_file.stem}")
        if self.env.time_stamp:
            if not self.env.argument_file.stem.endswith(self.env.time_stamp):
                self.env.argument_file = self.env.argument_file.with_stem(f"{self.env.argument_file.stem}-{self.env.time_stamp}")
            if self.env.logging_file:
                if not self.env.logging_file.stem.endswith(self.env.time_stamp):
                    self.env.logging_file = self.env.logging_file.with_stem(f"{self.env.logging_file.stem}-{self.env.time_stamp}")
        if self.env.output_home and self.env.logging_file:
            configure_dual_logger(level=self.env.msg_level, fmt=self.env.msg_format, datefmt=self.env.date_format, stream=sys.stdout,
                                  filename=self.env.output_home / self.env.logging_file)

    def save_args(self, to: Path | str = None) -> Path | None:
        if not self.env.output_home:
            return None
        args_file = to if to else self.env.output_home / self.env.argument_file
        args_json = self.to_json(default=str, ensure_ascii=False, indent=2)
        make_parent_dir(args_file).write_text(args_json, encoding="utf-8")
        return args_file

    def info_args(self):
        table = str_table(self.dataframe(), tablefmt="presto")
        lines = table.splitlines()
        lines = [lines[1]] + lines + [lines[1]]
        for line in lines:
            logger.info(line)
        return self

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
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


class ArgumentsUsing:  # TODO: Remove someday!
    def __init__(self, args: CommonArguments, delete_on_exit: bool = True):
        self.args: CommonArguments = args
        self.delete_on_exit: bool = delete_on_exit

    def __enter__(self) -> Path:
        self.args_file: Path | None = self.args.save_args()
        return self.args_file

    def __exit__(self, *exc_info):
        if self.delete_on_exit and self.args_file:
            self.args_file.unlink(missing_ok=True)


class JobTimer:
    def __init__(self, name=None, args: CommonArguments = None, prefix=None, postfix=None,
                 verbose=True, mt=0, mb=0, pt=0, pb=0, rt=0, rb=0, rc='-',
                 flush_sec=0.3, mute_loggers=None, mute_warning=None):
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
                self.args.save_args()
                if self.verbose:
                    self.args.info_args()
            self.t1 = datetime.now()
            return self
        except Exception as e:
            logger.error(f"[JobTimer.__enter__()] [{type(e)}] {e}")
            exit(11)

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
