import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin

from chrisbase.io import get_hostname, get_hostaddr, running_file, first_or, cwd, hr, str_table, flush_or, make_parent_dir, configure_dual_logger, configure_unit_logger
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
class ProjectEnv(TypedData):
    project: str = field()
    job_name: str = field(default=None)
    hostname: str = field(init=False)
    hostaddr: str = field(init=False)
    python_path: Path = field(init=False)
    working_path: Path = field(init=False)
    running_file: Path = field(init=False)
    command_args: List[str] = field(init=False)
    max_workers: int = field(default=os.cpu_count())
    output_home: str | Path | None = field(default=None)
    logging_file: str | Path = field(default="message.out")
    argument_file: str | Path = field(default="arguments.json")
    debugging: bool = field(default=False)
    msg_level: int = field(default=logging.INFO)
    msg_format: str = field(default=logging.BASIC_FORMAT)
    date_format: str = field(default="[%m.%d %H:%M:%S]")

    def set(self, name: str = None):
        self.job_name = name
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
        self.logging_file = Path(self.logging_file)
        self.argument_file = Path(self.argument_file)
        if self.output_home:
            self.output_home = Path(self.output_home)
            configure_dual_logger(level=self.msg_level, fmt=self.msg_format, datefmt=self.date_format,
                                  filename=self.output_home / self.logging_file)
        else:
            configure_unit_logger(level=self.msg_level, fmt=self.msg_format, datefmt=self.date_format,
                                  stream=sys.stdout)


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
    time: TimeChecker = field(default=TimeChecker())

    def __post_init__(self):
        super().__post_init__()
        if self.tag and not self.env.argument_file.stem.endswith(self.tag):
            self.env.argument_file = self.env.argument_file.with_stem(f"{self.env.argument_file.stem}-{self.tag}")
        if self.tag and not self.env.logging_file.stem.endswith(self.tag):
            self.env.logging_file = self.env.logging_file.with_stem(f"{self.env.logging_file.stem}-{self.tag}")
        self.env.output_home = self.env.output_home or Path("output")
        configure_dual_logger(level=self.env.msg_level, fmt=self.env.msg_format, datefmt=self.env.date_format,
                              filename=self.env.output_home / self.env.logging_file)

    def save_args(self, to: Path | str = None) -> Path | None:
        if not self.env.output_home:
            return None
        args_file = to if to else self.env.output_home / self.env.argument_file
        args_json = self.to_json(default=str, ensure_ascii=False, indent=2)
        make_parent_dir(args_file).write_text(args_json, encoding="utf-8")
        return args_file

    def info_args(self):
        table = str_table(self.dataframe(), tablefmt="presto")  # "plain", "presto"
        for line in table.splitlines() + [hr(c='-')]:
            logger.info(line)
        return self

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
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
