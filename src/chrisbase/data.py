import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin

from chrisbase.io import get_hostname, get_hostaddr, running_file, first_or, cwd, configure_dual_logger, configure_unit_logger, make_parent_dir, str_table, hr, LoggingFormat
from chrisbase.time import now, str_delta
from chrisbase.util import to_dataframe

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
    output_home: str | Path | None = field(default=None)
    logging_file: str | Path = field(default="message.out")
    argument_file: str | Path = field(default="arguments.json")
    debugging: bool = field(default=False)
    msg_level: int = field(default=logging.INFO)
    msg_format: str = field(default=logging.BASIC_FORMAT)
    date_format: str = field(default="[%m.%d %H:%M:%S]")
    try:
        import pytorch_lightning.loggers
        csv_logger: Optional[pytorch_lightning.loggers.CSVLogger] = field(init=False, default=None)
    except ImportError as e:
        print(f"pytorch_lightning.loggers.CSVLogger is not available: {e.msg}")

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
    tag = "common"
    env: ProjectEnv = field()
    time: TimeChecker = field(default=TimeChecker())

    def __post_init__(self):
        super().__post_init__()
        if not self.env.logging_file.stem.endswith(self.tag):
            self.env.logging_file = self.env.logging_file.with_stem(f"{self.env.logging_file.stem}-{self.tag}")
        if not self.env.argument_file.stem.endswith(self.tag):
            self.env.argument_file = self.env.argument_file.with_stem(f"{self.env.argument_file.stem}-{self.tag}")
        self.env.output_home = self.env.output_home or Path("output")
        configure_dual_logger(level=self.env.msg_level, fmt=self.env.msg_format, datefmt=self.env.date_format,
                              filename=self.env.output_home / self.env.logging_file)

    def save_arguments(self, to: Path | str = None) -> Path | None:
        if not self.env.output_home:
            return None
        args_file = to if to else self.env.output_home / self.env.argument_file
        args_json = self.to_json(default=str, ensure_ascii=False, indent=2)
        make_parent_dir(args_file).write_text(args_json, encoding="utf-8")
        return args_file

    def info_arguments(self):
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
