import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import typer
from dataclasses_json import DataClassJsonMixin
from pytorch_lightning.loggers import CSVLogger

from chrisbase.io import get_hostname, get_hostaddr, running_file, first_or, cwd, configure_dual_logger, configure_unit_logger


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
    csv_logger: CSVLogger | None = field(init=False, default=None)

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
