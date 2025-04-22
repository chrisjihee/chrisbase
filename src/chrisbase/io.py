import bz2
import csv
import gzip
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import traceback
import warnings
from io import IOBase
from ipaddress import IPv4Address
from itertools import chain
from pathlib import Path
from time import sleep
from typing import Iterable, List

import httpx
import netifaces
import pandas as pd
import yaml
from chrisbase.time import from_timestamp
from chrisbase.util import tupled, OX
from omegaconf import OmegaConf
from omegaconf._utils import get_omega_conf_dumper
from tabulate import tabulate
from tensorboard.backend.event_processing import event_accumulator

logger = logging.getLogger(__name__)
sys_stdout = sys.stdout
sys_stderr = sys.stderr

from enum import Enum


class LoggingFormat(Enum):
    PRINT_00 = ' ┇ '.join(['%(message)s'])
    PRINT_12 = ' ┇ '.join(['%(name)12s', '%(message)s'])
    PRINT_16 = ' ┇ '.join(['%(name)16s', '%(message)s'])
    PRINT_20 = ' ┇ '.join(['%(name)20s', '%(message)s'])
    BRIEF_00 = ' ┇ '.join(['%(asctime)s', '%(message)s'])
    BRIEF_12 = ' ┇ '.join(['%(asctime)s', '%(name)12s', '%(message)s'])
    BRIEF_16 = ' ┇ '.join(['%(asctime)s', '%(name)16s', '%(message)s'])
    BRIEF_20 = ' ┇ '.join(['%(asctime)s', '%(name)20s', '%(message)s'])
    CHECK_00 = ' ┇ '.join(['%(asctime)s', '%(levelname)-7s', '%(message)s'])
    CHECK_12 = ' ┇ '.join(['%(asctime)s', '%(levelname)-7s', '%(name)12s', '%(message)s'])
    CHECK_16 = ' ┇ '.join(['%(asctime)s', '%(levelname)-7s', '%(name)16s', '%(message)s'])
    CHECK_20 = ' ┇ '.join(['%(asctime)s', '%(levelname)-7s', '%(name)20s', '%(message)s'])
    CHECK_24 = ' ┇ '.join(['%(asctime)s', '%(levelname)-7s', '%(name)24s', '%(message)s'])
    CHECK_28 = ' ┇ '.join(['%(asctime)s', '%(levelname)-7s', '%(name)28s', '%(message)s'])
    CHECK_32 = ' ┇ '.join(['%(asctime)s', '%(levelname)-7s', '%(name)32s', '%(message)s'])
    CHECK_36 = ' ┇ '.join(['%(asctime)s', '%(levelname)-7s', '%(name)36s', '%(message)s'])
    CHECK_40 = ' ┇ '.join(['%(asctime)s', '%(levelname)-7s', '%(name)40s', '%(message)s'])
    CHECK_48 = ' ┇ '.join(['%(asctime)s', '%(levelname)-7s', '%(name)48s', '%(message)s'])
    TRACE_12 = ' ┇ '.join(['%(asctime)s', '%(filename)12s:%(lineno)-4d', '%(message)s'])
    TRACE_16 = ' ┇ '.join(['%(asctime)s', '%(filename)16s:%(lineno)-4d', '%(message)s'])
    TRACE_20 = ' ┇ '.join(['%(asctime)s', '%(filename)20s:%(lineno)-4d', '%(message)s'])
    TRACE_24 = ' ┇ '.join(['%(asctime)s', '%(filename)24s:%(lineno)-4d', '%(message)s'])
    TRACE_28 = ' ┇ '.join(['%(asctime)s', '%(filename)28s:%(lineno)-4d', '%(message)s'])
    TRACE_32 = ' ┇ '.join(['%(asctime)s', '%(filename)32s:%(lineno)-4d', '%(message)s'])
    TRACE_36 = ' ┇ '.join(['%(asctime)s', '%(filename)36s:%(lineno)-4d', '%(message)s'])
    TRACE_40 = ' ┇ '.join(['%(asctime)s', '%(filename)40s:%(lineno)-4d', '%(message)s'])
    DEBUG_00 = ' ┇ '.join(['%(pathname)60s:%(lineno)-5d', '%(asctime)s', '%(levelname)-7s', '%(message)s'])
    DEBUG_12 = ' ┇ '.join(['%(pathname)60s:%(lineno)-5d', '%(asctime)s', '%(levelname)-7s', '%(name)12s', '%(message)s'])
    DEBUG_16 = ' ┇ '.join(['%(pathname)60s:%(lineno)-5d', '%(asctime)s', '%(levelname)-7s', '%(name)16s', '%(message)s'])
    DEBUG_20 = ' ┇ '.join(['%(pathname)70s:%(lineno)-5d', '%(asctime)s', '%(levelname)-7s', '%(name)20s', '%(message)s'])
    DEBUG_24 = ' ┇ '.join(['%(pathname)70s:%(lineno)-5d', '%(asctime)s', '%(levelname)-7s', '%(name)24s', '%(message)s'])
    DEBUG_28 = ' ┇ '.join(['%(pathname)70s:%(lineno)-5d', '%(asctime)s', '%(levelname)-7s', '%(name)28s', '%(message)s'])
    DEBUG_32 = ' ┇ '.join(['%(pathname)90s:%(lineno)-5d', '%(asctime)s', '%(levelname)-7s', '%(name)32s', '%(message)s'])
    DEBUG_36 = ' ┇ '.join(['%(pathname)90s:%(lineno)-5d', '%(asctime)s', '%(levelname)-7s', '%(name)36s', '%(message)s'])
    DEBUG_40 = ' ┇ '.join(['%(pathname)120s:%(lineno)-5d', '%(asctime)s', '%(levelname)-7s', '%(name)40s', '%(message)s'])
    DEBUG_48 = ' ┇ '.join(['%(pathname)120s:%(lineno)-5d', '%(asctime)s', '%(levelname)-7s', '%(name)48s', '%(message)s'])


class LoggerWriter:
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        """
        A simple wrapper to use a logger like a file-like stream.

        :param logger: The logger instance to which messages will be sent.
        :param level: Logging level to use (default: logging.INFO).
        """
        self.logger = logger
        self.level = level

    def write(self, msg: str):
        """
        Emulates the behavior of a stream's write method.
        Non-empty lines are forwarded to the logger at the given level.
        """
        # Strip out extra whitespace/newlines and only log non-empty lines
        msg = msg.rstrip()
        if msg:
            self.logger.log(self.level, msg)

    def flush(self):
        """
        Emulates the behavior of a stream's flush method.
        In this context, we generally do not need to do anything special for flush.
        """
        pass


class MuteStd:
    def __init__(self, out=None, err=None, flush_sec=0.0, mute_warning=None, mute_logger=None):
        self.mute = open(os.devnull, 'w')
        self.stdout = out or self.mute
        self.stderr = err or self.mute
        self.preout = sys.stdout
        self.preerr = sys.stderr
        self.flush_sec = flush_sec
        assert isinstance(mute_logger, (type(None), str, list, tuple, set))
        assert isinstance(mute_warning, (type(None), str, list, tuple, set))
        self.mute_logger = tupled(mute_logger)
        self.mute_warning = tupled(mute_warning)

    def __enter__(self):
        try:
            self.mute_logger = [logging.getLogger(x) for x in self.mute_logger] if self.mute_logger else None
            if self.mute_logger:
                for x in self.mute_logger:
                    x.disabled = True
                    x.propagate = False
            if self.mute_warning:
                for x in self.mute_warning:
                    warnings.filterwarnings('ignore', category=UserWarning, module=x)
            flush_or(self.preout, self.preerr, sec=self.flush_sec if self.flush_sec else None)
            sys.stdout = self.stdout
            sys.stderr = self.stderr
        except Exception as e:
            print(f"[MuteStd.__enter__()] [{type(e)}] {e}", file=sys_stderr)
            exit(11)

    def __exit__(self, *exc_info):
        try:
            flush_or(self.stdout, self.stderr, sec=self.flush_sec if self.flush_sec else None)
            if self.mute_logger:
                for x in self.mute_logger:
                    x.disabled = False
                    x.propagate = True
            if self.mute_warning:
                for x in self.mute_warning:
                    warnings.filterwarnings('default', category=UserWarning, module=x)
            sys.stdout = self.preout
            sys.stderr = self.preerr
            self.mute.close()
        except Exception as e:
            print(f"[MuteStd.__exit__()] [{type(e)}] {e}", file=sys_stderr)
            exit(22)


def cwd(path=None) -> Path:
    if not path:
        return Path.cwd()
    else:
        os.chdir(path)
        return Path(path)


def get_call_stack():
    def deeper():
        return list(traceback.walk_stack(None))

    return tuple(
        {"file": frame.f_code.co_filename, "name": frame.f_code.co_name}
        for frame, _ in deeper() if 'plugins/python/helpers' not in frame.f_code.co_filename
    )


# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_notebook() -> bool:
    try:
        from IPython.core.getipython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False


def current_file(known_path: Path or str = None) -> Path:
    if known_path and exists_or(Path(known_path)):
        return Path(known_path)
    elif is_notebook():
        import ipynbname
        return ipynbname.path()
    else:
        for call_stack in get_call_stack():
            if call_stack['name'] == '<module>':
                return Path(call_stack['file'])
        raise RuntimeError("Cannot find current path")


def hr(c="=", w=137, t=0, b=0, title=''):
    # w=137-26 : %(asctime)s %(levelname)-7s %(message)s
    # w=137-47 : %(asctime)s %(levelname)-7s %(filename)15s:%(lineno)-4d %(message)s
    # w=137 (for ipynb on Chrome using D2Coding 13pt) with scroll
    # w=139 (for ipynb on Chrome using D2Coding 13pt) without scroll
    # w=165 (for ipynb on GitHub using D2Coding 13pt)
    if len(title) == 0:
        return "\n" * t + c * w + "\n" * b
    else:
        return "\n" * t + c * 3 + f" {title} " + c * (w - len(f" {title} ") - 3) + "\n" * b


def file_hr(*args, file=sys_stdout, c=None, ct=None, cb=None, title=None, sleep_sec=0.0, **kwargs):
    with MuteStd(out=file, err=file, flush_sec=sleep_sec):
        if c:
            print(hr(*args, c=c, **kwargs), file=file)
        elif ct:
            print(hr(*args, c=ct, **kwargs), file=file)
        else:
            print(hr(*args, **kwargs), file=file)
        if title:
            print(title, file=file)
            if c:
                print(hr(*args, c=c, **kwargs), file=file)
            elif cb:
                print(hr(*args, c=cb, **kwargs), file=file)
            else:
                print(hr(*args, **kwargs), file=file)
    with MuteStd(out=sys_stdout, err=sys_stderr, flush_sec=sleep_sec):
        pass


def out_hr(*args, **kwargs):
    file_hr(*args, file=sys_stdout, **kwargs)


def err_hr(*args, **kwargs):
    file_hr(*args, file=sys_stderr, **kwargs)


def str_table(tabular_data, headers=(), tablefmt="plain", showindex="default", transposed_df=False, **kwargs):
    if not headers and isinstance(tabular_data, pd.DataFrame):
        if showindex is True or showindex == "default" or showindex == "always" or \
                not isinstance(showindex, str) and isinstance(showindex, Iterable) and len(showindex) != len(tabular_data):
            if transposed_df:
                index_header = '#'
                if tabular_data.index.name:
                    index_header = tabular_data.index.name
                elif tabular_data.index.names:
                    index_header = ' '.join(tabular_data.index.names)
                if isinstance(tabular_data.columns, pd.RangeIndex):
                    headers = [index_header] + list(map(str(range(1, len(tabular_data.columns) + 1))))
                else:
                    headers = [index_header] + list(tabular_data.columns)
            else:
                headers = ['#'] + list(tabular_data.columns)
                showindex = range(1, len(tabular_data) + 1)
        else:
            headers = tabular_data.columns
    return tabulate(tabular_data, headers=headers, tablefmt=tablefmt, showindex=showindex, **kwargs)


def to_table_lines(*args, left='', c="-", w=137, tablefmt="plain", header_idx=0, border_idx=-1, bordered=False, **kwargs):
    table = str_table(*args, **kwargs, tablefmt=tablefmt)
    lines = table.splitlines()
    if bordered:
        border = hr(c=c, w=w)
        lines = ([border] + lines[:header_idx + 1] +
                 [border] + lines[header_idx + 1:] +
                 [border])
    elif border_idx >= 0:
        border = lines[border_idx]
        lines = (
                [] +
                [border] +
                lines[:header_idx + 1] +
                lines[header_idx + 1:] +
                [border] +
                []
        )
    for line in lines if not left else [left + line for line in lines]:
        yield line


def log_table(my_logger, *args, level=logging.INFO, **kwargs):
    for line in to_table_lines(*args, **kwargs):
        my_logger.log(level, line)


def file_table(*args, file=sys_stdout, **kwargs):
    print(str_table(*args, **kwargs), file=file)


def out_table(*args, **kwargs):
    file_table(*args, file=sys_stdout, **kwargs)


def err_table(*args, **kwargs):
    file_table(*args, file=sys_stderr, **kwargs)


def flush_or(*outs, sec):
    for out in outs:
        if out and not out.closed:
            out.flush()
            if sec and sec > 0.001:
                sleep(sec)


def read_or(path: str | Path):
    if not path:
        return None
    path = Path(path)
    return path.read_text() if path.is_file() else None


def write_or(path: str | Path, data: str):
    if not path:
        return None
    path = Path(path)
    path.write_text(data)
    return file_size(path)


def exists_or(path: str | Path):
    if not path:
        return None
    path = Path(path)
    return path if path.exists() else None


def first_path_or(path: str | Path):
    if not path:
        return None
    try:
        return next(iter(paths(path)))
    except StopIteration:
        return None


def first_or(xs):
    if not xs:
        return None
    try:
        return next(iter(xs))
    except StopIteration:
        return None


def parents_and_children(path):
    path = Path(path)
    return list(path.parents) + [path] + [x.absolute() for x in dirs("*")]


def paths(path, accept_fn=lambda _: True) -> List[Path]:
    assert path, f"No path: {path}"
    path = Path(path)
    if any(c in str(path.parent) for c in ["*", "?"]):
        parents = dirs(path.parent)
    else:
        parents = [path.parent]
    return sorted([x for x in chain.from_iterable(parent.glob(path.name) for parent in parents) if accept_fn(x)])


def dirs(path) -> List[Path]:
    return paths(path, accept_fn=lambda x: x.is_dir())


def files(path) -> List[Path]:
    return paths(path, accept_fn=lambda x: x.is_file())


def non_empty_files(path) -> List[Path]:
    return paths(path, accept_fn=lambda x: x.is_file() and x.stat().st_size > 0)


def glob_dirs(path, glob: str) -> List[Path]:
    path = Path(path)
    return sorted([x for x in path.glob(glob) if x.is_dir()])


def glob_files(path, glob: str) -> List[Path]:
    path = Path(path)
    return sorted([x for x in path.glob(glob) if x.is_file()])


def count_dirs(path, key, sub=None):
    path = Path(path)
    if not sub:
        return sum(1 for x in path.glob(f"*{key}*") if x.is_dir())
    else:
        return sum(1 for x in path.glob(f"*{key}*/*{sub}*") if x.is_dir())


def count_files(path, key, sub=None):
    path = Path(path)
    if not sub:
        return sum(1 for x in path.glob(f"*{key}*") if x.is_file())
    else:
        return sum(1 for x in path.glob(f"*{key}*/**/*{sub}*") if x.is_file())


def paths_info(*xs, to_pathlist=paths, to_filename=str, sort_key=None):
    from chrisbase.util import to_dataframe
    records = []
    all_paths = []
    for path in xs:
        all_paths += to_pathlist(path)
    for f in all_paths if not sort_key else sorted(all_paths, key=sort_key):
        records.append({'file': to_filename(f), 'size': f"{file_size(f):,d}", 'time': file_mtime(f)})
    return to_dataframe(records)


def files_info(*xs, **kwargs):
    return paths_info(*xs, to_pathlist=files, **kwargs)


def dirs_info(*xs):
    return paths_info(*xs, to_pathlist=dirs)


def file_mtime(path, fmt='%Y/%m/%d %H:%M:%S'):
    return from_timestamp(Path(path).stat().st_mtime, fmt=fmt)


def file_size(path):
    return Path(path).stat().st_size


def file_lines(file, encoding='utf-8'):
    def blocks(f, size=65536):
        while True:
            block = f.read(size)
            if not block:
                break
            yield block

    with Path(file).open(encoding=encoding) as inp:
        return sum(b.count("\n") for b in blocks(inp))


def num_lines(path, encoding='utf-8', mini=None):
    assert path, f"No path: {path}"
    path = Path(path)
    full = not mini or mini <= 0
    if not full:
        return mini
    with path.open(encoding=encoding) as inp:
        return sum(1 for _ in inp)


def all_lines(path, encoding='utf-8', mini=None):
    assert path, f"No path: {path}"
    path = Path(path)
    full = not mini or mini <= 0
    with path.open(encoding=encoding) as inp:
        return map(lambda x: x.rstrip(), inp.readlines() if full else inp.readlines()[:mini])


def all_line_list(path, encoding='utf-8', mini=None):
    return list(all_lines(path, encoding=encoding, mini=mini))


def tsv_lines(*args, encoding='utf-8', **kwargs):
    return map(lambda x: x.split('\t'), all_lines(*args, encoding=encoding, **kwargs))


def key_lines(key, *args, encoding='utf-8', **kwargs):
    return [x for x in all_lines(*args, encoding=encoding, **kwargs) if key in x]


def text_blocks(path, encoding='utf-8') -> Iterable[List[str]]:
    block = []
    with open(path, mode="r", encoding=encoding) as f:
        for line in f:
            line = line[:-1]
            if not line:
                if block:
                    yield block
                    block = []
            else:
                block.append(line)

        if block:
            yield block


def new_path(path, post=None, pre=None, sep='-') -> Path:
    path = Path(path)
    new_stem = (f"{pre}{sep}" if pre is not None else "") + path.stem + (f"{sep}{post}" if post is not None else "")
    return path.parent / (new_stem + path.suffix)


def new_file(infile, outfiles, blank=('', '*', '?')) -> Path:
    infile = Path(infile)
    outfiles = Path(outfiles)
    parent = outfiles.parent
    parent.mkdir(parents=True, exist_ok=True)

    suffix1 = ''.join(infile.suffixes)
    suffix2 = ''.join(outfiles.suffixes)
    suffix = suffix1 if suffix2 in blank else suffix2

    stem1 = infile.stem.strip()
    stem2 = outfiles.stem.strip()
    stem = stem1 if any(x and x in stem2 for x in blank) else stem2

    outfile: Path = parent / f"{stem}{suffix}"
    assert infile != outfile, f"infile({infile}) == outfile({outfile})"

    return outfile


def make_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_parent_dir(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def remove_dir(path, ignore_errors=False) -> Path:
    path = Path(path)
    shutil.rmtree(path, ignore_errors=ignore_errors)
    return path


def remove_dir_check(path, real=True, verbose=False, ignore_errors=True, file=None):
    path = Path(path)
    if verbose:
        print(f"- {str(path):<40}: {OX(path.exists())}", end='', file=file)
    if path.exists() and real:
        shutil.rmtree(path, ignore_errors=ignore_errors)
    if verbose:
        print(f" -> {OX(path.exists())}", file=file)
    return not path.exists()


def remove_any(path, sleep_sec=0.0) -> Path:
    path = Path(path)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    sleep(sleep_sec)
    return path


def load_json(path: str | Path, **kwargs) -> dict:
    file = Path(path)
    assert file.exists() and file.is_file(), f"file={file}"
    try:
        with file.open() as f:
            return json.load(f, **kwargs)
    except Exception as e:
        print(f"Error occurred from [load_json(path={path})]", file=sys_stderr)
        raise RuntimeError(f"Please validate json file!\n- path: {path}\n- type: {type(e).__qualname__}\n- detail: {e}")


# define function to normalize simple list in json
def normalize_simple_list_in_json(json_input):
    json_output = []
    pattern = re.compile(r"\[[^\[\]]+?]")
    if re.search(pattern, json_input):
        pre_end = 0
        for m in re.finditer(pattern, json_input):
            json_output.append(m.string[pre_end: m.start()])
            json_output.append("[" + " ".join(m.group().split()).removeprefix("[ ").removesuffix(" ]") + "]")
            pre_end = m.end()
        json_output.append(m.string[pre_end:])
        return ''.join(json_output)
    else:
        return json_input


def open_file(path: str | Path, mode: str = "rb", **kwargs) -> IOBase:
    file = Path(path)
    assert file.exists() and file.is_file(), f"No file: {file}"
    if file.suffix == ".gz":
        return gzip.open(file, mode, **kwargs)
    elif file.suffix == ".bz2":
        return bz2.open(file, mode, **kwargs)
    else:
        return file.open(mode, **kwargs)


def save_json(obj: dict | list, path: str | Path, **kwargs):
    file = make_parent_dir(Path(path))
    with file.open("w") as f:
        json.dump(obj, f, **kwargs)


def _path_to_str(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _path_to_str(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_path_to_str(v) for v in obj]
    return obj


def to_yaml(conf, *, resolve=False, sort_keys=False, **kwds):
    if not OmegaConf.is_config(conf):
        conf = OmegaConf.create(conf)
    container = _path_to_str(OmegaConf.to_container(conf, resolve=resolve, enum_to_str=True))
    return yaml.dump(container, Dumper=get_omega_conf_dumper(),
                     default_flow_style=False, allow_unicode=True, sort_keys=sort_keys, **kwds)


def save_yaml(conf, path, *, resolve=False, sort_keys=False):
    yaml_str = to_yaml(conf, resolve=resolve, sort_keys=sort_keys, width=4096)
    output_file = Path(path)
    output_file.write_text(yaml_str)
    return output_file


def merge_dicts(*xs) -> dict:
    items = list()
    for x in xs:
        if x:
            items += x.items()
    return dict(items)


def merge_lists(*xs) -> list:
    items = list()
    for x in xs:
        if x:
            items += x
    return items


def merge_sets(*xs) -> set:
    items = set()
    for x in xs:
        for e in x:
            items.add(e)
        # items = items.union(x)
    return items


def pop_keys(dic, keys) -> dict:
    if isinstance(dic, dict):
        for k in tupled(keys):
            if k in dic:
                dic.pop(k)
    return dic


def copy_dict(src: dict, dst: dict or None = None, keys=None) -> dict:
    if dst is None:
        dst = dict()
    else:
        dst = dict(dst)
    if keys is None:
        keys = tuple(src.keys())
    else:
        keys = tupled(keys)
    for k in keys:
        if k in src:
            dst[k] = src[k]
    return dst


def dict_to_cmd_args(obj: dict, keys=None) -> list:
    if not keys:
        keys = obj.keys()
    return list(chain.from_iterable([(f"--{key}", obj[key]) for key in keys]))


def dict_to_pairs(obj: dict, keys=None, eq='=') -> list:
    if not keys:
        keys = obj.keys()
    return [f"{key}{eq}{obj[key]}" for key in keys]


def save_attrs(obj: dict, file, keys=None, excl=None):
    if keys is not None and isinstance(keys, (list, tuple, set)):
        keys = [x for x in keys if x in obj.keys()]
    else:
        keys = obj.keys()
    if excl is not None and isinstance(excl, (list, tuple, set)):
        keys = [x for x in keys if x not in excl]
    save_json({key: obj[key] for key in keys}, file, ensure_ascii=False, indent=2, default=str)


def save_rows(rows, file, open_mode='w', keys=None, excl=None, with_column_name=False):
    rows = iter(rows)
    first = next(rows)
    if keys is not None and isinstance(keys, (list, tuple, set)):
        keys = [x for x in keys if x in first.keys()]
    else:
        keys = first.keys()
    if excl is not None and isinstance(excl, (list, tuple, set)):
        keys = [x for x in keys if x not in excl]
    with file.open(open_mode) as out:
        if with_column_name:
            print('\t'.join(keys), file=out)
        for row in chain([first], rows):
            print('\t'.join(map(str, [row[k] for k in keys])), file=out)


def run_command(*args, title=None, mt=0, mb=0, pt=0, pb=0, rt=0, rb=0, rc='-', bare=True, verbose=True, real=True):
    from chrisbase.data import JobTimer
    with JobTimer(name=None if bare else f"run_command({title})" if title else f"run_command{args}",
                  verbose=verbose, mt=mt, mb=mb, pt=pt, pb=pb, rt=rt, rb=rb, rc=rc) as scope:
        if real:
            subprocess.run(list(map(str, args)), stdout=None if verbose else scope.mute, stderr=None if verbose else scope.mute)


def read_command_out(*args):
    return subprocess.run(list(map(str, args)), stdout=subprocess.PIPE).stdout.decode('utf-8')


def read_command_err(*args):
    return subprocess.run(list(map(str, args)), stderr=subprocess.PIPE).stderr.decode('utf-8')


def get_valid_lines(lines,
                    accumulating_querys=(
                            ("(Epoch ", "training #1"),
                            ("(Epoch ", "metering #1"),
                    )):
    last_lines = {query: None for query in accumulating_querys}
    for line in lines:
        changed = True
        if len(str(line).strip()) > 0:
            for query in last_lines:
                if all(q in line for q in query):
                    last_lines[query] = line
                    changed = False
        if changed:
            for query in last_lines:
                if last_lines[query]:
                    yield last_lines[query]
                    last_lines[query] = None
            if len(str(line).strip()) > 0:
                yield line


def trim_output(infile, outfile):
    infile = Path(infile)
    outfile = Path(outfile)
    outfile.write_text('\n'.join(get_valid_lines(all_lines(infile))))


def get_hostname() -> str:
    return socket.gethostname()


def get_hostaddr(default="127.0.0.1") -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as st:
            st.connect(("8.8.8.8", 80))
            r = first_or(st.getsockname())
            return r if r else default
    except OSError:
        return default


def yield_local_addrs():
    for inf in netifaces.interfaces():
        inf_addrs = netifaces.ifaddresses(inf).get(netifaces.AF_INET)
        if inf_addrs:
            for inf_addr in [x.get('addr') for x in inf_addrs]:
                if inf_addr and IPv4Address(inf_addr).is_global:
                    yield inf_addr


class HttpClients(Iterable[httpx.Client]):
    http_clients = None

    def __init__(self, ip_addrs: Iterable[str]):
        self.http_clients = [
            httpx.Client(
                transport=httpx.HTTPTransport(local_address=ip_addr),
                timeout=httpx.Timeout(timeout=120.0),
            ) for ip_addr in ip_addrs
        ]

    def __iter__(self):
        return iter(self.http_clients)

    def __len__(self):
        return len(self.http_clients)

    def __getitem__(self, ii):
        return self.http_clients[ii % len(self)]

    def get_local_addr(self, ii):
        return self[ii]._transport._pool._local_address

    def __del__(self):
        if self.http_clients:
            for http_client in self.http_clients:
                http_client.close()


def get_http_clients():
    return HttpClients(yield_local_addrs())


def prepend_to_global_path(*xs):
    os.environ['PATH'] = os.pathsep.join(map(str, xs)) + os.pathsep + os.environ['PATH']


def append_to_global_path(*xs):
    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.pathsep.join(map(str, xs))


def environ_to_dataframe(max_value_len=200, columns=None):
    from chrisbase.util import to_dataframe
    return to_dataframe(copy_dict(dict(os.environ),
                                  keys=[x for x in sorted(os.environ.keys()) if len(str(os.environ[x])) <= max_value_len]),
                        columns=columns)


def setup_unit_logger(level=logging.INFO, force=True,
                      stream=sys_stdout, filename: str | Path = None, filemode="a", existing_content=None,
                      fmt=logging.BASIC_FORMAT, datefmt="[%m.%d %H:%M:%S]"):
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handlers = make_logging_handlers(formatter=formatter, stream=stream, filename=filename, filemode=filemode, existing_content=existing_content)[-1:]
    logging.basicConfig(level=level, force=force, handlers=handlers)
    update_existing_handlers(handlers=handlers)


def setup_dual_logger(level=logging.INFO, force=True,
                      stream=sys_stdout, filename: str | Path = "running.log", filemode="a", existing_content=None,
                      fmt=logging.BASIC_FORMAT, datefmt="[%m.%d %H:%M:%S]"):
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handlers = make_logging_handlers(formatter=formatter, stream=stream, filename=filename, filemode=filemode, existing_content=existing_content)
    logging.basicConfig(level=level, force=force, handlers=handlers)
    update_existing_handlers(handlers=handlers)


def make_logging_handlers(formatter, stream, filename, filemode, existing_content=None):
    handlers = []
    if stream:
        h = logging.StreamHandler(stream=stream)
        h.setFormatter(formatter)
        handlers.append(h)
    if filename and filemode:
        h = logging.FileHandler(filename=make_parent_dir(filename), mode=filemode, encoding="utf-8")
        h.setFormatter(formatter)
        handlers.append(h)
        if existing_content:
            h.stream.write(existing_content)
    assert len(handlers) > 0, f"Empty handlers: filename={filename}, filemode={filemode}, stream={stream}"
    return handlers


def update_existing_handlers(handlers, debug=False):
    for x in logging.Logger.manager.loggerDict.values():
        if isinstance(x, logging.Logger):
            if len(x.handlers) > 0:
                for h in x.handlers:
                    if isinstance(h, logging.FileHandler) and h.stream and not h.stream.closed:
                        h.stream.close()
                x.handlers.clear()
                for h in handlers:
                    x.addHandler(h)
                if debug:
                    logger.debug(f"logging.getLogger({x.name:<20s}) = Logger(level={x.level}, handlers={x.handlers}, disabled={x.disabled}, propagate={x.propagate}, parent={x.parent})")


def set_verbosity_debug(*names):
    for name in names:
        logging.getLogger(name).setLevel(logging.DEBUG)


def set_verbosity_info(*names):
    for name in names:
        logging.getLogger(name).setLevel(logging.INFO)


def set_verbosity_warning(*names):
    for name in names:
        logging.getLogger(name).setLevel(logging.WARNING)


def set_verbosity_error(*names):
    for name in names:
        logging.getLogger(name).setLevel(logging.ERROR)


def do_nothing(*args, **kwargs):
    pass


def info_r(x, *y, **z):
    x = str(x).rstrip()
    logger.info(x, *y, **z)


def tb_events_to_csv(
        event_file: str | Path,  # 단일 event 파일 경로 (예: "output/runs/events.out.tfevents.xxxx")
        output_file: str | Path,  # 내보낼 CSV 경로
        purge_orphaned_data=True,
):
    """
    지정한 TensorBoard 이벤트 로그(event_file)를 파싱하여
    CSV로 저장합니다.

    - out_csv_path가 이미 존재하면 덮어씁니다.
    - event_file에 여러 Scalar 태그가 존재할 경우, 모든 태그를 모아
      [wall_time, step, tag, value] 형태로 CSV 파일에 기록합니다.
    """
    ea = event_accumulator.EventAccumulator(
        str(event_file),
        purge_orphaned_data=purge_orphaned_data  # or False
    )
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wall_time", "step", "tag", "value"])  # CSV 헤더

        for tag in scalar_tags:
            # ea.Scalars(tag)는 해당 tag로 기록된 전체 log를 리스트 형태로 반환
            for event in ea.Scalars(tag):
                writer.writerow([
                    event.wall_time,  # float(Unix 시간)
                    event.step,  # int(전역 global_step)
                    tag,  # 예: "eval/loss", "train/loss" 등
                    event.value  # 실제 측정값
                ])


def convert_all_events_in_dir(log_dir: str | Path):
    """
    Converts all TensorBoard event files in `log_dir` to CSV.
    Each event file produces a separate CSV file.
    """
    input_files = os.path.join(log_dir, "**/events.out.tfevents.*")
    for input_file in files(input_files):
        if not input_file.name.endswith(".csv"):
            output_file = input_file.with_name(input_file.name + ".csv")
            logger.info(f"Convert {input_file} to csv")
            tb_events_to_csv(input_file, output_file)


def strip_lines(text):
    return "\n".join([line.strip() for line in text.splitlines()])
