from __future__ import annotations

import dataclasses
import logging
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import groupby
from operator import itemgetter, attrgetter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tqdm.std as tqdm_std
from sqlalchemy.util import OrderedSet

logger = logging.getLogger(__name__)

pd.options.display.width = 3000
pd.options.display.max_rows = 3000
pd.options.display.max_columns = 300
pd.options.display.max_colwidth = 300
plt.rcParams['figure.figsize'] = [10, 5]

# https://www.eso.org/~ndelmott/ascii.html
NO = ''  # None
HT = chr(9)  # Horizontal Tab
LF = chr(10)  # Line Feed
SP = chr(32)  # Space
FS = chr(47)  # Forward Slash
BS = chr(92)  # Backward Slash
VB = chr(124)  # Vertical Bar
PR = chr(46)  # Period
CM = chr(44)  # Comma
CO = chr(58)  # Colon
SC = chr(59)  # Semicolon
ES = chr(61)  # Equals Sign
LT = chr(60)  # Less-Than Sign
GT = chr(62)  # Greater-Than Sign
MS = chr(45)  # Minus Sign
PS = chr(43)  # Plus Sign
AS = chr(42)  # Asterisk
LB = chr(123)  # Left Brace
RB = chr(125)  # Right Brace
LP = chr(40)  # Left Parenthesis
RP = chr(41)  # Right Parenthesis
LK = chr(91)  # Left Bracket
RK = chr(93)  # Right Parenthesis
DQ = chr(34)  # Double Quote
SQ = chr(39)  # Single Quote


def OK(x: bool):
    return 'OK' if x else 'NO'


def OX(x: bool):
    return 'O' if x else 'X'


def ox(x: bool):
    return 'o' if x else 'x'


def tupled(x: any):
    if x and not isinstance(x, (tuple, list, set)):
        return x,
    else:
        return x


def shuffled(xs, seed=0, ran=None):
    if ran is None:
        ran = random.Random(seed)
    if not isinstance(xs, list):
        xs = list(xs)
    ran.shuffle(xs)
    return xs


def grouped(xs, **kwargs):
    key = kwargs.pop('key', None)
    if 'itemgetter' in kwargs:
        key = itemgetter(*tupled(kwargs.pop('itemgetter')))
    elif 'attrgetter' in kwargs:
        key = attrgetter(*tupled(kwargs.pop('attrgetter')))
    return groupby(sorted(xs, key=key, **kwargs), key=key)


def number_only(x):
    return NO.join(c for c in str(x) if c.isdigit())


def no_space(x, repl='＿'):
    return NO.join(c if c != ' ' else repl for c in str(x))


def no_replacement(x, repl='﹏'):
    return NO.join(c if c != '�' else repl for c in str(x))


def no_nonprintable(x, repl='﹍'):
    return NO.join(c if c.isprintable() else repl for c in str(x))


def percent(x, fmt='5.1f'):
    return f'{100 * x:{fmt}}%'


def to_prefix(x, sep='=', maxsplit=1, idx=0):
    return x.rsplit(sep, maxsplit=maxsplit)[idx]


def to_postfix(x, sep='=', maxsplit=1, idx=-1):
    return x.split(sep, maxsplit=maxsplit)[idx]


def counts_str(counts, name=None, ks=None, name_fmt='>10', key_fmt='>9', num_fmt='>9,', per_fmt='5.1f'):
    ks = sorted(counts.keys()) if ks is None else ks
    sx = sum(counts.values())
    head = f"{name:{name_fmt}} : " if name is not None else ""
    body = f"{sx:>10,} || {' | '.join(f'{k:{key_fmt}} = {counts[k]:{num_fmt}}[{percent(counts[k] / sx, fmt=per_fmt)}]' for k in ks)}"
    return head + body


def to_dataframe(raw, index=None, exclude=None, columns=None, data_exclude=None, data_prefix=None):
    if dataclasses.is_dataclass(raw):
        if not columns:
            columns = ["key", "value"]
        raw = {(f"{data_prefix}.{k}" if data_prefix else k): v
               for k, v in asdict(raw).items()
               if not data_exclude or k not in data_exclude}
        return to_dataframe(raw, index=index, exclude=exclude, columns=columns)
    elif isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], dict):
            return pd.DataFrame.from_records(raw, index=index, exclude=exclude, columns=columns)
        else:
            return pd.DataFrame.from_records([x for x in raw],
                                             index=index, exclude=exclude, columns=columns)
    elif isinstance(raw, dict):
        if not columns:
            columns = ["key", "value"]
        return pd.DataFrame.from_records(tuple(raw.items()),
                                         index=index, exclude=exclude, columns=columns)
    else:
        return pd.DataFrame.from_records(raw, index=index, exclude=exclude, columns=columns)


morpheme_pattern = re.compile("([^ ]+?/[A-Z]{2,3})[+]?")


def to_morphemes(text: str, pattern=morpheme_pattern):
    return ' '.join(x.group(1) for x in pattern.finditer(text))


def append_intersection(a, b):
    return list(OrderedSet(a).difference(b)) + list(OrderedSet(a).intersection(b))


def display_histogram(seqs, figsize=(10, 5), dpi=80, bins=20, rwidth=0.8, yaxis_major=-1, yaxis_minor=-1, title=None, show=True):
    plt.figure(figsize=figsize, dpi=dpi)
    axes = plt.axes()
    if yaxis_major > 0:
        axes.yaxis.set_major_locator(ticker.MultipleLocator(yaxis_major))
    if yaxis_minor > 0:
        axes.yaxis.set_minor_locator(ticker.MultipleLocator(yaxis_minor))
    _, edges, _ = plt.hist(seqs.values(), bins=bins, rwidth=rwidth)
    plt.xticks(np.round(edges), rotation=45)
    plt.legend(seqs.keys())
    plt.grid(True, alpha=0.7)
    if title:
        plt.title(title)
    if show:
        plt.show()


class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        def empty_fn(*args, **kwargs):
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return


class empty_tqdm_cls:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None

    def get_lock(self):
        pass


class mute_tqdm_cls:
    def to_desc(self, desc, pre=None):
        return NO.join([
            f'{pre} ' if pre else '',
            f'{desc:{self.aline}{self.desc_size}s}',
        ])

    def __init__(self, bar_size=50, desc_size=10, file=open(os.devnull, 'w'), aline='right'):
        self.desc_size = desc_size
        self.bar_size = bar_size
        self.file = file
        self.aline = '<' if str(aline).strip().lower() == 'left' else '>'

    def __call__(self, *args, **kwargs):
        if 'desc' not in kwargs or not kwargs['desc']:
            kwargs['desc'] = 'processing'
        kwargs['desc'] = self.to_desc(desc=kwargs['desc'],
                                      pre=kwargs.pop('pre') if 'pre' in kwargs else None)
        kwargs.pop('file', None)
        kwargs.pop('bar_format', None)
        return tqdm_std.tqdm(*args, bar_format=f"{{l_bar}}{{bar:{self.bar_size}}}{{r_bar}}", file=self.file, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None
        return tqdm_std.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        return tqdm_std.tqdm.get_lock()


def terminate_processes(pool: ProcessPoolExecutor):
    for proc in pool._processes.values():
        if proc.is_alive():
            proc.terminate()
