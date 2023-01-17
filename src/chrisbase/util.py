import random
import re
from itertools import groupby
from operator import itemgetter, attrgetter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sqlalchemy.util import OrderedSet

pd.options.display.width = 3000
pd.options.display.max_rows = 3000
pd.options.display.max_columns = 300
pd.options.display.max_colwidth = 300

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


def to_dataframe(records, index=None, exclude=None, columns=None):
    return pd.DataFrame.from_records(records, index=index, exclude=exclude, columns=columns)


morpheme_pattern = re.compile("([^ ]+?/[A-Z]{2,3})[+]?")


def to_morphemes(text: str, pattern=morpheme_pattern):
    return ' '.join(x.group(1) for x in pattern.finditer(text))


def append_intersection(a, b):
    return list(OrderedSet(a).difference(b)) + list(OrderedSet(a).intersection(b))


def display_histogram(seqs, figsize=(12, 6), dpi=100, bins=20, rwidth=0.8, yaxis_major=-1, yaxis_minor=-1, title=None, show=True):
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
