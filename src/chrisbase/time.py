import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone


def elasped_sec(x, *args, **kwargs):
    t1 = datetime.now()
    return x(*args, **kwargs), datetime.now() - t1


def now(fmt='[%m.%d %H:%M:%S]', prefix=None, delay=0) -> str:
    if delay:
        time.sleep(delay)
    if prefix:
        return f"{prefix} {datetime.now().strftime(fmt)}"
    else:
        return datetime.now().strftime(fmt)


def after(delta: timedelta, fmt='[%m.%d %H:%M:%S]', prefix=None):
    if prefix:
        return f"{prefix} {(datetime.now() + delta).strftime(fmt)}"
    else:
        return (datetime.now() + delta).strftime(fmt)


def before(delta: timedelta, fmt='[%m.%d %H:%M:%S]', prefix=None):
    if prefix:
        return f"{prefix} {(datetime.now() - delta).strftime(fmt)}"
    else:
        return (datetime.now() - delta).strftime(fmt)


def now_stamp(delay=0) -> float:
    if delay:
        time.sleep(delay)
    return datetime.now().timestamp()


def from_timestamp(stamp, fmt='[%m.%d %H:%M:%S]'):
    return datetime.fromtimestamp(stamp, tz=timezone.utc).astimezone().strftime(fmt)


def str_delta(x: timedelta):
    mm, ss = divmod(x.total_seconds(), 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02.0f}:{mm:02.0f}:{ss:06.3f}"


def gather_start_time() -> float:
    import accelerate.utils
    start_time = now_stamp()
    return sorted(accelerate.utils.gather_object([start_time]))[0]


def wait_for_everyone():
    import accelerate.utils
    return accelerate.utils.wait_for_everyone()


@contextmanager
def run_only_rank_zero(rank: int = int(os.getenv("LOCAL_RANK", -1))):
    wait_for_everyone()
    try:
        if rank == 0:
            yield
        else:
            yield None
    finally:
        wait_for_everyone()


@contextmanager
def run_only_main_process(is_main_process: bool = True):
    wait_for_everyone()
    try:
        if is_main_process:
            yield
        else:
            yield None
    finally:
        wait_for_everyone()


@contextmanager
def flush_and_sleep(delay: float = 0.1):
    try:
        yield
    finally:
        try:
            sys.stderr.flush()
            sys.stdout.flush()
        except Exception:
            pass
        time.sleep(delay)
