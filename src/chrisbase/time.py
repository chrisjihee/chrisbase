from datetime import datetime, timedelta, timezone


def elasped_sec(x, *args, **kwargs):
    t1 = datetime.now()
    return x(*args, **kwargs), datetime.now() - t1


def now(fmt='[%m.%d %H:%M:%S]', prefix=None):
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


def from_timestamp(stamp, fmt='%Y/%m/%d %H:%M:%S'):
    return datetime.fromtimestamp(stamp, tz=timezone.utc).astimezone().strftime(fmt)


def str_delta(x: timedelta):
    mm, ss = divmod(x.total_seconds(), 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02.0f}:{mm:02.0f}:{ss:06.3f}"
