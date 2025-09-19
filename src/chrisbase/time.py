from __future__ import annotations

import os
import signal
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import torch
import torch.distributed as dist

try:
    import accelerate
    import accelerate.utils
except Exception:
    accelerate = None
    _HAS_ACCELERATE = False

try:
    import deepspeed
    _HAS_DEEPSPEED = True
except Exception:
    deepspeed = None
    _HAS_DEEPSPEED = False


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
    start_time = now_stamp()
    return sorted(accelerate.utils.gather_object([start_time]))[0]


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _is_dist_initialized() else 1


def _rank() -> int:
    return dist.get_rank() if _is_dist_initialized() else 0


def _backend() -> Optional[str]:
    return dist.get_backend() if _is_dist_initialized() else None


def wait_for_everyone(local_cuda_sync: bool = True) -> None:
    """
    모든 프로세스가 여기서 만날 때까지 대기시킨 뒤 동시에 진행합니다.

    Args:
        local_cuda_sync (bool): NCCL 사용 시, 배리어 전/후로 CUDA 스트림을 로컬 동기화합니다.
                                커널이 여전히 실행 중인 상태로 배리어에 들어가 생기는 미묘한
                                교착을 줄이는 데 도움이 됩니다.
    Notes:
        - torch.distributed.barrier()를 사용합니다.
        - 분산 미초기화/싱글 프로세스면 아무 것도 하지 않습니다.
    """
    # 로컬 CUDA 스트림 동기화(선택)
    if local_cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    # 표준 torch.distributed 배리어
    if _is_dist_initialized() and _world_size() > 1:
        if _backend() == "nccl" and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.current_device() != local_rank:
                torch.cuda.set_device(local_rank)
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
        if local_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()


@contextmanager
def everyone_waits(tag: str = "", local_cuda_sync: bool = True):
    """
    with 블록 진입 직전에 모두 대기시키고, 블록 종료 후에도 다시 한번 대기시킵니다.
    디버깅이나 체크포인트 전후, 데이터 경계 구간에 유용합니다.

    Example:
        with everyone_waits("before_eval"):
            # 모든 rank가 여기서 맞춰진 뒤 평가 시작
            run_evaluation()
    """
    wait_for_everyone(local_cuda_sync=local_cuda_sync)
    try:
        yield
    finally:
        wait_for_everyone(local_cuda_sync=local_cuda_sync)


def sync_point(fn=None, *, tag: str = "", local_cuda_sync: bool = True):
    """
    함수 앞뒤로 전역 배리어를 자동으로 걸어주는 데코레이터.

    Example:
        @sync_point(tag="save_ckpt")
        def save_ckpt(...):
            ...
    """

    def _decorator(func):
        def _wrapped(*args, **kwargs):
            wait_for_everyone(local_cuda_sync=local_cuda_sync)
            out = func(*args, **kwargs)
            wait_for_everyone(local_cuda_sync=local_cuda_sync)
            return out

        _wrapped.__name__ = getattr(func, "__name__", "wrapped_sync_point")
        _wrapped.__doc__ = getattr(func, "__doc__", None)
        return _decorator if fn is None else _wrapped

    return _decorator if fn is None else _decorator(fn)


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


@contextmanager
def timeout_handler(seconds):
    def timeout_function(signum, frame):
        raise TimeoutError(f"API call timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_function)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Reset the alarm and handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
