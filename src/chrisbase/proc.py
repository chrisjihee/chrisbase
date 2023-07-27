import logging
import time
from concurrent.futures import ProcessPoolExecutor, Future, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import Iterable

logger = logging.getLogger(__name__)


def get_result(job: Future, timeout=None, default=None):
    try:
        return job.result(timeout=timeout)
    except TimeoutError:
        # job.cancel()
        return default
    except BrokenProcessPool:
        return default
    except Exception as e:
        print()
        logger.warning(f"{type(e).__qualname__} on future_result(job={job}, timeout={timeout}, default={default}): {e}")
        return default


def gather_results(pool: ProcessPoolExecutor, jobs: Iterable[Future], default=None, timeout=None):
    for job in jobs:
        get_result(job, default=default, timeout=timeout)
    for proc in pool._processes.values():
        proc.terminate()
    pool.shutdown()
    time.sleep(timeout)  # TODO: 과연 필요한가?
    return [get_result(job, default=default, timeout=0) for job in jobs]
