import logging
from concurrent.futures import ProcessPoolExecutor, Future, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import List

from chrisbase.util import time_tqdm_cls

logger = logging.getLogger(__name__)


def future_result(job: Future, timeout=None, default=None):
    try:
        return job.result(timeout=timeout)
    except TimeoutError:
        job.cancel()
        return default
    except BrokenProcessPool:
        return default
    # except Exception as e:
    #     print()
    #     logger.warning(f"[{type(e)}] on future_result: (e.args={e.args}) {e}")
    #     return default


def all_future_results(pool: ProcessPoolExecutor, jobs: List[Future], default=None, timeout=None, use_tqdm=True):
    loop = jobs
    if use_tqdm:
        tqdm = time_tqdm_cls(bar_size=100, desc_size=9)
        loop = tqdm(jobs, pre="┇", desc="crawling", unit="jobs")
    for job in loop:
        future_result(job, default=default, timeout=timeout)
    for process in pool._processes.values():
        process.terminate()
    pool.shutdown()
    return [future_result(job, default=default, timeout=0) for job in jobs]
