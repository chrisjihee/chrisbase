from concurrent.futures import ProcessPoolExecutor, Future, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import List


def future_result(job: Future, timeout=None, default=None):
    try:
        return job.result(timeout=timeout)
    except TimeoutError:
        job.cancel()
        return default
    except BrokenProcessPool:
        return default


def all_future_results(pool: ProcessPoolExecutor, jobs: List[Future], default=None, timeout=None):
    for job in jobs:
        future_result(job, default=default, timeout=timeout)
    for process in pool._processes.values():
        process.terminate()
    pool.shutdown()
    return [future_result(job, default=default, timeout=0) for job in jobs]
