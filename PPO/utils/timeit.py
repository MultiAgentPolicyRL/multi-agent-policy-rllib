from functools import wraps
import logging
import time


def timeit(func):  # pylint: disable = missing-function-docstring
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logging.debug(f"Function %s Took %f seconds", func.__name__, total_time)
        return result

    return timeit_wrapper
