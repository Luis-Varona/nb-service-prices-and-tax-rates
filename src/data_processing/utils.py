# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
import logging

from typing import Callable


# %%
def suppress_fastexcel_logging(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("fastexcel.types.dtype")
        default_level = logger.getEffectiveLevel()
        logger.setLevel(logging.ERROR)

        try:
            return func(*args, **kwargs)
        finally:
            logger.setLevel(default_level)

    return wrapper
