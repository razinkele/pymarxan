"""Planning unit status constants.

Canonical definitions live in :mod:`pymarxan.models.problem`.
This module re-exports them under shorter aliases for convenience.
"""

from pymarxan.models.problem import (
    STATUS_INITIAL_INCLUDE as INITIAL_INCLUDE,
)
from pymarxan.models.problem import (
    STATUS_LOCKED_IN as LOCKED_IN,
)
from pymarxan.models.problem import (
    STATUS_LOCKED_OUT as LOCKED_OUT,
)
from pymarxan.models.problem import (
    STATUS_NORMAL as AVAILABLE,
)

VALID_STATUSES = {AVAILABLE, INITIAL_INCLUDE, LOCKED_IN, LOCKED_OUT}
