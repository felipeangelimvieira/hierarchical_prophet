from typing import Any, Literal, Type

import numpyro


def create_literal_type(valid_values: list[str]) -> Any:
    """Create a Literal Type from list."""
    valid_values_tuple = tuple(valid_values)
    return Literal[valid_values_tuple]


OptimizerNames = create_literal_type([*numpyro.optim.__all__, "optax"])
