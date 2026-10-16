import pytest

from fair_intervention.data.synthetic import SyntheticConfig, generate
from fair_intervention.split import horizon_splits


@pytest.fixture(scope="session")
def tables():
    """A smaller synthetic dataset: four presentations x two modules x 400 students."""
    return generate(SyntheticConfig(students_per_presentation=400, seed=3))


@pytest.fixture(scope="session")
def splits(tables):
    return horizon_splits(tables)
