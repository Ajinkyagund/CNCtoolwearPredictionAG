"""Shared fixtures: a small synthetic dataset with the same columns as the NASA milling features."""
import pytest

from cncwear.config import load_config
from tests.helpers import make_synthetic

@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def quick_cfg():
    return load_config(quick=True)


@pytest.fixture()
def raw_df():
    return make_synthetic()
