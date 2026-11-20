import pandas as pd
import pytest

from frankfurt_rent.clean import clean
from frankfurt_rent.generate import generate_raw


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return generate_raw(n=1500, seed=7)


@pytest.fixture(scope="session")
def cleaned(raw: pd.DataFrame) -> pd.DataFrame:
    return clean(raw)[0]
