import pytest


@pytest.fixture(scope="session")
def spark():
    from stream.config import spark_session
    s = spark_session("tests", shuffle=2)
    yield s
    s.stop()


@pytest.fixture
def sample_events():
    from stream.generate import generate
    return generate(2000, seed=123)
