import pytest

from vllm_bench.mock_server import start_background


@pytest.fixture(scope="session")
def mock_url():
    srv, url = start_background(token_delay_ms=1.0, prefill_ms_per_1k=5.0)
    yield url
    srv.shutdown()
    srv.server_close()
