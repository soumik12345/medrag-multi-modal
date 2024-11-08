import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model-name",
        action="store",
        default="gemini-1.5-flash",
        help="Model name to use for evaluation",
    )


def pytest_configure(config):
    # Add model_name to pytest namespace for access in tests
    config.addinivalue_line(
        "markers", "model_name: mark test to run with specific model name"
    )


@pytest.fixture
def model_name(request):
    return request.config.getoption("--model-name")
