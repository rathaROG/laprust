"""Shared fixtures and explicit opt-ins for checks needing extra installations."""
import importlib
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-cross-package", action="store_true",
                     help="require both installed packages and test them together")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-cross-package"):
        skip = pytest.mark.skip(reason="enable --run-cross-package for this integration check")
        for item in items:
            if item.get_closest_marker("cross_package"):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def solver():
    return importlib.import_module("laprust")


@pytest.fixture(scope="session")
def both_packages():
    # Opted-in jobs must fail, rather than skip, if either install is missing.
    return tuple(importlib.import_module(name) for name in ("lapcpp", "laprust"))
