"""Explicit opt-in for checks that build and install into temporary environments."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-editable", action="store_true",
                     help="build editable/wheel installs in a fresh temporary virtualenv")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-editable"):
        skip = pytest.mark.skip(reason="enable --run-editable for this integration check")
        for item in items:
            if item.get_closest_marker("editable"):
                item.add_marker(skip)
