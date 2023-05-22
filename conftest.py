import pytest
from valohai.internals import global_state


@pytest.fixture()
def valohai_utils_global_state():
    global_state.flush_global_state()
    return global_state
