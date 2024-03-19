from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_python_command(command: List[str]):
    """Default method for executing shell commands with pytest and sh package."""
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg=msg)


def run_sh_bash_command(command: List[str]):
    """Default method for executing shell commands with pytest and sh package."""
    msg = None
    try:
        sh.bash(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg=msg)
