import pytest

from psycop.common.model_training_v2.loggers.multi_logger import MultiLogger
from psycop.common.model_training_v2.loggers.terminal_logger import TerminalLogger


def test_multilogger(capsys: pytest.CaptureFixture[str]):
    logger = MultiLogger(TerminalLogger(), TerminalLogger())
    logger.fail("Both loggers work!")
    captured = capsys.readouterr()
    assert captured.out.count("Both loggers work!") == 2
