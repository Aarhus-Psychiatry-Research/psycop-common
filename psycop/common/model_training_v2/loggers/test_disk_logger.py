import json
from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.loggers.disk_logger import DiskLogger


def test_disklogger_text(tmpdir: str):
    logger = DiskLogger(experiment_path=tmpdir)

    message = "This is a test message"
    logger.info(message)

    with logger.log_path.open("r") as file:
        logged_text = file.read()

    assert message in logged_text


def test_disklogger_log_config(tmpdir: str):
    logger = DiskLogger(experiment_path=tmpdir)

    config = Config({"parameters": {"param1": 123, "param2": "abc"}})
    logger.log_config(config)

    config_dict = json.loads(Path(logger.cfg_log_path).read_text())
    logged_config = Config(config_dict)
    assert logged_config == config


def test_disklogger_file(tmpdir: str):
    logger = DiskLogger(experiment_path=f"{tmpdir}/logger_dir")

    # Create the test file
    test_file = Path(tmpdir) / "test_file.txt"
    test_str = "This is a test file"
    test_file.write_text(test_str)

    # Log the artifact
    logger.log_artifact(local_path=test_file)

    # Check that the test file exists at the artifact path
    assert test_str in (logger.experiment_path / "test_file.txt").read_text()
