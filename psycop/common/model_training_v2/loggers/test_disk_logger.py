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

    logged_config = Config().from_disk(logger.cfg_log_path)
    assert logged_config == config