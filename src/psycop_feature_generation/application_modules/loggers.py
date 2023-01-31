"""Recommended logger to use."""
import logging
from datetime import datetime

import coloredlogs

from psycop_feature_generation.application_modules.project_setup import ProjectInfo


def init_root_logger(
    project_info: ProjectInfo,
    stdout_log_level: int = logging.INFO,
    log_file_level: int = logging.DEBUG,
) -> None:
    """Initializes the root logger with a file handler and a stream handler."""
    # Get the root logger
    root_log = logging.getLogger()

    # Set the root logger's level to the minimum of the stdout and file log
    # The root logger acts as the trunk of a tree, only messages with a level
    # equal to or higher than the root logger's level will be passed on to
    # the root logger's its branches (its handlers)
    root_log.setLevel(min(stdout_log_level, log_file_level))

    # Install the coloredlogs module on the root logger
    # to get prettier console output and to add a streamhandler,
    # which will write all logging messages from the root logger to
    # stdout
    coloredlogs.install(
        level=stdout_log_level,
        fmt="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Create a timestamped file handler which writes all logging messages from
    # the root logger to a file
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir_path = project_info.feature_set_path / "logs"
    log_dir_path.mkdir(exist_ok=True, parents=True)

    file_handler = logging.FileHandler(filename=log_dir_path / f"{now}.log")
    file_handler.setLevel(log_file_level)
    root_log.addHandler(file_handler)
