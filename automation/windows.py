import multiprocessing
import platform
import re
import shutil
from pathlib import Path
from typing import Optional

from invoke import Context, Result, task

NOT_WINDOWS = platform.system() != "Windows"
from .logger import msg_type


def on_ovartaci() -> bool:
    import platform

    if platform.node() == "RMAPPS1279":
        print(f"\n{msg_type.GOOD} On Ovartaci")
        return True

    print(f"\n{msg_type.GOOD} Not on Ovartaci")
    return False
