import multiprocessing
import platform
import re
import shutil
from pathlib import Path
from typing import Optional

from invoke import Context, Result, task

from automation.git import is_uncommitted_changes
from automation.windows import NOT_WINDOWS

NOT_WINDOWS = platform.system() != "Windows"


def on_ovartaci() -> bool:
    import platform

    if platform.node() == "RMAPPS1279":
        print(f"\n{msg_type.GOOD} On Ovartaci")
        return True

    print(f"\n{msg_type.GOOD} Not on Ovartaci")
    return False
