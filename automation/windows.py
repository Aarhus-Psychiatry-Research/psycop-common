import multiprocessing
import platform
import re
import shutil
from pathlib import Path
from typing import Optional

from invoke import Context, Result, task

NOT_WINDOWS = platform.system() != "Windows"


def on_ovartaci() -> bool:
    import platform

    if platform.node() == "RMAPPS1279":
        print("On Ovartaci")
        return True

    print("Not on Ovartaci")
    return False
