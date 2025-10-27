import os
from typing import List


def ensure_dirs(paths: List[str]):
    for p in paths:
        os.makedirs(p, exist_ok=True)
