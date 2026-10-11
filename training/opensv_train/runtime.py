import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def emit(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def atomic_save(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
