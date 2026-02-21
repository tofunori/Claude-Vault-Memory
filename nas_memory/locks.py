from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_lock(lock_path: Path, timeout_seconds: int = 30):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+")
    start = time.time()

    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start >= timeout_seconds:
                    raise TimeoutError(f"Lock timeout: {lock_path}")
                time.sleep(0.1)

        fh.seek(0)
        fh.truncate()
        fh.write(f"locked_at={int(time.time())}\n")
        fh.flush()
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
