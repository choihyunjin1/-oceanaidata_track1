"""Isolated process entry: deny network and optional original repository access."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def denied_path(path, repository, runtime):
    path = Path(path).resolve()
    return repository is not None and (path == repository or repository in path.parents) and not (
        path == runtime or runtime in path.parents
    )


def main():
    repository = Path(os.environ["P2_DENY_REPO"]).resolve() if os.environ.get("P2_DENY_REPO") else None
    runtime = Path(sys.prefix).resolve()
    package = Path(__file__).resolve().parents[1]
    if repository is not None and (package == repository or repository in package.parents):
        raise ValueError("repository-denial verification requires an outside-repository package")

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network disabled for portable execution")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            if denied_path(args[0], repository, runtime):
                raise PermissionError("original repository access denied; installed interpreter runtime excepted")

    sys.addaudithook(guard)
    target = Path(__file__).with_name("run.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
