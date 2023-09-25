#!/usr/bin/env python3
# system modules
import sys
import shlex
import os
import itertools
import re
import subprocess


def get_version():
    try:
        git_version = subprocess.check_output(
            shlex.split("git describe --always --tags --match 'v*' --dirty"),
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        d = re.fullmatch(
            pattern=r"[a-z]*(?P<tagversion>\d+(:?\.\d+)*)"
            r"(?:[^.\d]+(?P<revcount>\d+)[^.\da-z]+?(?P<commit>[a-z0-9]+))?"
            r"(?:[^.\d]+?(?P<dirty>dirty))?",
            string=git_version.decode(errors="ignore").strip(),
            flags=re.IGNORECASE,
        ).groupdict()
        return "+".join(
            filter(
                bool,
                itertools.chain(
                    [d.get("tagversion", "0")],
                    [".".join([d[k] for k in ("revcount", "commit") if d[k]])],
                ),
            )
        )
    except (
        subprocess.CalledProcessError,
        OSError,
        ModuleNotFoundError,
        AttributeError,
        TypeError,
        StopIteration,
    ) as e:
        print(e, file=sys.stderr)
        return None


print(get_version() or "0.0.0")
