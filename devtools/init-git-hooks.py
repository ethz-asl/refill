#!/usr/bin/env python
# Disable pylint filename and missing module member complaints.
# pylint: disable=C0103,E1101
""" Initializes git hooks for refill project. """

import shutil
import subprocess


def main():
    """ Copies pre-commit script to git hooks folder. """
    _, repo_root = subprocess.getstatusoutput("git rev-parse --show-toplevel")
    shutil.copy(repo_root + "/devtools/pre-commit", repo_root + "/.git/hooks/")


if __name__ == "__main__":
    main()
