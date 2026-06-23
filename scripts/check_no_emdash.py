#!/usr/bin/env python3
"""Pre-commit hook: fail the commit if an em dash appears in the given files.

Why this exists: this project keeps prose em-dash-free, the README most of all.
Checking for the character mechanically means a stray em dash never lands in the
history, instead of relying on anyone remembering the rule on every commit.

pre-commit passes the staged, matching file paths as arguments. Which files get
checked is controlled by the `files` / `types_or` keys in .pre-commit-config.yaml,
not here. This script simply scans whatever it is handed and reports precise
file:line:column locations so the offender is easy to find and fix.

Flagged by default:
  U+2014  EM DASH         (the long dash)
  U+2015  HORIZONTAL BAR  (a look-alike sometimes pasted in its place)

En dashes (U+2013) are NOT flagged by default, so numeric ranges like 10-20 written
with an en dash are left alone. Set EMDASH_CHECK_INCLUDE_EN=1 to flag them too.
"""

import os
import sys

FLAGGED = {
    "\u2014": "EM DASH",
    "\u2015": "HORIZONTAL BAR",
}
if os.environ.get("EMDASH_CHECK_INCLUDE_EN") == "1":
    FLAGGED["\u2013"] = "EN DASH"


def scan(path):
    """Yield (lineno, col, name) for every flagged character in a file."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except (OSError, UnicodeDecodeError):
        # Unreadable or not UTF-8 text (e.g. a binary asset): not our concern.
        return
    for lineno, line in enumerate(lines, start=1):
        for col, char in enumerate(line, start=1):
            if char in FLAGGED:
                yield lineno, col, FLAGGED[char], line.rstrip("\n")


def main(argv):
    found = False
    for path in argv:
        for lineno, col, name, text in scan(path):
            found = True
            print(f"{path}:{lineno}:{col}: found {name}")
            print(f"    {text}")
    if found:
        print()
        print("Em dash found. Replace it with a comma, a colon, parentheses, or two")
        print("separate sentences. Set EMDASH_CHECK_INCLUDE_EN=1 to also flag en dashes.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
