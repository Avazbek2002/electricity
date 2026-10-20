#!/usr/bin/env python3
"""Score any model in models/ the same way. See harness/cli.py and README.md.

  python evaluate.py --model nlinear_x --mode validate
"""
import harness  # noqa: F401  (puts core/ on the import path)
from harness.cli import main

if __name__ == "__main__":
    main()
