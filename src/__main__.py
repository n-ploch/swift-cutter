"""
Entry point for running the package as a module with `python -m src`.
"""

from .autoedit_cli import main
import sys

if __name__ == '__main__':
    sys.exit(main())
