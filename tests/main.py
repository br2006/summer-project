"""Run project smoke tests: python tests/main.py"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
