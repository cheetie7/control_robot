"""Repair pyrobotiqgripper 3.2.8's missing Optional import in this SDK environment."""

from importlib.util import find_spec
from pathlib import Path


def main() -> None:
    """Locate the installed package without importing its broken gripper module, then patch it."""

    spec = find_spec("pyrobotiqgripper")
    path = Path(spec.origin).parent / "gripper.py"
    source = path.read_text()
    if "from typing import Optional" not in source:
        path.write_text("from typing import Optional\n" + source)
    print(f"Optional import ready: {path}")


if __name__ == "__main__":
    main()
