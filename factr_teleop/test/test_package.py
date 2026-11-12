"""Smoke tests for the retained upstream FACTR ROS package."""

from pathlib import Path


def test_factr_package_manifest_exists() -> None:
    """Verify the package manifest is present; the test has no array output."""

    package_root = Path(__file__).parents[1]
    assert (package_root / "package.xml").is_file()
