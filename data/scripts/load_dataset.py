"""Dataset loader script for scanning and validating vision datasets.

This script recursively scans a specified dataset directory for supported image formats
(JPG, JPEG, PNG, BMP) and provides summary statistics and file previews.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Set

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_images(data_dir: Path) -> List[Path]:
    """Recursively search for supported image files within a directory.

    Args:
        data_dir (Path): Absolute or relative path to the dataset directory.

    Returns:
        List[Path]: Sorted list of Path objects for all matching image files.

    Raises:
        ValueError: If the path is not a directory or does not exist.
    """
    if not data_dir.exists():
        raise ValueError(f"Specified directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise ValueError(f"Specified path is not a directory: {data_dir}")

    image_paths: List[Path] = []
    for file_path in data_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(file_path)

    return sorted(image_paths)


def print_dataset_summary(data_dir: Path, image_paths: List[Path], preview_limit: int = 5) -> None:
    """Print a clean summary of discovered dataset images and a preview of file paths.

    Args:
        data_dir (Path): The dataset directory path.
        image_paths (List[Path]): List of discovered image file paths.
        preview_limit (int): Maximum number of image file paths to preview.
    """
    print("=" * 60)
    print("Dataset Loading Summary")
    print("=" * 60)
    print(f"Target Directory : {data_dir.resolve()}")
    print(f"Total Images Found: {len(image_paths)}")
    print(f"Supported Formats : {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    print("-" * 60)

    if image_paths:
        print(f"File Path Preview (showing up to {preview_limit}):")
        for idx, path in enumerate(image_paths[:preview_limit], start=1):
            rel_path = path.relative_to(data_dir) if path.is_relative_to(data_dir) else path
            print(f"  [{idx}] {rel_path}")
        if len(image_paths) > preview_limit:
            print(f"  ... and {len(image_paths) - preview_limit} more file(s).")
    else:
        print("No matching image files found in the specified directory.")
    print("=" * 60)


def main() -> None:
    """CLI entry point for dataset loading and validation."""
    parser = argparse.ArgumentParser(
        description="Recursively scan and validate dataset image files in a specified directory."
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="data/sample",
        help="Path to the dataset directory to scan (default: data/sample)",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=5,
        help="Maximum number of discovered image paths to display in preview (default: 5)",
    )

    args = parser.parse_args()
    target_path = Path(args.data_dir)

    try:
        images = discover_images(target_path)
        print_dataset_summary(target_path, images, preview_limit=args.preview_limit)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"Unexpected error while scanning dataset: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
