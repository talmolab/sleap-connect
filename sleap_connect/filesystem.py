"""Filesystem utilities for shared storage operations.

This module provides utilities for working with shared filesystems (NFS, local mounts)
using Python standard library. Future enhancement: Can integrate fsspec for cloud storage
(S3, GCS, Azure) support.
"""

import os
import shutil
from pathlib import Path
from typing import Optional


class PathValidationError(Exception):
    """Raised when path validation fails (security or accessibility)."""

    pass


class SharedStorageError(Exception):
    """Raised when shared storage operations fail."""

    pass


def validate_path_in_root(
    path: Path, root: Path, resolve_symlinks: bool = True
) -> Path:
    """Validate that a path is within the shared storage root directory.

    This function prevents path traversal attacks by ensuring the resolved path
    stays within the allowed root directory.

    Args:
        path: The path to validate (can be relative or absolute).
        root: The root directory that path must be within.
        resolve_symlinks: If True, resolve symlinks before validation (recommended
            for security). Default: True.

    Returns:
        The resolved absolute path if validation succeeds.

    Raises:
        PathValidationError: If path is outside root or validation fails.

    Examples:
        >>> root = Path("/shared/storage")
        >>> path = Path("/shared/storage/jobs/job_123/data.zip")
        >>> validate_path_in_root(path, root)
        PosixPath('/shared/storage/jobs/job_123/data.zip')

        >>> path = Path("/shared/storage/../../etc/passwd")
        >>> validate_path_in_root(path, root)
        PathValidationError: Path outside shared storage root
    """
    try:
        if resolve_symlinks:
            resolved_path = path.resolve()
            resolved_root = root.resolve()
        else:
            resolved_path = path.absolute()
            resolved_root = root.absolute()

        # Check if path is within root using relative_to
        # This will raise ValueError if path is not relative to root
        resolved_path.relative_to(resolved_root)

        return resolved_path

    except ValueError as e:
        raise PathValidationError(
            f"Path '{path}' is outside shared storage root '{root}'. "
            f"Resolved to: '{resolved_path if 'resolved_path' in locals() else 'unknown'}'"
        ) from e


def to_relative_path(path: Path, root: Path) -> Path:
    """Convert an absolute path to a relative path from root.

    Args:
        path: The absolute path to convert.
        root: The root directory to compute relative path from.

    Returns:
        The relative path from root.

    Raises:
        ValueError: If path is not within root.

    Examples:
        >>> root = Path("/shared/storage")
        >>> path = Path("/shared/storage/jobs/job_123/data.zip")
        >>> to_relative_path(path, root)
        PosixPath('jobs/job_123/data.zip')
    """
    return path.relative_to(root)


def to_absolute_path(relative_path: Path, root: Path) -> Path:
    """Convert a relative path to an absolute path using root.

    Args:
        relative_path: The relative path to convert.
        root: The root directory to resolve from.

    Returns:
        The absolute path.

    Examples:
        >>> root = Path("/shared/storage")
        >>> rel_path = Path("jobs/job_123/data.zip")
        >>> to_absolute_path(rel_path, root)
        PosixPath('/shared/storage/jobs/job_123/data.zip')
    """
    return root / relative_path


def safe_exists(path: Path) -> bool:
    """Check if a path exists safely (handles permission errors).

    Args:
        path: The path to check.

    Returns:
        True if path exists and is accessible, False otherwise.

    Examples:
        >>> safe_exists(Path("/shared/storage/file.zip"))
        True
        >>> safe_exists(Path("/nonexistent/file.zip"))
        False
    """
    try:
        return path.exists()
    except (PermissionError, OSError):
        return False


def safe_copy(src: Path, dst: Path, preserve_metadata: bool = True) -> None:
    """Copy a file safely with error handling.

    Args:
        src: Source file path.
        dst: Destination file path.
        preserve_metadata: If True, preserve file metadata (timestamps, permissions).
            Default: True.

    Raises:
        SharedStorageError: If copy operation fails.

    Examples:
        >>> src = Path("/local/training.zip")
        >>> dst = Path("/shared/storage/jobs/job_123/training.zip")
        >>> safe_copy(src, dst)
    """
    try:
        if preserve_metadata:
            shutil.copy2(src, dst)
        else:
            shutil.copy(src, dst)
    except PermissionError as e:
        raise SharedStorageError(
            f"Permission denied copying '{src}' to '{dst}'. " f"Check file permissions."
        ) from e
    except OSError as e:
        # Includes disk full, I/O errors, etc.
        if "No space left on device" in str(e):
            raise SharedStorageError(
                f"Insufficient disk space to copy '{src}' to '{dst}'."
            ) from e
        else:
            raise SharedStorageError(f"Failed to copy '{src}' to '{dst}': {e}") from e


def safe_mkdir(path: Path, parents: bool = True, exist_ok: bool = True) -> None:
    """Create a directory safely with error handling.

    Args:
        path: Directory path to create.
        parents: If True, create parent directories as needed. Default: True.
        exist_ok: If True, don't raise error if directory exists. Default: True.

    Raises:
        SharedStorageError: If directory creation fails.

    Examples:
        >>> safe_mkdir(Path("/shared/storage/jobs/job_123/output"))
    """
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except PermissionError as e:
        raise SharedStorageError(
            f"Permission denied creating directory '{path}'. "
            f"Check parent directory permissions."
        ) from e
    except OSError as e:
        if "No space left on device" in str(e):
            raise SharedStorageError(
                f"Insufficient disk space to create directory '{path}'."
            ) from e
        else:
            raise SharedStorageError(f"Failed to create directory '{path}': {e}") from e


def safe_remove(path: Path, recursive: bool = False) -> None:
    """Remove a file or directory safely with error handling.

    Args:
        path: Path to remove.
        recursive: If True, remove directories recursively. Default: False.

    Raises:
        SharedStorageError: If removal fails.

    Examples:
        >>> safe_remove(Path("/shared/storage/temp/file.zip"))
        >>> safe_remove(Path("/shared/storage/temp/"), recursive=True)
    """
    try:
        if recursive and path.is_dir():
            shutil.rmtree(path)
        else:
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
    except PermissionError as e:
        raise SharedStorageError(
            f"Permission denied removing '{path}'. Check permissions."
        ) from e
    except OSError as e:
        raise SharedStorageError(f"Failed to remove '{path}': {e}") from e


def get_file_info(path: Path) -> dict:
    """Get file metadata (size, modification time, etc.).

    Args:
        path: Path to file or directory.

    Returns:
        Dictionary containing file metadata:
            - size: File size in bytes
            - mtime: Modification time (timestamp)
            - is_dir: True if directory, False if file
            - is_file: True if regular file
            - exists: True if path exists

    Examples:
        >>> info = get_file_info(Path("/shared/storage/data.zip"))
        >>> print(f"Size: {info['size'] / 1e9:.2f} GB")
        Size: 5.23 GB
    """
    try:
        if not path.exists():
            return {"exists": False}

        stat = path.stat()
        return {
            "exists": True,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
        }
    except (PermissionError, OSError):
        return {"exists": False, "error": "Permission denied or inaccessible"}


def list_directory(path: Path, pattern: Optional[str] = None) -> list[Path]:
    """List contents of a directory.

    Args:
        path: Directory path to list.
        pattern: Optional glob pattern to filter results (e.g., "*.zip").

    Returns:
        List of Path objects for directory contents.

    Raises:
        SharedStorageError: If directory cannot be listed.

    Examples:
        >>> files = list_directory(Path("/shared/storage/jobs/job_123"))
        >>> zip_files = list_directory(Path("/shared/storage"), pattern="*.zip")
    """
    try:
        if pattern:
            return list(path.glob(pattern))
        else:
            return list(path.iterdir())
    except PermissionError as e:
        raise SharedStorageError(
            f"Permission denied listing directory '{path}'."
        ) from e
    except OSError as e:
        raise SharedStorageError(f"Failed to list directory '{path}': {e}") from e


def check_disk_space(path: Path, required_bytes: int) -> bool:
    """Check if sufficient disk space is available at path.

    Args:
        path: Path to check disk space for.
        required_bytes: Number of bytes required.

    Returns:
        True if sufficient space available, False otherwise.

    Examples:
        >>> # Check if 5GB available
        >>> if check_disk_space(Path("/shared/storage"), 5 * 1024**3):
        ...     print("Enough space for 5GB file")
    """
    try:
        stat = shutil.disk_usage(path)
        return stat.free >= required_bytes
    except OSError:
        # If we can't check, assume insufficient space (fail-safe)
        return False


def get_disk_usage(path: Path) -> dict:
    """Get disk usage statistics for the filesystem containing path.

    Args:
        path: Path to check disk usage for.

    Returns:
        Dictionary with keys:
            - total: Total disk space in bytes
            - used: Used disk space in bytes
            - free: Free disk space in bytes
            - percent_used: Percentage of disk used

    Examples:
        >>> usage = get_disk_usage(Path("/shared/storage"))
        >>> print(f"Free: {usage['free'] / 1e9:.1f} GB ({100 - usage['percent_used']:.1f}% free)")
        Free: 423.5 GB (42.3% free)
    """
    try:
        stat = shutil.disk_usage(path)
        return {
            "total": stat.total,
            "used": stat.used,
            "free": stat.free,
            "percent_used": (stat.used / stat.total * 100) if stat.total > 0 else 0,
        }
    except OSError as e:
        raise SharedStorageError(f"Failed to get disk usage for '{path}': {e}") from e
