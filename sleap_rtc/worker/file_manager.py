"""File transfer and storage management for worker nodes.

This module handles file transfer via RTC data channels, file compression,
and filesystem browsing for SLEAP-RTC workers.
"""

import asyncio
import base64
import fnmatch
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from aiortc import RTCDataChannel

from sleap_rtc.protocol import (
    MSG_FILE_UPLOAD_CACHE_HIT,
    MSG_FILE_UPLOAD_COMPLETE,
    MSG_FILE_UPLOAD_ERROR,
    MSG_FILE_UPLOAD_PROGRESS,
    MSG_FILE_UPLOAD_READY,
    MSG_SEPARATOR,
)

# Import sleap_io for SLP file operations (lazy import to avoid startup cost)
try:
    import sleap_io as sio
    from sleap_io.io.video_reading import HDF5Video

    SLEAP_IO_AVAILABLE = True
except ImportError:
    SLEAP_IO_AVAILABLE = False
    sio = None
    HDF5Video = None


class FileManager:
    """Manages file transfer, compression, and filesystem browsing for worker nodes.

    This class handles file transfer via RTC data channels, compression of
    training results, and browsing of configured mount points.

    Attributes:
        chunk_size: Size of chunks for file transfer (default 32KB).
        zipped_file: Path to most recently created zip file.
        save_dir: Local directory for saving files.
        output_dir: Output directory for job results.
        mounts: List of MountConfig objects for filesystem browsing.
        working_dir: Worker's current working directory.
    """

    # Constants for filesystem operations
    MAX_RESULTS = 20
    SEARCH_TIMEOUT = 10.0  # seconds
    MAX_SEARCH_DEPTH = 5
    MIN_PATTERN_CHARS = 3
    # fs.read is for small direct reads (inspecting a config/log/text file) —
    # bulk binary transfer goes through the blob API's content-addressed,
    # chunk-verified path instead. 4 MiB keeps a single call's JSON payload
    # reasonable; callers needing more page through with offset/length.
    MAX_READ_BYTES = 4 * 1024 * 1024

    def __init__(
        self,
        chunk_size: int = 32 * 1024,
        mounts: list = None,
        working_dir: str = None,
    ):
        """Initialize file manager.

        Args:
            chunk_size: Size of chunks for file transfer (default 32KB).
            mounts: List of MountConfig objects for filesystem browsing.
            working_dir: Worker's current working directory.
        """
        self.chunk_size = chunk_size
        self.save_dir = "."
        self.zipped_file = ""
        self.output_dir = ""
        self.mounts: list = mounts or []
        self.working_dir: str = working_dir

        # Client-to-worker upload state
        # Maps sha256 hex → absolute path for files received this session.
        self._upload_cache: Dict[str, str] = {}
        # Holds state for an in-progress upload (one at a time).
        self._upload_session: Optional[dict] = None

    async def send_file(
        self, channel: RTCDataChannel, file_path: str, output_dir: str = ""
    ):
        """Send a file to client via RTC data channel.

        Args:
            channel: RTC data channel for sending file.
            file_path: Path to file to send.
            output_dir: Output directory hint for client (where to save file).
        """
        if channel.readyState != "open":
            logging.error(f"Data channel not open: {channel.readyState}")
            return

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        # Send file metadata
        output_hint = output_dir or self.output_dir
        channel.send(f"FILE_META::{file_name}:{file_size}:{output_hint}")
        logging.info(f"Sending file: {file_name} ({file_size} bytes)")

        # Send file in chunks
        with open(file_path, "rb") as file:
            bytes_sent = 0
            while chunk := file.read(self.chunk_size):
                # Flow control: wait if buffer is too full
                while (
                    channel.bufferedAmount is not None
                    and channel.bufferedAmount > 16 * 1024 * 1024
                ):
                    await asyncio.sleep(0.1)

                channel.send(chunk)
                bytes_sent += len(chunk)

        # Signal end of file
        channel.send("END_OF_FILE")
        logging.info("File sent successfully")

    async def zip_results(self, file_name: str, dir_path: Optional[str] = None):
        """Zip directory contents into archive.

        Args:
            file_name: Name of the zip file to create (without .zip extension).
            dir_path: Path to directory to zip.

        Returns:
            Path to created zip file, or None if failed.
        """
        logging.info("Zipping results...")
        if not dir_path or not Path(dir_path).exists():
            logging.error(f"Directory does not exist: {dir_path}")
            return None

        try:
            shutil.make_archive(file_name, "zip", dir_path)
            self.zipped_file = f"{file_name}.zip"
            logging.info(f"Results zipped to {self.zipped_file}")
            return self.zipped_file
        except Exception as e:
            logging.error(f"Error zipping results: {e}")
            return None

    async def unzip_results(self, file_path: str):
        """Unzip archive to save directory.

        Args:
            file_path: Path to zip file to extract.

        Returns:
            Path to extracted directory, or None if failed.
        """
        logging.info("Unzipping results...")
        if not Path(file_path).exists():
            logging.error(f"File does not exist: {file_path}")
            return None

        try:
            shutil.unpack_archive(file_path, self.save_dir)
            logging.info(f"Results unzipped from {file_path} to {self.save_dir}")

            # Calculate unzipped directory path
            original_name = Path(file_path).stem  # Remove .zip extension
            unzipped_dir = f"{self.save_dir}/{original_name}"
            logging.info(f"Unzipped contents to {unzipped_dir}")
            return unzipped_dir
        except Exception as e:
            logging.error(f"Error unzipping results: {e}")
            return None

    # =========================================================================
    # Client-to-Worker Upload Methods
    # =========================================================================

    def check_upload_cache(self, sha256: str, filename: str) -> Optional[str]:
        """Check whether a previously-uploaded file is still on disk.

        Args:
            sha256: SHA-256 hex digest of the file to look up.
            filename: Filename (unused; reserved for future logging/hints).

        Returns:
            Absolute path string if a matching cached file exists on disk,
            or None if no cache entry exists or the cached file was deleted.
        """
        cached = self._upload_cache.get(sha256)
        if cached and Path(cached).exists():
            return cached
        # Remove stale entry so subsequent uploads repopulate it.
        if sha256 in self._upload_cache:
            del self._upload_cache[sha256]
        return None

    async def start_upload_session(
        self,
        channel: RTCDataChannel,
        filename: str,
        total_bytes: int,
        dest_dir: str,
        create_subdir: str,
    ) -> None:
        """Validate destination, open write handle, and signal readiness.

        Sends FILE_UPLOAD_READY on success or FILE_UPLOAD_ERROR on failure.

        Args:
            channel: RTC data channel to send responses on.
            filename: Base filename of the incoming file.
            total_bytes: Expected total file size in bytes.
            dest_dir: Absolute destination directory path on the worker.
            create_subdir: "1" to create a sleap_rtc_downloads/ subfolder,
                "0" to write directly into dest_dir.
        """
        dest_path = Path(dest_dir)

        # Security: destination must resolve within a configured mount.
        if not self._is_path_allowed(dest_path):
            channel.send(
                f"{MSG_FILE_UPLOAD_ERROR}{MSG_SEPARATOR}"
                "Destination outside configured mounts"
            )
            return

        if create_subdir == "1":
            dest_path = dest_path / "sleap_rtc_downloads"
            try:
                dest_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                channel.send(
                    f"{MSG_FILE_UPLOAD_ERROR}{MSG_SEPARATOR}"
                    f"Cannot create sleap_rtc_downloads/: {e}"
                )
                return

        file_path = dest_path / filename

        try:
            file_handle = open(file_path, "wb")  # noqa: WPS515
        except OSError as e:
            channel.send(
                f"{MSG_FILE_UPLOAD_ERROR}{MSG_SEPARATOR}Cannot open file for writing: {e}"
            )
            return

        self._upload_session = {
            "filename": filename,
            "total_bytes": total_bytes,
            "file_path": file_path,
            "file_handle": file_handle,
            "bytes_received": 0,
            "channel": channel,
            "last_progress_time": 0.0,
            "sha256_ctx": hashlib.sha256(),
        }

        channel.send(MSG_FILE_UPLOAD_READY)
        logging.info(
            f"Upload session started: {filename} ({total_bytes} bytes) → {file_path}"
        )

    def receive_upload_chunk(self, chunk: bytes) -> None:
        """Append an incoming binary chunk to the active upload session.

        Sends FILE_UPLOAD_PROGRESS at most every 500 ms.
        Sends FILE_UPLOAD_ERROR and cleans up on I/O failure.

        Args:
            chunk: Raw bytes received from the client.
        """
        if self._upload_session is None:
            logging.warning(
                "Received upload chunk with no active upload session; ignoring."
            )
            return

        session = self._upload_session
        try:
            session["file_handle"].write(chunk)
            session["sha256_ctx"].update(chunk)
            session["bytes_received"] += len(chunk)
        except OSError as e:
            logging.error(f"Upload write error: {e}")
            session["file_handle"].close()
            try:
                session["file_path"].unlink(missing_ok=True)
            except OSError:
                pass
            session["channel"].send(
                f"{MSG_FILE_UPLOAD_ERROR}{MSG_SEPARATOR}Write error: {e}"
            )
            self._upload_session = None
            return

        # Progress at most every 500 ms.
        now = time.time()
        if now - session["last_progress_time"] >= 0.5:
            session["channel"].send(
                f"{MSG_FILE_UPLOAD_PROGRESS}{MSG_SEPARATOR}"
                f"{session['bytes_received']}{MSG_SEPARATOR}{session['total_bytes']}"
            )
            session["last_progress_time"] = now

    async def finish_upload_session(self, channel: RTCDataChannel) -> None:
        """Finalise the active upload: close file, verify size, update cache.

        Sends FILE_UPLOAD_COMPLETE::{absolute_path} on success or
        FILE_UPLOAD_ERROR::{reason} on failure.

        Args:
            channel: RTC data channel to send the result on.
        """
        if self._upload_session is None:
            logging.warning(
                "Received FILE_UPLOAD_END with no active session; ignoring."
            )
            return

        session = self._upload_session
        self._upload_session = None

        try:
            session["file_handle"].close()
        except OSError as e:
            channel.send(
                f"{MSG_FILE_UPLOAD_ERROR}{MSG_SEPARATOR}Failed to close file: {e}"
            )
            return

        # Verify written size matches declared total.
        actual_size = session["file_path"].stat().st_size
        if actual_size != session["total_bytes"]:
            channel.send(
                f"{MSG_FILE_UPLOAD_ERROR}{MSG_SEPARATOR}"
                f"Size mismatch: expected {session['total_bytes']}, got {actual_size}"
            )
            session["file_path"].unlink(missing_ok=True)
            return

        # Cache by content hash so future uploads of the same file are instant.
        sha256 = session["sha256_ctx"].hexdigest()
        self._upload_cache[sha256] = str(session["file_path"])

        channel.send(f"{MSG_FILE_UPLOAD_COMPLETE}{MSG_SEPARATOR}{session['file_path']}")
        logging.info(f"Upload complete: {session['file_path']} (sha256={sha256[:12]}…)")

    # =========================================================================
    # Filesystem Browser Methods
    # =========================================================================

    def set_mounts(self, mounts: list) -> None:
        """Set the list of configured mount points.

        Args:
            mounts: List of MountConfig objects.
        """
        self.mounts = mounts or []

    def get_mounts(self) -> List[dict]:
        """Get list of available mount points.

        Returns:
            List of mount dictionaries with 'path' and 'label' fields.
        """
        return [{"path": mount.path, "label": mount.label} for mount in self.mounts]

    def get_worker_info(self, worker_id: str = None) -> dict:
        """Get worker information for browser status display.

        Args:
            worker_id: Optional worker identifier.

        Returns:
            Dictionary with worker_id, working_dir, and mounts.
        """
        return {
            "worker_id": worker_id or "unknown",
            "working_dir": self.working_dir or os.getcwd(),
            "mounts": self.get_mounts(),
        }

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if a path is within an allowed mount.

        Args:
            path: Path to check.

        Returns:
            True if path is within a configured mount, False otherwise.
        """
        if not self.mounts:
            return False

        # Resolve the path to handle symlinks and .. segments
        try:
            resolved = path.resolve()
        except (OSError, ValueError):
            return False

        for mount in self.mounts:
            mount_path = Path(mount.path).resolve()
            try:
                resolved.relative_to(mount_path)
                return True
            except ValueError:
                continue

        return False

    def _is_wildcard_pattern(self, pattern: str) -> bool:
        """Check if pattern contains wildcard characters.

        Args:
            pattern: The pattern to check.

        Returns:
            True if pattern contains wildcards.
        """
        return any(c in pattern for c in "*?[")

    def _validate_pattern(self, pattern: str) -> tuple:
        """Validate a search pattern.

        Args:
            pattern: The pattern to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not pattern:
            return False, "Pattern cannot be empty"

        # Count non-wildcard characters
        non_wildcard = sum(1 for c in pattern if c not in "*?[]")
        if non_wildcard < self.MIN_PATTERN_CHARS:
            return (
                False,
                f"Pattern must contain at least {self.MIN_PATTERN_CHARS} non-wildcard characters",
            )

        return True, None

    def _match_filename(self, pattern: str, filename: str) -> dict:
        """Match a filename against a pattern.

        Args:
            pattern: Pattern to match (may contain wildcards).
            filename: Filename to check.

        Returns:
            Match result with 'matches' and 'match_type' fields.
        """
        # Case-insensitive matching
        pattern_lower = pattern.lower()
        filename_lower = filename.lower()

        # Exact match
        if filename_lower == pattern_lower:
            return {"matches": True, "match_type": "exact"}

        # Wildcard match using fnmatch
        if self._is_wildcard_pattern(pattern):
            if fnmatch.fnmatch(filename_lower, pattern_lower):
                return {"matches": True, "match_type": "wildcard"}

        # Filename contains pattern (substring match)
        if pattern_lower in filename_lower:
            return {"matches": True, "match_type": "substring"}

        return {"matches": False, "match_type": None}

    def resolve_path(
        self,
        pattern: str,
        file_size: int = None,
        max_depth: int = None,
        mount_label: str = None,
    ) -> dict:
        """Resolve a file pattern to matching files within mounts.

        Args:
            pattern: Filename or pattern to search for.
            file_size: Expected file size for ranking (optional).
            max_depth: Maximum directory depth to search.
            mount_label: Label of specific mount to search, or None/"all" for all mounts.

        Returns:
            Dictionary with candidates, truncated, timeout, and search_time_ms.
        """
        if max_depth is None:
            max_depth = self.MAX_SEARCH_DEPTH

        # Validate pattern if it contains wildcards
        if self._is_wildcard_pattern(pattern):
            is_valid, error = self._validate_pattern(pattern)
            if not is_valid:
                return {
                    "candidates": [],
                    "truncated": False,
                    "timeout": False,
                    "error": error,
                    "error_code": "PATTERN_TOO_BROAD",
                }

        start_time = time.time()
        candidates = []
        timed_out = False

        # Extract just the filename if a full path was provided
        search_name = Path(pattern).name

        # Filter mounts if mount_label specified
        mounts_to_search = self.mounts
        if mount_label and mount_label.lower() != "all":
            mounts_to_search = [m for m in self.mounts if m.label == mount_label]
            if not mounts_to_search:
                return {
                    "candidates": [],
                    "truncated": False,
                    "timeout": False,
                    "error": f"Mount '{mount_label}' not found",
                    "error_code": "MOUNT_NOT_FOUND",
                }

        for mount in mounts_to_search:
            if timed_out:
                break

            mount_path = Path(mount.path)
            if not mount_path.exists():
                continue

            # Scan the mount directory
            for match in self._scan_directory(
                mount_path,
                search_name,
                0,
                max_depth,
                file_size,
                start_time,
            ):
                if match is None:
                    # Timeout signal
                    timed_out = True
                    break

                candidates.append(match)

                # Early termination after MAX_RESULTS
                if len(candidates) >= self.MAX_RESULTS:
                    break

        # Sort candidates by score (descending)
        candidates.sort(key=lambda c: c.get("score", 0), reverse=True)

        # Truncate to MAX_RESULTS
        truncated = len(candidates) > self.MAX_RESULTS
        candidates = candidates[: self.MAX_RESULTS]

        search_time_ms = int((time.time() - start_time) * 1000)

        return {
            "candidates": candidates,
            "truncated": truncated,
            "timeout": timed_out,
            "search_time_ms": search_time_ms,
        }

    def _scan_directory(
        self,
        dir_path: Path,
        pattern: str,
        depth: int,
        max_depth: int,
        file_size: int,
        start_time: float,
    ):
        """Lazily scan a directory for matching files.

        Args:
            dir_path: Directory to scan.
            pattern: Pattern to match.
            depth: Current depth.
            max_depth: Maximum depth.
            file_size: Expected file size for scoring.
            start_time: Search start time for timeout.

        Yields:
            Match dictionaries, or None if timeout.
        """
        # Check timeout
        if time.time() - start_time > self.SEARCH_TIMEOUT:
            yield None
            return

        # Check depth limit
        if depth > max_depth:
            return

        try:
            entries = list(dir_path.iterdir())
        except (PermissionError, OSError):
            # Skip directories we can't read
            return

        for entry in entries:
            # Check timeout periodically
            if time.time() - start_time > self.SEARCH_TIMEOUT:
                yield None
                return

            try:
                if entry.is_file():
                    match_result = self._match_filename(pattern, entry.name)
                    if match_result["matches"]:
                        stat = entry.stat()
                        score = self._calculate_score(
                            match_result["match_type"],
                            stat.st_size,
                            file_size,
                            str(entry),
                        )
                        yield {
                            "path": str(entry),
                            "name": entry.name,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "match_type": match_result["match_type"],
                            "score": score,
                        }

                elif entry.is_dir():
                    # Recursively scan subdirectories
                    for sub_match in self._scan_directory(
                        entry,
                        pattern,
                        depth + 1,
                        max_depth,
                        file_size,
                        start_time,
                    ):
                        yield sub_match

            except (PermissionError, OSError):
                # Skip files/directories we can't access
                continue

    def _calculate_score(
        self,
        match_type: str,
        actual_size: int,
        expected_size: int,
        path: str,
    ) -> float:
        """Calculate a ranking score for a match.

        Args:
            match_type: Type of match (exact, wildcard, substring).
            actual_size: Actual file size.
            expected_size: Expected file size from client.
            path: Full path for path token analysis.

        Returns:
            Score value (higher is better).
        """
        score = 0.0

        # Match type scoring (primary factor)
        if match_type == "exact":
            score += 100.0
        elif match_type == "wildcard":
            score += 50.0
        elif match_type == "substring":
            score += 25.0

        # Size match scoring (secondary factor)
        if expected_size and actual_size:
            if actual_size == expected_size:
                score += 20.0
            else:
                # Partial score for similar sizes (within 10%)
                size_diff = abs(actual_size - expected_size)
                if size_diff < expected_size * 0.1:
                    score += 10.0

        # Shorter paths get slight bonus (tertiary factor)
        path_depth = path.count(os.sep)
        score -= path_depth * 0.1

        return score

    def list_directory(self, path: str, offset: int = 0) -> dict:
        """List contents of a directory within allowed mounts.

        Args:
            path: Directory path to list.
            offset: Number of entries to skip (for pagination).

        Returns:
            Dictionary with path, entries, total_count, and has_more.
        """
        dir_path = Path(path)

        # Canonicalize path to prevent traversal attacks
        try:
            resolved_path = dir_path.resolve()
        except (OSError, ValueError) as e:
            return {
                "path": path,
                "entries": [],
                "total_count": 0,
                "has_more": False,
                "error": f"Invalid path: {e}",
                "error_code": "PATH_NOT_FOUND",
            }

        # Security check: path must be within allowed mounts
        if not self._is_path_allowed(resolved_path):
            return {
                "path": path,
                "entries": [],
                "total_count": 0,
                "has_more": False,
                "error": "Access denied: path is outside configured mounts",
                "error_code": "ACCESS_DENIED",
            }

        if not resolved_path.exists():
            return {
                "path": path,
                "entries": [],
                "total_count": 0,
                "has_more": False,
                "error": "Path does not exist",
                "error_code": "PATH_NOT_FOUND",
            }

        if not resolved_path.is_dir():
            return {
                "path": path,
                "entries": [],
                "total_count": 0,
                "has_more": False,
                "error": "Path is not a directory",
                "error_code": "PATH_NOT_FOUND",
            }

        try:
            all_entries = list(resolved_path.iterdir())
        except PermissionError:
            return {
                "path": path,
                "entries": [],
                "total_count": 0,
                "has_more": False,
                "error": "Permission denied",
                "error_code": "ACCESS_DENIED",
            }

        # Sort: directories first, then alphabetically
        all_entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))

        total_count = len(all_entries)

        # Apply offset and limit
        paginated = all_entries[offset : offset + self.MAX_RESULTS]

        entries = []
        for entry in paginated:
            try:
                stat = entry.stat()
                entries.append(
                    {
                        "name": entry.name,
                        "type": "directory" if entry.is_dir() else "file",
                        "size": stat.st_size if entry.is_file() else 0,
                        "modified": stat.st_mtime,
                    }
                )
            except (PermissionError, OSError):
                # Skip entries we can't stat
                continue

        has_more = offset + len(entries) < total_count

        return {
            "path": str(resolved_path),
            "entries": entries,
            "total_count": total_count,
            "has_more": has_more,
        }

    def stat_path(self, path: str) -> dict:
        """Get metadata for a single file or directory within allowed mounts.

        Mirrors `list_directory`'s security checks and error-shape
        convention (an `error`/`error_code` field on failure, rather than
        raising) exactly, for consistency with the existing `fs.list`.

        Args:
            path: File or directory path to stat.

        Returns:
            Dictionary with path, type ("file"|"directory"), size (0 for
            directories), and modified (mtime), or an error/error_code pair.
        """
        target = Path(path)

        try:
            resolved = target.resolve()
        except (OSError, ValueError) as e:
            return {
                "path": path,
                "error": f"Invalid path: {e}",
                "error_code": "PATH_NOT_FOUND",
            }

        if not self._is_path_allowed(resolved):
            return {
                "path": path,
                "error": "Access denied: path is outside configured mounts",
                "error_code": "ACCESS_DENIED",
            }

        if not resolved.exists():
            return {
                "path": path,
                "error": "Path does not exist",
                "error_code": "PATH_NOT_FOUND",
            }

        try:
            st = resolved.stat()
        except PermissionError:
            return {
                "path": path,
                "error": "Permission denied",
                "error_code": "ACCESS_DENIED",
            }

        return {
            "path": str(resolved),
            "type": "directory" if resolved.is_dir() else "file",
            "size": st.st_size if resolved.is_file() else 0,
            "modified": st.st_mtime,
        }

    def read_file(
        self, path: str, offset: int = 0, length: Optional[int] = None
    ) -> dict:
        """Read a range of bytes from a file within allowed mounts.

        For small direct reads (inspecting a config/log/text file) — NOT
        the bulk-transfer path, which goes through the blob API's content-
        addressed, chunk-verified reads instead. Content is base64-encoded
        since this travels inside a JSON envelope frame.

        Args:
            path: File path to read.
            offset: Byte offset to start reading from.
            length: Number of bytes to read. Defaults to (and is capped at)
                `MAX_READ_BYTES`; a caller reading a larger file pages
                through with repeated offset-advancing calls.

        Returns:
            Dictionary with path, content_base64, offset, size (bytes
            actually read), total_size, and eof (whether this read reached
            the end of the file) — or an error/error_code pair on failure.
        """
        target = Path(path)

        try:
            resolved = target.resolve()
        except (OSError, ValueError) as e:
            return {
                "path": path,
                "error": f"Invalid path: {e}",
                "error_code": "PATH_NOT_FOUND",
            }

        if not self._is_path_allowed(resolved):
            return {
                "path": path,
                "error": "Access denied: path is outside configured mounts",
                "error_code": "ACCESS_DENIED",
            }

        if not resolved.exists():
            return {
                "path": path,
                "error": "Path does not exist",
                "error_code": "PATH_NOT_FOUND",
            }

        if not resolved.is_file():
            return {
                "path": path,
                "error": "Path is not a file",
                "error_code": "PATH_NOT_FOUND",
            }

        if offset < 0:
            return {
                "path": path,
                "error": "offset must be non-negative",
                "error_code": "INVALID_RANGE",
            }

        read_len = (
            self.MAX_READ_BYTES if length is None else min(length, self.MAX_READ_BYTES)
        )

        try:
            total_size = resolved.stat().st_size
            with open(resolved, "rb") as f:
                f.seek(offset)
                data = f.read(read_len)
        except PermissionError:
            return {
                "path": path,
                "error": "Permission denied",
                "error_code": "ACCESS_DENIED",
            }
        except OSError as e:
            return {
                "path": path,
                "error": f"Read failed: {e}",
                "error_code": "IO_ERROR",
            }

        return {
            "path": str(resolved),
            "content_base64": base64.b64encode(data).decode("ascii"),
            "offset": offset,
            "size": len(data),
            "total_size": total_size,
            "eof": offset + len(data) >= total_size,
        }

    # =========================================================================
    # SLP Video Resolution Methods
    # =========================================================================

    def _is_video_embedded(self, video) -> bool:
        """Check if a video has embedded images (frames stored in SLP file).

        Args:
            video: A sleap_io Video object.

        Returns:
            True if the video has embedded images and doesn't need an external file.
        """
        # Priority 1: Live backend attribute (if backend is open)
        if HDF5Video is not None and isinstance(video.backend, HDF5Video):
            return video.backend.has_embedded_images

        # Priority 2: backend_metadata from SLP file (when open_videos=False)
        return video.backend_metadata.get("has_embedded_images", False)

    def check_video_accessibility(self, slp_path: str) -> dict:
        """Check if video paths in an SLP file are accessible on this filesystem.

        Loads the SLP file and checks if each video's filename exists on the
        Worker filesystem. Embedded videos (frames stored in the SLP) are skipped.

        Args:
            slp_path: Path to the SLP file to check.

        Returns:
            Dictionary with:
                - slp_path: The input SLP path
                - total_videos: Total number of video references in the SLP
                - missing: List of dicts with 'filename' and 'original_path' for
                  videos that don't exist on the Worker filesystem
                - accessible: Count of videos that are accessible
                - embedded: Count of videos with embedded frames (skipped)
                - error: Error message if loading failed (optional)
        """
        if not SLEAP_IO_AVAILABLE:
            return {
                "slp_path": slp_path,
                "total_videos": 0,
                "missing": [],
                "accessible": 0,
                "embedded": 0,
                "error": "sleap-io is not available on this Worker",
            }

        # Check SLP file exists
        slp_file = Path(slp_path)
        if not slp_file.exists():
            return {
                "slp_path": slp_path,
                "total_videos": 0,
                "missing": [],
                "accessible": 0,
                "embedded": 0,
                "error": f"SLP file not found: {slp_path}",
            }

        try:
            # Load SLP without opening video backends
            labels = sio.load_file(slp_path, open_videos=False)
        except Exception as e:
            return {
                "slp_path": slp_path,
                "total_videos": 0,
                "missing": [],
                "accessible": 0,
                "embedded": 0,
                "error": f"Failed to load SLP file: {e}",
            }

        missing = []
        accessible = 0
        embedded = 0

        for video in labels.videos:
            # Skip embedded videos (frames stored in SLP file)
            if self._is_video_embedded(video):
                embedded += 1
                continue

            # Get the video filename/path
            video_path = video.filename
            if isinstance(video_path, list):
                # Image sequence - check first image
                video_path = video_path[0] if video_path else ""

            if not video_path:
                continue

            # Check if the video file exists
            if Path(video_path).exists():
                accessible += 1
            else:
                # Extract just the filename for display
                filename = Path(video_path).name
                missing.append(
                    {
                        "filename": filename,
                        "original_path": video_path,
                    }
                )

        return {
            "slp_path": slp_path,
            "total_videos": len(labels.videos),
            "missing": missing,
            "accessible": accessible,
            "embedded": embedded,
        }

    def scan_directory_for_filenames(
        self, directory: str, filenames: list[str]
    ) -> dict:
        """Scan a directory for specific filenames (SLP Viewer style resolution).

        This method checks if the given filenames exist in the specified directory.
        Used for batch resolution of video paths when the user selects a video file
        and the system scans that directory for other missing videos.

        Args:
            directory: Path to the directory to scan.
            filenames: List of filenames to look for (without path).

        Returns:
            Dictionary with:
                - directory: The scanned directory path
                - found: Dict mapping filename → full path (or None if not found)
                - error: Error message if directory is invalid (optional)
                - error_code: Error code if applicable (optional)
        """
        dir_path = Path(directory)

        # Security check: directory must be within allowed mounts
        if not self._is_path_allowed(dir_path):
            return {
                "directory": directory,
                "found": {},
                "error": "Access denied: directory is outside configured mounts",
                "error_code": "ACCESS_DENIED",
            }

        # Check directory exists
        if not dir_path.exists():
            return {
                "directory": directory,
                "found": {},
                "error": f"Directory not found: {directory}",
                "error_code": "PATH_NOT_FOUND",
            }

        if not dir_path.is_dir():
            return {
                "directory": directory,
                "found": {},
                "error": f"Path is not a directory: {directory}",
                "error_code": "PATH_NOT_FOUND",
            }

        # Scan for each filename
        found = {}
        for filename in filenames:
            # Prevent path traversal in filenames
            if os.sep in filename or (os.altsep and os.altsep in filename):
                found[filename] = None
                continue
            if ".." in filename:
                found[filename] = None
                continue

            candidate = dir_path / filename
            if candidate.exists() and candidate.is_file():
                found[filename] = str(candidate)
            else:
                found[filename] = None

        return {
            "directory": str(dir_path),
            "found": found,
        }

    def write_slp_with_new_paths(
        self,
        slp_path: str,
        output_dir: str,
        filename_map: dict,
        output_filename: str = "",
    ) -> dict:
        """Write a new SLP file with updated video paths.

        Loads the SLP file, applies the filename_map to replace video paths,
        and saves a new SLP file with a timestamped name.

        Args:
            slp_path: Path to the original SLP file.
            output_dir: Directory to write the new SLP file to.
            filename_map: Dict mapping original paths to new resolved paths.
                          Example: {"/old/path/video.mp4": "/new/path/video.mp4"}
            output_filename: Optional custom filename. If not provided, generates
                             a default name with format: resolved_YYYYMMDD_<original>.slp

        Returns:
            Dictionary with:
                - output_path: Path to the newly written SLP file
                - videos_updated: Number of video paths that were updated
                - error: Error message if writing failed (optional)
        """
        if not SLEAP_IO_AVAILABLE:
            return {
                "error": "sleap-io is not available on this Worker",
            }

        # Validate SLP file exists
        slp_file = Path(slp_path)
        if not slp_file.exists():
            return {
                "error": f"SLP file not found: {slp_path}",
            }

        # Validate output_dir is within allowed mounts
        output_path_obj = Path(output_dir)
        if not self._is_path_allowed(output_path_obj):
            return {
                "error": "Output directory is outside configured mounts",
                "error_code": "ACCESS_DENIED",
            }

        # Check output_dir exists and is a directory
        if not output_path_obj.exists():
            return {
                "error": f"Output directory not found: {output_dir}",
            }

        if not output_path_obj.is_dir():
            return {
                "error": f"Output path is not a directory: {output_dir}",
            }

        try:
            # Load SLP without opening video backends
            labels = sio.load_file(slp_path, open_videos=False)
        except Exception as e:
            return {
                "error": f"Failed to load SLP file: {e}",
            }

        # Count how many videos will be updated
        videos_updated = 0
        for video in labels.videos:
            video_path = video.filename
            if isinstance(video_path, list):
                # Image sequence - check first image
                video_path = video_path[0] if video_path else ""

            if video_path in filename_map:
                videos_updated += 1

        # Apply filename replacements
        try:
            labels.replace_filenames(filename_map=filename_map, open_videos=False)
        except Exception as e:
            return {
                "error": f"Failed to replace filenames: {e}",
            }

        # Workaround for sleap-io bug: replace_filename sets
        # backend_metadata["filename"] to a list for image sequences, but
        # make_video expects it to be a string (with "filenames" as the list).
        for video in labels.videos:
            if isinstance(video.filename, list) and video.backend_metadata:
                video.backend_metadata["filename"] = video.filename[0]
                video.backend_metadata["filenames"] = video.filename

        # Use custom filename if provided, otherwise generate default
        if output_filename:
            final_filename = output_filename
        else:
            # Generate output filename: resolved_YYYYMMDD_<original>.slp
            from datetime import datetime

            date_str = datetime.now().strftime("%Y%m%d")
            original_name = slp_file.stem
            # Handle .pkg.slp extension
            if original_name.endswith(".pkg"):
                original_name = original_name[:-4]
                final_filename = f"resolved_{date_str}_{original_name}.pkg.slp"
            else:
                final_filename = f"resolved_{date_str}_{original_name}.slp"

        output_full_path = output_path_obj / final_filename

        # Save the updated SLP file
        try:
            labels.save(str(output_full_path))
        except Exception as e:
            return {
                "error": f"Failed to save SLP file: {e}",
            }

        return {
            "output_path": str(output_full_path),
            "videos_updated": videos_updated,
        }

    # =========================================================================
    # Prefix-Based Video Path Resolution Methods
    # =========================================================================

    def find_changed_subpath(self, old_path: str, new_path: str) -> tuple:
        """Find the differing prefix between two paths.

        Compares paths from the END to find the common suffix, then returns
        the differing initial portions (prefixes). This implements SLEAP's
        approach to detecting path prefix changes for auto-resolution.

        Args:
            old_path: Original path from SLP file.
            new_path: User-selected replacement path on Worker.

        Returns:
            Tuple of (old_prefix, new_prefix) where:
                - old_prefix: The initial portion of old_path that differs
                - new_prefix: The corresponding portion in new_path

        Example:
            >>> find_changed_subpath(
            ...     "/Volumes/talmo/project/day1/vid.mp4",
            ...     "/vast/project/day1/vid.mp4"
            ... )
            ("/Volumes/talmo", "/vast")
        """
        old_parts = Path(old_path).parts
        new_parts = Path(new_path).parts

        # Find where paths match from the end (common suffix)
        common_suffix_len = 0
        for i in range(1, min(len(old_parts), len(new_parts)) + 1):
            if old_parts[-i] == new_parts[-i]:
                common_suffix_len = i
            else:
                break

        # Extract prefixes (everything before the common suffix)
        if common_suffix_len > 0 and common_suffix_len < len(old_parts):
            # Normal case: both paths have parts before the common suffix
            old_prefix = str(Path(*old_parts[:-common_suffix_len]))
            new_prefix = str(Path(*new_parts[:-common_suffix_len]))
        elif common_suffix_len >= len(old_parts) and common_suffix_len < len(new_parts):
            # Special case: old path is entirely contained in new path (relative -> absolute)
            # e.g., old="project/video.mp4", new="/root/vast/project/video.mp4"
            # The old prefix is empty, new prefix is the extra leading parts
            old_prefix = ""
            new_prefix = str(Path(*new_parts[:-common_suffix_len]))
        else:
            # No common suffix found, return full paths as prefixes
            old_prefix = old_path
            new_prefix = new_path

        return old_prefix, new_prefix

    def compute_prefix_resolution(
        self,
        original_path: str,
        new_path: str,
        other_missing: list,
    ) -> dict:
        """Compute which missing videos would resolve with a prefix replacement.

        When a user manually locates one video, this method computes the prefix
        change and checks which other missing videos would resolve by applying
        the same prefix transformation.

        Args:
            original_path: The original path of the video the user selected.
            new_path: The new path the user browsed to on the Worker.
            other_missing: List of other missing video paths from the SLP.

        Returns:
            Dictionary with:
                - old_prefix: The detected old prefix to replace
                - new_prefix: The new prefix to use
                - would_resolve: List of dicts with 'original' and 'resolved'
                  paths for videos that would be resolved
                - would_not_resolve: List of paths that wouldn't resolve
                  (either different prefix or file doesn't exist)

        Example:
            >>> compute_prefix_resolution(
            ...     "/Volumes/talmo/project/day1/vid1.mp4",
            ...     "/vast/project/day1/vid1.mp4",
            ...     ["/Volumes/talmo/project/day2/vid2.mp4"]
            ... )
            {
                "old_prefix": "/Volumes/talmo",
                "new_prefix": "/vast",
                "would_resolve": [
                    {"original": "/Volumes/.../vid2.mp4", "resolved": "/vast/.../vid2.mp4"}
                ],
                "would_not_resolve": []
            }
        """
        old_prefix, new_prefix = self.find_changed_subpath(original_path, new_path)

        would_resolve = []
        would_not_resolve = []

        for missing_path in other_missing:
            # Check if this path shares the same old prefix
            if old_prefix == "":
                # Special case: old paths are relative, new paths are absolute
                # Prepend the new prefix to the missing path
                candidate = str(Path(new_prefix) / missing_path)
            elif missing_path.startswith(old_prefix):
                # Normal case: replace the old prefix with new prefix
                candidate = missing_path.replace(old_prefix, new_prefix, 1)
            else:
                # Different prefix - can't resolve with this transformation
                would_not_resolve.append(missing_path)
                continue

            # Check if the transformed path exists on the Worker filesystem
            if Path(candidate).exists():
                would_resolve.append(
                    {
                        "original": missing_path,
                        "resolved": candidate,
                    }
                )
            else:
                would_not_resolve.append(missing_path)

        return {
            "old_prefix": old_prefix,
            "new_prefix": new_prefix,
            "would_resolve": would_resolve,
            "would_not_resolve": would_not_resolve,
        }
