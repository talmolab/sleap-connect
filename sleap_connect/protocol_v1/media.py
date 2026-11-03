"""The `media.frame` method: one video frame as a small JPEG.

The app's receptive-field preview (training config dialog) draws one
labeled frame with the model's receptive field over it. When the labels
file lives on the worker, so does its video, and the app can't decode it
itself. It sends `{path, frame_idx}` and gets back that one frame, decoded
here and re-encoded as a JPEG with its longest side capped (a few hundred
KB), plus the frame's original size so the client can scale points and the
receptive-field box drawn on it.

Re-encoding instead of sending the video's own bytes: an MP4 frame usually
can't be decoded on its own (it needs everything back to the previous
keyframe, often megabytes), and not every client WebView can decode every
codec, while every client can show a JPEG.

`path` may also be a single image (one frame of an image-sequence video);
`frame_idx` is ignored then. Like the video paths inside a job's labels
file, it isn't confined to the shared folders.
"""

import asyncio
import base64
from pathlib import Path
from typing import Optional

from sleap_connect.protocol_v1.errors import (
    FS_NOT_FOUND,
    MEDIA_UNREADABLE,
    ProtocolError,
)
from sleap_connect.protocol_v1.server import Connection, ProtocolV1Server

DEFAULT_MAX_SIDE = 2048
# Upper bound a client may ask for; a full-size 4K frame is still well under
# the envelope size limit as a JPEG.
MAX_MAX_SIDE = 4096
JPEG_QUALITY = 85

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_frame_jpeg(
    path: str, frame_idx: int, max_side: int = DEFAULT_MAX_SIDE
) -> dict:
    """Decode one frame of `path` and return it as a JPEG.

    Args:
        path: A video file, or a single image file.
        frame_idx: Zero-based frame index (ignored for an image file).
        max_side: Longest side of the returned image, in pixels; smaller
            frames are returned at their own size.

    Returns:
        `{content_base64, mime, width, height, scaled_width, scaled_height}`:
        `width`/`height` are the frame's original size, `scaled_*` the
        JPEG's.

    Raises:
        ProtocolError: `FS_NOT_FOUND` if `path` doesn't exist,
            `MEDIA_UNREADABLE` if it can't be decoded or has no such frame.
    """
    import cv2

    target = Path(path)
    if not target.is_file():
        raise ProtocolError(FS_NOT_FOUND, f"No such file: {path}")

    if target.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(target), cv2.IMREAD_COLOR)
        if frame is None:
            raise ProtocolError(MEDIA_UNREADABLE, f"Couldn't read image: {path}")
    else:
        frame = _read_video_frame(cv2, target, frame_idx)

    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ProtocolError(MEDIA_UNREADABLE, f"Couldn't encode frame of {path}")

    scaled_height, scaled_width = frame.shape[:2]
    return {
        "content_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "mime": "image/jpeg",
        "width": width,
        "height": height,
        "scaled_width": scaled_width,
        "scaled_height": scaled_height,
    }


def _read_video_frame(cv2, path: Path, frame_idx: int):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ProtocolError(MEDIA_UNREADABLE, f"Couldn't open video: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0 and frame_idx >= count:
            raise ProtocolError(
                MEDIA_UNREADABLE,
                f"Frame {frame_idx} is past the end of {path} ({count} frames)",
            )
        if frame_idx > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ProtocolError(
            MEDIA_UNREADABLE, f"Couldn't decode frame {frame_idx} of {path}"
        )
    return frame


class MediaMethods:
    """Registers `media.frame` on a server."""

    def __init__(self, server: ProtocolV1Server):
        """Register the method.

        Args:
            server: The `ProtocolV1Server` to register on.
        """
        server.register("media.frame", self.frame)

    async def frame(self, params: dict, conn: Connection) -> dict:
        """Handle `media.frame` — `{path, frame_idx?, max_side?}`."""
        path = params.get("path")
        if not isinstance(path, str) or not path:
            raise ProtocolError(MEDIA_UNREADABLE, "media.frame needs a 'path'")
        frame_idx = _int_param(params, "frame_idx", 0)
        max_side = min(_int_param(params, "max_side", DEFAULT_MAX_SIDE), MAX_MAX_SIDE)
        if frame_idx < 0 or max_side < 1:
            raise ProtocolError(
                MEDIA_UNREADABLE, "frame_idx must be >= 0 and max_side >= 1"
            )
        # Decoding blocks (and seeking a long video can take a moment).
        return await asyncio.to_thread(read_frame_jpeg, path, frame_idx, max_side)


def _int_param(params: dict, name: str, default: int) -> int:
    value: Optional[object] = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(MEDIA_UNREADABLE, f"'{name}' must be an integer")
    return value
