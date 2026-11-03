"""Tests for the `media.frame` method."""

import base64

import cv2
import numpy as np
import pytest

from sleap_connect.protocol_v1.errors import FS_NOT_FOUND, MEDIA_UNREADABLE, ProtocolError
from sleap_connect.protocol_v1.media import MediaMethods, read_frame_jpeg
from sleap_connect.protocol_v1.server import ProtocolV1Server

N_FRAMES = 12


def _brightness(i: int) -> int:
    return 10 + i * 20


@pytest.fixture(params=[("mp4v", ".mp4"), ("MJPG", ".avi")], ids=["mp4", "avi"])
def video(request, tmp_path):
    """A small video whose frame i is a flat gray of `_brightness(i)`."""
    fourcc, suffix = request.param
    path = tmp_path / f"clip{suffix}"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 10, (64, 48))
    assert writer.isOpened()
    for i in range(N_FRAMES):
        writer.write(np.full((48, 64, 3), _brightness(i), dtype=np.uint8))
    writer.release()
    return path


def _decode(result: dict) -> np.ndarray:
    data = np.frombuffer(base64.b64decode(result["content_base64"]), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def test_returns_the_requested_frame_as_a_jpeg(video):
    result = read_frame_jpeg(str(video), 7)

    assert result["mime"] == "image/jpeg"
    assert (result["width"], result["height"]) == (64, 48)
    assert (result["scaled_width"], result["scaled_height"]) == (64, 48)
    image = _decode(result)
    assert image.shape == (48, 64, 3)
    # Frame 7, not a neighbor (they differ by 20 gray levels).
    assert abs(float(image.mean()) - _brightness(7)) < 6


def test_frame_zero(video):
    image = _decode(read_frame_jpeg(str(video), 0))
    assert abs(float(image.mean()) - _brightness(0)) < 6


def test_downscales_to_max_side_but_reports_the_original_size(tmp_path):
    path = tmp_path / "wide.png"
    cv2.imwrite(str(path), np.zeros((300, 3000, 3), dtype=np.uint8))

    result = read_frame_jpeg(str(path), 0, max_side=1500)

    assert (result["width"], result["height"]) == (3000, 300)
    assert (result["scaled_width"], result["scaled_height"]) == (1500, 150)
    assert _decode(result).shape == (150, 1500, 3)


def test_reads_a_single_image_ignoring_frame_idx(tmp_path):
    path = tmp_path / "frame_0001.png"
    cv2.imwrite(str(path), np.full((20, 30, 3), 200, dtype=np.uint8))

    result = read_frame_jpeg(str(path), 99)

    assert (result["width"], result["height"]) == (30, 20)
    assert abs(float(_decode(result).mean()) - 200) < 3


def test_missing_file_is_not_found(tmp_path):
    with pytest.raises(ProtocolError) as exc:
        read_frame_jpeg(str(tmp_path / "nope.mp4"), 0)
    assert exc.value.code == FS_NOT_FOUND


def test_frame_past_the_end_is_unreadable(video):
    with pytest.raises(ProtocolError) as exc:
        read_frame_jpeg(str(video), N_FRAMES + 5)
    assert exc.value.code == MEDIA_UNREADABLE


def test_a_file_that_isnt_a_video_is_unreadable(tmp_path):
    path = tmp_path / "notes.mp4"
    path.write_text("not a video")
    with pytest.raises(ProtocolError) as exc:
        read_frame_jpeg(str(path), 0)
    assert exc.value.code == MEDIA_UNREADABLE


class TestMethod:
    def test_is_registered_on_the_server(self):
        server = ProtocolV1Server(node_id="n")
        MediaMethods(server)
        assert "media.frame" in server._methods

    async def test_handles_a_request(self, video):
        methods = MediaMethods(ProtocolV1Server(node_id="n"))
        result = await methods.frame({"path": str(video), "frame_idx": 3}, conn=None)
        assert abs(float(_decode(result).mean()) - _brightness(3)) < 6

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"path": ""},
            {"path": "/x.mp4", "frame_idx": -1},
            {"path": "/x.mp4", "frame_idx": "3"},
            {"path": "/x.mp4", "frame_idx": True},
            {"path": "/x.mp4", "max_side": 0},
        ],
    )
    async def test_rejects_bad_params(self, params):
        methods = MediaMethods(ProtocolV1Server(node_id="n"))
        with pytest.raises(ProtocolError) as exc:
            await methods.frame(params, conn=None)
        assert exc.value.code == MEDIA_UNREADABLE
