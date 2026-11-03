"""SLP Context Panel widget for video path resolution.

This module provides a panel that shows video status when an SLP file is selected,
and allows fixing missing video paths.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, ListView, ListItem, Label
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding


@dataclass
class VideoInfo:
    """Information about a video in an SLP file."""

    filename: str
    original_path: str
    status: str = "unknown"  # "found", "missing", "embedded", "resolved_pending"
    resolved_path: str = ""

    @property
    def is_missing(self) -> bool:
        return self.status == "missing"

    @property
    def is_found(self) -> bool:
        return self.status == "found"

    @property
    def is_embedded(self) -> bool:
        return self.status == "embedded"

    @property
    def is_resolved_pending(self) -> bool:
        """Check if video has been resolved but not yet saved."""
        return self.status == "resolved_pending"

    @property
    def needs_resolution(self) -> bool:
        """Check if video still needs to be resolved (missing or unknown)."""
        return self.status in ("missing", "unknown")

    @property
    def status_icon(self) -> str:
        icons = {
            "found": "✓",
            "missing": "✗",
            "embedded": "◆",
            "resolved_pending": "✓",  # Checkmark to show it's resolved but pending save
            "unknown": "?",
        }
        return icons.get(self.status, "?")

    @property
    def display_path(self) -> str:
        """Get the path to display (resolved if available, otherwise original)."""
        return self.resolved_path if self.resolved_path else self.original_path


@dataclass
class SLPInfo:
    """Information about an SLP file and its videos."""

    path: str
    videos: list[VideoInfo] = field(default_factory=list)
    total_videos: int = 0
    accessible: int = 0
    missing: int = 0
    embedded: int = 0
    error: str = ""

    @classmethod
    def from_check_response(cls, path: str, data: dict) -> "SLPInfo":
        """Create SLPInfo from FS_CHECK_VIDEOS_RESPONSE data."""
        videos = []

        # Add accessible videos
        for v in data.get("accessible_videos", []):
            videos.append(
                VideoInfo(
                    filename=v.get("filename", ""),
                    original_path=v.get("path", ""),
                    status="found",
                )
            )

        # Add missing videos
        for v in data.get("missing", []):
            videos.append(
                VideoInfo(
                    filename=v.get("filename", ""),
                    original_path=v.get("original_path", ""),
                    status="missing",
                )
            )

        # Add embedded videos
        for v in data.get("embedded_videos", []):
            videos.append(
                VideoInfo(
                    filename=v.get("filename", "unknown"),
                    original_path="(embedded)",
                    status="embedded",
                )
            )

        return cls(
            path=path,
            videos=videos,
            total_videos=data.get("total_videos", len(videos)),
            accessible=data.get("accessible", 0),
            missing=len(data.get("missing", [])),
            embedded=data.get("embedded", 0),
            error=data.get("error", ""),
        )

    @property
    def all_resolved(self) -> bool:
        """Check if all videos are resolved (found, embedded, or resolved_pending)."""
        return self.missing == 0 and not self.error

    @property
    def has_pending_resolutions(self) -> bool:
        """Check if there are videos resolved but not yet saved."""
        return any(v.is_resolved_pending for v in self.videos)

    @property
    def pending_count(self) -> int:
        """Count of videos resolved but not yet saved."""
        return sum(1 for v in self.videos if v.is_resolved_pending)

    def get_filename_map(self) -> dict[str, str]:
        """Get mapping from original paths to resolved paths for all pending videos."""
        return {
            v.original_path: v.resolved_path
            for v in self.videos
            if v.is_resolved_pending and v.resolved_path
        }

    def mark_video_resolved(self, original_path: str, resolved_path: str):
        """Mark a video as resolved (pending save).

        Args:
            original_path: The original path to find.
            resolved_path: The new resolved path.
        """
        for video in self.videos:
            if video.original_path == original_path:
                video.status = "resolved_pending"
                video.resolved_path = resolved_path
                self.missing = sum(1 for v in self.videos if v.is_missing)
                break


class VideoListItem(ListItem):
    """A list item representing a video."""

    def __init__(self, video: VideoInfo, index: int, **kwargs):
        super().__init__(**kwargs)
        self.video = video
        self.index = index

    def compose(self) -> ComposeResult:
        status_class = f"video-{self.video.status}"

        if self.video.is_resolved_pending and self.video.resolved_path:
            # Show both old and new paths for resolved videos
            with Vertical(classes="video-paths"):
                yield Label(
                    f"{self.video.status_icon}  {self.video.original_path}",
                    classes="old-path",
                )
                yield Label(
                    f"   → {self.video.resolved_path}",
                    classes="new-path",
                )
        else:
            # Show just the filename with status
            yield Label(
                f"{self.video.status_icon}  {self.video.filename}",
                classes=status_class,
            )


class SLPContextPanel(Widget):
    """Panel showing SLP video status and resolution controls.

    This panel appears when an SLP file is selected in the Miller columns,
    showing video accessibility status and allowing path resolution.
    """

    DEFAULT_CSS = """
    SLPContextPanel {
        height: auto;
        min-height: 8;
        max-height: 20;
        border: solid $primary;
        padding: 1;
        margin: 1;
        display: none;
    }

    SLPContextPanel.visible {
        display: block;
    }

    #slp-header {
        height: 1;
        margin-bottom: 1;
    }

    #slp-title {
        text-style: bold;
        color: $primary;
    }

    #slp-summary {
        color: $text-muted;
    }

    #video-list {
        height: auto;
        max-height: 10;
        border: solid $primary-darken-2;
        margin-bottom: 1;
    }

    .video-found {
        color: $success;
    }

    .video-missing {
        color: $error;
    }

    .video-embedded {
        color: $text-muted;
    }

    .video-resolved_pending {
        color: $warning;
    }

    .video-paths {
        height: auto;
    }

    .old-path {
        color: $text-muted;
        text-style: strike;
    }

    .new-path {
        color: $success;
        text-style: bold;
    }

    #controls {
        height: auto;
    }

    #fix-hint {
        color: $warning;
    }

    #all-resolved {
        color: $success;
    }

    #ready-to-save {
        color: $warning;
        text-style: bold;
    }

    #error-message {
        color: $error;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("f", "fix_selected", "Fix selected video", show=True),
        Binding("escape", "hide_panel", "Hide panel", show=False),
    ]

    # Messages
    class FixRequested(Message):
        """Emitted when user wants to fix a missing video."""

        def __init__(self, video: VideoInfo, slp_info: SLPInfo):
            super().__init__()
            self.video = video
            self.slp_info = slp_info

    class PanelHidden(Message):
        """Emitted when panel is hidden."""

        pass

    # Reactive properties
    slp_path = reactive("")

    def __init__(
        self,
        on_fix_requested: Optional[Callable[[VideoInfo, SLPInfo], Any]] = None,
        **kwargs,
    ):
        """Initialize SLP context panel.

        Args:
            on_fix_requested: Callback when fix is requested for a video.
        """
        super().__init__(**kwargs)
        self.on_fix_requested = on_fix_requested
        self.slp_info: Optional[SLPInfo] = None
        self._selected_video_index: int = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="slp-header"):
            yield Static("SLP Video Status", id="slp-title")
            yield Static("", id="slp-summary")

        yield ListView(id="video-list")

        with Vertical(id="controls"):
            yield Static(
                "Navigate to video file, press 'f' to fix. Press 's' to save when done.",
                id="fix-hint",
            )
            yield Static(
                "All videos resolved! Press 's' to save new SLP.",
                id="ready-to-save",
                classes="hidden",
            )
            yield Static("All videos found!", id="all-resolved", classes="hidden")
            yield Static("", id="error-message", classes="hidden")

    def show(self, slp_info: SLPInfo):
        """Show the panel with SLP info.

        Args:
            slp_info: Information about the SLP file and its videos.
        """
        self.slp_info = slp_info
        self.slp_path = slp_info.path
        self.add_class("visible")

        # Update summary - include pending count if any
        summary = self.query_one("#slp-summary", Static)
        pending = slp_info.pending_count
        if pending > 0:
            summary.update(
                f"  ({slp_info.accessible} found, {slp_info.missing} missing, "
                f"{pending} pending, {slp_info.embedded} embedded)"
            )
        else:
            summary.update(
                f"  ({slp_info.accessible} found, {slp_info.missing} missing, "
                f"{slp_info.embedded} embedded)"
            )

        # Populate video list
        video_list = self.query_one("#video-list", ListView)
        video_list.clear()

        for i, video in enumerate(slp_info.videos):
            item = VideoListItem(video, i)
            video_list.append(item)

        # Update controls visibility
        fix_hint = self.query_one("#fix-hint", Static)
        ready_to_save = self.query_one("#ready-to-save", Static)
        all_resolved = self.query_one("#all-resolved", Static)
        error_msg = self.query_one("#error-message", Static)

        # Hide all first
        fix_hint.add_class("hidden")
        ready_to_save.add_class("hidden")
        all_resolved.add_class("hidden")
        error_msg.add_class("hidden")

        if slp_info.error:
            error_msg.remove_class("hidden")
            error_msg.update(f"Error: {slp_info.error}")
        elif slp_info.missing == 0 and slp_info.has_pending_resolutions:
            # All videos resolved (pending save)
            ready_to_save.remove_class("hidden")
        elif slp_info.all_resolved and not slp_info.has_pending_resolutions:
            # All videos already found (no changes needed)
            all_resolved.remove_class("hidden")
        else:
            # Still have missing videos
            fix_hint.remove_class("hidden")

        # Focus video list if there are missing videos
        if slp_info.missing > 0:
            video_list.focus()
            # Select first missing video
            for i, video in enumerate(slp_info.videos):
                if video.is_missing:
                    video_list.index = i
                    self._selected_video_index = i
                    break

    def hide(self):
        """Hide the panel."""
        self.remove_class("visible")
        self.slp_info = None
        self.post_message(self.PanelHidden())

    def refresh_display(self):
        """Refresh the panel display with current SLP info."""
        if self.slp_info:
            self.show(self.slp_info)

    def get_selected_video(self) -> Optional[VideoInfo]:
        """Get the currently selected video."""
        if not self.slp_info:
            return None

        video_list = self.query_one("#video-list", ListView)
        if video_list.index is not None and 0 <= video_list.index < len(
            self.slp_info.videos
        ):
            return self.slp_info.videos[video_list.index]
        return None

    def get_missing_videos(self) -> list[VideoInfo]:
        """Get list of missing videos."""
        if not self.slp_info:
            return []
        return [v for v in self.slp_info.videos if v.is_missing]

    def update_video_resolved(self, index: int, resolved_path: str):
        """Mark a video as resolved.

        Args:
            index: Index of the video in the list.
            resolved_path: The resolved path for the video.
        """
        if not self.slp_info or index >= len(self.slp_info.videos):
            return

        video = self.slp_info.videos[index]
        video.status = "found"
        video.resolved_path = resolved_path

        # Update counts
        self.slp_info.missing -= 1
        self.slp_info.accessible += 1

        # Refresh display
        self.show(self.slp_info)

    def on_list_view_selected(self, event: ListView.Selected):
        """Handle video selection."""
        item = event.item
        if isinstance(item, VideoListItem):
            self._selected_video_index = item.index

    def action_fix_selected(self):
        """Request fix for the selected video."""
        video = self.get_selected_video()
        if video and video.is_missing and self.slp_info:
            self.post_message(self.FixRequested(video, self.slp_info))
            if self.on_fix_requested:
                self.on_fix_requested(video, self.slp_info)

    def action_hide_panel(self):
        """Hide the panel."""
        self.hide()
