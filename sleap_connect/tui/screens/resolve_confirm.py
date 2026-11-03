"""Resolution confirmation screen for TUI.

This screen shows a confirmation dialog when the user attempts to fix video
paths, displaying which videos will be resolved and allowing confirmation.
"""

from typing import Optional, Callable

from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static, Button, Label
from textual.binding import Binding


class ResolveConfirmScreen(ModalScreen):
    """Modal screen for confirming video path resolution.

    Shows the prefix transformation and which videos will be resolved.
    """

    CSS = """
    ResolveConfirmScreen {
        align: center middle;
    }

    #confirm-box {
        width: 90%;
        max-width: 120;
        height: auto;
        max-height: 85%;
        padding: 2 4;
        border: solid $primary;
        background: $surface;
    }

    .title {
        text-style: bold;
        text-align: center;
        color: $primary;
        margin-bottom: 1;
    }

    #prefix-section {
        margin: 1 0;
        padding: 1;
        border: solid $primary-darken-2;
    }

    .prefix-label {
        color: $text-muted;
    }

    .old-prefix {
        color: $text-muted;
        text-style: strike;
        margin-left: 2;
    }

    .new-prefix {
        color: $success;
        margin-left: 2;
    }

    #videos-section {
        margin: 1 0;
    }

    .section-title {
        text-style: bold;
        margin-bottom: 1;
        color: $success;
    }

    #video-list {
        height: auto;
        max-height: 20;
        border: solid $primary-darken-2;
        padding: 1;
    }

    .video-item {
        margin-bottom: 1;
    }

    .video-old {
        color: $text-muted;
        text-style: strike;
    }

    .video-new {
        color: $success;
        text-style: bold;
    }

    #still-missing-section {
        margin: 1 0;
        padding: 1;
        border: solid $warning;
    }

    .warning-text {
        color: $warning;
    }

    #buttons {
        margin-top: 2;
        align: center middle;
        height: auto;
    }

    Button {
        margin: 0 1;
    }

    #confirm-btn {
        background: $success;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Confirm", show=False),
    ]

    def __init__(
        self,
        old_prefix: str,
        new_prefix: str,
        videos_to_resolve: list[dict],  # [{"original": ..., "resolved": ...}]
        still_missing: list[str],
        on_confirm: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
    ):
        """Initialize resolution confirmation screen.

        Args:
            old_prefix: The old prefix being replaced.
            new_prefix: The new prefix to use.
            videos_to_resolve: List of videos that will be resolved.
            still_missing: List of paths that won't be resolved.
            on_confirm: Callback when user confirms.
            on_cancel: Callback when user cancels.
            name: Screen name.
        """
        super().__init__(name=name)
        self.old_prefix = old_prefix
        self.new_prefix = new_prefix
        self.videos_to_resolve = videos_to_resolve
        self.still_missing = still_missing
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel

    def compose(self) -> ComposeResult:
        yield Container(
            Vertical(
                Static("Confirm Video Path Resolution", classes="title"),
                # Prefix transformation section
                Vertical(
                    Static("Prefix Transformation:", classes="prefix-label"),
                    Static(f"Old: {self.old_prefix or '(none)'}", classes="old-prefix"),
                    Static(f"New: {self.new_prefix}", classes="new-prefix"),
                    id="prefix-section",
                ),
                # Videos that will be resolved
                Vertical(
                    Static(
                        f"Videos to be resolved ({len(self.videos_to_resolve)}):",
                        classes="section-title",
                    ),
                    VerticalScroll(
                        *self._build_video_items(),
                        id="video-list",
                    ),
                    id="videos-section",
                ),
                # Still missing warning (if any)
                *(self._build_missing_section()),
                # Buttons
                Container(
                    Button("Confirm", variant="primary", id="confirm-btn"),
                    Button("Cancel", variant="default", id="cancel-btn"),
                    id="buttons",
                ),
                id="confirm-box",
            ),
        )

    def _build_video_items(self) -> list:
        """Build list of video items showing old → new paths."""
        items = []
        for video in self.videos_to_resolve:
            items.append(
                Vertical(
                    Label(f"  {video['original']}", classes="video-old"),
                    Label(f"  → {video['resolved']}", classes="video-new"),
                    classes="video-item",
                )
            )
        return items

    def _build_missing_section(self) -> list:
        """Build the still-missing warning section if needed."""
        if not self.still_missing:
            return []

        return [
            Vertical(
                Static(
                    f"Warning: {len(self.still_missing)} video(s) will still be missing:",
                    classes="warning-text",
                ),
                Static(
                    "\n".join(f"  - {p}" for p in self.still_missing[:5]),
                    classes="warning-text",
                ),
                *(
                    [
                        Static(
                            f"  ... and {len(self.still_missing) - 5} more",
                            classes="warning-text",
                        )
                    ]
                    if len(self.still_missing) > 5
                    else []
                ),
                id="still-missing-section",
            )
        ]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "confirm-btn":
            self.action_confirm()
        elif event.button.id == "cancel-btn":
            self.action_cancel()

    def action_confirm(self) -> None:
        """Confirm and apply resolution."""
        if self.on_confirm:
            self.on_confirm()
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel and dismiss."""
        if self.on_cancel:
            self.on_cancel()
        self.dismiss(False)
