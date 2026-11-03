"""Room secret input screen for PSK authentication.

This screen prompts the user to enter a room secret when PSK authentication
is required but no secret is configured.
"""

from typing import Callable, Optional

from textual.app import ComposeResult
from textual.containers import Vertical, Center
from textual.screen import ModalScreen
from textual.widgets import Static, Input, Button
from textual.binding import Binding


class SecretInputScreen(ModalScreen[Optional[str]]):
    """Modal screen for entering a room secret.

    This screen is shown when a worker requires PSK authentication
    but the client doesn't have a room secret configured.

    Returns:
        The entered secret string, or None if cancelled.
    """

    DEFAULT_CSS = """
    SecretInputScreen {
        align: center middle;
    }

    #secret-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: solid $primary;
    }

    #secret-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    #secret-message {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    #secret-error {
        text-align: center;
        color: $error;
        margin-bottom: 1;
    }

    #secret-input {
        margin: 1 0;
    }

    #button-row {
        align: center middle;
        margin-top: 1;
    }

    #button-row Button {
        margin: 0 1;
    }

    #submit-btn {
        background: $primary;
    }

    #cancel-btn {
        background: $surface-darken-1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "submit", "Submit", show=False),
    ]

    def __init__(
        self,
        room_id: str,
        error_message: Optional[str] = None,
        on_submit: Optional[Callable[[str], None]] = None,
        name: Optional[str] = None,
    ):
        """Initialize secret input screen.

        Args:
            room_id: Room ID for display purposes.
            error_message: Optional error message to display (e.g., from previous failed attempt).
            on_submit: Optional callback when secret is submitted.
            name: Screen name.
        """
        super().__init__(name=name)
        self.room_id = room_id
        self.error_message = error_message
        self.on_submit = on_submit

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="secret-dialog"):
                yield Static("Room Secret Required", id="secret-title")
                yield Static(
                    f"The worker requires PSK authentication.\n"
                    f"Enter the room secret for '{self.room_id}':",
                    id="secret-message",
                )
                if self.error_message:
                    yield Static(f"Error: {self.error_message}", id="secret-error")
                yield Input(
                    placeholder="Enter room secret...",
                    password=True,
                    id="secret-input",
                )
                with Center(id="button-row"):
                    yield Button("Submit", id="submit-btn", variant="primary")
                    yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        """Focus the input when mounted."""
        self.query_one("#secret-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "submit-btn":
            self.action_submit()
        elif event.button.id == "cancel-btn":
            self.action_cancel()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key in input."""
        self.action_submit()

    def action_submit(self) -> None:
        """Submit the entered secret."""
        input_widget = self.query_one("#secret-input", Input)
        secret = input_widget.value.strip()

        if not secret:
            # Show error for empty input
            self.app.notify("Please enter a room secret", severity="warning")
            return

        # Call callback if provided
        if self.on_submit:
            self.on_submit(secret)

        # Dismiss with the secret value
        self.dismiss(secret)

    def action_cancel(self) -> None:
        """Cancel and dismiss without a secret."""
        self.dismiss(None)
