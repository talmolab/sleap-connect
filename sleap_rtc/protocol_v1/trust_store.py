"""Persisted allowlist of paired client node IDs.

Separate from the legacy account/room credential schema (`credentials.py`)
— this is the pairing-model's trust store: just a set of client public keys
(`node_id`s) this worker has paired with and will accept `auth.prove` from
on future connections. No accounts, no rooms.
"""

import json
import os
import stat
from pathlib import Path
from typing import Set, Union

DEFAULT_TRUST_STORE_PATH = (
    Path.home() / ".sleap-rtc" / "protocol_v1_trusted_clients.json"
)


class TrustStore:
    """A durable set of trusted client `node_id`s."""

    def __init__(self, path: Union[str, Path] = DEFAULT_TRUST_STORE_PATH):
        """Load the trust store from `path` (empty if it doesn't exist yet).

        Args:
            path: Where to persist the trusted-client set. Defaults to a
                per-user path under the home directory; tests should pass a
                `tmp_path`.
        """
        self._path = Path(path)
        self._trusted: Set[str] = self._load()

    def is_trusted(self, node_id: str) -> bool:
        """Check whether `node_id` has been paired with this worker."""
        return node_id in self._trusted

    def add_trusted(self, node_id: str) -> None:
        """Add `node_id` to the trusted set and persist it immediately."""
        if node_id in self._trusted:
            return
        self._trusted.add(node_id)
        self._save()

    def list_trusted(self) -> Set[str]:
        """Return a copy of the current trusted-client set."""
        return set(self._trusted)

    def _load(self) -> Set[str]:
        if not self._path.exists():
            return set()
        data = json.loads(self._path.read_text())
        return set(data.get("trusted_node_ids", []))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps({"trusted_node_ids": sorted(self._trusted)}, indent=2)
        )
        os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temp_path, self._path)
