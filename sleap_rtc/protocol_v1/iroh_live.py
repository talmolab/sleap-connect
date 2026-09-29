"""The "live iroh reachability" file that bridges `serve --iroh` and `pair` (item 2.1).

`sleap-rtc serve` and `sleap-rtc pair` are separate OS processes (see
`pairing.py`'s module docstring for the same situation with pairing
secrets). Only the running `serve` process knows its iroh endpoint's
current relay URL and direct addresses, so it publishes them to a small
JSON file under ``--data-dir``; `pair` reads that file, best-effort, and
embeds it in the printed ticket as an optional ``iroh`` section. A client
that finds that section can dial the worker over iroh with no VPN, SSH
tunnel or port-forward.

The file is purely advisory: any problem reading it (missing, unparseable,
for a different identity, or left behind by a `serve` that has since
died) means `pair` prints exactly the ticket it printed before this
feature existed.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

# How often `keep_iroh_live_updated` re-checks the endpoint's address.
# The Python iroh bindings' `Endpoint.watch_addr` panics when called
# outside a Tokio runtime (verified by hand), so polling is the only option.
DEFAULT_REFRESH_INTERVAL_SECS = 5.0


def iroh_live_path(data_dir: Path) -> Path:
    """Where the live iroh reachability file lives under `data_dir`."""
    return Path(data_dir) / "iroh_live.json"


def build_iroh_section(
    node_id: str, relay_url: Optional[str], direct_addrs: List[str]
) -> Dict[str, Any]:
    """The ticket's ``iroh`` section: everything a client needs to dial over iroh.

    ``node_id`` is the worker's usual URL-safe-base64 identity — the same
    string as the ticket's top-level ``node_id``, and (since the iroh
    endpoint is bound with the identity's own key) the iroh endpoint id.
    ``relay_url`` is ``None`` until the endpoint has picked a home relay,
    or when relays are disabled.
    """
    return {
        "node_id": node_id,
        "relay_url": relay_url,
        "direct_addrs": list(direct_addrs),
    }


def snapshot_iroh_section(endpoint: Any, node_id: str) -> Dict[str, Any]:
    """The endpoint's current reachability, as a ticket ``iroh`` section."""
    addr = endpoint.addr()
    return build_iroh_section(
        node_id, addr.relay_url(), sorted(addr.direct_addresses())
    )


def write_iroh_live(path: Union[str, Path], section: Dict[str, Any]) -> None:
    """Atomically write the live file, tagging it with this process's pid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps({**section, "pid": os.getpid(), "updated_at": time.time()})
    )
    os.replace(temp_path, path)


def remove_iroh_live(path: Union[str, Path]) -> None:
    """Delete the live file if present (a worker that's no longer serving iroh)."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def read_iroh_live(
    path: Union[str, Path], expected_node_id: str
) -> Optional[Dict[str, Any]]:
    """Read the ``iroh`` ticket section, or ``None`` if there isn't a usable one.

    Never raises: this feeds an optional ticket field, so every failure
    mode collapses to "no iroh section".

    Args:
        path: The live file (see `iroh_live_path`).
        expected_node_id: The identity `pair` is minting a ticket for; a
            file for any other identity is ignored.
    """
    try:
        data = json.loads(Path(path).read_text())
        if data["node_id"] != expected_node_id:
            return None
        relay_url = data.get("relay_url")
        direct_addrs = data.get("direct_addrs", [])
        if relay_url is not None and not isinstance(relay_url, str):
            return None
        if not isinstance(direct_addrs, list) or not all(
            isinstance(a, str) for a in direct_addrs
        ):
            return None
        if not relay_url and not direct_addrs:
            return None  # nothing to dial yet
        pid = data.get("pid")
        if isinstance(pid, int) and not psutil.pid_exists(pid):
            return None  # left behind by a serve that crashed / was killed
        return build_iroh_section(expected_node_id, relay_url, direct_addrs)
    except (OSError, ValueError, KeyError, TypeError):
        return None


async def keep_iroh_live_updated(
    endpoint: Any,
    path: Union[str, Path],
    node_id: str,
    interval: Optional[float] = None,
) -> None:
    """Rewrite the live file whenever the endpoint's address changes.

    The relay URL typically appears a moment after bind, and direct
    addresses change with the network, so a single write at startup isn't
    enough. Runs until cancelled.
    """
    if interval is None:
        interval = DEFAULT_REFRESH_INTERVAL_SECS
    last: Optional[Dict[str, Any]] = None
    while True:
        try:
            current = snapshot_iroh_section(endpoint, node_id)
            if current != last:
                write_iroh_live(path, current)
                last = current
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("[serve] failed to refresh iroh live file")
        await asyncio.sleep(interval)
