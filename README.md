# SLEAP Connect (sleap-rtc)

Remote training and inference for [SLEAP](https://sleap.ai): run `sleap-nn` on a GPU machine
from [sleap-app](https://app.sleap.ai), with no account or server in between.

> The package and command are still called `sleap-rtc`. They'll be renamed to `sleap-connect`.

## Quick start

**1. Install on the GPU machine.** There's no release with this workflow yet, so install from `main`:

```bash
uv tool install --python 3.11 "sleap-rtc @ git+https://github.com/talmolab/sleap-connect" \
  --with "sleap-nn[torch]" \
  --with-executables-from sleap-nn \
  --torch-backend auto
```

**2. Start the worker in your data folder.**

```bash
cd /path/to/your/data
sleap-rtc serve
```

The first time, it asks whether to share this folder with the devices you pair. Answer **y**:
they can then browse it, and their jobs can read and write inside it. The answer is remembered.

The worker then runs in the background: you get your prompt back, and it keeps running after you
close the terminal or log out. Check on it with `sleap-rtc status` and stop it with
`sleap-rtc serve --stop`. So you don't forget it's running, new terminals show a one-line
reminder while it is (`sleap-rtc shell-hook uninstall` removes it). Use `--foreground` to keep it
attached to the terminal instead.

**3. Get a pairing code.**

```bash
sleap-rtc pair
```

It prints a one-line code starting with `sleap1`. It works once and expires after a few minutes.

**4. Pair from sleap-app.** In [app.sleap.ai](https://app.sleap.ai) or the desktop app, open
**Connect → Pair a worker** and paste the code. Then train or run inference from the Training
panel or the Connect window.

The full guide (connection routes, the Connect window, troubleshooting) is in the
[sleap-app docs](https://app.sleap.ai/docs/main/guides/remote-compute/).

## Commands

| Command | What it does |
|---|---|
| `sleap-rtc serve` | Start the worker in the background. `--stop` stops it; `--foreground` keeps it attached to the terminal (Ctrl-C to stop). |
| `sleap-rtc status` | Show whether the worker is running: PID, uptime, address, shared folders, paired devices, jobs. |
| `sleap-rtc pair` | Print a one-time pairing code for a new device. |
| `sleap-rtc folders` | List the folders shared with paired devices. |
| `sleap-rtc add-folder [FOLDER]` | Share another folder (relative paths are fine). |
| `sleap-rtc remove-folder FOLDER` | Stop sharing a folder (by path or label). |
| `sleap-rtc clients` | List paired devices: name, app, when paired, last seen. |
| `sleap-rtc clients remove NAME` | Unpair a device (by name or ID prefix). `--all` unpairs every device and cancels unused pairing codes. |
| `sleap-rtc doctor` | Check the machine's setup. |
| `sleap-rtc shell-hook uninstall` | Remove the new-terminal reminder (`install` adds it back; set `SLEAP_RTC_NO_SHELL_HOOK=1` to never add it). |

## How it works

- **The worker is the server.** `serve` listens for the app directly: a WebSocket on port 9631,
  result downloads on 9632, and an [iroh](https://iroh.computer) endpoint. There's no signaling
  server, room, or login.
- **Pairing decides who gets in.** A pairing code carries the worker's ID, its addresses and a
  one-time secret. The app claims it once; after that the worker trusts that device's key, and
  every connection proves it. Remove a device with `sleap-rtc clients remove`.
- **Two ways to connect.** On the same network or over a VPN such as Tailscale, the app connects
  to `ws://<worker>:9631`. The desktop app can also connect over iroh, which goes through a relay
  first and then switches to a direct connection when it can, so it works across networks.
  Relays only see encrypted traffic.
- **Shared folders.** Paired devices can only browse the folders the worker shares: the one
  `serve` started in, plus any added with `add-folder`. The worker's own settings folder
  (`~/.sleap-rtc`) is never shared.
- **Jobs live on the worker.** Jobs queue and keep running if the app disconnects or the worker
  restarts; the app catches up on progress when it reconnects.

## Installation options

Pick a PyTorch backend explicitly instead of `--torch-backend auto`:

```bash
# CUDA GPU
uv tool install --python 3.11 "sleap-rtc @ git+https://github.com/talmolab/sleap-connect" \
  --with "sleap-nn[torch-cuda130]" \
  --with-executables-from sleap-nn

# CPU only
uv tool install --python 3.11 "sleap-rtc @ git+https://github.com/talmolab/sleap-connect" \
  --with "sleap-nn[torch-cpu]" \
  --with-executables-from sleap-nn
```

To update, run the same command again with `--reinstall`.

## Legacy: rooms, dashboard and SLEAP GUI

The older workflow connects through a signaling server with GitHub login and rooms, over WebRTC.
It's still used by the [dashboard](https://talmolab.github.io/sleap-rtc/dashboard/index.html), the
SLEAP GUI's Remote Training dialog, and the `sleap-rtc train` / `sleap-rtc track` commands. It
will be retired; sleap-app uses the pairing workflow above.

<details>
<summary>Legacy setup</summary>

**Dashboard**

1. **Login** — Go to the [dashboard](https://talmolab.github.io/sleap-rtc/dashboard/index.html) and log in with GitHub.
2. **Create a room** — Click **Rooms** → **Create Room**.
3. **Generate an account key** — Go to **Account Keys** and create one. Copy it for the worker.
4. **Deploy a worker** — On your GPU machine:

   ```bash
   sleap-rtc login
   sleap-rtc config add-mount /path/to/your/data
   sleap-rtc worker --room <room-id>
   ```

5. **Submit a job** — In the dashboard, click **Submit Job** on your room card, select the worker, upload a config, and submit.

**CLI**

```bash
sleap-rtc login                              # GitHub login (opens browser)
sleap-rtc config add-mount /path/to/your/data
sleap-rtc worker --room <room-id>            # on the GPU machine
sleap-rtc train --room <room-id> ...         # from your machine
sleap-rtc track --room <room-id> ...
```

**Docker**

```bash
docker run --gpus all \
  -e SLEAP_RTC_ACCOUNT_KEY=<your-account-key> \
  -v /path/to/data:/app/shared_data \
  ghcr.io/talmolab/sleap-rtc:latest \
  worker --room <room-id>
```

</details>

## Links

- [SLEAP](https://sleap.ai) — pose estimation framework
- [sleap-nn](https://nn.sleap.ai) — neural network backend
- [sleap-app](https://app.sleap.ai) — SLEAP in the browser and on the desktop
- [Remote compute guide](https://app.sleap.ai/docs/main/guides/remote-compute/) — pairing, connection routes, troubleshooting
- [Dashboard](https://talmolab.github.io/sleap-rtc/dashboard/index.html) — legacy rooms and jobs
