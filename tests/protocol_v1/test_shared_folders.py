"""Tests for the shared-folder list, its FileManager, and the folder CLI."""

from pathlib import Path

import click
from click.testing import CliRunner

from sleap_rtc.protocol_v1.cli import _share_folder, add_folder, folders, remove_folder
from sleap_rtc.protocol_v1.shared_folders import (
    SharedFolderFileManager,
    SharedFolders,
    is_broad,
    resolve_folder,
)


def _store(tmp_path) -> SharedFolders:
    return SharedFolders(tmp_path / "data" / "shared_folders.json")


class TestSharedFolders:
    def test_add_list_remove(self, tmp_path):
        store = _store(tmp_path)
        a = tmp_path / "a"
        a.mkdir()
        added = store.add(a.resolve())
        assert added.label == "a"
        assert [f.path for f in store.list()] == [str(a.resolve())]
        assert store.remove("a") == added
        assert store.list() == []

    def test_adding_twice_keeps_one_entry(self, tmp_path):
        store = _store(tmp_path)
        store.add(tmp_path.resolve())
        store.add(tmp_path.resolve())
        assert len(store.list()) == 1

    def test_labels_stay_unique(self, tmp_path):
        store = _store(tmp_path)
        (tmp_path / "x" / "data").mkdir(parents=True)
        (tmp_path / "y" / "data").mkdir(parents=True)
        store.add((tmp_path / "x" / "data").resolve())
        second = store.add((tmp_path / "y" / "data").resolve())
        assert second.label == "data (2)"

    def test_remove_by_path(self, tmp_path):
        store = _store(tmp_path)
        store.add(tmp_path.resolve(), "lab")
        assert store.remove(str(tmp_path)).label == "lab"

    def test_find_covers_subfolders(self, tmp_path):
        store = _store(tmp_path)
        store.add(tmp_path.resolve())
        assert store.find((tmp_path / "deep" / "file.slp").resolve()) is not None
        assert store.find(tmp_path.parent.resolve()) is None

    def test_changes_from_another_process_are_seen(self, tmp_path):
        running = _store(tmp_path)
        assert running.list() == []
        _store(tmp_path).add(tmp_path.resolve())
        assert len(running.list()) == 1

    def test_unreadable_file_shares_nothing(self, tmp_path):
        store = _store(tmp_path)
        store.add(tmp_path.resolve())
        (tmp_path / "data" / "shared_folders.json").write_text("{not json")
        assert store.list() == []


class TestHelpers:
    def test_resolve_relative_and_home(self, tmp_path):
        assert resolve_folder("sub", base=tmp_path) == (tmp_path / "sub").resolve()
        assert resolve_folder("~") == Path.home().resolve()

    def test_is_broad(self, tmp_path):
        assert is_broad(Path.home().resolve())
        assert is_broad(Path("/"))
        assert not is_broad((Path.home() / "some" / "data").resolve())


class TestFileManager:
    def test_browsing_is_limited_to_shared_folders_and_never_the_data_dir(self, tmp_path):
        shared_dir = tmp_path / "project"
        data_dir = shared_dir / ".sleap-rtc"
        (data_dir).mkdir(parents=True)
        store = SharedFolders(data_dir / "shared_folders.json")
        fm = SharedFolderFileManager(store, data_dir, serve_folder=shared_dir.resolve())
        assert fm._is_path_allowed(shared_dir / "labels.slp") is False  # nothing shared yet
        store.add(shared_dir.resolve())
        assert fm._is_path_allowed(shared_dir / "labels.slp") is True
        assert fm._is_path_allowed(data_dir / "identity.json") is False
        assert fm._is_path_allowed(tmp_path / "elsewhere") is False

    def test_mounts_are_live_and_serve_folder_is_reported(self, tmp_path):
        store = _store(tmp_path)
        serve = (tmp_path / "project").resolve()
        serve.mkdir()
        fm = SharedFolderFileManager(store, tmp_path / "data", serve_folder=serve)
        assert fm.get_mounts() == [] and fm.serve_folder is None
        store.add(tmp_path.resolve(), "lab")
        assert fm.get_mounts() == [{"path": str(tmp_path.resolve()), "label": "lab"}]
        assert fm.serve_folder == str(serve)  # inside a shared folder counts


def _confirm_cmd(store, path):
    @click.command()
    def cmd():
        shared = _share_folder(store, path, yes=False, interactive=True)
        click.echo(f"shared={shared}")

    return cmd


class TestPrompt:
    def test_yes_shares_and_is_remembered(self, tmp_path):
        store = _store(tmp_path)
        result = CliRunner().invoke(_confirm_cmd(store, tmp_path.resolve()), input="y\n")
        assert "read and write inside it" in result.output
        assert "shared=True" in result.output
        # Already shared: no second question.
        again = CliRunner().invoke(_confirm_cmd(store, tmp_path.resolve()))
        assert "Share this folder?" not in again.output and "shared=True" in again.output

    def test_no_leaves_it_unshared(self, tmp_path):
        store = _store(tmp_path)
        result = CliRunner().invoke(_confirm_cmd(store, tmp_path.resolve()), input="n\n")
        assert "shared=False" in result.output
        assert store.list() == []

    def test_home_folder_gets_a_warning(self, tmp_path):
        store = _store(tmp_path)
        result = CliRunner().invoke(_confirm_cmd(store, Path.home().resolve()), input="n\n")
        assert "whole home folder" in result.output

    def test_without_a_terminal_nothing_is_shared(self, tmp_path):
        store = _store(tmp_path)
        assert _share_folder(store, tmp_path.resolve(), yes=False, interactive=False) is False
        assert _share_folder(store, tmp_path.resolve(), yes=True, interactive=False) is True


class TestFolderCommands:
    def _run(self, cmd, tmp_path, *args):
        return CliRunner().invoke(cmd, [*args, "--data-dir", str(tmp_path / "data")])

    def test_add_relative_folder_then_list_and_remove(self, tmp_path, monkeypatch):
        (tmp_path / "videos").mkdir()
        monkeypatch.chdir(tmp_path)
        added = self._run(add_folder, tmp_path, "videos", "--label", "lab", "--yes")
        assert added.exit_code == 0, added.output
        assert f'Shared {(tmp_path / "videos").resolve()} as "lab"' in added.output

        listed = self._run(folders, tmp_path)
        assert "lab" in listed.output and str((tmp_path / "videos").resolve()) in listed.output

        removed = self._run(remove_folder, tmp_path, "lab")
        assert 'Stopped sharing "lab"' in removed.output
        assert "No shared folders" in self._run(folders, tmp_path).output

    def test_add_without_a_terminal_needs_yes(self, tmp_path):
        result = self._run(add_folder, tmp_path, str(tmp_path))
        assert result.exit_code != 0
        assert "--yes" in result.output

    def test_add_a_folder_already_covered(self, tmp_path):
        (tmp_path / "sub").mkdir()
        self._run(add_folder, tmp_path, str(tmp_path), "--yes")
        result = self._run(add_folder, tmp_path, str(tmp_path / "sub"), "--yes")
        assert "Already shared" in result.output

    def test_add_a_missing_folder(self, tmp_path):
        result = self._run(add_folder, tmp_path, str(tmp_path / "nope"), "--yes")
        assert result.exit_code != 0 and "Not a folder" in result.output

    def test_remove_unknown(self, tmp_path):
        result = self._run(remove_folder, tmp_path, "nope")
        assert result.exit_code != 0 and "No shared folder matches" in result.output

    def test_missing_folders_are_flagged_in_the_list(self, tmp_path):
        gone = tmp_path / "gone"
        gone.mkdir()
        self._run(add_folder, tmp_path, str(gone), "--yes")
        gone.rmdir()
        assert "(missing)" in self._run(folders, tmp_path).output
