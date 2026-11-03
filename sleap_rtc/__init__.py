"""Old name of `sleap_connect`, kept so existing imports keep working.

`import sleap_rtc` and `import sleap_rtc.<anything>` resolve to the very same
module objects as `sleap_connect` / `sleap_connect.<anything>` (not copies),
so state and `isinstance` checks are shared. New code should import
`sleap_connect`.
"""

import importlib
import importlib.abc
import importlib.util
import sys

import sleap_connect

_OLD = __name__
_NEW = "sleap_connect"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str):
        self._target = target

    def create_module(self, spec):
        return importlib.import_module(self._target)

    def exec_module(self, module):
        # Already executed under its real name; nothing to run again.
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        new_name = _NEW + fullname[len(_OLD) :]
        if importlib.util.find_spec(new_name) is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(new_name))


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

sys.modules[_OLD] = sleap_connect
