"""A pickle_module for torch.load that tolerates un-importable classes.

RFdiffusion2 checkpoints embed an OmegaConf config and references to upstream
classes, so a plain `torch.load` needs the *entire* upstream dependency tree
(dgl, openbabel, rdkit, ...) importable just to read the tensors.

For the SOP §1.5 inventory we only want the tensors and the config values, so
any global that fails to import is replaced by a stub class that records what it
was. Tensors are unaffected: they come through torch's persistent-id storage
path, not through `find_class`.

This is ONLY used for inventory/loader-validation. `ref_dump.py` imports the
real, unmodified upstream modules, as the SOP requires.
"""
import pickle
from pickle import Pickler, dump, dumps  # noqa: F401  (torch expects these)

HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
DEFAULT_PROTOCOL = getattr(pickle, "DEFAULT_PROTOCOL", 2)

_STUBBED = set()


class Stub:
    """Placeholder for a class that could not be imported."""

    __qualname_stub__ = ("?", "?")

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state

    # OmegaConf/dataclass objects are sometimes rebuilt via these
    def __reduce__(self):
        return (Stub, ())

    def __repr__(self):
        m, n = self.__class__.__qualname_stub__
        return f"<stub {m}.{n}>"


def _make_stub(mod_name, name):
    _STUBBED.add(f"{mod_name}.{name}")
    return type(name, (Stub,), {"__qualname_stub__": (mod_name, name)})


class Unpickler(pickle.Unpickler):
    def find_class(self, mod_name, name):
        try:
            return super().find_class(mod_name, name)
        except Exception:
            return _make_stub(mod_name, name)


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()


def loads(data, **kwargs):
    import io
    return Unpickler(io.BytesIO(data), **kwargs).load()


def stubbed():
    """Names that had to be stubbed during the last load (for the record)."""
    return sorted(_STUBBED)
