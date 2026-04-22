__version__ = "1.0.1"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba

__all__ = [
    "selective_scan_fn",
    "mamba_inner_fn",
    "bimamba_inner_fn",
    "Mamba",
]


def __getattr__(name):
    if name == "MambaLMHeadModel":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        return MambaLMHeadModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
