import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional


def resolve_custom_bitsandbytes_root() -> Path:
    return Path(__file__).resolve().parents[2] / "custom_bitsandbytes"


def enable_custom_bitsandbytes() -> Optional[Path]:
    custom_bnb_root = resolve_custom_bitsandbytes_root()
    if not custom_bnb_root.exists():
        warnings.warn(
            f"Custom bitsandbytes root not found: {custom_bnb_root}. "
            "Falling back to the installed bitsandbytes package."
        )
        return None

    custom_bnb_root_str = str(custom_bnb_root)
    if custom_bnb_root_str not in sys.path:
        sys.path.insert(0, custom_bnb_root_str)
    return custom_bnb_root


enable_custom_bitsandbytes()


def iter_quant_act_modules(model) -> Iterable[object]:
    try:
        from bitsandbytes.quantization_utils.quant_modules import QuantAct
    except Exception:
        return ()

    return (
        module
        for _, module in model.named_modules()
        if isinstance(module, QuantAct)
    )


def set_quant_act_mode(model, calibrate=None, search=None) -> int:
    module_count = 0
    for module in iter_quant_act_modules(model):
        if calibrate is not None and hasattr(module, "set_calibrate"):
            module.set_calibrate(calibrate=calibrate)
        if search is not None and hasattr(module, "set_search"):
            module.set_search(search=search)
        module_count += 1
    return module_count
