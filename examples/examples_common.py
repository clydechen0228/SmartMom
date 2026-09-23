"""Shared bits for the example scripts."""
import os

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}


def pick_device():
    """Device for the examples, honouring $LAYA_DEVICE.

    Laya would otherwise pick 'mps' on any Mac with a Metal GPU, but autocast on mps
    only exists from torch 2.3. Intel Macs top out at torch 2.2.2 (no newer macOS
    x86_64 wheels on PyPI), where a forward pass dies with
    "unsupported autocast device_type 'mps'". Fall back to cpu there.
    """
    override = os.environ.get("LAYA_DEVICE")
    if override:
        return override

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.autocast(device_type="mps", dtype=torch.float16, enabled=False)
        except RuntimeError:
            return "cpu"
        return "mps"
    return "cpu"
