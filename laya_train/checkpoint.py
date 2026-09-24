"""Find, load and write Laya checkpoints.

A checkpoint is a directory holding `rl_agent_config.json`, `model.safetensors`,
`encoder/` (the encoder config) and `tokenizer/`, the layout `laya.Agent` loads. Names
("english", "multilingual", "typed-decisions") resolve to the published checkpoints.
"""
import json
import os
import shutil
import time
from typing import Optional, Tuple

import torch

from laya.agent import _fix_tokenizer_config
from laya.common import build_model

FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


def resolve(name_or_path: str, token: Optional[str] = None) -> str:
    """A local directory for a checkpoint name, hub id or path (downloads if needed)."""
    if os.path.isdir(name_or_path):
        if not os.path.exists(os.path.join(name_or_path, "rl_agent_config.json")):
            raise FileNotFoundError("%s has no rl_agent_config.json; not a Laya checkpoint" % name_or_path)
        return name_or_path
    from laya.router import DEFAULT_MODELS, normalise_name, _split
    from huggingface_hub import snapshot_download
    key = normalise_name(name_or_path)
    repo, sub = _split(DEFAULT_MODELS[key]) if key in DEFAULT_MODELS else (name_or_path, None)
    prefix = sub + "/" if sub else ""
    path = snapshot_download(repo, token=token or os.environ.get("HF_TOKEN"),
                             allow_patterns=[prefix + f for f in FILES])
    return os.path.join(path, sub) if sub else path


def read_config(ckpt: str) -> dict:
    with open(os.path.join(ckpt, "rl_agent_config.json"), encoding="utf-8") as f:
        return json.load(f)


def load_for_training(ckpt: str) -> Tuple[torch.nn.Module, object, dict]:
    """The model (on CPU, float32), its tokenizer and config, ready to train."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    ckpt = resolve(ckpt)
    _fix_tokenizer_config(ckpt)
    cfg = read_config(ckpt)
    tok = AutoTokenizer.from_pretrained(os.path.join(ckpt, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(ckpt, "encoder"))
    model.load_state_dict(load_file(os.path.join(ckpt, "model.safetensors")), strict=True)
    try:
        model.encoder.config.reference_compile = False     # as laya.Agent: no torch.compile
    except Exception:
        pass
    return model.float(), tok, cfg


def write(out_dir: str, cfg: dict, model: Optional[torch.nn.Module] = None, tok=None,
          source: Optional[str] = None, dtype: str = "fp16", link: bool = True) -> str:
    """Write a checkpoint. With `model`, its weights are saved; without, the weights,
    encoder and tokenizer of `source` are linked (or copied) and only the config changes,
    which is all a calibration needs."""
    os.makedirs(out_dir, exist_ok=True)
    if model is not None:
        from safetensors.torch import save_file
        cast = torch.float16 if dtype == "fp16" else torch.float32
        sd = {k: v.detach().to(cast).contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(out_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
        if tok is not None:
            tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    elif source is not None:
        for name in ("model.safetensors", "encoder", "tokenizer"):
            src, dst = os.path.realpath(os.path.join(source, name)), os.path.join(out_dir, name)
            if os.path.lexists(dst):
                if os.path.isdir(dst) and not os.path.islink(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            if link:
                os.symlink(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    else:
        raise ValueError("write() needs a model or a source checkpoint")
    cfg = dict(cfg)
    cfg.setdefault("history", [])
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return out_dir


def add_history(cfg: dict, step: dict) -> dict:
    """Record what was done to a checkpoint, newest last."""
    cfg = dict(cfg)
    cfg["history"] = list(cfg.get("history", [])) + [dict(step, at=time.strftime("%Y-%m-%d %H:%M"))]
    return cfg
