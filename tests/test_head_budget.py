"""Auto-allocate head_max_len so high-cardinality choice keeps 4 tokens per option.

Arithmetic only. Do not construct Agent (that downloads Hub weights). Cover
head_budget_for fully, and Agent via _resolve_head_budget plus source inspect.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import Agent  # noqa: E402
from laya.common import (  # noqa: E402
    HEAD_OPTION_SLACK,
    MIN_OPTION_TOKENS,
    build_sequence,
    head_budget_for,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


# --------------------------------------------------------------- named floor
check("MIN_OPTION_TOKENS is 4", MIN_OPTION_TOKENS, 4)
check("HEAD_OPTION_SLACK is 16", HEAD_OPTION_SLACK, 16)
_bs = inspect.getsource(build_sequence)
check_true("build_sequence uses MIN_OPTION_TOKENS", "MIN_OPTION_TOKENS" in _bs)
check_true("build_sequence uses HEAD_OPTION_SLACK", "HEAD_OPTION_SLACK" in _bs)


# --------------------------------------------------------------- k=4 stays put
b = head_budget_for(4, 192, 512, 512)
check("k4/tokens_per_option >= 4", b.tokens_per_option >= 4, True)
check("k4/raised False", b.raised, False)
check("k4/ok", b.ok, True)
check("k4/head unchanged", b.head_max_len, 192)
check("k4/max_len unchanged", b.max_len, 512)
check("k4/tpo formula", b.tokens_per_option, max(1, (192 - 16) // 4))


# --------------------------------------------------------------- Banking77 multilingual defaults
b = head_budget_for(77, 256, 1024, 1024)
check("k77/ok", b.ok, True)
check("k77/raised True", b.raised, True)
check_true("k77/tokens_per_option >= 4", b.tokens_per_option >= 4, "got %r" % b.tokens_per_option)
check_true("k77/head >= 16+4*77", b.head_max_len >= 16 + 4 * 77, "got %r" % b.head_max_len)
check("k77/smallest head", b.head_max_len, 16 + 4 * 77)
check("k77/tokens_per_option == 4", b.tokens_per_option, 4)
check("k77/max_len stays in encoder", b.max_len, 1024)


# --------------------------------------------------------------- 255 options cannot fit in 512
b = head_budget_for(255, 192, 512, 512)
check("k255/ok is False", b.ok, False)
check("k255/raised False", b.raised, False)
check("k255/keeps default head", b.head_max_len, 192)
check("k255/keeps default max_len", b.max_len, 512)


# --------------------------------------------------------------- k < 2 never raises
b = head_budget_for(1, 16, 512, 512)
check("k1/ok even if tpo < 4", b.ok, True)
check("k1/raised False", b.raised, False)
check("k0/ok", head_budget_for(0, 192, 512, 512).ok, True)
check("k0/raised", head_budget_for(0, 192, 512, 512).raised, False)


# --------------------------------------------------------------- english 77-way on a 1024 encoder
b = head_budget_for(77, 192, 512, 1024)
check("en77/ok", b.ok, True)
check("en77/raised", b.raised, True)
check("en77/head", b.head_max_len, 16 + 4 * 77)
check_true("en77/max_len grew with doc room", b.max_len >= 324 + 64, "got %r" % b.max_len)
check_true("en77/max_len <= encoder", b.max_len <= 1024, "got %r" % b.max_len)


# --------------------------------------------------------------- Agent._resolve_head_budget (no weights)
cfg = {"head_max_len": 256, "max_len": 1024}
low = {
    "dept": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"a": None, "b": None, "c": None, "d": None},
    }
}
head, mx, report = Agent._resolve_head_budget(low, cfg)
check("resolve/k4 head unchanged", head, 256)
check("resolve/k4 max unchanged", mx, 1024)
check("resolve/k4 raised", report["dept"]["raised"], False)
check("resolve/k4 k", report["dept"]["k"], 4)

labels = ["l%02d" % i for i in range(77)]
high = {
    "intent": {
        "type": "choice",
        "instructions": "Which Banking77 intent?",
        "criteria": labels,
    }
}
head, mx, report = Agent._resolve_head_budget(high, cfg)
check("resolve/k77 raised", report["intent"]["raised"], True)
check("resolve/k77 k", report["intent"]["k"], 77)
check_true("resolve/k77 tpo >= 4", report["intent"]["tokens_per_option"] >= 4,
           "got %r" % report["intent"]["tokens_per_option"])
check_true("resolve/k77 head >= 16+4*77", head >= 16 + 4 * 77, "got %r" % head)
check("resolve/k77 report head matches", report["intent"]["head_max_len"], head)

# one forward, one cfg: mixed low+high takes the max required head
mixed = dict(low)
mixed.update(high)
head, mx, report = Agent._resolve_head_budget(mixed, cfg)
check("resolve/mixed shares raised head", report["dept"]["head_max_len"], report["intent"]["head_max_len"])
check("resolve/mixed raised on both", report["dept"]["raised"] and report["intent"]["raised"], True)
check("resolve/mixed k preserved", (report["dept"]["k"], report["intent"]["k"]), (4, 77))

# overflow stays at defaults so predict can still raise
overflow = {
    "huge": {
        "type": "choice",
        "instructions": "pick",
        "criteria": ["c%d" % i for i in range(255)],
    }
}
head, mx, report = Agent._resolve_head_budget(overflow, {"head_max_len": 192, "max_len": 512, "encoder_max": 512})
check("resolve/255 keeps default head", head, 192)
check("resolve/255 report raised", report["huge"]["raised"], False)
check("resolve/255 k", report["huge"]["k"], 255)

# noul is 2 options; never needs a raise at defaults
noul = {"flag": {"type": "noul", "instructions": "is this true?"}}
head, mx, report = Agent._resolve_head_budget(noul, {"head_max_len": 192, "max_len": 512})
check("resolve/noul k", report["flag"]["k"], 2)
check("resolve/noul raised", report["flag"]["raised"], False)


# --------------------------------------------------------------- predict default path unchanged
import laya.agent as _agent  # noqa: E402

_src = inspect.getsource(_agent.Agent.system_one)
check_true("system_one/auto_head_budget defaults False",
           "auto_head_budget: bool = False" in _src)
check_true("system_one/persist defaults False", "persist: bool = False" in _src)
check_true("system_one/auto is a local flag", "auto = bool(auto_head_budget or self.cfg.get(\"auto_head_budget\"))" in _src)
check_true("system_one/resolve only when auto", "if auto:" in _src and "_resolve_head_budget" in _src)
check_true("system_one/persist writes cfg", 'self.cfg["head_max_len"] = head_max_len' in _src)
check_true("system_one/report only on auto path",
           'usage["head_budget"] = head_budget_report' in _src)
check_true("system_one/no two_stage_choice", "two_stage_choice" not in _src)
check_true("helper/no two_stage_choice", "two_stage_choice" not in inspect.getsource(head_budget_for))
check_true("predict is system_one", _agent.Agent.predict is _agent.Agent.system_one)

# staticmethod: callable without an instance / without Hub
check_true("_resolve_head_budget is static",
           isinstance(inspect.getattr_static(_agent.Agent, "_resolve_head_budget"), staticmethod))


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
