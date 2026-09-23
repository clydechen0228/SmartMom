"""The typed-decisions checkpoint and its four workflows.

Workflow routing is opt-in: pass `auto_task_detection=True`, and a question set whose
ids match a workflow signature exactly is sent to the typed-decisions checkpoint.

    python examples/05_typed_decisions.py
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import Router  # noqa: E402
from laya.router import _TYPED_DECISION_WORKFLOWS  # noqa: E402

from examples_common import pick_device  # noqa: E402

# Exactly the id set of the 'customer_service' workflow — any extra or missing id and
# the router falls back to language-based routing.
CUSTOMER_SERVICE = {
    "category": {"type": "choice", "instructions": "What is this about?",
                 "criteria": {"billing": "invoices, payments, refunds",
                              "technical": "bugs, outages, errors",
                              "account": "logins, settings, access",
                              "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "action": {"type": "choice", "instructions": "What should we do next?",
               "criteria": {"refund": "issue a refund", "escalate": "hand to a specialist",
                            "reply": "answer directly", "close": "no action needed"}},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to leave?"},
    "needs_human": {"type": "noul", "instructions": "Does this need a human agent?"},
}

STATE = {"from": "user@acme.com",
         "subject": "Duplicate charge on invoice #4411",
         "body": "We were billed twice for March. Refund the duplicate today or we cancel."}


def main():
    print("known workflows:")
    for name, sig in _TYPED_DECISION_WORKFLOWS.items():
        print("  %-28s %s" % (name, sorted(sig)))

    router = Router(preload=False, device=pick_device(), auto_task_detection=True)

    decision = router.route(STATE, CUSTOMER_SERVICE)
    print("\nroute -> %s\n  why: %s\n  workflow: %s"
          % (decision.model, decision.reason, decision["workflow"]))

    res = router.predict(STATE, CUSTOMER_SERVICE)
    a = res["answers"]
    print("\nanswers (model=%s):" % res["routing"]["model"])
    print("  category    : %s (%.2f)" % (a["category"]["choice"], a["category"]["confidence"]))
    print("  urgency     : %.2f" % a["urgency"]["score"])
    print("  action      : %s (%.2f)" % (a["action"]["choice"], a["action"]["confidence"]))
    print("  churn_risk  : %.2f" % a["churn_risk"]["noul"])
    print("  needs_human : %.2f" % a["needs_human"]["noul"])


if __name__ == "__main__":
    main()
