"""Routing only: which checkpoint would handle this state, and why.

No weights are loaded and no forward pass runs, so this is free and instant.

    python examples/01_routing.py
"""
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya  # noqa: E402
from laya import Router  # noqa: E402

from examples_common import QUESTIONS  # noqa: E402

STATES = {
    "english": {"subject": "Duplicate charge on invoice #4411",
                "body": "We were billed twice for March. Please refund today or we cancel."},
    "hindi": {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"},
    "chinese": {"body": "你们重复扣了我两次费用，请今天退款，否则我就取消订阅。"},
    "arabic": {"body": "لقد تم خصم المبلغ مرتين، يرجى استرداد المبلغ اليوم."},
    "french": {"body": "Nous avons ete factures deux fois en mars, merci de nous rembourser."},
}


def main():
    print("laya", laya.__version__)
    router = Router(preload=False)
    print("router:", router)

    t0 = time.time()
    for name, state in STATES.items():
        decision = router.route(state, QUESTIONS)
        print("%-9s -> %-14s | %s" % (name, decision.model, decision.reason))
    print("\n%d routing decisions in %.1f ms, zero weights loaded"
          % (len(STATES), (time.time() - t0) * 1000))


if __name__ == "__main__":
    main()
