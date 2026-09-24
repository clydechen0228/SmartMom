"""Real forward passes: the same typed questions across several languages.

Downloads the requested checkpoint from the Hub on first run.

    python examples/02_predict.py [model]        # default: multilingual
    LAYA_DEVICE=cpu python examples/02_predict.py
"""
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import Router  # noqa: E402

from examples_common import QUESTIONS, pick_device  # noqa: E402

STATES = {
    "EN": {"subject": "Duplicate charge on invoice #4411",
           "body": "Hi, we were billed twice for March. Please refund the duplicate "
                   "today or we will cancel our plan."},
    "ZH": {"body": "你们重复扣了我两次三月的费用，请今天退款，否则我就取消订阅。"},
    "HI": {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"},
    "TECH": {"body": "The API returns 502 on every request since the deploy. Production is down."},
}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "multilingual"
    device = pick_device()
    print("device: %s | model: %s" % (device, model))

    router = Router(preload=False, device=device)
    t0 = time.time()
    router.load(model)
    print("cold load: %.1f s | loaded=%s\n" % (time.time() - t0, router.loaded))

    for tag, state in STATES.items():
        t = time.time()
        res = router.predict(state, QUESTIONS, model=model)
        ms = (time.time() - t) * 1000
        a = res["answers"]
        print("--- %s  (%.0f ms total, %.1f ms/question) ---"
              % (tag, ms, ms / len(QUESTIONS)))
        # Each question type carries its answer under a different key.
        print("  department      : %s (%.2f)"
              % (a["department"]["choice"], a["department"]["confidence"]))
        print("  urgency         : %.2f  %s"
              % (a["urgency"]["score"], a["urgency"]["legend"]))
        print("  churn_risk      : %.2f" % a["churn_risk"]["noul"])
        print("  refund_requested: %.2f" % a["refund_requested"]["noul"])


if __name__ == "__main__":
    main()
