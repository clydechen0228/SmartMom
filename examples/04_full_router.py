"""All three checkpoints resident: automatic routing with no cold-load stalls.

Needs every checkpoint on disk (~4.7 GB total). Run 02_predict.py first if you
only want the multilingual one.

    python examples/04_full_router.py
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
    "FR": {"body": "Nous avons ete factures deux fois en mars, merci de nous rembourser aujourd'hui."},
}


def main():
    t0 = time.time()
    router = Router(preload=True, device=pick_device())
    print("preloaded %s in %.1f s\n" % (router.loaded, time.time() - t0))

    for tag, state in STATES.items():
        t = time.time()
        res = router.predict(state, QUESTIONS)
        ms = (time.time() - t) * 1000
        routing, a = res["routing"], res["answers"]
        print("--- %s -> %s  (%.0f ms) ---" % (tag, routing["model"], ms))
        print("    why: %s" % routing["reason"])
        print("    department=%s (%.2f)  urgency=%.2f  churn=%.2f  refund=%.2f"
              % (a["department"]["choice"], a["department"]["confidence"],
                 a["urgency"]["score"], a["churn_risk"]["noul"],
                 a["refund_requested"]["noul"]))


if __name__ == "__main__":
    main()
