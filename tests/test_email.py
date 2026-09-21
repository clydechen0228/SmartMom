"""Email cleaning: a disclaimer footer must not delete the sender's actual request.

Regression tests for `laya.email.clean_email_body`. `_DISCLAIMER` used to be applied to whole
paragraphs, so any paragraph that merely *mentioned* boilerplate was deleted outright. When the
footer ran on without a blank line, the request went with it:

    clean_email_body("My account is locked.\\nThis email is confidential...\\nPlease unlock it.")
    # before: ''      <- the whole body, request included, was deleted
    # after:  'My account is locked. Please unlock it.'

Dropping the request is silent and severe; leaving one boilerplate line behind is neither, so the
cleaning errs towards keeping text.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.email import clean_email_body, email_state  # noqa: E402

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


DISCLAIMER = "This email is confidential and intended solely for the named addressee."

# --------------------------------------------------------------- the request survives the footer
check(
    "inline footer/no blank line keeps the request",
    clean_email_body("My account is locked.\n%s\nPlease unlock it." % DISCLAIMER),
    "My account is locked. Please unlock it.",
)
check(
    "inline footer/no terminal punctuation still recovers the request",
    clean_email_body("My account is locked\n%s\nPlease unlock it." % DISCLAIMER),
    "Please unlock it.",
)
check(
    "inline footer/body is never emptied",
    clean_email_body("My account is locked. %s" % DISCLAIMER),
    "My account is locked.",
)
check_true(
    "inline footer/is not empty",
    clean_email_body("My account is locked. %s" % DISCLAIMER).strip() != "",
)
check(
    "email_state/body keeps the request",
    email_state("Locked out", "My account is locked. %s" % DISCLAIMER)["body"],
    "My account is locked.",
)

# --------------------------------------------------------------- a pure footer is still removed
check(
    "standalone footer paragraph is still dropped",
    clean_email_body("My account is locked.\n\n%s" % DISCLAIMER),
    "My account is locked.",
)
check(
    "wrapped standalone footer is still dropped",
    clean_email_body(
        "My account is locked.\n\nThis email and any files transmitted with it are\n"
        "confidential and intended solely for the named addressee."
    ),
    "My account is locked.",
)
check(
    "received-in-error footer is still dropped",
    clean_email_body(
        "Please reopen ticket 4411.\n\nIf you have received this message in error, delete it."
    ),
    "Please reopen ticket 4411.",
)

# --------------------------------------------------------------- unrelated cleaning is unchanged
check(
    "quoted history is still removed",
    clean_email_body("Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text"),
    "Thanks for the update.",
)
check(
    "signature block is still removed",
    clean_email_body("Hi team,\nCan you confirm the refund?\nRegards,\nAlice"),
    "Hi team,\nCan you confirm the refund?",
)
check("empty body stays empty", clean_email_body(""), "")

# --------------------------------------------------------------- an opener is not a sign-off
# A short line that merely starts with a closing word used to be taken for the sign-off, and
# everything below it -- the request -- was cut away.
for name, body in [
    ("thanks for", "Hi,\nThanks for getting back to me.\nThe refund still has not arrived.\nCan you check?"),
    ("thank you for", "Hello,\nThank you for the invoice.\nWe were charged twice for order 5521."),
    ("best time", "Hi support,\nBest time to reach me is after 5pm.\nOur production database is down."),
    ("cheers for", "Hey,\nCheers for the quick fix yesterday.\nIt broke again this morning."),
    ("kind of", "Hi,\nKind of urgent, the export job fails.\nLogs are attached."),
    ("thanks. sentence", "Hi,\nThanks. I tried that already.\nThe error is still there."),
    ("word prefix", "Hallo,\nBestellung 4471 ist nicht angekommen.\nBitte prüfen."),
]:
    check("opener/%s keeps the request" % name, clean_email_body(body), body)

check(
    "opener/real sign-off below it is still cut",
    clean_email_body("Hello,\nThank you for the invoice.\nWe were charged twice.\n\nBest regards,\nDana"),
    "Hello,\nThank you for the invoice.\nWe were charged twice.",
)

# --------------------------------------------------------------- real sign-offs are still cut
for closing in [
    "Thanks", "Thanks,", "Thanks!", "Thanks again!", "Thanks in advance,", "Thanks and regards,",
    "Many thanks,", "Thank you,", "Thank you very much.", "Best,", "Best regards,", "Best wishes,",
    "Kind regards,", "Warm regards,", "Warmest regards,", "Regards,", "Cheers,", "Sincerely,",
    "Thanks, Alice", "Best regards, Dr. Jane Doe", "Regards, Łukasz", "Kind  regards,",
]:
    check(
        "sign-off/%r is cut" % closing,
        clean_email_body("Hi team,\nCan you confirm the refund?\n%s\nAlice" % closing),
        "Hi team,\nCan you confirm the refund?",
    )


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
