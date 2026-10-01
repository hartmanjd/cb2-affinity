"""Free DeepSeek credit for visitors to the hosted research assistant.

When the app runs on a public server with the owner's DeepSeek key, every visitor's questions
are paid for by that key. This module keeps the spending bounded, in three layers:

1. Each visitor gets FREE_CREDIT_USD (default $0.25, about 50 questions at the measured
   $0.004-0.005 per question). The app shows it as a balance that goes down with each answer.
2. All visitors together get at most DAILY_CAP_USD per day (default $3), whoever they are.
3. Outside the app: a small prepaid balance on the DeepSeek account, the true worst case.

Telling visitors apart is approximate. A public page has no login, so a visitor is recognised
by their network address and browser (see visitor_key). Someone determined can reset their
credit by switching network or browser; that is why layer 2 exists, and why it is layer 2 and
3, not layer 1, that actually bound the cost. Balances live in memory and reset when the app
restarts, which on Streamlit Community Cloud happens after 12 hours without visitors.

Visitors who paste their own DeepSeek key are not limited and are never charged here.
"""
from datetime import date
import hashlib
import os
import threading

FREE_CREDIT_USD = float(os.environ.get("FREE_CREDIT_USD", "0.25"))
DAILY_CAP_USD = float(os.environ.get("DAILY_FREE_CREDIT_CAP_USD", "3.00"))
# Measured on the evaluations: 55 chemistry questions cost $0.23 and 18 dataset questions $0.09.
TYPICAL_QUESTION_USD = 0.005
# A question is not started with less than this left, so one answer cannot run far past zero.
MINIMUM_TO_ASK_USD = 0.01


def visitor_key(headers, ip_address, fallback):
    """A short anonymous key for one visitor: a hash of their network address and browser.

    Behind a hosting proxy the visitor's own address is the first entry of X-Forwarded-For;
    otherwise Streamlit's ip_address is used. Nothing identifying is stored, only the hash.
    When no address is available (e.g. running locally), `fallback` (a per-tab id) is used.
    """
    headers = {str(k).lower(): str(v) for k, v in dict(headers or {}).items()}
    forwarded = headers.get("x-forwarded-for", "").split(",")[0].strip()
    address = forwarded or ip_address
    if not address:
        return f"tab-{fallback}"
    browser = headers.get("user-agent", "")
    return hashlib.sha256(f"{address}|{browser}".encode()).hexdigest()[:16]


class CreditLedger:
    """How much free credit each visitor and the whole app have spent. Safe to share between tabs."""

    def __init__(self, per_visitor=FREE_CREDIT_USD, daily_cap=DAILY_CAP_USD, today=date.today):
        self.per_visitor, self.daily_cap, self.today = per_visitor, daily_cap, today
        self.lock = threading.Lock()
        self.spent = {}                 # visitor key -> dollars spent from their free credit
        self.day, self.spent_today = today(), 0.0

    def _roll_over(self):
        # The daily cap starts again each calendar day; each visitor's own credit does not.
        if self.today() != self.day:
            self.day, self.spent_today = self.today(), 0.0

    def remaining(self, visitor):
        with self.lock:
            return max(0.0, self.per_visitor - self.spent.get(visitor, 0.0))

    def remaining_today(self):
        with self.lock:
            self._roll_over()
            return max(0.0, self.daily_cap - self.spent_today)

    def can_ask(self, visitor):
        """(allowed, reason shown to the visitor when not allowed)."""
        if self.remaining(visitor) < MINIMUM_TO_ASK_USD:
            return False, (f"You have used your ${self.per_visitor:.2f} of free credit. Paste your own DeepSeek "
                           "API key in the sidebar to keep asking.")
        if self.remaining_today() < MINIMUM_TO_ASK_USD:
            return False, ("Today's free credit for all visitors is used up. It resets tomorrow; meanwhile you can "
                           "paste your own DeepSeek API key in the sidebar to keep asking.")
        return True, ""

    def charge(self, visitor, dollars):
        """Record the cost of one answer against the visitor and the day."""
        with self.lock:
            self._roll_over()
            self.spent[visitor] = self.spent.get(visitor, 0.0) + dollars
            self.spent_today += dollars
