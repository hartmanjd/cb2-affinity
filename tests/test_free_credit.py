"""Check the hosted demo's free DeepSeek credit, and the app page, without calling DeepSeek."""
from datetime import date
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import free_credit  # noqa: E402


class LedgerTests(unittest.TestCase):
    def test_each_visitor_has_their_own_credit(self):
        ledger = free_credit.CreditLedger(per_visitor=0.25, daily_cap=3.0)
        ledger.charge("ana", 0.10)
        self.assertAlmostEqual(ledger.remaining("ana"), 0.15)
        self.assertAlmostEqual(ledger.remaining("ben"), 0.25)
        self.assertTrue(ledger.can_ask("ana")[0])
        ledger.charge("ana", 0.145)   # Below the minimum needed to start another question.
        allowed, reason = ledger.can_ask("ana")
        self.assertFalse(allowed)
        self.assertIn("own DeepSeek", reason)
        self.assertAlmostEqual(ledger.remaining("ana"), 0.005)

    def test_the_daily_cap_stops_everyone_and_resets_next_day(self):
        days = [date(2026, 10, 1)]
        ledger = free_credit.CreditLedger(per_visitor=0.25, daily_cap=0.30, today=lambda: days[0])
        ledger.charge("ana", 0.20)
        ledger.charge("ben", 0.095)
        allowed, reason = ledger.can_ask("cara")   # Cara has spent nothing, but the day is used up.
        self.assertFalse(allowed)
        self.assertIn("Today's free credit", reason)
        days[0] = date(2026, 10, 2)
        self.assertTrue(ledger.can_ask("cara")[0])
        self.assertAlmostEqual(ledger.remaining("ana"), 0.05)   # A visitor's own credit does not reset.

    def test_visitors_are_recognised_by_address_and_browser_only_as_a_hash(self):
        firefox = {"X-Forwarded-For": "203.0.113.7, 10.0.0.1", "User-Agent": "Firefox"}
        key = free_credit.visitor_key(firefox, "10.0.0.1", "tab1")
        self.assertEqual(key, free_credit.visitor_key(firefox, "10.0.0.9", "tab2"))   # Same visitor, new tab.
        self.assertNotIn("203.0.113.7", key)
        self.assertNotEqual(key, free_credit.visitor_key({**firefox, "User-Agent": "Chrome"}, None, "tab1"))
        self.assertNotEqual(key, free_credit.visitor_key({"X-Forwarded-For": "198.51.100.2", "User-Agent": "Firefox"},
                                                         None, "tab1"))
        # No address at all (running locally): fall back to the browser tab.
        self.assertEqual(free_credit.visitor_key({}, None, "tab1"), "tab-tab1")


class AppPageTests(unittest.TestCase):
    """Drive the real Streamlit page with a stand-in for DeepSeek.

    The stand-ins are started in setUp and stopped by addCleanup, so they stay active for every
    question a test asks; nothing here can reach DeepSeek or spend money.
    """

    def start(self, environment, answer_cost):
        from streamlit.testing.v1 import AppTest
        import research_assistant as assistant

        def fake_chat_turn(client, data, messages, usage=None, **options):
            usage["cost_usd"] = usage.get("cost_usd", 0.0) + answer_cost
            usage["input_tokens"] = usage.get("input_tokens", 0) + 3000
            usage["output_tokens"] = usage.get("output_tokens", 0) + 400
            messages.append({"role": "assistant", "content": "An answer."})
            return "An answer."

        def no_network(*arguments, **options):
            raise AssertionError("A test tried to create a real DeepSeek client")

        for patcher in [patch.dict(os.environ, environment), patch.object(assistant, "chat_turn", fake_chat_turn),
                        patch.object(assistant, "make_client", lambda key=None: object()),
                        patch("openai.OpenAI", no_network)]:
            patcher.start()
            self.addCleanup(patcher.stop)
        # The credit ledger is cached for the whole server; start each test with an empty one.
        import streamlit
        streamlit.cache_resource.clear()   # Also reloads the dataset, which adds a few seconds per test.
        app = AppTest.from_file(str(ROOT / "scripts" / "assistant_app.py"), default_timeout=120)
        app.run()
        return app

    def meter(self, app):
        return app.sidebar.get("progress")[0].proto.text.replace("**", "")

    def test_public_demo_shows_a_balance_that_goes_down(self):
        app = self.start({"ASSISTANT_PUBLIC_DEMO": "1", "DEEPSEEK_API_KEY": "sk-test"}, answer_cost=0.004)
        self.assertEqual("Balance: $0.250", self.meter(app))
        app.chat_input[0].set_value("How many molecules?").run()
        self.assertEqual("Balance: $0.246", self.meter(app))
        self.assertTrue(any("3,400 tokens · $0.0040" in caption.value for caption in app.caption))

    def test_public_demo_stops_when_the_credit_is_used(self):
        app = self.start({"ASSISTANT_PUBLIC_DEMO": "1", "DEEPSEEK_API_KEY": "sk-test"}, answer_cost=0.245)
        app.chat_input[0].set_value("First question").run()
        app.chat_input[0].set_value("Second question").run()
        self.assertTrue(any("used your $0.25 of free credit" in warning.value for warning in app.warning))

    def test_a_visitors_own_key_is_not_limited(self):
        app = self.start({"ASSISTANT_PUBLIC_DEMO": "1", "DEEPSEEK_API_KEY": "sk-test"}, answer_cost=0.30)
        app.sidebar.text_input[0].set_value("sk-visitor").run()
        app.chat_input[0].set_value("First question").run()
        app.chat_input[0].set_value("Second question").run()
        self.assertEqual(len(app.warning), 0)
        self.assertEqual("Balance: $0.250", self.meter(app))   # Their own key: nothing deducted.

    def test_running_locally_is_not_limited(self):
        app = self.start({"ASSISTANT_PUBLIC_DEMO": "", "DEEPSEEK_API_KEY": "sk-test"}, answer_cost=0.30)
        self.assertEqual(len(app.sidebar.get("progress")), 0)   # No credit meter.
        app.chat_input[0].set_value("First question").run()
        app.chat_input[0].set_value("Second question").run()
        self.assertEqual(len(app.warning), 0)


if __name__ == "__main__":
    unittest.main()
