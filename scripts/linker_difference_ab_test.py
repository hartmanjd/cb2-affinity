"""A/B test: does reporting the difference between chain lengths stop a statistics error in linker answers?

Why this test exists. Asked how binding changes with the length of a chain between an aromatic ring and a
phenyl, the assistant called the 0-carbon vs 1-carbon difference "not a real effect" because the two mean
intervals overlapped and the difference was below the 0.54 pKi single-measurement noise floor. Both are
statistical mistakes: overlapping intervals do not mean equal averages, and averages of hundreds of molecules
are far more precise than one measurement. The difference is +0.18 pKi with a 95% interval of +0.04 to +0.31.

    A  find_linkers as it was: per-length means and intervals only
    B  find_linkers now: also each length's difference from the previous one, with its own interval, and a
       note on how to read it

Both variants are recreated here, in this process only. Each answers the same question several times.

    python scripts/linker_difference_ab_test.py --variant A --repeats 3 --out ab_linker_A.csv
"""
from pathlib import Path
import argparse
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_assistant as assistant  # noqa: E402

QUESTION = ("How does binding change as the carbon chain between an aromatic ring and a phenyl group gets longer? "
            "How much should I trust that trend?")

ORIGINAL_GUIDANCE = (
    "linker_atoms is the number of sp3 carbons between the core and the terminal group; 0 means directly attached. "
    "n_genuine_linker counts molecules where those carbons are outside every ring (a real flexible chain) and is "
    "what the statistics describe. n_ring_path_only counts molecules matched only through atoms inside a fused ring "
    "system: those are NOT linkers of that length and are excluded. A length with few genuine molecules, or "
    "n_papers of 1-2, cannot support a claim about linker-length SAR.")


def use_original_tool():
    """Variant A: the same tool output without the difference fields, and with the original guidance."""
    current = assistant.TOOL_FUNCTIONS["find_linkers"]

    def original(data, *arguments, **options):
        result = current(data, *arguments, **options)
        for row in result["lengths"]:
            row.pop("mean_difference_vs_previous_length", None)
            row.pop("mean_difference_vs_previous_length_ci95", None)
        result["how_to_read"] = ORIGINAL_GUIDANCE
        return result

    assistant.TOOL_FUNCTIONS["find_linkers"] = original


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    if arguments.variant == "A":
        use_original_tool()
    data, client = assistant.ResearchData(ROOT), assistant.make_client()
    rows = []
    for repeat in range(1, arguments.repeats + 1):
        usage, tools = {}, []
        messages = assistant.new_conversation() + [{"role": "user", "content": QUESTION}]
        started = time.time()
        answer = assistant.chat_turn(client, data, messages, usage=usage, on_tool=lambda name, *rest: tools.append(name))
        rows.append({"variant": arguments.variant, "repeat": repeat, "question": QUESTION, "answer": answer,
                     "seconds": round(time.time() - started, 1), "tools": ", ".join(tools),
                     "cost_usd": usage.get("cost_usd", 0.0)})
        print(f"{arguments.variant} pass {repeat}: {rows[-1]['seconds']} s, tools {rows[-1]['tools']}", flush=True)
    pd.DataFrame(rows).to_csv(arguments.out, index=False)


if __name__ == "__main__":
    main()
