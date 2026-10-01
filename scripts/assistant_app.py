"""Streamlit chat interface for the CB2 research assistant.

Run from the project root:

    streamlit run scripts/assistant_app.py

The tools, the system prompt and the DeepSeek chat loop live in research_assistant.py;
this file only draws the page. Every tool call is shown in an expandable box so a
researcher can check exactly what was computed behind each answer.

As a public demo (ASSISTANT_PUBLIC_DEMO=1 with the owner's DEEPSEEK_API_KEY, both set as
hosting secrets), each visitor gets a small free DeepSeek credit shown as a balance that
goes down with each answer; see free_credit.py. Run locally, nothing is limited.
"""
from pathlib import Path
import json
import os
import sys
import time
import uuid

import streamlit as st


def secrets_to_environment():
    """Expose hosting secrets (Streamlit's secrets.toml) as environment variables.

    research_assistant.py reads its settings and the API key from the environment, as it does
    locally from .env. Values already in the environment win. Locally there is usually no
    secrets file, which Streamlit reports as an error, so that case is simply skipped.
    """
    try:
        secrets = dict(st.secrets)
    except Exception:
        return
    for name, value in secrets.items():
        if isinstance(value, (str, int, float, bool)) and name not in os.environ:
            os.environ[name] = str(value)


secrets_to_environment()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_assistant as assistant  # noqa: E402
import free_credit  # noqa: E402
import home_page  # noqa: E402

PUBLIC_DEMO = os.environ.get("ASSISTANT_PUBLIC_DEMO", "").lower() in ("1", "true", "yes")

EXAMPLE_QUESTIONS = [
    "Give me an overview of this dataset. What can it tell me, and what can't it?",
    "Which scaffolds have the most compounds, and how potent is each series?",
    "Do compounds containing a morpholine bind CB2 more strongly? Is that a fair comparison?",
    "Show me the five largest activity cliffs and suggest what structural change drives each.",
    "Predict the CB2 pKi of delta-9-THC and tell me how much to trust the number.",
    "Plot pKi against cLogP and highlight the indoles. Is lipophilicity driving potency here?",
    "Which model predicts best, is its lead real, and how close does it get to the measurement-noise floor?",
    "How does affinity change as the chain between an aromatic core and a phenyl gets longer?",
    "Why does the project use a scaffold split as well as a random one, and how much harder is it?",
]

st.set_page_config(page_title="Affinity, Audited", page_icon="🧪", layout="wide")
# Small print (captions) much darker than Streamlit's default grey so it reads easily: 90% black on
# the light theme (and 90% white on the dark one), still a shade softer than the main text.
_caption_ink = "#e6e6e6" if getattr(st.context.theme, "type", "light") == "dark" else "#1a1a1a"
st.html(f"<style>[data-testid='stCaptionContainer'], [data-testid='stCaptionContainer'] p "
        f"{{color: {_caption_ink} !important;}}</style>")


@st.cache_resource(show_spinner="Loading the dataset, fingerprints and SVR model (about a minute on a fresh server)...")
def load_data():
    # Loaded once per server process and shared by every browser tab. A fresh server (such as
    # Streamlit Community Cloud) starts without the fitted models, which are kept outside git;
    # they are downloaded from the project's GitHub release and accepted only if their
    # checksums match the run manifests.
    from fetch_models import download_models, problem_models
    if problem_models():
        download_models()
    return assistant.ResearchData()


@st.cache_resource
def credit_ledger():
    # One ledger for the whole server, shared by every visitor's tab.
    return free_credit.CreditLedger()


def start_conversation():
    st.session_state.messages = assistant.new_conversation()   # What the LLM sees.
    st.session_state.display = []                              # What the page shows.
    st.session_state.usage = {}


def show_tool(call):
    """One tool call: a collapsed box with inputs and raw result, plus any image."""
    with st.expander(f"🔧 {call['name']}", expanded=False):
        st.caption("Inputs")
        st.code(json.dumps(json.loads(call["arguments"] or "{}"), indent=2), language="json")
        st.caption("Result sent to the model")
        st.code(call["result"][:6000], language="json")
    for png in call["images"]:
        st.image(png)


def show_entry(entry):
    with st.chat_message(entry["role"]):
        for call in entry.get("tools", []):
            show_tool(call)
        if entry.get("text"):
            st.markdown(entry["text"])
        if entry.get("cost"):
            st.caption(entry["cost"])


if "messages" not in st.session_state:
    start_conversation()
if "visitor" not in st.session_state:
    # Worked out once when the visitor arrives and kept for the session, so their credit cannot
    # reset mid-conversation. The random tab id is used only when no address is available.
    st.session_state.visitor = free_credit.visitor_key(st.context.headers, st.context.ip_address, uuid.uuid4().hex[:12])
data = load_data()
ledger = credit_ledger()
visitor = st.session_state.visitor

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("CB2 research assistant")
    st.caption(f"{len(data.compounds):,} molecules · {data.acquisition.get('chembl_version', 'ChEMBL')} · "
               "human CB2 Ki · predictions from the tuned SVR · reads the project's docs and notebooks")
    if not assistant.is_hosted():
        st.info(f"Local model at {assistant.BASE_URL}", icon="💻")
        typed_key = None
    elif assistant.has_api_key() and PUBLIC_DEMO:
        # The project's key pays, within the free credit, unless the visitor brings their own key.
        # Kept deliberately plain: what each answer cost is revealed under the answer itself.
        # Three decimals, because a typical answer costs about $0.004 and would not move a
        # two-decimal balance at all.
        remaining = ledger.remaining(visitor)
        st.progress(remaining / ledger.per_visitor, text=f"Balance: ${remaining:.3f}")
        typed_key = st.text_input("Your own DeepSeek API key (optional)", type="password",
                                  help="Paste one to keep asking after the free credit runs out. It is used for "
                                       "this browser tab only and never stored.") or None
        if typed_key:
            st.success("Using your key: no limit, billed to your DeepSeek account.", icon="🔑")
    elif assistant.has_api_key():
        st.success("DeepSeek API key found", icon="🔑")
        typed_key = None
    else:
        typed_key = st.text_input("DeepSeek API key", type="password",
                                  help="Used for this browser session only. To keep it, add DEEPSEEK_API_KEY=... "
                                       "to a .env file in the project root (git ignores it).")
    if assistant.is_hosted():
        default = assistant.AVAILABLE_MODELS.index(assistant.DEFAULT_MODEL) if assistant.DEFAULT_MODEL in assistant.AVAILABLE_MODELS else 0
        model = st.selectbox("Model", assistant.AVAILABLE_MODELS, index=default,
                             help="deepseek-v4-pro handles multi-step tool use better; deepseek-flash is about 4x cheaper.")
    else:
        # A local server has its own model names, so the name is typed rather than chosen from a list.
        model = st.text_input("Model", value=assistant.DEFAULT_MODEL, help="The name your local server uses.")
    thinking = st.toggle("Thinking mode", value=assistant.DEFAULT_THINKING,
                         help="The model plans before acting. Measured on the evaluation set it costs about a second "
                              "per question and uses fewer tool calls, so it is on by default.")
    usage = st.session_state.usage
    st.metric("Approximate cost this conversation", f"${usage.get('cost_usd', 0):.4f}",
              help="Peak-hour DeepSeek prices; off-peak is half.")
    st.caption(f"{usage.get('input_tokens', 0):,} input · {usage.get('output_tokens', 0):,} output tokens")
    if st.button("New conversation", width="stretch"):
        start_conversation()
        st.rerun()
    st.divider()
    st.caption("Try asking")
    clicked = None
    for question in EXAMPLE_QUESTIONS:
        if st.button(question, width="stretch"):
            clicked = question

# ---------------------------------------------------------------- Home page and conversation
def draw_png(smiles, legend):
    # The same drawing tool the assistant uses; it returns (text for the model, [PNG images]).
    return assistant.run_tool(data, "draw_molecules", {"smiles_list": [smiles], "legends": [legend]})[1][0]


st.title("Affinity, Audited")
st.markdown("#### Every molecule traced, every claim tested. What it takes to trust an AI smarter than you.")
if not st.session_state.display:
    # First visit or a new conversation: the tour. Once a question is asked it folds away below.
    home_page.render(st, data, assistant.ROOT, draw_png)
    st.divider()
else:
    with st.expander("About this project"):
        home_page.render(st, data, assistant.ROOT, draw_png)
st.caption("Answers are built from tool calls on the curated data, RDKit and the saved SVR. Open any 🔧 box to "
           "see exactly what was computed. Statements marked as general knowledge come from the language model "
           "itself and may need to be verified.")
st.caption("Questions are sent to DeepSeek's servers, so data privacy is non-existent. If that is a concern, you "
           "can [run this app on your own computer with a local model]"
           "(https://github.com/hartmanjd/cb2-affinity/blob/dev/README.md#running-a-model-on-your-own-machine-instead), "
           "though answers may be slower and less accurate.")
for entry in st.session_state.display:
    show_entry(entry)

# A chat box pinned to the bottom of the page makes Streamlit keep the page scrolled to the bottom,
# which is right for a conversation but would skip past the home page on arrival. So on the home
# page the box sits inline at the end of the tour, and once a conversation starts it is pinned.
PROMPT = "Ask about compounds, scaffolds, SAR, predictions, papers..."
if st.session_state.display:
    question = st.chat_input(PROMPT)
else:
    with st.container():
        question = st.chat_input(PROMPT)
question = question or clicked
# Free credit applies only to the public demo's own key; a visitor's pasted key is never limited.
on_free_credit = PUBLIC_DEMO and assistant.is_hosted() and not typed_key
if question and on_free_credit:
    allowed, reason = ledger.can_ask(visitor)
    if not allowed:
        st.warning(reason, icon="💳")
        st.stop()
if question:
    show_entry({"role": "user", "text": question})
    st.session_state.display.append({"role": "user", "text": question})
    st.session_state.messages.append({"role": "user", "content": question})
    saved_length = len(st.session_state.messages)
    entry = {"role": "assistant", "text": "", "tools": []}
    with st.chat_message("assistant"):
        status = st.status("Working...", expanded=True)

        def on_tool(name, arguments, result, images):
            # Called by the chat loop after every tool, so progress appears live.
            call = {"name": name, "arguments": arguments, "result": result, "images": images}
            entry["tools"].append(call)
            with status:
                show_tool(call)

        usage = st.session_state.usage
        cost_before, tokens_before = usage.get("cost_usd", 0.0), usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        started = time.time()
        try:
            client = assistant.make_client(typed_key)
            entry["text"] = assistant.chat_turn(client, data, st.session_state.messages, model=model,
                                                thinking=thinking, on_tool=on_tool, usage=st.session_state.usage)
            status.update(label=f"Done · {len(entry['tools'])} tool call(s)", state="complete", expanded=False)
        except Exception as error:   # Missing key, network or API errors: show them and keep the chat usable.
            status.update(label="Failed", state="error")
            entry["text"] = f"⚠️ {type(error).__name__}: {error}"
            # Drop the half-finished turn so the next request starts from a valid conversation.
            del st.session_state.messages[saved_length - 1:]
        # What this one answer cost, shown under it; with free credit it is also deducted.
        cost = usage.get("cost_usd", 0.0) - cost_before
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0) - tokens_before
        if on_free_credit:
            ledger.charge(visitor, cost)
        if tokens:
            entry["cost"] = f"{tokens:,} tokens · ${cost:.4f} · {time.time() - started:.1f} s"
        st.markdown(entry["text"])
        if entry.get("cost"):
            st.caption(entry["cost"])
    st.session_state.display.append(entry)
    st.rerun()
