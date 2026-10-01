"""The research assistant's home page: a quick tour of the project for first-time visitors.

Every number on the page is computed here from the project's own files (the download and
selection manifests, the curated data, the saved model scores and the evaluation results), so
the page cannot drift from the data. facts() does the computing and is tested on its own;
render() only draws.
"""
from pathlib import Path
import json

import pandas as pd

# A nominal Olympic swimming pool: 50 m x 25 m x 2 m = 2,500 cubic metres.
OLYMPIC_POOL_LITRES = 2_500_000

# Chart colours from the data-visualisation guide's validated reference palette: one accent
# (blue) for the model the chart is about, its muted grey for the rest, and ink for each theme.
COLOURS = {"light": {"accent": "#2a78d6", "muted": "#898781", "ink": "#0b0b0b", "secondary": "#52514e"},
           "dark": {"accent": "#3987e5", "muted": "#898781", "ink": "#ffffff", "secondary": "#c3c2b7"}}


def facts(data, root):
    """Everything the home page shows, as plain numbers and text."""
    from provenance_support import current_selection
    import research_assistant as assistant
    root = Path(root)
    acquisition = data.acquisition
    selection = json.loads((root / current_selection(root)).read_text())
    stages = selection["stage_counts"]
    measurements = data.measurements
    retained = measurements[measurements["status"] == "retained"]
    compounds = data.compounds

    funnel = [
        ("Every CB2 activity record in ChEMBL", acquisition["activity_records"]),
        ("Human binding Ki values", stages["Human binding Ki before cleaning"]),
        ("Exact values, not just \"less than\"", stages["Exact Ki measurements"]),
        ("After validity and duplicate checks", stages["Unique activity records"]),
        ("After removing lab disagreements and mixtures", len(retained)),
        ("Unique molecules (repeats combined)", len(compounds)),
    ]

    # The strongest binder, and how little of it a swimming pool would need.
    best = compounds.loc[compounds["pki"].idxmax()]
    ki_molar = 10 ** -best["pki"]
    pool_mg = ki_molar * OLYMPIC_POOL_LITRES * best["mol_weight"] * 1000

    models = pd.DataFrame(json.loads(assistant.run_tool(data, "compare_models", {"include_dummies": True})[0])["models"])
    scaffold = models[models["split_strategy"] == "scaffold"]
    # One bar per model family: its better variant. The dummy ("guess the average") is a reference line.
    real = scaffold[~scaffold["model"].str.startswith("Dummy")]
    families = real.loc[real.groupby("model")["rmse_pki"].idxmin()].sort_values("rmse_pki")
    dummy_rmse = scaffold[scaffold["model"].str.startswith("Dummy")]["rmse_pki"].min()

    chemistry = pd.read_csv(root / "results/chemistry_evaluation.csv")
    dataset = pd.read_csv(root / "results/assistant_evaluation.csv")

    return {
        "molecules": len(compounds), "measurements": len(retained),
        "papers": int(retained["document_chembl_id"].nunique()),
        "first_year": int(retained["year"].min()), "last_year": int(retained["year"].max()),
        "chembl_version": acquisition["chembl_version"].replace("_", " "),
        "retrieved": acquisition["retrieved_at_utc"][:10],
        "funnel": funnel,
        "disagreements": int((measurements["status"] == "measurement_conflict").sum()),
        "mixtures": int((measurements["status"] == "structure_review").sum()),
        "limits_dropped": selection["quarantine_sheets"]["Ki limits"],
        "strongest": {"chembl_id": best["chembl_ids"].split(";")[0], "pki": float(best["pki"]),
                      "ki_pm": ki_molar * 1e12, "smiles": best["smiles"], "pool_mg": pool_mg},
        "top_journal": retained["journal"].value_counts().index[0],
        "top_journal_share": float(retained["journal"].value_counts(normalize=True).iloc[0]),
        "series": int(compounds["scaffold"].nunique()),
        "pki_5th": float(compounds["pki"].quantile(0.05)), "pki_95th": float(compounds["pki"].quantile(0.95)),
        "chembl_target_url": f"https://www.ebi.ac.uk/chembl/explore/target/{acquisition['target_chembl_id']}",
        "models": families[["model", "variant", "rmse_pki"]].to_dict("records"),
        "noise_floor": assistant.NOISE_FLOOR_PKI, "dummy_rmse": float(dummy_rmse),
        "chemistry_score": (int(chemistry["correct"].sum()), len(chemistry)),
        "dataset_score": (int(dataset["correct"].sum()), len(dataset)),
    }


def funnel_chart(funnel, colours):
    import altair as alt
    frame = pd.DataFrame(funnel, columns=["stage", "count"])
    frame["order"] = range(len(frame))
    frame["label"] = frame["count"].map("{:,}".format)
    base = alt.Chart(frame).encode(
        y=alt.Y("stage:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelLimit=320, labelPadding=8)),
        x=alt.X("count:Q", title=None, axis=None, scale=alt.Scale(domainMax=frame["count"].max() * 1.15)))
    bars = base.mark_bar(size=20, color=colours["accent"], cornerRadiusEnd=4).encode(
        tooltip=[alt.Tooltip("stage:N", title="Stage"), alt.Tooltip("count:Q", title="Count", format=",")])
    labels = base.mark_text(align="left", dx=6, color=colours["secondary"]).encode(text="label:N")
    return (bars + labels).properties(height=len(frame) * 34).configure_view(strokeWidth=0)


def model_chart(models, noise_floor, dummy_rmse, colours):
    import altair as alt
    frame = pd.DataFrame(models)
    frame["best"] = frame["rmse_pki"] == frame["rmse_pki"].min()
    frame["label"] = frame["rmse_pki"].map("{:.2f}".format)
    top = max(dummy_rmse, frame["rmse_pki"].max()) * 1.12
    base = alt.Chart(frame).encode(
        y=alt.Y("model:N", sort=alt.SortField("rmse_pki"), title=None, axis=alt.Axis(labelLimit=260, labelPadding=8)),
        x=alt.X("rmse_pki:Q", title="Typical prediction error (RMSE, pKi): lower is better",
                scale=alt.Scale(domain=[0, top]), axis=alt.Axis(grid=False, tickCount=5)))
    # Emphasis: the best model in the accent colour, the rest in muted grey.
    bars = base.mark_bar(size=20, cornerRadiusEnd=4).encode(
        color=alt.condition("datum.best", alt.value(colours["accent"]), alt.value(colours["muted"])),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("variant:N", title="Variant"),
                 alt.Tooltip("rmse_pki:Q", title="RMSE (pKi)", format=".3f")])
    labels = base.mark_text(align="left", dx=6, color=colours["secondary"]).encode(text="label:N")
    lines = pd.DataFrame([{"x": noise_floor, "text": f"lab noise floor {noise_floor:.2f}"},
                          {"x": dummy_rmse, "text": f"just guess the average {dummy_rmse:.2f}"}])
    rules = alt.Chart(lines).mark_rule(strokeDash=[4, 4], color=colours["secondary"]).encode(x="x:Q")
    rule_labels = alt.Chart(lines).mark_text(align="left", dx=4, dy=-6, baseline="bottom", fontSize=11,
                                             color=colours["secondary"]).encode(x="x:Q", y=alt.value(0), text="text:N")
    return (bars + labels + rules + rule_labels).properties(height=len(frame) * 38 + 24).configure_view(strokeWidth=0)


_FACTS = {}   # facts() for each loaded dataset, computed once per server rather than on every click.


def render(st, data, root, draw_png):
    """Draw the home page. `draw_png(smiles, legend)` returns a PNG of one molecule."""
    if id(data) not in _FACTS:
        _FACTS[id(data)] = facts(data, root)
    f = _FACTS[id(data)]
    theme = "dark" if getattr(st.context.theme, "type", "light") == "dark" else "light"
    colours = COLOURS[theme]

    st.markdown(
        "AI is improving faster than almost anyone predicted. Models that stumbled over arithmetic a few years ago "
        "now plan multi-step analyses, write their own code and reason about chemistry. The open question is no "
        "longer whether they are capable, but when we can trust them. An old rule may be the answer: **trust, but "
        "verify**. Let the AI do the thinking, but make every number it reports traceable and every claim testable. "
        "Built that way, a tool like this lets someone like me take on research well beyond my own training, "
        "because nothing has to be taken on faith. This project is a study of that idea, using how tightly "
        "molecules bind to the cannabinoid receptor CB2 as the test case.")
    st.markdown(
        "**CB2** is one of the body's two main cannabinoid receptors, the same family that THC acts on. Unlike "
        "CB1 in the brain, CB2 sits mostly on immune cells, which makes it a target for treating pain and "
        "inflammation without the high. This project asks: **can a computer predict how tightly a new "
        "molecule will bind to CB2, just from its structure?** And it comes with an AI assistant that answers "
        "your questions about the data, showing its working.")

    columns = st.columns(5)
    for column, (label, value) in zip(columns, [
            ("Molecules", f"{f['molecules']:,}"), ("Measurements", f"{f['measurements']:,}"),
            ("Research papers", f"{f['papers']:,}"), ("Years", f"{f['first_year']}–{f['last_year']}"),
            ("Chemical families", f"{f['series']:,}")]):
        column.metric(label, value)
    st.caption(f"Human CB2 binding data from [{f['chembl_version']}]({f['chembl_target_url']}), a free public database of bioactive molecules "
               f"(downloaded {f['retrieved']}). Binding strength is measured as **Ki**, the concentration at which a "
               "molecule fills half the receptors; the project uses **pKi**, where every +1 means ten times tighter.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader(f"From {f['funnel'][0][1]:,} records to a clean dataset")
        st.altair_chart(funnel_chart(f["funnel"], colours), width="stretch")
        st.caption(f"{f['limits_dropped']:,} values were only limits (such as \"Ki > 10 µM\"), not measurements. "
                   f"{f['disagreements']:,} were dropped because labs measuring the same molecule disagreed by "
                   f"ten times or more, and {f['mixtures']} because the sample was a mixture.")
    with right:
        strongest = f["strongest"]
        st.subheader("The strongest binder")
        st.image(draw_png(strongest["smiles"], f"{strongest['chembl_id']}  ·  pKi {strongest['pki']:.2f}"))
        st.markdown(f"Ki of about **{strongest['ki_pm']:.0f} picomolar**. About **{strongest['pool_mg']:.0f} mg** "
                    "of it, roughly a grain of rice, dissolved in an Olympic swimming pool would still reach that "
                    "concentration.")

    st.subheader("Can a model beat the lab?")
    st.altair_chart(model_chart(f["models"], f["noise_floor"], f["dummy_rmse"], colours), width="stretch")
    best = f["models"][0]
    st.caption(f"Each model was tested on chemical families it never saw in training, the honest test of predicting "
               f"new chemistry. The winner, a tuned **{best['model'].lower()}**, misses by about {best['rmse_pki']:.2f} "
               f"pKi on a typical molecule. The same molecule measured in two different labs already differs by about "
               f"{f['noise_floor']:.2f}, so no model can be expected to do much better. For scale, 90% of the "
               f"molecules fall between pKi {f['pki_5th']:.1f} and {f['pki_95th']:.1f}, a "
               f"{10 ** (f['pki_95th'] - f['pki_5th']):,.0f}-fold span in binding strength; a miss of "
               f"{best['rmse_pki']:.2f} means a predicted Ki is typically within about "
               f"{10 ** best['rmse_pki']:.0f}-fold of the measured one. A fine-tuned chemistry transformer "
               "(ChemBERTa) did not beat the simpler model.")

    chemistry, dataset = f["chemistry_score"], f["dataset_score"]
    st.subheader("How the assistant works")
    st.markdown(
        "Most people still would not trust a chatbot with hard numbers, and for good reason: language models can "
        "state a wrong figure as confidently as a right one. So this assistant is built so that the language model "
        "(DeepSeek) **never produces a number itself**. It reads your question, decides what to look up and "
        "explains the results. Everything it reports comes from ordinary, checkable code: pandas for the data, "
        "RDKit for chemistry, the tuned model for predictions. Very little of any answer is actually generated; "
        "the language model writes the sentences around numbers it was handed.")
    st.markdown(
        "The guard rails behind that:\n"
        "- **Every number comes from a tool, and you can see it.** Open any 🔧 box under an answer for the exact "
        "inputs and raw result.\n"
        "- **Every claim is labelled** as measured (from the data), predicted (from the model) or general "
        "knowledge (from the language model, worth checking).\n"
        "- **Every number carries its uncertainty**: confidence intervals, prediction intervals and the range lab "
        "measurements themselves vary over.\n"
        "- **The model is never asked to read a structure.** Formulas, ring names and stereochemistry come from "
        "RDKit, because testing showed language models guessing them confidently and wrongly.\n"
        "- **Held-back test molecules are never predicted**, so the project's final test stays honest.\n"
        f"- **It is tested against answers it could not have guessed**: {chemistry[0]}/{chemistry[1]} chemistry "
        f"questions and {dataset[0]}/{dataset[1]} questions about the data correct, and every change to the "
        "assistant was measured before it was kept.")
    st.markdown(
        "Language models still make mistakes. The point is not that this one cannot, but that when it does, you "
        "can see exactly where.")
    st.markdown("**Try one of the questions in the sidebar, or ask your own below.** The full project, notebooks "
                "and data are on [GitHub](https://github.com/hartmanjd/cb2-affinity).")
