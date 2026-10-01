"""The research assistant's home page: a quick tour of the project for first-time visitors.

Every number on the page is computed here from the project's own files (the download and
selection manifests, the curated data, the saved model scores and the evaluation results), so
the page cannot drift from the data. facts() does the computing and is tested on its own;
render() only draws.
"""
from pathlib import Path
import json

import pandas as pd

# Chart colours from the data-visualisation guide's validated reference palette: one accent
# (blue) for the model the chart is about, its muted grey for the rest, and ink for each theme.
COLOURS = {"light": {"page": "#ffffff", "accent": "#2a78d6", "outline": "#184f95", "muted": "#898781", "ink": "#0b0b0b", "secondary": "#52514e"},
           "dark": {"page": "#0e1117", "accent": "#3987e5", "outline": "#86b6ef", "muted": "#898781", "ink": "#ffffff", "secondary": "#c3c2b7"}}


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

    quarantine = selection["quarantine_sheets"]
    conflicts = int((measurements["status"] == "measurement_conflict").sum())
    mixtures = int((measurements["status"] == "structure_review").sum())
    # Each step of the clean-up: what is left, and what that step took away.
    funnel = [
        ("Every CB2 activity record in ChEMBL", acquisition["activity_records"], ""),
        ("Human binding Ki values", stages["Human binding Ki before cleaning"],
         "other kinds of result, such as IC50 or functional assays, or not clearly human CB2"),
        ("Exact values", stages["Exact Ki measurements"],
         f"only limits such as \"Ki > 10 µM\" ({quarantine['Ki limits']:,}), or missing or invalid values"),
        ("Passed validity and duplicate checks", stages["Unique activity records"],
         f"flagged as possible duplicates ({quarantine['Potential duplicates']:,}) or invalid by ChEMBL"),
        ("Labs agree, single compounds", len(retained),
         f"labs disagreed tenfold or more ({conflicts}), or the sample was a mixture ({mixtures})"),
        ("Molecules, one value each", len(compounds), "repeat measurements of the same molecule, merged"),
    ]
    for (_, before, _), (_, after, _) in zip(funnel, funnel[1:]):
        assert after <= before, "each clean-up step can only remove records"

    def binder(row):
        # One molecule and its five nearest neighbours in the dataset, by fingerprint (Tanimoto) similarity,
        # from the same find_similar tool the assistant uses. The molecule itself (similarity 1) is dropped.
        found = json.loads(assistant.run_tool(data, "find_similar", {"smiles": row["smiles"], "top_k": 6})[0])
        chembl_id = row["chembl_ids"].split(";")[0]
        neighbours = [n for n in found["neighbours"] if n["chembl_ids"] != row["chembl_ids"]][:5]
        return {"chembl_id": chembl_id, "name": row["name"] if isinstance(row["name"], str) else None,
                "pki": float(row["pki"]), "smiles": row["smiles"],
                "neighbours": [{"label": n["name"] or n["chembl_ids"].split(";")[0], "similarity": n["tanimoto"],
                                "pki": n["pki"], "chembl_id": n["chembl_ids"].split(";")[0], "name": n["name"],
                                "smiles": n["smiles"]} for n in neighbours]}

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
        "disagreements": conflicts, "mixtures": mixtures,
        "limits_dropped": selection["quarantine_sheets"]["Ki limits"],
        "strongest": binder(compounds.loc[compounds["pki"].idxmax()]),
        "weakest": binder(compounds.loc[compounds["pki"].idxmin()]),
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


# Six steps of one blue, light (everything) to dark (what survives), from the reference palette's
# sequential ramp; no lighter than step 250, so even the outer ring stands out from the page.
RING_BLUES = ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]


def rings_chart(funnel, colours, size=340):
    """The clean-up as nested circles: each circle's area is the records left after that step.

    Drawn largest first, so every smaller circle sits inside the one before it and the visible ring
    between two circles is what that step removed. A thin page-coloured outline separates the rings.
    """
    import altair as alt
    largest = funnel[0][1]
    full_area = 3.14159 * (size / 2 - 4) ** 2   # Circle marks are sized by area in square pixels.
    frame = pd.DataFrame([{"order": i, "stage": stage, "count": count, "area": full_area * count / largest,
                           "removed": (funnel[i - 1][1] - count) if i else 0, "colour": RING_BLUES[i]}
                          for i, (stage, count, _) in enumerate(funnel)])
    return alt.Chart(frame).mark_circle(opacity=1, stroke=colours["page"], strokeWidth=2).encode(
        x=alt.value(size / 2), y=alt.value(size / 2),
        size=alt.Size("area:Q", scale=None, legend=None),
        color=alt.Color("colour:N", scale=None, legend=None),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("stage:N", title="Step"), alt.Tooltip("count:Q", title="Left", format=","),
                 alt.Tooltip("removed:Q", title="Removed by this step", format=",")],
    ).properties(width=size, height=size).configure_view(strokeWidth=0)


def rings_legend(funnel, colours):
    """The key beside the rings: each step's colour, what is left, and what it removed."""
    rows = []
    for i, (stage, count, removed) in enumerate(funnel):
        taken = f"<div style='margin-left:22px;color:{colours['secondary']};font-size:0.85rem'>" \
                f"−{funnel[i - 1][1] - count:,}: {removed}</div>" if i else ""
        rows.append(f"<div style='margin:0 0 10px 0'><span style='display:inline-block;width:14px;height:14px;"
                    f"border-radius:50%;background:{RING_BLUES[i]};vertical-align:-2px;margin-right:8px'></span>"
                    f"<b>{count:,}</b> &nbsp;{stage}{taken}</div>")
    return "".join(rows)


def neighbour_chart(molecule, colours, gutter=190):
    """Its five nearest neighbours' pKi as bars, with the molecule's own pKi as a dashed line (a Vega-Lite spec).

    Bars and names can both be clicked: both carry the point selection "pick" on chembl_id, which
    Streamlit reports back so the page can show that neighbour (see picked_neighbour). Axis labels
    cannot be clicked, so the names are drawn as text in a left-hand gutter of `gutter` pixels
    instead, and the bars' scale starts after it.
    """
    import altair as alt
    frame = pd.DataFrame(molecule["neighbours"])
    frame["order"] = range(len(frame))
    frame["value"] = frame["pki"].map("{:.2f}".format) + "  ·  " + (frame["similarity"] * 100).round().astype(int).astype(str) + "% similar"
    pick = alt.selection_point(name="pick", fields=["chembl_id"], on="click")
    # A fixed 40 px per row with 40% of it left as a gap, so all five rows fit and bars never touch.
    y = alt.Y("chembl_id:N", sort=alt.SortField("order"), title=None, axis=None, scale=alt.Scale(paddingInner=0.4))
    scale = alt.Scale(domain=[0, 12], range=[gutter, {"expr": "width"}])   # Bars start after the name gutter.
    x = alt.X("pki:Q", title="pKi of its 5 nearest neighbours (click a name or bar to see it)", scale=scale,
              axis=alt.Axis(grid=False, values=[0, 2, 4, 6, 8, 10, 12]))
    # Thin darker outline on every bar; a clicked one gets a bold outline and the others fade.
    bars = alt.Chart(frame).mark_bar(color=colours["accent"], cornerRadiusEnd=4, cursor="pointer").encode(
        x=x, y=y, stroke=alt.value(colours["outline"]),
        strokeWidth=alt.condition(pick, alt.value(2.5), alt.value(1), empty=False),
        opacity=alt.condition(pick, alt.value(1.0), alt.value(0.35)),
        tooltip=[alt.Tooltip("label:N", title="Molecule"), alt.Tooltip("similarity:Q", title="Similarity", format=".0%"),
                 alt.Tooltip("pki:Q", title="pKi", format=".2f")]).add_params(pick)
    names = alt.Chart(frame).mark_text(align="right", baseline="middle", dx=-10, cursor="pointer",
                                       color=colours["accent"], fontWeight="bold").encode(
        x=alt.value(gutter), y=y, text="label:N",
        opacity=alt.condition(pick, alt.value(1.0), alt.value(0.45)))
    values = alt.Chart(frame).mark_text(align="left", baseline="middle", dx=5, color=colours["secondary"]).encode(
        x=x, y=y, text="value:N")
    own = pd.DataFrame([{"x": molecule["pki"], "text": f"itself {molecule['pki']:.2f}"}])
    rule = alt.Chart(own).mark_rule(strokeDash=[4, 4], color=colours["secondary"]).encode(x=alt.X("x:Q", scale=scale))
    rule_label = alt.Chart(own).mark_text(align="center", dy=-6, baseline="bottom", fontSize=11,
                                          color=colours["secondary"]).encode(x=alt.X("x:Q", scale=scale),
                                                                             y=alt.value(0), text="text:N")
    spec = (bars + names + values + rule + rule_label).properties(height=len(frame) * 40 + 10) \
        .configure_view(strokeWidth=0).to_dict()
    # Altair attaches a selection to one layer only; Vega-Lite allows several. Register the names layer as
    # a second view of "pick", so a click on a name selects that molecule just like a click on its bar.
    spec["layer"][1]["name"] = "neighbour_names"
    spec["params"][0]["views"].append("neighbour_names")
    return spec


def picked_neighbour(selection_state, molecule):
    """The neighbour a visitor clicked in neighbour_chart, or None (nothing clicked, or click cleared).

    `selection_state` is what Streamlit keeps for the chart: {"selection": {"pick": [{"chembl_id": ...}]}}.
    """
    try:
        points = selection_state["selection"]["pick"]
    except (KeyError, TypeError):
        return None
    chosen = {point.get("chembl_id") for point in points or []}
    return next((n for n in molecule["neighbours"] if n["chembl_id"] in chosen), None)


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


# The audit workbook (scripts/build_assistant_audit.py) on GitHub, as a direct download.
AUDIT_URL = "https://github.com/hartmanjd/cb2-affinity/raw/dev/results/assistant_audit.xlsx"

ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]


def frozen_sentence(models):
    """One sentence on ChemBERTa without fine-tuning, worded from where it actually placed."""
    names = [m["model"] for m in models]
    if "Frozen ChemBERTa + SVR" not in names or "Fine-tuned ChemBERTa" not in names:
        return ""
    frozen = models[names.index("Frozen ChemBERTa + SVR")]["rmse_pki"]
    fine_tuned = models[names.index("Fine-tuned ChemBERTa")]["rmse_pki"]
    place = ORDINALS[names.index("Frozen ChemBERTa + SVR")]
    if frozen < fine_tuned:
        return (f" The very same ChemBERTa left frozen, used only to describe each molecule for a simpler model to "
                f"learn from, did better ({frozen:.2f} against {fine_tuned:.2f}) and placed {place}.")
    return f" Left frozen, used only to describe molecules for a simpler model, it placed {place} ({frozen:.2f})."


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

    st.subheader(f"Whittling {f['funnel'][0][1]:,} records down to {f['funnel'][-1][1]:,} molecules")
    rings, key_column = st.columns([2, 3], gap="large", vertical_alignment="center")
    with rings:
        st.altair_chart(rings_chart(f["funnel"], colours), width="content")
    with key_column:
        st.html(rings_legend(f["funnel"], colours))

    # The two ends of the scale, side by side: each molecule, then its closest relatives in the data.
    for column, title, key in zip(st.columns(2, gap="large"), ["The strongest binder", "The weakest binder"],
                                  ["strongest", "weakest"]):
        molecule = f[key]
        name = f"{molecule['name']} ({molecule['chembl_id']})" if molecule["name"] else molecule["chembl_id"]
        # Clicking a neighbour's bar reruns the page with the click stored under the chart's key, so the
        # drawing above the chart can show that neighbour. "Back" swaps in a fresh chart (a new key),
        # which clears the click.
        resets = st.session_state.setdefault(f"{key}_resets", 0)
        chart_key = f"{key}_neighbours_{resets}"
        shown = picked_neighbour(st.session_state.get(chart_key), molecule)
        with column:
            st.subheader(title)
            if shown is None:
                st.image(draw_png(molecule["smiles"], f"{name}  ·  pKi {molecule['pki']:.2f}"))
            else:
                label = f"{shown['name']} ({shown['chembl_id']})" if shown["name"] else shown["chembl_id"]
                st.image(draw_png(shown["smiles"], f"{label}  ·  pKi {shown['pki']:.2f}"))
            # Always present (greyed out until a neighbour is shown), so the panel never changes height.
            st.button(f"Back to {molecule['chembl_id']}", key=f"{key}_back", disabled=shown is None,
                      on_click=lambda k=key: st.session_state.update({f"{k}_resets": st.session_state[f"{k}_resets"] + 1}))
            st.vega_lite_chart(neighbour_chart(molecule, colours), width="stretch", on_select="rerun", key=chart_key)

    st.subheader("Can a model support the lab?")
    st.altair_chart(model_chart(f["models"], f["noise_floor"], f["dummy_rmse"], colours), width="stretch")
    best = f["models"][0]
    st.caption(f"Each model was tested on chemical families it never saw in training, the honest test of predicting "
               f"new chemistry. The winner, a tuned **{best['model'].lower()}**, misses by about {best['rmse_pki']:.2f} "
               f"pKi on a typical molecule. The same molecule measured in two different labs already differs by about "
               f"{f['noise_floor']:.2f}, so no model can be expected to do much better. For scale, 90% of the "
               f"molecules fall between pKi {f['pki_5th']:.1f} and {f['pki_95th']:.1f}, a "
               f"{10 ** (f['pki_95th'] - f['pki_5th']):,.0f}-fold span in binding strength; a miss of "
               f"{best['rmse_pki']:.2f} means a predicted Ki is typically within about "
               f"{10 ** best['rmse_pki']:.0f}-fold of the measured one. "
               + (f"Interestingly, the most sophisticated model came last: ChemBERTa, a chemistry transformer "
                  f"pretrained on 77 million molecules and then fine-tuned on this project's data, was the least "
                  f"accurate on new chemical families. It learned its training molecules too well and carried less "
                  f"of that over to unfamiliar chemistry."
                  if f["models"][-1]["model"] == "Fine-tuned ChemBERTa" else
                  "A fine-tuned chemistry transformer (ChemBERTa), pretrained on 77 million molecules, did not beat "
                  "the simpler model.")
               + frozen_sentence(f["models"]))

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
        f"- **It is tested against answers it could not have guessed**: [{chemistry[0]}/{chemistry[1]} chemistry "
        f"questions and {dataset[0]}/{dataset[1]} questions about the data]({AUDIT_URL}) correct, and every change "
        "to the assistant was measured before it was kept. The linked workbook shows every question, how its true "
        "answer was computed, the answer given, and every condition held fixed.")
    st.markdown(
        "Language models still make mistakes. The point is not that this one cannot, but that when it does, you "
        "can see exactly where.")
    st.markdown("**Try one of the questions in the sidebar, or ask your own below.** The full project, notebooks "
                "and data are on [GitHub](https://github.com/hartmanjd/cb2-affinity).")
