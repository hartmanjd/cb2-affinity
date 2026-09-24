## CB2 Affinity Project Instructions

- This is a beginner-focused AI/ML capstone project.
- Prefer clear, explicit code over highly abstract or optimized code.

## Project goals

- Curate a reproducible human CB2 receptor affinity dataset.
- Preserve scientifically relevant assay and chemistry metadata.
- Build and compare machine-learning baselines.
- Evaluate models using chemically meaningful validation strategies.
- Investigate molecular activity cliffs where supported by the data.

## Notebook style

When creating or editing Jupyter notebooks:

- Add frequent explanatory `#` comments to code cells and blocks according to their logic.
- Comments should explain both WHAT the code is doing and WHY it is being done.
- Assume the reader has basic Python knowledge but limited experience with APIs,
  cheminformatics, machine learning, and scientific data processing.
- Explain unfamiliar API structures, loops, filters, validation checks, and safeguards.
- Keep useful safeguards and validation checks, but explain their purpose with comments.
- Prefer visible intermediate steps over hiding important logic behind abstractions.
- Avoid unnecessary functions when straightforward sequential notebook code is easier
  to understand.
- Do not remove scientific validation or reproducibility measures merely to shorten code.
- Keep each notebook focused on one stage of the project.

## Existing code

When asked to add explanations or comments to existing working code, do not change
the executable code unless explicitly requested. Add comments around the existing
implementation instead.

## Development principles

- Prefer clear, beginner-readable Python.
- Explain scientific and machine-learning decisions in comments.
- Avoid unnecessary abstraction.
- Keep notebooks readable and presentation-friendly.
- Do not change scientific assumptions without documenting the reason.
- Keep raw source data unchanged.

## Tables and printed summaries

When printing table-like output in notebooks, format the output so columns are clearly aligned and easy to scan.

- Do not rely on plain `|` separators without fixed column widths.
- Use formatted f-strings with explicit alignment and width specifiers.
- Left-align text labels and right-align numeric values.
- Include thousands separators for large counts where helpful.
- Keep column widths wide enough for the longest expected label.
- Prefer readable console tables over compact but misaligned output.
- Do not change the underlying analysis just to improve formatting.

Example:

```python
print(
    f"{'Stage':<30} | "
    f"{'Measurements':>12} | "
    f"{'Molecule IDs':>12} | "
    f"{'Assays':>8} | "
    f"{'Removed':>10}"
)

print(
    f"{name:<30} | "
    f"{len(rows):>12,} | "
    f"{molecule_count:>12,} | "
    f"{assay_count:>8,} | "
    f"{removed:>10,}"
)
```

## Rough-draft reference

A second folder may be attached to the Codex project containing the
earlier rough-draft notebooks.

Treat that folder as reference material.

Do not modify files in the rough-draft folder unless explicitly asked.

## Modeling

- Keep one accepted, complete run per notebook. Replace it when a revised notebook successfully completes verification. Acceptance means the implementation is complete and correct—not that its scores improved.