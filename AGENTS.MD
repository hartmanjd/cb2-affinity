# CB2 Affinity ML Project

This repository is the clean, professional rebuild of an earlier
experimental CB2 machine-learning project.

## Project goals

- Curate a reproducible human CB2 receptor affinity dataset.
- Preserve scientifically relevant assay and chemistry metadata.
- Build and compare machine-learning baselines.
- Evaluate models using chemically meaningful validation strategies.
- Investigate molecular activity cliffs where supported by the data.

## Development principles

- Prefer clear, beginner-readable Python.
- Explain scientific and machine-learning decisions in comments.
- Avoid unnecessary abstraction.
- Keep notebooks readable and presentation-friendly.
- Do not change scientific assumptions without documenting the reason.
- Keep raw source data unchanged.

## Rough-draft reference

A second folder may be attached to the Codex project containing the
earlier rough-draft notebooks.

Treat that folder as reference material.

Do not modify files in the rough-draft folder unless explicitly asked.

When rebuilding something from the rough draft:
1. Explain what the old implementation does and compare it to the new implementation. 
2. The new implementation will likely be less sophisticated but more beginner friendly.
3. Identify anything scientifically important that must be preserved.
4. Rebuild it cleanly in this repository.