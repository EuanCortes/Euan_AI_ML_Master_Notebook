# The Applied AI Engineer's Compendium: Project Plan & Architecture

This document covers **architecture and execution strategy**. The curriculum
itself lives in [AI_ML_Master_Checklist.md](AI_ML_Master_Checklist.md), which is
the single source of truth for chapter content — this file deliberately does not
duplicate it.

## 1. Project Overview

This repository is a living portfolio that combines rigorous mathematical theory
(via LaTeX) with production-ready machine learning engineering (via containerized
Python code).

The curriculum runs in three phases:

*   **Phase 1 — Mathematical & Engineering Foundations (Chapters 1–2).** The
    mathematics the models are written in, then the infrastructure that makes
    everything after it reproducible.
*   **Phase 2 — Applied Machine Learning & AI (Chapters 3–8).** Classical ML
    through to generative AI, deriving the theory and building the models.
*   **Phase 3 — Production & Deployment (Chapter 9).** The capstone: a trained
    model deployed as a live, multi-container service.

The defining structural decision is that **engineering infrastructure is Chapter
2, not an afterthought**. Dependency management, data versioning, branch
protection, and CI/CD are all in place before the first model is trained, so
every chapter from 3 onward inherits a reproducible pipeline rather than
retrofitting one.

## 2. Repository Architecture

A modular structure keeps the repository scalable as chapters and projects are
added.

```text
Euan_AI_ML_Master_Notebook/
├── theory/                      # LaTeX source for the master notebook
│   ├── main.tex                 # Master document — compile this
│   ├── preamble.tex             # Packages, styling, theorem boxes, math macros
│   ├── chapters/                # One file per chapter (01_math_foundations.tex …)
│   ├── figures/                 # All images; \graphicspath set in preamble
│   └── references.bib           # Single shared bibliography
├── src/
│   └── mlnotebook/              # Shared library code, installed via uv
├── projects/                    # Executable code, one folder per chapter
│   ├── 02_infrastructure/       # Chapter 2 — DVC, CI/CD, tooling setup
│   ├── 03_classical_ml/         # Chapter 3 — e.g. housing_price_xgboost/
│   ├── 08_gen_ai/               # Chapter 8 — e.g. local_rag_pipeline/
│   └── 09_capstone/             # Chapter 9 — FastAPI + Postgres + Docker stack
├── docs/                        # Planning and reference material
│   ├── AI_ML_Master_Checklist.md    # Curriculum (source of truth)
│   ├── ProjectPlanOverview.md       # This file
│   └── GEMINI.md                    # Mentor system prompt
├── .github/workflows/           # CI/CD pipelines
│   └── compile_theory.yml       # Builds the theory book PDF
├── pyproject.toml               # Root project, uv build backend
└── .python-version              # Pinned interpreter (3.12)
```

Chapter *N* of the book always maps to `projects/0N_*/` in the code half of the
repository. The mapping table in the
[master checklist](AI_ML_Master_Checklist.md) is authoritative.

## 3. Future-Proofing Guidelines

These are the standing rules. Chapter 2 is where they get implemented; from then
on they are simply enforced.

*   **Modular LaTeX:** Use `\input{chapters/...}` in `main.tex`. Never write
    chapter prose in the master file.
*   **Automated PDF Compilation:** GitHub Actions compiles the LaTeX into a PDF
    on every push and pull request touching `theory/`, and uploads it as a build
    artifact.
*   **Isolated Dependencies:** No global `requirements.txt`. The root project
    uses `uv` with a committed `uv.lock`; projects with conflicting requirements
    get their own `pyproject.toml` or a dedicated `Dockerfile`.
*   **Data Version Control (DVC):** Datasets never enter Git history. DVC holds
    lightweight pointers to data in a cloud bucket (AWS S3, Google Drive) for
    reproducibility.
*   **Branch Protection:** `main` is protected. Work happens on feature branches
    and merges via pull request with passing status checks and a linear history.
*   **Production-Ready Python:** Object-oriented where it earns its place, type
    hints throughout, docstrings on public interfaces, and `pytest` coverage for
    library code.

## 4. Execution Strategy

Work proceeds chapter by chapter. For each chapter:

1.  **Write the theory.** Derive the mathematics in
    `theory/chapters/0N_*.tex`, working through it by hand first.
2.  **Build the implementation.** Write the corresponding code in
    `projects/0N_*/`, applying the Chapter 2 standards.
3.  **Wire it to CI.** Tests run on pull request; the theory book recompiles on
    merge.
4.  **Merge via pull request.** No direct commits to `main`.

The two foundation chapters come first and in order — Chapter 1 because the
mathematics underpins everything, Chapter 2 because the infrastructure carries
everything. Chapter 9 is the capstone and deliberately comes last, since it
deploys a model built in an earlier chapter rather than introducing a new one.
