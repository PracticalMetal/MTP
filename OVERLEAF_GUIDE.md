# Overleaf Upload Guide — Research-Standard Thesis

## Files to upload to Overleaf

```
THESIS.tex            ← main document (Overleaf > Menu > Main document = THESIS.tex)
references.bib        ← bibliography (BibTeX)
figures/
  ├── inference_scaling.pdf
  ├── label_efficiency_baselines.pdf     ← the 5-method comparison plot
  ├── label_efficiency_n03.pdf           ← low-noise label-efficiency curve
  ├── label_efficiency_n06.pdf           ← high-noise label-efficiency curve
  └── selectivity.pdf                    ← edge-type selectivity bar chart
```

## Compile settings (Overleaf > Menu)

- **Compiler:** pdfLaTeX
- **TeX Live version:** 2021 or later
- **Main document:** `THESIS.tex`

## First compile sequence

Overleaf auto-runs `pdflatex → bibtex → pdflatex → pdflatex` on the first
compile. Hit "Recompile" a second time to populate the table of contents,
list of figures, list of tables, and acronym list.

## Expected output

- **~30 pages** total
- Title page + abstract + ToC + LoF + LoT + acronyms (3 pp.)
- 8 chapters: Introduction, Background & Related Work, Problem
  Formulation, Proposed Method, Experimental Methodology, Results,
  Discussion, Conclusion (24 pp.)
- 5 PDF figures + 2 TikZ diagrams (gap diagram, architecture diagram)
- 8 tables (related-work summary, methods compared, main results,
  inference scaling, selectivity, label-efficiency low/high noise,
  baseline comparison, ablation)
- 1 algorithm pseudocode block
- References (~1.5 pp.)

## Required LaTeX packages

All standard on Overleaf:
- `geometry`, `setspace`, `lmodern`, `microtype`
- `amsmath`, `amssymb`, `bm`, `mathtools`
- `algorithm`, `algpseudocode`
- `graphicx`, `xcolor`, `tikz` (with libraries `shapes.geometric`,
  `arrows.meta`, `positioning`, `fit`, `backgrounds`, `calc`)
- `booktabs`, `multirow`, `array`, `tabularx`, `makecell`
- `caption`, `subcaption`
- `glossaries` (acronym package)
- `listings`, `enumitem`, `titlesec`, `hyperref`

## What's in each chapter

| Ch. | Title | Key content |
|---|---|---|
| 1 | Introduction | Motivation, two issues (graph noise + label asymmetry), 3 RQs, 4 contributions |
| 2 | Background & Related Work | GNNs, 4 pruning families, anomaly-detection methods, IoT-IDS literature, **research gap diagram** |
| 3 | Problem Formulation | Formal definitions, 2 problems, label-asymmetric reformulation, design desiderata |
| 4 | Proposed Method | Architecture diagram, all 3 stages, joint loss, **Algorithm 1 pseudocode**, theoretical intuition for class-conditional choice, deployment protocol |
| 5 | Experimental Methodology | Dataset details, methods compared (6-way), evaluation protocol, metrics, hyperparameters, reproducibility |
| 6 | Results | Main comparison, inference scaling, selectivity analysis, label-efficiency (low+high noise), **benchmark vs NeuralSparse/DropEdge**, ablation study |
| 7 | Discussion | Findings summary, why it works, limitations, threats to validity, panel-question Q&A |
| 8 | Conclusion + Future Work | Contributions summary, 6 concrete future research directions |

## Notes for the panel

The key plots are:
1. **`figures/label_efficiency_baselines.pdf`** — the 5-method comparison curve. This single plot is the strongest novelty defence.
2. **`figures/selectivity.pdf`** — proves the class-conditional choice is what drives noise-targeting.
3. **`figures/inference_scaling.pdf`** — proves the deployment-mode speed-up grows with graph size.

The key tables are:
- Table 6.6 (Baseline comparison) — proves we beat NeuralSparse and DropEdge.
- Table 6.7 (Ablation) — proves each design element contributes something.

## If anything looks off

- **Glossaries error**: if Overleaf complains about acronyms, ensure
  the compiler is set to "pdfLaTeX + makeglossaries" (it should auto-run).
- **Bibliography blank**: hit Recompile a second time.
- **Page count over 30**: tighten `\linespread{1.18}` to `\linespread{1.10}` in the preamble.
- **Page count under 28**: relax `\linespread{1.18}` to `\onehalfspacing`.

## Working artefacts on disk

| File | Purpose |
|------|---------|
| `THESIS.tex` | The Overleaf-ready LaTeX (research-standard report class) |
| `references.bib` | Bibliography (18 entries) |
| `THESIS_REPORT.md` | Markdown mirror with all results |
| `SESSION_CONTEXT.md` | Project state and headline numbers |
| `figures/*.pdf` | Plots (regenerable via `make_thesis_plots.py`) |
| `results/*.json` | Raw experimental results |
| `results/le_log_*.txt` | Full training logs |

## Regenerating plots after data changes

```
python make_thesis_plots.py
```
This reads `results/label_efficiency_n*.json` and writes `figures/*.pdf`.
