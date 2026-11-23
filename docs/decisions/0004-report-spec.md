# 0004 — The participant report

Date: 2026-11-23. Status: accepted. Revisit when the group level arrives (0.3.0) or when a result file
gains a field the report should show.

## Question

After a participant run the user has, for each recording, result files, an edge table, a diagnostics
file and a run log. The plan for the report asked for one page per participant with eight sections
(summary, input, diagnostics, connectivity, controls, statistics, methods, errors) and a methods text
with references. It was written before the pipeline of 0.2.0 existed in its final form. Where does the
report get its figures, what does it show, and what does it leave out because the pipeline does not
keep the data?

## What was built

- `dirconn.reports.build` turns the derivatives of one subject into *reportlets* (a JSON record with a
  title, a caption, a status and an optional table, plus an SVG), `dirconn.reports.assemble` arranges
  them into `sub-XX.html` following `reports/_data/spec.yml`, and `dirconn.reports.boilerplate` writes
  the methods text and the references. The command writes them after every participant run and
  rebuilds them with `--reports-only` and `--boilerplate-only` ([usage](../usage.rst), [the
  report](../report.rst)).
- The page is one HTML file with inline CSS and SVG, no script and no external resource except the DOI
  links of the references. Every value comes through Jinja2 autoescaping; the only text inserted as it
  is, is the SVG read from a reportlet file.

## Decisions that differ from the plan

1. **Reportlets are built from the derivatives at report time, not by the pipeline steps.** The plan
   had each step write its figures while it ran. The pipeline package is not allowed to import plotting
   code (`workflows` may not import `viz` or `reports`, an import-linter contract). Reading the files that are on
   disk makes the report reproducible from the dataset alone: `--reports-only` gives the same page
   without the BIDS input or the configuration of the run, and a report of a dataset from an earlier
   run can be rebuilt. The cost is that the report can only show what the files hold, which
   leads to the next four items.
2. **Input has no variance before the z-score.** The diagnostics file records the standard deviation of
   each channel after preparation only. The section shows that, the share of valid samples, the
   sampling frequency and the preparation steps; the plan's bar chart of the variance before and after
   the z-score would need a quantity nothing stored.
3. **Diagnostics has one order curve per recording.** PR #13 selects the VAR order once per recording
   and every method uses it, so there is one `order_selection` record (in the diagnostics file) and
   one figure, not one per method. With a fixed order the reportlet is a note. The other checks (stability,
   residual whiteness, sampling adequacy) are read from each result's assumption checklist; they are
   not recomputed.
4. **Controls has no leakage pairs and no null distributions.** `controls.leakage_pairs` was removed
   from the configuration (decision 0003): a real recording has no head model to say which pairs leak.
   The time-reversal control is shown as a matrix and as an agreement table with ordinary Granger
   causality (`trgc` against `gc`). The time-reversal control is not tested, so it has no null distribution to show.
5. **Statistics shows the tests and the p values, not the null distributions.** The sidecar records
   which test ran, with which generator, how many surrogates and which seed, and the edge table has
   the p values; the surrogate samples themselves are not saved.
   The section therefore has a table of the tests, the significant rows and a histogram of the p
   values. A result that was not tested (time-resolved on a continuous record) shows the reason from
   the sidecar and nothing else; it is a description, not an error.
6. **References are in `reports/_data/references.json`.** Each method names its references in
   `MethodInfo.citations`; a test requires that every shipped method cites keys that exist. The entries
   were taken from Crossref (titles, venues, DOIs), with OpenAlex only for page ranges and the split of
   author names where Crossref lacked them, never from memory.
7. **The methods text goes to `logs/CITATION.{html,md,tex,bib}`.** It describes the dataset (all
   subjects), so it belongs with the logs and not with one participant; each participant page repeats
   it for the participant's own results. The text is built from the configuration, the selected orders,
   the parameters read from the result files (not from the configuration, since dirconn fills in some)
   and the tests and controls that ran.
8. **A report failure never touches the results.** Reports are built after the tasks and inside their
   own guards: a figure that raises is replaced by a `FAIL` placeholder and listed in the Errors
   section, a page that cannot be written leaves a `crash-report-<subject>.txt`, and the exit code
   becomes 1. Results and task files are only read.

## Rulings that shaped the result

- **Subject-level reportlets live in the subject directory.** `summary/overview` and `summary/errors`
  belong to no single recording, so they are written to `sub-XX/figures/` with the label `sub-XX`;
  recording reportlets stay next to the recording's diagnostics. Reading a label matches
  `<label>_desc-` exactly, so `sub-01` never picks up the files of `sub-01_task-rest`.
- **Time-frequency maps are raster images in the SVG.** A vector map of a time-resolved result had
  about 200 time points and made one SVG 14.7 MB and one report 46 MB. The mesh is now
  rasterized (axes, labels and colour bar stay vector); the largest SVG of the test data fell to 190 KB
  and the report to 1.47 MB. A test bounds the size of each reportlet SVG and of the page. The price
  is that the page contains `data:image/png` images, which the offline test allows (nothing is loaded).
- **SVG ids are prefixed per reportlet.** Matplotlib reuses ids (`figure_1`, `axes_1`, clip paths) in
  every figure, and inlining several SVGs in one HTML document makes the ids collide and the clip
  paths point to the wrong figure. Each inlined SVG gets a running prefix, and the `url(#..)` and
  `href="#.."` references are rewritten with it.
- **Statuses use rules of thumb and say so.** The thresholds of the checks are those of the estimators
  and diagnostics; the report adds three of its own (fewer than half of the samples valid, a selected
  order at the largest candidate, a sign disagreement in the time-reversal control). None of them is
  calibrated, and the report text, the captions and the documentation say that.

## Not done

- **No group-level report.** The group level arrives in 0.3.0.
- **No leakage annotation, no null-distribution figures, no pre-z-score variance** (items 2, 4 and 5).
- **No other output language and no `output.figures` switch.** The report is English; the configuration
  has no option for figures or language.
- **No JavaScript.** Navigation is anchors and `<details>`; the page must work from a mail attachment
  and in a locked-down browser.
