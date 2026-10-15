# E017 — SIWC run01 bounded 500-total resume

- **Date:** 2026-10-15
- **Status:** Inconclusive operational attempt; stopped below target.
- **Parent/baseline:** E016 (original unbounded run01 terminal failure); implementation baseline D014, commit `8229473`.
- **Hypothesis:** A total checkpoint cap of 500 can safely bound the existing run while preserving its 15 saved records and stopping on the first failed request.
- **Changed variable:** The run scope was explicitly bound to `--limit 500`; model, prompt, service, and account configuration were unchanged. This was one separately authorized monitored resume, with no retry.
- **Protocol:** Resume the existing checkpoint once, monitor through terminal status, and compare safe checkpoint/canary/capture-preservation evidence. Do not inspect or publish report, response, evidence, label, or identifier contents.
- **Observed result:** Exit 2 after 24.02 seconds; 1 request started, 0 completed, 1 failed; checkpoint remained 15/500. The safe category was `case_only`. No new rejected-evidence quote was captured. The prior 15 records, 14-case canary, and existing private capture remained unchanged. The previous unbounded terminal job metadata was preserved separately; the current scope marker remains 500-bound.
- **Conclusion:** The bounded resume did not reach its target. `case_only` is a static diagnostic category, not proof of the response's actual text difference or cause. No label-quality, evidence-relevance, or clinical-validity conclusion follows. No automatic retry or additional request was made.
- **Limitations / next step:** The single failed attempt does not establish why the fresh response was rejected. Any further inference requires a new explicit user authorization and must honor the existing 500-bound scope unless separately changed. Preserve prior checkpoints and private artifacts.
