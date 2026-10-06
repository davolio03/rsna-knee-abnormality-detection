# Competition Contract

Phase 0 is still in progress. Official overview, evaluation, and data descriptions were previously checked through indexed content; the full rules, host clarification, and submissions page were directly read in an authenticated Chrome session on 2026-10-02 under explicit read-only authorization. This confirmed web access and the accepted-rules UI, not Kaggle API/CLI access. No personal account identifiers are recorded.

## Published contract

| Area | Published description | Source |
|---|---|---|
| Schedule | Start: 2026-07-30. Entry and team-merger deadline: 2026-10-15 23:59 UTC. Final deadline: 2026-10-22 23:59 UTC. Winners' requirements: 2026-11-05 23:59 UTC. Organizers may update the timeline. | [Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) |
| Notebook | CPU/GPU notebook runtime is limited to 9 hours; internet is disabled. The overview permits freely/publicly available external data and pretrained models. Required output filename: `submission.csv`. | [Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) |
| Evaluation | Mean ROC-AUC across 12 targets at study level. Published sample prediction is 0.5 for each target; this is not a project result. | [Evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation) |
| Submissions | Up to 5 submissions per day and up to 2 final submissions. If fewer than 2 are selected, Kaggle may auto-select from the highest-scoring submissions; the fallback does not establish what will be selected in a particular case. | [Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules), [Submissions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions) |
| Team and sharing | Teams may have up to 5 members. Private code/data sharing is limited to registered team members; public code is to be shared on the competition forum or notebooks and must satisfy applicable licenses. | [Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) |
| Winning obligations | Winners must provide training/inference code, weights, documentation, and runtime-environment details; the page describes public distribution of model weights through a Kaggle dataset. | [Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) |

The scoring columns are `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`.

## External data, tools, and report processing

Rules require external data, models, and tools to offer fair and reasonable access, and place limits around cost/accessibility; the host assesses unfair advantage. The rules also prohibit redistributing competition data to nonparticipants and incorporate the [RSNA MIRA license](http://rsna.org/mira-license). The full license was read on 2026-10-02 and contains a broad restriction on derivative works; its scope over competition-derived labels, features, or trained weights has not been clarified. Review provider terms and applicable access/cost constraints before selecting or operating an external service.

The [host discussion, “Use of Commercially Hosted LLMs”](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) was directly read. Its author had a Competition Host badge. The post dated 2026-08-09 says commercially hosted LLM services may process report text for label extraction subject to competition rules, reasonable access/cost, and provider terms. A host reply dated 2026-08-27 explicitly says LLM APIs may read reports and generate labels. This verifies conditional host permission, not blanket approval for every provider/workflow and not authorization to transmit this project's reports. D006 records the evidence; provider, model, prompt, and service strategy remain unselected.

## Licensing ambiguity and remaining unknowns

The rules page names CC-BY-NC 4.0 in the winner-license area and rule 2.5, while nearby text calls the license OSI-approved and says it must not restrict commercial use. Those statements conflict on their face. The MIRA license also has a broad no-derivative-work restriction whose application to labels, features, or weights remains unclear. Seek organizer clarification before distributing competition-derived artifacts or commercial deployment; do not resolve legal precedence here or select a project license. This ambiguity is not recorded as a blanket ban on private competition research, including research intended to inform a saleable product. It does not grant any additional permission to access or download competition data.

The rules do not resolve practical GPU/VRAM/RAM availability or establish a representative inference runtime, and no local metadata/sample-file schema has been verified. Kaggle web acceptance/access is confirmed by the authenticated UI; API/CLI access is not. Do not substitute generic platform defaults for these unknowns.

## Local evidence and smoke-test path

No competition images, reports, credential values, or hidden scoring data were inspected or used. The local package currently contains empty `__init__.py` files; that is repository structure, not proof of an executable inference path.

`scripts/smoke_submission.py` generates and reads back an ignored local CSV using three explicitly fictitious identifiers. Separately, `notebooks/phase0_kaggle_submission.py` is the reproducible source for the authorized Kaggle smoke test; it creates constant predictions and reads its output back. One authorized constant-prediction smoke submission completed successfully with public score 0.500. This is end-to-end submission/scoring-path evidence, not a model-performance result. Private execution metadata is intentionally omitted from this public record. No raw data was downloaded, no training occurred, and no final-selection change was made. Kaggle API/CLI access and the full test schema remain unverified; the result does not establish future inference practicality or model quality.
