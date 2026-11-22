# State families and splits

`families_v1.json`: 252 state families (histories of 2–3 clues leaving 4–16 codes possible), distinct under the
game's symmetries and from every Experiment 1 starting position, each assigned to one fixed split.
`manifest_v1.json`: seed, counts and sha256. Both were committed before any confirmation run
(`experiments/study_b/make_splits.py`).

| Split | Families | Used for |
|---|---|---|
| `development` | 24 | B2 battery runs for developing the analyses (not used in the B3 fits or tests) |
| `processing_calibration` | 60 | B3 fits |
| `confirmation` | 80 | B3 held-out comparison |
| `reserve` | 40 | the pre-stated M1a confirmation test |
| `choice_calibration` | 24 | native question choice |
| `support_calibration` | 24 | not used in Studies A and B |
