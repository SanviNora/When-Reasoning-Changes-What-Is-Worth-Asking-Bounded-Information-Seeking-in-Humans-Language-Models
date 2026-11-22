# Human data

The human data are not included in this repository. They are the public release accompanying
Cheyette, Callaway, Bramley, Nelson & Tenenbaum (2025), *Representational constraints shape human
information-seeking* (PsyArXiv, doi 10.31234/osf.io/w4b3y), on OSF (project `vjrqu`). No license is
stated, so the files are linked rather than redistributed.

Download the OSF project and place it at `data/human/cheyette2025_osf_vjrqu/`. The scripts read:

| File | Used for |
|---|---|
| `Experiments/E1/mastermind_E1.csv` | Experiment 1: games continued from supplied guesses |
| `Experiments/E2/mastermind_E2.csv` | Experiment 2: forced choice between a simpler and a more informative guess (33 participants released) |
| `Experiments/E3/mastermind_E3.csv` | Experiment 3: free play, 10 games per person |
| `Experiments/E3/analysis/e3_human_preprocessed.csv` | Experiment 3 exclusions (39 participants after cleaning) |
| `Models/Mastermind/` | the authors' grammar, used to rebuild the bounded model's representation cache |

The authors' fitted-parameter files and approximation cache are not part of the release; the cache is
rebuilt by `experiments/study_a/build_cache.py` (see `data/model_cache/`).
