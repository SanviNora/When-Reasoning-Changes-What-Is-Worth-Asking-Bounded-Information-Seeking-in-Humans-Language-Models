# Bounded-model representation caches

Rebuilt by `experiments/study_a/build_cache.py` (the authors' cache is not released). Each cache has the
approximations as arrays (`.npz`), the same approximations as readable expressions (`_expressions.json`) and a
manifest with the grammar, search settings, seeds and hash (`_manifest.json`).

| Cache | Used for |
|---|---|
| `bounded_cache_v1` | the bounded model in Study A and B1b, and M2c in B3 (conservative distance, as in the paper) |
| `bounded_cache_v1_rep2` | the cache-seed check: an independent rebuild with new seeds |
| `symmetric_cache_v1` | M2s in B3 (errors in both directions count) |
