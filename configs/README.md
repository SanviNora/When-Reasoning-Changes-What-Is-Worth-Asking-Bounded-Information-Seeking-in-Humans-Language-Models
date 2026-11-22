# Configuration

| File | Contents |
|---|---|
| `model_registry.json` | every model: exact version (weights revision or API id), serving settings, decoding settings, thinking modes |
| `model_selection.json` | the panel: floor rule, screening results, admitted models and the depth model |
| `prompt_freeze_v1.json` | hashes of the frozen prompt templates for every task and condition (T0–T4) |

Each file was fixed before the runs it controls.
