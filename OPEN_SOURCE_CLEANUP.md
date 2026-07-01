# Open-Source Cleanup

This folder has been sanitized for source release.

Removed or avoided:

- Python bytecode/cache directories (`__pycache__`, `.pyc`, `.ruff_cache`).
- Internal experiment helper folder with machine-specific absolute paths.
- Machine-specific reproduction paths in documentation and reproducibility metadata.
- Internal output paths, server names, and local user/host path references.

Retained intentionally:

- Frozen distance-head checkpoints required for RASR reproduction.
- Public method/configuration files required to reproduce the archived PairUAV result.
- `REPRODUCIBILITY.json` with path placeholders and expected SHA-256.

Expected archive SHA-256:

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```
