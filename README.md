# RH

Standalone extraction of the `rh` research code from `iontheprize`.

## Contents
- `rh.py`: core functions
- `twoadic.py`: 2-adic utilities
- `pascal.py`: Pascal-related helpers
- `deep_rh_analysis.py`: analysis scripts
- `rieman.py`: consolidated demo entry
- `test_*.py`: tests

## Getting started

```bash
uv run python rieman.py
```

## Certification

Generate a certification report (defaults: depth=4, gamma=3, d=0.05):

```bash
uv run python certify.py --out .out/certs
```

This writes a TOML file under `.out/certs/` with on-line/off-line lock rates. CI runs the same check and uploads artifacts.

### Latest generated certification (example)

Path:

```
.out/certs/cert-depth4-N17-20250902-031805.toml
```

Key summary from that run:

```toml
[summary]
online_locked_rate = 1.0
offline_locked_rate = 0.0
online_total = 33
offline_total = 33
online_locked = 33
offline_locked = 0
```

### What the numbers mean (plain language)

- On-line: we test points on the critical line (σ = 1/2) over a small window in t. Locked means the integer sandwich gap ≥ γ using the Metanion–Pascal phaselock. A good certificate has a high on-line locked rate (near 100%).
- Off-line: we test points slightly off the line (σ = 1/2 + d) over the same window. A good certificate has a low off-line locked rate (near 0%).

Parameters at defaults:
- depth=4 → N = 2^depth + 1 = 17 (resolution)
- γ=3 → required integer gap to count as locked
- d=0.05 → how far off the line we test
- window, step → size and granularity of the t-sweep around zeros

In short: the certificate succeeds if it locks on the line and fails off the line across the window.

## License
MIT unless noted otherwise.
