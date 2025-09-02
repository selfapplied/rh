# RH

Standalone extraction of the `rh` research code from `iontheprize`.

## Contents
- `rh.py`: core functions
- `twoadic.py`: 2-adic utilities
- `pascal.py`: Pascal-related helpers
- `deep_rh_analysis.py`: analysis scripts
- `simple_demo.py`: quick demo
- `main.py`: CLI entrypoint
- `test_*.py`: tests

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python rieman.py
```

## Certification

Generate a certification report (defaults: depth=4, gamma=3, d=0.05):

```bash
python certify.py --out .out/certs
```

This writes a TOML file under `.out/certs/` with on-line/off-line lock rates. CI runs the same check and uploads artifacts.

## License
MIT unless noted otherwise.
