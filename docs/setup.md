# Setup

Python 3.11.15 and pip are the supported installation path. Create the core
and GEDI environments with the Python standard library:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .

python -m venv .venv-gedi
.venv-gedi/bin/python -m pip install -r environments/gedi/requirements.txt
```

`requirements.txt` pins the complete core and test dependency graph.
`environments/gedi/requirements.txt` independently pins GEDI 1.0.8 and its
older scientific stack. Do not install GEDI into the core environment.

Verify both interpreters:

```bash
.venv/bin/python -c "import process_discovery_cash, pm4py"
.venv-gedi/bin/python -c "import gedi, smac"
```

Repository entry points can be invoked directly without activating either
environment:

```bash
.venv/bin/python scripts/run_discovery.py --help
.venv/bin/python scripts/run_metric.py --help
```

Host storage can be relocated with `DATA_ROOT`, `RESULTS_ROOT`, and `LOG_ROOT`.
A `.env` file may define those names, but real environment variables take
precedence.
