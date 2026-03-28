# AML Assignment 1 Auto Researcher

Auto-research harness for Oxford-IIIT Pet breed classification.

## Contents

- `pet_researcher/`: reusable training, data, hierarchy, and evaluation code
- `scripts/`: staged experiment runners
- `auto_researcher_workbench.ipynb`: Colab-friendly orchestration notebook
- `COLAB_SETUP.md`: quick setup guide for Google Colab

## Colab Quick Start

1. Enable a GPU runtime in Colab.
2. Clone this repo or mount it from Google Drive.
3. Install the small extra dependency set:

```bash
python -m pip install -q -r requirements-colab.txt
```

4. Run either:

```bash
python scripts/run_auto_research.py
```

or

```bash
python scripts/run_stage2_sweep.py
```

See `COLAB_SETUP.md` for the full workflow and what outputs to send back for the next optimization round.
