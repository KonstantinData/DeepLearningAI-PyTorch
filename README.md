# Notebook support files

This folder contains the minimal files needed to run `C1_M1_Lab_1_simple_nn.ipynb` in your repo.

## Files
- `helper_utils.py` – plotting helpers imported by the notebook
- `requirements.txt` – Python dependencies

## Setup (venv example)
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Jupyter
```bash
pip install jupyter
jupyter lab
```

Place `helper_utils.py` in the same folder as the notebook (or ensure it’s on your `PYTHONPATH`).
