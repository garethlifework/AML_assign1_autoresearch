# Google Colab Setup

Use this if you want to run the new `pet_researcher` harness on a GPU-backed Colab runtime.

## 1. Put the project somewhere Colab can read it

Choose one:

- Upload this folder to Google Drive
- Push it to GitHub and clone it from Colab
- Zip the folder and upload it into the Colab session

GitHub is the cleanest option if you are going to iterate repeatedly.

The dataset does not need to be committed to the repo. The harness will download Oxford-IIIT Pet automatically on first run if `./data/oxford-iiit-pet` is missing.

## 2. Open Colab and enable GPU

- `Runtime` -> `Change runtime type`
- `Hardware accelerator` -> `GPU`

## 3. Bootstrap the environment

If using Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/AML_AutoResearcher
```

If using GitHub:

```bash
!git clone <your-repo-url> /content/AML_AutoResearcher
%cd /content/AML_AutoResearcher
```

Install Python dependencies:

```bash
!python -m pip install -q -r requirements-colab.txt
```

Check the runtime:

```python
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

## 4. Run the research harness

Quick staged run:

```bash
!python scripts/run_auto_research.py
```

Focused ResNet18 sweep:

```bash
!python scripts/run_stage2_sweep.py
```

Notebook-driven workflow:

- Open `auto_researcher_workbench.ipynb`
- Run cells stage by stage

## 5. What to send back here

I cannot directly log into your Colab account. The easiest collaboration loop is:

- run the script or notebook in Colab
- send me the console output or the resulting files under `runs/`
- especially send:
  - `runs/<experiment>/metrics.json`
  - `runs/<experiment>/val_predictions.csv`
  - `runs/<experiment>/val_confusion.png`
  - `runs/final_test_report.json` when you get there

If you want tighter iteration, zip the `runs/` folder and place it back in this workspace so I can inspect it directly.

## 6. Recommended first execution order

1. `stage1_resnet18_baseline_exec`
2. `scripts/run_stage2_sweep.py`
3. pick the top 1-2 validation winners
4. evaluate `resnet50` and the hierarchical specialist setup
5. run final shortlist with TTA on the held-out test split
