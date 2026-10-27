# Run in Google Colab

## 1. Upload and extract

Upload `hybrid-mmwave-mimo-benchmark-v3.zip` using:

```python
from google.colab import files
uploaded = files.upload()
```

Then extract the ZIP:

```python
import zipfile
zip_path = "/content/hybrid-mmwave-mimo-benchmark-v3.zip"
with zipfile.ZipFile(zip_path) as z:
    z.extractall("/content")
%cd /content/hybrid-mmwave-mimo-benchmark-v3
```

## 2. Install dependencies and run tests

```python
%pip install -r requirements.txt
!python -m unittest discover -s tests -v
```

All eleven tests should pass before the main experiment.

## 3. Run a short smoke test

```python
!python main.py --n-realizations 5 --outdir results_smoke
```

This creates small result files to verify the complete experiment pipeline.

## 4. Run the full benchmark

```python
!python main.py --n-realizations 100 --seed 2026 --outdir results
```

The run evaluates SNR, CSI-error, and RF-chain sweeps. Runtime depends on the Colab CPU/GPU instance; this implementation uses NumPy and does not require a GPU.

## 5. Display results

```python
from pathlib import Path
from IPython.display import display, Image
import pandas as pd

results = Path("results")
summary = pd.read_csv(results / "summary.csv")
display(summary.head(20))
for image_path in sorted(results.glob("*.png")):
    print(image_path.name)
    display(Image(filename=str(image_path)))
```

Use `summary.csv` for mean and standard deviation across Monte Carlo channel realizations. Do not write resume performance claims until the full benchmark completes and the comparisons are reviewed.
