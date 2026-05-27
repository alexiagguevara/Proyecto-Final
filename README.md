# AstroMetrix

AstroMetrix is a desktop application for astrocyte morphology analysis from GFAP epifluorescence microscopy images acquired in the green fluorescence channel.

It provides two main analysis modes:

* **State Classifier**
  Binary classification of astrocyte morphology as **Control** or **Pro-inflammatory**.

* **Progression Profiler**
  Continuous morphology scoring in two modes:

  * **Experiment-Anchored**: computes a recovery-oriented score using control and inflamed reference images from the same experiment.
  * **Reference-Free**: computes an inflammatory progression score using population-level anchors from the training dataset.

---

## Features

### State Classifier

Classifies a single `.tif` image based on three selected morphological features:

* `median_thickness`
* `median_tortuosity`
* `median_segment_length`

### Progression Profiler

Two analysis modes are available:

#### 1. Experiment-Anchored

* Uses multiple control and inflamed reference images from the same experiment
* Outputs a **recovery score** and an **inflammatory score**
* Reduces between-experiment variability

#### 2. Reference-Free

* Does not require reference images from the current experiment
* Outputs an **inflammatory score**
* Uses population-level anchors from the training dataset

### Additional outputs

For each analysis, the app can display:

* wavelet image
* segmentation mask
* skeleton
* extracted features
* recent analysis history with saved result snapshots

---

## Input requirements

AstroMetrix is designed for `.tif` epifluorescence microscopy images from GFAP-labelled samples acquired in the **green fluorescence channel**.

The current analysis pipeline assumes that:

* the relevant GFAP signal is contained in the **green channel**
* the sample was labelled using a **secondary antibody conjugated to a green fluorophore**
* morphology is extracted from that green fluorescence signal

Images acquired from other fluorescence channels, other stains, brightfield modalities, or unrelated imaging pipelines are not supported and may produce invalid results.

---

## Requirements

* Python 3.11 or 3.12 recommended
* macOS or Windows

---

## Run from source

From the project root:

### 1. Create and activate a virtual environment

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r app_ui/requirements.txt
```

### 3. Run the app

```bash
python app_ui/main.py
```

---

## Build desktop app

### Build requirements

```bash
pip install -r app_ui/requirements-build.txt
```

---

## macOS build

Build the macOS app from macOS:

```bash
rm -rf build dist
pyinstaller AstroMetrix-mac.spec
```

Output:

```text
dist/AstroMetrix.app
```

### Notes

* The macOS app is intended for local use or informal sharing.
* It is not notarized for App Store distribution.

---

## Windows build

Build the Windows executable from Windows:

```powershell
Remove-Item -Recurse -Force build, dist
pyinstaller AstroMetrix-windows.spec
```

Output will be created in:

```text
dist/
```

### Important

Windows builds should be generated on a Windows machine.

---

## Local data storage

AstroMetrix stores recent analyses locally on the user machine.

### macOS

```text
~/Library/Application Support/AstroMetrix/
```

### Windows

```text
%APPDATA%\AstroMetrix
```

Stored files include:

* `recent_analyses.json`
* `recent_assets/`

These are user-generated files and are not stored inside the packaged app bundle.

---

## Project structure

```text
Proyecto-Final/
├── app_ui/
│   ├── main.py
│   ├── app.py
│   ├── widgets.py
│   ├── paths.py
│   ├── recent_store.py
│   ├── core/
│   │   └── pipeline.py
│   ├── views/
│   │   ├── home.py
│   │   ├── classifier.py
│   │   └── progression.py
│   ├── assets/
│   ├── requirements.txt
│   └── requirements-build.txt
├── final_binary_model.py
├── temporal/
│   └── pipeline_temp.py
├── final_binary_model.joblib
├── final_binary_model_metadata.joblib
├── temporal_score_metadata.joblib
├── AstroMetrix-mac.spec
└── AstroMetrix-windows.spec
```

---

## Model resources

The app requires these files at runtime:

* `final_binary_model.joblib`
* `final_binary_model_metadata.joblib`
* `temporal_score_metadata.joblib`

These are bundled during desktop app packaging and are also required when running from source.

---

## Notes

* Input images must be `.tif` files.
* The app is intended for GFAP epifluorescence images with signal in the green channel.
* Images from other channels or staining configurations are not supported.
* Recent analyses are stored locally on the machine.
* The current UI supports click-to-browse upload.
* macOS and Windows may render some fonts and wrap behavior slightly differently.
* Windows packaging and testing should always be validated on a Windows machine.

---

## Status

Current validated setup:

* macOS desktop app build working
* Windows source execution working
* Cross-platform user data storage implemented

---

## License

No license has been specified yet.
