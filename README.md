# AI helmet detection system

Streamlit image and video detection powered by Ultralytics YOLO. The
repository includes a pretrained helmet-status model and the configuration
and attribution needed to add the separately licensed training dataset.

## Run the app

From this folder, install the dependencies and start Streamlit:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The project includes a pretrained helmet-status model at `best.pt`, so the app
starts with helmet-specific detection. It predicts `With Helmet` and
`Without Helmet` directly. Select **Upload helmet model (.pt)** to use another
YOLO checkpoint, or select the general-object demo to try image/video
processing without helmet classification.

The bundled model is MIT-tagged by its [Hugging Face model page](https://huggingface.co/nnsohamnn/helmet-detection-yolo11);
source and checksum details are in [MODEL_ATTRIBUTION.txt](./MODEL_ATTRIBUTION.txt).
Its training-data provenance is not documented, and its predictions can be
wrong on small riders, occlusions, pedestrians, or background objects. Review
the annotated results rather than treating them as definitive. Inference
defaults to 1280 pixels to improve detection of small riders; lower the image
size in the sidebar for faster processing.

## Train a helmet model

Training requires a labeled YOLO dataset. The expected layout is:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
dataset.yaml
```

Each label file must have the same base filename as its image and contain
YOLO-format rows: `class_id x_center y_center width height`, with normalized
coordinates. The included dataset uses these classes:

```text
0: helmet
1: motorcycle
```

The bundled model predicts helmet status from two classes. Models trained on
the included dataset instead detect `helmet` and `motorcycle`; for those
weights, the app estimates **No helmet** when no helmet is associated with a
motorcycle. That estimate is not a separately trained no-helmet class, so
missed or occluded helmets can cause false warnings. The included dataset has
no explicit no-helmet class. For stronger results, use rider-level
`with_helmet` and `without_helmet` annotations and validate against images
representative of your deployment.

The dataset archive and images are intentionally excluded from this GitHub
repository because of their size. To train, download version 2 of the
[HCMC Whole dataset from Roboflow Universe](https://universe.roboflow.com/segmentation-z6r2a/hcmc-whole/dataset/2),
then place the color `helmet_original` train, validation, and test splits in
the layout above. The dataset is licensed under CC BY 4.0; its attribution is
included in `dataset/ATTRIBUTION.txt`.

The downloaded dataset contains `helmet` and `motorcycle` classes, not an
explicit no-helmet class. Training also requires local initial YOLO weights;
pass their location with `--model` if `yolov8n.pt` is not present. Then train
and validate:

```powershell
.\venv\Scripts\python.exe train.py
```

Training outputs are written under `runs/` and are not included in this
repository.

Optional training settings:

```powershell
.\venv\Scripts\python.exe train.py --epochs 100 --imgsz 640 --batch 8 --device 0
```

Use `--device cpu` to force CPU training or `--device auto` to let Ultralytics
select the available device. To use a dataset YAML elsewhere, pass
`--data "C:\path\to\dataset.yaml"`. The YAML's `path` setting controls where
its relative image paths are resolved.

Training writes its best weights under `runs/detect/helmet_detection/weights/`.
Restart or refresh the app after training; it will find the trained model
automatically.

## Troubleshooting

- **Dataset folders are missing:** download and extract the dataset as
  described above. Creating empty folders is not enough; train and validation
  images and their annotations are required.
- **No helmet weights found:** use the general-object demo, upload an existing
  helmet `.pt` model, or train one with your labeled dataset.
- **No helmet status:** the general-object demo cannot identify helmets.
  Download the dataset before training, or upload helmet-specific weights.
