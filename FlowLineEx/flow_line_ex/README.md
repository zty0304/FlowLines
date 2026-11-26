# Flow Line Extraction and Long-Curve Generator

A cleaned two-stage training and inference project extracted from the original `flow_line_gen` workspace. The original source directory was left unchanged.

## Project layout

- `train/train_stage1.py`: trains a HairStruct U-Net to predict hair centerline and orientation fields.
- `train/train_long_curve.py`: freezes the stage-1 model and trains a long Bézier flow-line generator.
- `inference/infer_long_curve.py`: runs single-image inference and writes a centerline probability map and SVG.
- `preprocessing/`, `models/`, `datasets/`, and `utils/`: local modules required by the training and inference scripts.
- `weights/stage1_last.pt` and `weights/long_curve_best.pt`: matching pretrained checkpoints.
- `configs/train_stage1.yaml`: stage-1 training configuration.
- `environment.yml` and `requirements.txt`: tested Conda and pip dependency specifications.
- `ENVIRONMENT.md`: tested environment details and installation guide for another Linux machine.

Training data is intentionally not copied into this project. Place it in this layout, or provide custom paths to the training script:

```text
data/raw/
  train/{images,masks,lines,svg}/
  val/{images,masks,lines,svg}/
```

Image, mask, line, and SVG filenames should share the same six-digit ID (for example, `000123.png` and `000123.svg`). Stage 1 uses images, masks, and line annotations. Stage 2 uses images, masks, and SVG annotations.

## Environment setup

The project was tested on Linux with Python 3.10, PyTorch 2.11.0 + CUDA 12.8, and an NVIDIA GPU. For a matching environment:

```bash
conda env create -f environment.yml
conda activate flow-line-ex
python -m pip install -r requirements.txt
```

`pydiffvg` is only needed for stage-2 training, where it differentiably rasterizes Bézier curves. It is not needed for inference, which writes the predicted curves directly to SVG. The compiled extension is installed separately from its pinned source revision; see [ENVIRONMENT.md](ENVIRONMENT.md) for CUDA Toolkit, compiler, and installation instructions. For CPU-only machines or a different CUDA/PyTorch combination, follow the PyTorch selector and adjust the PyTorch wheel line in `requirements.txt` before installing.

Verify the required modules and GPU runtime:

```bash
python -c "import torch, torchvision, cairosvg, cv2; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

## Training

Run commands from the project root. Preprocess the raw line annotations into stage-1 `.npz` samples:

```bash
python preprocessing/prepare_dataset.py --raw-root data/raw/train --out-dir data/processed/train --size 512
python preprocessing/prepare_dataset.py --raw-root data/raw/val --out-dir data/processed/val --size 512
```

Train stage 1:

```bash
python train/train_stage1.py --config configs/train_stage1.yaml
```

Train the long-curve stage 2 (uses `weights/stage1_last.pt` and the raw SVG dataset by default):

```bash
python train/train_long_curve.py
```

Use `--train-image-dir`, `--train-mask-dir`, and `--train-svg-dir`, plus the corresponding `--val-*` options, if the data is stored elsewhere. New checkpoints and previews go under `runs/`; the bundled weights are not overwritten.

## Inference

Inference requires an input image and its corresponding grayscale or binary hair mask. Both are resized to the requested square resolution:

```bash
python inference/infer_long_curve.py \
  --image /path/to/image.png \
  --mask /path/to/hair_mask.png
```

The default checkpoints are `weights/stage1_last.pt` and `weights/long_curve_best.pt`. Results are written to `outputs/inference/`. Override checkpoint and output paths with `--stage1`, `--svg_model`, and `--output`.

## Checkpoint notes

- `stage1_last.pt` was copied from the original `outputs/exp2/checkpoints/last.pt`. The bundled long-curve checkpoint records this stage-1 model as its dependency, so the pair matches.
- `long_curve_best.pt` was copied from the original `outputs/svg_long_curve_gen/checkpoints/best.pt` (epoch 68).
- If stage 1 is retrained, train stage 2 with that new stage-1 checkpoint; do not mix checkpoints from different runs.
- The bundled stage-2 checkpoint has `lambda_conf=0`, so its confidence head was not meaningfully supervised. Inference falls back to retaining every predicted curve if all scores are below the confidence threshold. For future training, add confidence supervision or select a threshold on a validation set.
