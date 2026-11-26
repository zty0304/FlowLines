# Environment and Reinstallation Guide

## Tested reference environment

The existing `hairstruct` environment was inspected on Linux. These are the versions reported by that machine:

| Component | Tested version |
|---|---:|
| OS | Ubuntu 24.04 |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| TorchVision | 0.26.0+cu128 |
| PyTorch CUDA build | 12.8 |
| NVIDIA driver | 580.159.04 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| CUDA Toolkit / `nvcc` | 12.8.93 |
| CMake | 3.28.3 |
| GCC | 13.3.0 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0 |
| Pillow | 11.1.0 |
| Matplotlib | 3.9.1 (installed; not required by the cleaned entry points) |
| tqdm | 4.67.3 |
| PyYAML | 6.0.3 |
| scikit-image | 0.25.2 |
| SciPy | 1.15.2 |
| CairoSVG | 2.9.0 |
| cairocffi | 1.7.1 |
| diffvg / `pydiffvg` (stage-2 training only) | 0.0.1 |

The original Conda YAML was not an exact lockfile: it specified a CUDA 12.1 Conda runtime, while the active environment reported PyTorch wheels built for CUDA 12.8. The files in this project describe the tested Python stack more closely. The NVIDIA driver and CUDA Toolkit are system-level prerequisites for building and running the CUDA-enabled `pydiffvg` extension.

## Recreate the tested Linux GPU environment

1. Install an NVIDIA driver compatible with CUDA 12.8 and the CUDA Toolkit 12.8, including `nvcc`. Install CMake and a working C++ compiler. On the tested machine these were driver 580.159.04, CUDA Toolkit 12.8.93, CMake 3.28.3, and GCC 13.3.0.
2. Create and activate the Conda environment, then install the Python packages:

   ```bash
   conda env create -f environment.yml
   conda activate flow-line-ex
   python -m pip install -r requirements.txt
   ```

3. If you plan to train stage 2, install the pinned DiffVG source. The recursive clone fetches its own source submodules. This step is not required for inference or stage-1 training:

   ```bash
   mkdir -p third_party
   git clone --recursive https://github.com/BachiLi/diffvg.git third_party/diffvg
   cd third_party/diffvg
   git checkout 85802a71fbcc72d79cb75716eb4da4392fd09532
   git submodule update --init --recursive
   DIFFVG_CUDA=1 python setup.py install
   cd ../..
   ```

4. Confirm imports and CUDA visibility:

   ```bash
   python -c "import torch, torchvision, pydiffvg, cairosvg, cv2; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```

`pydiffvg` must be built after installing PyTorch and with the CUDA Toolkit/compiler available. If the build reports that CUDA is unavailable, check `nvcc --version`, `which nvcc`, `which c++`, and that PyTorch reports `torch.cuda.is_available() == True` before building.

## CPU-only or different CUDA systems

The bundled `requirements.txt` pins the tested CUDA 12.8 PyTorch wheels. For CPU-only systems or another CUDA runtime, install a PyTorch/TorchVision pair selected for the target system from the official PyTorch installation selector, and edit/remove the two PyTorch lines in `requirements.txt` before installing the remaining packages. Build DiffVG with `DIFFVG_CUDA=0` for CPU-only use:

```bash
DIFFVG_CUDA=0 python setup.py install
```

The current inference script automatically falls back to CPU when CUDA is unavailable. Training can also be run with `--device cpu`, though it will be substantially slower.

## Package inventory

The direct dependencies for inference and stage-1 training are PyTorch, TorchVision, NumPy, OpenCV, Pillow, tqdm, PyYAML, scikit-image, SciPy, and CairoSVG. Stage-2 training additionally requires DiffVG/`pydiffvg` to render Bézier curves as raster masks while preserving gradients back to the curve parameters. It is built from source rather than installed from the ordinary PyPI package index because its C++/CUDA extension must match the installed PyTorch and CUDA Toolkit.
