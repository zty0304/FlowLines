import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import HairStructUNet
from train.train_long_curve import LongCurveSVGGenerator, save_svg_prediction, save_mask


def extract_six_digit_id(path: Path):
    m = re.search(r"(\d{6})", path.stem)
    return None if m is None else m.group(1)


def load_rgb(path, size):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def load_mask(path, size):
    img = Image.open(path).convert("L").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr)[None, None]


def farthest_point_sample_xy(xy, n, seed=0):
    if len(xy) == 0:
        return np.random.rand(n, 2).astype(np.float32)

    selected = [np.random.randint(len(xy))]
    dist = np.sum((xy - xy[selected[0]]) ** 2, axis=1)

    for _ in range(1, n):
        idx = int(np.argmax(dist))
        selected.append(idx)
        d = np.sum((xy - xy[idx]) ** 2, axis=1)
        dist = np.minimum(dist, d)

    pts = xy[selected]
    return pts.astype(np.float32)


def anchors_from_center(center, num_strokes, thr=0.15):
    center_np = center[0, 0].cpu().numpy()
    h, w = center_np.shape

    ys, xs = np.where(center_np > thr)
    if len(xs) > 0:
        xy = np.stack([xs / (w - 1), ys / (h - 1)], axis=1)
    else:
        xy = np.zeros((0, 2))

    anchors = farthest_point_sample_xy(xy, num_strokes)
    return torch.from_numpy(anchors).float()[None]


@torch.no_grad()
def infer_center(stage1, image, mask):
    x = torch.cat([image, mask], dim=1)
    pred = stage1(x)
    center = pred["center"] * mask
    return center.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--stage1", default=str(PROJECT_ROOT / "weights/stage1_last.pt"))
    parser.add_argument("--svg_model", default=str(PROJECT_ROOT / "weights/long_curve_best.pt"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs/inference"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # load input
    image = load_rgb(args.image, args.size).to(device)
    mask = load_mask(args.mask, args.size).to(device)

    # load stage1
    stage1 = HairStructUNet(in_channels=4, base_ch=32, out_channels=3).to(device)
    stage1.load_state_dict(torch.load(args.stage1, map_location="cpu")["model"])
    stage1.eval()

    # load svg model
    ckpt = torch.load(args.svg_model, map_location="cpu")
    model = LongCurveSVGGenerator(
        in_ch=5,
        hidden=ckpt["args"]["hidden"],
        min_len=ckpt["args"]["min_len"],
        max_len=ckpt["args"]["max_len"],
        max_bend=ckpt["args"]["max_bend"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # center
    center = infer_center(stage1, image, mask)

    # anchors
    anchors = anchors_from_center(center, ckpt["args"]["num_strokes"]).to(device)

    # forward
    inp = torch.cat([image, mask, center], dim=1)
    pred = model(inp, anchors)

    stem = Path(args.image).stem

    save_mask(center[0, 0].cpu().numpy(), f"{args.output}/{stem}_center.png")
    save_svg_prediction(pred, f"{args.output}/{stem}.svg", args.size)

    print("done.")


if __name__ == "__main__":
    main()