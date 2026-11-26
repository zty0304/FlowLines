import argparse
import io
import os
import re
import sys
from pathlib import Path

import cairosvg
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.unet import HairStructUNet

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def extract_six_digit_id(path: Path):
    m = re.search(r"(\d{6})", path.stem)
    return None if m is None else m.group(1)


def find_file_by_id(six_id: str, folder, exts):
    folder = Path(folder)
    for ext in exts:
        p = folder / f"{six_id}{ext}"
        if p.exists():
            return p
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts and extract_six_digit_id(p) == six_id:
            return p
    return None


def load_rgb(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def load_mask(path, size):
    img = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


def rasterize_svg_to_mask(svg_path, size):
    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=size,
        output_height=size,
        background_color="white",
    )
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    rgb = np.asarray(img).astype(np.float32) / 255.0
    mask = 1.0 - np.min(rgb, axis=-1)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def save_mask(x, path):
    x = np.clip(x, 0.0, 1.0)
    Image.fromarray((x * 255).astype(np.uint8)).save(path)


def farthest_point_sample_xy(xy, n, seed=0):
    if len(xy) == 0:
        rng = np.random.default_rng(seed)
        return rng.random((n, 2), dtype=np.float32)

    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(xy))
    selected = [first]
    dist = np.sum((xy - xy[first]) ** 2, axis=1)

    for _ in range(1, n):
        idx = int(np.argmax(dist))
        selected.append(idx)
        d = np.sum((xy - xy[idx]) ** 2, axis=1)
        dist = np.minimum(dist, d)

    pts = xy[selected]
    if len(pts) < n:
        pad = np.repeat(pts[-1:], n - len(pts), axis=0)
        pts = np.concatenate([pts, pad], axis=0)
    return pts.astype(np.float32)


class HairSVGLongCurveDataset(Dataset):
    def __init__(self, image_dir, mask_dir, svg_dir, size, num_strokes, anchor_thr=0.15):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.svg_dir = Path(svg_dir)
        self.size = size
        self.num_strokes = num_strokes
        self.anchor_thr = anchor_thr

        self.items = []
        svg_files = sorted([p for p in self.svg_dir.iterdir() if p.is_file() and p.suffix.lower() == ".svg"])

        skipped_no_image = 0
        skipped_no_mask = 0
        skipped_no_id = 0
        for svg_p in svg_files:
            six_id = extract_six_digit_id(svg_p)
            if six_id is None:
                skipped_no_id += 1
                continue
            image_p = find_file_by_id(six_id, self.image_dir, IMG_EXTS)
            mask_p = find_file_by_id(six_id, self.mask_dir, IMG_EXTS)
            if image_p is None:
                skipped_no_image += 1
                continue
            if mask_p is None:
                skipped_no_mask += 1
                continue
            self.items.append((image_p, mask_p, svg_p, six_id))

        print("=" * 60)
        print(f"SVG files: {len(svg_files)}")
        print(f"Matched samples: {len(self.items)}")
        print(f"Skipped no six-digit id: {skipped_no_id}")
        print(f"Skipped no image: {skipped_no_image}")
        print(f"Skipped no mask: {skipped_no_mask}")
        print("=" * 60)

        if len(self.items) == 0:
            raise RuntimeError("No matched samples found.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_p, mask_p, svg_p, six_id = self.items[idx]

        image = load_rgb(img_p, self.size)
        mask = load_mask(mask_p, self.size)
        gt_svg = rasterize_svg_to_mask(svg_p, self.size)

        # Anchor from GT SVG raster only. No skeletonization.
        # The model will generate a long Bezier ray from each anchor.
        ys, xs = np.where(gt_svg > self.anchor_thr)
        if len(xs) > 0:
            xy = np.stack([xs / (self.size - 1), ys / (self.size - 1)], axis=1).astype(np.float32)
        else:
            xy = np.zeros((0, 2), dtype=np.float32)
        anchors = farthest_point_sample_xy(xy, self.num_strokes, seed=int(six_id))

        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float(),
            "mask": torch.from_numpy(mask)[None].float(),
            "gt_svg": torch.from_numpy(gt_svg)[None].float(),
            "anchors": torch.from_numpy(anchors).float(),
            "stem": six_id,
        }


class LongCurveSVGGenerator(nn.Module):
    """
    Long-curve Bezier generator.

    Important difference from the old AnchorSVGGenerator:
    old: P0/P1/P2/P3 = anchor + small independent offsets -> local small bent curves.
    new: P0 = anchor, P3 = anchor + direction * length,
         P1/P2 lie along the same direction with limited normal bend.

    This forces each predicted curve to behave like a long hair line instead of a tiny local arc.
    """
    def __init__(self, in_ch=5, hidden=256, min_len=0.10, max_len=0.55, max_bend=0.08):
        super().__init__()
        self.min_len = min_len
        self.max_len = max_len
        self.max_bend = max_bend

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )

        self.anchor_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # angle, length_raw, bend1, bend2, reverse_logit
        self.curve_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 5),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.width_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, anchors):
        feat = self.encoder(x)
        memory = feat.flatten(2).transpose(1, 2)

        q = self.anchor_mlp(anchors)
        h = self.decoder(q, memory)

        curve = self.curve_head(h)

        # Full 360-degree direction.
        angle = curve[..., 0] * np.pi
        direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)

        # Allow flipping direction so the model can choose either way from the anchor.
        reverse = torch.tanh(curve[..., 4])[..., None]
        direction = direction * torch.sign(reverse + 1e-6)

        length = self.min_len + (self.max_len - self.min_len) * torch.sigmoid(curve[..., 1])
        bend1 = torch.tanh(curve[..., 2]) * self.max_bend
        bend2 = torch.tanh(curve[..., 3]) * self.max_bend

        normal = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)

        p0 = anchors
        p1 = anchors + direction * (length[..., None] * 0.33) + normal * bend1[..., None]
        p2 = anchors + direction * (length[..., None] * 0.66) + normal * bend2[..., None]
        p3 = anchors + direction * length[..., None]

        points = torch.stack([p0, p1, p2, p3], dim=2).clamp(0.0, 1.0)

        conf = torch.sigmoid(self.conf_head(h)).squeeze(-1)
        width = 1.0 + 2.0 * torch.sigmoid(self.width_head(h)).squeeze(-1)
        return {"points": points, "conf": conf, "width": width}


def load_stage1(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = HairStructUNet(in_channels=4, base_ch=32, out_channels=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def infer_center(stage1, image, mask):
    x = torch.cat([image, mask], dim=1)
    pred = stage1(x)
    center = pred["center"] * mask
    return center.clamp(0.0, 1.0)


def gaussian_blur(x, k=15, sigma=3.0):
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, k, k)
    return F.conv2d(x, kernel, padding=k // 2)


def blur_mse_loss(pred, target, k=15, sigma=3.0):
    return F.mse_loss(gaussian_blur(pred, k, sigma), gaussian_blur(target, k, sigma))


def render_one_sample(points, conf, width, canvas_size, device):
    import pydiffvg

    h = w = canvas_size
    shapes = []
    shape_groups = []

    pts_px = points.clone()
    pts_px[..., 0] = pts_px[..., 0] * (w - 1)
    pts_px[..., 1] = pts_px[..., 1] * (h - 1)

    for i in range(points.shape[0]):
        p = pts_px[i]
        path_points = torch.stack([p[0], p[1], p[2], p[3]], dim=0)
        path = pydiffvg.Path(
            num_control_points=torch.tensor([2], dtype=torch.int32, device=device),
            points=path_points,
            stroke_width=width[i],
            is_closed=False,
        )
        shapes.append(path)

        color = torch.stack([
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            conf[i].clamp(0.0, 1.0),
        ])
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i], dtype=torch.int32, device=device),
            fill_color=None,
            stroke_color=color,
        )
        shape_groups.append(group)

    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img = pydiffvg.RenderFunction.apply(w, h, 2, 2, 0, None, *scene_args)
    return img[:, :, 3].clamp(0.0, 1.0)


def render_batch(pred, size, device):
    masks = []
    for i in range(pred["points"].shape[0]):
        alpha = render_one_sample(
            pred["points"][i], pred["conf"][i], pred["width"][i], size, device
        )
        masks.append(alpha[None, None])
    return torch.cat(masks, dim=0)


def outside_mask_loss(render, mask):
    return (render * (1.0 - mask)).mean()


def bezier_sample(points, t_num=16):
    # points: [B, N, 4, 2]
    t = torch.linspace(0.0, 1.0, t_num, device=points.device, dtype=points.dtype)
    t = t.view(1, 1, t_num, 1)
    p0 = points[:, :, 0:1]
    p1 = points[:, :, 1:2]
    p2 = points[:, :, 2:3]
    p3 = points[:, :, 3:4]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def curve_length_regularization(points, target_min=0.12):
    pts = bezier_sample(points, t_num=16)
    seg = pts[:, :, 1:] - pts[:, :, :-1]
    length = torch.sqrt((seg ** 2).sum(dim=-1) + 1e-6).sum(dim=-1)
    return F.relu(target_min - length).mean()


def curve_bend_regularization(points):
    pts = bezier_sample(points, t_num=16)
    d1 = pts[:, :, 1:] - pts[:, :, :-1]
    d2 = d1[:, :, 1:] - d1[:, :, :-1]
    return (d2 ** 2).mean()


@torch.no_grad()
def save_svg_prediction(pred, out_svg, size, conf_thr=0.15):
    points = pred["points"][0].detach().cpu().numpy()
    conf = pred["conf"][0].detach().cpu().numpy()
    width = pred["width"][0].detach().cpu().numpy()
    keep = conf >= conf_thr
    if not np.any(keep):
        # The confidence head has no direct supervision in this training setup
        # (lambda_conf defaults to zero), so its scores can be uniformly low.
        keep = np.ones_like(conf, dtype=bool)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    kept = 0
    for i in range(points.shape[0]):
        if not keep[i]:
            continue
        p = points[i].copy()
        p[:, 0] *= size - 1
        p[:, 1] *= size - 1
        d = (
            f"M {p[0,0]:.2f} {p[0,1]:.2f} "
            f"C {p[1,0]:.2f} {p[1,1]:.2f}, "
            f"{p[2,0]:.2f} {p[2,1]:.2f}, "
            f"{p[3,0]:.2f} {p[3,1]:.2f}"
        )
        lines.append(
            f'<path d="{d}" fill="none" stroke="#000000" '
            f'stroke-width="{float(width[i]):.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        kept += 1
    lines.append("</svg>")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return kept


@torch.no_grad()
def save_preview(stage1, model, loader, args, device, epoch):
    stage1.eval()
    model.eval()

    batch = next(iter(loader))
    image = batch["image"].to(device)
    mask = batch["mask"].to(device)
    gt_svg = batch["gt_svg"].to(device)
    anchors = batch["anchors"].to(device)
    stems = batch["stem"]

    center = infer_center(stage1, image, mask)
    inp = torch.cat([image, mask, center], dim=1)
    pred = model(inp, anchors)
    render = render_batch(pred, args.size, device)

    out_dir = Path(args.output_dir) / "previews" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.preview_num, image.shape[0])):
        stem = stems[i]
        save_mask(center[i, 0].cpu().numpy(), out_dir / f"{stem}_center.png")
        save_mask(gt_svg[i, 0].cpu().numpy(), out_dir / f"{stem}_gt_svg.png")
        save_mask(render[i, 0].cpu().numpy(), out_dir / f"{stem}_pred_raster.png")
        one_pred = {
            "points": pred["points"][i:i + 1],
            "conf": pred["conf"][i:i + 1],
            "width": pred["width"][i:i + 1],
        }
        kept = save_svg_prediction(one_pred, out_dir / f"{stem}_pred.svg", args.size, args.conf_thr)
        print(f"preview {stem}: kept {kept} curves")


def train_one_epoch(stage1, model, loader, optimizer, args, device, epoch):
    stage1.eval()
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        gt_svg = batch["gt_svg"].to(device, non_blocking=True)
        anchors = batch["anchors"].to(device, non_blocking=True)

        with torch.no_grad():
            center = infer_center(stage1, image, mask)

        inp = torch.cat([image, mask, center], dim=1)
        pred = model(inp, anchors)
        render = render_batch(pred, args.size, device)

        loss_gt = blur_mse_loss(render, gt_svg, args.blur_k, args.blur_sigma)
        loss_center = blur_mse_loss(render, center, args.blur_k, args.blur_sigma)
        loss_out = outside_mask_loss(render, mask)
        loss_long = curve_length_regularization(pred["points"], args.target_min_len)
        loss_bend = curve_bend_regularization(pred["points"])
        loss_conf = pred["conf"].mean()

        loss = (
            args.lambda_gt * loss_gt
            + args.lambda_center * loss_center
            + args.lambda_outside * loss_out
            + args.lambda_length * loss_long
            + args.lambda_bend * loss_bend
            + args.lambda_conf * loss_conf
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += float(loss.detach().cpu())
        pbar.set_postfix(
            loss=f"{float(loss):.5f}",
            gt=f"{float(loss_gt):.5f}",
            center=f"{float(loss_center):.5f}",
            out=f"{float(loss_out):.5f}",
            long=f"{float(loss_long):.5f}",
            conf=f"{float(pred['conf'].mean()):.3f}/{float(pred['conf'].max()):.3f}",
        )

    return total / max(len(loader), 1)


def main():
    import pydiffvg

    parser = argparse.ArgumentParser()

    parser.add_argument("--stage1-checkpoint", type=str, default=str(PROJECT_ROOT / "weights/stage1_last.pt"))
    parser.add_argument("--train-image-dir", type=str, default=str(PROJECT_ROOT / "data/raw/train/images"))
    parser.add_argument("--train-mask-dir", type=str, default=str(PROJECT_ROOT / "data/raw/train/masks"))
    parser.add_argument("--train-svg-dir", type=str, default=str(PROJECT_ROOT / "data/raw/train/svg"))
    parser.add_argument("--val-image-dir", type=str, default=str(PROJECT_ROOT / "data/raw/val/images"))
    parser.add_argument("--val-mask-dir", type=str, default=str(PROJECT_ROOT / "data/raw/val/masks"))
    parser.add_argument("--val-svg-dir", type=str, default=str(PROJECT_ROOT / "data/raw/val/svg"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "runs/long_curve"))

    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--num-strokes", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--anchor-thr", type=float, default=0.15)

    # Long-curve constraints.
    parser.add_argument("--min-len", type=float, default=0.10)
    parser.add_argument("--max-len", type=float, default=0.55)
    parser.add_argument("--max-bend", type=float, default=0.08)
    parser.add_argument("--target-min-len", type=float, default=0.12)

    parser.add_argument("--blur-k", type=int, default=15)
    parser.add_argument("--blur-sigma", type=float, default=3.0)

    parser.add_argument("--lambda-gt", type=float, default=1.0)
    parser.add_argument("--lambda-center", type=float, default=1.2)
    parser.add_argument("--lambda-outside", type=float, default=2.0)
    parser.add_argument("--lambda-length", type=float, default=0.08)
    parser.add_argument("--lambda-bend", type=float, default=0.02)
    parser.add_argument("--lambda-conf", type=float, default=0.0)

    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--preview-num", type=int, default=4)
    parser.add_argument("--conf-thr", type=float, default=0.15)

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(Path(args.output_dir) / "checkpoints")
    ensure_dir(Path(args.output_dir) / "previews")

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    pydiffvg.set_use_gpu(device != "cpu")

    train_ds = HairSVGLongCurveDataset(
        args.train_image_dir,
        args.train_mask_dir,
        args.train_svg_dir,
        args.size,
        args.num_strokes,
        args.anchor_thr,
    )
    val_ds = HairSVGLongCurveDataset(
        args.val_image_dir,
        args.val_mask_dir,
        args.val_svg_dir,
        args.size,
        args.num_strokes,
        args.anchor_thr,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
        drop_last=False,
    )

    stage1 = load_stage1(args.stage1_checkpoint, device)
    for p in stage1.parameters():
        p.requires_grad_(False)

    model = LongCurveSVGGenerator(
        in_ch=5,
        hidden=args.hidden,
        min_len=args.min_len,
        max_len=args.max_len,
        max_bend=args.max_bend,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(stage1, model, train_loader, optimizer, args, device, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, Path(args.output_dir) / "checkpoints" / "last.pt")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(ckpt, Path(args.output_dir) / "checkpoints" / "best.pt")
            print(f"Saved best: {best_loss:.6f}")

        if epoch % args.save_every == 0:
            torch.save(ckpt, Path(args.output_dir) / "checkpoints" / f"epoch_{epoch:03d}.pt")

        if epoch % args.preview_every == 0:
            save_preview(stage1, model, val_loader, args, device, epoch)


if __name__ == "__main__":
    main()
