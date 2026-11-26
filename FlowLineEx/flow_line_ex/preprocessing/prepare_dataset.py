import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


def load_rgb(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def load_mask(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert('L').resize((size, size), Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


def load_line(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert('L').resize((size, size), Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0

    # 默认兼容：黑线白底 -> 反相后让线=1
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    line = (arr > 0.5).astype(np.float32)
    return line


def build_skeleton(line: np.ndarray, mask: np.ndarray) -> np.ndarray:
    line_bin = (line > 0.5).astype(np.uint8)

    # 闭运算，修一点小断裂
    line_bin = cv2.morphologyEx(
        line_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    skel = skeletonize(line_bin > 0).astype(np.float32)
    skel = skel * mask.astype(np.float32)
    return skel


def build_centerline_from_skeleton(skel: np.ndarray, mask: np.ndarray, dilate_iter: int = 1) -> np.ndarray:
    skel_bin = (skel > 0).astype(np.uint8)

    # 轻微膨胀，让监督别太稀疏
    if dilate_iter > 0:
        skel_bin = cv2.dilate(
            skel_bin, np.ones((3, 3), np.uint8), iterations=dilate_iter
        )

    center = skel_bin.astype(np.float32) * mask.astype(np.float32)
    return center


def build_orientation(skeleton: np.ndarray) -> np.ndarray:
    line = (skeleton > 0).astype(np.uint8) * 255
    blur = cv2.GaussianBlur(line, (0, 0), sigmaX=1.0, sigmaY=1.0)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

    # 梯度法线 -> 切线
    tangent_x = -gy
    tangent_y = gx
    mag = np.sqrt(tangent_x ** 2 + tangent_y ** 2) + 1e-6

    tangent_x = tangent_x / mag
    tangent_y = tangent_y / mag

    orient = np.stack([tangent_x, tangent_y], axis=-1).astype(np.float32)
    orient[skeleton <= 0] = 0.0
    return orient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--dilate-iter', type=int, default=1)
    args = parser.parse_args()

    img_dir = os.path.join(args.raw_root, 'images')
    mask_dir = os.path.join(args.raw_root, 'masks')
    line_dir = os.path.join(args.raw_root, 'lines')
    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(img_dir, '*')))
    if not image_paths:
        raise FileNotFoundError(f'No images found in {img_dir}')

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = None
        line_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
            mp = os.path.join(mask_dir, stem + ext)
            lp = os.path.join(line_dir, stem + ext)
            if mask_path is None and os.path.exists(mp):
                mask_path = mp
            if line_path is None and os.path.exists(lp):
                line_path = lp
        if mask_path is None or line_path is None:
            print(f'Skip {stem}: missing mask or line')
            continue

        image = load_rgb(img_path, args.size)
        mask = load_mask(mask_path, args.size)
        line = load_line(line_path, args.size) * mask

        skeleton = build_skeleton(line, mask)
        center = build_centerline_from_skeleton(
            skeleton, mask, dilate_iter=args.dilate_iter
        )
        orient = build_orientation(skeleton)

        out_path = os.path.join(args.out_dir, f'{stem}.npz')
        np.savez_compressed(
            out_path,
            image=image.astype(np.float32),
            mask=mask.astype(np.float32),
            line_gt=line.astype(np.float32),
            skeleton_gt=skeleton.astype(np.float32),
            centerline_gt=center.astype(np.float32),
            orientation_gt=orient.astype(np.float32),
        )
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()