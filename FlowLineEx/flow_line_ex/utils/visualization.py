import os
import numpy as np
from PIL import Image


def save_mask(mask: np.ndarray, path: str) -> None:
    m = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(m).save(path)


def orientation_to_rgb(orient: np.ndarray, mag: np.ndarray | None = None) -> np.ndarray:
    # orient: HxWx2 in [-1, 1]
    x = orient[..., 0]
    y = orient[..., 1]
    angle = np.arctan2(y, x)
    hue = (angle + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = np.ones_like(hue) if mag is None else np.clip(mag, 0, 1)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i = i % 6
    conditions = [i == k for k in range(6)]
    rgb = np.zeros(hsv.shape, dtype=np.float32)
    rgb[conditions[0]] = np.stack([v, t, p], axis=-1)[conditions[0]]
    rgb[conditions[1]] = np.stack([q, v, p], axis=-1)[conditions[1]]
    rgb[conditions[2]] = np.stack([p, v, t], axis=-1)[conditions[2]]
    rgb[conditions[3]] = np.stack([p, q, v], axis=-1)[conditions[3]]
    rgb[conditions[4]] = np.stack([t, p, v], axis=-1)[conditions[4]]
    rgb[conditions[5]] = np.stack([v, p, q], axis=-1)[conditions[5]]
    return rgb


def save_rgb(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)
