"""
Texture métallique procédurale carrelable (sans raccord visible aux bords).

Le bruit est défini sur un tore (grille interpolée avec indices modulo),
donc l'image se répète sans couture.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def torus_value_noise(
    height: int,
    width: int,
    grid_cells: int,
    seed: int,
) -> np.ndarray:
    """Bruit lissé périodique de période (height, width)."""
    rng = np.random.default_rng(seed)
    g = rng.random((grid_cells, grid_cells), dtype=np.float64)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    u = (xx / float(width)) * grid_cells
    v = (yy / float(height)) * grid_cells
    j0 = (np.floor(u).astype(np.int64) % grid_cells)
    j1 = (j0 + 1) % grid_cells
    i0 = (np.floor(v).astype(np.int64) % grid_cells)
    i1 = (i0 + 1) % grid_cells
    tu = _smoothstep(u - np.floor(u))
    tv = _smoothstep(v - np.floor(v))
    a = g[i0, j0]
    b = g[i0, j1]
    c = g[i1, j0]
    d = g[i1, j1]
    top = a * (1.0 - tu) + b * tu
    bot = c * (1.0 - tu) + d * tu
    return (top * (1.0 - tv) + bot * tv).astype(np.float32)


def _torus_gradients(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients avec wrap torique (échantillonnage symétrique)."""
    d_dy = 0.5 * (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0))
    d_dx = 0.5 * (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1))
    return d_dx.astype(np.float32), d_dy.astype(np.float32)


def make_seamless_metal_texture(
    width: int,
    height: int,
    seed: int = 0,
) -> Image.Image:
    """
    Génère une image RGB carrelable.

    ``width`` et ``height`` doivent être au moins 8 pour l'interpolation.
    """
    if width < 8 or height < 8:
        raise ValueError("Les dimensions minimales sont 8×8 px.")

    macro = torus_value_noise(height, width, 14, seed)
    meso = torus_value_noise(height, width, 28, seed + 101)
    fine = torus_value_noise(height, width, 9, seed + 303)
    hmap = (0.52 * macro + 0.33 * meso + 0.15 * fine).astype(np.float32)
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

    d_dx, d_dy = _torus_gradients(hmap)
    nx = -d_dx
    ny = -d_dy
    nz = np.ones_like(hmap, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-8)
    nx *= inv_len
    ny *= inv_len
    nz *= inv_len
    lx, ly, lz = 0.45, -0.32, 0.83
    shade = nx * lx + ny * ly + nz * lz
    shade = np.clip(shade, 0.0, 1.0)

    grain = torus_value_noise(height, width, 56, seed + 707)
    grain = (grain - 0.5) * 14.0

    r = 158.0 + 48.0 * hmap * shade + grain
    g = 162.0 + 42.0 * hmap * shade + grain * 0.95
    b = 172.0 + 38.0 * hmap * shade + grain * 1.05
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    img = Image.fromarray(rgb, "RGB")
    img = ImageEnhance.Contrast(img).enhance(1.07)
    return img
