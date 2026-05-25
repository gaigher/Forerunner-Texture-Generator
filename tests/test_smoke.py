from PIL import Image
import os
from pathlib import Path
from effets.traitement import process_image

def test_process_runs(tmp_path):
    p = tmp_path / "in.png"
    out = tmp_path / "out.png"
    Image.new('RGBA',(64,64),(128,128,128,255)).save(p)
    opts = {
        'niveaux': 0,
        'glow': {'enabled': False},
        'output_path': str(out)
    }
    result_path = process_image(str(p), opts)
    assert Path(result_path).exists()


def test_seamless_metal_texture():
    from ui.metal_procedural import make_seamless_metal_texture

    im = make_seamless_metal_texture(64, 48, seed=3)
    assert im.size == (64, 48)
    assert im.mode == "RGB"


def test_prepare_metal_texture_rgba_modes():
    from effets.utilitaires import (
        prepare_metal_texture_rgba,
        TM_FIT_ANISOTROPIC,
        TM_FIT_UNIFORM_COVER,
        TM_FIT_TILE,
        TM_FIT_TILE_APPROX_STRETCH,
    )

    tm = Image.new("RGBA", (32, 16), (200, 50, 50, 255))
    tw, th = 100, 40
    for mode in (
        TM_FIT_ANISOTROPIC,
        TM_FIT_UNIFORM_COVER,
        TM_FIT_TILE,
        TM_FIT_TILE_APPROX_STRETCH,
    ):
        out = prepare_metal_texture_rgba(tm, tw, th, mode)
        assert out.size == (tw, th)
        assert out.mode == "RGBA"
