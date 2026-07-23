from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_cord_copy_move import forge_copy_move


def test_copy_move_is_deterministic_and_masked(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    pixels = np.arange(96 * 64 * 3, dtype=np.uint8).reshape(64, 96, 3)
    Image.fromarray(pixels).save(source)
    first_image = tmp_path / "first.png"
    first_mask = tmp_path / "first-mask.png"
    second_image = tmp_path / "second.png"
    second_mask = tmp_path / "second-mask.png"
    forge_copy_move(source, first_image, first_mask, "fixture")
    forge_copy_move(source, second_image, second_mask, "fixture")
    assert first_image.read_bytes() == second_image.read_bytes()
    assert first_mask.read_bytes() == second_mask.read_bytes()
    with Image.open(first_mask) as mask:
        values = np.asarray(mask)
    assert set(np.unique(values)) == {0, 255}
    assert 0 < np.count_nonzero(values) < values.size
