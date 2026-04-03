"""Unit tests for pattern bbox margin handling."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from migrama.cli.main import app
from migrama.core.pattern import DetectorParameters, PatternDetector
from migrama.core.pattern.source import PatternFovSource


class ArrayPatternSource(PatternFovSource):
    """Minimal in-memory pattern source for unit tests."""

    def __init__(self, frames: dict[int, np.ndarray]) -> None:
        self._frames = frames

    @property
    def n_fovs(self) -> int:
        """Return the number of FOVs."""
        return len(self._frames)

    def iter_fovs(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield stored FOVs in sorted order."""
        yield from sorted(self._frames.items())


def test_detector_uses_zero_bbox_margin_by_default() -> None:
    """Default detector output should match the raw contour bbox."""
    image = np.zeros((40, 50), dtype=np.uint8)
    image[10:20, 15:25] = 255

    source = ArrayPatternSource({0: image})
    detector = PatternDetector(
        source=source,
        parameters=DetectorParameters(
            gaussian_blur_size=(1, 1),
            morph_dilate_size=(1, 1),
            edge_tolerance=0,
        ),
    )

    records = detector.detect_fov(0)

    assert len(records) == 1
    assert (records[0].x, records[0].y, records[0].w, records[0].h) == (15, 10, 10, 10)


def test_detector_expands_bbox_and_clips_to_image() -> None:
    """Configured bbox margin should expand each side without exceeding image bounds."""
    image = np.zeros((12, 12), dtype=np.uint8)
    image[1:6, 2:7] = 255

    source = ArrayPatternSource({0: image})
    detector = PatternDetector(
        source=source,
        parameters=DetectorParameters(
            gaussian_blur_size=(1, 1),
            morph_dilate_size=(1, 1),
            edge_tolerance=0,
            bbox_margin=4,
        ),
    )

    records = detector.detect_fov(0)

    assert len(records) == 1
    assert (records[0].x, records[0].y, records[0].w, records[0].h) == (0, 0, 11, 10)


def test_pattern_cli_passes_margin(monkeypatch, tmp_path: Path) -> None:
    """CLI --margin option should be forwarded into detector parameters."""
    captured: dict[str, int] = {}

    class FakeDetector:
        def __init__(self, source, parameters=None) -> None:
            del source
            captured["bbox_margin"] = parameters.bbox_margin
            self.n_fovs = 1

        def detect_fov(self, fov_idx: int) -> list[object]:
            del fov_idx
            return []

        def detect_all(self, fov_filter=None) -> list[object]:
            del fov_filter
            return []

        def save_csv(self, records: list[object], output_path: str) -> None:
            del records
            Path(output_path).write_text("cell,fov,x,y,w,h\n", encoding="utf-8")

    class FakeSource:
        n_fovs = 1

    import migrama.core.pattern as pattern_module
    import migrama.core.pattern.source as source_module

    monkeypatch.setattr(source_module, "Nd2PatternFovSource", lambda path: FakeSource())
    monkeypatch.setattr(pattern_module, "PatternDetector", FakeDetector)

    runner = CliRunner()
    patterns_path = tmp_path / "patterns.nd2"
    output_path = tmp_path / "patterns.csv"
    patterns_path.write_text("placeholder", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "pattern",
            "--patterns",
            str(patterns_path),
            "--output",
            str(output_path),
            "--fovs",
            "0",
            "--margin",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert captured["bbox_margin"] == 7
