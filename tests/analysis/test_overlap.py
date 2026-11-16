# SPDX-License-Identifier: BSD-3-Clause

"""Question overlap on the reference experiment's two task numberings."""

from __future__ import annotations

import pytest

pytest.importorskip("reasoning_gym")

from analysis.overlap import main, overlap


class TestOverlap:
    @pytest.mark.parametrize(
        "ladder,shared,distinct",
        [
            ("rfc", [100, 47, 0, 0, 0, 0, 0, 0, 0, 0], [75, 99] + [100] * 8),
            ("staircase", [100, 38, 1, 0, 0, 0, 0, 0, 0, 0], [76, 98] + [100] * 8),
        ],
    )
    def test_reference_questions(
        self, ladder: str, shared: list[int], distinct: list[int]
    ) -> None:
        rows = overlap(ladder)
        assert [row["level"] for row in rows] == list(range(10))
        assert [row["overlap"] for row in rows] == shared
        assert [row["distinct_test"] for row in rows] == distinct

    def test_cli(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--ladder", "rfc"])
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 10
        assert lines[0] == "level 0: overlap=100, distinct_test=75"
        assert lines[-1] == "level 9: overlap=0, distinct_test=100"

    def test_cli_refuses_unknown_ladder(self) -> None:
        with pytest.raises(SystemExit, match="2"):
            main(["--ladder", "unknown"])
