# SPDX-License-Identifier: BSD-3-Clause

"""Pin the copied environment to its published source commit."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHA = "7186e70e9e0fe74c78a56b87f40d17a7c6698271"
HASHES = {
    "models.py": "4ab5a8b10c9df52523f116fac6082bbeb5b407ee921f82131c73ac202b1d0765",
    "server/reasoning_gym_environment.py": "171910d6a878cca11c085344c56dee8bd2501e1ff09310b45d40f2c732414b9e",
}


class TestVendor:
    def test_hashes(self) -> None:
        for name, digest in HASHES.items():
            assert (
                hashlib.sha256(
                    (ROOT / "trial/vendor/reasoning_gym_env" / name).read_bytes()
                ).hexdigest()
                == digest
            )

    def test_source_commit(self) -> None:
        clone = ROOT.parent / "OpenEnv"
        message = f"needs an OpenEnv clone that has commit {SHA}"
        if not clone.exists():
            pytest.skip(message)
        probe = subprocess.run(
            ["git", "-C", str(clone), "show", "--format=", "--no-patch", SHA],
            capture_output=True,
        )
        if probe.returncode:
            pytest.skip(message)
        for name in HASHES:
            source = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(clone),
                    "show",
                    f"{SHA}:envs/reasoning_gym_env/{name}",
                ]
            )
            assert (
                ROOT / "trial/vendor/reasoning_gym_env" / name
            ).read_bytes() == source
