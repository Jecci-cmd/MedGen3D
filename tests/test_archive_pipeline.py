from __future__ import annotations

import hashlib
import io
import sys
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medicalmodel_data.layout import ensure_layout
from scripts.prepare_abdomenatlas import (
    command_extract,
    command_verify_archives,
    discover_cases,
    safe_extract,
)


def test_archive_verification_and_staged_extraction(tmp_path: Path) -> None:
    paths = ensure_layout(tmp_path / "dataset")
    source = tmp_path / "source"
    for number in (1, 2):
        case = source / f"BDMAP_{number:08d}"
        case.mkdir(parents=True)
        image = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4))
        nib.save(image, case / "ct.nii.gz")
        nib.save(image, case / "combined_labels.nii.gz")
    archive_name = "cases.tar.gz"
    archive = paths["raw/archives"] / archive_name
    with tarfile.open(archive, "w:gz") as handle:
        for case in sorted(source.iterdir()):
            handle.add(case, arcname=case.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    config = {
        "dataset": {
            "expected_cases": 2,
            "archives": {
                archive_name: {
                    "bytes": archive.stat().st_size,
                    "etag_sha256": digest,
                    "case_start": 1,
                    "case_end": 2,
                }
            },
        }
    }
    command_verify_archives(config, paths)
    command_extract(config, paths, archives=None)
    assert [item[0] for item in discover_cases(paths["raw/extracted"])] == [
        "BDMAP_00000001",
        "BDMAP_00000002",
    ]
    command_extract(config, paths, archives=None)
    tampered = paths["raw/extracted"] / "BDMAP_00000001" / "ct.nii.gz"
    tampered.write_bytes(tampered.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="differs from verified archive"):
        command_extract(config, paths, archives=None)


def test_safe_extract_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("BDMAP_00000001/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        handle.addfile(member, io.BytesIO())
    with pytest.raises(RuntimeError, match="special/link"):
        safe_extract(archive, tmp_path / "output")
