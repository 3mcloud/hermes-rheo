"""Tests for reader tolerance to missing / unknown instrument serial numbers.

The instrument serial number is used by ``TriosRheoReader`` only to select
the correct OWChirp wave-metadata parser (Arbitrary Wave layout differs
between DHR and ARES G2 / G3 / Discovery HR).  Non-OWChirp measurements
do not need it, so a missing or unknown serial must never break parsing
of e.g. Creep, Frequency Sweep, Flow Ramp or Temperature Ramp files.

Only when :class:`OWChirpAnalysis` is applied to an Arbitrary Wave step
that has no wave metadata should a targeted error be raised.
"""

import os
import shutil

import pytest

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis


def _copy_and_patch_serial(datadir, src_name: str, replacement: str) -> str:
    """Copy a fixture file, rewriting its ``Instrument serial number`` line."""
    src = datadir.join(src_name)
    dst = datadir.join(f"_patched_{src_name}")
    with open(str(src), "r") as f:
        contents = f.read()
    new_lines = []
    for line in contents.splitlines():
        if line.startswith("Instrument serial number"):
            new_lines.append(f"Instrument serial number\t{replacement}")
        else:
            new_lines.append(line)
    with open(str(dst), "w") as f:
        f.write("\n".join(new_lines))
    return str(dst)


def test_reader_accepts_missing_serial_number(datadir):
    """A Creep file with an empty serial-number line must still parse."""
    filepath = _copy_and_patch_serial(datadir, "test_creep_apac.txt", "")
    experiment = TriosRheoReader().data_from_filepath(
        filepath,
        create_composite_datasets=True,
    )
    assert len(experiment.measurements) > 0


def test_reader_accepts_unknown_serial_prefix(datadir):
    """A file with an unrecognised serial prefix must still parse."""
    filepath = _copy_and_patch_serial(
        datadir,
        "test_creep_apac.txt",
        "9999-0000",
    )
    experiment = TriosRheoReader().data_from_filepath(
        filepath,
        create_composite_datasets=True,
    )
    assert len(experiment.measurements) > 0


def test_owchirp_raises_when_wave_metadata_missing(datadir):
    """``OWChirpAnalysis`` must raise a targeted error on Arbitrary Wave
    steps whose wave metadata could not be extracted (unknown serial)."""
    filepath = _copy_and_patch_serial(
        datadir,
        "test_owchirp_DHR.txt",
        "9999-0000",
    )
    experiment = TriosRheoReader().data_from_filepath(
        filepath,
        create_composite_datasets=True,
    )
    with pytest.raises(ValueError, match="[Ww]ave metadata"):
        OWChirpAnalysis()(experiment)
