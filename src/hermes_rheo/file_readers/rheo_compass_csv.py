"""
Reader for data exported from Rheo Compass software in .csv format.

Classes
-------
RheoCompassReader - Reader for files in the RheoCompass .csv format.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Tuple, Union
from io import StringIO
import numpy as np
import codecs

from piblin.dataio.fileio.read import FileParsingException
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset
from piblin.data.data_collections import measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.dataio.fileio.read.file_reader as file_reader


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ordered per-measurement default-coordinate rules
# ---------------------------------------------------------------------------
#
# RheoCompass method labels are highly free-form (e.g.
# ``"Interval 1 (Fs - 100.11)"`` for a frequency sweep at 100.11 °C, or
# ``"Interval 1 (822.884 - s)"`` for a stress relaxation).  Rather than
# parsing the label we detect the measurement type from the column set:
# whichever rule below has *all* of its required columns present wins.
#
# Each rule is ``(required_columns, default_independent, default_dependent)``.
# Order matters — earlier rules win.  The final fallback ``(time, shear
# stress)`` matches the historical default and applies to genuinely
# ambiguous cases (e.g. creep vs stress relaxation cannot be told apart
# from the columns alone).
_DEFAULT_COORD_RULES: List[Tuple[Tuple[str, ...], str, str]] = [
    # Oscillatory sweep (frequency, amplitude, TTS)
    (("angular frequency", "storage modulus"), "angular frequency", "storage modulus"),
    # Rotational / flow sweep
    (("shear rate", "viscosity"), "shear rate", "viscosity"),
    # Fallback: matches almost every transient/step measurement
    (("time", "shear stress"), "time", "shear stress"),
]


def _lookup_rheocompass_defaults(
    variable_names: List[str],
    fallback_independent: str = "time",
) -> Tuple[str, str]:
    """Choose ``(default_x, default_y)`` for a RheoCompass measurement.

    Walks :data:`_DEFAULT_COORD_RULES` in order and returns the first rule
    whose required columns are all present in *variable_names*.  When no
    rule matches, falls back to ``(fallback_independent, first_other_column)``.

    Parameters
    ----------
    variable_names : list of str
        Names of the data-array columns of the composite dataset.
    fallback_independent : str, optional
        Column used as independent when no rule matches.  Default ``"time"``.

    Returns
    -------
    default_x : str
    default_y : str
    """
    for required, default_x, default_y in _DEFAULT_COORD_RULES:
        if all(col in variable_names for col in required):
            return default_x, default_y

    # Final safety net
    _logger.debug(
        "No RheoCompass default-coord rule matched columns %r; falling back.",
        variable_names,
    )
    default_y = next(
        (n for n in variable_names if n != fallback_independent),
        variable_names[0],
    )
    return fallback_independent, default_y


class RheoCompassReader(file_reader.FileReader):
    """Reader for files in the RheoCompass .csv format"""

    supported_extensions = {"csv"}

    @property
    def default_mode(self) -> str:
        return "r"

    def _parse_metadata(self, lines: List[str]) -> Dict[str, Union[str, List[str]]]:
        metadata, current_key, buf = {}, None, []
        for ln in lines:
            if not ln.strip():
                continue
            if ":" in ln and not ln.startswith("\t"):
                if current_key:
                    metadata[current_key] = "\n".join(buf)
                    buf = []
                key, val = ln.split(":", 1)
                current_key = key.strip()
                buf.append(val.strip())
            else:
                buf.append(ln.strip())
        if current_key:
            metadata[current_key] = "\n".join(buf)
        return metadata

    def _read_file_contents(self, filepath: str, **read_kwargs):
        delimiter = read_kwargs.get("delimiter", "\t")

        with codecs.open(filepath, mode="r", encoding="utf-16") as f:
            lines = f.readlines()

        first_interval = None
        for i, ln in enumerate(lines):
            if ln.lstrip().lower().startswith("interval and data points"):
                first_interval = i
                break
        if first_interval is None:
            raise FileParsingException(
                "No 'Interval and data points:' blocks found in the .csv file."
            )

        global_metadata = self._parse_metadata([ln.rstrip("\n") for ln in lines[:first_interval]])

        interval_positions = [
            i
            for i, ln in enumerate(lines)
            if ln.lstrip().lower().startswith("interval and data points")
        ]

        result_positions = []
        for idx, ln in enumerate(lines):
            text = ln.lstrip()
            if text.lower().startswith("result"):
                parts = text.rstrip("\n").split(":", 1)
                val = parts[1].strip() if len(parts) > 1 else ""
                result_positions.append((idx, val))

        intervals = []
        for idx in interval_positions:
            prev = [val for pos, val in result_positions if pos < idx]
            current_result = prev[-1] if prev else ""

            interval_line = lines[idx].rstrip("\n")
            interval_fields = interval_line.split(delimiter)
            interval_id = interval_fields[1].strip()
            n_points = int(interval_fields[2].strip())

            header_idx = next(
                (
                    j
                    for j in range(idx + 1, len(lines))
                    if lines[j].lstrip().lower().startswith("interval data")
                ),
                None,
            )
            if header_idx is None:
                raise FileParsingException(f"Missing 'Interval data:' header after line {idx}")

            header_tokens = lines[header_idx].rstrip("\n").split(delimiter)[2:]
            header_tokens = [h.strip() for h in header_tokens if h.strip()]

            units_fields = lines[header_idx + 2].rstrip("\n").split(delimiter)[2:]
            units_fields = [u.strip() for u in units_fields[: len(header_tokens)]]

            # end of block = next interval or next result
            end_candidates = [p for p in interval_positions if p > idx] + [
                p for p, _ in result_positions if p > idx
            ]
            end_idx = min(end_candidates) if end_candidates else len(lines)

            # data starts after units line
            data_rows: List[List[str]] = []
            data_start = header_idx + 3

            for j in range(data_start, end_idx):
                line = lines[j].rstrip("\n")
                if not line.strip():
                    continue
                low = line.lstrip().lower()
                if low.startswith(("interval and data points", "result", "interval data")):
                    break

                tokens = line.split(delimiter)
                if len(tokens) >= len(header_tokens) + 2:
                    data_rows.append(tokens[2 : 2 + len(header_tokens)])

            data_str = f"{delimiter.join(header_tokens)}\n" + "\n".join(
                delimiter.join(row) for row in data_rows
            )
            buf = StringIO(data_str)

            block_data = np.genfromtxt(
                buf,
                delimiter=delimiter,
                names=True,
                autostrip=True,
                dtype=float,
                invalid_raise=False,
            )

            units_dict = {}
            for headers, units in zip(header_tokens, units_fields):
                key = headers.replace(" ", "_")
                units_dict[key] = units

            intervals.append(
                {
                    "interval": f"Interval {interval_id}",
                    "points": n_points,
                    "data": block_data,
                    "result": current_result,
                    "units": units_dict,
                }
            )

        if not intervals:
            raise FileParsingException("No intervals were parsed.")

        return {"metadata": global_metadata, "intervals": intervals}

    @classmethod
    def _data_from_file_contents(cls, file_contents, file_location=None, **kwargs):
        metadata = file_contents["metadata"]
        intervals = file_contents["intervals"]

        measurements = []
        for seg in intervals:
            raw = seg["data"]
            orig_names = list(raw.dtype.names)

            # only requirement: lowercase display names
            clean_names = [n.replace("_", " ").lower() for n in orig_names]

            # units dict keyed by dtype names
            clean_units = [seg["units"].get(n, "") for n in orig_names]

            res_parts = seg.get("result", "").split()
            direction = res_parts[0].capitalize() if res_parts else ""
            cycle = res_parts[1] if len(res_parts) > 1 else ""
            method_name = f"{seg['interval']} ({direction} - {cycle})"

            data_arrays = [np.atleast_1d(raw[field]) for field in orig_names]

            # Choose (independent, dependent) from the column set — see
            # _DEFAULT_COORD_RULES for the ordered rule list.
            default_x, default_y = _lookup_rheocompass_defaults(clean_names)

            dataset = one_d_composite_dataset.OneDimensionalCompositeDataset(
                data_arrays=data_arrays,
                data_array_names=clean_names,
                data_array_units=clean_units,
                default_independent_name=default_x,
                default_dependent_name=default_y,
                source="RheoCompass .csv file",
            )

            details = dict(metadata)

            measurements.append(
                measurement.Measurement(
                    datasets=[dataset],
                    conditions={"method": method_name},
                    details=details,
                )
            )

        return measurement_set.MeasurementSet(measurements, merge_redundant=False)
