"""
Recovery Rheology Transform
===========================

Implements oscillatory recovery rheology (OCR) and step-function recovery
analysis as a piblin and hermes_rheo transform.

Oscillatory method
------------------
Reproduces the MATLAB pipeline of ``ocr_rec_anal_mcr`` followed by
``ocr_mcr_shift_deriv``:

1. **Pairing** – consecutive measurements are grouped as forward / reverse
   pairs with positive and negative strain amplitudes: (1,2), (3,4), …
   If the last measurement has no partner the forward run is used alone.
2. **Averaging** – forward raw data (A) and reverse raw data (B) are
   averaged as (A − B) / 2 to reduce material-drift effects, then centred
   on the mean of the final oscillation interval.
3. **FFT moduli** – the average stress response is converted to
   frequency-domain moduli (G′ and G″) through a Fourier transform.
4. **Transient decomposition** – a closed half-period loop is built from
   the recovery-analysis matrix and mirrored to complete the period.
   Energy-informed dynamic moduli are then determined from total strain,
   with G′_solid, G″_solid, and G″_fluid determined from the recoverable
   and unrecoverable contributions.

Step-function method
--------------------
Reproduces ``Trios_recovery_rheology_creep.m``:

1. Non-zero strain- or stress-controlled deformation blocks are identified
   in a stepwise measurement.
2. Blocks are classified as positive or negative steps.
3. Positive/negative pairs are averaged to obtain total, unrecoverable,
   and recoverable strain and strain rates as functions of the experimental
   time of the input deformation.

Usage
-----
::

    from hermes_rheo.transforms.recovery_rheology import RecoveryRheology

    # oscillatory (RheoCompass data)
    pipeline = RecoveryRheology(method='oscillatory')
    result = pipeline(experiment)

    # step-function deformation (Trios creep/recovery data)
    pipeline = RecoveryRheology(method='step')
    result = pipeline(experiment)

    # access results
    rr = result.result
    # or equivalently: rr = result.details['recovery_rheology_result']
"""

from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import piblin.data.data_collections.experiment_set as experiment_set
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset
from piblin.data import Measurement, MeasurementSet
from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RecoveryRheologyResult:
    """Container for all outputs of a recovery-rheology analysis."""

    method: str
    recov_anal_data: np.ndarray
    pair_indices: np.ndarray
    mod_fft: np.ndarray | None = None
    amplitudes: np.ndarray | None = None
    trad_mod: np.ndarray | None = None
    raw_mod: np.ndarray | None = None
    decomp_waveform: np.ndarray | None = None
    omega: float | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)

    # ---- convenience --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "recov_anal_data": self.recov_anal_data,
            "pair_indices": self.pair_indices,
            "mod_fft": self.mod_fft,
            "amplitudes": self.amplitudes,
            "trad_mod": self.trad_mod,
            "raw_mod": self.raw_mod,
            "decomp_waveform": self.decomp_waveform,
            "omega": self.omega,
            "metadata": self.metadata or {},
        }

    # ---- export -------------------------------------------------------------

    def to_excel(self, path: str) -> str:
        """
        Write transient data and moduli to an Excel file.

        Output format matches the reference notebook
        (``test_rr_new.ipynb``).

        Parameters
        ----------
        path : str
            Output file path (.xlsx).

        Returns
        -------
        str
            The path that was written.
        """
        sheets: dict[str, pd.DataFrame] = {}

        if self.decomp_waveform is not None:
            sheets["transient data"] = pd.DataFrame(
                {
                    "t": self.decomp_waveform[:, 0],
                    "total_strain": self.decomp_waveform[:, 1],
                    "unrec_strain": self.decomp_waveform[:, 2],
                    "rec_strain": self.decomp_waveform[:, 3],
                    "total_rate": self.decomp_waveform[:, 4],
                    "unrec_rate": self.decomp_waveform[:, 5],
                    "rec_rate": self.decomp_waveform[:, 6],
                    "stress": self.decomp_waveform[:, 7],
                }
            )

        if self.mod_fft is not None and self.trad_mod is not None:
            # mod_fft is the windowed average [G'_FT, G''_FT]
            # (MATLAB: mod_anal_use = sum(mod_anal_mat(idx,:)) / n_halfp)
            if self.mod_fft.ndim == 1:
                Gp_FT = float(self.mod_fft[0])
                Gpp_FT = float(self.mod_fft[1])
            else:
                Gp_FT = float(np.nanmean(self.mod_fft[:, 0]))
                Gpp_FT = float(np.nanmean(self.mod_fft[:, 1]))

            trad = self.trad_mod
            raw = self.raw_mod
            amp = self.amplitudes
            meta = self.metadata or {}

            row: dict[str, float] = {
                # FFT moduli
                "G'_FT": Gp_FT,
                "G''_FT": Gpp_FT,
                # Energy-based (traditional) moduli – normalised by
                # total-strain amplitude (MATLAB: mod_trad)
                "trad_G'_tot": float(trad[0]),
                "trad_G'_unrec": float(trad[1]),
                "trad_G'_rec": float(trad[2]),
                "trad_G''_tot": float(trad[3]),
                "trad_G''_unrec": float(trad[4]),
                "trad_G''_rec": float(trad[5]),
                # Raw moduli – each component normalised by its own
                # amplitude (MATLAB: mod_raw)
                "raw_G'_tot": float(raw[0]),
                "raw_G'_unrec": float(raw[1]),
                "raw_G'_rec": float(raw[2]),
                "raw_G''_tot": float(raw[3]),
                "raw_G''_unrec": float(raw[4]),
                "raw_G''_rec": float(raw[5]),
                # Amplitudes (fraction for strains, 1/s for rates, Pa for stress)
                "amp_total_strain": float(amp[0]),
                "amp_unrec_strain": float(amp[1]),
                "amp_rec_strain": float(amp[2]),
                "amp_total_rate": float(amp[3]),
                "amp_unrec_rate": float(amp[4]),
                "amp_rec_rate": float(amp[5]),
                "amp_stress": float(amp[6]),
                # Analysis metadata
                "n_halfp_auto": meta.get("n_halfp_auto", np.nan),
                "n_halfp_est": meta.get("n_halfp_est", np.nan),
                "n_use_auto": meta.get("n_use_auto", np.nan),
                "join_score": meta.get("join_score", np.nan),
                "dt_auto": meta.get("dt_auto", np.nan),
            }

            sheets["moduli"] = pd.DataFrame([row])

        if self.method == "step":
            sheets["step_results"] = pd.DataFrame(
                self.recov_anal_data,
                columns=[
                    "time",
                    "total_strain",
                    "total_stress",
                    "rec_time",
                    "unrec_strain",
                    "rec_strain",
                ],
            )

        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
                for name, df in sheets.items():
                    df.to_excel(xw, sheet_name=name, index=False)
        except Exception:
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                for name, df in sheets.items():
                    df.to_excel(xw, sheet_name=name, index=False)

        return path

    # ---- plotting -----------------------------------------------------------

    def plot_lissajous(self, figsize=(6, 4)):
        """
        Plot stress versus total, recoverable, and unrecoverable responses.

        Generates three separate figures, each showing forward and reverse
        half-periods. The unrecoverable panel uses strain rate rather than
        strain to show the fluid contribution to the stress response.

        Parameters
        ----------
        figsize : tuple, optional
            Size of each individual figure (default ``(6, 4)``).

        Returns
        -------
        figs : list[matplotlib.figure.Figure]
        axes : list[matplotlib.axes.Axes]
        """
        if self.decomp_waveform is None:
            raise ValueError(
                "No decomposition waveform available. "
                "plot_lissajous() is only valid for method='oscillatory'. "
                "For step-function results use plot_step_results() instead."
            )

        decomp = self.decomp_waveform
        n_halfp = (decomp.shape[0] - 1) // 2
        f_idx = np.arange(0, n_halfp + 1)
        r_idx = np.arange(n_halfp, min(2 * n_halfp + 1, decomp.shape[0]))

        titles = [
            "Stress vs total strain",
            "Stress vs unrecoverable strain rate",
            "Stress vs recoverable strain",
        ]
        labels = [
            ("Total (forward)", "Total (reverse)"),
            ("Unrecoverable (forward)", "Unrecoverable (reverse)"),
            ("Recoverable (forward)", "Recoverable (reverse)"),
        ]
        cols = [1, 5, 3]
        color_pairs = [("C0", "C1"), ("C2", "C3"), ("C4", "C5")]

        figs, axes = [], []
        for col, title, (lf, lr), (cf, cr) in zip(cols, titles, labels, color_pairs):
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(decomp[f_idx, col], decomp[f_idx, 7], lw=1.5, color=cf, label=lf)
            ax.plot(decomp[r_idx, col], decomp[r_idx, 7], lw=1.5, color=cr, label=lr)
            ax.set_xlabel("Unrecoverable strain rate [1/s]" if col == 5 else "Strain [-]")
            ax.set_ylabel("Stress [Pa]")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            figs.append(fig)
            axes.append(ax)

        return figs, axes

    def plot_step_results(self, figsize=(8, 5)):
        """
        Plot total, unrecoverable, and recoverable strain versus experimental time.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (default ``(8, 5)``).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        if self.method != "step":
            raise ValueError("plot_step_results is only for method='step'.")
        data = self.recov_anal_data
        # columns: [time, total_strain, total_stress, rec_time, unrec_strain, rec_strain]
        t = data[:, 0]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, data[:, 4], "o-", color="C0", label="Unrecoverable strain")
        ax.plot(t, data[:, 5], "s-", color="C1", label="Recoverable strain")
        ax.plot(t, data[:, 1], "^--", color="C2", alpha=0.5, label="Total strain")
        ax.set_xlabel("Experiment Time [s]")
        ax.set_ylabel("Strain [-]")
        ax.legend(frameon=False)
        ax.set_title("Step-function recovery: strain decomposition vs experimental time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax

    def plot_transient(self, figsize=(10, 6), direction="both", logx=True):
        """
        Plot transient recoverable strain vs recovery time.

        For each stepwise deformation, the recoverable strain at recovery
        time *t* is defined as::

            γ_rec(t_recovery) = γ_tot(t_exp) − γ(t_recovery)

        i.e. how much strain has been recovered by time *t*.  This
        starts at 0 (nothing recovered yet) and asymptotically
        approaches the final recoverable strain.

        Curves are coloured by deformation duration.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (default ``(10, 6)``).
        direction : str, optional
            Which step directions to include:
            ``'pos'``, ``'neg'``, or ``'both'`` (default).
        logx : bool, optional
            Use logarithmic x-axis (default ``True``).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        if self.method != "step":
            raise ValueError("plot_transient is only for method='step'.")

        meta = self.metadata or {}
        curves = meta.get("transient_curves", [])
        if not curves:
            raise ValueError(
                "No transient curve data available. "
                "This data is only collected when using RheoCompass "
                "step-strain files via the merged extraction path."
            )

        # Filter by direction
        if direction == "pos":
            curves = [c for c in curves if c["sign"] > 0]
        elif direction == "neg":
            curves = [c for c in curves if c["sign"] < 0]

        fig, ax = plt.subplots(figsize=figsize)

        # Use a colormap based on creep duration
        cmap = plt.cm.viridis
        all_creep_durations = []
        for c in curves:
            if c["rec_time"].size > 0:
                ct = c["creep_time"][-1] - c["creep_time"][0] if c["creep_time"].size > 1 else 0
                all_creep_durations.append(ct)

        if not all_creep_durations:
            raise ValueError("No recovery transient data with finite length.")

        t_min = min(all_creep_durations)
        t_max = max(all_creep_durations)
        if t_max == t_min:
            t_max = t_min + 1.0  # avoid division by zero

        for c in curves:
            rec_t = c["rec_time"]
            rec_g = c["rec_strain"]
            if rec_t.size == 0:
                continue

            # Shift recovery time to start at 0
            rec_t_shifted = rec_t - rec_t[0]

            # Recoverable strain: γ_rec(t) = γ(end of creep) − γ(t)
            # The strain at the start of recovery equals the strain at
            # the end of creep (rec_g[0]).
            gamma_rec_transient = np.abs(rec_g[0] - rec_g)

            # Colour by creep duration
            ct = c["creep_time"][-1] - c["creep_time"][0] if c["creep_time"].size > 1 else 0
            color = cmap((ct - t_min) / (t_max - t_min))

            sign_label = "+" if c["sign"] > 0 else "−"
            label = f"t_creep = {ct:.1f} s ({sign_label})"
            ax.plot(rec_t_shifted, gamma_rec_transient, color=color, lw=1.2, alpha=0.8, label=label)

        if logx:
            ax.set_xscale("log")
        ax.set_xlabel("Recovery time [s]")
        ax.set_ylabel("Recoverable strain [-]")
        ax.set_title("Transient recoverable strain vs recovery time")
        ax.legend(fontsize="small", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax


# ---------------------------------------------------------------------------
# Main transform class
# ---------------------------------------------------------------------------


class RecoveryRheology(MeasurementSetTransform):
    """
    Recovery rheology transform for hermes-rheo / piblin.

    Parameters
    ----------
    method : str
        ``'oscillatory'`` or ``'step'``.
    n_osc : int
        Harmonic bin index for FFT (oscillatory only, default 2).
    n_use : int or None
        Initial guess for the half-period starting row (1-based,
        matching MATLAB convention).  When ``None`` (default) the
        optimal value is selected automatically by minimising a
        join-discontinuity score – matching the behaviour of the
        reference MATLAB code.
    diff_state : int
        Stencil width for periodic centred differentiation (1–4, default 4).
    k_step : int
        Step size for derivative stencil (default 1).
    omega : float or None
        Angular frequency in rad/s.  If ``None`` (default) the value is
        estimated automatically from the FFT of the oscillation data.
        When the experiment frequency is known it should be supplied
        explicitly to match the reference MATLAB pipeline.
    stress_zero_tol : float
        Threshold below which stress is treated as zero (step only).
    """

    def __init__(
        self,
        method: str = "oscillatory",
        n_osc: int = 2,
        n_use: int | None = None,
        diff_state: int = 4,
        k_step: int = 1,
        strain_shift_state: bool = False,
        omega: float | None = None,
        stress_zero_tol: float = 1e-10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.method = method.lower()
        self.n_osc = n_osc
        self.n_use = n_use
        self.diff_state = diff_state
        self.k_step = k_step
        self.strain_shift_state = strain_shift_state
        self.omega = omega
        self.stress_zero_tol = stress_zero_tol
        self.result: RecoveryRheologyResult | None = None
        self._transient_curves: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # piblin transform hook
    # ------------------------------------------------------------------

    def _apply(self, target, **kwargs):
        if self.method == "oscillatory":
            rr_result = self._run_oscillatory(target)
        elif self.method in ("step", "step strain", "creep"):
            rr_result = self._run_step(target)
        else:
            raise ValueError(f"method must be 'oscillatory' or 'step', got '{self.method}'")

        # Store result on the transform instance for easy access
        self.result = rr_result

        # Store on the returned target so callers can use
        # ``processed.result`` directly instead of ``pipeline.result``
        try:
            target.result = rr_result
        except (AttributeError, TypeError):
            pass

        # Try to store on target details if possible
        try:
            if hasattr(target, "details") and isinstance(target.details, dict):
                target.details["recovery_rheology"] = rr_result.to_dict()
                target.details["recovery_rheology_result"] = rr_result
        except Exception:
            pass

        # Build a decomposition dataset if available
        if rr_result.decomp_waveform is not None:
            ds = self._build_decomposition_dataset(rr_result.decomp_waveform)
            try:
                if hasattr(target, "details") and isinstance(target.details, dict):
                    target.details["recovery_rheology_dataset"] = ds
            except Exception:
                pass

        return target

    # ==================================================================
    #  OSCILLATORY  workflow
    # ==================================================================

    def _run_oscillatory(self, target) -> RecoveryRheologyResult:
        """
        Full oscillatory recovery-rheology pipeline.

        Works with both:
        - **RheoCompass** data where measurements are already split by
          Result and contain interval IDs inside each dataset.
          Intervals belonging to the same run (e.g. "Forward - 17") are
          automatically merged before pairing.
        - **Trios** data where each measurement is one forward or reverse
          run and interval structure is embedded in the data arrays.
        """
        measurements = self._get_measurements(target)

        # ----------------------------------------------------------
        # Stage 1 – build the 3-D data cube  (max_rows × 4 × n_meas)
        #   columns: [interval_id, time, strain, stress]
        #
        # RheoCompass creates one measurement per interval.  We need to
        # group intervals belonging to the same Result (e.g. "Forward - 17")
        # into a single stacked array so the pairing logic can find the
        # oscillation / pre-load / recovery interval structure.
        # ----------------------------------------------------------
        stacks: list[np.ndarray] = []

        # Try to detect RheoCompass-style method labels like
        # "Interval 1 (Forward - 17)" and group by the part after "Interval N"
        grouped: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
        ungrouped: list[np.ndarray] = []

        for meas in measurements:
            arr = self._extract_oscillatory_arrays(meas)
            if arr is None or arr.shape[0] == 0:
                continue

            method = ""
            if hasattr(meas, "conditions") and meas.conditions and "method" in meas.conditions:
                method = str(meas.conditions["method"])

            # Match "Interval <N> (<direction> - <cycle>)"
            m = re.match(
                r"Interval\s+(\d+)\s+\((.+)\)",
                method,
                flags=re.IGNORECASE,
            )
            if m:
                interval_id = int(m.group(1))
                run_key = m.group(2).strip()  # e.g. "Forward - 17"
                # Override the interval column with the parsed interval id
                arr[:, 0] = float(interval_id)
                grouped[run_key].append((interval_id, arr))
            else:
                ungrouped.append(arr)

        if grouped:
            # Sort runs by their cycle number so forward/reverse alternate
            # correctly: "Forward - 1", "Reverse - 1", "Forward - 2", ...
            def _run_sort_key(key: str):
                parts = key.split("-")
                try:
                    cycle = int(parts[-1].strip())
                except (ValueError, IndexError):
                    cycle = 0
                direction = parts[0].strip().lower()
                # forward before reverse within the same cycle
                dir_order = 0 if direction.startswith("f") else 1
                return (cycle, dir_order)

            for run_key in sorted(grouped.keys(), key=_run_sort_key):
                intervals = grouped[run_key]
                # Sort intervals within a run by interval id
                intervals.sort(key=lambda x: x[0])
                merged = np.vstack([arr for _, arr in intervals])
                stacks.append(merged)
        else:
            stacks = ungrouped

        if not stacks:
            raise ValueError("No usable oscillatory measurements found.")

        num_meas = len(stacks)
        size_meas = np.array([s.shape[0] for s in stacks], dtype=int)
        mm = int(size_meas.max())

        data_in = np.full((mm, 4, num_meas), np.nan, dtype=float)
        for i, stk in enumerate(stacks):
            n = stk.shape[0]
            data_in[:n, :, i] = stk

        # ----------------------------------------------------------
        # Stage 2 – pairing + FFT  (ocr_rec_anal_mcr)
        # ----------------------------------------------------------
        omega_user = self.omega  # may be None

        (avg_data, recov_anal_data, mod_tot_data, num_pairs, omega_list) = self._ocr_rec_anal_mcr(
            data_in,
            num_meas,
            size_meas,
            self.n_osc,
            omega_override=omega_user,
        )

        # ----------------------------------------------------------
        # Filter valid rows  (keep only rows with ALL finite values,
        # matching the reference MATLAB / notebook pipeline)
        # ----------------------------------------------------------
        valid = np.all(np.isfinite(recov_anal_data), axis=1)
        recov_anal_data = recov_anal_data[valid]
        mod_tot_data = mod_tot_data[valid]
        omega_list = omega_list[valid]
        num_pairs = int(valid.sum())

        if num_pairs == 0:
            raise ValueError(
                "No valid recovery-analysis pairs remain after filtering. "
                "Check that the data contains oscillation + recovery "
                "intervals and that strain/stress columns are present."
            )

        # ----------------------------------------------------------
        # Stage 3 – shift / derivative / energy  (ocr_mcr_shift_deriv)
        # ----------------------------------------------------------
        if omega_user is not None and np.isfinite(omega_user) and omega_user > 0:
            omega_energy = float(omega_user)
        else:
            finite_mask = np.isfinite(omega_list)
            omega_energy = (
                float(np.nanmedian(omega_list[finite_mask])) if finite_mask.any() else 1.0
            )
            if not np.isfinite(omega_energy) or omega_energy <= 0:
                omega_energy = 1.0

        anal_res = self._ocr_mcr_shift_deriv(
            rec_anal_mat=recov_anal_data,
            mod_anal_mat=mod_tot_data,
            omega=omega_energy,
        )

        return RecoveryRheologyResult(
            method="oscillatory",
            recov_anal_data=recov_anal_data,
            pair_indices=np.arange(num_pairs, dtype=int),
            mod_fft=anal_res["mod_fft"],
            amplitudes=anal_res["amp"],
            trad_mod=anal_res["trad_mod"],
            raw_mod=anal_res["raw_mod"],
            decomp_waveform=anal_res["decomp"],
            omega=omega_energy,
            metadata={
                "num_pairs": num_pairs,
                "n_osc": self.n_osc,
                "n_halfp_auto": anal_res["n_halfp_auto"],
                "n_halfp_est": anal_res["n_halfp_est"],
                "n_use_auto": anal_res["n_use_auto"],
                "join_score": anal_res["join_score"],
                "dt_auto": anal_res["dt_auto"],
                "diff_state": self.diff_state,
                "k_step": self.k_step,
                "omega_energy": omega_energy,
                "omega_per_pair": omega_list,
                "mod_tot_per_pair": mod_tot_data,
            },
        )

    # ------------------------------------------------------------------
    # Extract arrays from a single measurement
    # ------------------------------------------------------------------

    def _extract_oscillatory_arrays(self, meas: Measurement) -> np.ndarray | None:
        """
        Return an (N, 4) array  [interval_id, time, strain, stress]
        from *meas.datasets[0]*.

        Tries several common column-name variants.
        """
        ds = meas.datasets[0]
        names_lower = [n.lower() for n in ds.data_array_names]

        def _find(candidates):
            for c in candidates:
                for i, n in enumerate(names_lower):
                    if c == n:
                        return i
            return None

        idx_time = _find(["time", "interval time", "step time"])
        idx_strain = _find(["shear strain", "strain", "strain (step)"])
        idx_stress = _find(["shear stress", "stress", "stress (step)"])

        if idx_time is None or idx_strain is None or idx_stress is None:
            return None

        time_arr = np.asarray(ds.data_arrays[idx_time], dtype=float)
        strain_arr = np.asarray(ds.data_arrays[idx_strain], dtype=float)
        stress_arr = np.asarray(ds.data_arrays[idx_stress], dtype=float)

        # Check if strain is in % and convert
        strain_unit = (
            ds.data_array_units[idx_strain] if idx_strain < len(ds.data_array_units) else ""
        )
        if strain_unit == "%":
            strain_arr = strain_arr * 0.01

        # Try to get interval id
        idx_interval = _find(["interval", "interval id"])
        if idx_interval is not None:
            interval_arr = np.asarray(ds.data_arrays[idx_interval], dtype=float)
        else:
            # If no interval column, assign a single interval
            interval_arr = np.ones_like(time_arr)

        n = min(len(time_arr), len(strain_arr), len(stress_arr), len(interval_arr))
        return np.column_stack(
            [
                interval_arr[:n],
                time_arr[:n],
                strain_arr[:n],
                stress_arr[:n],
            ]
        )

    # ------------------------------------------------------------------
    # Stage 2 – pairing + FFT  (MATLAB: ocr_rec_anal_mcr_modify)
    # ------------------------------------------------------------------

    def _ocr_rec_anal_mcr(
        self,
        data_in: np.ndarray,
        num_meas: int,
        size_meas: np.ndarray,
        n_osc: int,
        omega_override: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        """
        Pair measurements as (1,2), (3,4), … and compute per-pair
        recovery data and FFT moduli.

        This is a faithful port of MATLAB ``ocr_rec_anal_mcr_modify``.

        Parameters
        ----------
        omega_override : float or None
            If supplied and finite, use this omega (rad/s) for the G″
            calculation instead of estimating per pair.

        Returns
        -------
        avg_data        : (maxN, 4, num_pairs)
        recov_anal_data : (num_pairs, 5)  [t_n, total, unrec, rec, stress]
        mod_tot_data    : (num_pairs, 2)  [G'_FT, G''_FT]
        num_pairs       : int
        omega_list      : (num_pairs,)  rad/s per pair
        """
        # Pair as (1,2), (3,4), … like the modified MATLAB code
        num_pairs = math.ceil(num_meas / 2)
        maxN = int(np.max(size_meas))

        avg_data = np.full((maxN, 4, num_pairs), np.nan, dtype=float)
        recov_anal_data = np.full((num_pairs, 5), np.nan, dtype=float)
        mod_tot_data = np.full((num_pairs, 2), np.nan, dtype=float)
        omega_list = np.full(num_pairs, np.nan, dtype=float)

        use_fixed_omega = (
            omega_override is not None and np.isfinite(omega_override) and omega_override > 0
        )

        print(f"num_meas  = {num_meas}")
        print(f"num_pairs = {num_pairs}")

        for ii in range(num_pairs):
            meas_f = 2 * ii  # forward (0-based)
            meas_r = 2 * ii + 1  # reverse (0-based)
            has_reverse = meas_r < num_meas

            if has_reverse:
                N = min(int(size_meas[meas_f]), int(size_meas[meas_r]))
                # interval & time from forward
                avg_data[:N, 0:2, ii] = data_in[:N, 0:2, meas_f]
                # strain & stress averaged as (forward - reverse) / 2
                avg_data[:N, 2:4, ii] = (data_in[:N, 2:4, meas_f] - data_in[:N, 2:4, meas_r]) / 2.0
            else:
                N = int(size_meas[meas_f])
                avg_data[:N, :, ii] = data_in[:N, :, meas_f]

            avg_block = avg_data[:N, :, ii]

            # Max interval number
            n_int = np.nanmax(avg_block[:, 0])

            # Final full oscillation interval (n_int - 2)
            final_osc_idx = np.where(avg_block[:, 0] == (n_int - 2))[0]
            if final_osc_idx.size == 0:
                warnings.warn(
                    f"Pair {ii + 1}: final oscillation interval not found. Skipping pair.",
                    stacklevel=2,
                )
                continue

            # Centre strain and stress on final-oscillation mean
            center_gamma = float(np.nanmean(avg_block[final_osc_idx, 2]))
            center_sigma = float(np.nanmean(avg_block[final_osc_idx, 3]))
            avg_data[:N, 2, ii] -= center_gamma
            avg_data[:N, 3, ii] -= center_sigma

            # Refresh block after centering
            avg_block = avg_data[:N, :, ii]

            # End-of-oscillation and end-of-recovery indices
            osc_end_idx = np.where(avg_block[:, 0] == (n_int - 1))[0]
            rec_end_idx = np.where(avg_block[:, 0] == n_int)[0]

            if osc_end_idx.size == 0 or rec_end_idx.size == 0:
                warnings.warn(
                    f"Pair {ii + 1}: recovery interval endpoints not found. "
                    f"Skipping recovery extraction.",
                    stacklevel=2,
                )
            else:
                i_osc_end = osc_end_idx[-1]
                i_rec_end = rec_end_idx[-1]

                recov_anal_data[ii, :] = [
                    avg_block[i_osc_end, 1],  # time
                    avg_block[i_osc_end, 2],  # total strain
                    avg_block[i_rec_end, 2],  # unrec strain
                    avg_block[i_osc_end, 2] - avg_block[i_rec_end, 2],  # rec strain
                    avg_block[i_osc_end, 3],  # stress
                ]

            # ---- FFT moduli (explicit AB rotation, MATLAB-faithful) ----
            g_osc = avg_block[final_osc_idx, 2]
            s_osc = avg_block[final_osc_idx, 3]
            n_f = final_osc_idx.size

            Gg = np.fft.fft(g_osc, n=n_f)
            Gs = np.fft.fft(s_osc, n=n_f)
            h = n_osc  # harmonic bin (0-based in Python)

            if h >= n_f:
                warnings.warn(
                    f"Pair {ii + 1}: n_osc={n_osc} too large for n_fosc={n_f}. "
                    f"Skipping modulus calculation.",
                    stacklevel=2,
                )
                continue

            # MATLAB: strain_AB1 = [2*real(fft[h+1])/n, -2*imag(fft[h+1])/n]
            strain_AB1 = np.array(
                [
                    2.0 * np.real(Gg[h]) / n_f,
                    -2.0 * np.imag(Gg[h]) / n_f,
                ]
            )
            stress_AB1 = np.array(
                [
                    2.0 * np.real(Gs[h]) / n_f,
                    -2.0 * np.imag(Gs[h]) / n_f,
                ]
            )

            if abs(strain_AB1[1]) < np.finfo(float).eps:
                warnings.warn(
                    f"Pair {ii + 1}: strain_AB1[1] too small for stable modulus calculation.",
                    stacklevel=2,
                )
                continue

            delta_strain = math.atan(strain_AB1[0] / strain_AB1[1])

            c_d, s_d = math.cos(delta_strain), math.sin(delta_strain)
            strain_AB = np.array(
                [
                    strain_AB1[0] * c_d - strain_AB1[1] * s_d,
                    strain_AB1[1] * c_d + strain_AB1[0] * s_d,
                ]
            )
            stress_AB = np.array(
                [
                    stress_AB1[0] * c_d - stress_AB1[1] * s_d,
                    stress_AB1[1] * c_d + stress_AB1[0] * s_d,
                ]
            )

            if abs(strain_AB[1]) < np.finfo(float).eps:
                warnings.warn(
                    f"Pair {ii + 1}: shifted strain amplitude too small for modulus calculation.",
                    stacklevel=2,
                )
                continue

            # Estimate ω from time series
            t_osc = avg_block[final_osc_idx, 1]
            if use_fixed_omega:
                omega_i = float(omega_override)
            else:
                omega_i = self._estimate_omega(t_osc, g_osc)
            omega_list[ii] = omega_i

            # MATLAB: mod_tot_data = [stress_AB(2)/strain_AB(2),
            #                         stress_AB(1)/(strain_AB(2)*omega)]
            mod_tot_data[ii, 0] = stress_AB[1] / strain_AB[1]
            if np.isfinite(omega_i) and omega_i > 0:
                mod_tot_data[ii, 1] = stress_AB[0] / (strain_AB[1] * omega_i)

        return avg_data, recov_anal_data, mod_tot_data, num_pairs, omega_list

    # ------------------------------------------------------------------
    # Stage 3 – shift / derivative / energy
    # (MATLAB: ocr_mcr_shift_deriv_modify)
    # ------------------------------------------------------------------

    def _ocr_mcr_shift_deriv(
        self,
        rec_anal_mat: np.ndarray,
        mod_anal_mat: np.ndarray,
        omega: float,
    ) -> dict[str, Any]:
        """
        Build the closed half-period loop, differentiate, compute
        energy-based moduli.

        This is a faithful port of MATLAB ``ocr_mcr_shift_deriv_modify``
        and matches ``test_rr_new.ipynb`` exactly.

        n_halfp is always auto-determined from omega and dt (MATLAB
        convention).

        Parameters
        ----------
        rec_anal_mat : (n_meas, 5)
            [time, total_strain, unrec_strain, rec_strain, stress]
        mod_anal_mat : (n_meas, 2)
            [G'_FT, G''_FT] per pair
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        dict with keys: decomp, amp, trad_mod, raw_mod, mod_fft,
        lump, n_halfp_auto, n_halfp_est, n_use_auto, join_score, dt_auto
        """
        rec_anal_mat = np.asarray(rec_anal_mat, dtype=float)
        mod_anal_mat = np.asarray(mod_anal_mat, dtype=float)
        n_meas = rec_anal_mat.shape[0]

        if rec_anal_mat.size == 0 or n_meas < 3:
            raise ValueError("rec_anal_mat must contain at least 3 rows.")

        # Initial n_use (MATLAB: default 1, 1-based)
        if self.n_use is not None and np.isfinite(self.n_use):
            n_use_init = max(1, int(round(self.n_use)))
        else:
            n_use_init = 1

        # ---- auto-determine n_halfp from omega & dt ----
        # MATLAB: dt = median(diff(rec_anal_mat(:,1)))
        dt_vec = np.diff(rec_anal_mat[:, 0])
        dt_vec = dt_vec[np.isfinite(dt_vec) & ~np.isinf(dt_vec) & (dt_vec > 0)]
        if dt_vec.size == 0:
            raise ValueError("Could not determine a valid positive dt from rec_anal_mat[:,0].")

        dt = float(np.median(dt_vec))
        half_period = math.pi / omega
        n_halfp_est = int(round(half_period / dt))
        if n_halfp_est < 1:
            n_halfp_est = 1

        # MATLAB: max_halfp = n_meas - n_use
        max_halfp = n_meas - n_use_init
        if max_halfp < 1:
            raise ValueError(f"Not enough rows in rec_anal_mat for n_use = {n_use_init}.")

        n_halfp = min(n_halfp_est, max_halfp)
        print(
            f"Auto-selected n_halfp = {n_halfp} (estimated {n_halfp_est}, max allowed {max_halfp})"
        )

        # ---- determine all half-period options ----
        n_opts = n_meas - n_halfp
        if n_opts < 1:
            raise ValueError("Not enough rows to construct half-period options.")

        # MATLAB: meas_opts(nn:(nn+n_halfp), nn) = ...
        meas_opts = np.full((n_meas, n_opts), np.nan, dtype=float)

        for nn in range(n_opts):
            diff_cyc = -rec_anal_mat[nn + n_halfp, 4] - rec_anal_mat[nn, 4]
            meas_opts[nn : nn + n_halfp + 1, nn] = (
                rec_anal_mat[nn : nn + n_halfp + 1, 4] + diff_cyc / 2.0
            )

        # ---- auto-select n_use by minimizing discontinuity ----
        best_score = np.inf
        best_use = min(max(1, n_use_init), n_opts)  # MATLAB-style 1-based

        for cand0 in range(n_opts):
            idx = slice(cand0, cand0 + n_halfp)

            first_half = np.column_stack(
                [
                    rec_anal_mat[idx, 1:4],
                    meas_opts[cand0 : cand0 + n_halfp, cand0],
                ]
            )
            second_half = np.column_stack(
                [
                    -rec_anal_mat[idx, 1:4],
                    -meas_opts[cand0 : cand0 + n_halfp, cand0],
                ]
            )

            mid_jump = np.linalg.norm(first_half[-1, :] - second_half[0, :])
            close_jump = np.linalg.norm(second_half[-1, :] - first_half[0, :])
            score = mid_jump + close_jump

            if score < best_score:
                best_score = score
                best_use = cand0 + 1  # MATLAB-style 1-based

        n_use = best_use
        print(f"Auto-selected n_use = {n_use} with join score = {best_score:.6g}")

        # Convert to 0-based for indexing
        n_use0 = n_use - 1

        # ---- optional strain shift ----
        if self.strain_shift_state:
            if (n_use0 + n_halfp) >= n_meas:
                raise ValueError(
                    f"strain_shift_state requires row {n_use0 + n_halfp + 1}, "
                    f"but rec_anal_mat has only {n_meas} rows."
                )
            gamma_shift = (-rec_anal_mat[n_use0 + n_halfp, 1:4] - rec_anal_mat[n_use0, 1:4]) / 2.0
        else:
            gamma_shift = np.array([0.0, 0.0, 0.0])

        # ---- build full period ----
        # MATLAB: idx = n_use:(n_use+(n_halfp-1))
        idx = np.arange(n_use0, n_use0 + n_halfp)

        rec_anal_use = np.vstack(
            [
                np.column_stack(
                    [
                        rec_anal_mat[idx, 0],
                        rec_anal_mat[idx, 1:4] + gamma_shift,
                        np.zeros((n_halfp, 3)),
                        meas_opts[idx, n_use0],
                    ]
                ),
                np.column_stack(
                    [
                        rec_anal_mat[idx, 0] + dt * n_halfp,
                        -rec_anal_mat[idx, 1:4] - gamma_shift,
                        np.zeros((n_halfp, 3)),
                        -meas_opts[idx, n_use0],
                    ]
                ),
            ]
        )

        # MATLAB: mod_anal_use = sum(mod_anal_mat(idx,:)) / n_halfp
        # Use nanmean so that NaN entries (skipped pairs) don't poison
        # the average and leave G'_FT / G''_FT empty.
        mod_anal_use = np.nanmean(mod_anal_mat[idx, :], axis=0)

        # ---- smooth discontinuities ----
        # Midpoint join: rows n_halfp-1 and n_halfp
        join_mid = (rec_anal_use[n_halfp - 1, 1:8] + rec_anal_use[n_halfp, 1:8]) / 2.0
        rec_anal_use[n_halfp - 1, 1:8] = join_mid
        rec_anal_use[n_halfp, 1:8] = join_mid

        # End/start closure: row 2*n_halfp-1 and row 0
        join_end = (rec_anal_use[2 * n_halfp - 1, 1:8] + rec_anal_use[0, 1:8]) / 2.0
        rec_anal_use[2 * n_halfp - 1, 1:8] = join_end
        rec_anal_use[0, 1:8] = join_end

        # ---- differentiate (periodic centred stencil) ----

        for col in range(3):
            rec_anal_use[:, 4 + col] = self._centered_derivative_periodic(
                rec_anal_use[:, 1 + col],
                k=self.k_step,
                stencil=self.diff_state,
                dt=dt,
            )

        # ---- close curve (append first row) ----
        rec_anal_use = np.vstack([rec_anal_use, rec_anal_use[0, :][None, :]])

        # ---- energy / Lissajous ----
        npts = 2 * n_halfp
        rec_anal_nrg_s = np.zeros((npts, 3), dtype=float)
        rec_anal_nrg_d_dot = np.zeros((npts, 3), dtype=float)

        for q in range(3):
            # MATLAB: rec_anal_nrg_s(:,q) = strain * stress / 2
            rec_anal_nrg_s[:, q] = rec_anal_use[:npts, q + 1] * rec_anal_use[:npts, 7] / 2.0
            # MATLAB: rec_anal_nrg_d_dot(:,q) = rate * stress
            rec_anal_nrg_d_dot[:, q] = rec_anal_use[:npts, q + 4] * rec_anal_use[:npts, 7]

        avg_nrg_s = np.mean(rec_anal_nrg_s, axis=0)
        avg_nrg_d_dot = np.mean(rec_anal_nrg_d_dot, axis=0)

        # MATLAB: rec_anal_amp = max(abs(rec_anal_use(:,2:8)))
        rec_anal_amp = np.max(np.abs(rec_anal_use[:, 1:8]), axis=0)

        # MATLAB: mod_trad = [4*avg_nrg_s/rec_anal_amp(1)^2,
        #                     2*avg_nrg_d_dot/(rec_anal_amp(1)*omega)^2]
        trad_mod = np.concatenate(
            [
                4.0 * avg_nrg_s / (rec_anal_amp[0] ** 2),
                2.0 * avg_nrg_d_dot / ((rec_anal_amp[0] * omega) ** 2),
            ]
        )

        # MATLAB: mod_raw = [4*avg_nrg_s./rec_anal_amp(1:3).^2,
        #                    2*avg_nrg_d_dot./rec_anal_amp(4:6).^2]
        raw_mod = np.concatenate(
            [
                4.0 * avg_nrg_s / (rec_anal_amp[0:3] ** 2),
                2.0 * avg_nrg_d_dot / (rec_anal_amp[3:6] ** 2),
            ]
        )

        # MATLAB: lump = [mod_anal_use, mod_trad, mod_raw, rec_anal_amp]
        lump = np.concatenate([mod_anal_use, trad_mod, raw_mod, rec_anal_amp])

        return {
            "decomp": rec_anal_use,
            "amp": rec_anal_amp,
            "trad_mod": trad_mod,
            "raw_mod": raw_mod,
            "mod_fft": mod_anal_use,
            "lump": lump,
            "n_halfp_auto": n_halfp,
            "n_halfp_est": n_halfp_est,
            "n_use_auto": n_use,
            "join_score": best_score,
            "dt_auto": dt,
        }

    # ==================================================================
    #  STEP-FUNCTION workflow  (MATLAB: Trios_recovery_rheology_creep.m)
    # ==================================================================

    def _run_step(self, target) -> RecoveryRheologyResult:
        """
        Step-function recovery analysis.

        1. Extract time, strain, stress from each measurement.
        2. Merge into a single continuous time series if needed (handles
           RheoCompass data where each interval is a separate measurement).
        3. Identify non-zero strain- or stress-controlled deformation blocks
           and classify as +/−.
        4. Positive/negative pairs are averaged to obtain total strain,
           unrecoverable strain, and recoverable strain and strain rate as
           functions of experimental time.
        """
        measurements = self._get_measurements(target)

        # ----------------------------------------------------------
        # Try per-measurement block detection first (Trios-style: each
        # measurement contains a full stress+recovery sequence).
        # ----------------------------------------------------------
        rows_pos: list[np.ndarray] = []
        rows_neg: list[np.ndarray] = []

        for meas in measurements:
            pos, neg = self._extract_step_blocks(meas)
            rows_pos.extend(pos)
            rows_neg.extend(neg)

        # ----------------------------------------------------------
        # If no blocks found OR all recovery data is NaN (meaning
        # per-measurement detection found stress blocks but couldn't
        # find recovery within the same measurement), fall back to
        # merged approach (RheoCompass-style: each interval is a
        # separate measurement, need to concatenate).
        # ----------------------------------------------------------
        all_recovery_nan = (rows_pos or rows_neg) and all(
            np.isnan(r[4]) for r in rows_pos + rows_neg
        )
        if not rows_pos and not rows_neg or all_recovery_nan:
            rows_pos, rows_neg = self._extract_step_blocks_merged(measurements)

        if not rows_pos and not rows_neg:
            raise ValueError("No valid step-strain recovery blocks found.")

        arr_pos = np.array(rows_pos) if rows_pos else np.empty((0, 7))
        arr_neg = np.array(rows_neg) if rows_neg else np.empty((0, 7))

        n = min(arr_pos.shape[0], arr_neg.shape[0])
        if n == 0:
            # Only one direction available – use what we have
            combined = arr_pos if arr_pos.shape[0] > 0 else arr_neg
            pair_avg = np.column_stack(
                [
                    combined[:, 1],  # time
                    np.abs(combined[:, 2]),  # total strain
                    np.abs(combined[:, 3]),  # total stress
                    combined[:, 4],  # recovery time
                    np.abs(combined[:, 5]),  # unrec strain
                    np.abs(combined[:, 2]) - np.abs(combined[:, 5]),  # rec strain
                ]
            )
        else:
            pair_avg = np.full((n, 6), np.nan, dtype=float)
            for j in range(n):
                time_nz = np.nanmean([arr_pos[j, 1], arr_neg[j, 1]])
                tot_strain = np.nanmean([abs(arr_pos[j, 2]), abs(arr_neg[j, 2])])
                tot_stress = np.nanmean([abs(arr_pos[j, 3]), abs(arr_neg[j, 3])])
                rec_time = np.nanmean([arr_pos[j, 4], arr_neg[j, 4]])
                unrec_strain = np.nanmean([abs(arr_pos[j, 5]), abs(arr_neg[j, 5])])
                rec_strain = tot_strain - unrec_strain

                pair_avg[j, :] = [
                    time_nz,
                    tot_strain,
                    tot_stress,
                    rec_time,
                    unrec_strain,
                    rec_strain,
                ]

        return RecoveryRheologyResult(
            method="step",
            recov_anal_data=pair_avg,
            pair_indices=np.arange(pair_avg.shape[0], dtype=int),
            metadata={
                "n_pairs": pair_avg.shape[0],
                "n_pos": arr_pos.shape[0],
                "n_neg": arr_neg.shape[0],
                "stress_zero_tol": self.stress_zero_tol,
                "transient_curves": getattr(self, "_transient_curves", []),
            },
        )

    def _extract_step_blocks(self, meas: Measurement) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        From one measurement, find non-zero stress blocks, classify
        as positive or negative, and extract the recovery point.

        Returns two lists (pos_rows, neg_rows) where each row is
        [sign, nz_time, nz_strain, nz_stress, z_time, z_strain, z_stress].
        """
        ds = meas.datasets[0]
        names_lower = [n.lower() for n in ds.data_array_names]

        def _find(candidates):
            for c in candidates:
                for i, n in enumerate(names_lower):
                    if c == n:
                        return i
            return None

        idx_time = _find(["time", "step time", "interval time", "average time"])
        idx_strain = _find(["strain", "shear strain", "strain (step)"])
        idx_stress = _find(["stress", "shear stress", "stress (step)"])

        if idx_time is None or idx_strain is None or idx_stress is None:
            return [], []

        time_arr = np.asarray(ds.data_arrays[idx_time], dtype=float)
        strain_arr = np.asarray(ds.data_arrays[idx_strain], dtype=float)
        stress_arr = np.asarray(ds.data_arrays[idx_stress], dtype=float)

        N = min(len(time_arr), len(strain_arr), len(stress_arr))
        time_arr = time_arr[:N]
        strain_arr = strain_arr[:N]
        stress_arr = stress_arr[:N]

        tol = self.stress_zero_tol
        check_zero = np.abs(stress_arr) <= tol
        find_nonzero = ~check_zero

        # Find contiguous non-zero blocks
        d = np.diff(np.r_[False, find_nonzero, False].astype(int))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1

        rows_pos, rows_neg = [], []

        for i1, i2 in zip(starts, ends):
            sgn = np.sign(stress_arr[i1])
            if sgn == 0:
                continue
            if np.any(np.sign(stress_arr[i1 : i2 + 1]) != sgn):
                continue

            nz_t = time_arr[i2]
            nz_g = strain_arr[i2]
            nz_s = stress_arr[i2]

            # Look for following zero-stress block
            z1 = i2 + 1
            if z1 >= N or not check_zero[z1]:
                z_t, z_g, z_s = np.nan, np.nan, np.nan
            else:
                z2 = z1
                while z2 + 1 < N and check_zero[z2 + 1]:
                    z2 += 1
                z_t = time_arr[z2]
                z_g = strain_arr[z2]
                z_s = stress_arr[z2]

            row = np.array([sgn, nz_t, nz_g, nz_s, z_t, z_g, z_s])
            if sgn > 0:
                rows_pos.append(row)
            else:
                rows_neg.append(row)

        return rows_pos, rows_neg

    def _extract_step_blocks_merged(
        self, measurements: list[Measurement]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Merge all measurements into a single continuous time series and
        then run block detection.

        This handles RheoCompass-style data where each interval is stored
        as a separate measurement (so a single measurement never contains
        both a non-zero and a zero stress block).

        Parameters
        ----------
        measurements : list of Measurement

        Returns
        -------
        rows_pos, rows_neg : lists of (7,) arrays
            Same format as ``_extract_step_blocks``.

        Side Effects
        ------------
        Sets ``self._transient_curves`` to a list of dicts, one per
        non-zero stress block, containing the full recovery transient:
        ``{'sign', 'creep_time', 'creep_strain', 'rec_time', 'rec_strain'}``.
        """
        time_parts: list[np.ndarray] = []
        strain_parts: list[np.ndarray] = []
        stress_parts: list[np.ndarray] = []

        for meas in measurements:
            ds = meas.datasets[0]
            names_lower = [n.lower() for n in ds.data_array_names]

            def _find(candidates, names_lower=names_lower):
                for c in candidates:
                    for i, n in enumerate(names_lower):
                        if c == n:
                            return i
                return None

            idx_time = _find(["time", "step time", "interval time", "average time"])
            idx_strain = _find(["strain", "shear strain", "strain (step)"])
            idx_stress = _find(["stress", "shear stress", "stress (step)"])

            if idx_time is None or idx_strain is None or idx_stress is None:
                continue

            t = np.asarray(ds.data_arrays[idx_time], dtype=float)
            g = np.asarray(ds.data_arrays[idx_strain], dtype=float)
            s = np.asarray(ds.data_arrays[idx_stress], dtype=float)

            n = min(len(t), len(g), len(s))
            time_parts.append(t[:n])
            strain_parts.append(g[:n])
            stress_parts.append(s[:n])

        if not time_parts:
            return [], []

        time_arr = np.concatenate(time_parts)
        strain_arr = np.concatenate(strain_parts)
        stress_arr = np.concatenate(stress_parts)

        # Do NOT sort by time: RheoCompass resets time per Result, so
        # sorting would interleave different stress cycles.  The reader
        # already produces measurements in file order which is the
        # correct temporal sequence (stress interval → recovery interval
        # for each Result).

        N = len(time_arr)
        tol = self.stress_zero_tol
        check_zero = np.abs(stress_arr) <= tol
        find_nonzero = ~check_zero

        # Find contiguous non-zero blocks
        d = np.diff(np.r_[False, find_nonzero, False].astype(int))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1

        rows_pos: list[np.ndarray] = []
        rows_neg: list[np.ndarray] = []
        transient_curves: list[dict[str, Any]] = []

        for i1, i2 in zip(starts, ends):
            sgn = np.sign(stress_arr[i1])
            if sgn == 0:
                continue
            if np.any(np.sign(stress_arr[i1 : i2 + 1]) != sgn):
                continue

            nz_t = time_arr[i2]
            nz_g = strain_arr[i2]
            nz_s = stress_arr[i2]

            # Full creep transient (during non-zero stress)
            creep_t = time_arr[i1 : i2 + 1].copy()
            creep_g = strain_arr[i1 : i2 + 1].copy()

            # Look for following zero-stress block
            z1 = i2 + 1
            if z1 >= N or not check_zero[z1]:
                z_t, z_g, z_s = np.nan, np.nan, np.nan
                rec_t_full = np.array([])
                rec_g_full = np.array([])
            else:
                z2 = z1
                while z2 + 1 < N and check_zero[z2 + 1]:
                    z2 += 1
                z_t = time_arr[z2]
                z_g = strain_arr[z2]
                z_s = stress_arr[z2]
                # Full recovery transient
                rec_t_full = time_arr[z1 : z2 + 1].copy()
                rec_g_full = strain_arr[z1 : z2 + 1].copy()

            # Store transient curve data
            transient_curves.append(
                {
                    "sign": float(sgn),
                    "creep_time": creep_t,
                    "creep_strain": creep_g,
                    "rec_time": rec_t_full,
                    "rec_strain": rec_g_full,
                    "nz_stress": nz_s,
                }
            )

            row = np.array([sgn, nz_t, nz_g, nz_s, z_t, z_g, z_s])
            if sgn > 0:
                rows_pos.append(row)
            else:
                rows_neg.append(row)

        # Store transient data on the transform instance for later use
        self._transient_curves = transient_curves

        return rows_pos, rows_neg

    # ==================================================================
    #  Numerical helpers
    # ==================================================================

    @staticmethod
    def _safe_dt(t: np.ndarray) -> float:
        d = np.diff(np.asarray(t, dtype=float))
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) if d.size else np.nan

    def _estimate_omega(self, t: np.ndarray, x: np.ndarray) -> float:
        """Estimate ω (rad/s) from the FFT peak of x(t)."""
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        mask = np.isfinite(t) & np.isfinite(x)
        t, x = t[mask], x[mask]
        if t.size < 8:
            return np.nan
        dt = self._safe_dt(t)
        if not np.isfinite(dt) or dt <= 0:
            return np.nan
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(x.size, d=dt) * 2.0 * np.pi
        pos = freqs >= 0
        amps = np.abs(X[pos])
        if amps.size < 3:
            return np.nan
        amps[0] = 0.0
        return float(freqs[pos][np.argmax(amps)])

    @staticmethod
    def _centered_derivative_periodic(X: np.ndarray, k: int, stencil: int, dt: float) -> np.ndarray:
        """
        Periodic centred finite-difference derivative.

        Faithful port of the MATLAB loop that uses modular indexing
        for periodic boundary conditions.

        Parameters
        ----------
        X       : signal array (length L, will be treated as periodic)
        k       : step multiplier
        stencil : 1 → 3-pt, 2 → 5-pt, 3 → 7-pt, 4 → 9-pt
        dt      : time step
        """
        X = np.asarray(X, dtype=float)
        L = X.shape[0]
        d = np.zeros_like(X, dtype=float)

        for p in range(L):
            p1 = (p + k) % L
            p2 = (p + 2 * k) % L
            p3 = (p + 3 * k) % L
            p4 = (p + 4 * k) % L

            pn1 = (p - k) % L
            pn2 = (p - 2 * k) % L
            pn3 = (p - 3 * k) % L
            pn4 = (p - 4 * k) % L

            if stencil == 1:
                d[p] = (X[p1] - X[pn1]) / (2.0 * k * dt)
            elif stencil == 2:
                d[p] = (-X[p2] + 8 * X[p1] - 8 * X[pn1] + X[pn2]) / (12.0 * k * dt)
            elif stencil == 3:
                d[p] = (X[p3] - 9 * X[p2] + 45 * X[p1] - 45 * X[pn1] + 9 * X[pn2] - X[pn3]) / (
                    60.0 * k * dt
                )
            elif stencil == 4:
                d[p] = (
                    -3 * X[p4]
                    + 32 * X[p3]
                    - 168 * X[p2]
                    + 672 * X[p1]
                    - 672 * X[pn1]
                    + 168 * X[pn2]
                    - 32 * X[pn3]
                    + 3 * X[pn4]
                ) / (840.0 * k * dt)
            else:
                raise ValueError("Choose stencil (diff_state) between 1 and 4.")

        return d

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _get_measurements(self, target) -> list[Measurement]:
        if isinstance(target, experiment_set.ExperimentSet):
            return list(target.measurements)
        if isinstance(target, MeasurementSet):
            return list(target.measurements)
        if hasattr(target, "measurements"):
            return list(target.measurements)
        return list(target)

    def _build_decomposition_dataset(self, decomp: np.ndarray):
        """Wrap the decomposition waveform in a 1-D composite dataset."""
        return one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=[decomp[:, i] for i in range(8)],
            data_array_names=[
                "time",
                "total strain",
                "unrecoverable strain",
                "recoverable strain",
                "total strain rate",
                "unrecoverable strain rate",
                "recoverable strain rate",
                "stress",
            ],
            data_array_units=["s", "–", "–", "–", "1/s", "1/s", "1/s", "Pa"],
            default_independent_name="time",
            default_dependent_name="stress",
            source="Recovery rheology decomposition",
        )


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def batch_oscillatory_recovery(
    directory: str,
    pattern: str = "*.csv",
    export_dir: str | None = None,
    **recovery_kwargs: Any,
) -> pd.DataFrame:
    """
    Run oscillatory recovery rheology on every RheoCompass file in a directory.

    Parameters
    ----------
    directory : str
        Folder containing one RheoCompass ``.csv`` export per amplitude.
    pattern : str
        Glob pattern used to select the files (default ``"*.csv"``).
    export_dir : str or None
        If given, a sub-folder per file is created here holding the Excel
        results and the Lissajous figures.  Nothing is written when ``None``.
    **recovery_kwargs
        Passed to :class:`RecoveryRheology` (``n_osc``, ``diff_state``, ...).

    Returns
    -------
    pandas.DataFrame
        One row per file, sorted by strain amplitude, with the columns
        ``file``, ``omega``, ``strain_amplitude``, ``stress_amplitude``,
        ``storage_modulus_solid``, ``loss_modulus_solid`` and
        ``loss_modulus_fluid``.
    """
    import os
    from glob import glob

    from hermes_rheo.file_readers.read import read_file
    from hermes_rheo.file_readers.rheo_compass_csv import RheoCompassReader

    paths = sorted(glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")

    rows = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        experiment = read_file(path, file_reader=RheoCompassReader, create_composite_dataset=True)
        recovery = RecoveryRheology(method="oscillatory", **recovery_kwargs)(experiment).result

        if export_dir is not None:
            file_dir = os.path.join(export_dir, stem)
            os.makedirs(file_dir, exist_ok=True)
            recovery.to_excel(os.path.join(file_dir, "recovery_oscillatory_results.xlsx"))
            figures, _ = recovery.plot_lissajous(figsize=(4, 4))
            names = ("total", "unrecoverable_rate", "recoverable")
            for name, figure in zip(names, figures):
                figure.savefig(
                    os.path.join(file_dir, f"lissajous_{name}.png"), dpi=300, bbox_inches="tight"
                )
                plt.close(figure)

        rows.append(
            {
                "file": os.path.basename(path),
                "omega": float(recovery.omega),
                "strain_amplitude": float(recovery.amplitudes[0]),
                "stress_amplitude": float(recovery.amplitudes[6]),
                "storage_modulus_solid": float(recovery.trad_mod[2]),
                "loss_modulus_solid": float(recovery.trad_mod[5]),
                "loss_modulus_fluid": float(recovery.trad_mod[4]),
            }
        )

    return pd.DataFrame(rows).sort_values("strain_amplitude").reset_index(drop=True)


def plot_decomposed_moduli(
    summary: pd.DataFrame,
    x_axis: str = "strain",
    title: str | None = None,
    figsize: tuple[float, float] = (6, 4),
):
    """
    Plot G'_solid, G''_solid and G''_fluid against the oscillation amplitude.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`batch_oscillatory_recovery`.
    x_axis : {"strain", "stress"}
        Amplitude shown on the horizontal axis.
    title : str or None
        Axis title.  Defaults to the median angular frequency.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if x_axis not in {"strain", "stress"}:
        raise ValueError("x_axis must be 'strain' or 'stress'")

    if x_axis == "strain":
        x_values = 100 * summary["strain_amplitude"]
        x_label = r"Strain amplitude $\gamma_0$ [%]"
    else:
        x_values = summary["stress_amplitude"]
        x_label = r"Stress amplitude $\sigma_0$ [Pa]"

    figure, axis = plt.subplots(figsize=figsize)
    series = (
        ("storage_modulus_solid", "o-", r"$G'_{\mathrm{solid}}$"),
        ("loss_modulus_solid", "s-", r"$G''_{\mathrm{solid}}$"),
        ("loss_modulus_fluid", "^-", r"$G''_{\mathrm{fluid}}$"),
    )
    for column, marker, label in series:
        axis.plot(x_values, summary[column], marker, label=label)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Modulus [Pa]")
    if title is None:
        title = rf"$\omega = {summary['omega'].median():.3g}$ rad/s"
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    return figure, axis
