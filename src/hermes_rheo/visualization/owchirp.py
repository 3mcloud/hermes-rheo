"""Plots for processed OWChirp (optimally windowed chirp) experiments.

OWChirp probes a broad frequency range in one short arbitrary-wave step by
applying a windowed chirp and Fourier-transforming the strain and stress
signals to obtain ``G'`` and ``G''`` [1]_.

After :class:`~hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis`,
each measurement holds two datasets:

- ``datasets[0]``: frequency domain (angular frequency, storage and loss
  modulus, complex modulus and viscosity, tan(delta), Fourier transforms of
  strain and stress);
- ``datasets[1]``: time domain (adjusted time, step time, adjusted strain,
  adjusted stress, temperature).

Axis labels and units are read from these datasets.

References
----------
.. [1] Geri, M., Keshavarz, B., McKinley, G. H., and Bonnecaze, R. T.,
   Phys. Rev. X 8, 041042 (2018).

Examples
--------
>>> from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
>>> from hermes_rheo.visualization import OWChirpVisualizer
>>> viz = OWChirpVisualizer(OWChirpAnalysis()(experiment))  # doctest: +SKIP
>>> viz.plot_waves(steps=[0, 1, 2])  # doctest: +SKIP
>>> viz.plot_fourier(step=10)  # doctest: +SKIP
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from hermes_rheo.visualization import _common as c
from hermes_rheo.visualization import style
from hermes_rheo.visualization.units import axis_label, get_unit

StepSelector = int | str | Sequence[int] | None

# (data key, dataset index, array name)
_DATA_ARRAYS = (
    ("angular_frequencies", 0, c.ANGULAR_FREQUENCY),
    ("storage_moduli", 0, c.STORAGE_MODULUS),
    ("loss_moduli", 0, c.LOSS_MODULUS),
    ("complex_moduli", 0, c.COMPLEX_MODULUS),
    ("complex_viscosities", 0, c.COMPLEX_VISCOSITY),
    ("phase_angles", 0, c.TAN_DELTA),
    ("step_times", 1, c.ADJUSTED_TIME),
    ("strains", 1, c.ADJUSTED_STRAIN),
    ("stresses", 1, c.ADJUSTED_STRESS),
    ("temperatures", 1, c.TEMPERATURE),
)


class OWChirpVisualizer:
    """Plot the waves of a processed OWChirp experiment.

    Parameters
    ----------
    processed_experiment : ExperimentSet
        Output of
        :class:`~hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis`.
    moduli_ylim : tuple of float, optional
        ``(ymin, ymax)`` of the modulus axis in :meth:`plot_wave`.
        Autoscaled if ``None``.

    Attributes
    ----------
    n_waves : int
        Number of waves (measurements).
    """

    def __init__(
        self,
        processed_experiment: Any,
        moduli_ylim: tuple[float, float] | None = None,
    ) -> None:
        self._experiment = processed_experiment
        self.moduli_ylim = moduli_ylim
        self._meas = list(processed_experiment.measurements)
        self.n_waves = len(self._meas)
        self._data: dict[str, list[np.ndarray]] | None = None

    @property
    def data(self) -> dict[str, list[np.ndarray]]:
        """Arrays of every wave, computed once.

        Returns
        -------
        dict of list of numpy.ndarray
            Keys are ``angular_frequencies``, ``storage_moduli``,
            ``loss_moduli``, ``complex_moduli``, ``complex_viscosities``,
            ``phase_angles``, ``step_times``, ``strains``, ``stresses`` and
            ``temperatures``; each list has ``n_waves`` entries. Arrays
            missing from a dataset are returned as empty arrays.
        """
        if self._data is None:
            data: dict[str, list[np.ndarray]] = {key: [] for key, _, _ in _DATA_ARRAYS}
            for meas in self._meas:
                for key, index, name in _DATA_ARRAYS:
                    values = c.array_by_name(meas.datasets[index], name)
                    data[key].append(np.array([]) if values is None else values.copy())
            self._data = data
        return self._data

    def _resolve_steps(self, steps: StepSelector) -> list[int]:
        """Return the sorted wave indices selected by *steps*.

        Parameters
        ----------
        steps : None, str, int or sequence of int
            ``None`` or ``"all"`` selects every wave; ``"first-middle-last"``
            (or ``"endpoints"``) selects the first, middle and last wave;
            an int or a sequence of ints selects those waves.

        Returns
        -------
        list of int

        Raises
        ------
        ValueError
            If *steps* is an unknown string.
        """
        n = self.n_waves
        if steps is None or steps == "all":
            return list(range(n))
        if isinstance(steps, str):
            if steps in ("first-middle-last", "endpoints"):
                return sorted({0, n // 2, n - 1})
            raise ValueError(
                f"Unknown step selector {steps!r}. Use 'all', "
                "'first-middle-last', an int, or a sequence of ints."
            )
        if isinstance(steps, int):
            return [steps]
        return sorted({int(s) for s in steps})

    def _title(self, step: int) -> str:
        """Return ``"Arbitrary Wave N"`` plus the mean temperature if known."""
        title = f"Arbitrary Wave {step + 1}"
        temperature = self.data["temperatures"][step]
        if temperature.size:
            unit = c.temperature_unit_symbol(get_unit(self._meas[step].datasets[1], c.TEMPERATURE))
            title += f", {c.TEMPERATURE} = {np.mean(temperature):.2f} {unit}".rstrip()
        return title

    def plot_wave(
        self,
        step: int,
        figsize: tuple[float, float] = style.FIGSIZE_WIDE,
        save_dir: str | None = None,
    ) -> tuple[matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]]:
        """Plot the moduli, strain and stress of one wave.

        The moduli versus angular frequency are on the left; the strain and
        stress versus time are stacked on the right.

        Parameters
        ----------
        step : int
            Wave index (0-based).
        figsize : tuple of float, optional
            Figure size.
        save_dir : str, optional
            If given, write the figure to ``<save_dir>/plot_<step+1>.png``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : tuple of matplotlib.axes.Axes
            ``(moduli_ax, strain_ax, stress_ax)``.
        """
        freq_ds, time_ds = self._meas[step].datasets[:2]
        d = self.data

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
        ax_mod = fig.add_subplot(gs[:, 0])
        ax_strain = fig.add_subplot(gs[0, 1])
        ax_stress = fig.add_subplot(gs[1, 1])

        omega = d["angular_frequencies"][step]
        ax_mod.plot(
            omega,
            d["storage_moduli"][step],
            marker=style.MARKER_STORAGE,
            color=style.COLOR_STORAGE,
            linewidth=style.LINEWIDTH,
            markersize=style.MARKERSIZE,
            label=c.STORAGE_MODULUS,
        )
        ax_mod.plot(
            omega,
            d["loss_moduli"][step],
            marker=style.MARKER_LOSS,
            color=style.COLOR_LOSS,
            linewidth=style.LINEWIDTH,
            markersize=style.MARKERSIZE,
            label=c.LOSS_MODULUS,
        )
        c.decorate_axes(
            ax_mod,
            xlabel=axis_label(freq_ds, c.ANGULAR_FREQUENCY),
            ylabel=axis_label(freq_ds, c.STORAGE_MODULUS, text="modulus"),
            title=self._title(step),
            logx=True,
            logy=True,
            ylim=self.moduli_ylim,
        )
        c.add_legend(ax_mod)

        time_label = axis_label(time_ds, c.ADJUSTED_TIME, text="time")
        for ax, key, name, color in (
            (ax_strain, "strains", c.ADJUSTED_STRAIN, style.COLOR_STRAIN),
            (ax_stress, "stresses", c.ADJUSTED_STRESS, style.COLOR_STRESS),
        ):
            ax.plot(d["step_times"][step], d[key][step], linewidth=style.LINEWIDTH, color=color)
            ax.axhline(0, color=style.COLOR_GUIDE, linestyle="--", linewidth=style.LINEWIDTH_THIN)
            c.decorate_axes(
                ax,
                xlabel=time_label,
                ylabel=axis_label(time_ds, name, text=name.removeprefix("adjusted ")),
            )

        fig.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"plot_{step + 1}.png"), dpi=100)
        return fig, (ax_mod, ax_strain, ax_stress)

    def plot_waves(
        self,
        steps: StepSelector = "first-middle-last",
        figsize: tuple[float, float] = style.FIGSIZE_WIDE,
        save_dir: str | None = None,
    ) -> tuple[list[matplotlib.figure.Figure], list[Sequence[matplotlib.axes.Axes]]]:
        """Plot several waves with :meth:`plot_wave`.

        Parameters
        ----------
        steps : None, str, int or sequence of int, optional
            Waves to plot; see :meth:`_resolve_steps`.
        figsize : tuple of float, optional
            Size of each figure.
        save_dir : str, optional
            If given, write each figure to ``<save_dir>/plot_<step+1>.png``.

        Returns
        -------
        figs : list of matplotlib.figure.Figure
        axes : list of tuple of matplotlib.axes.Axes
        """
        figs, axes = [], []
        for step in self._resolve_steps(steps):
            fig, ax = self.plot_wave(step, figsize=figsize, save_dir=save_dir)
            figs.append(fig)
            axes.append(ax)
        return figs, axes

    def save_frames(self, output_dir: str) -> list[str]:
        """Save every wave as ``plot_<n>.png`` without showing it.

        Parameters
        ----------
        output_dir : str
            Output directory, created if needed.

        Returns
        -------
        list of str
            Paths of the PNG files, in wave order.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for step in range(self.n_waves):
            fig, _ = self.plot_wave(step, save_dir=output_dir)
            plt.close(fig)
            paths.append(os.path.join(output_dir, f"plot_{step + 1}.png"))
        return paths

    def to_video(
        self,
        output_dir: str,
        filename: str = "plot_video.avi",
        fps: int = 2,
        codec: str = "DIVX",
    ) -> str:
        """Combine the ``plot_<n>.png`` frames into a video.

        Frames are created with :meth:`save_frames` if *output_dir* has none.

        Parameters
        ----------
        output_dir : str
            Directory of the frames and the output video.
        filename : str, optional
            Video file name.
        fps : int, optional
            Frames per second.
        codec : str, optional
            FourCC codec code.

        Returns
        -------
        str
            Path of the video file.

        Raises
        ------
        ImportError
            If ``opencv-python`` is not installed.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "to_video() requires opencv-python. Install it with "
                '`pip install "hermes-rheo[notebooks]"` or '
                "`pip install opencv-python`."
            ) from exc

        def _frame_paths() -> list[str]:
            return sorted(
                (
                    os.path.join(output_dir, f)
                    for f in os.listdir(output_dir)
                    if f.startswith("plot_") and f.endswith(".png")
                ),
                key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
            )

        if not os.path.isdir(output_dir) or not _frame_paths():
            self.save_frames(output_dir)
        frames = _frame_paths()

        first = cv2.imread(frames[0])
        height, width = first.shape[:2]
        video_path = os.path.join(output_dir, filename)
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        for frame_path in frames:
            frame = cv2.imread(frame_path)
            # The writer drops frames whose size differs from the first one.
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        writer.release()
        return video_path

    def plot_fourier(
        self,
        step: int = 0,
        figsize: tuple[float, float] = style.FIGSIZE_GRID,
    ) -> tuple[matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]]:
        """Plot strain and stress in the time and Fourier domains.

        The left column shows the strain (top) and stress (bottom) versus
        time. The right column shows the real part, imaginary part and
        magnitude of their Fourier transforms versus angular frequency.

        Parameters
        ----------
        step : int, optional
            Wave index (0-based).
        figsize : tuple of float, optional
            Figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : tuple of matplotlib.axes.Axes
            ``(strain_time, strain_fourier, stress_time, stress_fourier)``.
        """
        freq_ds, time_ds = self._meas[step].datasets[:2]
        omega = c.array_by_name(freq_ds, c.ANGULAR_FREQUENCY)
        t = c.array_by_name(time_ds, c.STEP_TIME)
        if t is None:
            t = c.array_by_name(time_ds, c.ADJUSTED_TIME)
        time_label = axis_label(time_ds, c.STEP_TIME, text="time")
        omega_label = axis_label(freq_ds, c.ANGULAR_FREQUENCY)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        rows = (
            (c.ADJUSTED_STRAIN, c.FOURIER_STRAIN, style.COLOR_STRAIN),
            (c.ADJUSTED_STRESS, c.FOURIER_STRESS, style.COLOR_STRESS),
        )
        for (time_name, fourier_name, color), (ax_time, ax_ft) in zip(rows, axes):
            signal = c.array_by_name(time_ds, time_name)
            spectrum = c.array_by_name(freq_ds, fourier_name)
            quantity = time_name.removeprefix("adjusted ")

            ax_time.plot(t, signal, color=color, linewidth=style.LINEWIDTH, label=quantity)
            c.decorate_axes(
                ax_time,
                xlabel=time_label,
                ylabel=axis_label(time_ds, time_name, text=quantity),
            )

            for part, values, part_color in (
                ("real", np.real(spectrum), style.COLOR_GUIDE),
                ("imaginary", np.imag(spectrum), style.COLOR_MARK),
                ("magnitude", np.abs(spectrum), color),
            ):
                ax_ft.plot(
                    omega, values, color=part_color, linewidth=style.LINEWIDTH_THIN, label=part
                )
            c.decorate_axes(
                ax_ft,
                xlabel=omega_label,
                ylabel=axis_label(freq_ds, fourier_name, text=f"FT of {quantity}"),
            )
            c.add_legend(ax_ft)

        fig.tight_layout()
        return fig, (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
