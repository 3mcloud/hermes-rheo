"""OWChirp visualisation helpers for processed OWChirp experiments.

This module wraps the boilerplate needed to visualise the output of
:class:`hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis` into a
single convenient object, :class:`OWChirpVisualizer`.

After running ``OWChirpAnalysis`` each measurement carries two datasets:

* ``datasets[0]`` — the frequency-domain moduli
  (``angular frequency``, ``storage modulus``, ``loss modulus``,
  ``complex modulus``, ``complex viscosity``, ``tan(delta)`` …).
* ``datasets[1]`` — the bias-corrected time-domain signal
  (``adjusted time``/``step time``, ``adjusted strain``,
  ``adjusted stress``, ``temperature``).

:class:`OWChirpVisualizer` reads those datasets once and exposes methods
to plot individual waves, save every wave as a PNG frame, stitch the
frames into an animation, and inspect the Fourier-domain strain/stress
signals — without exposing the extraction/plotting boilerplate in a
notebook.

Usage::

    from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
    from hermes_rheo.visualization import OWChirpVisualizer

    processed = OWChirpAnalysis()(experiment)
    v = OWChirpVisualizer(processed)

    v.plot_waves(steps=[0, 1, 2])   # plot waves by measurement index
    v.save_frames("plots")          # write every wave PNG
    v.to_video("plots")             # stitch frames -> .avi
    v.plot_fourier(step=10)         # Fourier-domain analysis
"""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

StepSelector = int | str | Sequence[int] | None


class OWChirpVisualizer:
    """Convenience visualiser for a processed OWChirp experiment.

    Parameters
    ----------
    processed_experiment : piblin.data.ExperimentSet
        The result of applying
        :class:`~hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis`
        to an OWChirp (Arbitrary Wave / Frequency Chirp) experiment. Each
        step must carry the frequency-domain dataset at ``datasets[0]``
        and the time-domain dataset at ``datasets[1]``.
    moduli_ylim : tuple of float, optional
        ``(ymin, ymax)`` limits for the moduli axis in :meth:`plot_wave`.
        Defaults to ``(1e3, 5e8)``.

    Attributes
    ----------
    n_waves : int
        Number of OWChirp waves (steps) in the experiment.

    Examples
    --------
    >>> from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
    >>> from hermes_rheo.visualization import OWChirpVisualizer
    >>> processed = OWChirpAnalysis()(experiment)          # doctest: +SKIP
    >>> viz = OWChirpVisualizer(processed)                 # doctest: +SKIP
    >>> figs, axes = viz.plot_waves(steps="first-middle-last")  # doctest: +SKIP
    """

    def __init__(
        self,
        processed_experiment: Any,
        moduli_ylim: tuple[float, float] = (1e3, 5e8),
    ) -> None:
        self._experiment = processed_experiment
        self.moduli_ylim = moduli_ylim
        self._meas = list(processed_experiment.measurements)
        self.n_waves = len(self._meas)
        self._data: dict[str, list[np.ndarray]] | None = None

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _xy(dataset, x_name: str, y_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, y)`` for two named arrays without mutating *dataset*."""
        ds = copy.deepcopy(dataset)
        ds.switch_coordinates(independent_name=x_name, dependent_name=y_name)
        return copy.deepcopy(ds.x_values), copy.deepcopy(ds.y_values)

    @property
    def data(self) -> dict[str, list[np.ndarray]]:
        """Per-wave extracted arrays (lazily computed and cached).

        Returns
        -------
        dict of list of numpy.ndarray
            Keys: ``angular_frequencies``, ``storage_moduli``,
            ``loss_moduli``, ``complex_moduli``, ``complex_viscosities``,
            ``phase_angles``, ``step_times``, ``strains``, ``stresses``,
            ``temperatures``. Each value is a list of length
            :attr:`n_waves`.
        """
        if self._data is not None:
            return self._data

        keys = [
            "angular_frequencies",
            "storage_moduli",
            "loss_moduli",
            "complex_moduli",
            "complex_viscosities",
            "phase_angles",
            "step_times",
            "strains",
            "stresses",
            "temperatures",
        ]
        data: dict[str, list[np.ndarray]] = {k: [] for k in keys}

        for step in range(self.n_waves):
            freq = self._meas[step].datasets[0]
            time = self._meas[step].datasets[1]

            omega, storage = self._xy(freq, "angular frequency", "storage modulus")
            loss, complex_mod = self._xy(freq, "loss modulus", "complex modulus")
            complex_visc, phase = self._xy(freq, "complex viscosity", "tan(delta)")

            t, strain = self._xy(time, "adjusted time", "adjusted strain")
            temperature, stress = self._xy(time, "temperature", "adjusted stress")

            data["angular_frequencies"].append(omega)
            data["storage_moduli"].append(storage)
            data["loss_moduli"].append(loss)
            data["complex_moduli"].append(complex_mod)
            data["complex_viscosities"].append(complex_visc)
            data["phase_angles"].append(phase)
            data["step_times"].append(t)
            data["strains"].append(strain)
            data["stresses"].append(stress)
            data["temperatures"].append(temperature)

        self._data = data
        return data

    # ------------------------------------------------------------------
    # Step selection
    # ------------------------------------------------------------------

    def _resolve_steps(self, steps: StepSelector) -> list[int]:
        """Turn a step selector into a concrete, sorted list of indices.

        Parameters
        ----------
        steps : int, str, sequence of int, or None
            * ``None`` or ``"all"`` — every wave.
            * ``"first-middle-last"`` (or ``"endpoints"``) — the first,
              middle and last wave.
            * ``int`` — a single wave.
            * sequence of int — those waves.

        Returns
        -------
        list of int
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

    # ------------------------------------------------------------------
    # Per-wave moduli / signal plot
    # ------------------------------------------------------------------

    def plot_wave(
        self,
        step: int,
        figsize: tuple[float, float] = (12, 5),
        save_dir: str | None = None,
    ) -> tuple[matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]]:
        """Plot the moduli and time-domain signal for a single wave.

        The layout is a moduli panel (``G'``/``G''`` vs angular frequency)
        on the left and the input strain and output stress traces on the
        right.

        Parameters
        ----------
        step : int
            Index of the wave to plot (0-based).
        figsize : tuple of float, optional
            Figure size. Defaults to ``(12, 5)``.
        save_dir : str, optional
            When given, the figure is written to
            ``<save_dir>/plot_<step+1>.png`` at 100 dpi.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : tuple of matplotlib.axes.Axes
            ``(moduli_ax, strain_ax, stress_ax)``.
        """
        d = self.data
        f = d["angular_frequencies"][step]
        stor = d["storage_moduli"][step]
        loss = d["loss_moduli"][step]
        t = d["step_times"][step]
        strain = d["strains"][step]
        stress = d["stresses"][step]
        average_temperature = np.average(d["temperatures"][step])
        title = f"Arbitrary Wave {step + 1}, Temperature = {average_temperature:.2f} \u00b0C"

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 2])

        ax_mod = plt.subplot(gs[:, 0])
        ax_mod.plot(f, stor, marker="s", label=r"$G'$", color="red", markersize=10, linestyle="-")
        ax_mod.plot(
            f,
            loss,
            marker="^",
            label=r"$G''$",
            color="blue",
            markersize=10,
            markerfacecolor="none",
            linestyle="--",
        )
        ax_mod.set_title(title, fontsize=15)
        ax_mod.set_xscale("log")
        ax_mod.set_yscale("log")
        ax_mod.set_xlabel(r"$\omega$ (rad/s)", fontsize=20)
        ax_mod.set_ylabel(r"$G'$, $G''$ (Pa)", fontsize=20)
        ax_mod.legend(loc="best", frameon=False, prop={"size": 15}, ncol=1)
        ax_mod.set_ylim(*self.moduli_ylim)

        ax_strain = plt.subplot(gs[0, 1])
        ax_strain.plot(t, strain, linewidth=2, color="seagreen")
        ax_strain.set_xlabel("Time (s)", fontsize=16)
        ax_strain.set_ylabel("Strain", fontsize=16)
        ax_strain.axhline(0, color="black", linestyle="--", linewidth=1)

        ax_stress = plt.subplot(gs[1, 1])
        ax_stress.plot(t, stress, linewidth=2, color="darkorange")
        ax_stress.set_xlabel("Time (s)", fontsize=16)
        ax_stress.set_ylabel("Stress (Pa)", fontsize=16)
        ax_stress.axhline(0, color="black", linestyle="--", linewidth=1)

        fig.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"plot_{step + 1}.png"), dpi=100)

        return fig, (ax_mod, ax_strain, ax_stress)

    def plot_waves(
        self,
        steps: StepSelector = "first-middle-last",
        figsize: tuple[float, float] = (12, 5),
        save_dir: str | None = None,
    ) -> tuple[list[matplotlib.figure.Figure], list[Sequence[matplotlib.axes.Axes]]]:
        """Plot a selection of waves.

        Parameters
        ----------
        steps : int, str, or sequence of int, optional
            Which waves to plot; see :meth:`_resolve_steps`. Defaults to
            ``"first-middle-last"``.
        figsize : tuple of float, optional
            Size of each figure. Defaults to ``(12, 5)``.
        save_dir : str, optional
            When given, each figure is also written to
            ``<save_dir>/plot_<step+1>.png``.

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
        """Render and save every wave as a PNG frame (no inline display).

        Parameters
        ----------
        output_dir : str
            Directory to write ``plot_<n>.png`` files into (created if
            necessary).

        Returns
        -------
        list of str
            The paths of the written PNG files, ordered by wave.
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
        """Stitch the per-wave PNG frames into a video.

        Frames are (re)generated with :meth:`save_frames` if they are not
        already present in *output_dir*, then combined in wave order.

        Parameters
        ----------
        output_dir : str
            Directory holding (or to hold) the ``plot_<n>.png`` frames and
            the output video.
        filename : str, optional
            Output video filename. Defaults to ``"plot_video.avi"``.
        fps : int, optional
            Frames per second. Defaults to ``2``.
        codec : str, optional
            FourCC codec string. Defaults to ``"DIVX"``.

        Returns
        -------
        str
            Path to the written video file.

        Raises
        ------
        ImportError
            If ``opencv-python`` (``cv2``) is not installed.
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
            writer.write(cv2.imread(frame_path))
        writer.release()
        return video_path

    # ------------------------------------------------------------------
    # Fourier-domain analysis
    # ------------------------------------------------------------------

    def plot_fourier(
        self,
        step: int = 0,
        figsize: tuple[float, float] = (10, 7),
    ) -> tuple[matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]]:
        """Plot the time- and Fourier-domain strain/stress for one wave.

        The 2x2 layout shows the input strain (top-left) and output stress
        (bottom-left) in the time domain, and the real, imaginary and
        absolute parts of their Fourier transforms (right column).

        Parameters
        ----------
        step : int, optional
            Index of the wave to analyse. Defaults to ``0``.
        figsize : tuple of float, optional
            Figure size. Defaults to ``(10, 7)``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : tuple of matplotlib.axes.Axes
            ``(strain_time_ax, strain_fft_ax, stress_time_ax, stress_fft_ax)``.
        """
        freq = self._meas[step].datasets[0]
        f, strain_hat = self._xy(freq, "angular frequency", "fourier_transform_strain")
        _, stress_hat = self._xy(freq, "angular frequency", "fourier_transform_stress")

        time_ds = self._meas[step].datasets[1]
        t, strain = self._xy(time_ds, "step time", "adjusted strain")
        _, stress = self._xy(time_ds, "step time", "adjusted stress")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        ax1.plot(t, strain, "blue", label="Original Strain")
        ax1.set_xlabel("t (s)")
        ax1.set_ylabel(r"$\gamma$")
        ax1.set_xlim(0, 15)

        ax2.plot(f, np.real(strain_hat), "black", label=r"$\mathfrak{Re}(\tilde\gamma)$")
        ax2.plot(f, np.imag(strain_hat), "lightgreen", label=r"$\mathfrak{Im}(\tilde\gamma)$")
        ax2.plot(
            f,
            np.abs(strain_hat),
            "blue",
            label=r"$\mid\mathfrak{Re}(\tilde\gamma) + i\mathfrak{Im}(\tilde\gamma)\mid$",
        )
        ax2.set_xlabel(r"$\omega$ (rad/s)")
        ax2.set_ylabel(r"$\tilde\gamma$")
        ax2.set_xlim(0, 60)
        ax2.legend(fontsize=8).get_frame().set_edgecolor("white")

        ax3.plot(t, stress, "red", label="Original Stress")
        ax3.set_xlabel("t (s)")
        ax3.set_ylabel(r"$\sigma$ (Pa)")
        ax3.set_xlim(0, 15)

        ax4.plot(f, np.real(stress_hat), "black", label=r"$\mathfrak{Re}(\tilde\sigma)$")
        ax4.plot(f, np.imag(stress_hat), "lightgreen", label=r"$\mathfrak{Im}(\tilde\sigma)$")
        ax4.plot(
            f,
            np.abs(stress_hat),
            "red",
            label=r"$\mid\mathfrak{Re}(\tilde\sigma) + i\mathfrak{Im}(\tilde\sigma)\mid$",
        )
        ax4.set_xlabel(r"$\omega$ (rad/s)")
        ax4.set_ylabel(r"$\tilde\sigma$")
        ax4.set_xlim(0, 60)
        ax4.legend(fontsize=8).get_frame().set_edgecolor("white")

        fig.tight_layout()
        return fig, (ax1, ax2, ax3, ax4)
