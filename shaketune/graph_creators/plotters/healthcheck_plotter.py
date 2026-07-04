# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: healthcheck_plotter.py
# Description: Plotter for the SHAKETUNE_HEALTHCHECK graph

from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..base_models import PlotterStrategy
from ..computation_results import HealthCheckResult
from ..plotting_utils import AxesConfiguration, PlottingConstants

AXIS_ORDER = ['X', 'Y']  # Preferred display order when both axes are present

# The figure width is fixed while its height grows with the number of measured axes (1 or 2).
# add_logo()'s default position assumes a fixed (15, 7) figsize, so on this narrower/taller
# canvas the logo is pinned to an explicit square physical size instead (same trick as
# trend_plotter.py), and the title's left edge is derived from it.
FIG_WIDTH_IN = 12.0
PANEL_TOP_MARGIN_IN = 1.05
PANEL_BOTTOM_MARGIN_IN = 0.55
PANEL_HEIGHT_IN = 3.6
LOGO_MARGIN_IN = 0.05
LOGO_SIZE_IN = 0.75
TITLE_X_IN = 1.05


class HealthCheckPlotter(PlotterStrategy):
    """Plotter for the SHAKETUNE_HEALTHCHECK graph: one stacked panel per measured axis (X, Y),
    each showing the current PSD against the stored baseline (when available) with resonant
    frequency markers, so mechanical drift shows up as a visibly shifted/reshaped curve. PASS/WARN
    verdict text isn't part of this plot: it's computed and printed parent-side (see
    helpers/healthcheck_report.py) since it needs to reach the gcode console, not just the PNG"""

    def plot(self, result: HealthCheckResult) -> Figure:
        data = result.get_plot_data()
        per_axis: Dict[str, Any] = data.get('per_axis') or {}
        baseline: Dict[str, Any] = data.get('baseline') or {}
        baseline_axes: Dict[str, Any] = baseline.get('per_axis') or {}
        version = data.get('st_version', 'unknown')

        axes_present = [a for a in AXIS_ORDER if a in per_axis] or sorted(per_axis)
        if not axes_present:
            return self._plot_placeholder(version)

        n_panels = len(axes_present)
        fig_height = PANEL_TOP_MARGIN_IN + PANEL_BOTTOM_MARGIN_IN + PANEL_HEIGHT_IN * n_panels
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=(FIG_WIDTH_IN, fig_height),
            gridspec_kw={
                'top': 1 - (PANEL_TOP_MARGIN_IN / fig_height),
                'bottom': PANEL_BOTTOM_MARGIN_IN / fig_height,
                'left': 0.09,
                'right': 0.96,
                'hspace': 0.5,
            },
        )
        if n_panels == 1:
            axes = [axes]

        for ax, axis in zip(axes, axes_present):
            self._plot_axis_panel(ax, axis, per_axis.get(axis) or {}, baseline_axes.get(axis))

        self._add_titles(fig, data, has_baseline=bool(baseline_axes), fig_height=fig_height)
        self.add_logo(fig, position=self._logo_position(fig_height))
        self.add_version_text(fig, version)

        return fig

    @staticmethod
    def _logo_position(fig_height: float) -> List[float]:
        """Logo axes rect (figure fraction) that always renders as a LOGO_SIZE_IN square in the
        top-left corner, regardless of the figure's height"""
        return [
            LOGO_MARGIN_IN / FIG_WIDTH_IN,
            1 - (LOGO_MARGIN_IN + LOGO_SIZE_IN) / fig_height,
            LOGO_SIZE_IN / FIG_WIDTH_IN,
            LOGO_SIZE_IN / fig_height,
        ]

    def _plot_axis_panel(self, ax, axis: str, current: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> None:
        """Plot one axis' current PSD, overlaid with its baseline PSD (dashed) when available,
        with vertical markers at the current (and baseline) resonant frequency"""
        freqs = current.get('freqs')
        psd = current.get('psd')

        if not freqs or not psd:
            ax.axis('off')
            ax.text(
                0.5,
                0.5,
                f'No {axis} axis data available',
                ha='center',
                va='center',
                fontsize=12,
                color=PlottingConstants.KLIPPAIN_COLORS['dark_purple'],
                transform=ax.transAxes,
            )
            return

        ax.plot(freqs, psd, label='Current', color=PlottingConstants.KLIPPAIN_COLORS['orange'], zorder=3)

        fr = current.get('fr')
        if fr is not None:
            ax.axvline(
                fr,
                color=PlottingConstants.KLIPPAIN_COLORS['orange'],
                linestyle=':',
                linewidth=1.5,
                label=f'Current fr: {fr:.1f} Hz',
            )

        if baseline and baseline.get('freqs') and baseline.get('psd'):
            ax.plot(
                baseline['freqs'],
                baseline['psd'],
                label='Baseline',
                color=PlottingConstants.KLIPPAIN_COLORS['purple'],
                linestyle='--',
                zorder=2,
            )
            baseline_fr = baseline.get('fr')
            if baseline_fr is not None:
                ax.axvline(
                    baseline_fr,
                    color=PlottingConstants.KLIPPAIN_COLORS['purple'],
                    linestyle='--',
                    linewidth=1.5,
                    label=f'Baseline fr: {baseline_fr:.1f} Hz',
                )
        else:
            ax.text(
                0.99,
                0.95,
                'No baseline recorded yet for this axis',
                ha='right',
                va='top',
                fontsize=9,
                style='italic',
                color=PlottingConstants.KLIPPAIN_COLORS['dark_purple'],
                transform=ax.transAxes,
            )

        ax.set_xlim([freqs[0], freqs[-1]])
        ax.set_ylim(bottom=0)
        AxesConfiguration.configure_axes(
            ax,
            xlabel='Frequency (Hz)',
            ylabel='Power spectral density',
            title=f'{axis} axis',
            sci_axes='y',
            legend=True,
        )

    def _add_titles(self, fig: Figure, data: Dict[str, Any], has_baseline: bool, fig_height: float) -> None:
        """Add the figure title, positioned at a fixed distance from the top/left edges
        regardless of how many panels (and therefore how tall the figure) ended up being
        rendered, and clear of the logo (see _logo_position)"""
        title_line2 = self._format_date_subtitle(data)
        title_line3 = (
            'Compared against stored baseline'
            if has_baseline
            else 'No baseline recorded yet -- run MODE=BASELINE first'
        )

        title_x = TITLE_X_IN / FIG_WIDTH_IN
        title_y = 1 - (0.35 / fig_height)
        subtitle_y = 1 - (0.62 / fig_height)
        subtitle2_y = 1 - (0.85 / fig_height)
        self.add_title(
            fig,
            [
                {
                    'x': title_x,
                    'y': title_y,
                    'va': 'top',
                    'text': 'SHAKE&TUNE HEALTHCHECK',
                    'fontsize': 20,
                    'color': PlottingConstants.KLIPPAIN_COLORS['purple'],
                    'weight': 'bold',
                },
                {'x': title_x, 'y': subtitle_y, 'va': 'top', 'fontsize': 11, 'text': title_line2},
                {'x': title_x, 'y': subtitle2_y, 'va': 'top', 'fontsize': 11, 'text': title_line3},
            ],
        )

    @staticmethod
    def _format_date_subtitle(data: Dict[str, Any]) -> str:
        """Best-effort extraction of a human-readable run date from the first measurement's
        name (e.g. 'axis_X_20260704_101010'), falling back to the raw name on any parsing issue"""
        try:
            measurements = data.get('measurements') or []
            filename = measurements[0]['name']
            parts = filename.split('_')
            dt = datetime.strptime(f'{parts[2]} {parts[3]}', '%Y%m%d %H%M%S')
            return dt.strftime('%x %X')
        except Exception:
            measurements = data.get('measurements') or []
            return measurements[0]['name'] if measurements else ''

    def _plot_placeholder(self, version: str) -> Figure:
        """Render a clean 'no data' placeholder figure instead of an empty/broken plot"""
        fig_height = 5.0
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, fig_height))
        ax.axis('off')
        ax.text(
            0.5,
            0.5,
            'No healthcheck data available',
            ha='center',
            va='center',
            fontsize=16,
            color=PlottingConstants.KLIPPAIN_COLORS['dark_purple'],
            transform=ax.transAxes,
        )

        self.add_title(
            fig,
            [
                {
                    'x': TITLE_X_IN / FIG_WIDTH_IN,
                    'y': 1 - (0.35 / fig_height),
                    'va': 'top',
                    'text': 'SHAKE&TUNE HEALTHCHECK',
                    'fontsize': 20,
                    'color': PlottingConstants.KLIPPAIN_COLORS['purple'],
                    'weight': 'bold',
                },
            ],
        )
        self.add_logo(fig, position=self._logo_position(fig_height))
        self.add_version_text(fig, version)

        return fig
