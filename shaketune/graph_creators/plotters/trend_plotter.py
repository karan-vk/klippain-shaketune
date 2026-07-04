# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: trend_plotter.py
# Description: Plotter for the metrics-history trend graph

from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..base_models import PlotterStrategy
from ..computation_results import TrendResult
from ..plotting_utils import AxesConfiguration, PlottingConstants

# One subplot ("panel") per metric family. Each entry is (panel_title, y-axis label for the
# primary (left) axis, matcher for the primary series, secondary (right, twinx) label, matcher
# for the secondary series). A family is only rendered when at least one of its series is present
_PANEL_COLORS = [
    PlottingConstants.KLIPPAIN_COLORS['purple'],
    PlottingConstants.KLIPPAIN_COLORS['orange'],
    PlottingConstants.KLIPPAIN_COLORS['dark_orange'],
    PlottingConstants.KLIPPAIN_COLORS['red_pink'],
    PlottingConstants.KLIPPAIN_COLORS['dark_purple'],
]

PANEL_TOP_MARGIN_IN = 1.15
PANEL_BOTTOM_MARGIN_IN = 0.55
PANEL_HEIGHT_IN = 3.0

# The figure width is fixed while its height grows with the number of panels. add_logo()'s
# default position assumes a fixed (15, 7) figsize, so on a narrower/taller canvas the logo
# would render disproportionately large (matplotlib scales an 'equal' aspect image to the
# SMALLER of the box's two physical dimensions) and collide with the title. Pin the logo to an
# explicit square physical size instead, and derive the title's left edge from it, so both stay
# correctly laid out no matter how tall the figure ends up being
FIG_WIDTH_IN = 10.0
LOGO_MARGIN_IN = 0.05
LOGO_SIZE_IN = 0.75
TITLE_X_IN = 1.05


class TrendPlotter(PlotterStrategy):
    """Plotter for the Shake&Tune metrics-history trend graph: renders one panel per metric
    family that has at least one data point, or a placeholder when there is no history yet"""

    def plot(self, result: TrendResult) -> Figure:
        data = result.get_plot_data()
        version = data.get('st_version', 'unknown')
        series: Dict[str, Tuple[List[str], List[float]]] = data.get('series') or {}
        n_records = data.get('n_records') or 0

        panels = self._build_panels(series) if series and n_records else []
        if not panels:
            return self._plot_placeholder(version)

        n_panels = len(panels)
        fig_height = PANEL_TOP_MARGIN_IN + PANEL_BOTTOM_MARGIN_IN + PANEL_HEIGHT_IN * n_panels
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=(10, fig_height),
            gridspec_kw={
                'top': 1 - (PANEL_TOP_MARGIN_IN / fig_height),
                'bottom': PANEL_BOTTOM_MARGIN_IN / fig_height,
                'left': 0.09,
                'right': 0.93,
                'hspace': 0.55,
            },
        )
        if n_panels == 1:
            axes = [axes]

        for ax, (title, primary_label, primary, secondary_label, secondary) in zip(axes, panels):
            self._plot_panel(ax, title, primary_label, primary, secondary_label, secondary)

        self._add_titles(fig, n_records, fig_height)
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

    def _build_panels(
        self, series: Dict[str, Tuple[List[str], List[float]]]
    ) -> List[Tuple[str, str, Dict[str, Any], str, Dict[str, Any]]]:
        """Group the flat {label: (ts, values)} series dict into one entry per metric family
        that has data, as (title, primary_label, primary_series, secondary_label, secondary_series)"""
        panels = []

        shaper_fr = {k: v for k, v in series.items() if k.startswith('shaper fr [')}
        shaper_damping = {k: v for k, v in series.items() if k.startswith('shaper damping [')}
        if shaper_fr or shaper_damping:
            panels.append(
                ('Input shaper resonance & damping', 'Frequency (Hz)', shaper_fr, 'Damping ratio', shaper_damping)
            )

        belts_similarity = {k: v for k, v in series.items() if k == 'belt similarity %'}
        belts_tension = {k: v for k, v in series.items() if k == 'belt tension Δf (Hz)'}
        if belts_similarity or belts_tension:
            panels.append(('Belts comparison', 'Similarity (%)', belts_similarity, 'Tension Δf (Hz)', belts_tension))

        vib_symmetry = {k: v for k, v in series.items() if k == 'vibration symmetry %'}
        vib_motor_fr = {k: v for k, v in series.items() if k == 'motor resonance (Hz)'}
        if vib_symmetry or vib_motor_fr:
            panels.append(('Vibrations profile', 'Symmetry (%)', vib_symmetry, 'Motor resonance (Hz)', vib_motor_fr))

        healthcheck_fr = {k: v for k, v in series.items() if k.startswith('healthcheck fr [')}
        if healthcheck_fr:
            panels.append(('Healthcheck', 'Frequency (Hz)', healthcheck_fr, '', {}))

        return panels

    def _plot_panel(
        self,
        ax,
        title: str,
        primary_label: str,
        primary: Dict[str, Tuple[List[str], List[float]]],
        secondary_label: str,
        secondary: Dict[str, Tuple[List[str], List[float]]],
    ) -> None:
        """Plot every series of a panel as a marker+line, primary series on the left axis and
        (when present) secondary series on a twin right axis"""
        color_idx = 0
        any_dates = False
        all_x: List[Any] = []
        for label, (ts_list, values) in sorted(primary.items()):
            x_values, is_dates = self._x_values(ts_list)
            any_dates = any_dates or is_dates
            all_x.extend(x_values)
            ax.plot(
                x_values,
                values,
                marker='o',
                markersize=4,
                linewidth=1.6,
                label=label,
                color=_PANEL_COLORS[color_idx % len(_PANEL_COLORS)],
            )
            color_idx += 1

        ax_secondary = None
        for label, (ts_list, values) in sorted(secondary.items()):
            if ax_secondary is None:
                ax_secondary = ax.twinx()
            x_values, is_dates = self._x_values(ts_list)
            any_dates = any_dates or is_dates
            all_x.extend(x_values)
            ax_secondary.plot(
                x_values,
                values,
                marker='s',
                markersize=4,
                linewidth=1.4,
                linestyle='--',
                label=label,
                color=_PANEL_COLORS[color_idx % len(_PANEL_COLORS)],
            )
            color_idx += 1

        self._apply_x_padding(ax, all_x, any_dates)

        xlabel = 'Date' if any_dates else 'Run #'
        fontP = AxesConfiguration.configure_axes(
            ax, xlabel=xlabel, ylabel=primary_label, title=title, legend=bool(primary)
        )
        if ax_secondary is not None:
            ax_secondary.set_ylabel(secondary_label)
            ax_secondary.legend(loc='upper right', prop=fontP)

    @staticmethod
    def _apply_x_padding(ax, all_x: List[Any], any_dates: bool) -> None:
        """matplotlib's default autoscale margin degenerates into a multi-year-wide date range
        when a panel has only a single (or a couple of same-valued) x point(s); pick an explicit,
        sane padding instead so single-run history still renders a tight, readable x-axis. Falls
        back to matplotlib's own autoscale (never raises) if the data isn't cleanly comparable,
        e.g. a corrupted history mixing parsed timestamps and plain run indices in one panel"""
        if not all_x:
            return
        try:
            xmin, xmax = min(all_x), max(all_x)
            if any_dates:
                span = xmax - xmin
                pad = span * 0.1 if span > timedelta(0) else timedelta(days=1)
            else:
                span = xmax - xmin
                pad = span * 0.1 if span > 0 else 1
            ax.set_xlim(xmin - pad, xmax + pad)
        except Exception:
            return

    @staticmethod
    def _x_values(ts_list: List[str]) -> Tuple[List[Any], bool]:
        """Best-effort parse of a series' timestamp strings into datetimes for a chronological
        x-axis; falls back to a plain run index (0..N-1) if any timestamp fails to parse"""
        parsed = []
        for ts in ts_list:
            try:
                parsed.append(datetime.fromisoformat(ts))
            except (ValueError, TypeError):
                return list(range(len(ts_list))), False
        return parsed, True

    def _add_titles(self, fig: Figure, n_records: int, fig_height: float) -> None:
        """Add the figure title, positioned at a fixed distance from the top/left edges
        regardless of how many panels (and therefore how tall the figure) ended up being
        rendered, and clear of the logo (see _logo_position)"""
        title_x = TITLE_X_IN / FIG_WIDTH_IN
        title_y = 1 - (0.35 / fig_height)
        subtitle_y = 1 - (0.62 / fig_height)
        self.add_title(
            fig,
            [
                {
                    'x': title_x,
                    'y': title_y,
                    'va': 'top',
                    'text': 'SHAKE&TUNE METRICS TREND',
                    'fontsize': 20,
                    'color': PlottingConstants.KLIPPAIN_COLORS['purple'],
                    'weight': 'bold',
                },
                {
                    'x': title_x,
                    'y': subtitle_y,
                    'va': 'top',
                    'fontsize': 11,
                    'text': f'Based on {n_records} recorded run{"s" if n_records != 1 else ""} from the Shake&Tune history',
                },
            ],
        )

    def _plot_placeholder(self, version: str) -> Figure:
        """Render a clean 'no data yet' placeholder figure instead of an empty/broken plot"""
        fig_height = 5.0
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, fig_height))
        ax.axis('off')
        ax.text(
            0.5,
            0.5,
            'No Shake&Tune history yet — run some calibrations first',
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
                    'text': 'SHAKE&TUNE METRICS TREND',
                    'fontsize': 20,
                    'color': PlottingConstants.KLIPPAIN_COLORS['purple'],
                    'weight': 'bold',
                },
            ],
        )
        self.add_logo(fig, position=self._logo_position(fig_height))
        self.add_version_text(fig, version)

        return fig
