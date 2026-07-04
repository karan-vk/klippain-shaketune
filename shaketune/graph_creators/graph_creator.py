# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: graph_creator.py
# Description: Base class for creating graphs using composition-based architecture


import abc
from pathlib import Path
from typing import Optional, Type

from matplotlib.figure import Figure

from ..helpers.accelerometer import MeasurementsManager
from ..shaketune_config import ShakeTuneConfig
from .base_models import Computation, PlotterStrategy


class GraphCreator(abc.ABC):
    """Base class for graph creators using composition-based architecture"""

    registry = {}
    writes_history: bool = True

    @classmethod
    def register(cls, graph_type: str):
        """Decorator to register graph creator subclasses"""

        def decorator(subclass):
            cls.registry[graph_type] = subclass
            subclass.graph_type = graph_type
            return subclass

        return decorator

    def __init__(
        self, config: ShakeTuneConfig, computation_class: Type[Computation], plotter_class: Type[PlotterStrategy]
    ):
        self._config = config
        self._version = ShakeTuneConfig.get_git_version()
        self._type = self.__class__.graph_type
        self._folder = self._config.get_results_folder(self._type)
        self._output_target: Optional[Path] = None

        self._computation_class = computation_class
        self._plotter = plotter_class()

    @abc.abstractmethod
    def configure(self, **kwargs) -> None:
        """Configure the graph creator with specific parameters"""
        pass

    @abc.abstractmethod
    def _create_computation(self, measurements_manager: MeasurementsManager) -> Computation:
        """Create the computation instance with proper configuration"""
        pass

    def create_graph(self, measurements_manager: MeasurementsManager) -> None:
        """Create and save the graph. This pipeline is fixed and not meant to be overridden;
        subclasses that need extra post-processing (e.g. persisting a baseline) hook in via
        _after_save() rather than reimplementing the pipeline."""
        computation = self._create_computation(measurements_manager)
        result = computation.compute()
        fig = self._plotter.plot(result)
        self._save_figure(fig)
        if self.writes_history:
            self._save_summary(result)
        self._after_save(result)

    def _after_save(self, result) -> None:  # noqa: B027 (intentional concrete no-op hook, not abstract)
        """Extension hook run at the very end of create_graph(), after the figure and summary have
        been saved. No-op by default; subclasses may override to add post-processing without
        touching the fixed create_graph() pipeline."""
        pass

    def _save_summary(self, result) -> None:
        """Persist the computation result's JSON-safe summary next to the output file and
        append it to the run history. Never raises: a metrics failure (including a buggy
        get_summary() implementation) must never fail the graph it's attached to"""
        from ..helpers.console_output import ConsoleOutput
        from ..helpers.metrics_store import write_run_artifacts

        try:
            summary = result.get_summary()
        except Exception as e:
            ConsoleOutput.print(f'Warning: unable to compute Shake&Tune metrics summary: {e}')
            return

        write_run_artifacts(self._output_target, self._config, self._type, summary)

    def _save_figure(self, fig: Figure) -> None:
        """Save the figure to disk"""
        if self._output_target is None:
            raise ValueError(
                'Output target not defined. Please call define_output_target() before trying to save the figure!'
            )

        fig.savefig(f'{self._output_target.with_suffix(".png")}', dpi=self._config.dpi)
        if not self._config.keep_raw_data:
            self._output_target.with_suffix('.stdata').unlink(missing_ok=True)

    def get_type(self) -> str:
        """Get the graph type"""
        return self._type

    def get_folder(self) -> Path:
        """Get the output folder"""
        return self._folder

    def define_output_target(self, filepath: Path) -> None:
        """Define the output file path"""
        # Remove the extension if it exists (to be safer when using the CLI mode)
        if filepath.suffix:
            filepath = filepath.with_suffix('')
        self._output_target = filepath

    def clean_old_files(self, keep_results: int = 10) -> None:
        """Clean old result files"""
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)
        if len(files) <= keep_results:
            return  # No need to delete any files
        for old_png_file in files[keep_results:]:
            stdata_file = old_png_file.with_suffix('.stdata')
            stdata_file.unlink(missing_ok=True)
            old_png_file.with_suffix('.json').unlink(missing_ok=True)
            old_png_file.unlink()
