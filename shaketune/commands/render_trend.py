# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: render_trend.py
# Description: Provides the SHAKETUNE_TREND command that renders how the Shake&Tune metrics
#              history evolves over time. Unlike every other command in this package, this one
#              performs no toolhead move and no accelerometer recording at all: it only reads
#              back the run history that was already persisted by previous calibrations.

from datetime import datetime

from ..helpers.console_output import ConsoleOutput
from ..shaketune_process import ShakeTuneProcess


def render_trend(gcmd, klipper_config, st_process: ShakeTuneProcess) -> None:
    last_n = gcmd.get_int('LAST_N', default=None, minval=1)

    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    creator = st_process.get_graph_creator()
    filename = creator.get_folder() / f'trend_{date}'

    creator.configure(last_n=last_n)
    creator.define_output_target(filename)

    ConsoleOutput.print('Generating Shake&Tune trend graph from history...')
    st_process.run(None)
    st_process.wait_for_completion()
