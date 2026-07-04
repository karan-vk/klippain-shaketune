# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: healthcheck.py
# Description: Provides the SHAKETUNE_HEALTHCHECK command: a quick per-axis resonance sweep
#              (faster and lower resolution than AXES_SHAPER_CALIBRATION) compared against a
#              stored baseline to catch mechanical drift over time (belt stretch, loosening
#              screws, degrading bearings, ...). MODE=BASELINE captures the reference; MODE=CHECK
#              (the default) compares the current sweep against it and prints a PASS/WARN verdict.

from datetime import datetime

from ..helpers import healthcheck_report, metrics_store
from ..helpers.accelerometer import Accelerometer, MeasurementsManager
from ..helpers.common_func import AXIS_CONFIG
from ..helpers.compat import KlipperCompatibility
from ..helpers.console_output import ConsoleOutput
from ..helpers.resonance_test import vibrate_axis
from ..shaketune_process import ShakeTuneProcess


def healthcheck(gcmd, klipper_config, st_process: ShakeTuneProcess) -> None:
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    mode = gcmd.get('MODE', default='check').lower()
    if mode not in ('check', 'baseline'):
        raise gcmd.error('MODE selection invalid. Should be either CHECK or BASELINE!')

    # Fail fast, before any toolhead motion, if a CHECK is requested with no baseline recorded yet
    if mode == 'check' and not healthcheck_report.baseline_path(st_process.get_st_config()).exists():
        raise gcmd.error('No healthcheck baseline found! Run SHAKETUNE_HEALTHCHECK MODE=BASELINE first.')

    printer = klipper_config.get_printer()
    toolhead = printer.lookup_object('toolhead')
    res_tester = printer.lookup_object('resonance_tester')
    systime = printer.get_reactor().monotonic()

    # Get the default acceleration per Hz value (the frequency range itself is intentionally
    # narrower and quicker than the full AXES_SHAPER_CALIBRATION sweep, see FREQ_START/FREQ_END below)
    compat = KlipperCompatibility(klipper_config)
    res_config = compat.get_res_tester_config()
    _, _, default_accel_per_hz, test_points = res_config

    min_freq = gcmd.get_float('FREQ_START', default=30.0, minval=1)
    max_freq = gcmd.get_float('FREQ_END', default=115.0, minval=1)
    hz_per_sec = gcmd.get_float('HZ_PER_SEC', default=4.0, minval=1)
    accel_per_hz = gcmd.get_float('ACCEL_PER_HZ', default=None)
    feedrate_travel = gcmd.get_float('TRAVEL_SPEED', default=120.0, minval=20.0)
    z_height = gcmd.get_float('Z_HEIGHT', default=None, minval=1)
    accel_chip = gcmd.get('ACCEL_CHIP', default=None)

    if accel_per_hz == '':
        accel_per_hz = None
    if accel_chip == '':
        accel_chip = None

    if accel_per_hz is None:
        accel_per_hz = default_accel_per_hz

    gcode = printer.lookup_object('gcode')

    max_accel = max_freq * accel_per_hz

    # Move to the starting point
    if len(test_points) > 1:
        raise gcmd.error('Only one test point in the [resonance_tester] section is supported by Shake&Tune.')
    if test_points[0] == (-1, -1, -1):
        if z_height is None:
            raise gcmd.error(
                'Z_HEIGHT parameter is required if the test_point in [resonance_tester] section is set to -1,-1,-1'
            )
        # Use center of bed in case the test point in [resonance_tester] is set to -1,-1,-1
        # This is usefull to get something automatic and is also used in the Klippain modular config
        kin_info = toolhead.kin.get_status(systime)
        mid_x = (kin_info['axis_minimum'].x + kin_info['axis_maximum'].x) / 2
        mid_y = (kin_info['axis_minimum'].y + kin_info['axis_maximum'].y) / 2
        point = (mid_x, mid_y, z_height)
    else:
        x, y, z = test_points[0]
        if z_height is not None:
            z = z_height
        point = (x, y, z)

    # Read and save the current acceleration limits so they can be restored later
    toolhead_info = toolhead.get_status(systime)
    old_accel = toolhead_info['max_accel']
    old_mcr = toolhead_info.get('minimum_cruise_ratio')  # None on Klipper < v0.12.0-239

    input_shaper = printer.lookup_object('input_shaper', None)
    creator = st_process.get_graph_creator()
    filename = creator.get_folder() / f'{creator.get_type()}_{date}'
    measurements_manager = MeasurementsManager(st_process.get_st_config().chunk_size, printer.get_reactor(), filename)
    filtered_config = [a for a in AXIS_CONFIG if a['axis'] in ('x', 'y')]

    # Everything that mutates machine state (velocity limits, input shaping) is wrapped so it is
    # always restored in the finally block, even if a measurement raises mid-run
    try:
        # Set the acceleration values needed for the test
        if old_mcr is not None:
            gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={max_accel} MINIMUM_CRUISE_RATIO=0')
        else:
            gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={max_accel}')

        # Deactivate input shaper if it is active to get raw movements
        if input_shaper is not None:
            input_shaper.disable_shaping()

        # Run the quick sweep for each axis, into one shared measurements file (like COMPARE_BELTS_RESPONSES)
        for config in filtered_config:
            toolhead.manual_move(point, feedrate_travel)
            toolhead.dwell(0.5)
            toolhead.wait_moves()

            # First we need to find the accelerometer chip suited for the axis (if not provided by the user)
            current_accel_chip = accel_chip  # Use manually specified chip if provided
            if current_accel_chip is None:
                current_accel_chip = Accelerometer.find_axis_accelerometer(printer, config['axis'])
            if current_accel_chip is None:
                raise gcmd.error('No suitable accelerometer found for measurement!')
            k_accelerometer = printer.lookup_object(current_accel_chip, None)
            if k_accelerometer is None:
                raise gcmd.error(f'Accelerometer chip "{current_accel_chip}" not found!')
            accelerometer = Accelerometer(k_accelerometer, printer.get_reactor())

            ConsoleOutput.print(f'Measuring {config["label"]}...')
            accelerometer.start_recording(measurements_manager, name=config['label'], append_time=True)
            vibrate_axis(
                toolhead,
                gcode,
                config['direction'],
                min_freq,
                max_freq,
                hz_per_sec,
                accel_per_hz,
                res_tester,
                klipper_config,
            )
            accelerometer.stop_recording()
            toolhead.dwell(0.5)
            toolhead.wait_moves()

        # Run post-processing: a single combined graph (and history entry) for both axes
        ConsoleOutput.print('Healthcheck frequency profile generation...')
        creator.configure(mode=mode)
        creator.define_output_target(filename)
        measurements_manager.save_stdata()
        st_process.run(filename)
        st_process.wait_for_completion()
    finally:
        # Always restore the machine state, even if a measurement above raised
        if input_shaper is not None:
            input_shaper.enable_shaping()
        if old_mcr is not None:
            gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_accel} MINIMUM_CRUISE_RATIO={old_mcr}')
        else:
            gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_accel}')

    # Success path only (skipped if the sweep raised): read the result and print the verdict
    current = metrics_store.read_current_summary(filename)
    if mode == 'baseline':
        ConsoleOutput.print('Healthcheck baseline captured.')
    else:
        baseline = healthcheck_report.load_baseline(st_process.get_st_config())
        status, lines = healthcheck_report.compare_to_baseline(current or {}, baseline or {})
        for line in lines:
            ConsoleOutput.print(line)
        ConsoleOutput.print(f'Healthcheck result: {status}')
