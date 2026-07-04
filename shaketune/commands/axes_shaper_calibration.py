# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: axes_shaper_calibration.py
# Description: Provides a command for calibrating the input shaper of a 3D printer's axes using an accelerometer.
#              The script performs resonance tests along specified axes, starts and stops measurements,
#              and generates graphs for each axis to analyze the collected data.

import json
from datetime import datetime

from ..helpers.accelerometer import Accelerometer, MeasurementsManager
from ..helpers.common_func import AXIS_CONFIG
from ..helpers import metrics_store
from ..helpers.compat import KlipperCompatibility
from ..helpers.console_output import ConsoleOutput
from ..helpers.resonance_test import vibrate_axis
from ..shaketune_process import ShakeTuneProcess


def axes_shaper_calibration(gcmd, klipper_config, st_process: ShakeTuneProcess) -> None:
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    printer = klipper_config.get_printer()
    toolhead = printer.lookup_object('toolhead')
    res_tester = printer.lookup_object('resonance_tester')
    systime = printer.get_reactor().monotonic()
    toolhead_info = toolhead.get_status(systime)

    # Get the default values for the frequency range and the acceleration per Hz
    compat = KlipperCompatibility(klipper_config)
    res_config = compat.get_res_tester_config()
    default_min_freq, default_max_freq, default_accel_per_hz, test_points = res_config

    min_freq = gcmd.get_float('FREQ_START', default=default_min_freq, minval=1)
    max_freq = gcmd.get_float('FREQ_END', default=default_max_freq, minval=1)
    hz_per_sec = gcmd.get_float('HZ_PER_SEC', default=1, minval=1)
    accel_per_hz = gcmd.get_float('ACCEL_PER_HZ', default=None)
    axis_input = gcmd.get('AXIS', default='all').lower()
    if axis_input not in {'x', 'y', 'all'}:
        raise gcmd.error('AXIS selection invalid. Should be either x, y, or all!')
    scv = gcmd.get_float('SCV', default=toolhead_info['square_corner_velocity'], minval=0)
    max_sm = gcmd.get_float('MAX_SMOOTHING', default=None, minval=0)
    feedrate_travel = gcmd.get_float('TRAVEL_SPEED', default=120.0, minval=20.0)
    z_height = gcmd.get_float('Z_HEIGHT', default=None, minval=1)
    max_scale = gcmd.get_int('MAX_SCALE', default=None, minval=1)
    accel_chip = gcmd.get('ACCEL_CHIP', default=None)
    apply = bool(gcmd.get_int('APPLY', default=0, minval=0, maxval=1))
    apply_target = gcmd.get('APPLY_TARGET', default='low_vibration').lower()

    if accel_per_hz == '':
        accel_per_hz = None
    if accel_chip == '':
        accel_chip = None
    if apply_target == '':
        apply_target = 'low_vibration'

    if accel_per_hz is None:
        accel_per_hz = default_accel_per_hz

    if apply_target not in ('low_vibration', 'performance'):
        raise gcmd.error('APPLY_TARGET selection invalid. Should be either low_vibration or performance!')

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

    # set the needed acceleration values for the test
    toolhead_info = toolhead.get_status(systime)
    old_accel = toolhead_info['max_accel']
    if 'minimum_cruise_ratio' in toolhead_info:  # minimum_cruise_ratio found: Klipper >= v0.12.0-239
        old_mcr = toolhead_info['minimum_cruise_ratio']
        gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={max_accel} MINIMUM_CRUISE_RATIO=0')
    else:  # minimum_cruise_ratio not found: Klipper < v0.12.0-239
        old_mcr = None
        gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={max_accel}')

    # Deactivate input shaper if it is active to get raw movements
    input_shaper = printer.lookup_object('input_shaper', None)
    if input_shaper is not None:
        input_shaper.disable_shaping()
    else:
        input_shaper = None

    creator = st_process.get_graph_creator()

    # Filter axis configurations based on user input, assuming 'axis_input' can be 'x', 'y', 'all' (that means 'x' and 'y')
    filtered_config = [
        a for a in AXIS_CONFIG if a['axis'] == axis_input or (axis_input == 'all' and a['axis'] in ('x', 'y'))
    ]
    per_axis_results = {}
    for config in filtered_config:
        filename = creator.get_folder() / f'{creator.get_type().replace(" ", "")}_{date}_{config["label"]}'
        measurements_manager = MeasurementsManager(
            st_process.get_st_config().chunk_size, printer.get_reactor(), filename
        )

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

        # Then do the actual measurements
        ConsoleOutput.print(f'Measuring {config["label"]}...')
        accelerometer.start_recording(measurements_manager, name=config['label'], append_time=True)
        test_params = vibrate_axis(
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

        # And finally generate the graph for each measured axis
        ConsoleOutput.print(f'{config["axis"].upper()} axis frequency profile generation...')
        ConsoleOutput.print('This may take some time (1-3min)')
        creator.configure(scv, max_sm, test_params, max_scale)
        creator.define_output_target(filename)
        measurements_manager.save_stdata()
        st_process.run(filename)
        st_process.wait_for_completion()
        metrics_store.print_run_summary(st_process.get_st_config(), filename, creator.get_type())

        summary_path = filename.with_suffix('.json')
        try:
            record = json.loads(summary_path.read_text())
            per_axis_results[config['axis']] = record.get('summary') if isinstance(record, dict) else None
        except (OSError, json.JSONDecodeError) as e:
            ConsoleOutput.print(f'Warning: unable to read Shake&Tune results for {config["label"]}: {e}')
            per_axis_results[config['axis']] = None

        toolhead.dwell(1)

    _print_shaper_recommendation_block(per_axis_results, apply_target)
    if apply:
        _apply_shaper_recommendation(printer, gcode, per_axis_results, apply_target)

    # Re-enable the input shaper if it was active
    if input_shaper is not None:
        input_shaper.enable_shaping()

    # Restore the previous acceleration values
    if old_mcr is not None:  # minimum_cruise_ratio found: Klipper >= v0.12.0-239
        gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_accel} MINIMUM_CRUISE_RATIO={old_mcr}')
    else:  # minimum_cruise_ratio not found: Klipper < v0.12.0-239
        gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_accel}')


def _selected_shaper(summary, apply_target):
    """Pick the recommended shaper dict {type, freq, max_accel} for apply_target, falling back
    to the low_vibration recommendation when the requested target isn't available"""
    if not summary:
        return None
    recommendations = summary.get('recommendations') or {}
    return recommendations.get(apply_target) or recommendations.get('low_vibration')


def _print_shaper_recommendation_block(per_axis_results, apply_target) -> None:
    """Print a ready-to-paste [input_shaper] config block for every successfully measured axis"""
    ConsoleOutput.print('Recommended input shaper configuration (copy into your [input_shaper] section):')
    ConsoleOutput.print('[input_shaper]')

    compat_axes = []
    for axis in ('x', 'y'):
        summary = per_axis_results.get(axis)
        rec = _selected_shaper(summary, apply_target)
        if not rec:
            continue

        ConsoleOutput.print(f'shaper_type_{axis} = {rec["type"]}')
        ConsoleOutput.print(f'shaper_freq_{axis} = {rec["freq"]:.1f}')
        if summary.get('damping_ratio_measured') and summary.get('zeta') is not None:
            ConsoleOutput.print(f'damping_ratio_{axis} = {summary["zeta"]:.3f}')

        if summary.get('compat'):
            compat_axes.append(axis.upper())

    if compat_axes:
        ConsoleOutput.print(
            f'Note: {", ".join(compat_axes)} axis measurement(s) ran in Klipper compatibility mode; '
            'results may be slightly less accurate.'
        )
    ConsoleOutput.print(
        'This configuration was NOT saved automatically (no SAVE_CONFIG was issued): copy it into '
        'printer.cfg and restart Klipper to make it permanent, or use APPLY=1 to set it for this session only.'
    )


def _apply_shaper_recommendation(printer, gcode, per_axis_results, apply_target) -> None:
    """Apply the recommended shaper for this session only, via SET_INPUT_SHAPER. Only measured
    axes are included so unmeasured axes keep their current Klipper values"""
    if printer.lookup_object('dual_carriage', None) is not None:
        ConsoleOutput.print(
            'Warning: APPLY is not supported on printers with dual_carriage; please review the recommended '
            'values above and apply them manually.'
        )
        return

    apply_parts = []
    for axis in ('x', 'y'):
        summary = per_axis_results.get(axis)
        rec = _selected_shaper(summary, apply_target)
        if not rec:
            continue

        axis_upper = axis.upper()
        apply_parts.append(f'SHAPER_TYPE_{axis_upper}={rec["type"]}')
        apply_parts.append(f'SHAPER_FREQ_{axis_upper}={rec["freq"]:.3f}')
        if summary.get('damping_ratio_measured') and summary.get('zeta') is not None:
            apply_parts.append(f'DAMPING_RATIO_{axis_upper}={summary["zeta"]:.4f}')

    if apply_parts:
        gcode.run_script_from_command('SET_INPUT_SHAPER ' + ' '.join(apply_parts))
        ConsoleOutput.print(
            f'Applied the {apply_target.replace("_", " ")} input shaper settings for this session only '
            '(not persisted -- use the block above and SAVE_CONFIG to make it permanent).'
        )
