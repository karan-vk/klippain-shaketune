# Machine healthcheck

The `SHAKETUNE_HEALTHCHECK` macro is a quick, repeatable "is my machine still where it was?" check. It runs a short resonance sweep on the X then Y axis, extracts each axis' main resonant frequency and damping ratio, and compares them against a **baseline** you captured earlier. Mechanical drift — a belt slowly stretching, a screw working loose, a bearing starting to wear — shows up as a change in these numbers well before it ruins a print, so this is a good thing to run every so often (for example as a pre-print-day ritual).

It is intentionally **faster and lower resolution** than [`AXES_SHAPER_CALIBRATION`](./axes_shaper_calibrations.md): it is a health *check*, not a precision input-shaper calibration. Use `AXES_SHAPER_CALIBRATION` when you actually want to (re)tune your filters.


## Usage

First capture a baseline on a machine you're happy with:

```
SHAKETUNE_HEALTHCHECK MODE=BASELINE
```

Then, later, run a check against it:

```
SHAKETUNE_HEALTHCHECK
```

The check prints a per-axis `PASS`/`WARN` verdict to the console. A `WARN` means an axis has drifted enough to be worth a look (it tells you what changed and by how much); it is a hint to inspect the machine, not an automatic failure. Available parameters:

| parameters | default value | description |
|-----------:|---------------|-------------|
|MODE|check|`BASELINE` captures the current sweep as the new reference; `CHECK` (the default) compares the current sweep against the stored baseline. A `CHECK` with no baseline yet will stop immediately and ask you to capture one first|
|FREQ_START|30|starting excitation frequency (narrower/faster than a full shaper calibration on purpose)|
|FREQ_END|115|maximum excitation frequency|
|HZ_PER_SEC|4|number of Hz per second for the sweep (faster than the `AXES_SHAPER_CALIBRATION` default, trading a little frequency resolution for speed — enough for a clean single-peak drift check)|
|ACCEL_PER_HZ|None (default to `[resonance_tester]` value)|accel per Hz value used for the test|
|TRAVEL_SPEED|120|speed in mm/s used for the travel movements|
|Z_HEIGHT|None|Z height for the test, to override the `probe_point` Z of your `[resonance_tester]` config section if needed|
|ACCEL_CHIP|None|accelerometer chip to use. If not provided, the best chip is found automatically from your `[resonance_tester]` config section|

  > **Note**
  >
  > The comparison thresholds are deliberately conservative: an axis is flagged only when its resonant frequency moves by more than ~3 Hz, or its damping ratio changes by both a large relative and absolute amount. A frequency shift and its accompanying damping wobble are reported as a single finding to avoid double-warning about the same underlying cause.


## What it produces

For each axis the healthcheck graph overlays the **current** power spectral density against the **baseline** one (when a baseline is available), with the resonant frequency of each marked, so you can see at a glance whether and how the response has moved. The scalar results are also recorded in the Shake&Tune history, so `SHAKETUNE_HEALTHCHECK` results feed into [`SHAKETUNE_TREND`](./shaketune_trend.md) alongside your other calibrations.
