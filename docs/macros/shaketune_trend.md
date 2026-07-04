# Metrics trend over time

The `SHAKETUNE_TREND` macro plots how your printer's calibration metrics evolve **over time**, from the history that Shake&Tune records automatically. It takes **no new measurement** — it just reads the recorded history and draws it — so it's instant and safe to run any time.

Every time you run a Shake&Tune calibration, a small JSON-safe summary of that run (the scalar results, not the raw data) is appended to a `history.jsonl` file in your results folder, and a compact delta versus the previous run of the same type is printed to the console (e.g. `X resonant frequency: 52.4 Hz (was 53.1 Hz, Δ-0.7)`). `SHAKETUNE_TREND` turns that accumulated history into graphs, which makes slow drift — belt stretch, a loosening frame, degrading bearings — visible as a trend line rather than something you'd have to spot by eyeballing old PNGs.


## Usage

```
SHAKETUNE_TREND
```

| parameters | default value | description |
|-----------:|---------------|-------------|
|LAST_N|None (all)|only plot the most recent N recorded runs. Useful to zoom in on recent history once you have a lot of it|

If there's no history yet, the macro produces a friendly placeholder graph rather than failing — just run a few calibrations first and it will start filling in.


## What it produces

One panel per metric family present in your history, each showing the metric versus run:
  - **Input shaper**: resonant frequency and damping ratio, per axis
  - **Belts**: estimated similarity, and the belt-tension frequency difference (Δf)
  - **Vibrations profile**: machine vibration symmetry and the motors' resonant frequency
  - **Healthcheck**: per-axis resonant frequency (see [`SHAKETUNE_HEALTHCHECK`](./shaketune_healthcheck.md))

What you're looking for is **stability**: healthy metrics stay roughly flat over time. A steady drift in one of them is an early sign that something mechanical is changing and is worth investigating before it affects your prints.

  > **Note**
  >
  > The history lives at `history.jsonl` in your Shake&Tune results folder and is never pruned automatically (unlike the graph images), so your trend keeps its full history. When using the [CLI](../cli_usage.md), a run writes its history next to the output file instead, so it never touches a real printer's history.
