# Layout Selection Artifact

This folder is a standalone artifact for the treewidth-based exact layout-selection solver.
It contains:

- the C++ implementation of the interaction-graph construction, nice tree-decomposition conversion, and dynamic program
- wrapper scripts to build and run the full pipeline
- two sample input instances

## Requirements

- Linux
- `g++` with C++17 support
- `make`
- `git`
- Java (`javac` and `java`)
- GNU `timeout`

The solver records RSS from `/proc/self/status`, so the memory metrics are Linux-specific.

## Build

From this directory:

```bash
make
```

This builds `./ls_opt` from `src/ls_opt.cpp`.

## External Treewidth Solver

This artifact does not vendor the external PACE 2017 Track A solver. Instead, fetch it from the public GitHub repository:

Source:
- `https://github.com/TCS-Meiji/PACE2017-TrackA`

One simple setup is:

```bash
mkdir -p external
git clone https://github.com/TCS-Meiji/PACE2017-TrackA external/PACE2017-TrackA
make -C external/PACE2017-TrackA exact
```

The default runner expects the solver wrapper at:

```text
external/PACE2017-TrackA/tw-exact
```

If you place it elsewhere, set:

```bash
export TW_EXACT=/path/to/PACE2017-TrackA/tw-exact
```

## Run One Instance

```bash
./run_layout_selection.sh samples/ls-0660a9-smt-sg0001.txt
```

By default this writes intermediate and final files to:

```text
out/pipeline/ls-0660a9-smt-sg0001/
```

The final solved dump contains a filled `# layout_selection` section.

## Run Both Samples

```bash
./scripts/run_samples.sh
```

## Useful Environment Variables

- `LS_TIMEOUT_MS`: per-stage timeout in milliseconds, default `600000`
- `LS_MEM_CAP_MB`: per-stage memory cap in MB, default `8192`
- `TW_EXACT`: path to the external `tw-exact` wrapper
- `TW_SOLVER_DIR`: path to the external `PACE2017-TrackA` checkout
- `TW_XMX`, `TW_XMS`, `TW_XSS`: JVM settings forwarded to `tw-exact`

## Folder Layout

- `src/ls_opt.cpp`: main implementation
- `run_layout_selection.sh`: end-to-end pipeline
- `samples/`: small example inputs
- `scripts/`: helper scripts for running samples
