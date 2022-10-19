# TAPA-HiSparse

TODO
## Workflow

### Hardware

1. Setup the TAPA and Vitis 2022.1 environments before running `run_tapa.sh`.

2. Run `run_tapa.sh` to start the TAPA and then AutoBridge process. The DSE results will locate on `spmv/run/run-*` directory.

3. Enter `spmv/run/run-*` and run `spmv_generate_bitstream.sh` to synthesize and implement HW.

4. Build host and benchmark in `spmv/sw` via `make host benchmark`.

### Software Emulation

Simply build the host in `spmv/sw` directory via `make host`, and execute host directly.

Note: The environment variable `DATASETS` should be set to the path of datasets, before running `host` or `bench.sh`.