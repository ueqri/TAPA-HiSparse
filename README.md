# TAPA-HiSparse

[HiSparse](https://github.com/cornell-zhang/HiSparse) is a HLS library which targets High-performance Sparse Linear Algebra, such as SpMV. Compared to the version in FPGA'22, current HiSparse is enhanced in many levels such as portability and compatibility (for latest vendor tool), and also equipped with a **[multi-HBM SpMSpV](https://github.com/ueqri/TAPA-HiSparse/tree/main/spmspv) as a new case study**.

[TAPA](https://github.com/UCLA-VAST/tapa) is a dataflow HLS framework from [UCLA VAST](https://vast.cs.ucla.edu/) group, which features fast compilation, expressive programming model and generates high-frequency FPGA accelerators.

This project aims to migrate the latest HiSparse library from vanilla Vitis HLS to TAPA framework, to exert [AutoBridge](https://github.com/UCLA-VAST/AutoBridge) workflow for better floorplan quality and pipelining; in this way, we could get higher frequency and thus higher throughput for sparse computing. Further works still focus on improving HiSparse frequency & scalability (scale to more HBM channels), and the integration to [GraphLily](https://github.com/cornell-zhang/GraphLily) (already have some milestones [here](https://github.com/cornell-zhang/GraphLily/tree/hang_integration)).

## Prerequisites

### Basic
- TAPA framework: 0.0.20220807.1 or later
- Xilinx Vitis Tool: 2022.1.1
- Package cnpy: latest

### Hardware-specific
- FPGA Card: Xilinx Alveo U280
- Hardware Platform: [xilinx_u280_gen3x16_xdma_base_1](https://docs.xilinx.com/r/en-US/ug1120-alveo-platforms/U280-Gen3x16-XDMA-base_1-Platform)
- XRT: 2022.1

## Workflow

### Hardware

1. Setup the TAPA and Vitis 2022.1 environments before running `run_tapa.sh`.

2. Run `run_tapa.sh` to start the TAPA and then AutoBridge process. The DSE results will locate on `spmv/run/run-*` directory.

3. Enter `spmv/run/run-*` and run `spmv_generate_bitstream.sh` to synthesize and implement HW. (Also, in case you use the TAPA w/o Vitis 2022.1 supports, please run `sed -i 's/pfm_top_i\/dynamic_region/level0_i\/ulp/g' spmv_floorplan.tcl` before generating bitstream.)

4. Build host and benchmark in `spmv/sw` via `make host benchmark`.

### Software Emulation

Simply build the host in `spmv/sw` directory via `make host`, and execute host directly.

### Note
The environment variable `DATASETS` should be set to the path of datasets (e.g., googleplus, ogbl-ppa, etc.) before running `host` or `bench.sh`. The datasets including graph and pruned_nn are available [here](https://drive.google.com/file/d/1VCus77NffWdEfppD5xE6sIIZtx7yNZ6m).