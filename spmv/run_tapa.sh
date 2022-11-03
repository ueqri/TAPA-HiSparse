#! /bin/bash

WORK_DIR=run
mkdir -p "${WORK_DIR}"

tapac \
  --work-dir "${WORK_DIR}" \
  --top spmv \
  --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
  --clock-period 3.33 \
  --read-only-args "matrix_hbm.*" \
  --write-only-args "packed_dense_.*" \
  --run-floorplan-dse \
  --enable-synth-util \
  -o "${WORK_DIR}/spmv.xo" \
  --floorplan-strategy "QUICK_FLOORPLANNING" \
  --floorplan-output "${WORK_DIR}/spmv_floorplan.tcl" \
  --connectivity ./src/spmv.ini \
  ./src/spmv.cpp
