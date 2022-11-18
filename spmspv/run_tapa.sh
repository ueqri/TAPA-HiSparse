#! /bin/bash

WORK_DIR=run
mkdir -p "${WORK_DIR}"

tapac \
  --work-dir "${WORK_DIR}" \
  --top spmspv \
  --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
  --clock-period 3.33 \
  --read-only-args "mat_.*" \
  --read-only-args "vector" \
  --write-only-args "result" \
  --run-floorplan-dse \
  --enable-synth-util \
  --enable-hbm-binding-adjustment \
  -o "${WORK_DIR}/spmspv.xo" \
  --floorplan-output "${WORK_DIR}/spmspv_floorplan.tcl" \
  --connectivity ./src/spmspv.ini \
  ./src/spmspv.cpp
