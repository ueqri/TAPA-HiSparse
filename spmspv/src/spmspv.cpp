#include <tapa.h>
#include <ap_fixed.h>
#include <assert.h>
#include "spmspv.h"

#ifndef __SYNTHESIS__
#include <iostream>
#include <iomanip>
static bool line_tracing_spmspv_load_data = false;
static bool line_tracing_spmspv_write_back = false;
#endif

// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
// matrix loader

    // data loader for SpMSpV from one HBM channel
    void load_matrix_from_gmem(
        // |--partptr--|--indptr--|--matrix data--|
        tapa::mmap<SPMSPV_MAT_PKT_MMAP> matrix,
        IDX_T num_parts,
        IDX_T num_cols,
        unsigned channel_id,
        // fifos
        tapa::istream<IDX_VAL_INST_T> &VL2ML,
        tapa::ostream<INST_T> &DL_to_MG_inst,
        tapa::ostreams<UPDATE_PLD_T, PACK_SIZE> &DL_to_MG_stream
    ) {
        IDX_T num_cols_this_channel = (num_cols - 1 - channel_id) / SPMSPV_NUM_HBM_CHANNEL + 1;
        // partition base
        IDX_T mat_indptr_base = 0;
        IDX_T mat_row_id_base = 0;
        IDX_T mat_indptr_offset = (num_parts + 2 * PACK_SIZE - 1) / (2 * PACK_SIZE);

        load_matrix_from_gmem_loop_over_parts:
        for (unsigned int part_id = 0; part_id < num_parts; part_id++) {
            #pragma HLS pipeline off

            bool exit = false;

            // CSC part pointers start from 0
            IDX_T partptr_pack_idx = part_id / (2 * PACK_SIZE);
            IDX_T partptr_pack_pos = part_id % (2 * PACK_SIZE);
            IDX_T mat_data_addr_base = partptr_pack_pos / PACK_SIZE ?
                MAT_PKT_CAST_VALS(matrix[partptr_pack_idx], partptr_pack_pos % PACK_SIZE) :
                MAT_PKT_CAST_INDICES(matrix[partptr_pack_idx], partptr_pack_pos % PACK_SIZE);

            DL_to_MG_inst.write(SOD); // no need to fill `DL_to_MG_stream` with SOD anymore

            // loop over all active columns
            loop_over_active_columns_ML:
            while (!exit) {

                // slice out the current column out of the active columns
                IDX_VAL_INST_T pld = VL2ML.read();
                if (pld.inst == EOS) {
                    exit = true;
                } else if (pld.inst != SOD && pld.inst != EOD) {

                    IDX_T current_column_id = pld.index;
                    VAL_T vec_val = pld.val;
                    // [0] for start, [1] for end
                    // write like this to make sure it uses burst read
                    IDX_T col_slice[2];
                    #pragma HLS array_partition variable=col_slice complete

                    loop_get_column_len_ML:
                    for (unsigned int i = 0; i < 2; i++) {
                        #pragma HLS unroll
                        IDX_T raw_mat_indptr_idx = current_column_id + mat_indptr_base + i;
                        // CSC index pointers start from `mat_indptr_offset`
                        IDX_T indptr_pack_idx = raw_mat_indptr_idx / (2 * PACK_SIZE);
                        IDX_T indptr_pack_pos = raw_mat_indptr_idx % (2 * PACK_SIZE);
                        SPMV_MAT_PKT_MMAP pkt = matrix[mat_indptr_offset + indptr_pack_idx];
                        col_slice[i] = indptr_pack_pos / PACK_SIZE ?
                            MAT_PKT_CAST_VALS(pkt, indptr_pack_pos % PACK_SIZE) :
                            MAT_PKT_CAST_INDICES(pkt, indptr_pack_pos % PACK_SIZE);
                    }

                    loop_over_pkts_ML:
                    for (unsigned int i = 0; i < (col_slice[1] - col_slice[0]); i++) {
                        #pragma HLS pipeline II=1
                        SPMSPV_MAT_PKT_MMAP mat_pkt = matrix[i + mat_data_addr_base + col_slice[0]];
                        DL_to_MG_inst.write(0);

                        loop_unpack_ML_unroll:
                        for (unsigned int k = 0; k < PACK_SIZE; k++) {
                            #pragma HLS unroll
                            UPDATE_PLD_T input_to_MG;
                            input_to_MG.mat_val(31,0) = MAT_PKT_CAST_VALS(mat_pkt, k);
                            input_to_MG.vec_val = vec_val;
                            input_to_MG.row_idx = MAT_PKT_CAST_INDICES(mat_pkt, k) - mat_row_id_base;
                            input_to_MG.inst = 0;
                            // discard paddings is done in data merger
                            DL_to_MG_stream[k].write(input_to_MG);
                        }
                    }

                }

            }

            // no need to fill `DL_to_MG_stream` with end inst anymore
            DL_to_MG_inst.write(EOD);
            DL_to_MG_inst.write(EOS);

            #ifndef __SYNTHESIS__
            if (line_tracing_spmspv_load_data) {
                static int part_id = 0;
                std::cout << "INFO: [kernel SpMSpV] Load matrix finished, part_id = "
                        << part_id++ << std::endl << std::flush;
            }
            #endif

            mat_indptr_base += num_cols_this_channel + 1;
            mat_row_id_base += SPMSPV_OUT_BUF_LEN;
        }
    }

    // merge streams of matrix loader in different HBM, and forward the available
    // output to shuffle stream
    void merge_load_streams(
        tapa::istreams<INST_T, SPMSPV_NUM_HBM_CHANNEL> &ML_to_MG_insts,
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &ML_to_MG_channel_0,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &ML_to_MG_channel_1,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &ML_to_MG_channel_2,
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &ML_to_MG_channel_3,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 6)
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &ML_to_MG_channel_4,
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &ML_to_MG_channel_5,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &ML_to_MG_channel_6,
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &ML_to_MG_channel_7,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 10)
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &ML_to_MG_channel_8,
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &ML_to_MG_channel_9,
#endif
        tapa::ostreams<UPDATE_PLD_T, PACK_SIZE> &MG_to_SF_streams,
        IDX_T num_parts
    ) {
        merge_load_streams_loop_over_parts:
        for (unsigned int part_id = 0; part_id < num_parts; part_id++) {
            #pragma HLS pipeline off

            for (unsigned int k = 0; k < PACK_SIZE; k++) {
                #pragma HLS unroll
                MG_to_SF_streams[k].write(UPDATE_PLD_SOD);
            }

            bool exit = false;
            char current_input = 0; // read from multiple matrix loader streams
            ap_uint<SPMSPV_NUM_HBM_CHANNEL> finished = 0;

            spmspv_merge_load_streams_loop:
            while (!exit) {
                #pragma HLS pipeline II=1

                INST_T ctrl;
                if (!finished[current_input] && ML_to_MG_insts[current_input].try_read(ctrl)) {
                    if (ctrl == EOS) {
                        finished[current_input] = true;
                    } else if (ctrl != SOD && ctrl != EOD) {

                        forward_mat_pkt_MG_unroll:
                        for (unsigned int k = 0; k < PACK_SIZE; k++) {
                            #pragma HLS unroll
                            UPDATE_PLD_T pld_to_SF;
                            switch (current_input) {
                                #if (SPMSPV_NUM_HBM_CHANNEL >= 1)
                                    case 0: pld_to_SF = ML_to_MG_channel_0[k].read(); break;
                                #endif
                                #if (SPMSPV_NUM_HBM_CHANNEL >= 2)
                                    case 1: pld_to_SF = ML_to_MG_channel_1[k].read(); break;
                                #endif
                                #if (SPMSPV_NUM_HBM_CHANNEL >= 4)
                                    case 2: pld_to_SF = ML_to_MG_channel_2[k].read(); break;
                                    case 3: pld_to_SF = ML_to_MG_channel_3[k].read(); break;
                                #endif
                                #if (SPMSPV_NUM_HBM_CHANNEL >= 6)
                                    case 4: pld_to_SF = ML_to_MG_channel_4[k].read(); break;
                                    case 5: pld_to_SF = ML_to_MG_channel_5[k].read(); break;
                                #endif
                                #if (SPMSPV_NUM_HBM_CHANNEL >= 8)
                                    case 6: pld_to_SF = ML_to_MG_channel_6[k].read(); break;
                                    case 7: pld_to_SF = ML_to_MG_channel_7[k].read(); break;
                                #endif
                                #if (SPMSPV_NUM_HBM_CHANNEL >= 10)
                                    case 8: pld_to_SF = ML_to_MG_channel_8[k].read(); break;
                                    case 9: pld_to_SF = ML_to_MG_channel_9[k].read(); break;
                                #endif
                            }
                            if (pld_to_SF.mat_val != /*Zero*/0) {
                                // only forward non-zero payload, and SOD/EOS/EOD to shuffle
                                // needs to take care manually
                                MG_to_SF_streams[k].write(pld_to_SF);
                            }
                        }

                    }
                }

                exit = finished.and_reduce();

                if ( (++current_input) == SPMSPV_NUM_HBM_CHANNEL) {
                    current_input = 0;
                }
            }

            for (unsigned int k = 0; k < PACK_SIZE; k++) {
                #pragma HLS unroll
                MG_to_SF_streams[k].write(UPDATE_PLD_EOD);
                MG_to_SF_streams[k].write(UPDATE_PLD_EOS);
            }

            #ifndef __SYNTHESIS__
            if (line_tracing_spmspv_load_data) {
                std::cout << "INFO: [kernel SpMSpV] Merge streams finished" << std::endl
                        << std::flush;
            }
            #endif
        }
    }
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
// 1 kinds of arbiter

    // latency of the arbiter, which is pipeline_depth - 1
    const unsigned ARBITER_LATENCY = 5;

    // arbiter for read responses (UPDATE_PLD_T) (depends on row_idx)
    void arbiter_for_read_resp(
        const UPDATE_PLD_T in_pld[PACK_SIZE],
        UPDATE_PLD_T resend_pld[PACK_SIZE],
        const ap_uint<PACK_SIZE> in_valid,
        ap_uint<PACK_SIZE> &in_resend,
        unsigned xbar_sel[PACK_SIZE],
        ap_uint<PACK_SIZE> &out_valid,
        const unsigned rotate_priority
    ) {
        #pragma HLS pipeline II=1 enable_flush
        #pragma HLS latency min=ARBITER_LATENCY max=ARBITER_LATENCY
        #pragma HLS array_partition variable=xbar_sel complete

        // prioritized valid and addr
        ap_uint<PACK_SIZE> arb_p_in_valid = in_valid;
        IDX_T arb_p_in_addr[PACK_SIZE];
        IDX_T in_addr[PACK_SIZE];
        #pragma HLS array_partition variable=in_addr complete
        #pragma HLS array_partition variable=arb_p_in_addr complete

        for (unsigned i = 0; i < PACK_SIZE; i++) {
            #pragma HLS unroll
            arb_p_in_addr[i] = in_pld[(i + rotate_priority) % PACK_SIZE].row_idx;
            in_addr[i] = in_pld[i].row_idx;
        }

        arb_p_in_valid.rrotate(rotate_priority);

        loop_A_arbsearch:
        for (unsigned OLid = 0; OLid < PACK_SIZE; OLid++) {
            #pragma HLS unroll
            bool found = false;
            unsigned chosen_port = 0;

            loop_ab_logic_encoder_unroll:
            for (unsigned ILid_plus_1 = PACK_SIZE; ILid_plus_1 > 0; ILid_plus_1--) {
                #pragma HLS unroll
                if (arb_p_in_valid[ILid_plus_1 - 1] && ((arb_p_in_addr[ILid_plus_1 - 1] % PACK_SIZE) == OLid)) {
                    chosen_port = ILid_plus_1 - 1;
                    found = true;
                }
            }
            if (!found) {
                out_valid[OLid] = 0;
                xbar_sel[OLid] = 0;
            } else {
                out_valid[OLid] = 1;
                xbar_sel[OLid] = (chosen_port + rotate_priority) % PACK_SIZE;
            }
        }

        loop_A_grant:
        for (unsigned ILid = 0; ILid < PACK_SIZE; ILid++) {
            #pragma HLS unroll
            unsigned requested_olid = in_addr[ILid] % PACK_SIZE;
            bool in_granted = (in_valid[ILid]
                            && out_valid[requested_olid]
                            && (xbar_sel[requested_olid] == ILid));
            in_resend[ILid] = (in_valid[ILid] && !in_granted) ? 1 : 0;
            resend_pld[ILid] = in_pld[ILid];
        }
    }
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
// 1 kind of shuffler_cores

    // shuffler states
    #define SF_WORKING 0 // normal working state
    #define SF_ENDING 1 // flushing the remaining packets in the arbiter

    // shuffler core for read responses: works on 1 partition
    void shuffler_core_for_read_resp(
        // fifos
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &input_lanes,
        tapa::ostreams<UPDATE_PLD_T, PACK_SIZE> &output_lanes
    ) {
        const unsigned shuffler_extra_iters = (ARBITER_LATENCY + 1) * 2 * PACK_SIZE;
        // pipeline control variables
        ap_uint<PACK_SIZE> fetch_complete = 0;
        unsigned loop_extra_iters = shuffler_extra_iters;
        ap_uint<1> state = SF_WORKING;
        bool loop_exit = false;

        // payloads
        UPDATE_PLD_T payload[PACK_SIZE];
        #pragma HLS array_partition variable=payload complete
        ap_uint<PACK_SIZE> valid = 0;

        // resend control
        #define SF_REG (ARBITER_LATENCY+2)
        UPDATE_PLD_T payload_resend[SF_REG][PACK_SIZE];
        #pragma HLS array_partition variable=payload_resend type=complete dim=0
        ap_uint<PACK_SIZE> resend[SF_REG];
        #pragma HLS array_partition variable=resend type=complete dim=0

        for (unsigned k = 0; k < SF_REG; k++) {
            #pragma HLS unroll
            resend[k] = 0;
        }

        shuffler_resend_payload_resend_reset_unroll:
        for (unsigned i = 0; i < PACK_SIZE; i++) {
            #pragma HLS unroll
            for (unsigned k = 0; k < SF_REG; k++) {
                #pragma HLS unroll
                payload_resend[k][i] = (UPDATE_PLD_T){0,0,0,0};
            }
        }

        // arbiter outputs
        unsigned xbar_sel[PACK_SIZE];
        ap_uint<PACK_SIZE> xbar_valid = 0;
        #pragma HLS array_partition variable=xbar_sel complete
        // arbiter priority rotation
        unsigned rotate_priority = 0;
        unsigned next_rotate_priority = 0;

        loop_shuffle_pipeline:
        while (!loop_exit) {
            #pragma HLS pipeline II=1
            #pragma HLS dependence variable=resend intra true
            #pragma HLS dependence variable=payload_resend intra true

            // Fetch stage (F)
            for (unsigned ILid = 0; ILid < PACK_SIZE; ILid++) {
                #pragma HLS unroll
                if (resend[0][ILid]) {
                    valid[ILid] = 1;
                    payload[ILid] = payload_resend[0][ILid];
                } else if (fetch_complete[ILid]) {
                    valid[ILid] = 0;
                    payload[ILid] = (UPDATE_PLD_T){0,0,0,0};
                } else {
                    if (input_lanes[ILid].try_read(payload[ILid])) {
                        if (payload[ILid].inst == EOD) {
                            fetch_complete[ILid] = 1;
                            valid[ILid] = 0;
                        } else {
                            valid[ILid] = 1;
                        }
                    } else {
                        valid[ILid] = 0;
                        payload[ILid] = (UPDATE_PLD_T){0,0,0,0};
                    }
                }
            }

            switch (state) {
            case SF_WORKING:
                if (fetch_complete.and_reduce()) {
                    state = SF_ENDING;
                }
                break;
            case SF_ENDING:
                loop_extra_iters--;
                loop_exit = (loop_extra_iters == 0);
                break;
            default:
                break;
            }
            // ------- end of F stage

            for (unsigned k = 0; k < SF_REG - 1; ++k) {
                #pragma HLS unroll
                resend[k] = resend[k + 1];
            }

            for (unsigned k = 0; k < SF_REG - 1; ++k) {
                #pragma HLS unroll
                for (unsigned i = 0; i < PACK_SIZE; i++) {
                    #pragma HLS unroll
                    payload_resend[k][i] = payload_resend[k + 1][i];
                }
            }

            // Arbiter stage (A) pipeline arbiter, depth = 6
            rotate_priority = next_rotate_priority;
            arbiter_for_read_resp(
                payload,
                payload_resend[SF_REG - 1], // as return value
                valid,
                resend[SF_REG - 1], // as return value
                xbar_sel,
                xbar_valid,
                rotate_priority
            );
            next_rotate_priority = (rotate_priority + 1) % PACK_SIZE;
            // ------- end of A stage

            // crossbar stage (C)
            for (unsigned OLid = 0; OLid < PACK_SIZE; OLid++) {
                #pragma HLS unroll
                if (xbar_valid[OLid]) {
                    if (valid[xbar_sel[OLid]]) {
                        output_lanes[OLid].write(payload[xbar_sel[OLid]]);
                    }
                }

            }
            // ------- end of C stage
        } // main while() loop ends here

        for (unsigned OLid = 0; OLid < PACK_SIZE; OLid++) {
            #pragma HLS unroll
            output_lanes[OLid].write(UPDATE_PLD_EOD);
        }
    }
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
// 1 kind of shufflers

    void shuffler_read_resp(
        tapa::istreams<UPDATE_PLD_T, PACK_SIZE> &input_lanes,
        tapa::ostreams<UPDATE_PLD_T, PACK_SIZE> &output_lanes
    ) {
        bool first_launch = true;
        ap_uint<PACK_SIZE> got_EOS = 0;
        while (!got_EOS.and_reduce()) {
            #pragma HLS pipeline off
            ap_uint<PACK_SIZE> got_SOD = 0;

            if (first_launch) {
                loop_sync_on_SOD:
                while (!got_SOD.and_reduce()) {
                    #pragma HLS pipeline II=1
                    for (unsigned ILid = 0; ILid < PACK_SIZE; ILid++) {
                        #pragma HLS unroll
                        if (!got_SOD[ILid]) {
                            UPDATE_PLD_T p;
                            if (input_lanes[ILid].try_read(p)) {
                                if (p.inst == SOD) {
                                    got_SOD[ILid] = 1;
                                }
                            }
                        }
                    }
                } // while() : sync on first SOD
                first_launch = false;
            } // first launch SOD sync

            for (unsigned OLid = 0; OLid < PACK_SIZE; OLid++) {
                #pragma HLS unroll
                output_lanes[OLid].write(UPDATE_PLD_SOD);
            }

            shuffler_core_for_read_resp(input_lanes, output_lanes);

            got_SOD = 0;
            loop_sync_on_SOD_EOS:
            while (!(got_SOD.and_reduce() || got_EOS.and_reduce())) {
                #pragma HLS pipeline II=1
                for (unsigned ILid = 0; ILid < PACK_SIZE; ILid++) {
                    #pragma HLS unroll
                    if (!(got_SOD[ILid] || got_EOS[ILid])) {
                        UPDATE_PLD_T p;
                        if (input_lanes[ILid].try_read(p)) {
                            if (p.inst == EOS) {
                                got_EOS[ILid] = 1;
                            } else if (p.inst == SOD) {
                                got_SOD[ILid] = 1;
                            }
                        }
                    }
                }
            } // while() : EOS or SOD sync
        } // while() : EOS sync

        for (unsigned OLid = 0; OLid < PACK_SIZE; OLid++) {
            #pragma HLS unroll
            output_lanes[OLid].write(UPDATE_PLD_EOS);
        }
    }
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
// process engine

    #ifdef __SYNTHESIS__
    #include "utils/x_hls_utils.h" // for reg() function
    #else
    #ifndef REG_FOR_SW_EMU
    #define REG_FOR_SW_EMU
    template<typename T>
    T reg(T in) {
        return in;
    }
    #endif
    #endif

    // data type for IFWQ
    struct IN_FLIGHT_WRITE {
        bool valid;
        IDX_T addr;
        VAL_T value;
    };

    // pe process pipeline
    void ufixed_pe_process(
        tapa::istream<UPDATE_PLD_T> &input,
        VAL_T output_buffer[OB_BANK_SIZE]
    ) {
        bool exit = false;

        // in-flight write queue for data-forwarding
        // designed for URAM latnecy=3 (RDL=3, WRL=2)
        IN_FLIGHT_WRITE ifwq[5];
        #pragma HLS array_partition variable=ifwq complete;
        ifwq[0] = (IN_FLIGHT_WRITE){false, 0, 0};
        ifwq[1] = (IN_FLIGHT_WRITE){false, 0, 0};
        ifwq[2] = (IN_FLIGHT_WRITE){false, 0, 0};
        ifwq[3] = (IN_FLIGHT_WRITE){false, 0, 0};
        ifwq[4] = (IN_FLIGHT_WRITE){false, 0, 0};

        pe_process_loop:
        while (!exit) {
            #pragma HLS pipeline II=1
            #pragma HLS dependence variable=output_buffer inter false
            #pragma HLS dependence variable=ifwq intra true
            #pragma HLS dependence variable=ifwq inter false
            bool valid = false;
            UPDATE_PLD_T pld;
            if(input.try_read(pld)) {
    #ifdef PE_LINE_TRACING
            std::cout << "  input payload: " << pld << std::endl;
    #endif
                if (pld.inst == EOD) {
                    exit = true;
                    valid = false;
                } else {
                    exit = false;
                    valid = true;
                }
            } else {
                #ifndef __SYNTHESIS__
                pld = (UPDATE_PLD_T){0,0,0,0};
                #endif
                valid = false;
            }

            IN_FLIGHT_WRITE ifwq_new_entry;
            IDX_T bank_addr = pld.row_idx / PACK_SIZE;
            VAL_T incr = pld.mat_val * pld.vec_val;
            VAL_T q = output_buffer[bank_addr];
            VAL_T q_fwd = ((bank_addr == ifwq[0].addr) && ifwq[0].valid) ? ifwq[0].value :
                        ((bank_addr == ifwq[1].addr) && ifwq[1].valid) ? ifwq[1].value :
                        ((bank_addr == ifwq[2].addr) && ifwq[2].valid) ? ifwq[2].value :
                        ((bank_addr == ifwq[3].addr) && ifwq[3].valid) ? ifwq[3].value :
                        ((bank_addr == ifwq[4].addr) && ifwq[4].valid) ? ifwq[4].value :
                        q;
            VAL_T new_q = q_fwd + incr;
            #pragma HLS bind_op variable=new_q op=add impl=dsp latency=0
            VAL_T new_q_reg = reg(new_q); // force a register after addition
            ifwq_new_entry.addr = bank_addr;
            ifwq_new_entry.value = new_q;
            ifwq_new_entry.valid = valid;

            if (valid) {
                output_buffer[bank_addr] = new_q_reg;
            }

            ifwq[4] = ifwq[3];
            ifwq[3] = ifwq[2];
            ifwq[2] = ifwq[1];
            ifwq[1] = ifwq[0];
            ifwq[0] = ifwq_new_entry;

        }
    }

    // pe output pipeline
    void ufixed_pe_output_sparse(
        tapa::ostream<VEC_PLD_T> &output,
        VAL_T output_buffer[OB_BANK_SIZE],
        const unsigned id,
        const unsigned used_buf_len
    ) {
        pe_output_loop:
        for (unsigned dump_count = 0; dump_count < used_buf_len; dump_count++) {
            #pragma HLS pipeline II=1
            VAL_T q = output_buffer[dump_count];
            if (q != 0) {
                VEC_PLD_T out_pld;
                out_pld.val = q;
                out_pld.idx = dump_count * PACK_SIZE + id;
                out_pld.inst = 0x0;
                output.write(out_pld);
    #ifdef PE_LINE_TRACING
                std::cout << "  write output: " << VEC_PLD_EOD << std::endl;
    #endif
            }
        }
    }

    // unsigned fixed-point pe
    void pe_bram_sparse(
        tapa::istream<UPDATE_PLD_T> &input,
        tapa::ostream<VEC_PLD_T> &output,
        unsigned id,
        IDX_T num_rows
        // unsigned used_buf_len
    ) {
        IDX_T num_parts = (num_rows + SPMSPV_OUT_BUF_LEN - 1) / SPMSPV_OUT_BUF_LEN;
        IDX_T num_rows_last_part = (num_rows % SPMSPV_OUT_BUF_LEN) ? (num_rows % SPMSPV_OUT_BUF_LEN) : SPMSPV_OUT_BUF_LEN;

        pe_bram_sparse_loop_over_parts:
        for (unsigned int part_id = 0; part_id < num_parts; part_id++) {
            #pragma HLS pipeline off
            IDX_T num_rows_this_part = (part_id == (num_parts - 1)) ? num_rows_last_part : SPMSPV_OUT_BUF_LEN;
            IDX_T used_buf_len = (num_rows_this_part + PACK_SIZE - 1) / PACK_SIZE;

            VAL_T output_buffer[OB_BANK_SIZE];
            #pragma HLS bind_storage variable=output_buffer type=RAM_2P impl=BRAM latency=3
            // reset output buffer before doing anything
            loop_reset_ob:
            for (unsigned i = 0; i < used_buf_len; i++) {
                #pragma HLS pipeline II=1
                output_buffer[i] = 0;
            }

            // wait on the first SOD
            bool got_SOD = false;
            pe_sync_SOD:
            while (!got_SOD) {
                #pragma HLS pipeline II=1
                UPDATE_PLD_T p = input.read();
                got_SOD = (p.inst == SOD);
            }

            // start processing
            bool exit = false;
            pe_main_loop:
            while (!exit) {
                #pragma HLS pipeline off
                // this function will exit upon EOD
                ufixed_pe_process(input, output_buffer);

                // read the next payload and decide whether continue processing or exit
                bool got_valid_pld = false;
                pe_sync_SODEOS:
                while (!got_valid_pld) {
                    #pragma HLS pipeline II=1
                    UPDATE_PLD_T p = input.read();
                    if (p.inst == SOD) {
                        got_valid_pld = true;
                        exit = false;
                    } else if (p.inst == EOS) {
                        got_valid_pld = true;
                        exit = true;
                    } else {
                        got_valid_pld = false;
                        exit = false;
                    }
                }
            }

            // dump results
            output.write(VEC_PLD_SOD);
            ufixed_pe_output_sparse(output, output_buffer, id,  used_buf_len);
            output.write(VEC_PLD_EOD);
            output.write(VEC_PLD_EOS);
        }
    }
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
// vector loader and result draining unit

    void load_vector_from_gmem(
        // vector data, row_id
        tapa::mmap<IDX_VAL_MMAP> vector,
        // number of non-zeros
        IDX_T vec_num_nnz,
        // number of matrix rows
        IDX_T num_parts,
        // fifo
        tapa::ostreams<IDX_VAL_INST_T, SPMSPV_NUM_HBM_CHANNEL> &VL_to_ML_stream
    ) {
        load_vector_from_gemm_loop_over_parts:
        for (unsigned int part_id = 0; part_id < num_parts; part_id++) {
            #pragma HLS pipeline off

            for (unsigned int k = 0; k < SPMSPV_NUM_HBM_CHANNEL; k++) {
                #pragma HLS unroll
                VL_to_ML_stream[k].write(IDX_VAL_INST_SOD);
            }

            loop_over_vector_values:
            for (unsigned int vec_nnz_cnt = 0; vec_nnz_cnt < vec_num_nnz; vec_nnz_cnt++) {
                #pragma HLS pipeline II=1
                IDX_VAL_INST_T instruction_to_ml;
                IDX_T index = IDX_VAL_CAST_INDEX(vector[vec_nnz_cnt + 1]);
                instruction_to_ml.index = index / SPMSPV_NUM_HBM_CHANNEL;
                instruction_to_ml.val(31,0) = IDX_VAL_CAST_VALUE(vector[vec_nnz_cnt + 1]);
                instruction_to_ml.inst = 0;
                VL_to_ML_stream[index % SPMSPV_NUM_HBM_CHANNEL].write(instruction_to_ml);
            }

            for (unsigned int k = 0; k < SPMSPV_NUM_HBM_CHANNEL; k++) {
                #pragma HLS unroll
                VL_to_ML_stream[k].write(IDX_VAL_INST_EOD);
                VL_to_ML_stream[k].write(IDX_VAL_INST_EOS);
            }

            #ifndef __SYNTHESIS__
            if (line_tracing_spmspv_load_data) {
                std::cout << "INFO: [kernel SpMSpV] Load vector finished, vec_num_nnz = "
                        << vec_num_nnz << std::endl << std::flush;
            }
            #endif
        }
    }

    void write_back_results (
        tapa::istreams<VEC_PLD_T, PACK_SIZE> &PE2WB,
        tapa::mmap<IDX_VAL_MMAP> result,
        IDX_T num_parts
    ) {
        IDX_T nnz_cnt = 0;
        IDX_T mat_row_id_base = 0;
        read_from_pe_streams_loop_over_parts:
        for (unsigned int part_id = 0; part_id < num_parts; part_id++) {
            #pragma HLS pipeline off

            bool exit = false;
            char current_input = 0; // read from multiple PE output streams
            ap_uint<PACK_SIZE> finished = 0;

            spmspv_write_back_loop:
            while (!exit) {
                #pragma HLS pipeline II=1

                VEC_PLD_T pld;
                if (!finished[current_input]) {
                    if (PE2WB[current_input].try_read(pld)) {
                        if (pld.inst == EOS) {
                            finished[current_input] = true;
                            current_input = (current_input + 1) % PACK_SIZE; // switch to next pe
                        } else if (pld.inst != SOD && pld.inst != EOD) {
                            IDX_T index = mat_row_id_base + pld.idx;
                            // IDX_VAL_T res_pld;
                            // res_pld.index = index;
                            // res_pld.val = pld.val;
                            nnz_cnt++;
                            IDX_VAL_CAST_INDEX(result[nnz_cnt]) = index;
                            IDX_VAL_CAST_VALUE(result[nnz_cnt]) = pld.val(31,0);
                        }
                    } else {
                        current_input = (current_input + 1) % PACK_SIZE; // switch to next pe
                    }
                } else {
                    current_input = (current_input + 1) % PACK_SIZE; // switch to next pe
                }
                exit = finished.and_reduce();
            }

            #ifndef __SYNTHESIS__
                if (line_tracing_spmspv_write_back) {
                    std::cout << "INFO: [kernel SpMSpV] Cummulative NNZ:"
                                << " " << nnz_cnt << std::endl << std::flush;
                }
            #endif

            mat_row_id_base += SPMSPV_OUT_BUF_LEN;
        }

        IDX_VAL_CAST_INDEX(result[0]) = nnz_cnt;
    }

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

// top-level kernel function
void spmspv(
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
    tapa::mmap<SPMSPV_MAT_PKT_MMAP> mat_0,  // in,   HBM[0]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
    tapa::mmap<SPMSPV_MAT_PKT_MMAP> mat_1,  // in,   HBM[1]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
    tapa::mmap<SPMSPV_MAT_PKT_MMAP> mat_2,  // in,   HBM[2]
    tapa::mmap<SPMSPV_MAT_PKT_MMAP> mat_3,  // in,   HBM[3]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 6)
    tapa::mmap<SPMSPV_MAT_PKT_MMAP> mat_4,  // in,   HBM[4]
    tapa::mmap<SPMSPV_MAT_PKT_MMAP> mat_5,  // in,   HBM[5]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
    tapa::mmap<SPMSPV_MAT_PKT_MMAP> mat_6,  // in,   HBM[6]
    tapa::mmap<SPMSPV_MAT_PKT_MMAP> mat_7,  // in,   HBM[7]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 10)
    tapa::mmap<SPMSPV_MAT_PKT_MMAP> mat_8,  // in,   HBM[8]
    tapa::mmap<SPMSPV_MAT_PKT_MMAP> mat_9,  // in,   HBM[9]
#endif
    tapa::mmap<IDX_VAL_MMAP> vector,        // in,   HBM[30]
    tapa::mmap<IDX_VAL_MMAP> result,        // out,  HBM[31]
    IDX_T num_rows,                         // in
    IDX_T num_parts,                        // in
    IDX_T num_cols,                         // in
    IDX_T num_vec_nnz                       // in
) {
    tapa::streams<IDX_VAL_INST_T, SPMSPV_NUM_HBM_CHANNEL, FIFO_DEPTH> VL2ML;
    tapa::streams<INST_T, SPMSPV_NUM_HBM_CHANNEL, FIFO_DEPTH> ML2MG_inst;
    // Note: tapa::streams<> array[N] is not valid for tapac flow, even through
    //       it passes sw_emu.
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> ML2MG_0;
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> ML2MG_1;
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> ML2MG_2;
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> ML2MG_3;
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 6)
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> ML2MG_4;
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> ML2MG_5;
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> ML2MG_6;
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> ML2MG_7;
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 10)
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> ML2MG_8;
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> ML2MG_9;
#endif
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> MG2SF;
    tapa::streams<UPDATE_PLD_T, PACK_SIZE, FIFO_DEPTH> SF2PE;
    tapa::streams<VEC_PLD_T, PACK_SIZE, FIFO_DEPTH> PE2WB;

    tapa::task()
    .invoke(load_vector_from_gmem,
        vector,
        num_vec_nnz,
        num_parts,
        VL2ML
    )
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
    .invoke(load_matrix_from_gmem,
        mat_0,
        num_parts,
        num_cols,
        0,
        VL2ML[0],
        ML2MG_inst[0],
        ML2MG_0
    )
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
    .invoke(load_matrix_from_gmem,
        mat_1,
        num_parts,
        num_cols,
        1,
        VL2ML[1],
        ML2MG_inst[1],
        ML2MG_1
    )
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
    .invoke(load_matrix_from_gmem,
        mat_2,
        num_parts,
        num_cols,
        2,
        VL2ML[2],
        ML2MG_inst[2],
        ML2MG_2
    )
    .invoke(load_matrix_from_gmem,
        mat_3,
        num_parts,
        num_cols,
        3,
        VL2ML[3],
        ML2MG_inst[3],
        ML2MG_3
    )
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 6)
    .invoke(load_matrix_from_gmem,
        mat_4,
        num_parts,
        num_cols,
        4,
        VL2ML[4],
        ML2MG_inst[4],
        ML2MG_4
    )
    .invoke(load_matrix_from_gmem,
        mat_5,
        num_parts,
        num_cols,
        5,
        VL2ML[5],
        ML2MG_inst[5],
        ML2MG_5
    )
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
    .invoke(load_matrix_from_gmem,
        mat_6,
        num_parts,
        num_cols,
        6,
        VL2ML[6],
        ML2MG_inst[6],
        ML2MG_6
    )
    .invoke(load_matrix_from_gmem,
        mat_7,
        num_parts,
        num_cols,
        7,
        VL2ML[7],
        ML2MG_inst[7],
        ML2MG_7
    )
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 10)
    .invoke(load_matrix_from_gmem,
        mat_8,
        num_parts,
        num_cols,
        8,
        VL2ML[8],
        ML2MG_inst[8],
        ML2MG_8
    )
    .invoke(load_matrix_from_gmem,
        mat_9,
        num_parts,
        num_cols,
        9,
        VL2ML[9],
        ML2MG_inst[9],
        ML2MG_9
    )
#endif
    .invoke(merge_load_streams,
        ML2MG_inst,
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
        ML2MG_0,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
        ML2MG_1,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
        ML2MG_2,
        ML2MG_3,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 6)
        ML2MG_4,
        ML2MG_5,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
        ML2MG_6,
        ML2MG_7,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 10)
        ML2MG_8,
        ML2MG_9,
#endif
        MG2SF,
        num_parts
    ).invoke(shuffler_read_resp,
        MG2SF,
        SF2PE
    ).invoke(pe_bram_sparse,
        SF2PE[0],
        PE2WB[0],
        0,
        num_rows
    ).invoke(pe_bram_sparse,
        SF2PE[1],
        PE2WB[1],
        1,
        num_rows
    ).invoke(pe_bram_sparse,
        SF2PE[2],
        PE2WB[2],
        2,
        num_rows
    ).invoke(pe_bram_sparse,
        SF2PE[3],
        PE2WB[3],
        3,
        num_rows
    ).invoke(pe_bram_sparse,
        SF2PE[4],
        PE2WB[4],
        4,
        num_rows
    ).invoke(pe_bram_sparse,
        SF2PE[5],
        PE2WB[5],
        5,
        num_rows
    ).invoke(pe_bram_sparse,
        SF2PE[6],
        PE2WB[6],
        6,
        num_rows
    ).invoke(pe_bram_sparse,
        SF2PE[7],
        PE2WB[7],
        7,
        num_rows
    ).invoke(write_back_results,
        PE2WB,
        result,
        num_parts
    );

} // kernel
