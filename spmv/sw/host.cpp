#include "common.h"

#include "data_loader.h"
#include "data_formatter.h"

#include <iostream>
#include <iomanip>
#include <assert.h>

#include <gflags/gflags.h>
#include <tapa.h>

void spmv(
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_0,       // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_1,       // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_2,       // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_3,       // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_4,       // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_5,       // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_6,       // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_7,       // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_8,       // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_9,       // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_10,      // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_11,      // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_12,      // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_13,      // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_14,      // in
    tapa::mmap<SPMV_MAT_PKT_T> matrix_hbm_15,      // in
    tapa::mmap<PACKED_VAL_T> packed_dense_vector,  // in
    tapa::mmap<PACKED_VAL_T> packed_dense_result,  // out
    unsigned num_columns,                          // in
    unsigned num_partitions,                       // in
    unsigned num_col_partitions,                   // in
    unsigned row_partition_idx,                    // in
    unsigned rows_per_c_in_partition               // in
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

//--------------------------------------------------------------------------------------------------
// reference and verify utils
//--------------------------------------------------------------------------------------------------

void compute_ref(
    spmv_::io::CSRMatrix<float> &mat,
    std::vector<float> &vector,
    std::vector<float> &ref_result
) {
    ref_result.resize(mat.num_rows);
    std::fill(ref_result.begin(), ref_result.end(), 0);
    for (size_t row_idx = 0; row_idx < mat.num_rows; row_idx++) {
        IDX_T start = mat.adj_indptr[row_idx];
        IDX_T end = mat.adj_indptr[row_idx + 1];
        for (size_t i = start; i < end; i++) {
            IDX_T idx = mat.adj_indices[i];
            ref_result[row_idx] += mat.adj_data[i] * vector[idx];
        }
    }
}

bool verify(std::vector<float> reference_results,
            std::vector<VAL_T> kernel_results) {
    float epsilon = 0.0001;
    if (reference_results.size() != kernel_results.size()) {
        std::cout << "Error: Size mismatch"
                      << std::endl;
        std::cout   << "  Reference result size: " << reference_results.size()
                    << "  Kernel result size: " << kernel_results.size()
                    << std::endl;
        return false;
    }
    for (size_t i = 0; i < reference_results.size(); i++) {
        bool match = abs(float(kernel_results[i]) - reference_results[i]) < epsilon;
        if (!match) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "  i = " << i
                      << "  Reference result = " << reference_results[i]
                      << "  Kernel result = " << kernel_results[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

void unpack_vector(
    aligned_vector<PACKED_VAL_T> &pdv,
    std::vector<VAL_T> &dv
) {
    dv.resize(pdv.size() * PACK_SIZE);
    for (size_t i = 0; i < pdv.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            dv[i * PACK_SIZE + k] = pdv[i].data[k];
        }
    }
}



//---------------------------------------------------------------
// test harness
//---------------------------------------------------------------

bool spmv_test_harness (
    std::string bitstream,
    spmv_::io::CSRMatrix<float> &ext_matrix,
    bool skip_empty_rows
) {
    using namespace spmv_::io;

    //--------------------------------------------------------------------
    // load and format the matrix
    //--------------------------------------------------------------------
    std::cout << "INFO : Test started" << std::endl;
    util_round_csr_matrix_dim<float>(ext_matrix, PACK_SIZE * NUM_HBM_CHANNELS * INTERLEAVE_FACTOR, PACK_SIZE);
    CSRMatrix<VAL_T> mat = csr_matrix_convert_from_float<VAL_T>(ext_matrix);

    size_t num_row_partitions = (mat.num_rows + LOGICAL_OB_SIZE - 1) / LOGICAL_OB_SIZE;
    size_t num_col_partitions = (mat.num_cols + LOGICAL_VB_SIZE - 1) / LOGICAL_VB_SIZE;
    size_t num_partitions = num_row_partitions * num_col_partitions;
    size_t num_virtual_hbm_channels = NUM_HBM_CHANNELS * INTERLEAVE_FACTOR;
    CPSRMatrix<PACKED_VAL_T, PACKED_IDX_T, PACK_SIZE> cpsr_matrix
        = csr2cpsr<PACKED_VAL_T, PACKED_IDX_T, VAL_T, IDX_T, PACK_SIZE>(
            mat,
            IDX_MARKER,
            LOGICAL_OB_SIZE,
            LOGICAL_VB_SIZE,
            num_virtual_hbm_channels,
            skip_empty_rows
        );
    using partition_indptr_t = struct {IDX_T start; PACKED_IDX_T nnz;};
    using ch_partition_indptr_t = std::vector<partition_indptr_t>;
    using ch_packed_idx_t = std::vector<PACKED_IDX_T>;
    using ch_packed_val_t = std::vector<PACKED_VAL_T>;
    using ch_mat_pkt_t = aligned_vector<SPMV_MAT_PKT_T>;
    std::vector<ch_partition_indptr_t> channel_partition_indptr(num_virtual_hbm_channels);
    for (size_t c = 0; c < num_virtual_hbm_channels; c++) {
        channel_partition_indptr[c].resize(num_partitions);
        channel_partition_indptr[c][0].start = 0;
    }
    std::vector<ch_packed_idx_t> channel_indices(num_virtual_hbm_channels);
    std::vector<ch_packed_val_t> channel_vals(num_virtual_hbm_channels);
    std::vector<ch_mat_pkt_t> channel_packets(NUM_HBM_CHANNELS);
    // Iterate virtual channels and map virtual channels (vc) to physical channels (pc)
    for (size_t pc = 0; pc < NUM_HBM_CHANNELS; pc++) {
        for (size_t j = 0; j < num_row_partitions; j++) {
            for (size_t i = 0; i < num_col_partitions; i++) {
                size_t num_packets_each_virtual_channel[INTERLEAVE_FACTOR];
                for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
                    size_t vc = pc + f * NUM_HBM_CHANNELS;
                    auto indptr_partition = cpsr_matrix.get_packed_indptr(j, i, vc);
                    uint32_t num_packets = *std::max_element(indptr_partition.back().data,
                                                             indptr_partition.back().data + PACK_SIZE);
                    num_packets_each_virtual_channel[f] = num_packets;
                }
                uint32_t max_num_packets = *std::max_element(num_packets_each_virtual_channel,
                                                             num_packets_each_virtual_channel + INTERLEAVE_FACTOR);
                for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
                    size_t vc = pc + f * NUM_HBM_CHANNELS;
                    auto indices_partition = cpsr_matrix.get_packed_indices(j, i, vc);
                    channel_indices[vc].insert(channel_indices[vc].end(), indices_partition.begin(), indices_partition.end());
                    auto vals_partition = cpsr_matrix.get_packed_data(j, i, vc);
                    channel_vals[vc].insert(channel_vals[vc].end(), vals_partition.begin(), vals_partition.end());
                    channel_indices[vc].resize(channel_partition_indptr[vc][j*num_col_partitions + i].start
                                               + max_num_packets);
                    channel_vals[vc].resize(channel_partition_indptr[vc][j*num_col_partitions + i].start
                                            + max_num_packets);
                    assert(channel_indices[vc].size() == channel_vals[vc].size());
                    auto indptr_partition = cpsr_matrix.get_packed_indptr(j, i, vc);
                    channel_partition_indptr[vc][j*num_col_partitions + i].nnz = indptr_partition.back();
                    if (!((j == (num_row_partitions - 1)) && (i == (num_col_partitions - 1)))) {
                        channel_partition_indptr[vc][j*num_col_partitions + i + 1].start =
                            channel_partition_indptr[vc][j*num_col_partitions + i].start + max_num_packets;
                    }
                }
            }
        }

        channel_packets[pc].resize(num_partitions*(1+INTERLEAVE_FACTOR) + channel_indices[pc].size()*INTERLEAVE_FACTOR);
        // partition indptr
        for (size_t ij = 0; ij < num_partitions; ij++) {
            channel_packets[pc][ij*(1+INTERLEAVE_FACTOR)].indices.data[0] =
                channel_partition_indptr[pc][ij].start * INTERLEAVE_FACTOR;
            for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
                size_t vc = pc + f * NUM_HBM_CHANNELS;
                channel_packets[pc][ij*(1+INTERLEAVE_FACTOR) + 1 + f].indices = channel_partition_indptr[vc][ij].nnz;
            }
        }
        // matrix indices and vals
        uint32_t offset = num_partitions*(1+INTERLEAVE_FACTOR);
        for (size_t i = 0; i < channel_indices[pc].size(); i++) {
            for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
                size_t vc = pc + f * NUM_HBM_CHANNELS;
                size_t ii = i*INTERLEAVE_FACTOR + f;
                channel_packets[pc][offset + ii].indices = channel_indices[vc][i];
                channel_packets[pc][offset + ii].vals = channel_vals[vc][i];
            }
        }
    }
    std::cout << "INFO : Matrix loading/preprocessing complete!" << std::endl;

    //--------------------------------------------------------------------
    // generate input vector
    //--------------------------------------------------------------------
    std::vector<float> vector_f(ext_matrix.num_cols);
    std::generate(vector_f.begin(), vector_f.end(), [&](){return float(rand() % 2);});
    aligned_vector<PACKED_VAL_T> vector(mat.num_cols / PACK_SIZE);
    for (size_t i = 0; i < vector.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            vector[i].data[k] = VAL_T(vector_f[i*PACK_SIZE + k]);
        }
    }

    //--------------------------------------------------------------------
    // allocate space for results
    //--------------------------------------------------------------------
    aligned_vector<PACKED_VAL_T> result(mat.num_rows / PACK_SIZE);
    for (size_t i = 0; i < result.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            result[i].data[k] = 0;
        }
    }
    std::cout << "INFO : Input/result initialization complete!" << std::endl;

    //--------------------------------------------------------------------
    // invoke kernel
    //--------------------------------------------------------------------
    std::cout << "INFO : Invoking kernel:" << std::endl;
    std::cout << "  row_partitions: " << num_row_partitions << std::endl;
    std::cout << "  col_partitions: " << num_col_partitions << std::endl;

    size_t rows_per_ch_in_last_row_part;
    if (mat.num_rows % LOGICAL_OB_SIZE == 0) {
        rows_per_ch_in_last_row_part = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    } else {
        rows_per_ch_in_last_row_part = mat.num_rows % LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    }

    for (size_t row_part_id = 0; row_part_id < num_row_partitions; row_part_id++) {
        unsigned part_len = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
        if (row_part_id == num_row_partitions - 1) {
            part_len = rows_per_ch_in_last_row_part;
        }
        std::cout << "INFO : SpMV Kernel Started: row partition " << row_part_id
                  << " with " << part_len << " rows per cluster" << std::endl;

        double kernel_time_taken_ns
            = tapa::invoke(spmv, bitstream,
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[0]),   // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[1]),   // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[2]),   // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[3]),   // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[4]),   // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[5]),   // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[6]),   // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[7]),   // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[8]),   // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[9]),   // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[10]),  // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[11]),  // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[12]),  // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[13]),  // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[14]),  // in
                            tapa::read_only_mmap<SPMV_MAT_PKT_T>(channel_packets[15]),  // in
                            tapa::read_only_mmap<PACKED_VAL_T>(vector),                 // in
                            tapa::write_only_mmap<PACKED_VAL_T>(result),                // out
                            (unsigned)mat.num_cols,                                     // in
                            (unsigned)num_partitions,                                   // in
                            (unsigned)num_col_partitions,                               // in
                            (unsigned)row_part_id,                                      // in
                            (unsigned)part_len                                          // in
                );

        std::cout << "INFO : SpMV Kernel Finished: row partition " << row_part_id << std::endl;
        std::cout << "INFO : SpMV Kernel Time is " << kernel_time_taken_ns * 1e-6 << "ms" << std::endl;
    }
    std::cout << "INFO : SpMV kernel complete!" << std::endl;

    //--------------------------------------------------------------------
    // compute reference
    //--------------------------------------------------------------------
    std::vector<float> ref_result;
    compute_ref(ext_matrix, vector_f, ref_result);
    std::cout << "INFO : Compute reference complete!" << std::endl;

    //--------------------------------------------------------------------
    // verify
    //--------------------------------------------------------------------
    std::cout << "INFO : Device -> Host data transfer complete!" << std::endl;

    std::vector<VAL_T> upk_result;
    unpack_vector(result, upk_result);
    return verify(ref_result, upk_result);
}

//---------------------------------------------------------------
// test case utils
//---------------------------------------------------------------
spmv_::io::CSRMatrix<float> create_dense_CSR (
    unsigned num_rows,
    unsigned num_cols
) {
    spmv_::io::CSRMatrix<float> mat_f;
    mat_f.num_rows = num_rows;
    mat_f.num_cols = num_cols;
    mat_f.adj_data.resize(num_rows * num_cols);
    mat_f.adj_indices.resize(num_rows * num_cols);
    mat_f.adj_indptr.resize(num_rows + 1);

    for (auto &x : mat_f.adj_data) {x = 1;}

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            mat_f.adj_indices[i*num_cols + j] = j;
        }
    }
    for (size_t i = 0; i < num_rows + 1; i++) {
        mat_f.adj_indptr[i] = num_cols*i;
    }
    return mat_f;
}

spmv_::io::CSRMatrix<float> create_uniform_sparse_CSR (
    unsigned num_rows,
    unsigned num_cols,
    unsigned nnz_per_row
) {
    spmv_::io::CSRMatrix<float> mat_f;
    mat_f.num_rows = num_rows;
    mat_f.num_cols = num_cols;
    mat_f.adj_data.resize(num_rows * nnz_per_row);
    mat_f.adj_indices.resize(num_rows * nnz_per_row);
    mat_f.adj_indptr.resize(num_rows + 1);

    for (auto &x : mat_f.adj_data) {x = 1;}

    unsigned indice_step = num_cols / nnz_per_row;
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < nnz_per_row; j++) {
            mat_f.adj_indices[i*nnz_per_row + j] = (indice_step*j + i) % num_cols;
        }
    }
    for (size_t i = 0; i < num_rows + 1; i++) {
        mat_f.adj_indptr[i] = nnz_per_row*i;
    }
    return mat_f;
}

//---------------------------------------------------------------
// test cases
//---------------------------------------------------------------
std::string DATASET_DIR = std::getenv("DATASETS");
std::string GRAPH_DATASET_DIR = DATASET_DIR + "/graph/";
std::string NN_DATASET_DIR = DATASET_DIR + "/pruned_nn/";

bool test_basic(std::string bitstream) {
    std::cout << "------ Running test: on basic dense matrix " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f = create_dense_CSR(128, 128);
    for (auto &x : mat_f.adj_data) {x = 1;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_basic_sparse(std::string bitstream) {
    std::cout << "------ Running test: on basic sparse matrix " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f = create_uniform_sparse_CSR(1000, 1024, 10);
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_medium_sparse(std::string bitstream) {
    std::cout << "------ Running test: on uniform 10K 10 (100K, 1M) " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f = create_uniform_sparse_CSR(10000, 10000, 10);
    for (auto &x : mat_f.adj_data) {x = 1;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_gplus(std::string bitstream) {
    std::cout << "------ Running test: on google_plus (108K, 13M) " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(GRAPH_DATASET_DIR + "gplus_108K_13M_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_ogbl_ppa(std::string bitstream) {
    std::cout << "------ Running test: on ogbl_ppa (576K, 42M) " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(GRAPH_DATASET_DIR + "ogbl_ppa_576K_42M_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_transformer_50_t(std::string bitstream) {
    std::cout << "------ Running test: on transformer-50-t" << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(NN_DATASET_DIR + "transformer_50_512_33288_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, true)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_transformer_95_t(std::string bitstream) {
    std::cout << "------ Running test: on transformer-95-t" << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(NN_DATASET_DIR + "transformer_95_512_33288_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, true)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

//---------------------------------------------------------------
// main
//---------------------------------------------------------------

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

int main (int argc, char** argv) {

    bool passed = true;
    passed = passed && test_basic(FLAGS_bitstream);
    passed = passed && test_basic_sparse(FLAGS_bitstream);
    passed = passed && test_medium_sparse(FLAGS_bitstream);
    passed = passed && test_gplus(FLAGS_bitstream);
    passed = passed && test_ogbl_ppa(FLAGS_bitstream);
    passed = passed && test_transformer_50_t(FLAGS_bitstream);
    passed = passed && test_transformer_95_t(FLAGS_bitstream);

    std::cout << (passed ? "===== All Test Passed! =====" : "===== Test FAILED! =====") << std::endl;
    return passed ? 0 : 1;
}
