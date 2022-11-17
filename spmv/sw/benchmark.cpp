#include "common.h"

#include "data_loader.h"
#include "data_formatter.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <assert.h>

#include <gflags/gflags.h>
#include <tapa.h>

const unsigned NUM_RUNS = 32;

void spmv(
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_0,       // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_1,       // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_2,       // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_3,       // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_4,       // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_5,       // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_6,       // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_7,       // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_8,       // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_9,       // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_10,      // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_11,      // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_12,      // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_13,      // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_14,      // in
    tapa::mmap<_SPMV_MAT_PKT_T> matrix_hbm_15,      // in
    tapa::mmap<_PACKED_VAL_T> packed_dense_vector,  // in
    tapa::mmap<_PACKED_VAL_T> packed_dense_result,  // out
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
    aligned_vector<_PACKED_VAL_T> &pdv,
    std::vector<VAL_T> &dv
) {
    dv.resize(pdv.size() * PACK_SIZE);
    for (size_t i = 0; i < pdv.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            dv[i * PACK_SIZE + k](31,0) = pdv[i](31+32*k, 32*k);
        }
    }
}

inline _SPMV_MAT_PKT_T bits(SPMV_MAT_PKT_T &mat_pkt) {
    _SPMV_MAT_PKT_T temp;
    for (size_t i = 0; i < PACK_SIZE; i++) {
        temp(31+32*i, 32*i) = mat_pkt.indices.data[i];
    }
    for (size_t i = 0; i < PACK_SIZE; i++) {
        temp(32*PACK_SIZE+31+32*i, 32*PACK_SIZE+32*i) = VAL_T(mat_pkt.vals.data[i])(31,0);
    }
    return temp;
}

struct benchmark_result {
    std::string benchmark_name;
    double preprocess_time_s;
    double spmv_time_ms;
    double throughput_GBPS;
    double throughput_GOPS;
    bool verified;
};

template <typename T>
inline std::string fmt_key_val(std::string key, T val) {
    std::stringstream ss;
    ss << "\"" << key << "\": \"" << val << "\"";
    return ss.str();
}

// print the benchmark result in JSON format
std::ostream& operator<<(std::ostream& os, const benchmark_result &p) {
    os << "{ "
       << fmt_key_val("Benchmark", p.benchmark_name) << ", "
       << fmt_key_val("Preprocessing_s", p.preprocess_time_s) << ", "
       << fmt_key_val("SpMV_ms", p.spmv_time_ms) << ", "
       << fmt_key_val("TP_GBPS", p.throughput_GBPS) << ", "
       << fmt_key_val("TP_GOPS", p.throughput_GOPS) << ", "
       << fmt_key_val("verified", (int)p.verified) << " }";
    return os;
}


//---------------------------------------------------------------
// test harness
//---------------------------------------------------------------

benchmark_result spmv_test_harness (
    std::string bitstream,
    std::string bench_name,
    spmv_::io::CSRMatrix<float> &ext_matrix,
    bool skip_empty_rows
) {
    using namespace spmv_::io;
    using namespace std::chrono;

    benchmark_result rec = {bench_name, 0, 0, 0, 0, false};

    //--------------------------------------------------------------------
    // load and format the matrix
    //--------------------------------------------------------------------
    std::cout << "INFO : Test started" << std::endl;
    auto t0 = high_resolution_clock::now();

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

    auto t1 = high_resolution_clock::now();
    rec.preprocess_time_s = double(duration_cast<microseconds>(t1 - t0).count()) / 1000000;
    std::cout << "INFO : Matrix loading/preprocessing complete!" << std::endl;

    //--------------------------------------------------------------------
    // generate input vector
    //--------------------------------------------------------------------
    std::vector<float> vector_f(ext_matrix.num_cols);
    std::generate(vector_f.begin(), vector_f.end(), [&](){return float(rand() % 2);});
    aligned_vector<_PACKED_VAL_T> vector(mat.num_cols / PACK_SIZE);
    for (size_t i = 0; i < vector.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            vector[i](31+32*k, 32*k) = VAL_T(vector_f[i*PACK_SIZE + k])(31, 0);
        }
    }

    //--------------------------------------------------------------------
    // allocate space for results
    //--------------------------------------------------------------------
    aligned_vector<_PACKED_VAL_T> result(mat.num_rows / PACK_SIZE);
    for (size_t i = 0; i < result.size(); i++) {
        result[i] = 0;
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

    std::vector<aligned_vector<_SPMV_MAT_PKT_T>> _channel_packets(NUM_HBM_CHANNELS);
    for (size_t i = 0; i < NUM_HBM_CHANNELS; i++) {
        for (size_t j = 0; j < channel_packets[i].size(); j++) {
            _channel_packets[i].push_back(bits(channel_packets[i][j]));
        }
    }

    double total_run_time_ms = 0;

    for (unsigned run_iter = 0; run_iter < NUM_RUNS; run_iter++) {
        for (size_t row_part_id = 0; row_part_id < num_row_partitions; row_part_id++) {
            unsigned part_len = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
            if (row_part_id == num_row_partitions - 1) {
                part_len = rows_per_ch_in_last_row_part;
            }
            std::cout << "INFO : SpMV Kernel Started: row partition " << row_part_id
                    << " with " << part_len << " rows per cluster" << std::endl;

            double kernel_time_taken_ns
                = tapa::invoke(spmv, bitstream,
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[0]),   // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[1]),   // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[2]),   // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[3]),   // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[4]),   // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[5]),   // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[6]),   // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[7]),   // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[8]),   // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[9]),   // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[10]),  // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[11]),  // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[12]),  // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[13]),  // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[14]),  // in
                                tapa::read_only_mmap<_SPMV_MAT_PKT_T>(_channel_packets[15]),  // in
                                tapa::read_only_mmap<_PACKED_VAL_T>(vector),                 // in
                                tapa::write_only_mmap<_PACKED_VAL_T>(result),                // out
                                (unsigned)mat.num_cols,                                     // in
                                (unsigned)num_partitions,                                   // in
                                (unsigned)num_col_partitions,                               // in
                                (unsigned)row_part_id,                                      // in
                                (unsigned)part_len                                          // in
                    );
            total_run_time_ms += kernel_time_taken_ns * 1e-6;
        }
    }

    unsigned Nnz = mat.adj_data.size();
    double Mops = 2 * Nnz / 1000 / 1000;
    double gbs = double(Nnz * 2 * 4) / 1024.0 / 1024.0 / 1024.0;
    rec.spmv_time_ms = total_run_time_ms / NUM_RUNS;
    rec.throughput_GBPS = gbs / (rec.spmv_time_ms / 1000);
    rec.throughput_GOPS = Mops / rec.spmv_time_ms;

    std::cout << "INFO : SpMV kernel complete " << NUM_RUNS << " runs!" << std::endl;

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
    rec.verified = verify(ref_result, upk_result);
    return rec;
}

//---------------------------------------------------------------
// main
//---------------------------------------------------------------

int main (int argc, char** argv) {
    // ./benchmark <dataset_name> <dataset_path> <hw_xclbin> <log_path>
    std::string name = argv[1], dataset = argv[2], bitstream = argv[3], metric = argv[4];

    std::cout << "------ Running test: on " << name << std::endl;
    spmv_::io::CSRMatrix<float> mat_f = spmv_::io::load_csr_matrix_from_float_npz(dataset);
    for (auto &x : mat_f.adj_data) {x = 1.0 / mat_f.num_cols;}
    auto result = spmv_test_harness(bitstream, name, mat_f, false);

    std::ofstream log(metric);
    log << result;
    log.close();

    if (result.verified) {
        std::cout << "INFO : Testcase passed." << std::endl;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
    }
    return result.verified;
}
