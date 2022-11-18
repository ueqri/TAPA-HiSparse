#include "spmspv.h"

#include "data_loader.h"
#include "data_formatter.h"

#include <iostream>
#include <iomanip>
#include <assert.h>

#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>

#define SPMSPV_MAT_ARGS(x) \
tapa::mmap<SPMSPV_MAT_PKT_T> mat_##x

// top-level kernel function
void spmspv(
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
    SPMSPV_MAT_ARGS(0),             // in,  HBM[0]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
    SPMSPV_MAT_ARGS(1),             // in,  HBM[1]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
    SPMSPV_MAT_ARGS(2),             // in,  HBM[2]
    SPMSPV_MAT_ARGS(3),             // in,  HBM[3]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 6)
    SPMSPV_MAT_ARGS(4),             // in,  HBM[4]
    SPMSPV_MAT_ARGS(5),             // in,  HBM[5]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
    SPMSPV_MAT_ARGS(6),             // in,  HBM[6]
    SPMSPV_MAT_ARGS(7),             // in,  HBM[7]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 10)
    SPMSPV_MAT_ARGS(8),             // in,  HBM[8]
    SPMSPV_MAT_ARGS(9),             // in,  HBM[9]
#endif
    tapa::mmap<IDX_VAL_T> vector,   // inout, HBM[30]
    tapa::mmap<IDX_VAL_T> result,   // out,   HBM[31]
    unsigned num_rows,              // in
    unsigned num_parts,             // in
    unsigned num_cols,              // in
    unsigned num_vec_nnz            // in
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

// using aligned_dense_float_vec_t = aligned_vector<float>;
// using aligned_sparse_float_vec_t = std::vector<IDX_FLOAT_T>;

// using packet_t = struct {IDX_T indices[PACK_SIZE]; VAL_T vals[PACK_SIZE];};
// using aligned_packet_t = aligned_vector<packet_t>;

using spmv_::io::CSRMatrix;
using spmv_::io::load_csr_matrix_from_float_npz;

using spmspv_::io::csr2csc;
using spmspv_::io::CSCMatrix;
using spmspv_::io::FormattedCSCMatrix;
using spmspv_::io::ColumnCyclicSplitCSC;
using spmspv_::io::formatCSC;
using spmspv_::io::csc_matrix_convert_from_float;

//--------------------------------------------------------------------------------------------------
// reference and verify utils
//--------------------------------------------------------------------------------------------------
void compute_ref(
    CSCMatrix<float> &mat,
    std::vector<IDX_FLOAT_T> &vector,
    std::vector<float> &ref_result,
    uint32_t &involved_Nnz
) {
    // measure dimensions
    unsigned vec_nnz_total = vector[0].index;
    involved_Nnz = 0;

    // create result container
    ref_result.resize(mat.num_rows);
    std::fill(ref_result.begin(), ref_result.end(), 0);

    // indices of active columns are stored in vec_idx
    // number of active columns = vec_nnz_total
    // loop over all active columns
    for (unsigned active_col_id = 0; active_col_id < vec_nnz_total; active_col_id++) {

        float nnz_from_vec = vector[active_col_id + 1].val;
        unsigned current_col_id = vector[active_col_id + 1].index;

        // slice out the current column out of the active columns
        unsigned col_start = mat.adj_indptr[current_col_id];
        unsigned col_end = mat.adj_indptr[current_col_id + 1];

        // measure the involved Nnz in one SpMSpV run (only measure the matrix)
        involved_Nnz += col_end - col_start;

        // loop over all nnzs in the current column
        for (unsigned mat_element_id = col_start; mat_element_id < col_end; mat_element_id++) {
            unsigned current_row_id = mat.adj_indices[mat_element_id];
            float nnz_from_mat = mat.adj_data[mat_element_id];
            ref_result[current_row_id] += nnz_from_mat * nnz_from_vec;
        }
    }
}

template<typename data_t>
bool verify(std::vector<float> reference_results,
            std::vector<data_t> kernel_results) {
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

// convert a sparse vector to dense
template<typename sparse_vec_t, typename dense_vec_t>
void convert_sparse_vec_to_dense_vec(const sparse_vec_t &sparse_vector,
                                            dense_vec_t &dense_vector,
                                         const unsigned range) {
    dense_vector.resize(range);
    std::fill(dense_vector.begin(), dense_vector.end(), 0);
    int nnz = sparse_vector[0].index;
    for (int i = 1; i < nnz + 1; i++) {
        dense_vector[sparse_vector[i].index] = sparse_vector[i].val;
    }
}

// inline _SPMV_MAT_PKT_T bits(SPMV_MAT_PKT_T &mat_pkt) {
//     _SPMV_MAT_PKT_T temp;
//     for (size_t i = 0; i < PACK_SIZE; i++) {
//         temp(31+32*i, 32*i) = mat_pkt.indices.data[i];
//     }
//     for (size_t i = 0; i < PACK_SIZE; i++) {
//         temp(32*PACK_SIZE+31+32*i, 32*PACK_SIZE+32*i) = VAL_T(mat_pkt.vals.data[i])(31,0);
//     }
//     return temp;
// }

struct benchmark_result {
    std::string benchmark_name;
    float spmspv_sparsity;
    double preprocess_time_s;
    double spmspv_time_ms;
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
       << fmt_key_val("Sparsity", p.spmspv_sparsity) << ", "
       << fmt_key_val("Preprocessing_s", p.preprocess_time_s) << ", "
       << fmt_key_val("SpMSpV_ms", p.spmspv_time_ms) << ", "
       << fmt_key_val("TP_GBPS", p.throughput_GBPS) << ", "
       << fmt_key_val("TP_GOPS", p.throughput_GOPS) << ", "
       << fmt_key_val("verified", (int)p.verified) << " }";
    return os;
}

using namespace std::chrono;

// helper function to generate a dense CSC matrix
CSCMatrix<float> generate_dense_float_csc(unsigned dim) {
    CSCMatrix<float> float_csc;
    float_csc.num_rows = dim;
    float_csc.num_cols = dim;
    unsigned nnz = float_csc.num_rows*float_csc.num_cols;
    float_csc.adj_data.resize(nnz);
    float_csc.adj_indices.resize(nnz);
    float_csc.adj_indptr.resize(float_csc.num_cols + 1);
    std::fill(float_csc.adj_data.begin(), float_csc.adj_data.end(), 1.0/dim);
    float_csc.adj_indptr[0] = 0;
    for (size_t c = 0; c < dim; c++) {
        for (size_t r = 0; r < dim; r++) {
            float_csc.adj_indices[c * dim + r] = r;
        }
        float_csc.adj_indptr[c + 1] = float_csc.adj_indptr[c] + dim;
    }
    return float_csc;
}

//---------------------------------------------------------------
// test harness
//---------------------------------------------------------------

bool spmspv_test_harness (
    std::string bitstream,
    CSCMatrix<float> &csc_matrix_float,
    float vector_sparsity
) {
    //--------------------------------------------------------------------
    // load and format the matrix
    //--------------------------------------------------------------------
    std::cout << "INFO : Test started" << std::endl;

    CSCMatrix<VAL_T> csc_matrix = csc_matrix_convert_from_float<VAL_T>(csc_matrix_float);
    // std::vector<aligned_packet_t> channel_packets_uncast(SPMSPV_NUM_HBM_CHANNEL);
    std::vector<aligned_vector<SPMSPV_MAT_PKT_T>> channel_packets(SPMSPV_NUM_HBM_CHANNEL);

    std::vector<CSCMatrix<VAL_T>> csc_matrices = ColumnCyclicSplitCSC<VAL_T>(csc_matrix, SPMSPV_NUM_HBM_CHANNEL);
    FormattedCSCMatrix<SPMSPV_MAT_PKT_T> formatted_csc_matrices[SPMSPV_NUM_HBM_CHANNEL];
    for (uint32_t c = 0; c < SPMSPV_NUM_HBM_CHANNEL; c++) {
        formatted_csc_matrices[c] = formatCSC<VAL_T, SPMSPV_MAT_PKT_T>(csc_matrices[c],
                                                               PACK_SIZE,
                                                               SPMSPV_OUT_BUF_LEN);
        channel_packets[c] = formatted_csc_matrices[c].get_fused_matrix<VAL_T>();
    }
    uint32_t num_row_partitions = formatted_csc_matrices[0].num_row_partitions;

    std::cout << "INFO : Matrix loading/preprocessing complete!" << std::endl;

    //--------------------------------------------------------------------
    // generate input vector
    //--------------------------------------------------------------------
    unsigned vector_length = csc_matrix.num_cols;
    unsigned vector_nnz_cnt = (unsigned)floor(vector_length * (1 - vector_sparsity));
    unsigned vector_indices_increment = vector_length / vector_nnz_cnt;

    std::vector<IDX_FLOAT_T> vector_float(vector_nnz_cnt);
    for (size_t i = 0; i < vector_nnz_cnt; i++) {
        vector_float[i].val = (float)(rand() % 10) / 10;
        vector_float[i].index = i * vector_indices_increment;
    }
    IDX_FLOAT_T vector_head = {.index = vector_nnz_cnt, .val = 0};
    vector_float.insert(vector_float.begin(), vector_head);

    aligned_vector<IDX_VAL_T> vector(vector_float.size());
    for (size_t i = 0; i < vector_nnz_cnt + 1; i++) {
        vector[i].index = vector_float[i].index;
        vector[i].val = vector_float[i].val;
    }

    //--------------------------------------------------------------------
    // allocate space for results
    //--------------------------------------------------------------------
    aligned_vector<IDX_VAL_T> result(csc_matrix.num_rows + 1);
    std::fill(result.begin(), result.end(), (IDX_VAL_T){0, 0});
    std::cout << "INFO : Input/result initialization complete!" << std::endl;

    //--------------------------------------------------------------------
    // invoke kernel
    //--------------------------------------------------------------------
    // set kernel arguments that won't change across row iterations
    std::cout << "INFO : Invoking kernel:";
    std::cout << "  row_partitions: " << num_row_partitions << std::endl;

    unsigned num_parts = (csc_matrix.num_rows + SPMSPV_OUT_BUF_LEN - 1) / SPMSPV_OUT_BUF_LEN;

    double kernel_time_taken_ns
            = tapa::invoke(spmspv, bitstream,
                        #if (SPMSPV_NUM_HBM_CHANNEL >= 1)
                            tapa::read_only_mmap<SPMSPV_MAT_PKT_T>(channel_packets[0]),  // in,  HBM[0]
                        #endif
                        #if (SPMSPV_NUM_HBM_CHANNEL >= 2)
                            tapa::read_only_mmap<SPMSPV_MAT_PKT_T>(channel_packets[1]),  // in,  HBM[1]
                        #endif
                        #if (SPMSPV_NUM_HBM_CHANNEL >= 4)
                            tapa::read_only_mmap<SPMSPV_MAT_PKT_T>(channel_packets[2]),  // in,  HBM[2]
                            tapa::read_only_mmap<SPMSPV_MAT_PKT_T>(channel_packets[3]),  // in,  HBM[3]
                        #endif
                        #if (SPMSPV_NUM_HBM_CHANNEL >= 6)
                            tapa::read_only_mmap<SPMSPV_MAT_PKT_T>(channel_packets[4]),  // in,  HBM[4]
                            tapa::read_only_mmap<SPMSPV_MAT_PKT_T>(channel_packets[5]),  // in,  HBM[5]
                        #endif
                        #if (SPMSPV_NUM_HBM_CHANNEL >= 8)
                            tapa::read_only_mmap<SPMSPV_MAT_PKT_T>(channel_packets[6]),  // in,  HBM[6]
                            tapa::read_only_mmap<SPMSPV_MAT_PKT_T>(channel_packets[7]),  // in,  HBM[7]
                        #endif
                        #if (SPMSPV_NUM_HBM_CHANNEL >= 10)
                            tapa::read_only_mmap<SPMSPV_MAT_PKT_T>(channel_packets[8]),  // in,  HBM[8]
                            tapa::read_only_mmap<SPMSPV_MAT_PKT_T>(channel_packets[9]),  // in,  HBM[9]
                        #endif
                            tapa::read_only_mmap<IDX_VAL_T>(vector),                     // in
                            tapa::write_only_mmap<IDX_VAL_T>(result),                    // out
                            (unsigned)csc_matrix.num_rows,                               // in
                            (unsigned)num_parts,                                         // in
                            (unsigned)csc_matrix.num_cols,                               // in
                            (unsigned)vector_nnz_cnt                                     // in
            );

    std::cout << "INFO : SpMSpV Kernel Time is " << kernel_time_taken_ns * 1e-6 << "ms" << std::endl;
    std::cout << "INFO : SpMSpV Kernel complete!"<< std::endl;

    //--------------------------------------------------------------------
    // compute reference
    //--------------------------------------------------------------------
    uint32_t involved_Nnz = 0;
    std::vector<float> ref_result;
    compute_ref(csc_matrix_float, vector_float, ref_result, involved_Nnz);
    std::cout << "INFO : Compute reference complete!" << std::endl;

    //--------------------------------------------------------------------
    // verify
    //--------------------------------------------------------------------
    std::cout << "INFO : Device -> Host data transfer complete!" << std::endl;
    std::cout << "INFO : Involved Nnz during calculation: " << involved_Nnz << std::endl;
    std::cout << "INFO : Result Nnz: " << result[0].index << std::endl;

    std::vector<VAL_T> upk_result;
    convert_sparse_vec_to_dense_vec(result, upk_result, csc_matrix.num_rows);
    return verify(ref_result, upk_result);
}

//---------------------------------------------------------------
// test case utils
//---------------------------------------------------------------
CSRMatrix<float> create_dense_CSR (
    unsigned num_rows,
    unsigned num_cols
) {
    CSRMatrix<float> mat_f;
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

CSRMatrix<float> create_uniform_sparse_CSR (
    unsigned num_rows,
    unsigned num_cols,
    unsigned nnz_per_row
) {
    CSRMatrix<float> mat_f;
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

bool test_dense32(std::string bitstream) {
    std::cout << "------ Running test: on basic dense matrix " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_dense_CSR(32, 32));
    for (auto &x : mat_f.adj_data) x = 1.0;
    if (spmspv_test_harness(bitstream, mat_f, 0.0)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_basic(std::string bitstream) {
    std::cout << "------ Running test: on basic dense matrix " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_dense_CSR(128, 128));
    for (auto &x : mat_f.adj_data) x = 1.0;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_basic_sparse(std::string bitstream) {
    std::cout << "------ Running test: on basic sparse matrix " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_uniform_sparse_CSR(1000, 1024, 10));
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_medium_sparse(std::string bitstream) {
    std::cout << "------ Running test: on uniform 10K 10 (100K, 1M) " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_uniform_sparse_CSR(10000, 10000, 10));
    for (auto &x : mat_f.adj_data) x = 1.0;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_gplus(std::string bitstream) {
    std::cout << "------ Running test: on google_plus (108K, 13M) " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(GRAPH_DATASET_DIR + "gplus_108K_13M_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_ogbl_ppa(std::string bitstream) {
    std::cout << "------ Running test: on ogbl_ppa (576K, 42M) " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(GRAPH_DATASET_DIR + "ogbl_ppa_576K_42M_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_transformer_50_t(std::string bitstream) {
    std::cout << "------ Running test: on transformer-50-t" << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(NN_DATASET_DIR + "transformer_50_512_33288_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_transformer_95_t(std::string bitstream) {
    std::cout << "------ Running test: on transformer-95-t" << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(NN_DATASET_DIR + "transformer_95_512_33288_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
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
    passed = passed && test_dense32(FLAGS_bitstream);
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
