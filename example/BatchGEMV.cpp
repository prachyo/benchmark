#include "BatchGEMV_Utils.h"

int main(int argc, char *argv[]) {

    auto *in = new Inputs<double>(1000, 1000, 0, 0);
    in->ExpName = "GEMV_With_PAPI";
    in->CorrectSol = new double[in->Dim1];

    // generate a random matrix
    in->A = new double[in->Dim1 * in->Dim2];
    for (int i = 0; i < in->Dim1 * in->Dim2; ++i) {
        in->A[i] = (double) rand() / RAND_MAX;
    }

    // generate a random vector
    in->x = new double[in->Dim2];
    for (int i = 0; i < in->Dim2; ++i) {
        in->x[i] = (double) rand() / RAND_MAX;
    }

    in->y = new double[in->Dim1];
    in->NumTrials = 2;

    auto *st = new Stats( "GEMV_Batch", "MV", in->NumTrials);
    for (int i = 0; i < in->Dim1; ++i) {
        in->CorrectSol[i] = 0;
    }

    in->Threshold = 1e-6;
    auto *gemv_naive = new GEMVBench_Naive(in, st);
    gemv_naive->run();
    auto headerStat = gemv_naive->printStatsHeader();
    auto gemv_naive_stat = gemv_naive->printStats();
    delete gemv_naive;

    std::map<std::string, std::string> stats;

    #ifdef OPENMP_ENABLED
    auto *gemv_openmp = new GEMVBench_OpenMP(in, st);
    gemv_openmp->run();
    stats["GEMV_OpenMP"] = gemv_openmp->printStats();
    delete gemv_openmp;
    #endif

    #ifdef TBB_ENABLED
    auto *gemv_tbb = new GEMVBench_TBB(in, st);
    gemv_tbb->run();
    stats["GEMV_TBB"] = gemv_tbb->printStats();
    delete gemv_tbb;
    #endif

    #ifdef CUDA_ENABLED
    auto *gemv_cuda = new GEMVBench_CUDA(in, st);
    gemv_cuda->run();
    stats["GEMV_CUDA"] = gemv_cuda->printStats();
    delete gemv_cuda;
    #endif

    #ifdef OPENBLAS_ENABLED
    auto *gemv_blas = new GEMVBench_BLAS(in, st);
    gemv_blas->run();
    stats["GEMV_BLAS"] = gemv_blas->printStats();
    delete gemv_blas;
    #endif

    auto inHeader = in->printCSVHeader();
    auto inStat = in->printCSV();
    delete in;
    delete st;

    /// Exporting the results
    std::cout<<headerStat<<inHeader<<std::endl;
    std::cout<<gemv_naive_stat<<inStat<<std::endl;
    for (const auto& stat : stats) {
        std::cout << stat.second << inStat << std::endl;
    }

  return 0;
}