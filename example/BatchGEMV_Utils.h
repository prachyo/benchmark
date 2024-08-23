#ifndef BATCH_GEMV_UTILS_H
#define BATCH_GEMV_UTILS_H

#include "SWTensorBench.h"
#include "Timer.h"
#include <immintrin.h>
#include <thread>
#include <vector>
#ifdef OPENMP_ENABLED
#include <omp.h>
#endif
#ifdef TBB_ENABLED
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif
#ifdef OPENBLAS_ENABLED
#include <cblas.h>
#endif

using namespace swiftware::benchmark;

void GEMV_Naive(int m, int n, double *A, double *x, double *y) {
    for (int i = 0; i < m; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[j * m + i] * x[j];
        }
        y[i] = sum;
    }
}

void GEMV_LoopUnrolling(int m, int n, double *A, double *x, double *y) {
    for (int i = 0; i < m; i++) {
        double sum = 0;
        int j = 0;
        for (; j <= n - 4; j += 4) {
            sum += A[j * m + i] * x[j] + A[(j + 1) * m + i] * x[j + 1] + A[(j + 2) * m + i] * x[j + 2] + A[(j + 3) * m + i] * x[j + 3];
        }
        for (; j < n; j++) {
            sum += A[j * m + i] * x[j];
        }
        y[i] = sum;
    }
}

void GEMV_SIMD(int m, int n, double *A, double *x, double *y) {
    for (int i = 0; i < m; i++) {
        __m256d sum = _mm256_setzero_pd();
        int j = 0;
        for (; j <= n - 4; j += 4) {
            __m256d a = _mm256_loadu_pd(&A[j * m + i]);
            __m256d b = _mm256_loadu_pd(&x[j]);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
        }
        double temp[4];
        _mm256_storeu_pd(temp, sum);
        y[i] = temp[0] + temp[1] + temp[2] + temp[3];
        for (; j < n; j++) {
            y[i] += A[j * m + i] * x[j];
        }
    }
}

void GEMV_CacheBlocking(int m, int n, double *A, double *x, double *y) {
    int blockSize = 64;
    for (int i = 0; i < m; i++) {
        y[i] = 0.0;
    }

    for (int jj = 0; jj < n; jj += blockSize) {
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = jj; j < std::min(jj + blockSize, n); j++) {
                sum += A[j * m + i] * x[j];
            }
            y[i] += sum;
        }
    }
}

void GEMV_ThreadWorker(int m, int n, double *A, double *x, double *y, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[j * m + i] * x[j];
        }
        y[i] = sum;
    }
}

void GEMV_MultiThreading(int m, int n, double *A, double *x, double *y) {
  int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int rowsPerThread = m / numThreads;

    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? m : startRow + rowsPerThread;
        threads.emplace_back(GEMV_ThreadWorker, m, n, A, x, y, startRow, endRow);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

#ifdef OPENMP_ENABLED
void GEMV_OpenMP(int m, int n, double *A, double *x, double *y) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        double sum = 0;
        for (int j = 0; j < n/4; j+=4) {
            sum += A[j * m + i] * x[j];
            sum += A[(j+1) * m + i] * x[j+1];
            sum += A[(j+2) * m + i] * x[j+2];
            sum += A[(j+3) * m + i] * x[j+3];
        }
        for(int j = n/4*4; j < n; j++)
            sum += A[j * m + i] * x[j];
        y[i] = sum;
    }
}
#endif

#ifdef TBB_ENABLED
void GEMV_TBB(int m, int n, double *A, double *x, double *y) {
    tbb::parallel_for(tbb::blocked_range<int>(0, m), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i != r.end(); ++i) {
            double sum = 0;
            for (int j = 0; j < n/4; j+=4) {
                sum += A[j * m + i] * x[j];
                sum += A[(j+1) * m + i] * x[j+1];
                sum += A[(j+2) * m + i] * x[j+2];
                sum += A[(j+3) * m + i] * x[j+3];
            }
            for(int j = n/4*4; j < n; j++)
                sum += A[j * m + i] * x[j];
            y[i] = sum;
        }
    });
}
#endif

#ifdef CUDA_ENABLED
__global__ void GEMV_CUDA(int m, int n, double *A, double *x, double *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        double sum = 0;
        for (int j = 0; j < n/4; j+=4) {
            sum += A[j * m + i] * x[j];
            sum += A[(j+1) * m + i] * x[j+1];
            sum += A[(j+2) * m + i] * x[j+2];
            sum += A[(j+3) * m + i] * x[j+3];
        }
        for(int j = n/4*4; j < n; j++)
            sum += A[j * m + i] * x[j];
        y[i] = sum;
    }
}

void GEMV_CUDA_Launch(int m, int n, double *A, double *x, double *y) {
    double *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, m * n * sizeof(double));
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_y, m * sizeof(double));
    
    cudaMemcpy(d_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (m + blockSize - 1) / blockSize;
    GEMV_CUDA<<<numBlocks, blockSize>>>(m, n, d_A, d_x, d_y);
    
    cudaMemcpy(y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}
#endif

#ifdef OPENBLAS_ENABLED
void GEMV_BLAS(int m, int n, double *A, double *x, double *y) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A, n, x, 1, 0.0, y, 1);
}
#endif

class GEMVBench_Naive : public SWTensorBench<double> {
protected:
  void setup() override {
    Out = new Outputs<double>();
    Out->Out = new double[In->Dim1]();
  }

  void preExecute() override {
    for (int i = 0; i < In->Dim1; ++i) {
      In->CorrectSol[i] = 0;
      Out->Out[i] = 0;
    }
  }

  Timer execute() override {
    Timer t;
    t.start();
    GEMV_Naive(In->Dim1, In->Dim2, In->A, In->x, In->y);
    t.stop();
    return t;
  }

public:
  GEMVBench_Naive(Inputs<double> *In1, Stats *Stat1) : SWTensorBench<double>(In1, Stat1)
  {}

  ~GEMVBench_Naive(){
  }
};

class GEMVBench_LoopUnrolling : public GEMVBench_Naive {
protected:
  Timer execute() override {
    Timer t;
    t.start();
    GEMV_LoopUnrolling(In->Dim1, In->Dim2, In->A, In->x, In->y);
    t.stop();
    return t;
  }

public:
  GEMVBench_LoopUnrolling(Inputs<double> *In1, Stats *Stat1) : GEMVBench_Naive(In1, Stat1)
  {}

  ~GEMVBench_LoopUnrolling(){
  }
};

class GEMVBench_SIMD : public GEMVBench_Naive {
protected:
  Timer execute() override {
    Timer t;
    t.start();
    GEMV_SIMD(In->Dim1, In->Dim2, In->A, In->x, In->y);
    t.stop();
    return t;
  }

public:
  GEMVBench_SIMD(Inputs<double> *In1, Stats *Stat1) : GEMVBench_Naive(In1, Stat1)
  {}

  ~GEMVBench_SIMD(){
  }
};

class GEMVBench_CacheBlocking : public GEMVBench_Naive {
protected:
  Timer execute() override {
    Timer t;
    t.start();
    GEMV_CacheBlocking(In->Dim1, In->Dim2, In->A, In->x, In->y);
    t.stop();
    return t;
  }

public:
  GEMVBench_CacheBlocking(Inputs<double> *In1, Stats *Stat1) : GEMVBench_Naive(In1, Stat1)
  {}

  ~GEMVBench_CacheBlocking(){
  }
};

class GEMVBench_MultiThreading: public GEMVBench_Naive {
protected:
  Timer execute() override {
    Timer t;
    t.start();
    GEMV_MultiThreading(In->Dim1, In->Dim2, In->A, In->x, In->y);
    t.stop();
    return t;
  }

public:
  GEMVBench_MultiThreading(Inputs<double> *In1, Stats *Stat1) : GEMVBench_Naive(In1, Stat1)
  {}

  ~GEMVBench_MultiThreading(){
  }
};

#ifdef OPENMP_ENABLED
class GEMVBench_OpenMP : public GEMVBench_Naive {
protected:
  Timer execute() override {
    Timer t;
    t.start();
    GEMV_OpenMP(In->Dim1, In->Dim2, In->A, In->x, In->y);
    t.stop();
    return t;
  }

public:
  GEMVBench_OpenMP(Inputs<double> *In1, Stats *Stat1) : GEMVBench_Naive(In1, Stat1)
  {}

  ~GEMVBench_OpenMP(){
  }
};
#endif

#ifdef TBB_ENABLED
class GEMVBench_TBB : public GEMVBench_Naive {
protected:
  Timer execute() override {
    Timer t;
    t.start();
    GEMV_TBB(In->Dim1, In->Dim2, In->A, In->x, In->y);
    t.stop();
    return t;
  }

public:
  GEMVBench_TBB(Inputs<double> *In1, Stats *Stat1) : GEMVBench_Naive(In1, Stat1)
  {}

  ~GEMVBench_TBB(){
  }
};
#endif

#ifdef CUDA_ENABLED
class GEMVBench_CUDA: public GEMVBench_Naive{
protected:
  Timer execute() override {
    Timer t;
    t.start();
    GEMV_CUDA_Launch(In->Dim1, In->Dim2, In->A, In->x, In->y);
    t.stop();
    return t;
  }

public:
  GEMVBench_CUDA(Inputs<double> *In1, Stats *Stat1) : GEMVBench_Naive(In1, Stat1)
  {}

  ~GEMVBench_CUDA(){
  }
};
#endif

#ifdef OPENBLAS_ENABLED
class GEMVBench_BLAS: public GEMVBench_Naive {
protected:
  Timer execute() override {
    Timer t;
    t.start();
    GEMV_BLAS(In->Dim1, In->Dim2, In->A, In->x, In->y);
    t.stop();
    return t;
  }

public:
  GEMVBench_BLAS(Inputs<double> *In1, Stats *Stat1) : GEMVBench_Naive(In1, Stat1)
  {}

  ~GEMVBench_BLAS(){
  }
};
#endif

#endif //BATCH_GEMV_UTILS_H