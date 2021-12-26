/*
    by zhumakhan.nazir@nu.edu.kz
*/

#include "utils.cpp"
#include <stdio.h>

#define TILE_SIZE 16

__global__ void mmSharedRR(float *, float *, float *, int, int, int);
__global__ void mmSharedRC(float *, float *, float *, int, int, int);
__global__ void mmSharedCR(float *, float *, float *, int, int, int);
__global__ void mmSharedCC(float *, float *, float *, int, int, int);

void test(void (*kernel)(float *, float *, float *, int, int, int),
    dim3 &threads, dim3 &blocks, int M, int K, int N, 
    int a_major, int b_major, int c_major);


int main(int argc, char *argv[]){
    int M = std::atoi(argv[1]);
    int K = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);

    printf("M=%d K=%d N=%d\n",M,K,N);
    

    dim3 threads(   TILE_SIZE, TILE_SIZE    );
    dim3 blocks(    ( N + threads.x - 1 ) / threads.x, ( M + threads.y - 1 ) / threads.y   );

    std::cout << "Row Row" << std::endl;
    test(&mmSharedRR, threads, blocks, M, K, N, utils::ROW_MAJOR, utils::ROW_MAJOR, utils::ROW_MAJOR);
    std::cout << "Row Column" << std::endl;
    test(&mmSharedRC, threads, blocks, M, K, N, utils::ROW_MAJOR, utils::COLUMN_MAJOR, utils::ROW_MAJOR);
    std::cout << "Column Row" << std::endl;
    test(&mmSharedCR, threads, blocks, M, K, N, utils::COLUMN_MAJOR, utils::ROW_MAJOR, utils::ROW_MAJOR);
    std::cout << "Column Column" << std::endl;
    test(&mmSharedCC, threads, blocks, M, K, N, utils::COLUMN_MAJOR, utils::COLUMN_MAJOR, utils::ROW_MAJOR);

    return 0;
}

__global__ void mmSharedRR(float *A, float *B, float *C, int M, int K, int N){
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    volatile __shared__ float As[ TILE_SIZE ][ TILE_SIZE ];
    volatile __shared__ float Bs[ TILE_SIZE ][ TILE_SIZE ];

    int aBegin  = K * TILE_SIZE * by;
    int aEnd    = aBegin + K;
    int aStep   = TILE_SIZE;

    int bBegin  = TILE_SIZE * bx;
    int bStep   = TILE_SIZE * N;

    float Csub = 0;

    for (int i = aBegin, j = bBegin; i < aEnd; i += aStep, j += bStep) {
        As[ ty ][ tx ] = A[ i + K * ty + tx ];
        Bs[ tx ][ ty ] = B[ j + N * tx + ty ];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            Csub += As[ ty ][ k ] * Bs[ k ][ tx ];
        }
        
        __syncthreads();
    }
    int cIdx = N * TILE_SIZE * by + TILE_SIZE * bx;
    C[ cIdx + N * ty + tx ] = Csub;
}

__global__ void mmSharedRC(float *A, float *B, float *C, int M, int K, int N){
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    volatile __shared__ float As[TILE_SIZE][TILE_SIZE];
    volatile __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int aBegin  = K * TILE_SIZE * by;
    int aEnd    = aBegin + K;
    int aStep   = TILE_SIZE;

    int bBegin  = K * TILE_SIZE * bx;
    int bStep   = TILE_SIZE;

    float Csub  = 0;

    for (int i = aBegin, j = bBegin; i < aEnd; i += aStep, j += bStep) {
        As[ty][tx] = A[i + K * ty + tx];
        Bs[tx][ty] = B[j + K * ty + tx];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            Csub += As[ty][k]*Bs[k][tx];
        }
        
        __syncthreads();
    }
    int cIdx = N * TILE_SIZE * by + TILE_SIZE * bx;
    C[cIdx + N * ty + tx] = Csub;
}

__global__ void mmSharedCR(float *A, float *B, float *C, int M, int K, int N){
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    volatile __shared__ float As[TILE_SIZE][TILE_SIZE];
    volatile __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int aBegin  = TILE_SIZE * by;
    int aEnd    = aBegin + K*M;
    int aStep   = TILE_SIZE * M;

    int bBegin  = TILE_SIZE * bx;
    int bStep   = TILE_SIZE * N;

    float Csub = 0;

    for (int i = aBegin, j = bBegin; i < aEnd; i += aStep, j += bStep) {
        As[ty][tx] = A[i + M * tx + ty];
        Bs[tx][ty] = B[j + N * tx + ty];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            Csub += As[ty][k]*Bs[k][tx];
        }
        
        __syncthreads();
    }
    int cIdx = N * TILE_SIZE * by + TILE_SIZE * bx;
    C[cIdx + N * ty + tx] = Csub;
}

__global__ void mmSharedCC(float *A, float *B, float *C, int M, int K, int N){
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    volatile __shared__ float As[TILE_SIZE][TILE_SIZE];
    volatile __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int aBegin  = TILE_SIZE * by;
    int aEnd    = aBegin + K*M;
    int aStep   = TILE_SIZE * M;

    int bBegin  = K * TILE_SIZE * bx;
    int bStep   = TILE_SIZE;

    float Csub  = 0;

    for (int i = aBegin, j = bBegin; i < aEnd; i += aStep, j += bStep) {
        As[ty][tx] = A[i + M * tx + ty];
        Bs[tx][ty] = B[j + K * ty + tx];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            Csub += As[ty][k]*Bs[k][tx];
        }
        
        __syncthreads();
    }
    int cIdx = N * TILE_SIZE * by + TILE_SIZE * bx;
    C[cIdx + N * ty + tx] = Csub;
}

void test(void (*kernel)(float *, float *, float *, int, int, int),
    dim3 &threads, dim3 &blocks, int M, int K, int N, 
    int a_major, int b_major, int c_major){
    
    float *A = utils::random_matrix_gpu<float>(M, K, a_major,-1,1);
    float *B = utils::random_matrix_gpu<float>(K, N, b_major,-1,1);
    float *C = (float*)malloc(sizeof(float)*M*N);
    
    float *dA, *dB, *dC;

    cudaMalloc((void**)&dA,sizeof(float)*M*K);
    cudaMalloc((void**)&dB,sizeof(float)*K*N);
    cudaMalloc((void**)&dC,sizeof(float)*M*N);

    cudaMemcpy(dA,A,sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(float)*K*N, cudaMemcpyHostToDevice);

    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    (*kernel)<<<blocks,threads>>>(dA,dB,dC,M,K,N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t cuda_error = cudaGetLastError();
    if(cuda_error != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
    }

    cudaMemcpy(C, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

#ifdef CHECK
    std::cout << (utils::check_mul<float>(A, B, C, M, K, N, a_major, b_major, c_major) 
            ? "Correct!!" : "Wrong Answer!") << std::endl;
#endif
#ifdef DEBUG
    std::cout << "Matrix A:" << std::endl;
    utils::print_mat_gpu(A, M, K, a_major);
    std::cout << "Matrix B:" << std::endl;
    utils::print_mat_gpu(B, K, N, b_major);
    std::cout << "Matrix C:" << std::endl;
    utils::print_mat_gpu(C, M, N, c_major);
#endif

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A);
    free(B);
    free(C);

    printf("%f ms\n\n", ms);

}



