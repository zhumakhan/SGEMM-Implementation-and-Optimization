/*
    by zhumakhan.nazir@nu.edu.kz
*/

#include "utils.cpp"
#include "test_cublas.cu"
#include "tiling_with_shared_memory.cu"
#include "computation_optimization.cu"
#include "loop_unrolling.cu"
#include "prefetching.cu"

#include <stdio.h>
// #include <unistd.h>


#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
#ifndef VECTOR_SIZE
#define VECTOR_SIZE 4
#endif

/*
    A is a row major matrix ( M x K )
    B is a row major matrix ( K x N )

*/

int main(int argc, char *argv[]){
    const int M = std::atoi(argv[1]);
    const int K = std::atoi(argv[2]);
    const int N = std::atoi(argv[3]);

    printf("M=%d K=%d N=%d\n",M,K,N);

    
    test_cublas( M, K, N );
    

    float *A = utils::random_matrix_gpu<float>(M, K, utils::ROW_MAJOR,-50,50);
    float *B = utils::random_matrix_gpu<float>(K, N, utils::ROW_MAJOR,-50,50);
    float *C = (float*)malloc(sizeof(float)*M*N);
    
    float *dA, *dB, *dC;


    cudaMalloc((void**)&dA,sizeof(float)*M*K);
    cudaMalloc((void**)&dB,sizeof(float)*K*N);
    cudaMalloc((void**)&dC,sizeof(float)*N*M);

    cudaMemcpy(dA,A,sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(float)*K*N, cudaMemcpyHostToDevice);

    const dim3 threads1( TILE_SIZE, TILE_SIZE );
    const dim3 blocks1(N / TILE_SIZE, M / TILE_SIZE);

    const dim3 threads2( TILE_SIZE, VECTOR_SIZE );
    const dim3 blocks2(N / (TILE_SIZE * VECTOR_SIZE), M / TILE_SIZE);
    

    // mmSharedRR <<< blocks1, threads1 >>> ( dA, dB, dC, M, K, N );

    // mmCompOpt_v1 <<< blocks2, threads2 >>> ( dA, dB, dC, M, K, N );
// 
    // mmLoopUnrolling <<< blocks2, threads2 >>> ( dA, dB, dC, M, K, N );

    mmPrefetching <<< blocks2, threads2 >>> ( dA, dB, dC, M, K, N );
        
        
    cudaError_t cuda_error = cudaGetLastError();
    if(cuda_error != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
      exit(-1);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A);
    free(B);
    free(C);

    return 0;
}
