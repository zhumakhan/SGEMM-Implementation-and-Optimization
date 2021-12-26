/*
    by zhumakhan.nazir@nu.edu.kz
*/

#include "utils.cpp"
#include "tiling_with_shared_memory.cu"
#include "computation_optimization.cu"
#include "loop_unrollung.cu"
#include "prefetching.cu"

#include <stdio.h>

#define TILE_SIZE 16
#define VECTOR_SIZE 4

/*
    A is a row major matrix ( M x K )
    B is a row major matrix ( K x N )

*/

typedef void (*FunctionPointer_t)(float *, float *, float *, const int, const int, const int);


int main(int argc, char *argv[]){
    int M = std::atoi(argv[1]);
    int K = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);

    printf("M=%d K=%d N=%d\n",M,K,N);

    float *A = utils::random_matrix_gpu<float>(M, K, utils::ROW_MAJOR,-50,50);
    float *B = utils::random_matrix_gpu<float>(K, N, utils::ROW_MAJOR,-50,50);
    float *C = (float*)malloc(sizeof(float)*M*N);
    
    float ms;
    float *dA, *dB, *dC;

    cudaEvent_t start, stop;

    cudaMalloc((void**)&dA,sizeof(float)*M*K);
    cudaMalloc((void**)&dB,sizeof(float)*K*N);
    cudaMalloc((void**)&dC,sizeof(float)*N*M);

    cudaMemcpy(dA,A,sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(float)*K*N, cudaMemcpyHostToDevice);

    dim3 threads1( TILE_SIZE, TILE_SIZE );
    dim3 blocks1(N / TILE_SIZE, M / TILE_SIZE);

    dim3 threads2( TILE_SIZE, VECTOR_SIZE );
    dim3 blocks2(N / (TILE_SIZE * VECTOR_SIZE), M / TILE_SIZE);
    
    FunctionPointer_t kernels [ 4 ] = {
        &mmSharedRR,
        &mmCompOpt_v1,
        &mmLoopUnrolling,
        &mmPrefetching
    };

    for(int i = 0; i < 4; i++){

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        // (*kernels[i])<<<blocks,threads>>>(dA,dB,dC,M,K,N);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        cudaError_t cuda_error = cudaGetLastError();
        if(cuda_error != cudaSuccess)
        {
          printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
          exit(-1);
        }

        cudaMemcpy(C,dC,sizeof(float)*N*M, cudaMemcpyDeviceToHost);

        #ifdef CHECK
        std::cout << (utils::check_mul<float>(A, B, C, M, K, N, utils::ROW_MAJOR, utils::ROW_MAJOR, utils::ROW_MAJOR) 
                ? "Correct!!" : "Wrong Answer!") << std::endl;
        #endif

        printf("%f\n",ms);
    }


#ifdef DEBUG
    std::cout << "Matrix A:" << std::endl;
    utils::print_mat_gpu(a, M, K, utils::ROW_MAJOR);
    std::cout << "Matrix B:" << std::endl;
    utils::print_mat_gpu(b, K, N, utils::ROW_MAJOR);
    std::cout << "Matrix C:" << std::endl;
    utils::print_mat_gpu(c, M, N, utils::ROW_MAJOR);
#endif

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A);
    free(B);
    free(C);

    return 0;
}
