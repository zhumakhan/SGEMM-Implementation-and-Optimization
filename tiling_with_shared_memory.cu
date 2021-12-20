#include "utils.cpp"
#include <stdio.h>


#define BS 16

__global__ void mmShared(float *A, float *B, float *C, int M, int K, int N);

int main(int argc, char *argv[]){
    int M = std::atoi(argv[1]), K = std::atoi(argv[2]), N = std::atoi(argv[3]);
    printf("M=%d K=%d N=%d\n",M,K,N);

    float *A = utils::random_matrix_gpu<float>(M, K, utils::ROW_MAJOR,-50,50);
    float *B = utils::random_matrix_gpu<float>(K, N, utils::ROW_MAJOR,-50,50);
    float *C = (float*)malloc(sizeof(float)*M*N);
    
    float ms;
    float *dA, *dB, *dC;

    cudaMalloc((void**)&dA,sizeof(float)*M*K);
    cudaMalloc((void**)&dB,sizeof(float)*K*N);
    cudaMalloc((void**)&dC,sizeof(float)*M*N);

    cudaMemcpy(dA,A,sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(float)*K*N, cudaMemcpyHostToDevice);

    dim3 threads(BS,BS);
    dim3 blocks( (M+threads.x-1)/threads.x, (N+threads.y-1)/threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    mmShared<<<blocks,threads>>>(dA,dB,dC,M,K,N);

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
    std::cout << (utils::check_mul<float>(A, B, C, M, K, N, utils::ROW_MAJOR, utils::ROW_MAJOR, utils::ROW_MAJOR) 
            ? "Correct!!" : "Wrong Answer!") << std::endl;
#endif
#ifdef DEBUG
    std::cout << "Matrix A:" << std::endl;
    utils::print_mat_gpu(A, M, K, utils::ROW_MAJOR);
    std::cout << "Matrix B:" << std::endl;
    utils::print_mat_gpu(B, K, N, utils::ROW_MAJOR);
    std::cout << "Matrix C:" << std::endl;
    utils::print_mat_gpu(C, M, N, utils::ROW_MAJOR);
#endif

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A);
    free(B);
    free(C);

    printf("%f\n",ms);
    return 0;
}

__global__ void mmShared(float *A, float *B, float *C, int M, int K, int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if( i >= M or j >= N){
        return;
    }

    int ii = threadIdx.x;
    int jj = threadIdx.y;

  
    __shared__ float sA[BS][BS], sB[BS][BS];

    float temp = 0;
    int k,m;

    for(k = 0; k < K; k += BS){
        sA[ii][jj] = A[ IDXR(i,k+jj, M, K) ];
        sB[ii][jj] = B[ IDXR(k+ii,j, K, N) ];
        
        __syncthreads();

         for(m = 0; m < BS and k+m < K; m += 1){
            temp += sA[ii][m] * sB[m][jj];
        }
        __syncthreads();
    }
    C[ IDXR(i,j,M,N) ] = temp;
}


