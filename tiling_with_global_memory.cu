#include <stdio.h>
// #define N 1024
// #define P 512
// #define M 256
/*
#define N 512
#define P 256
#define M 128
*/
#define IDX(i,j,row,col) (i*col+j)
#define Tx 1
#define Ty 64 //max value for Tx * Ty is 1024

__global__ void mmGlobal(float *A, float *B, float *C, int M, int K, int N);

int main(int argc, char *argv[]){
    int M = std::atoi(argv[1]), K = std::atoi(argv[2]), N = std::atoi(argv[3]);
    printf("M=%d K=%d N=%d\n",M,K,N);
    float ms;
    float error = 0.0f;
    float temp;

    float *A = (float*)malloc(M*K*sizeof(float));
    float *B = (float*)malloc(K*N*sizeof(float));
    float *C = (float*)malloc(M*N*sizeof(float));
    
    float *dA;
    float *dB;
    float *dC;
    
    int i,j,k;

    for(i=0;i<M;i++){
        for(j=0;j<K;j++){
            A[IDX(i,j,M,K)]=1.0f;
        }
    }
    for(i=0;i<K;i++){
        for(j=0;j<N;j++){
            B[IDX(i,j,K,N)]=2.0f;
        }
    }
    cudaMalloc((void**)&dA,sizeof(float)*M*K);
    cudaMalloc((void**)&dB,sizeof(float)*K*N);
    cudaMalloc((void**)&dC,sizeof(float)*N*M);

    cudaMemcpy(dA,A,sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(float)*K*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC,C,sizeof(float)*N*M, cudaMemcpyHostToDevice);

    dim3 threads(Tx,Ty);
    dim3 blocks( (M+threads.x-1)/threads.x, (N+threads.y-1)/threads.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    mmGlobal<<<blocks,threads>>>(dA,dB,dC,M,K,N);
    
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

    for(i=0;i<N;i++){
        for(j=0;j<M;j++){
            temp = 0;
            for(k=0;k<K;k++){
                temp += A[IDX(i,k,N,K)] * B[IDX(k,j,K,M)];
            }
            error += abs(C[IDX(i,j,M,N)]-temp);
        }
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A);
    free(B);
    free(C);

    printf("%f\n",error);
    printf("%f\n",ms);
    return 0;
}
__global__ void mmGlobal(float *A, float *B, float *C, int M, int K, int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    float temp = 0;
    if(i < M and j < N){
        for(int k = 0; k < K; ++k){
          temp += A[ IDX(i,k,M,K) ] * B[ IDX(k,j,K,N) ];
      }
      C[ IDX(i,j,M,N) ]=temp;
    }
}