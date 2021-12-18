#include <stdio.h>
#define N 1024
#define P 512
#define M 256
/*
#define N 512
#define P 256
#define M 128
*/
#define Tx 1
#define Ty 64 //max value for Tx * Ty is 1024

__global__ void mmGlobal(float A[N][P], float B[P][M], float C[N][M]);

int main(){
    float A[N][P];
    float B[P][M];
    float C[N][M];
    
    float (*dA)[P];
    float (*dB)[M];
    float (*dC)[M];
    
    int i,j,k;

    for(i=0;i<N;i++){
        for(j=0;j<P;j++){
            A[i][j]=1.0f;
        }
    }
    for(i=0;i<P;i++){
        for(j=0;j<M;j++){
            B[i][j]=2.0f;
        }
    }
    cudaMalloc(&dA,sizeof(float)*N*P);
    cudaMalloc(&dB,sizeof(float)*P*M);
    cudaMalloc(&dC,sizeof(float)*N*M);

    cudaMemcpy(dA,A,sizeof(float)*N*P, cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(float)*P*M, cudaMemcpyHostToDevice);
    cudaMemcpy(dC,C,sizeof(float)*N*M, cudaMemcpyHostToDevice);

    dim3 threads(Tx,Ty);
    dim3 blocks( (N+threads.x-1)/threads.x, (M+threads.y-1)/threads.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    mmGlobal<<<blocks,threads>>>(dA,dB,dC);
    cudaDeviceSynchronize();
    
    cudaError_t cuda_error = cudaGetLastError();
    if(cuda_error != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
      exit(-1);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C,dC,sizeof(float)*N*M, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    float error = 0.0f;
    float temp;

    for(i=0;i<N;i++){
        for(j=0;j<M;j++){
            temp = 0;
            for(k=0;k<P;k++){
                temp += A[i][k] * B[k][j];
            }
            error += abs(C[i][j]-temp);
        }
    }
    printf("%f\n",error);
    printf("%f\n",ms);
    printf("%f\n",(float)(sizeof(float)*(N*M + N*M*P*2))/1.0e6/ms);
    return 0;
}
__global__ void mmGlobal(float A[N][P], float B[P][M], float C[N][M]){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    float temp = 0;
    if(i < N and j < M){
        for(int k=0;k<P;++k){
          temp += A[i][k]*B[k][j];
      }
      C[i][j]=temp;
    }
}