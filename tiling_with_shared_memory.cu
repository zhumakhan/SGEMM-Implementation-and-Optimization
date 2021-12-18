#include <stdio.h>

#define N 512
#define P 256
#define M 128

#define N 1024
#define P 512
#define M 256

#define BS 16 //keep BS*BS <= 1024

__global__ void mmShared(float A[N][P], float B[P][M], float C[N][M]);

int main(){
    
    float ms;
    float error = 0.0f;
    float temp;

    float A[N][P];
    float B[P][M];
    float C[N][M];
    
    float (*dA)[P];
    float (*dB)[M];
    float (*dC)[M];
    
    int i,j,k;

    for(i=0;i<N;i++){
        for(j=0;j<P;j++){
            A[i][j]=i-j;
        }
    }
    for(i=0;i<P;i++){
        for(j=0;j<M;j++){
            B[i][j]=i+j;
        }
    }
    cudaMalloc(&dA,sizeof(float)*N*P);
    cudaMalloc(&dB,sizeof(float)*P*M);
    cudaMalloc(&dC,sizeof(float)*N*M);

    cudaMemcpy(dA,A,sizeof(float)*N*P, cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(float)*P*M, cudaMemcpyHostToDevice);
    cudaMemcpy(dC,C,sizeof(float)*N*M, cudaMemcpyHostToDevice);

    dim3 threads(BS,BS);
    dim3 blocks( (N+threads.x-1)/threads.x, (M+threads.y-1)/threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    mmShared<<<blocks,threads>>>(dA,dB,dC);

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

    cudaMemcpy(C, dC, sizeof(float)*N*M, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

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
    printf("%f\n",(float)(sizeof(float)*(N*M*P/BS * 2 + N*M))/1.0e6/ms);
    return 0;
}

__global__ void mmShared(float A[N][P], float B[P][M], float C[N][M]){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if( i >= N or j >= M){
        return;
    }

    int ii = threadIdx.x;
    int jj = threadIdx.y;
    
  
    __shared__ float sA[BS][BS], sB[BS][BS];

    float temp = 0;
    int k,m;

    for(k = 0; k < P/BS; k++){
        sA[ii][jj] = A[i][k*BS + jj];
        sB[ii][jj] = B[k*BS + ii][j];
        
        __syncthreads();

         for(m=0; m < BS; m++){
            temp += sA[ii][m] * sB[m][jj];
        }
        __syncthreads();
    }

    C[i][j] = temp;
}