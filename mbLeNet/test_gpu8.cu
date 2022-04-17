#include "utils.cpp"
#include <stdio.h>
__device__ void update1(float *a, float b, float *c)
{
for (int i = 0; i < 16; i++)
c[i] += a[i] * b;
}

__device__ void update2(float *a, float b, float *c)
{
    for (int i = 0; i < 16; i++)
        c[i] += a[i * 4] * b;
}

__global__ void GPU8 (float *a, float *b, float *c, int n)
{// thread code to compute one column of a 16 x 128 sub-matrix of c
// use shared memory to hold the transpose of a
// 16 x 64 sub-matrix of 1 x 4 sub-vectors of a
__shared__ float as[16][65];
// registers for column of c sub-matrix
float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int nDiv64 = n/64;
int sRow = threadIdx.y;
int sRow4 = sRow*4;
int sCol = threadIdx.x;
int tid = sRow*16+sCol;
int aNext = (16*blockIdx.y+sRow)*n+sCol*4;
int bNext = 128*blockIdx.x + tid;
int cNext = 16*blockIdx.y*n + 128*blockIdx.x + tid;
int nTimes2 = 2*n;
int nTimes3 = 3*n;
int nTimes4 = 4*n;
a += aNext;
b += bNext;
c += cNext;
float4 *a4 = (float4 *)a;
for (int i = 0; i < nDiv64; i++)
{
*( (float4 *)(&as[sCol][sRow4]) ) = a4[0];
*( (float4 *)(&as[sCol][sRow4+32]) ) = a4[nTimes2];
__syncthreads(); // wait for read to complete
float br0 = b[0];
float br1 = b[n];
float br2 = b[nTimes2];
float br3 = b[nTimes3];
b += nTimes4;
#pragma unroll
for (int k = 0; k < 15; k++)
{
update2 (&as[k][0], br0, cr); br0 = b[0];
update2 (&as[k][1], br1, cr); br1 = b[n];
update2 (&as[k][2], br2, cr); br2 = b[nTimes2];
update2 (&as[k][3], br3, cr); br3 = b[nTimes3];
b+= nTimes4;
}
update2 (&as[15][0], br0, cr);
update2 (&as[15][1], br1, cr);
update2 (&as[15][2], br2, cr);
update2 (&as[15][3], br3, cr);
a4 += 16;
__syncthreads(); // wait for computation to complete
}
for (int j = 0; j < 16; j++)
{
c[0] = cr[j];
c += n;
}
}


__global__ void GPU6 (float *a, float *b, float *c, int n)
{// thread code to compute one column of a 16 x 128 sub-matrix of c
// use shared memory to hold the transpose of a 16 x 32 sub-matrix of a
__shared__ float as[32][17];
// registers for column of c sub-matrix
float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int nDiv32 = n/32;
int sRow = threadIdx.y;
int sCol = threadIdx.x;
int sCol2 = sCol*2;
int sCol2Plus1 = sCol2+1;
int tid = sRow*16+sCol;
int aNext = (16*blockIdx.y+sRow)*n+sCol*2;
int bNext = 128*blockIdx.x+tid;
int sRowPlus8 = sRow+8;
int nTimes8 = 8*n;
a += aNext;
b += bNext;
int i, j;
float2 temp;
for (i = 0; i< nDiv32; i++)
{// threads in a thread block collectively read a 16 x 32
// sub-matrix of a from device memory to shared memory
temp = *(float2 *)a;
as[sCol2][sRow] = temp.x;
as[sCol2Plus1][sRow] = temp.y;
temp = *(float2 *)(a+nTimes8);
as[sCol2][sRowPlus8] = temp.x;
as[sCol2Plus1][sRowPlus8] = temp.y;
__syncthreads(); // wait for read to complete
#pragma unroll
for (j = 0; j < 32; j++)
{
float br = b[0];
b += n;
update1 (&as[j][0], br, cr);
}
a += 32;
__syncthreads(); // wait for computation to complete
}

// output cr[]
int cNext = 16*blockIdx.y*n + 128*blockIdx.x + tid;
c += cNext;
for (int i = 0; i < 16; i++)
{
c[0] = cr[i];
c += n;
}
}


__global__ void GPU7 (float *a, float *b, float *c, int n)
{// thread code to compute one column of a 16 x 128 sub-matrix of c
// use shared memory to hold the transpose of a 16 x 32 sub-matrix of a
__shared__ float as[32][17];
// registers for column of c sub-matrix
float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int nDiv32 = n/32;
int sRow = threadIdx.y;
int sCol = threadIdx.x;
int sCol2 = sCol*2;
int sCol2Plus1 = sCol2+1;
int tid = sRow*16+sCol;
int aNext = (16*blockIdx.y+sRow)*n+sCol*2;
int bNext = 128*blockIdx.x+tid;
int sRowPlus8 = sRow+8;
int nTimes2 = 2*n;
int nTimes3 = 3*n;
int nTimes4 = 4*n;
int nTimes8 = 8*n;
a += aNext;
b += bNext;
int i, j;
float2 temp;
for (i = 0; i< nDiv32; i++)
{// threads in a thread block collectively read a 16 x 32
// sub-matrix of a from device memory to shared memory
temp = *(float2 *)a;
as[sCol2][sRow] = temp.x;
as[sCol2Plus1][sRow] = temp.y;
temp = *(float2 *)(a+nTimes8);
as[sCol2][sRowPlus8] = temp.x;
as[sCol2Plus1][sRowPlus8] = temp.y;
__syncthreads(); // wait for read to complete
float br0 = b[0];
float br1 = b[n];
float br2 = b[nTimes2];
float br3 = b[nTimes3];
#pragma unroll
for (j = 0; j< 7; j++)
{
b += nTimes4;
update1 (&as[j*4][0], br0, cr); br0 = b[0];
update1 (&as[j*4+1][0], br1, cr); br1 = b[n];
update1 (&as[j*4+2][0], br2, cr); br2 = b[nTimes2];
update1 (&as[j*4+3][0], br3, cr); br3 = b[nTimes3];
}
b += nTimes4;
update1 (&as[28][0], br0, cr);
update1 (&as[29][0], br1, cr);
update1 (&as[30][0], br2, cr);
update1 (&as[31][0], br3, cr);

a += 32;
__syncthreads(); // wait for computation to complete
}

// output cr[]
int cNext = 16*blockIdx.y*n + 128*blockIdx.x + tid;
c += cNext;
for (int i = 0; i < 16; i++)
{
c[0] = cr[i];
c += n;
}
}






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

    cudaMalloc((void**)&dA,sizeof(float)*M*K);
    cudaMalloc((void**)&dB,sizeof(float)*K*N);
    cudaMalloc((void**)&dC,sizeof(float)*N*M);

    cudaMemcpy(dA,A,sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(float)*K*N, cudaMemcpyHostToDevice);

    dim3 threads( 16, 8 );
    dim3 blocks(N / 128, N / 16);

    printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    GPU7<<<blocks,threads>>>(dA,dB,dC,N);
    
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

    printf("%f\n",ms);
    return 0;
}


