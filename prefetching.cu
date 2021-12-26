/*
    by zhumakhan.nazir@nu.edu.kz
*/

#include "utils.cpp"
#include <stdio.h>
#include <algorithm>

#define TILE_SIZE 16
#define VECTOR_SIZE 4

/*
    A is a row major matrix ( M x K )
    B is a row major matrix ( K x N )

    Asub: TILE_SIZE x TILE_SIZE
    Bsub: TILE_SIZE x ( TILE_SIZE * VECTOR_SIZE )

    dim3 threads ( TILE_SIZE, VECTOR_SIZE )
    dim3 blocks  ( N / ( TILE_SIZE * VECTOR_SIZE ), M / TILE_SIZE )
*/

__global__ void mmPrefetching(float *A, float *B, float *C, const int M, const int K, const int N){
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int aBegin  = TILE_SIZE * K * by;
    const int aEnd    = aBegin + K;
    const int aStep   = TILE_SIZE;

    const int bBegin  = TILE_SIZE * VECTOR_SIZE * bx;
    const int bStep   = TILE_SIZE * N;

    __shared__ float As1[ TILE_SIZE * TILE_SIZE ];
    __shared__ float As2[ TILE_SIZE * TILE_SIZE ];

    float Cv[ TILE_SIZE ] = { 0 };

    int i, j;
    
    float *ptr1, *ptr2;
    float bValue;

    // to avoid repeated computations 
    const int t1 = tx * TILE_SIZE + ty;
    const int t2 = ty * K + tx;
    const int t3 = ty * TILE_SIZE + tx;
    const int t4 = TILE_SIZE / VECTOR_SIZE;
    int t10      = 0;

    float *pre1 = As1;
    float *pre2 = As2;

// prefecth to first shared memory block;
    ptr1 = &pre1[ t1 ];
    ptr2 = &A[ aBegin + t2 ];

    for(i = 0; i < t4; ++i){
        ptr1[ t10 ] = ptr2[ t10 * K ];
        t10         += VECTOR_SIZE;
    }
    __syncthreads();



    for(int a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep){
        
        ptr1    = &pre2[ t1 ];
        ptr2    = &A[ a + aStep + t2 ];
        t10     = 0;
        for(i = 0; i < t4; ++i){
            // load elements to As in column major way from matrix A
            ptr1[ t10 ] = ptr2[ t10 * K ];
            t10         += VECTOR_SIZE;
        }

        ptr1 = pre1;
        ptr2 = &B[ b + t3 ];

        for(i = 0; i < TILE_SIZE; ++i){
            bValue = *ptr2;

            for(j = 0; j < TILE_SIZE; ++j){
                Cv[ j ] += ptr1[ j ] * bValue;
            }

            ptr1 += TILE_SIZE;
            ptr2 += N;
        }

        __syncthreads();

//swap pointers to shared spaces
        ptr1 = pre1;
        pre1 = pre2;
        pre2 = ptr1;

    }

    j = N * TILE_SIZE * by + TILE_SIZE * VECTOR_SIZE * bx;
    // c += TILE_SIZE * ty + tx;
    j += t3;

    for(i = 0; i < TILE_SIZE; ++i){
        C[ j ] = Cv[ i ];
        j += N;
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

    dim3 threads( TILE_SIZE, VECTOR_SIZE );
    dim3 blocks(N / (TILE_SIZE * VECTOR_SIZE), M / TILE_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    mmPrefetching<<<blocks,threads>>>(dA,dB,dC,M,K,N);
    
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


