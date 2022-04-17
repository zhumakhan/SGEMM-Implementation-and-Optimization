/*
    by zhumakhan.nazir@nu.edu.kz
*/

#include "utils.cpp"
#include <stdio.h>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
#ifndef VECTOR_SIZE
#define VECTOR_SIZE 8
#endif

/*
    A is a row major matrix ( M x K )
    B is a row major matrix ( K x N )

    Asub: TILE_SIZE x TILE_SIZE
    Bsub: TILE_SIZE x ( TILE_SIZE * VECTOR_SIZE )

    dim3 threads ( TILE_SIZE, VECTOR_SIZE )
    dim3 blocks  ( N / ( TILE_SIZE * VECTOR_SIZE ), M / TILE_SIZE )
*/

__global__ void mmCompOpt(float *A, float *B, float *C, int M, int K, int N){
    
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[ TILE_SIZE * TILE_SIZE ];

    float Cv[ TILE_SIZE ] = { 0 };

    int aBegin  = K * TILE_SIZE * by;
    int aEnd    = aBegin + K;

    int bBegin  = TILE_SIZE * VECTOR_SIZE * bx;
    int bStep   = TILE_SIZE * N;

    int i, j;
    
    float *aPtr, *bPtr;
    float bValue;

    // to avoid repeated computations 
    // int t1 = tx * TILE_SIZE + ty;
    // int t2 = ty * K + tx;
    // int t3 = ty * TILE_SIZE + tx;
    // int t4;

    for(int a = aBegin, b = bBegin; a < aEnd; a += TILE_SIZE, b += bStep){

        for(i = 0; i < TILE_SIZE / VECTOR_SIZE; ++i){
            // load elements to As in column major way from matrix A
            // t4 = i * VECTOR_SIZE;
            As[ tx * TILE_SIZE + ty + i * VECTOR_SIZE ] = A[ a + K * (i * VECTOR_SIZE + ty) + tx ];
            // As[ t1 + t4 ] = A[ a + t4 * K + t2 ];
        }
        
        __syncthreads();

        aPtr = As;
        bPtr = &B[ b + TILE_SIZE * ty + tx ];
        // bPtr = &B[b + t3 ];

        for(i = 0; i < TILE_SIZE; ++i){
            bValue = *bPtr;

            for(j = 0; j < TILE_SIZE; ++j){
                Cv[ j ] += aPtr[ j ] * bValue;
            }

            aPtr += TILE_SIZE;
            bPtr += N;
        }

        __syncthreads();

    }

    int c = N * TILE_SIZE * by + TILE_SIZE * VECTOR_SIZE * bx;
    c += TILE_SIZE * ty + tx;
    // c += t3;

    for(i = 0; i < TILE_SIZE; ++i){
        C[ c ] = Cv[ i ];
        c += N;
    }
}

__global__ void mmCompOpt_v1(float *A, float *B, float *C, const int M, const int K, const int N){
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int aBegin  = TILE_SIZE * K * by;
    const int aEnd    = aBegin + K;

    const int bBegin  = TILE_SIZE * VECTOR_SIZE * bx;
    const int bStep   = TILE_SIZE * N;

    __shared__ float As[ TILE_SIZE * TILE_SIZE ];

    float Cv[ TILE_SIZE ] = { 0 };

    int i, j;
    
    float *aPtr, *bPtr;
    float bValue;

    const int t1 = tx * TILE_SIZE + ty;
    const int t2 = ty * K + tx;
    const int t3 = ty * TILE_SIZE + tx;
    const int t4 = TILE_SIZE / VECTOR_SIZE;
    int t10      = 0;
    
    for(int a = aBegin, b = bBegin; a < aEnd; a += TILE_SIZE, b += bStep){
        
        aPtr    = &As[ t1 ];
        bPtr    = &A[ a + t2 ];
        t10     = 0;

        #pragma unroll
        for(i = 0; i < t4; ++i){
            aPtr[ t10 ] = bPtr[ t10*K ];
            t10         += VECTOR_SIZE;
        }
        
        __syncthreads();

        aPtr = As;
        bPtr = &B[ b + t3 ];

        #pragma unroll
        for(i = 0; i < TILE_SIZE; i += 1){
            bValue = *bPtr;

            #pragma unroll
            for(j = 0; j < TILE_SIZE; j += 1){
                Cv[ j ] += aPtr[ j ] * bValue;
            }

            aPtr += TILE_SIZE;
            bPtr += N;
        }

        __syncthreads();

    }

    j = bStep * by + bBegin;
    j += t3;

    #pragma unroll
    for(i = 0; i < TILE_SIZE; i += 1){
        C[ j ] = Cv[ i ];
        j += N;
    }
}


__global__ void mmCompOpt_v2(float *A, float *B, float *C, const int M, const int K, const int N){
    
 
    const int bBegin  = TILE_SIZE * VECTOR_SIZE * blockIdx.x;
    const int bStep   = TILE_SIZE * N;

    __shared__ float As[ TILE_SIZE * TILE_SIZE ];

    float Cv[ TILE_SIZE ] = { 0 };

    int i, j;
    
    float *aPtr, *bPtr;
    float bValue;

    const int t1 = threadIdx.x * TILE_SIZE + threadIdx.y;
    const int t2 = threadIdx.y * K + threadIdx.x;
    const int t3 = threadIdx.y * TILE_SIZE + threadIdx.x;

    int t10      = 0;
    
    for(int a = TILE_SIZE * K * blockIdx.y, b = bBegin; a < TILE_SIZE * K * blockIdx.y + K; a += TILE_SIZE, b += bStep){
        
        aPtr    = &As[ t1 ];
        bPtr    = &A[ a + t2 ];
        t10     = 0;

        #pragma unroll
        for(i = 0; i < TILE_SIZE / VECTOR_SIZE; ++i){
            aPtr[ t10 ] = bPtr[ t10*K ];
            t10         += VECTOR_SIZE;
        }
        
        __syncthreads();

        aPtr = As;
        bPtr = &B[ b + t3 ];

        #pragma unroll
        for(i = 0; i < TILE_SIZE; i += 1){
            bValue = *bPtr;

            #pragma unroll
            for(j = 0; j < TILE_SIZE; j += 1){
                Cv[ j ] += aPtr[ j ] * bValue;
            }

            aPtr += TILE_SIZE;
            bPtr += N;
        }

        __syncthreads();

    }

    j = bStep * blockIdx.y + bBegin;
    j += t3;

    #pragma unroll
    for(i = 0; i < TILE_SIZE; i += 1){
        C[ j ] = Cv[ i ];
        j += N;
    }
}

int main(int argc, char *argv[]){
    int M = std::atoi(argv[1]);
    int K = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);

    printf("M=%d K=%d N=%d\n",M,K,N);

    float *A = utils::random_matrix_gpu<float>(M, K, utils::ROW_MAJOR,-1,1);
    float *B = utils::random_matrix_gpu<float>(K, N, utils::ROW_MAJOR,-1,1);
    float *C = (float*)malloc(sizeof(float)*M*N);
    
    float *dA, *dB, *dC;

    cudaMalloc((void**)&dA,sizeof(float)*M*K);
    cudaMalloc((void**)&dB,sizeof(float)*K*N);
    cudaMalloc((void**)&dC,sizeof(float)*N*M);

    cudaMemcpy(dA,A,sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(float)*K*N, cudaMemcpyHostToDevice);

    dim3 threads( TILE_SIZE, VECTOR_SIZE );
    dim3 blocks(N / (TILE_SIZE * VECTOR_SIZE), M / TILE_SIZE);
    
    printf("%d %d\n", blocks.x, blocks.y);

    mmCompOpt_v2<<<blocks,threads>>>(dA,dB,dC,M,K,N);
    
    
    cudaError_t cuda_error = cudaGetLastError();
    if(cuda_error != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
      exit(-1);
    }

    cudaMemcpy(C,dC,sizeof(float)*N*M,cudaMemcpyDeviceToHost);

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

    return 0;
}




