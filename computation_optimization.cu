/*
    by zhumakhan.nazir@nu.edu.kz
*/

#include "utils.cpp"
#include <stdio.h>

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

__global__ void mmCompOpt(float *A, float *B, float *C, int M, int K, int N){
    
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    volatile __shared__ float As[ TILE_SIZE * TILE_SIZE ];

    float Cv[ TILE_SIZE ] = { 0 };

    int aBegin  = K * TILE_SIZE * by;
    int aEnd    = aBegin + K;
    int aStep   = TILE_SIZE;

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

    for(int a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep){

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





