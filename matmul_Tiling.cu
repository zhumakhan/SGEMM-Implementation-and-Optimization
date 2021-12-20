#include "utils.cpp"

const int TILE_SIZE = 16;

template <typename T>

__global__ void matmul_Tiling(T *A, T *B, T *C, int M, int K, int N) {
	/* Basic tiling implementation of matrix multiplication.
	 * Based on a more mathematically reasonable indexing method.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ T As[TILE_SIZE][TILE_SIZE];
	__shared__ T Bs[TILE_SIZE][TILE_SIZE];

	int aBegin = K * TILE_SIZE * by;
	int aEnd = aBegin + K - 1;
	int aStep = TILE_SIZE;

	int bBegin = TILE_SIZE * bx;
	int bStep = TILE_SIZE * N;

	T Csub = 0;

	for (int i = aBegin, j = bBegin; i <= aEnd; i += aStep, j += bStep) {
		As[ty][tx] = A[i + K * ty + tx];
		Bs[tx][ty] = B[j + N * tx + ty];

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k) {
			Csub += As[ty][k]*Bs[k][tx];
		}
		
		__syncthreads();
	}
	int cIdx = N * TILE_SIZE * by + TILE_SIZE * bx;
	C[cIdx + N * ty + tx] = Csub;
}

int main(int argc, char *argv[]) {
	int M = std::atoi(argv[1]), K = std::atoi(argv[2]), N = std::atoi(argv[3]);
	printf("M=%d K=%d N=%d\n",M,K,N);

	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 grid(N / TILE_SIZE, M / TILE_SIZE);

	float *a = utils::random_matrix_gpu<float>(M, K, utils::ROW_MAJOR);
	float *b = utils::random_matrix_gpu<float>(K, N, utils::ROW_MAJOR);
	float *c = (float*)malloc(sizeof(float)*M*N);
	
	float *dev_a, *dev_b, *dev_c;
	float ms;

	cudaMalloc((void**)&dev_a, M*K*sizeof(float));
	cudaMalloc((void**)&dev_b, K*N*sizeof(float));
	cudaMalloc((void**)&dev_c, M*N*sizeof(float));

	cudaMemcpy(dev_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

	matmul_Tiling<float><<<grid, threads>>>(dev_a, dev_b, dev_c, M, K, N);

	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	cudaMemcpy(c, dev_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);
#ifdef CHECK
	std::cout << (utils::check_mul<float>(a, b, c, M, K, N, utils::ROW_MAJOR, utils::ROW_MAJOR, utils::ROW_MAJOR) ? "Correct!!" : "Wrong Answer!") << std::endl;
#endif
#ifdef DEBUG
    std::cout << "Matrix A:" << std::endl;
    utils::print_mat_gpu(a, M, K, utils::ROW_MAJOR);
    std::cout << "Matrix B:" << std::endl;
    utils::print_mat_gpu(b, K, N, utils::ROW_MAJOR);
    std::cout << "Matrix C:" << std::endl;
    utils::print_mat_gpu(c, M, N, utils::ROW_MAJOR);
#endif

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);

    printf("%f\n",ms);
	return 0;
}

