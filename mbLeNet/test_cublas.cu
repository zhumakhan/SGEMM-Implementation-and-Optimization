#include "utils.cpp"
#include <cublas.h>
#include <cublas_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void test_cublas(const int M, const int K, const int N) {

    float *a = utils::random_matrix_gpu<float>(M, K, utils::COLUMN_MAJOR,-1,1);
    float *b = utils::random_matrix_gpu<float>(K, N, utils::COLUMN_MAJOR,-1,1);
    float *c = (float*)malloc(M*N*sizeof(float));

    float *dev_a, *dev_b, *dev_c;
    float ms;

    cudaMalloc((void**)&dev_a, M*K*sizeof(float));
    cudaMalloc((void**)&dev_b, K*N*sizeof(float));
    cudaMalloc((void**)&dev_c, M*N*sizeof(float));

    cudaMemcpy(dev_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice);
    
    cublasStatus_t status;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha=1.0f, beta=0;
    // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
	//	    &al, dev_a, M, dev_b, K, &bet, dev_c, M);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
		    &alpha, dev_a, M, dev_b, K, &beta, dev_c, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaError_t cuda_error = cudaGetLastError();
    if(cuda_error != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
      return;
    }

    switch(status){
        case (CUBLAS_STATUS_SUCCESS):{
            break;
        }
        case (CUBLAS_STATUS_NOT_INITIALIZED):{
            printf("the library was not initialized\n");
            break;
        }
        case (CUBLAS_STATUS_INVALID_VALUE):{
            printf("the parameters m,n,k<0\n");
            break;
        }
        case (CUBLAS_STATUS_ARCH_MISMATCH):{
            printf("Architecture problem. in the case of cublasHgemm the device does not support math in half precision.\n");
            break;
        }
        case (CUBLAS_STATUS_EXECUTION_FAILED):{
            printf("the function failed to launch on the GPU\n");
            break;
        }
        default:{
            printf("Unknown error occured in cublasSgemm\n");
            break;
        }
    }    
    bool correct = true;

#ifdef CHECK
    cudaMemcpy(c, dev_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    correct = utils::check_mul<float>(a, b, c, M, K, N, utils::COLUMN_MAJOR, utils::COLUMN_MAJOR, utils::COLUMN_MAJOR);
#endif


#ifdef DEBUG

#ifndef CHECK
    cudaMemcpy(C,dC,sizeof(float)*N*M, cudaMemcpyDeviceToHost);
#endif

    std::cout << "Matrix A:" << std::endl;
    utils::print_mat_gpu(a, M, K, utils::COLUMN_MAJOR);
    std::cout << "\nMatrix B:" << std::endl;
    utils::print_mat_gpu(b, K, N, utils::COLUMN_MAJOR);
    std::cout << "\nMatrix C:" << std::endl;
    utils::print_mat_gpu(c, M, N, utils::COLUMN_MAJOR);
#endif

    cublasDestroy(handle);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    free(a);
    free(b);
    free(c);
    
    std::cout << "cublas" << " " << ms << " " << correct << std::endl;
    return;
}
    
int main(int argc, char * argv[]){
    int M = std::atoi(argv[1]);
    int K = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);

    printf("M=%d K=%d N=%d\n",M,K,N);
    test_cublas(M,K,N);
}