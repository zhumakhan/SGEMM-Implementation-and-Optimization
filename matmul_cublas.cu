#include "utils.cpp"
#include <cublas.h>
#include <cublas_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char *argv[]) {
    int M = std::atoi(argv[1]), K = std::atoi(argv[2]), N = std::atoi(argv[3]);
    float *a = utils::random_matrix_gpu<float>(M, K, utils::COLUMN_MAJOR);
    float *b = utils::random_matrix_gpu<float>(K, N, utils::COLUMN_MAJOR);
    float *c = (float*)malloc(M*N*sizeof(float));

    float *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, M*K*sizeof(float));
    cudaMalloc((void**)&dev_b, K*N*sizeof(float));
    cudaMalloc((void**)&dev_c, M*N*sizeof(float));

    cudaMemcpy(dev_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    float al=1.0f, bet=0;
    // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
	//	    &al, dev_a, M, dev_b, K, &bet, dev_c, M);
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
		    &al, dev_a, M, dev_b, K, &bet, dev_c, M);
    
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

    cudaMemcpy(c, dev_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);
#ifdef CHECK
    std::cout << (utils::check_mul<float>(a, b, c, M, K, N, utils::COLUMN_MAJOR) 
		    ? "Correct!!" : "Wrong Answer!") << std::endl;
#endif
#ifdef DEBUG
    std::cout << "Matrix A:" << std::endl;
    utils::print_mat_gpu(a, M, K, utils::COLUMN_MAJOR);
    std::cout << "\nMatrix B:" << std::endl;
    utils::print_mat_gpu(b, K, N, utils::COLUMN_MAJOR);
    std::cout << "\nMatrix C:" << std::endl;
    utils::print_mat_gpu(c, M, N, utils::COLUMN_MAJOR);
#endif

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    return 0;
}
    
