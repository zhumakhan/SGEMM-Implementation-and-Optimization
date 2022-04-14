#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <random>
#include <cmath>

#ifndef MB_NET_UTILS
#define MB_NET_UTILS

#ifndef IDXR
#define IDXR(i,j,row,col) (((i)*(col))+(j))
#endif

#ifndef IDXC
#define IDXC(i,j,row,col) (((j)*(row))+(i))
#endif

namespace utils {
int ROW_MAJOR = 1;
int COLUMN_MAJOR = 2;



__inline__ int IDX(int i, int j, int row, int col, int order) {
    /*
    0-based indexing
    converts 2d row/column major matrix indices (i,j) to
    1d array index
    */
    return (order == ROW_MAJOR ? IDXR(i,j,row,col) : IDXC(i,j,row,col));
}

template <typename T>
T** random_fill_matrix(int row, int col, T min=0, T max=100) {
    /* A function to quickly generate a matrix in some range [min, max]
     * Parameters:
     *   row: number of rows of matrix
     *   col: number of columns of matrix
     *   min, max: the range of random number. default to [0, 100]
     * Returns:
     *   a specific type 2d pointer pointed to the matrix
     */
    T** mat = new T*[row];
    for (int i = 0; i < row; ++i) {
        mat[i] = new T[col];
    }

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> unif(min, max);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
	       mat[i][j] = unif(mt);
        }
    }
    return mat;
}

template <typename T>
T* random_matrix_gpu(int row, int col, int order_type, T min=-50, T max=50) {
    /* A function to quickly generate a matrix in some range [min, max)
     * Note that it is very hard to allocate 2-d array on GPU,
     * so in most of the cases, we pass the 2-d array as a 1-d array
     * to the device following row-major or column-major order.
     *
     * Parameters:
     * ----------
     *   row: number of rows of matrix
     *   col: number of columns of matrix
     *   min, max: the range of random number. default to [-50, 50)
     *
     * Returns:
     * -------
     *    a specific type 1d pinter pointed to the matrix
     */
    T* mat = new T[row*col];
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> unif(min, max);
    
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            mat[ IDX(i,j,row,col,order_type) ] = unif(mt);
        }
    }
    return mat;
}

template <typename T>
void print_mat_gpu(T* mat, int row, int col, int order_type) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            std::cout << mat[ IDX(i,j,row,col,order_type) ] << " ";
	    }
        std::cout << std::endl;
    }
}

template <typename T>
void print_mat(T** mat, int row, int col) {
    // Display the matrix for visualizatoin
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            std::cout << mat[i][j] << " ";
        }std::cout << std::endl;
    }
}

template <typename T>
bool check_sum(T* a, T* b, T* c, int row, int col, int order_type_a, int order_type_b, int order_type_c) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
	    	if (a[ IDX(i,j,row,col,order_type_a) ] + b[ IDX(i,j,row,col,order_type_b) ] != c[ IDX(i,j,row,col,order_type_c) ]) {
				std::cout << a[ IDX(i,j,row,col,order_type_a) ] << " " << b[ IDX(i,j,row,col,order_type_b) ] << " "
				 << c[ IDX(i,j,row,col,order_type_c) ] << std::endl;
				return false;
	    	}
		}
    }
	return true;
}

template <typename T>
bool check_mul(T* a, T* b, T* c, int M, int K, int N, int order_type_a, int order_type_b, int order_type_c) {
    /* Check if the result of matrix multiplication is right.*/
    std::cout<<std::setprecision(10);
    T value = 0;
    
    int err_count = 0;

	for (int i = 0; i < M; ++i) {
	    for (int j = 0; j < N; ++j) {
            value = 0;
            for (int k = 0; k < K; ++k) {
                value += a[ IDX(i, k, M, K, order_type_a) ] * b[ IDX(k, j, K, N, order_type_b) ];
            }
            if ( fabs(value - c[ IDX(i, j, M, N, order_type_c) ] ) > 0.01) {
                // std::cout << c[ IDX(i, j, M, N, order_type_c) ] << " " << value << std::endl;
                err_count += 1;
                // return false;
            }
        }
    }

    printf("Err count: %d\n", err_count);

    return err_count == 0;
}
//end of utils namespace
}

// int main() {
//     // int a[4] = {1, 2, 3, 4};
//     // int b[4] = {1, 2, 3, 4};
//     // int c[4] = {6, 10, 15, 22};
//     // std::cout << utils::check_mul<int>(a, b, c, 2, 2, 2, utils::C_ORDER) << std::endl;
//     // std::cout << fabs(-1.34) << std::endl;
//     float *a = utils::random_matrix_gpu<float>(10,5,utils::COLUMN_MAJOR,-50,50);
//     utils::print_mat_gpu<float>(a,10,5,utils::COLUMN_MAJOR);
//     delete[] a;
//     return 0;
// }

#endif

