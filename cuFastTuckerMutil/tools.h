#include <numeric>
#include <iomanip>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <sys/time.h>

#define type_of_data float

using namespace std;

inline double Seconds() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void Getting_Input(char* InputPath_train, char* InputPath_test, int order,
		int** dimen, int* nnz_train, int* nnz_test, int*** index_train,
		type_of_data** value_train, int** index_test, type_of_data** value_test,
		double* data_norm);

void Multi_Parameter_Initialization(int order, int gpu_number,
		int *multi_gpu_part, int *part_per_gpu, int core_kernel,
		int core_length, int core_dimen, int* dimen, double data_norm,
		type_of_data*** parameter_a, type_of_data*** parameter_b, int nnz_train,
		int** index_train, type_of_data* value_train, int nnz_test,
		int* index_test, type_of_data* value_test, int **multi_nnz_train,
		int ***multi_index_train, type_of_data ***multi_value_train,
		int **multi_nnz_test, int ***multi_index_test,
		type_of_data ***multi_value_test, int ***multi_part,
		int ***multi_start);

void Mutil_Cuda_Parameter_Initialization(int order, int gpu_number,
		int multi_gpu_part, int part_per_gpu, int* dimen, int core_kernel,
		int core_dimen, int *multi_nnz_train, int **multi_index_train_host,
		int ***multi_index_train_device, int *multi_nnz_test,
		int **multi_index_test_host, int ***multi_index_test_device,
		type_of_data **multi_value_train_host,
		type_of_data ***multi_value_train_device,
		type_of_data **multi_value_test_host,
		type_of_data ***multi_value_test_device,
		type_of_data** parameter_a_host, type_of_data** parameter_b_host,
		type_of_data**** parameter_a_device,
		type_of_data**** parameter_b_device,
		type_of_data**** parameter_a_host_to_device,
		type_of_data**** parameter_b_host_to_device);

void Select_Best_Result(type_of_data* train_rmse, type_of_data* train_mae,
type_of_data* test_rmse, type_of_data* test_mae,
type_of_data* best_train_rmse, type_of_data* best_train_mae,
type_of_data* best_test_rmse, type_of_data* best_test_mae);
