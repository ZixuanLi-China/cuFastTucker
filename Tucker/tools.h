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
		type_of_data*** value_train, int** index_test,
		type_of_data** value_test, double* data_norm);

void Parameter_Initialization(int order, int core_length, int core_dimen,
		int* dimen, double data_norm, type_of_data*** parameter_a,
		type_of_data** parameter_g);

void Cuda_Parameter_Initialization(int order, int core_length, int core_dimen,
		int* dimen_host, int nnz_train, type_of_data **value_train_host,
		type_of_data ***value_train_device,
		type_of_data ***value_train_host_to_device, int **index_train_host,
		int ***index_train_device, int ***index_train_host_to_device,
		int nnz_test, type_of_data *value_test_host,
		type_of_data **value_test_device, int *index_test_host,
		int **index_test_device, type_of_data** parameter_a_host,
		type_of_data* parameter_g_host, type_of_data*** parameter_a_device,
		type_of_data** parameter_g_device,
		type_of_data*** parameter_a_host_to_device, type_of_data*** g_device,
		type_of_data*** g_host_to_device);

void Select_Best_Result(type_of_data* train_rmse, type_of_data* train_mae,
type_of_data* test_rmse, type_of_data* test_mae,
type_of_data* best_train_rmse, type_of_data* best_train_mae,
type_of_data* best_test_rmse, type_of_data* best_test_mae);
