#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#define type_of_data float

void Mutil_Update_Parameter_A(const int order, const int gpu_number,
		const int part_per_gpu, const int core_kernel, const int core_dimen,
		const int *multi_nnz_train,
		type_of_data **multi_value_train_device, int **multi_index_train_device,
		type_of_data ***parameter_a_device,
		type_of_data ***parameter_a_host_to_device,
		type_of_data ***parameter_b_device, int **multi_start, int **multi_part,
		type_of_data learn_rate_a, type_of_data lambda_a, double *memcpy_time_a,
		const int model);

void Mutil_Update_Parameter_B(const int order, const int gpu_number,
		const int nnz, const int part_per_gpu, const int core_kernel,
		const int core_dimen, const int *multi_nnz_train,
		type_of_data **multi_value_train_device, int **multi_index_train_device,
		type_of_data ***parameter_a_device, type_of_data ***parameter_b_device,
		type_of_data ***parameter_b_host_to_device, type_of_data learn_rate_b,
		type_of_data lambda_b, double *memcpy_time_b, const int model);

void Mutil_GET_RMSE_AND_MAE(const int order, const int gpu_number,
		const int multi_gpu_part, const int part_per_gpu, const int core_kernel,
		const int core_dimen, const int *multi_nnz_train,
		type_of_data **multi_value_train_device, int **multi_index_train_device,
		type_of_data ***parameter_a_device, type_of_data ***parameter_b_device,
		type_of_data *rmse, type_of_data *mae);
