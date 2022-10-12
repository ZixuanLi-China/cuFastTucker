#include <cuda_runtime.h>
#include <curand_kernel.h>

#define type_of_data float

void Update_Parameter_A(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a_device,
		type_of_data **parameter_b_device, const int nnz_train,
		type_of_data **value_train_device, int **index_train_device,
		type_of_data learn_rate_a, type_of_data lambda_a, const int model);

void Update_Parameter_B_Batch(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, type_of_data **value,
		int **index, const type_of_data learn_rate_b,
		const type_of_data lambda_b, const int model);

void GET_RMSE_AND_MAE(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		type_of_data **value, int **index, type_of_data *rmse,
		type_of_data *mae);

void GET_RMSE_AND_MAE(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, type_of_data *value,
		int *index, type_of_data *rmse, type_of_data *mae);
