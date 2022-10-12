#include <cuda_runtime.h>
#include <curand_kernel.h>

#define type_of_data float

void Update_Parameter_A(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a, const int nnz,
		type_of_data **value, int **index, type_of_data **g,
		const type_of_data learn_rate_a, const type_of_data lambda_a);

void Update_Parameter_G_Batch(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data *parameter_g, const int nnz, type_of_data **value,
		int **index, const type_of_data learn_rate_g,
		const type_of_data lambda_g, int model);

void GET_RMSE_AND_MAE(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data *parameter_g, const int nnz, type_of_data **value,
		int **index, type_of_data **g, type_of_data *rmse,
		type_of_data *mae);

void GET_RMSE_AND_MAE(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data *parameter_g, const int nnz, const type_of_data *value,
		const int *index, type_of_data **g, type_of_data *rmse,
		type_of_data *mae);

