#include <fstream>
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>
#include "parameter.h"

type_of_data frand(type_of_data x, type_of_data y) {
	return ((y - x) * ((type_of_data) rand() / RAND_MAX)) + x;
}

void Getting_Input(char *InputPath_train, char *InputPath_test, int order,
		int **dimen, int *nnz_train, int *nnz_test, int ***index_train,
		type_of_data ***value_train, int **index_test,
		type_of_data **value_test, double *data_norm) {

	*data_norm = 0.0;
	*dimen = (int*) malloc(sizeof(int) * order);
	for (int i = 0; i < order; i++) {
		(*dimen)[i] = 0;
	}

	char tmp[1024];

	FILE *train_file_count = fopen(InputPath_train, "r");
	FILE *train_file = fopen(InputPath_train, "r");

	FILE *test_file_count = fopen(InputPath_test, "r");
	FILE *test_file = fopen(InputPath_test, "r");

	*nnz_train = 0;
	*nnz_test = 0;

	while (fgets(tmp, 1024, train_file_count)) {
		(*nnz_train)++;
	}

	while (fgets(tmp, 1024, test_file_count)) {
		(*nnz_test)++;
	}

	fclose(train_file_count);
	fclose(test_file_count);

	int data_per_part = (*nnz_train) / data_part + 1;
	*index_train = (int**) malloc(sizeof(int*) * data_part);
	*value_train = (type_of_data**) malloc(sizeof(type_of_data*) * data_part);

	for (int i = 0; i < data_part - 1; i++) {
		(*index_train)[i] = (int*) malloc(sizeof(int) * data_per_part * order);
		(*value_train)[i] = (type_of_data*) malloc(
				sizeof(type_of_data) * data_per_part);
	}

	(*index_train)[data_part - 1] = (int*) malloc(
			sizeof(int) * ((*nnz_train) - (data_part - 1) * data_per_part)
					* order);
	(*value_train)[data_part - 1] = (type_of_data*) malloc(
			sizeof(type_of_data)
					* ((*nnz_train) - (data_part - 1) * data_per_part));

	*index_test = (int*) malloc(sizeof(int) * (*nnz_test) * order);
	*value_test = (type_of_data*) malloc(sizeof(type_of_data) * (*nnz_test));

	char *p;
	for (int i = 0; i < (*nnz_train); i++) {
		fgets(tmp, 1024, train_file);
		p = strtok(tmp, "\t");
		for (int j = 0; j < order; j++) {
			int int_temp = atoi(p);
			(*index_train)[i / data_per_part][(i % data_per_part) * order + j] =
					int_temp - 1;
			p = strtok(NULL, "\t");
			if (int_temp > (*dimen)[j]) {
				(*dimen)[j] = int_temp;
			}
		}
		type_of_data double_temp = atof(p);
		(*value_train)[i / data_per_part][i % data_per_part] = double_temp;
		(*data_norm) += double_temp * double_temp;
	}
	*data_norm = sqrt((*data_norm) / (*nnz_train));

	for (int i = 0; i < (*nnz_test); i++) {
		fgets(tmp, 1024, test_file);
		p = strtok(tmp, "\t");
		for (int j = 0; j < order; j++) {
			int int_temp = atoi(p);
			(*index_test)[i * order + j] = int_temp - 1;
			p = strtok(NULL, "\t");
			if (int_temp > (*dimen)[j]) {
				(*dimen)[j] = int_temp;
			}
		}
		(*value_test)[i] = atof(p);
	}

	fclose(train_file);
	fclose(test_file);

}

void Parameter_Initialization(int order, int core_kernel, int core_length,
		int core_dimen, int *dimen, double data_norm,
		type_of_data ***parameter_a, type_of_data ***parameter_b) {

	srand((unsigned) time(NULL));

	*parameter_a = (type_of_data**) malloc(sizeof(type_of_data*) * order);
	*parameter_b = (type_of_data**) malloc(sizeof(type_of_data*) * order);

	for (int i = 0; i < order; i++) {
		(*parameter_a)[i] = (type_of_data*) malloc(
				sizeof(type_of_data) * dimen[i] * core_dimen);
		(*parameter_b)[i] = (type_of_data*) malloc(
				sizeof(type_of_data) * core_dimen * core_kernel);
	}

	for (int i = 0; i < order; i++) {

		for (int j = 0; j < dimen[i] * core_dimen; j++) {
			(*parameter_a)[i][j] = pow(data_norm / core_length, 1.0 / order)
					* frand(0.5, 1.5);
		}

		for (int j = 0; j < core_kernel * core_dimen; j++) {
			(*parameter_b)[i][j] = pow(1.0 / core_kernel, 1.0 / order)
					* frand(0.5, 1.5);
		}
	}
}

void Cuda_Parameter_Initialization(int order, int core_kernel, int core_dimen,
		int *dimen, int nnz_train,
		type_of_data **value_train_host,
		type_of_data ***value_train_device,
		type_of_data ***value_train_host_to_device, int **index_train_host,
		int ***index_train_device, int ***index_train_host_to_device,
		int nnz_test,
		type_of_data *value_test_host, type_of_data **value_test_device,
		int *index_test_host, int **index_test_device,
		type_of_data **parameter_a_host, type_of_data **parameter_b_host,
		type_of_data ***parameter_a_device, type_of_data ***parameter_b_device,
		type_of_data ***parameter_a_host_to_device,
		type_of_data ***parameter_b_host_to_device) {

	*parameter_a_host_to_device = (type_of_data**) malloc(
			sizeof(type_of_data*) * order);
	*parameter_b_host_to_device = (type_of_data**) malloc(
			sizeof(type_of_data*) * order);
	*index_train_host_to_device = (int**) malloc(sizeof(int*) * data_part);
	*value_train_host_to_device = (type_of_data**) malloc(
			sizeof(type_of_data*) * data_part);

	cudaMalloc((void**) &(*value_train_device),
			sizeof(type_of_data*) * data_part);
	cudaMalloc((void**) &(*index_train_device), sizeof(int*) * data_part);

	cudaMalloc((void**) &(*value_test_device), sizeof(type_of_data) * nnz_test);
	cudaMalloc((void**) &(*index_test_device), sizeof(int) * nnz_test * order);

	cudaMemcpy(*value_test_device, value_test_host,
			sizeof(type_of_data) * nnz_test, cudaMemcpyHostToDevice);
	cudaMemcpy(*index_test_device, index_test_host,
			sizeof(type_of_data) * nnz_test * order, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &(*parameter_a_device), sizeof(type_of_data*) * order);
	cudaMalloc((void**) &(*parameter_b_device), sizeof(type_of_data*) * order);

	for (int i = 0; i < order; i++) {

		type_of_data *temp_a;
		cudaMalloc((void**) &temp_a,
				sizeof(type_of_data) * dimen[i] * core_dimen);
		(*parameter_a_host_to_device)[i] = temp_a;
		cudaMemcpy(temp_a, parameter_a_host[i],
				sizeof(type_of_data) * dimen[i] * core_dimen,
				cudaMemcpyHostToDevice);

		type_of_data *temp_b;
		cudaMalloc((void**) &temp_b,
				sizeof(type_of_data) * core_kernel * core_dimen);
		(*parameter_b_host_to_device)[i] = temp_b;
		cudaMemcpy(temp_b, parameter_b_host[i],
				sizeof(type_of_data) * core_kernel * core_dimen,
				cudaMemcpyHostToDevice);

	}

	int data_per_part = nnz_train / data_part + 1;
	for (int i = 0; i < data_part - 1; i++) {

		int *temp_index;
		cudaMalloc((void**) &temp_index, sizeof(int) * data_per_part * order);
		(*index_train_host_to_device)[i] = temp_index;
		cudaMemcpy(temp_index, index_train_host[i],
				sizeof(int) * data_per_part * order, cudaMemcpyHostToDevice);

		type_of_data *temp_value;
		cudaMalloc((void**) &temp_value, sizeof(type_of_data) * data_per_part);
		(*value_train_host_to_device)[i] = temp_value;
		cudaMemcpy(temp_value, value_train_host[i],
				sizeof(type_of_data) * data_per_part, cudaMemcpyHostToDevice);
	}

	int *temp_index;
	cudaMalloc((void**) &temp_index,
			sizeof(int) * (nnz_train - (data_part - 1) * data_per_part)
					* order);
	(*index_train_host_to_device)[data_part - 1] = temp_index;
	cudaMemcpy(temp_index, index_train_host[data_part - 1],
			sizeof(int) * (nnz_train - (data_part - 1) * data_per_part) * order,
			cudaMemcpyHostToDevice);

	type_of_data *temp_value;
	cudaMalloc((void**) &temp_value,
			sizeof(type_of_data)
					* (nnz_train - (data_part - 1) * data_per_part));
	(*value_train_host_to_device)[data_part - 1] = temp_value;
	cudaMemcpy(temp_value, value_train_host[data_part - 1],
			sizeof(type_of_data)
					* (nnz_train - (data_part - 1) * data_per_part),
			cudaMemcpyHostToDevice);

	cudaMemcpy(*index_train_device, *index_train_host_to_device,
			sizeof(int*) * data_part, cudaMemcpyHostToDevice);
	cudaMemcpy(*value_train_device, *value_train_host_to_device,
			sizeof(type_of_data*) * data_part, cudaMemcpyHostToDevice);

	cudaMemcpy(*parameter_a_device, *parameter_a_host_to_device,
			sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);
	cudaMemcpy(*parameter_b_device, *parameter_b_host_to_device,
			sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);

}

void Select_Best_Result(type_of_data *train_rmse, type_of_data *train_mae,
		type_of_data *test_rmse, type_of_data *test_mae,
		type_of_data *best_train_rmse, type_of_data *best_train_mae,
		type_of_data *best_test_rmse, type_of_data *best_test_mae) {
	if (*train_rmse < *best_train_rmse) {
		*best_train_rmse = *train_rmse;
	}
	if (*train_mae < *best_train_mae) {
		*best_train_mae = *train_mae;
	}
	if (*test_rmse < *best_test_rmse) {
		*best_test_rmse = *test_rmse;
	}
	if (*test_mae < *best_test_mae) {
		*best_test_mae = *test_mae;
	}
}
