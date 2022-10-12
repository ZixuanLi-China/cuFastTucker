#include "tools.h"
#include <numeric>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define type_of_data float
#define data_part 4

using namespace std;

type_of_data frand(type_of_data x, type_of_data y) {
	return ((y - x) * ((type_of_data) rand() / RAND_MAX)) + x;
}

void Getting_Input(char* InputPath_train, char* InputPath_test, int order,
		int** dimen, int* nnz_train, int* nnz_test, int*** index_train,
		type_of_data** value_train, int** index_test,
		type_of_data** value_test, double* data_norm) {

	*data_norm = 0.0;
	*dimen = (int *) malloc(sizeof(int) * order);
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
	*index_train = (int **) malloc(sizeof(int*) * data_part);
	for (int i = 0; i < data_part; i++) {
		(*index_train)[i] = (int *) malloc(sizeof(int) * data_per_part * order);
	}

	*value_train = (type_of_data *) malloc(sizeof(type_of_data) * (*nnz_train));

	*index_test = (int *) malloc(sizeof(int) * (*nnz_test) * order);
	*value_test = (type_of_data *) malloc(sizeof(type_of_data) * (*nnz_test));

	char* p;
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
		(*value_train)[i] = double_temp;
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

void Multi_Parameter_Initialization(int order, int gpu_number,
		int *multi_gpu_part, int *part_per_gpu, int core_length, int core_dimen,
		int* dimen, double data_norm, type_of_data*** parameter_a,
		type_of_data** parameter_g, int nnz_train, int** index_train,
		type_of_data* value_train, int nnz_test, int* index_test,
		type_of_data* value_test, int **multi_nnz_train,
		int ***multi_index_train, type_of_data ***multi_value_train,
		int **multi_nnz_test, int ***multi_index_test,
		type_of_data ***multi_value_test, int ***multi_part,
		int ***multi_start) {

	*parameter_a = (type_of_data **) malloc(sizeof(type_of_data*) * order);
	*parameter_g = (type_of_data *) malloc(sizeof(type_of_data) * core_length);

	for (int i = 0; i < order; i++) {
		(*parameter_a)[i] = (type_of_data *) malloc(
				sizeof(type_of_data) * dimen[i] * core_dimen);
	}

	for (int i = 0; i < order; i++) {
		for (int j = 0; j < dimen[i] * core_dimen; j++) {
			(*parameter_a)[i][j] = pow(data_norm / core_length, 1.0 / order)
					* 1.0 * frand(0.5, 1.5);
		}
	}
	for (int j = 0; j < core_length; j++) {
		(*parameter_g)[j] = 1.0 * frand(0.5, 1.5);
	}

	(*multi_gpu_part) = 1;
	for (int i = 0; i < order; i++) {
		(*multi_gpu_part) *= gpu_number;
	}

	(*part_per_gpu) = (*multi_gpu_part) / gpu_number;

	*multi_part = (int **) malloc(sizeof(int*) * order);
	*multi_start = (int **) malloc(sizeof(int*) * order);

	for (int i = 0; i < order; i++) {
		(*multi_part)[i] = (int *) malloc(sizeof(int) * gpu_number);
		(*multi_start)[i] = (int *) malloc(sizeof(int) * gpu_number);
		int quotient = dimen[i] / gpu_number;
		int remainder = dimen[i] % gpu_number;
		int sum = 0;
		for (int j = 0; j < remainder; j++) {
			(*multi_part)[i][j] = quotient + 1;
			(*multi_start)[i][j] = sum;
			sum += quotient + 1;
		}
		for (int j = remainder; j < gpu_number; j++) {
			(*multi_part)[i][j] = quotient;
			(*multi_start)[i][j] = sum;
			sum += quotient;
		}
	}

	*multi_nnz_train = (int *) malloc(sizeof(int) * (*multi_gpu_part));
	*multi_index_train = (int **) malloc(sizeof(int*) * (*multi_gpu_part));
	*multi_value_train = (type_of_data **) malloc(
			sizeof(type_of_data*) * (*multi_gpu_part));
	memset(*multi_nnz_train, 0, (*multi_gpu_part) * sizeof(int));

	int data_per_part = nnz_train / data_part + 1;
	for (int i = 0; i < nnz_train; i++) {
		int part_temp = 0;
		int weight_temp = (*multi_gpu_part);
		for (int j = 0; j < order; j++) {
			weight_temp /= gpu_number;
			part_temp +=
					(index_train[i / data_per_part][(i % data_per_part) * order + j]
							% gpu_number) * weight_temp;
		}
		(*multi_nnz_train)[part_temp]++;
	}

	for (int i = 0; i < (*multi_gpu_part); i++) {
		(*multi_index_train)[i] = (int *) malloc(
				sizeof(int) * (*multi_nnz_train)[i] * order);
		(*multi_value_train)[i] = (type_of_data *) malloc(
				sizeof(type_of_data) * (*multi_nnz_train)[i]);
	}

	int *count_train = (int *) malloc(sizeof(int) * (*multi_gpu_part));
	memset(count_train, 0, (*multi_gpu_part) * sizeof(int));

	for (int i = 0; i < nnz_train; i++) {
		int part_temp = 0;
		int weight_temp = (*multi_gpu_part);
		for (int j = 0; j < order; j++) {
			weight_temp /= gpu_number;
			part_temp +=
					(index_train[i / data_per_part][(i % data_per_part) * order + j]
							% gpu_number) * weight_temp;
		}
		for (int j = 0; j < order; j++) {
			int quotient_all = dimen[j] / gpu_number;
			int remainder_all = dimen[j] % gpu_number;
			int quotient = index_train[i / data_per_part][(i % data_per_part) * order
					+ j] / gpu_number;
			int remainder = index_train[i / data_per_part][(i % data_per_part) * order
					+ j] % gpu_number;
			if (remainder < remainder_all) {
				(*multi_index_train)[part_temp][count_train[part_temp] * order
						+ j] = remainder * (quotient_all + 1) + quotient;
			} else {
				(*multi_index_train)[part_temp][count_train[part_temp] * order
						+ j] = remainder_all * (quotient_all + 1)
						+ (remainder - remainder_all) * quotient_all + quotient;
			}
		}
		(*multi_value_train)[part_temp][count_train[part_temp]] =
				value_train[i];
		count_train[part_temp] += 1;
	}
	free(count_train);

	*multi_nnz_test = (int *) malloc(sizeof(int) * (*multi_gpu_part));
	*multi_index_test = (int **) malloc(sizeof(int*) * (*multi_gpu_part));
	*multi_value_test = (type_of_data **) malloc(
			sizeof(type_of_data*) * (*multi_gpu_part));
	memset(*multi_nnz_test, 0, (*multi_gpu_part) * sizeof(int));

	for (int i = 0; i < nnz_test; i++) {
		int part_temp = 0;
		int weight_temp = (*multi_gpu_part);
		for (int j = 0; j < order; j++) {
			weight_temp /= gpu_number;
			part_temp += (index_test[i * order + j] % gpu_number) * weight_temp;
		}
		(*multi_nnz_test)[part_temp]++;
	}

	for (int i = 0; i < (*multi_gpu_part); i++) {
		(*multi_index_test)[i] = (int *) malloc(
				sizeof(int) * (*multi_nnz_test)[i] * order);
		(*multi_value_test)[i] = (type_of_data *) malloc(
				sizeof(type_of_data) * (*multi_nnz_test)[i]);
	}

	int *count_test = (int *) malloc(sizeof(int) * (*multi_gpu_part));
	memset(count_test, 0, (*multi_gpu_part) * sizeof(int));

	for (int i = 0; i < nnz_test; i++) {
		int part_temp = 0;
		int weight_temp = (*multi_gpu_part);
		for (int j = 0; j < order; j++) {
			weight_temp /= gpu_number;
			part_temp += (index_test[i * order + j] % gpu_number) * weight_temp;
		}
		for (int j = 0; j < order; j++) {
			int quotient_all = dimen[j] / gpu_number;
			int remainder_all = dimen[j] % gpu_number;
			int quotient = index_test[i * order + j] / gpu_number;
			int remainder = index_test[i * order + j] % gpu_number;
			if (remainder < remainder_all) {
				(*multi_index_test)[part_temp][count_test[part_temp] * order + j] =
						remainder * (quotient_all + 1) + quotient;
			} else {
				(*multi_index_test)[part_temp][count_test[part_temp] * order + j] =
						remainder_all * (quotient_all + 1)
								+ (remainder - remainder_all) * quotient_all
								+ quotient;
			}
		}
		(*multi_value_test)[part_temp][count_test[part_temp]] = value_test[i];
		count_test[part_temp] += 1;
	}
	free(count_test);

}

void Mutil_Cuda_Parameter_Initialization(int order, int gpu_number,
		int multi_gpu_part, int part_per_gpu, int* dimen, int core_length,
		int core_dimen, int *multi_nnz_train, int **multi_index_train_host,
		int ***multi_index_train_device, int *multi_nnz_test,
		int **multi_index_test_host, int ***multi_index_test_device,
		type_of_data **multi_value_train_host,
		type_of_data ***multi_value_train_device,
		type_of_data **multi_value_test_host,
		type_of_data ***multi_value_test_device,
		type_of_data** parameter_a_host, type_of_data* parameter_g_host,
		type_of_data**** parameter_a_device, type_of_data*** parameter_g_device,
		type_of_data**** parameter_a_host_to_device, type_of_data**** g_device,
		type_of_data**** g_host_to_device) {

	*multi_index_train_device = (int **) malloc(sizeof(int*) * multi_gpu_part);
	*multi_value_train_device = (type_of_data **) malloc(
			sizeof(type_of_data*) * multi_gpu_part);

	*multi_index_test_device = (int **) malloc(sizeof(int*) * multi_gpu_part);
	*multi_value_test_device = (type_of_data **) malloc(
			sizeof(type_of_data*) * multi_gpu_part);

	*parameter_a_device = (type_of_data ***) malloc(
			sizeof(type_of_data**) * gpu_number);
	*parameter_a_host_to_device = (type_of_data ***) malloc(
			sizeof(type_of_data**) * gpu_number);

	*parameter_g_device = (type_of_data **) malloc(
			sizeof(type_of_data*) * gpu_number);

	*g_device = (type_of_data ***) malloc(sizeof(type_of_data**) * gpu_number);
	*g_host_to_device = (type_of_data ***) malloc(
			sizeof(type_of_data**) * gpu_number);

#pragma omp parallel for
	for (int i = 0; i < gpu_number; i++) {

		cudaSetDevice(i);

		cudaMalloc((void**) &((*parameter_g_device)[i]),
				sizeof(type_of_data) * core_length);
		cudaMemcpy((*parameter_g_device)[i], parameter_g_host,
				sizeof(type_of_data) * core_length, cudaMemcpyHostToDevice);

		cudaMalloc((void**) &((*parameter_a_device)[i]),
				sizeof(type_of_data*) * order);
		(*parameter_a_host_to_device)[i] = (type_of_data **) malloc(
				sizeof(type_of_data*) * order);

		cudaMalloc((void**) &((*g_device)[i]), sizeof(type_of_data*) * order);
		(*g_host_to_device)[i] = (type_of_data **) malloc(
				sizeof(type_of_data*) * order);

		for (int j = 0; j < order; j++) {

			type_of_data* temp_a;
			cudaMalloc((void**) &temp_a,
					sizeof(type_of_data) * dimen[j] * core_dimen);
			(*parameter_a_host_to_device)[i][j] = temp_a;
			cudaMemcpy(temp_a, parameter_a_host[j],
					sizeof(type_of_data) * dimen[j] * core_dimen,
					cudaMemcpyHostToDevice);

			type_of_data* temp_g;
			cudaMalloc((void**) &temp_g, sizeof(type_of_data) * core_length);
			cudaMemset(temp_g, 0, core_length * sizeof(type_of_data));
			(*g_host_to_device)[i][j] = temp_g;

		}
		cudaMemcpy((*parameter_a_device)[i], (*parameter_a_host_to_device)[i],
				sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);

		cudaMemcpy((*g_device)[i], (*g_host_to_device)[i],
				sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);

		for (int j = 0; j < part_per_gpu; j++) {

			cudaMalloc(
					(void**) &((*multi_index_train_device)[i * part_per_gpu + j]),
					sizeof(int) * multi_nnz_train[i * part_per_gpu + j]
							* order);
			cudaMemcpy((*multi_index_train_device)[i * part_per_gpu + j],
					multi_index_train_host[i * part_per_gpu + j],
					sizeof(int) * multi_nnz_train[i * part_per_gpu + j] * order,
					cudaMemcpyHostToDevice);

			cudaMalloc(
					(void**) &((*multi_value_train_device)[i * part_per_gpu + j]),
					sizeof(type_of_data)
							* multi_nnz_train[i * part_per_gpu + j]);
			cudaMemcpy((*multi_value_train_device)[i * part_per_gpu + j],
					multi_value_train_host[i * part_per_gpu + j],
					sizeof(type_of_data)
							* multi_nnz_train[i * part_per_gpu + j],
					cudaMemcpyHostToDevice);

			cudaMalloc(
					(void**) &((*multi_index_test_device)[i * part_per_gpu + j]),
					sizeof(int) * multi_nnz_test[i * part_per_gpu + j]
							* order);
			cudaMemcpy((*multi_index_test_device)[i * part_per_gpu + j],
					multi_index_test_host[i * part_per_gpu + j],
					sizeof(int) * multi_nnz_test[i * part_per_gpu + j] * order,
					cudaMemcpyHostToDevice);

			cudaMalloc(
					(void**) &((*multi_value_test_device)[i * part_per_gpu + j]),
					sizeof(type_of_data)
							* multi_nnz_test[i * part_per_gpu + j]);
			cudaMemcpy((*multi_value_test_device)[i * part_per_gpu + j],
					multi_value_test_host[i * part_per_gpu + j],
					sizeof(type_of_data) * multi_nnz_test[i * part_per_gpu + j],
					cudaMemcpyHostToDevice);
		}
	}

}

void Select_Best_Result(type_of_data* train_rmse, type_of_data* train_mae,
type_of_data* test_rmse, type_of_data* test_mae,
type_of_data* best_train_rmse, type_of_data* best_train_mae,
type_of_data* best_test_rmse, type_of_data* best_test_mae) {

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
