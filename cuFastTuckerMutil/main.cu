#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "tools.h"
#include "kernel.h"

#define type_of_data float

using namespace std;

int model;

type_of_data learn_alpha_a;
type_of_data learn_beta_a;
type_of_data lambda_a;

type_of_data learn_alpha_b;
type_of_data learn_beta_b;
type_of_data lambda_b;

int iter_number;

type_of_data learn_rate_a;
type_of_data learn_rate_b;

type_of_data train_rmse;
type_of_data train_mae;

type_of_data test_rmse;
type_of_data test_mae;

type_of_data best_train_rmse;
type_of_data best_train_mae;

type_of_data best_test_rmse;
type_of_data best_test_mae;

char* InputPath_train;
char* InputPath_test;

int order;
int core_kernel;
int core_length;

int *dimen;
int core_dimen;

double data_norm;

int nnz_train;
int **index_train_host;
int *index_train_device;
type_of_data *value_train_host;
type_of_data *value_train_device;
int *multi_nnz_train;
int **multi_index_train_host;
int **multi_index_train_device;
type_of_data **multi_value_train_host;
type_of_data **multi_value_train_device;

int nnz_test;
int *index_test_host;
int *index_test_device;
type_of_data *value_test_host;
type_of_data *value_test_device;
int *multi_nnz_test;
int **multi_index_test_host;
int **multi_index_test_device;
type_of_data **multi_value_test_host;
type_of_data **multi_value_test_device;

int gpu_number;
int multi_data_part;
int part_per_gpu;

int **multi_part;
int **multi_start;

type_of_data** parameter_a_host;
type_of_data*** parameter_a_host_to_device;
type_of_data*** parameter_a_device;

type_of_data** parameter_b_host;
type_of_data*** parameter_b_host_to_device;
type_of_data*** parameter_b_device;

double time_spend = 0.0;
double start_time;
double mid_time;
double stop_time;
double memcpy_time_a;
double memcpy_time_b;

int main(int argc, char* argv[]) {

	if (argc == 14) {

		InputPath_train = argv[1];
		InputPath_test = argv[2];
		core_kernel = atoi(argv[3]);
		order = atoi(argv[4]);
		core_dimen = atoi(argv[5]);

		iter_number = atoi(argv[6]);

		learn_alpha_a = atof(argv[7]);
		learn_beta_a = atof(argv[8]);
		lambda_a = atof(argv[9]);

		learn_alpha_b = atof(argv[10]);
		learn_beta_b = atof(argv[11]);
		lambda_b = atof(argv[12]);

		model = atoi(argv[13]);

		core_length = 1;
		for (int i = 0; i < order; i++) {
			core_length *= core_dimen;
		}

	}

	cudaGetDeviceCount(&gpu_number);
	
	printf("number of GPUs:%d\n", gpu_number);

	Getting_Input(InputPath_train, InputPath_test, order, &dimen, &nnz_train,
			&nnz_test, &index_train_host, &value_train_host, &index_test_host,
			&value_test_host, &data_norm);

	printf("nnz_train:\t%d\n", nnz_train);
	printf("nnz_test:\t%d\n", nnz_test);

	for (int i = 0; i < order; i++) {
		printf("order %d:\t%d\n", i + 1, dimen[i]);
	}
	printf("data_norm:\t%f\n", data_norm);

	Multi_Parameter_Initialization(order, gpu_number, &multi_data_part,
			&part_per_gpu, core_kernel, core_length, core_dimen, dimen,
			data_norm, &parameter_a_host, &parameter_b_host, nnz_train,
			index_train_host, value_train_host, nnz_test, index_test_host,
			value_test_host, &multi_nnz_train, &multi_index_train_host,
			&multi_value_train_host, &multi_nnz_test, &multi_index_test_host,
			&multi_value_test_host, &multi_part, &multi_start);

	Mutil_Cuda_Parameter_Initialization(order, gpu_number, multi_data_part,
			part_per_gpu, dimen, core_kernel, core_dimen, multi_nnz_train,
			multi_index_train_host, &multi_index_train_device, multi_nnz_test,
			multi_index_test_host, &multi_index_test_device,
			multi_value_train_host, &multi_value_train_device,
			multi_value_test_host, &multi_value_test_device, parameter_a_host,
			parameter_b_host, &parameter_a_device, &parameter_b_device,
			&parameter_a_host_to_device, &parameter_b_host_to_device);

	Mutil_GET_RMSE_AND_MAE(order, gpu_number, multi_data_part, part_per_gpu,
			core_kernel, core_dimen, multi_nnz_train, multi_value_train_device,
			multi_index_train_device, parameter_a_device, parameter_b_device,
			&best_train_rmse, &best_train_mae);
	Mutil_GET_RMSE_AND_MAE(order, gpu_number, multi_data_part, part_per_gpu,
			core_kernel, core_dimen, multi_nnz_test, multi_value_test_device,
			multi_index_test_device, parameter_a_device, parameter_b_device,
			&best_test_rmse, &best_test_mae);

	printf(
			"initial:\ttrain rmse:%f\ttest rmse:%f\ttrain mae:%f\ttest mae:%f\t\n",
			best_train_rmse, best_test_rmse, best_train_mae, best_test_mae);
	printf(
			"iter\ttrain rmse\ttest rmse\ttrain mae\ttest mae\tfactor time\tfactor memcpy\tcore time\tcore memcpy\ttotal time\tcumulative time\n");

	for (int iter = 0; iter < iter_number; iter++) {

		learn_rate_a = learn_alpha_a / (1 + learn_beta_a * pow(iter, 1.5));
		learn_rate_b = learn_alpha_b / (1 + learn_beta_b * pow(iter, 1.5));

		start_time = Seconds();

		Mutil_Update_Parameter_A(order, gpu_number, part_per_gpu, core_kernel,
				core_dimen, multi_nnz_train, multi_value_train_device,
				multi_index_train_device, parameter_a_device,
				parameter_a_host_to_device, parameter_b_device, multi_start,
				multi_part, learn_rate_a, lambda_a, &memcpy_time_a, model);

		mid_time = Seconds();

		Mutil_Update_Parameter_B(order, gpu_number, nnz_train, part_per_gpu,
				core_kernel, core_dimen, multi_nnz_train,
				multi_value_train_device, multi_index_train_device,
				parameter_a_device, parameter_b_device,
				parameter_b_host_to_device, learn_rate_b, lambda_b,
				&memcpy_time_b, model);

		stop_time = Seconds();
		time_spend += stop_time - start_time;

		Mutil_GET_RMSE_AND_MAE(order, gpu_number, multi_data_part, part_per_gpu,
				core_kernel, core_dimen, multi_nnz_train,
				multi_value_train_device, multi_index_train_device,
				parameter_a_device, parameter_b_device, &train_rmse,
				&train_mae);
		Mutil_GET_RMSE_AND_MAE(order, gpu_number, multi_data_part, part_per_gpu,
				core_kernel, core_dimen, multi_nnz_test,
				multi_value_test_device, multi_index_test_device,
				parameter_a_device, parameter_b_device, &test_rmse, &test_mae);
		Select_Best_Result(&train_rmse, &train_mae, &test_rmse, &test_mae,
				&best_train_rmse, &best_train_mae, &best_test_rmse,
				&best_test_mae);
		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", iter, train_rmse,
				test_rmse, train_mae, test_mae, mid_time - start_time,
				memcpy_time_a, stop_time - mid_time, memcpy_time_b,
				stop_time - start_time, time_spend);
	}
	printf("best:\ttrain rmse:%f\ttest rmse:%f\ttrain mae:%f\ttest mae:%f\t\n",
			best_train_rmse, best_test_rmse, best_train_mae, best_test_mae);
	return 0;
}
