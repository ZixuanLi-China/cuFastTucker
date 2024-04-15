#include <stdio.h>
#include "tools.h"
#include "kernel.h"

//using namespace std;

type_of_data learn_alpha_a;
type_of_data learn_beta_a;
type_of_data lambda_a;

type_of_data learn_alpha_g;
type_of_data learn_beta_g;
type_of_data lambda_g;

int iter_number;

type_of_data learn_rate_a;
type_of_data learn_rate_g;

type_of_data train_rmse;
type_of_data train_mae;

type_of_data test_rmse;
type_of_data test_mae;

type_of_data best_train_rmse;
type_of_data best_train_mae;

type_of_data best_test_rmse;
type_of_data best_test_mae;

char *InputPath_train;
char *InputPath_test;

int order;
int core_length;

int *dimen;
int core_dimen;

double data_norm;

type_of_data rmse_train;
type_of_data rmse_test;

int nnz_train;
type_of_data **value_train_host;
type_of_data **value_train_device;
type_of_data **value_train_host_to_device;
int **index_train_host;
int **index_train_device;
int **index_train_host_to_device;

int nnz_test;
type_of_data *value_test_host;
type_of_data *value_test_device;
int *index_test_host;
int *index_test_device;

type_of_data **parameter_a_host;
type_of_data **parameter_a_host_to_device;
type_of_data **parameter_a_device;

type_of_data *parameter_g_host;
type_of_data *parameter_g_device;

type_of_data **g_device;
type_of_data **g_host_to_device;

double time_spend = 0.0;
double start_time;
double mid_time;
double stop_time;

int main(int argc, char *argv[]) {

	if (argc == 12) {

		InputPath_train = argv[1];
		InputPath_test = argv[2];

		order = atoi(argv[3]);
		core_dimen = atoi(argv[4]);

		iter_number = atoi(argv[5]);

		learn_alpha_a = atof(argv[6]);
		learn_beta_a = atof(argv[7]);
		lambda_a = atof(argv[8]);

		learn_alpha_g = atof(argv[9]);
		learn_beta_g = atof(argv[10]);
		lambda_g = atof(argv[11]);

		core_length = 1;
		for (int i = 0; i < order; i++) {
			core_length *= core_dimen;
		}

	}
	printf("learn_alpha_a:%f\tlearn_beta_a:%f\tlambda_a:%f\n", learn_alpha_a,
			learn_beta_a, lambda_a);
	printf("learn_alpha_g:%f\tlearn_beta_g:%f\tlambda_g:%f\n", learn_alpha_g,
			learn_beta_g, lambda_g);
	Getting_Input(InputPath_train, InputPath_test, order, &dimen, &nnz_train,
			&nnz_test, &index_train_host, &value_train_host, &index_test_host,
			&value_test_host, &data_norm);
	printf("nnz_train:\t%d\n", nnz_train);
	printf("nnz_test:\t%d\n", nnz_test);
	for (int i = 0; i < order; i++) {
		printf("order %d:\t%d\n", i + 1, dimen[i]);
	}
	printf("data_norm:\t%f\n", data_norm);

	Parameter_Initialization(order, core_length, core_dimen, dimen, data_norm,
			&parameter_a_host, &parameter_g_host);

	Cuda_Parameter_Initialization(order, core_length, core_dimen, dimen,
			nnz_train, value_train_host, &value_train_device,
			&value_train_host_to_device, index_train_host, &index_train_device,
			&index_train_host_to_device, nnz_test, value_test_host,
			&value_test_device, index_test_host, &index_test_device,
			parameter_a_host, parameter_g_host, &parameter_a_device,
			&parameter_g_device, &parameter_a_host_to_device, &g_device,
			&g_host_to_device);

	GET_RMSE_AND_MAE(order, core_length, core_dimen, parameter_a_device,
			parameter_g_device, nnz_train, value_train_host_to_device,
			index_train_host_to_device, g_device, &best_train_rmse,
			&best_train_mae);

	GET_RMSE_AND_MAE(order, core_length, core_dimen, parameter_a_device,
			parameter_g_device, nnz_test, value_test_device, index_test_device,
			g_device, &best_test_rmse, &best_test_mae);

	printf(
			"initial:\ttrain rmse:%f\ttest rmse:%f\ttrain mae:%f\ttest mae:%f\t\n",
			best_train_rmse, best_test_rmse, best_train_mae, best_test_mae);
	printf(
			"iter\ttrain rmse\ttest rmse\ttrain mae\ttest mae\tfactor time\tcore time\ttotal time\tcumulative time\n");

	for (int iter = 0; iter < iter_number; iter++) {

		learn_rate_a = learn_alpha_a / (1 + learn_beta_a * pow(iter, 1.5));
		learn_rate_g = learn_alpha_g / (1 + learn_beta_g * pow(iter, 1.5));

		start_time = Seconds();

		Update_Parameter_A(order, core_length, core_dimen, parameter_a_device,
				nnz_train, value_train_host_to_device,
				index_train_host_to_device, g_device, learn_rate_a, lambda_a);

		mid_time = Seconds();

		Update_Parameter_G_Batch(order, core_length, core_dimen,
				parameter_a_device, parameter_g_device, nnz_train,
				value_train_host_to_device, index_train_host_to_device,
				learn_rate_g, lambda_g);

		stop_time = Seconds();
		time_spend += stop_time - start_time;

		GET_RMSE_AND_MAE(order, core_length, core_dimen, parameter_a_device,
				parameter_g_device, nnz_train, value_train_host_to_device,
				index_train_host_to_device, g_device, &train_rmse, &train_mae);

		GET_RMSE_AND_MAE(order, core_length, core_dimen, parameter_a_device,
				parameter_g_device, nnz_test, value_test_device,
				index_test_device, g_device, &test_rmse, &test_mae);

		Select_Best_Result(&train_rmse, &train_mae, &test_rmse, &test_mae,
				&best_train_rmse, &best_train_mae, &best_test_rmse,
				&best_test_mae);
		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", iter, train_rmse,
				test_rmse, train_mae, test_mae, mid_time - start_time,
				stop_time - mid_time, stop_time - start_time, time_spend);

	}

	printf("best:\ttrain rmse:%f\ttest rmse:%f\ttrain mae:%f\ttest mae:%f\t\n",
			best_train_rmse, best_test_rmse, best_train_mae, best_test_mae);

	return 0;
}
