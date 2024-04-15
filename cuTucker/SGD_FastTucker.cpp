#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <armadillo>
#include <string.h>
#include <omp.h>

using namespace std;
using namespace arma;

double learn_alpha_a = 0.0005;
double learn_beta_a = 0.05;
double lambda_a = 0.01;
double learn_alpha_b = 0.001;
double learn_beta_b = 0.05;
double lambda_b = 0.01;

double learn_rate_a;
double learn_rate_b;

char *InputPath_train;
char *InputPath_test;

int order;
int *dimen;
int nnz_train, nnz_test;
double *value_train, *value_test;
int *index_train, *index_test;
double data_norm;

int core_count;
int *core_dimen;
int core_kernel;
int threads = 20;

mat *parameter_a;
mat *parameter_b;

double train_rmse, rmse_test;

inline double seconds() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void Getting_Input() {

	dimen = (int*) malloc(sizeof(int) * order);
	for (int i = 0; i < order; i++) {
		dimen[i] = 0;
	}
	data_norm = 0.0;

	char tmp[1024];

	FILE *train_file_count = fopen(InputPath_train, "r");
	FILE *train_file = fopen(InputPath_train, "r");

	FILE *test_file_count = fopen(InputPath_test, "r");
	FILE *test_file = fopen(InputPath_test, "r");

	nnz_train = 0;
	nnz_test = 0;

	while (fgets(tmp, 1024, train_file_count)) {
		nnz_train++;
	}

	while (fgets(tmp, 1024, test_file_count)) {
		nnz_test++;
	}

	index_train = (int*) malloc(sizeof(int) * nnz_train * order);
	value_train = (double*) malloc(sizeof(double) * nnz_train);

	index_test = (int*) malloc(sizeof(int) * nnz_test * order);
	value_test = (double*) malloc(sizeof(double) * nnz_test);

	char *p;

	for (int i = 0; i < nnz_train; i++) {
		fgets(tmp, 1024, train_file);
		p = strtok(tmp, "\t");
		for (int j = 0; j < order; j++) {
			int int_temp = atoi(p);
			index_train[i * order + j] = int_temp - 1;
			p = strtok(NULL, "\t");
			if (int_temp > dimen[j]) {
				dimen[j] = int_temp;
			}
		}
		double double_temp = atof(p);
		value_train[i] = double_temp;
		data_norm += double_temp * double_temp;
	}
	data_norm = sqrt(data_norm / nnz_train);

	for (int i = 0; i < nnz_test; i++) {
		fgets(tmp, 1024, test_file);
		p = strtok(tmp, "\t");
		for (int j = 0; j < order; j++) {
			int int_temp = atoi(p);
			index_test[i * order + j] = int_temp - 1;
			p = strtok(NULL, "\t");
			if (int_temp > dimen[j]) {
				dimen[j] = int_temp;
			}
		}
		value_test[i] = atof(p);
	}
}

void Parameter_Initialization() {

	parameter_a = new mat[order];
	parameter_b = new mat[order];

	for (int i = 0; i < order; i++) {

		parameter_a[i] = pow(data_norm / core_count, 1.0 / order)
				* (0.5 + randu(dimen[i], core_dimen[i]));

		parameter_b[i] = pow(1.0 / core_kernel, 1.0 / order)
				* (0.5 + randu(core_dimen[i], core_kernel));

	}

}

void Update_Parameter_A() {

	for (int update_order_index = 0; update_order_index < order;
			update_order_index++) {

#pragma omp parallel for
		for (int nnz_index = 0; nnz_index < nnz_train; nnz_index++) {

			mat e = ones(core_kernel, 1);
			mat d;
			mat temp;
			mat x_r;
			for (int order_index = 0; order_index < order; order_index++) {
				if (order_index != update_order_index) {
					temp =
							parameter_b[order_index].t()
									* parameter_a[order_index].row(
											index_train[nnz_index * order
													+ order_index]).t();
					e = e % temp;
				}
			}
			d = parameter_b[update_order_index] * e;
			x_r = parameter_a[update_order_index].row(
					index_train[nnz_index * order + update_order_index]) * d;
			x_r(0, 0) -= value_train[nnz_index];
			parameter_a[update_order_index].row(
					index_train[nnz_index * order + update_order_index]) -=
					learn_rate_a
							* (x_r * d.t()
									+ lambda_a
											* parameter_a[update_order_index].row(
													index_train[nnz_index
															* order
															+ update_order_index]));
		}
	}
}

void Update_Parameter_B() {

	omp_set_num_threads(threads);
	mat *gard = new mat[threads];

	for (int update_order_index = 0; update_order_index < order;
			update_order_index++) {

		for (int i = 0; i < threads; i++) {
			gard[i] = ones(core_dimen[update_order_index], core_kernel);
		}

#pragma omp parallel for
		for (int nnz_index = 0; nnz_index < nnz_train; nnz_index++) {

			mat e = ones(core_kernel, 1);
			mat temp;
			mat x_r;
			mat gard_temp;
			for (int order_index = 0; order_index < order; order_index++) {
				if (order_index != update_order_index) {
					temp =
							parameter_b[order_index].t()
									* parameter_a[order_index].row(
											index_train[nnz_index * order
													+ order_index]).t();
					e = e % temp;
				}
			}
			x_r = parameter_a[update_order_index].row(
					index_train[nnz_index * order + update_order_index])
					* parameter_b[update_order_index] * e;
			x_r(0, 0) -= value_train[nnz_index];
			gard_temp = parameter_a[update_order_index].row(
					index_train[nnz_index * order + update_order_index]).t()
					* e.t() * x_r(0, 0)
					+ lambda_b * parameter_b[update_order_index];
			int thread_id = omp_get_thread_num();
			gard[thread_id] += gard_temp;

		}
		for (int i = 0; i < threads; i++) {
			gard[0] += gard[i];
		}
		parameter_b[update_order_index] -= learn_rate_b * (gard[0] / nnz_train);

	}
	delete[] gard;
}

double Get_RMSE_Train() {

	double return_rmse = 0.0;
#pragma omp parallel for reduction(+:return_rmse)
	for (int nnz_index = 0; nnz_index < nnz_train; nnz_index++) {
		mat e = ones(core_kernel, 1);
		mat d;
		mat temp;
		mat x_r;
		for (int order_index = 0; order_index < order; order_index++) {
			if (order_index != 0) {
				temp =
						parameter_b[order_index].t()
								* parameter_a[order_index].row(
										index_train[nnz_index * order
												+ order_index]).t();
				e = e % temp;
			}
		}
		d = parameter_b[0] * e;
		x_r = parameter_a[0].row(index_train[nnz_index * order]) * d;
		x_r(0, 0) -= value_train[nnz_index];
		return_rmse += x_r(0, 0) * x_r(0, 0);
	}
	return sqrt(return_rmse / nnz_train);
}

double Get_RMSE_Test() {

	double return_rmse = 0.0;
#pragma omp parallel for reduction(+:return_rmse)
	for (int nnz_index = 0; nnz_index < nnz_test; nnz_index++) {
		mat e = ones(core_kernel, 1);
		mat d;
		mat temp;
		mat x_r;
		for (int order_index = 0; order_index < order; order_index++) {
			if (order_index != 0) {
				temp =
						parameter_b[order_index].t()
								* parameter_a[order_index].row(
										index_test[nnz_index * order
												+ order_index]).t();
				e = e % temp;
			}
		}
		d = parameter_b[0] * e;
		x_r = parameter_a[0].row(index_test[nnz_index * order]) * d;
		x_r(0, 0) -= value_test[nnz_index];
		return_rmse += x_r(0, 0) * x_r(0, 0);
	}
	return sqrt(return_rmse / nnz_test);
}

double Get_MAE_Train() {

	double return_mae = 0.0;
#pragma omp parallel for reduction(+:return_mae)
	for (int nnz_index = 0; nnz_index < nnz_train; nnz_index++) {
		mat e = ones(core_kernel, 1);
		mat d;
		mat temp;
		mat x_r;
		for (int order_index = 0; order_index < order; order_index++) {
			if (order_index != 0) {
				temp =
						parameter_b[order_index].t()
								* parameter_a[order_index].row(
										index_train[nnz_index * order
												+ order_index]).t();
				e = e % temp;
			}
		}
		d = parameter_b[0] * e;
		x_r = parameter_a[0].row(index_train[nnz_index * order]) * d;
		x_r(0, 0) -= value_train[nnz_index];
		return_mae += abs(x_r(0, 0));
	}
	return return_mae / nnz_train;
}

double Get_MAE_Test() {

	double return_mae = 0.0;
#pragma omp parallel for reduction(+:return_mae)
	for (int nnz_index = 0; nnz_index < nnz_test; nnz_index++) {
		mat e = ones(core_kernel, 1);
		mat d;
		mat temp;
		mat x_r;
		for (int order_index = 0; order_index < order; order_index++) {
			if (order_index != 0) {
				temp =
						parameter_b[order_index].t()
								* parameter_a[order_index].row(
										index_test[nnz_index * order
												+ order_index]).t();
				e = e % temp;
			}
		}
		d = parameter_b[0] * e;
		x_r = parameter_a[0].row(index_test[nnz_index * order]) * d;
		x_r(0, 0) -= value_test[nnz_index];
		return_mae += abs(x_r(0, 0));
	}
	return return_mae / nnz_test;
}

int main(int argc, char *argv[]) {

	if (argc == 5 + atoi(argv[4])) {

		InputPath_train = argv[1];
		InputPath_test = argv[2];
		core_kernel = atoi(argv[3]);
		order = atoi(argv[4]);
		core_count = 1;
		core_dimen = (int*) malloc(sizeof(int) * order);
		for (int i = 0; i < order; i++) {
			core_dimen[i] = atoi(argv[5 + i]);
			core_count *= atoi(argv[5 + i]);
		}

	}
	Getting_Input();
	Parameter_Initialization();
	printf("nnz_train:\t%d\n", nnz_train);
	printf("nnz_test:\t%d\n", nnz_test);
	for (int i = 0; i < order; i++) {
		printf("order %d:\t%d\n", i + 1, dimen[i]);
	}
	printf(
			"initial:\ttrain rmse:%f\ttest rmse:%f\ttrain mae:%f\ttest mae:%f\t\n",
			Get_RMSE_Train(), Get_RMSE_Test(), Get_MAE_Train(), Get_MAE_Test());

	double time_spend = 0.0, start_time, mid_time, end_time;
	printf(
			"it\ttrain rmse\ttest rmse\ttrain mae\ttest mae\tfactor time\tcore time\ttrain time\n");
	for (int i = 0; i < 50; i++) {
		learn_rate_a = learn_alpha_a / (1 + learn_beta_a * pow(i, 1.5));
		learn_rate_b = learn_alpha_b / (1 + learn_beta_b * pow(i, 1.5));
		start_time = seconds();
		Update_Parameter_A();
		mid_time = seconds();
		Update_Parameter_B();
		end_time = seconds();
		time_spend += end_time - start_time;
		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", i, Get_RMSE_Train(),
				Get_RMSE_Test(), Get_MAE_Train(), Get_MAE_Test(),
				mid_time - start_time, end_time - mid_time, time_spend);
	}
	return 0;
}
