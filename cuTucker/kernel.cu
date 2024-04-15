#include <cublas_v2.h>
#include "parameter.h"

__global__ void Structure_Core_Tensor(const int order, const int core_length,
		const int core_dimen, type_of_data *parameter_g, type_of_data **g) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int order_index = 0; order_index < order; order_index++) {
		int length = core_length / core_dimen;
		for (int index = worker_id; index < length; index += workers) {

			int g_index = 0;
			int weight = core_length;
			int parameter_b_index = index;
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				weight /= core_dimen;
				if (inner_order_index != order_index) {
					g_index += (parameter_b_index % core_dimen) * weight;
					parameter_b_index /= core_dimen;
				} else {
					g_index += lane_id * weight;
				}
			}
			g[order_index][index * core_dimen + lane_id] = parameter_g[g_index];
		}
	}
}

__global__ void Update_Parameter_A_SGD(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a, const int nnz,
		type_of_data *value, int *index, type_of_data **g,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		const int update_order) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		int order_index = nnz_index * order;
		type_of_data gs = 0.0;
		for (int g_index = 0; g_index < core_length / core_dimen; g_index++) {
			int parameter_a_index = g_index;
			type_of_data s = 1.0;
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != update_order) {
					s *= parameter_a[inner_order_index][index[order_index
							+ inner_order_index] * core_dimen
							+ parameter_a_index % core_dimen];
					parameter_a_index /= core_dimen;
				}

			}
			gs += s * g[update_order][g_index * core_dimen + lane_id];
		}

		type_of_data p_a_temp = parameter_a[update_order][index[order_index
				+ update_order] * core_dimen + lane_id];

		type_of_data p_a_gs = p_a_temp * gs;

		if (core_dimen == 4) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0, 4);
		} else if (core_dimen == 8) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0, 8);
		} else if (core_dimen == 16) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0, 16);
		} else if (core_dimen == 32) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 16);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0);
		}

		parameter_a[update_order][index[order_index + update_order] * core_dimen
				+ lane_id] -=
				learn_rate_a
						* (-value[nnz_index] * gs + p_a_gs * gs
								+ lambda_a
										* parameter_a[update_order][index[order_index
												+ update_order] * core_dimen
												+ lane_id]);

	}
}

void Update_Parameter_A(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a, const int nnz,
		type_of_data **value, int **index, type_of_data **g,
		const type_of_data learn_rate_a, const type_of_data lambda_a) {

	int data_per_part = nnz / data_part + 1;
	for (int update_order = 0; update_order < order; update_order++) {

		for (int i = 0; i < data_part - 1; i++) {
			Update_Parameter_A_SGD<<<grid_size,
			block_size>>>(order, core_length, core_dimen, parameter_a,
					data_per_part, value[i], index[i], g, learn_rate_a,
					lambda_a, update_order);
			cudaDeviceSynchronize();
		}
		Update_Parameter_A_SGD<<<grid_size,
		block_size>>>(order, core_length, core_dimen, parameter_a,
				nnz - (data_part - 1) * data_per_part, value[data_part - 1],
				index[data_part - 1], g, learn_rate_a, lambda_a, update_order);
		cudaDeviceSynchronize();

	}

}

__global__ void Update_Parameter_G_Gradient(const int order,
		const int core_length, const int core_dimen, type_of_data **parameter_a,
		type_of_data *parameter_g, const int nnz, const type_of_data *value,
		const int *index, type_of_data *g_sum,
		type_of_data *h_shared) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		int order_index = nnz_index * order;
		type_of_data x_r = 0.0;
		for (int g_index = 0; g_index < core_length / core_dimen; g_index++) {
			int parameter_a_and_b_index = g_index;
			type_of_data h = parameter_a[0][index[order_index] * core_dimen
					+ lane_id];
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != 0) {
					h *= parameter_a[inner_order_index][index[order_index
							+ inner_order_index] * core_dimen
							+ parameter_a_and_b_index % core_dimen];
					parameter_a_and_b_index /= core_dimen;
				}
			}
			h_shared[(nnz_index % sm) * g_index * core_dimen + lane_id] = h;
			x_r += h * parameter_g[g_index * core_dimen + lane_id];
		}

		if (core_dimen == 4) {
			x_r += __shfl_down_sync(mask, x_r, 2);
			x_r += __shfl_down_sync(mask, x_r, 1);
			x_r = __shfl_sync(mask, x_r, 0, 4);
		} else if (core_dimen == 8) {
			x_r += __shfl_down_sync(mask, x_r, 4);
			x_r += __shfl_down_sync(mask, x_r, 2);
			x_r += __shfl_down_sync(mask, x_r, 1);
			x_r = __shfl_sync(mask, x_r, 0, 8);
		} else if (core_dimen == 16) {
			x_r += __shfl_down_sync(mask, x_r, 8);
			x_r += __shfl_down_sync(mask, x_r, 4);
			x_r += __shfl_down_sync(mask, x_r, 2);
			x_r += __shfl_down_sync(mask, x_r, 1);
			x_r = __shfl_sync(mask, x_r, 0, 16);
		} else if (core_dimen == 32) {
			x_r += __shfl_down_sync(mask, x_r, 16);
			x_r += __shfl_down_sync(mask, x_r, 8);
			x_r += __shfl_down_sync(mask, x_r, 4);
			x_r += __shfl_down_sync(mask, x_r, 2);
			x_r += __shfl_down_sync(mask, x_r, 1);
			x_r = __shfl_sync(mask, x_r, 0);
		}

		x_r -= value[nnz_index];

		for (int g_index = 0; g_index < core_length / core_dimen; g_index++) {
			atomicAdd(
					&g_sum[(nnz_index % sum_size) * core_length
							+ g_index * core_dimen + lane_id],
					x_r
							* h_shared[(nnz_index % sm) * g_index * core_dimen
									+ lane_id]);
		}

	}
}

__global__ void Parameter_G_Gradient_Sum(const int order, const int core_length,
		const int core_dimen, const int nnz,
		type_of_data *g_sum, type_of_data *g_grad) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int g_index = worker_id; g_index < core_length / core_dimen; g_index +=
			workers) {
		for (int sum_size_index = 0; sum_size_index < sum_size;
				sum_size_index++) {
			g_grad[g_index * core_dimen + lane_id] += g_sum[sum_size_index
					* core_length + g_index * core_dimen + lane_id];
		}
		g_grad[g_index * core_dimen + lane_id] /= nnz;
	}
}

__global__ void Update_Parameter_G(const int order, const int core_length,
		const int core_dimen, type_of_data *parameter_g, type_of_data *g_grad,
		const type_of_data learn_rate_g, const type_of_data lambda_g) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int g_index = worker_id; g_index < core_length / core_dimen; g_index +=
			workers) {
		parameter_g[g_index * core_dimen + lane_id] -=
				learn_rate_g
						* (g_grad[g_index * core_dimen + lane_id]
								+ lambda_g
										* parameter_g[g_index * core_dimen
												+ lane_id]);

	}
}

void Update_Parameter_G_Batch(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data *parameter_g, const int nnz, type_of_data **value,
		int **index, const type_of_data learn_rate_g,
		const type_of_data lambda_g) {

	type_of_data *g_sum;
	type_of_data *g_grad;
	cudaMalloc((void**) &g_sum, sum_size * core_length * sizeof(type_of_data));
	cudaMalloc((void**) &g_grad, core_length * sizeof(type_of_data));
	cudaMemset(g_sum, 0, sum_size * core_length * sizeof(type_of_data));
	cudaMemset(g_grad, 0, core_length * sizeof(type_of_data));

	int data_per_part = nnz / data_part + 1;

	type_of_data *h_shared;
	cudaMalloc((void**) &h_shared, sm * core_length * sizeof(type_of_data));
	cudaMemset(h_shared, 0, sm * core_length * sizeof(type_of_data));
	for (int i = 0; i < data_part - 1; i++) {
		Update_Parameter_G_Gradient<<<grid_size, block_size>>>(order,
				core_length, core_dimen, parameter_a, parameter_g,
				data_per_part, value[i], index[i], g_sum, h_shared);
		cudaDeviceSynchronize();
	}
	Update_Parameter_G_Gradient<<<
	grid_size, block_size>>>(order, core_length, core_dimen, parameter_a,
			parameter_g, nnz - (data_part - 1) * data_per_part,
			value[data_part - 1], index[data_part - 1], g_sum, h_shared);
	cudaDeviceSynchronize();
	cudaFree(h_shared);

	Parameter_G_Gradient_Sum<<<core_length / (block_size / core_dimen) + 1,
	block_size>>>(order, core_length, core_dimen, nnz, g_sum, g_grad);
	cudaDeviceSynchronize();
	Update_Parameter_G<<<core_length / (block_size / core_dimen) + 1,
	block_size>>>(order, core_length, core_dimen, parameter_g, g_grad,
			learn_rate_g, lambda_g);
	cudaDeviceSynchronize();

	cudaFree(g_sum);
	cudaFree(g_grad);

}

__global__ void RMSE_AND_MAE(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a, const int nnz,
		const type_of_data *value, const int *index, type_of_data **g,
		type_of_data *rmse, type_of_data *mae) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		int order_index = nnz_index * order;
		type_of_data gs = 0.0;
		for (int g_index = 0; g_index < core_length / core_dimen; g_index++) {
			int parameter_a_index = g_index;
			type_of_data s = 1.0;
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != 0) {
					s *= parameter_a[inner_order_index][index[order_index
							+ inner_order_index] * core_dimen
							+ parameter_a_index % core_dimen];
					parameter_a_index /= core_dimen;
				}

			}
			gs += s * g[0][g_index * core_dimen + lane_id];
		}

		type_of_data p_a_temp = parameter_a[0][index[order_index] * core_dimen
				+ lane_id];

		type_of_data p_a_gs = p_a_temp * gs;

		if (core_dimen == 4) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0, 4);
		} else if (core_dimen == 8) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0, 8);
		} else if (core_dimen == 16) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0, 16);
		} else if (core_dimen == 32) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 16);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0);
		}

		p_a_gs -= value[nnz_index];

		if (lane_id == 0) {
			atomicAdd(&rmse[nnz_index % error_size], p_a_gs * p_a_gs);
			atomicAdd(&mae[nnz_index % error_size], abs(p_a_gs));
		}

	}
}

void GET_RMSE_AND_MAE(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data *parameter_g, const int nnz, const type_of_data *value,
		const int *index, type_of_data **g, type_of_data *rmse,
		type_of_data *mae) {

	Structure_Core_Tensor<<<core_length / (block_size / core_dimen) + 1,
	block_size>>>(order, core_length, core_dimen, parameter_g, g);
	cudaDeviceSynchronize();

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse, error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0, error_size * sizeof(type_of_data));

	RMSE_AND_MAE<<<nnz / (block_size / core_dimen) + 1, block_size>>>(order,
			core_length, core_dimen, parameter_a, nnz, value, index, g,
			errors_rmse, errors_mae);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));
	type_of_data *mae_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();
	cublasSasum(handle_mae, error_size, errors_mae, 1, mae_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	*mae = (*mae_sum) / nnz;
	cudaFree(errors_rmse);
	cudaFree(errors_mae);
	cublasDestroy(handle_rmse);
	cublasDestroy(handle_mae);
	free(rmse_sum);
	free(mae_sum);

}

void GET_RMSE_AND_MAE(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data *parameter_g, const int nnz, type_of_data **value,
		int **index, type_of_data **g, type_of_data *rmse,
		type_of_data *mae) {

	Structure_Core_Tensor<<<core_length / (block_size / core_dimen) + 1,
	block_size>>>(order, core_length, core_dimen, parameter_g, g);
	cudaDeviceSynchronize();

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse, error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0, error_size * sizeof(type_of_data));

	int data_per_part = nnz / data_part + 1;
	for (int i = 0; i < data_part - 1; i++) {
		RMSE_AND_MAE<<<data_per_part / (block_size / core_dimen) + 1,
		block_size>>>(order, core_length, core_dimen, parameter_a,
				data_per_part, value[i], index[i], g, errors_rmse, errors_mae);
		cudaDeviceSynchronize();
	}

	RMSE_AND_MAE<<<data_per_part / (block_size / core_dimen) + 1, block_size>>>(
			order, core_length, core_dimen, parameter_a,
			nnz - (data_part - 1) * data_per_part, value[data_part - 1],
			index[data_part - 1], g, errors_rmse, errors_mae);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));
	type_of_data *mae_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();
	cublasSasum(handle_mae, error_size, errors_mae, 1, mae_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	*mae = (*mae_sum) / nnz;
	cudaFree(errors_rmse);
	cudaFree(errors_mae);
	cublasDestroy(handle_rmse);
	cublasDestroy(handle_mae);
	free(rmse_sum);
	free(mae_sum);

}
