#include <cublas_v2.h>
#include "parameter.h"

__global__ void Update_Parameter_A_SGD(const int order,
		const int core_kernel, const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		const type_of_data *value, const int *index,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		const int update_order) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	__shared__ type_of_data shared_b[order_size][core_kernel_size][core_dimen_size];

	for (int order_index = 0; order_index < order_size; order_index++) {
		for (int core_kernel_index = 0; core_kernel_index < core_kernel;
				core_kernel_index++) {
			shared_b[order_index][core_kernel_index][lane_id] =
					parameter_b[order_index][core_kernel_index * core_dimen
							+ lane_id];
		}
	}
	__syncthreads();

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data p_a_gs = 0.0;
		type_of_data gs = 0.0;
		for (int core_kernel_index = 0; core_kernel_index < core_kernel;
				core_kernel_index++) {
			type_of_data gs_temp =
					shared_b[update_order][core_kernel_index][lane_id];
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != update_order) {
					type_of_data temp =
							parameter_a[inner_order_index][index[nnz_index
									* order + inner_order_index] * core_dimen
									+ lane_id]
									* shared_b[inner_order_index][core_kernel_index][lane_id];

					if (core_dimen == 4) {
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 4);
					} else if (core_dimen == 8) {
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 8);
					} else if (core_dimen == 16) {
						temp += __shfl_down_sync(mask, temp, 8);
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 16);
					} else if (core_dimen == 32) {
						temp += __shfl_down_sync(mask, temp, 16);
						temp += __shfl_down_sync(mask, temp, 8);
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0);
					}

					gs_temp *= temp;
				}
			}
			gs += gs_temp;
		}

		p_a_gs = parameter_a[update_order][index[nnz_index * order
				+ update_order] * core_dimen + lane_id] * gs;

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

		parameter_a[update_order][index[nnz_index * order + update_order]
				* core_dimen + lane_id] -= learn_rate_a
				* (p_a_gs * gs
						+ lambda_a
								* parameter_a[update_order][index[nnz_index
										* order + update_order] * core_dimen
										+ lane_id]);
	}
}

void Update_Parameter_A(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a_device,
		type_of_data **parameter_b_device, const int nnz_train,
		type_of_data **value_train_device, int **index_train_device,
		type_of_data learn_rate_a, type_of_data lambda_a) {

	int data_per_part = nnz_train / data_part + 1;

	for (int i = 0; i < data_part - 1; i++) {
		for (int j = 0; j < order; j++) {
			Update_Parameter_A_SGD<<<grid_size, block_size>>>(order,
					core_kernel, core_dimen, parameter_a_device,
					parameter_b_device, data_per_part, value_train_device[i],
					index_train_device[i], learn_rate_a, lambda_a, j);
			cudaDeviceSynchronize();
		}
	}
	for (int j = 0; j < order; j++) {
		Update_Parameter_A_SGD<<<grid_size, block_size>>>(order,
				core_kernel, core_dimen, parameter_a_device, parameter_b_device,
				nnz_train - (data_part - 1) * data_per_part,
				value_train_device[data_part - 1],
				index_train_device[data_part - 1], learn_rate_a, lambda_a, j);
		cudaDeviceSynchronize();
	}

}

__global__ void Parameter_B_SGD_Gradient(const int order,
		const int core_kernel, const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		const type_of_data *value, const int *index,
		type_of_data *b_sum, int update_order) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	__shared__ type_of_data shared_b[order_size][core_kernel_size][core_dimen_size];
	__shared__ type_of_data shared_ho[core_kernel_size][block_size];

	for (int order_index = 0; order_index < order_size; order_index++) {
		for (int core_kernel_index = 0; core_kernel_index < core_kernel;
				core_kernel_index++) {
			shared_b[order_index][core_kernel_index][lane_id] =
					parameter_b[order_index][core_kernel_index * core_dimen
							+ lane_id];
		}
	}
	__syncthreads();

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		int order_index = nnz_index * order;

		type_of_data p_a_gs = 0.0;

		for (int core_kernel_index = 0; core_kernel_index < core_kernel;
				core_kernel_index++) {
			type_of_data ho = parameter_a[update_order][index[order_index
					+ update_order] * core_dimen + lane_id];
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != update_order) {
					type_of_data temp =
							parameter_a[inner_order_index][index[order_index
									+ inner_order_index] * core_dimen + lane_id]
									* shared_b[inner_order_index][core_kernel_index][lane_id];

					if (core_dimen == 4) {
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 4);
					} else if (core_dimen == 8) {
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 8);
					} else if (core_dimen == 16) {
						temp += __shfl_down_sync(mask, temp, 8);
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 16);
					} else if (core_dimen == 32) {
						temp += __shfl_down_sync(mask, temp, 16);
						temp += __shfl_down_sync(mask, temp, 8);
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0);
					}
					ho *= temp;
				}
			}
			shared_ho[core_kernel_index][threadIdx.x] = ho;
			ho *= shared_b[update_order][core_kernel_index][lane_id];
			p_a_gs += ho;
		}

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

		for (int core_kernel_index = 0; core_kernel_index < core_kernel;
				core_kernel_index++) {
			atomicAdd(
					&b_sum[(nnz_index % sum_size) * core_kernel * core_dimen
							+ core_kernel_index * core_dimen + lane_id],
					p_a_gs * shared_ho[core_kernel_index][threadIdx.x]);
		}
	}
}

__global__ void Parameter_B_Gradient_Sum(const int core_kernel,
		const int core_dimen, const int nnz,
		type_of_data *b_sum, type_of_data *b_grad) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int core_kernel_index = worker_id; core_kernel_index < core_kernel;
			core_kernel_index += workers) {
		for (int sum_size_index = 0; sum_size_index < sum_size;
				sum_size_index++) {
			b_grad[core_kernel_index * core_dimen + lane_id] +=
					b_sum[sum_size_index * core_kernel * core_dimen
							+ core_kernel_index * core_dimen + lane_id];
		}
		b_grad[core_kernel_index * core_dimen + lane_id] /= nnz;
	}
}

__global__ void Update_Parameter_B(const int update_order,
		const int core_kernel, const int core_dimen,
		type_of_data **parameter_b,
		type_of_data *b_grad, const type_of_data learn_rate_b,
		const type_of_data lambda_b) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int core_kernel_index = worker_id; core_kernel_index < core_kernel;
			core_kernel_index += workers) {

		parameter_b[update_order][core_kernel_index * core_dimen + lane_id] -=
				learn_rate_b
						* (b_grad[core_kernel_index * core_dimen + lane_id]
								+ lambda_b
										* parameter_b[update_order][core_kernel_index
												* core_dimen + lane_id]);
	}
}

void Update_Parameter_B_Batch(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, type_of_data **value,
		int **index, const type_of_data learn_rate_b,
		const type_of_data lambda_b) {

	type_of_data *b_sum;
	type_of_data *b_grad;

	cudaMalloc((void**) &b_sum,
	sum_size * core_kernel * core_dimen * sizeof(type_of_data));
	cudaMalloc((void**) &b_grad,
			core_kernel * core_dimen * sizeof(type_of_data));

	for (int order_index = 0; order_index < order; order_index++) {

		cudaMemset(b_sum, 0,
		sum_size * core_kernel * core_dimen * sizeof(type_of_data));
		cudaMemset(b_grad, 0, core_kernel * core_dimen * sizeof(type_of_data));

		int data_per_part = nnz / data_part + 1;

		for (int i = 0; i < data_part - 1; i++) {
			Parameter_B_SGD_Gradient <<<grid_size,
			block_size>>>(order, core_kernel, core_dimen, parameter_a,
					parameter_b, data_per_part, value[i], index[i], b_sum,
					order_index);
			cudaDeviceSynchronize();
		}
		Parameter_B_SGD_Gradient <<<grid_size, block_size>>>(order,
				core_kernel, core_dimen, parameter_a, parameter_b,
				nnz - (data_part - 1) * data_per_part, value[data_part - 1],
				index[data_part - 1], b_sum, order_index);
		cudaDeviceSynchronize();

		Parameter_B_Gradient_Sum<<<
		core_kernel / (block_size / core_dimen) + 1, block_size>>>(
				core_kernel, core_dimen, nnz, b_sum, b_grad);
		cudaDeviceSynchronize();

		Update_Parameter_B <<< core_kernel / (block_size / core_dimen) + 1,
		block_size>>>(order_index, core_kernel, core_dimen, parameter_b, b_grad,
				learn_rate_b, lambda_b);
		cudaDeviceSynchronize();
	}

	cudaFree(b_sum);
	cudaFree(b_grad);

}

__global__ void RMSE_AND_MAE(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		const type_of_data *value, const int *index,
		type_of_data *rmse,
		type_of_data *mae) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data p_a_gs = 0.0;
		type_of_data gs = 0.0;

		for (int core_kernel_index = 0; core_kernel_index < core_kernel;
				core_kernel_index++) {
			type_of_data gs_temp = parameter_b[0][core_kernel_index * core_dimen
					+ lane_id];

			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != 0) {
					type_of_data temp =
							parameter_a[inner_order_index][index[nnz_index
									* order + inner_order_index] * core_dimen
									+ lane_id]
									* parameter_b[inner_order_index][core_kernel_index
											* core_dimen + lane_id];

					if (core_dimen == 4) {
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 4);
					} else if (core_dimen == 8) {
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 8);
					} else if (core_dimen == 16) {
						temp += __shfl_down_sync(mask, temp, 8);
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 16);
					} else if (core_dimen == 32) {
						temp += __shfl_down_sync(mask, temp, 16);
						temp += __shfl_down_sync(mask, temp, 8);
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0);
					}

					gs_temp *= temp;

				}
			}
			gs += gs_temp;
		}

		p_a_gs = parameter_a[0][index[nnz_index * order] * core_dimen + lane_id]
				* gs;

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

void GET_RMSE_AND_MAE(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		type_of_data **value, int **index, type_of_data *rmse,
		type_of_data *mae) {

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse,
	error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0, error_size * sizeof(type_of_data));

	int data_per_part = nnz / data_part + 1;
	for (int i = 0; i < data_part - 1; i++) {
		RMSE_AND_MAE <<<data_per_part / block_size + 1, block_size>>>(order,
				core_kernel, core_dimen, parameter_a, parameter_b,
				data_per_part, value[i], index[i], errors_rmse, errors_mae);
		cudaDeviceSynchronize();
	}
	RMSE_AND_MAE<<<data_per_part / block_size + 1, block_size>>>(order,
			core_kernel, core_dimen, parameter_a, parameter_b,
			nnz - (data_part - 1) * data_per_part, value[data_part - 1],
			index[data_part - 1], errors_rmse, errors_mae);
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

void GET_RMSE_AND_MAE(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		type_of_data *value, int *index,
		type_of_data *rmse,
		type_of_data *mae) {

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse,
	error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0, error_size * sizeof(type_of_data));

	RMSE_AND_MAE<<<nnz / block_size + 1, block_size>>>(order, core_kernel,
			core_dimen, parameter_a, parameter_b, nnz, value, index,
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
