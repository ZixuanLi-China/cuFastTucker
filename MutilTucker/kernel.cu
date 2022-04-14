#include "kernel.h"
#include <omp.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "tools.h"

#define type_of_data float
#define grid_size 1024
#define block_size 128
#define warp_size 32
#define sum_size 1024
#define error_size 1024
#define sm 1024

using namespace std;

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
			int parameter_g_index = index;
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				weight /= core_dimen;
				if (inner_order_index != order_index) {
					g_index += (parameter_g_index % core_dimen) * weight;
					parameter_g_index /= core_dimen;
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
		const type_of_data *value, const int *index, type_of_data **g,
		const type_of_data learn_rate_a, const type_of_data lambda_a) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data gs_shared[];

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		for (int order_index = 0; order_index < order; order_index++) {
			type_of_data gs = 0.0;
			for (int g_index = 0; g_index < core_length / core_dimen;
					g_index++) {
				int parameter_a_index = g_index;
				type_of_data s = 1.0;
				for (int inner_order_index = 0; inner_order_index < order;
						inner_order_index++) {
					if (inner_order_index != order_index) {
						s *= parameter_a[inner_order_index][index[nnz_index
								* order + inner_order_index] * core_dimen
								+ parameter_a_index % core_dimen];
						parameter_a_index /= core_dimen;
					}

				}
				gs += s * g[order_index][g_index * core_dimen + lane_id];
			}
			gs_shared[order_index * block_size + threadIdx.x] = gs;
		}
		type_of_data p_a_temp = parameter_a[0][index[nnz_index * order]
				* core_dimen + lane_id];

		type_of_data p_a_gs = p_a_temp * gs_shared[threadIdx.x];

		int temp = core;
		while (temp != 1) {
			temp /= 2;
			p_a_gs += __shfl_down_sync(0xffffffff, p_a_gs, temp);
		}

		p_a_gs = __shfl_sync(0xffffffff, p_a_gs, (local_id % local) * core);
		for (int order_index = 0; order_index < order; order_index++) {
			parameter_a[order_index][index[nnz_index * order + order_index]
					* core_dimen + lane_id] -= learn_rate_a
					* (-value[nnz_index]
							* gs_shared[order_index * block_size + threadIdx.x]
							+ p_a_gs
									* gs_shared[order_index * block_size
											+ threadIdx.x]
							+ lambda_a
									* parameter_a[order_index][index[nnz_index
											* order + order_index] * core_dimen
											+ lane_id]);
		}
	}
}

__global__ void Update_Parameter_A_SGD_Shared(const int order,
		const int core_length, const int core_dimen, type_of_data **parameter_a,
		const int nnz, const type_of_data *value, const int *index,
		type_of_data **g, const type_of_data learn_rate_a,
		const type_of_data lambda_a) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];

	type_of_data *gs_shared = shared;
	type_of_data *g_shared = (type_of_data*) &shared[order * block_size];

	for (int i = local_id; i < order * core_length / core_dimen; i += worker) {
		g_shared[i * core_dimen + lane_id] =
				g[i / (core_length / core_dimen)][(i
						% (core_length / core_dimen)) * core_dimen + lane_id];
	}
	__syncthreads();

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		for (int order_index = 0; order_index < order; order_index++) {
			type_of_data gs = 0.0;
			for (int g_index = 0; g_index < core_length / core_dimen;
					g_index++) {
				int parameter_a_index = g_index;
				type_of_data s = 1.0;
				for (int inner_order_index = 0; inner_order_index < order;
						inner_order_index++) {
					if (inner_order_index != order_index) {
						s *= parameter_a[inner_order_index][index[nnz_index
								* order + inner_order_index] * core_dimen
								+ parameter_a_index % core_dimen];
						parameter_a_index /= core_dimen;
					}

				}
				gs += s
						* g_shared[order_index * core_length
								+ g_index * core_dimen + lane_id];
			}
			gs_shared[order_index * block_size + threadIdx.x] = gs;
		}
		type_of_data p_a_temp = parameter_a[0][index[nnz_index * order]
				* core_dimen + lane_id];

		type_of_data p_a_gs = p_a_temp * gs_shared[threadIdx.x];

		int temp = core;
		while (temp != 1) {
			temp /= 2;
			p_a_gs += __shfl_down_sync(0xffffffff, p_a_gs, temp);
		}

		p_a_gs = __shfl_sync(0xffffffff, p_a_gs, (local_id % local) * core);
		for (int order_index = 0; order_index < order; order_index++) {
			parameter_a[order_index][index[nnz_index * order + order_index]
					* core_dimen + lane_id] -= learn_rate_a
					* (-value[nnz_index]
							* gs_shared[order_index * block_size + threadIdx.x]
							+ p_a_gs
									* gs_shared[order_index * block_size
											+ threadIdx.x]
							+ lambda_a
									* parameter_a[order_index][index[nnz_index
											* order + order_index] * core_dimen
											+ lane_id]);
		}
	}
}

void Mutil_Update_Parameter_A(const int order, const int gpu_number,
		const int part_per_gpu, const int core_length, const int core_dimen,
		const int *multi_nnz_train, type_of_data **multi_value_train_device,
		int **multi_index_train_device, type_of_data ***parameter_a_device,
		type_of_data ***parameter_a_host_to_device, type_of_data ***g_device,
		int **multi_start, int **multi_part, type_of_data learn_rate_a,
		type_of_data lambda_a, double *memcpy_time_a, const int model) {

	omp_set_num_threads(gpu_number);
	double *memcpy_all_time = (double*) malloc(sizeof(double) * gpu_number);
#pragma omp parallel
	{
		int i = omp_get_thread_num();
		cudaSetDevice(i);

		int *select = (int*) malloc(sizeof(int) * order);
		for (int j = 0; j < order; j++) {
			select[j] = i;
		}

		memcpy_all_time[i] = 0.0;
		double memcpy_start_time;
		double memcpy_end_time;

		for (int j = 0; j < part_per_gpu; j++) {

			int weight = 1;
			int part_index = 0;
			for (int k = order - 1; k > -1; k--) {
				part_index += weight * select[k];
				weight *= gpu_number;
			}
			if (model == 0) {
				Update_Parameter_A_SGD<<<grid_size, block_size,
				order * block_size * sizeof(type_of_data)>>>(order, core_length,
						core_dimen, parameter_a_device[i],
						multi_nnz_train[part_index],
						multi_value_train_device[part_index],
						multi_index_train_device[part_index], g_device[i],
						learn_rate_a, lambda_a);
				cudaDeviceSynchronize();
			} else if (model == 1) {
				Update_Parameter_A_SGD_Shared<<<grid_size, block_size,
				(order * block_size + order * core_length)
				* sizeof(type_of_data)>>>(order, core_length, core_dimen,
						parameter_a_device[i], multi_nnz_train[part_index],
						multi_value_train_device[part_index],
						multi_index_train_device[part_index], g_device[i],
						learn_rate_a, lambda_a);
				cudaDeviceSynchronize();
			}
			int temp_transport = j;
			int count_transport = 0;
			for (int k = 0; k < order - 1; k++) {
				if (temp_transport % gpu_number == 0 && temp_transport != 0) {
					count_transport++;
					temp_transport /= gpu_number;
				} else {
					break;
				}
			}

			memcpy_start_time = Seconds();
			cudaMemcpyPeer(
					parameter_a_host_to_device[(i + 1) % gpu_number][order - 1
							- count_transport]
							+ multi_start[order - 1 - count_transport][select[order
									- 1 - count_transport]] * core_dimen,
					(i + 1) % gpu_number,
					parameter_a_host_to_device[i][order - 1 - count_transport]
							+ multi_start[order - 1 - count_transport][select[order
									- 1 - count_transport]] * core_dimen, i,
					sizeof(type_of_data)
							* multi_part[order - 1 - count_transport][select[order
									- 1 - count_transport]] * core_dimen);
			memcpy_end_time = Seconds();
			memcpy_all_time[i] += memcpy_end_time - memcpy_start_time;

#pragma omp barrier

			int temp_update = j + 1;
			int count_update = 0;
			for (int k = 0; k < order - 1; k++) {
				if (temp_update % gpu_number == 0) {
					count_update++;
					temp_update /= gpu_number;
				} else {
					break;
				}
			}
			select[order - 1 - count_update] = (select[order - 1 - count_update]
					+ 1) % gpu_number;
		}

	}
	*memcpy_time_a = 0.0;
	for (int i = 0; i < gpu_number; i++) {
		if (memcpy_all_time[i] > *memcpy_time_a) {
			*memcpy_time_a = memcpy_all_time[i];
		}
	}
	free(memcpy_all_time);

}

__global__ void Update_Parameter_G_Gradient(const int order,
		const int core_length, const int core_dimen, type_of_data **parameter_a,
		type_of_data *parameter_g, const int nnz, const type_of_data *value,
		const int *index, type_of_data *g_sum) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data h_shared[];

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data value_nnz = value[nnz_index];
		type_of_data h;
		type_of_data x_r = 0.0;
		for (int g_index = 0; g_index < core_length / core_dimen; g_index++) {
			int parameter_a_and_b_index = g_index;
			h = parameter_a[0][index[nnz_index * order] * core_dimen + lane_id];
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != 0) {
					h *= parameter_a[inner_order_index][index[nnz_index * order
							+ inner_order_index] * core_dimen
							+ parameter_a_and_b_index % core_dimen];
					parameter_a_and_b_index /= core_dimen;
				}
			}
			h_shared[g_index * block_size + threadIdx.x] = h;
			x_r += h * parameter_g[g_index * core_dimen + lane_id];
		}

		int temp = core;
		while (temp != 1) {
			temp /= 2;
			x_r += __shfl_down_sync(0xffffffff, x_r, temp);
		}

		x_r = __shfl_sync(0xffffffff, x_r, (local_id % local) * core);

		x_r -= value_nnz;

		for (int g_index = 0; g_index < core_length / core_dimen; g_index++) {
			atomicAdd(
					&g_sum[(nnz_index % sum_size) * core_length
							+ g_index * core_dimen + lane_id],
					x_r * h_shared[g_index * block_size + threadIdx.x]);
		}

	}
}

__global__ void Update_Parameter_G_Gradient_Shared(const int order,
		const int core_length, const int core_dimen, type_of_data **parameter_a,
		type_of_data *parameter_g, const int nnz, const type_of_data *value,
		const int *index, type_of_data *g_sum) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];

	type_of_data *h_shared = shared;
	type_of_data *g_shared = (type_of_data*) &shared[core_length
			* (block_size / core_dimen)];

	for (int i = local_id; i < core_length / core_dimen; i += worker) {
		g_shared[i * core_dimen + lane_id] = parameter_g[i * core_dimen
				+ lane_id];
	}
	__syncthreads();

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data value_nnz = value[nnz_index];
		type_of_data h;
		type_of_data x_r = 0.0;
		for (int g_index = 0; g_index < core_length / core_dimen; g_index++) {
			int parameter_a_and_b_index = g_index;
			h = parameter_a[0][index[nnz_index * order] * core_dimen + lane_id];
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != 0) {
					h *= parameter_a[inner_order_index][index[nnz_index * order
							+ inner_order_index] * core_dimen
							+ parameter_a_and_b_index % core_dimen];
					parameter_a_and_b_index /= core_dimen;
				}
			}
			h_shared[g_index * block_size + threadIdx.x] = h;
			x_r += h * g_shared[g_index * core_dimen + lane_id];
		}

		int temp = core;
		while (temp != 1) {
			temp /= 2;
			x_r += __shfl_down_sync(0xffffffff, x_r, temp);
		}

		x_r = __shfl_sync(0xffffffff, x_r, (local_id % local) * core);

		x_r -= value_nnz;

		for (int g_index = 0; g_index < core_length / core_dimen; g_index++) {
			atomicAdd(
					&g_sum[(nnz_index % sum_size) * core_length
							+ g_index * core_dimen + lane_id],
					x_r * h_shared[g_index * block_size + threadIdx.x]);
		}

	}
}

__global__ void Update_Parameter_G_Gradient_Gobal(const int order,
		const int core_length, const int core_dimen, type_of_data **parameter_a,
		type_of_data *parameter_g, const int nnz, const type_of_data *value,
		const int *index, type_of_data *g_sum,
		type_of_data *h_shared) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data value_nnz = value[nnz_index];
		type_of_data h;
		type_of_data x_r = 0.0;
		for (int g_index = 0; g_index < core_length / core_dimen; g_index++) {
			int parameter_a_and_b_index = g_index;
			h = parameter_a[0][index[nnz_index * order] * core_dimen + lane_id];
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != 0) {
					h *= parameter_a[inner_order_index][index[nnz_index * order
							+ inner_order_index] * core_dimen
							+ parameter_a_and_b_index % core_dimen];
					parameter_a_and_b_index /= core_dimen;
				}
			}
			h_shared[(nnz_index % sm) * g_index * core_dimen + lane_id] = h;
			x_r += h * parameter_g[g_index * core_dimen + lane_id];
		}

		int temp = core;
		while (temp != 1) {
			temp /= 2;
			x_r += __shfl_down_sync(0xffffffff, x_r, temp);
		}

		x_r = __shfl_sync(0xffffffff, x_r, (local_id % local) * core);

		x_r -= value_nnz;

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

__global__ void Parameter_G_Gradient_Sum_Part(const int core_length,
		const int core_dimen, type_of_data *g_sum,
		type_of_data *g_grad) {

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
	}
}

__global__ void Parameter_G_Gradient_Sum_Whole(const int core_length,
		const int core_dimen, const int nnz, type_of_data *g_grad,
		const int gpu_number) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int g_index = worker_id; g_index < core_length / core_dimen; g_index +=
			workers) {
		for (int gpu_number_index = 1; gpu_number_index < gpu_number;
				gpu_number_index++) {
			g_grad[g_index * core_dimen + lane_id] += g_grad[gpu_number_index
					* core_length + g_index * core_dimen + lane_id];
		}
		g_grad[g_index * core_dimen + lane_id] /= nnz;
	}
}

__global__ void Update_Parameter_G(const int core_length, const int core_dimen,
type_of_data *parameter_g, type_of_data *g_grad,
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

void Mutil_Update_Parameter_G(const int order, const int gpu_number,
		const int nnz, const int part_per_gpu, const int core_length,
		const int core_dimen, const int *multi_nnz_train,
		type_of_data **multi_value_train_device, int **multi_index_train_device,
		type_of_data ***parameter_a_device, type_of_data **parameter_g_device,
		type_of_data learn_rate_g, type_of_data lambda_g, double *memcpy_time_g,
		const int model) {

	type_of_data **mutil_g_sum;
	type_of_data **mutil_g_grad;

	mutil_g_sum = (type_of_data**) malloc(sizeof(type_of_data*) * gpu_number);
	mutil_g_grad = (type_of_data**) malloc(sizeof(type_of_data*) * gpu_number);
	double *memcpy_all_time = (double*) malloc(sizeof(double) * gpu_number);

	omp_set_num_threads(gpu_number);
#pragma omp parallel
	{
		int i = omp_get_thread_num();
		cudaSetDevice(i);

		memcpy_all_time[i] = 0.0;
		double memcpy_start_time;
		double memcpy_end_time;

		cudaMalloc((void**) &(mutil_g_sum[i]),
		sum_size * core_length * sizeof(type_of_data));
		cudaMemset(mutil_g_sum[i], 0,
		sum_size * core_length * sizeof(type_of_data));

		if (i == 0) {
			cudaMalloc((void**) &(mutil_g_grad[i]),
					gpu_number * core_length * sizeof(type_of_data));
			cudaMemset(mutil_g_grad[i], 0,
					gpu_number * core_length * sizeof(type_of_data));
		} else {
			cudaMalloc((void**) &(mutil_g_grad[i]),
					core_length * sizeof(type_of_data));
			cudaMemset(mutil_g_grad[i], 0, core_length * sizeof(type_of_data));
		}

		for (int j = 0; j < part_per_gpu; j++) {

			if (model == 0) {
				Update_Parameter_G_Gradient<<<grid_size, block_size,
				core_length * (block_size / core_dimen)
				* sizeof(type_of_data)>>>(order, core_length, core_dimen,
						parameter_a_device[i], parameter_g_device[i],
						multi_nnz_train[i * part_per_gpu + j],
						multi_value_train_device[i * part_per_gpu + j],
						multi_index_train_device[i * part_per_gpu + j],
						mutil_g_sum[i]);
				cudaDeviceSynchronize();
			} else if (model == 1) {
				Update_Parameter_G_Gradient_Shared<<<grid_size, block_size,
				(core_length * (block_size / core_dimen) + core_length)
				* sizeof(type_of_data)>>>(order, core_length, core_dimen,
						parameter_a_device[i], parameter_g_device[i],
						multi_nnz_train[i * part_per_gpu + j],
						multi_value_train_device[i * part_per_gpu + j],
						multi_index_train_device[i * part_per_gpu + j],
						mutil_g_sum[i]);
				cudaDeviceSynchronize();
			} else if (model == 2) {
				type_of_data *h_shared;
				cudaMalloc((void**) &h_shared,
				sm * core_length * sizeof(type_of_data));
				cudaMemset(h_shared, 0,
				sm * core_length * sizeof(type_of_data));

				Update_Parameter_G_Gradient_Gobal<<<grid_size, block_size>>>(
						order, core_length, core_dimen, parameter_a_device[i],
						parameter_g_device[i],
						multi_nnz_train[i * part_per_gpu + j],
						multi_value_train_device[i * part_per_gpu + j],
						multi_index_train_device[i * part_per_gpu + j],
						mutil_g_sum[i], h_shared);
				cudaDeviceSynchronize();
				cudaFree(h_shared);
			}
		}

		Parameter_G_Gradient_Sum_Part<<<
		core_length / (block_size / core_dimen) + 1,
		block_size>>>(core_length, core_dimen, mutil_g_sum[i], mutil_g_grad[i]);
		cudaDeviceSynchronize();

		if (i != 0) {
			memcpy_start_time = Seconds();
			cudaMemcpyPeer(mutil_g_grad[0] + i * core_length, 0, mutil_g_sum[i],
					i, sizeof(type_of_data) * core_length);
			memcpy_end_time = Seconds();
			memcpy_all_time[i] += memcpy_end_time - memcpy_start_time;
		}
#pragma omp barrier
		if (i == 0) {

			Parameter_G_Gradient_Sum_Whole<<<
			core_length / (block_size / core_dimen) + 1, block_size>>>(
					core_length, core_dimen, nnz, mutil_g_grad[0], gpu_number);
			cudaDeviceSynchronize();

			Update_Parameter_G<<<core_length / (block_size / core_dimen) + 1,
			block_size>>>(core_length, core_dimen, parameter_g_device[0],
					mutil_g_grad[0], learn_rate_g, lambda_g);
			cudaDeviceSynchronize();

			for (int j = 1; j < gpu_number; j++) {
				memcpy_start_time = Seconds();
				cudaMemcpyPeer(parameter_g_device[j], j, parameter_g_device[0],
						0, sizeof(type_of_data) * core_length);
				memcpy_end_time = Seconds();
				memcpy_all_time[i] += memcpy_end_time - memcpy_start_time;
			}
		}
		cudaFree(mutil_g_sum[i]);
		cudaFree(mutil_g_grad[i]);
	}
	*memcpy_time_g = 0.0;
	for (int i = 0; i < gpu_number; i++) {
		if (memcpy_all_time[i] > *memcpy_time_g) {
			*memcpy_time_g = memcpy_all_time[i];
		}
	}
	free(mutil_g_sum);
	free(mutil_g_grad);
	free(memcpy_all_time);
}

__global__ void RMSE_AND_MAE(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a, const int nnz,
		const type_of_data *value, const int *index, type_of_data **g,
		type_of_data *rmse, type_of_data *mae) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data gs = 0.0;
		for (int g_index = 0; g_index < core_length / core_dimen; g_index++) {
			int parameter_a_index = g_index;
			type_of_data s = 1.0;
			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != 0) {
					s *= parameter_a[inner_order_index][index[nnz_index * order
							+ inner_order_index] * core_dimen
							+ parameter_a_index % core_dimen];
					parameter_a_index /= core_dimen;
				}

			}
			gs += s * g[0][g_index * core_dimen + lane_id];
		}

		type_of_data p_a_temp = parameter_a[0][index[nnz_index * order + 0]
				* core_dimen + lane_id];

		type_of_data p_a_gs = p_a_temp * gs;

		int temp = core;
		while (temp != 1) {
			temp /= 2;
			p_a_gs += __shfl_down_sync(0xffffffff, p_a_gs, temp);
		}

		p_a_gs = __shfl_sync(0xffffffff, p_a_gs, (local_id % local) * core);
		p_a_gs -= value[nnz_index];

		if (lane_id == 0) {
			atomicAdd(&rmse[nnz_index % error_size], p_a_gs * p_a_gs);
			atomicAdd(&mae[nnz_index % error_size], abs(p_a_gs));
		}

	}
}

void GET_RMSE_AND_MAE(const int order, const int core_length,
		const int core_dimen, type_of_data **parameter_a, const int nnz,
		const type_of_data *value, const int *index, type_of_data **g,
		type_of_data *rmse, type_of_data *mae) {

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

void Mutil_GET_RMSE_AND_MAE(const int order, const int gpu_number,
		const int multi_gpu_part, const int part_per_gpu, const int core_length,
		const int core_dimen, const int *multi_nnz_train,
		type_of_data **multi_value_train_device, int **multi_index_train_device,
		type_of_data ***parameter_a_device, type_of_data **parameter_g_device,
		type_of_data ***g_device, type_of_data *rmse, type_of_data *mae) {

	type_of_data *mul_rmse = (type_of_data*) malloc(
			sizeof(type_of_data) * multi_gpu_part);
	type_of_data *mul_mae = (type_of_data*) malloc(
			sizeof(type_of_data) * multi_gpu_part);

	int nnz_sum = 0;
	type_of_data rmse_sum = 0.0;
	type_of_data mae_sum = 0.0;

	omp_set_num_threads(gpu_number);
#pragma omp parallel
	{
		int i = omp_get_thread_num();

		cudaSetDevice(i);

		Structure_Core_Tensor<<<core_length / (block_size / core_dimen) + 1,
		block_size>>>(order, core_length, core_dimen, parameter_g_device[i],
				g_device[i]);
		cudaDeviceSynchronize();

		for (int j = 0; j < part_per_gpu; j++) {

			GET_RMSE_AND_MAE(order, core_length, core_dimen,
					parameter_a_device[i],
					multi_nnz_train[i * part_per_gpu + j],
					multi_value_train_device[i * part_per_gpu + j],
					multi_index_train_device[i * part_per_gpu + j], g_device[i],
					&(mul_rmse[i * part_per_gpu + j]),
					&(mul_mae[i * part_per_gpu + j]));

			nnz_sum += multi_nnz_train[i * part_per_gpu + j];
			rmse_sum += multi_nnz_train[i * part_per_gpu + j]
					* mul_rmse[i * part_per_gpu + j];
			mae_sum += multi_nnz_train[i * part_per_gpu + j]
					* mul_mae[i * part_per_gpu + j];
		}
	}
	(*rmse) = rmse_sum / (type_of_data) nnz_sum;
	(*mae) = mae_sum / (type_of_data) nnz_sum;
	free(mul_rmse);
	free(mul_mae);

}
