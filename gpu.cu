#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#define NUM_THREADS 256
#define MAX_PARTICLES_PER_BLOCK 2048 // Ensure this matches shared memory capacity

// Static global variables
int blks;
int* bin_indices_gpu;
int* bin_counters_gpu;
int* bins_gpu;
double bin_size;
int num_bins_x;
int num_bins_y;
int num_bins;
int total_num_threads;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    // Short-range repulsive force
    double coef = (1.0 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__device__ int get_bin_index_gpu(particle_t& particle, double bin_size, int num_bins_x) {
    int x = particle.x / bin_size;
    int y = particle.y / bin_size;
    return x + num_bins_x * y;
}

__global__ void zero_out_bin_counts(int* bin_indices_gpu, int* bin_counters_gpu, int num_bins, int total_num_threads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < num_bins + 1; i += total_num_threads) {
        bin_indices_gpu[i] = 0;
        bin_counters_gpu[i] = 0;
    }
}

__global__ void compute_bin_counts_gpu(particle_t* particles, int num_parts, int total_num_threads, double bin_size, int num_bins_x, int* bin_indices_gpu) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < num_parts; i += total_num_threads) {
        int bin_index = get_bin_index_gpu(particles[i], bin_size, num_bins_x);
        atomicAdd(&(bin_indices_gpu[bin_index]), 1);
    }
}

__global__ void compute_bins_gpu(particle_t* particles, int num_parts, int total_num_threads, double bin_size, int num_bins_x, int* bins_gpu, int* bin_counters_gpu, int* bin_indices_gpu) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < num_parts; i += total_num_threads) {
        int bin_index = get_bin_index_gpu(particles[i], bin_size, num_bins_x);
        int count = atomicAdd(&bin_counters_gpu[bin_index], 1);
        int particle_index = bin_indices_gpu[bin_index] + count;
        bins_gpu[particle_index] = i; 
    }
}

__global__ void initialize_accelerations(particle_t* particles, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particles[tid].ax = 0;
    particles[tid].ay = 0;
}

// Same-bin kernel uses dynamic shared memory for up to MAX_PARTICLES_PER_BLOCK:
extern __shared__ particle_t shared_particles[];

__global__ void compute_forces_same_bin(particle_t* particles, int* bins_gpu, int* bin_indices_gpu, int num_bins_x, int num_bins_y, double bin_size) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    int bin_x = blockIdx.x; 
    int bin_y = blockIdx.y; 
    int bin_index = bin_x + num_bins_x * bin_y;

    int bin_start_index = bin_indices_gpu[bin_index];
    int bin_end_index   = bin_indices_gpu[bin_index + 1];
    int num_particles_in_bin = bin_end_index - bin_start_index;

    // Protect against overflow
    if (num_particles_in_bin > MAX_PARTICLES_PER_BLOCK) {
        printf("Error: Too many particles in bin. Increase MAX_PARTICLES_PER_BLOCK.\n");
        return;
    }

    // Strided load into shared memory
    int totalThreads = blockDim.x * blockDim.y;
    int particles_per_thread = (num_particles_in_bin + totalThreads - 1) / totalThreads;
    int start = tid * particles_per_thread;
    int end   = min(start + particles_per_thread, num_particles_in_bin);

    for (int i = start; i < end; i++) {
        shared_particles[i] = particles[bins_gpu[bin_start_index + i]];
    }
    __syncthreads();

    // Compute forces
    for (int i = tid; i < num_particles_in_bin; i += totalThreads) {
        particle_t& p = shared_particles[i];
        for (int j = 0; j < num_particles_in_bin; j++) {
            apply_force_gpu(p, shared_particles[j]);
        }
    }
    __syncthreads();

    // Write back updated states
    for (int i = tid; i < num_particles_in_bin; i += totalThreads) {
        int global_idx = bins_gpu[bin_start_index + i];
        particles[global_idx] = shared_particles[i];
    }
}

// Neighbor-bin kernel:
__global__ void compute_forces_neighbor_bins(particle_t* particles, int* bins_gpu, int* bin_indices_gpu, int num_bins_x, int num_bins_y, double bin_size) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    int bin_x = blockIdx.x;
    int bin_y = blockIdx.y; 
    int bin_index = bin_x + num_bins_x * bin_y;

    int bin_start_index = bin_indices_gpu[bin_index];
    int bin_end_index   = bin_indices_gpu[bin_index + 1];

    // Shared memory where neighbor-bin particles will be loaded
    __shared__ particle_t shared_particles_nb[MAX_PARTICLES_PER_BLOCK];

    int totalThreads = blockDim.x * blockDim.y;
    // For each particle in this bin:
    for (int local_i = tid; local_i < (bin_end_index - bin_start_index); local_i += totalThreads) {
        int global_p_idx = bins_gpu[bin_start_index + local_i];
        particle_t& particle = particles[global_p_idx];

        // Check neighbors:
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                // Skip the same bin
                if (dx == 0 && dy == 0) continue;

                int nx = bin_x + dx;
                int ny = bin_y + dy;
                // Bounds check
                if (nx < 0 || nx >= num_bins_x || ny < 0 || ny >= num_bins_y) {
                    continue;
                }

                int neighbor_bin_index = nx + num_bins_x * ny;
                int neighbor_bin_start = bin_indices_gpu[neighbor_bin_index];
                int neighbor_bin_end   = bin_indices_gpu[neighbor_bin_index + 1];
                int neighbor_count     = neighbor_bin_end - neighbor_bin_start;

                // Prevent overflow in shared memory
                if (neighbor_count > MAX_PARTICLES_PER_BLOCK) {
                    printf("Error: Too many particles in neighbor bin. Increase MAX_PARTICLES_PER_BLOCK.\n");
                    continue;
                }

                // Load neighbor particles into shared memory with 2D stride
                for (int j = tid; j < neighbor_count; j += totalThreads) {
                    shared_particles_nb[j] = particles[bins_gpu[neighbor_bin_start + j]];
                }
                __syncthreads();

                // Apply forces with loaded neighbors
                for (int j = 0; j < neighbor_count; j++) {
                    apply_force_gpu(particle, shared_particles_nb[j]);
                }
                __syncthreads();
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particle_t* p = &particles[tid];
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    // Reflect boundary conditions
    while (p->x < 0 || p->x > size) {
        p->x  = p->x < 0 ? 0 : size;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y  = p->y < 0 ? 0 : size;
        p->vy = -(p->vy);
    }
}

// Initialization and simulation routines
void init_simulation(particle_t* parts, int num_parts, double size) {
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    total_num_threads = blks * NUM_THREADS;

    bin_size = cutoff;
    num_bins_x = size / bin_size + 1;
    num_bins_y = size / bin_size + 1;
    num_bins   = num_bins_x * num_bins_y;

    cudaMalloc((void **)&bin_indices_gpu,   sizeof(int) * (num_bins + 1));
    cudaMalloc((void **)&bin_counters_gpu,  sizeof(int) * (num_bins + 1));
    cudaMalloc((void **)&bins_gpu,          sizeof(int) * num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // First, reset accelerations
    initialize_accelerations<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Zero out bin structures
    zero_out_bin_counts<<<blks, NUM_THREADS>>>(bin_indices_gpu, bin_counters_gpu, num_bins, total_num_threads);
    cudaDeviceSynchronize();

    // Count how many particles go in each bin
    compute_bin_counts_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, total_num_threads, bin_size, num_bins_x, bin_indices_gpu);
    cudaDeviceSynchronize();

    // Prefix sum to determine bin start indices
    thrust::device_ptr<int> bin_counts_gpu_ptr(bin_indices_gpu);
    thrust::exclusive_scan(bin_counts_gpu_ptr, bin_counts_gpu_ptr + num_bins + 1, bin_counts_gpu_ptr);

    // Fill bins_gpu array with actual particle indices
    compute_bins_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, total_num_threads, bin_size, num_bins_x, bins_gpu, bin_counters_gpu, bin_indices_gpu);
    cudaDeviceSynchronize();

    // Launch force kernels
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_bins_x + 15) / 16, (num_bins_y + 15) / 16);

    // Shared mem for same-bin kernel
    size_t shared_mem_size = MAX_PARTICLES_PER_BLOCK * sizeof(particle_t);
    compute_forces_same_bin<<<numBlocks, threadsPerBlock, shared_mem_size>>>(parts, bins_gpu, bin_indices_gpu, num_bins_x, num_bins_y, bin_size);
    compute_forces_neighbor_bins<<<numBlocks, threadsPerBlock>>>(parts, bins_gpu, bin_indices_gpu, num_bins_x, num_bins_y, bin_size);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
