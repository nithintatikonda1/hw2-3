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

// Device utility to compute the force between two particles
__device__ void apply_force_gpu(particle_t& particle, const particle_t& neighbor) {
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

// Compute bin index from a particle’s position
__device__ int get_bin_index_gpu(const particle_t& p, double bin_size, int num_bins_x) {
    int x = p.x / bin_size;
    int y = p.y / bin_size;
    return x + num_bins_x * y;
}

// Zero out bin counts for a new iteration
__global__ void zero_out_bin_counts(int* bin_indices_gpu, int* bin_counters_gpu, int num_bins, int total_num_threads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < num_bins + 1; i += total_num_threads) {
        bin_indices_gpu[i] = 0;
        bin_counters_gpu[i] = 0;
    }
}

// Count how many particles land in each bin
__global__ void compute_bin_counts_gpu(
    particle_t* particles, int num_parts, int total_num_threads,
    double bin_size, int num_bins_x, int* bin_indices_gpu)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < num_parts; i += total_num_threads) {
        int bin_index = get_bin_index_gpu(particles[i], bin_size, num_bins_x);
        atomicAdd(&(bin_indices_gpu[bin_index]), 1);
    }
}

// Populate bins_gpu with the actual particle indices
__global__ void compute_bins_gpu(
    particle_t* particles, int num_parts, int total_num_threads,
    double bin_size, int num_bins_x, int* bins_gpu,
    int* bin_counters_gpu, int* bin_indices_gpu)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < num_parts; i += total_num_threads) {
        int bin_index = get_bin_index_gpu(particles[i], bin_size, num_bins_x);
        int offset = atomicAdd(&bin_counters_gpu[bin_index], 1);
        int global_idx = bin_indices_gpu[bin_index] + offset;
        bins_gpu[global_idx] = i;
    }
}

// Reset accelerations before each iteration
__global__ void initialize_accelerations(particle_t* particles, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    particles[tid].ax = 0;
    particles[tid].ay = 0;
}

// Move particles after forces are computed
__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particle_t* p = &particles[tid];
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    // Reflect at boundaries
    while (p->x < 0 || p->x > size) {
        p->x  = p->x < 0 ? 0 : size;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y  = p->y < 0 ? 0 : size;
        p->vy = -(p->vy);
    }
}

//-------------------------------------------------------------------------------
// Unified kernel: loads all particles from the bin and its neighbors (old states)
// into shared memory, then computes forces, then writes updated accel back once.
//-------------------------------------------------------------------------------
__global__ void compute_forces_unified(
    particle_t* particles,
    int* bins_gpu,
    int* bin_indices_gpu,
    int num_bins_x,
    int num_bins_y,
    double bin_size)
{
    // Determine which bin this block handles
    int bin_x = blockIdx.x;
    int bin_y = blockIdx.y;
    int bin_index = bin_x + num_bins_x * bin_y;

    // Count how many particles in current bin
    int bin_start_index = bin_indices_gpu[bin_index];
    int bin_end_index   = bin_indices_gpu[bin_index + 1];
    int bin_count       = bin_end_index - bin_start_index;

    // Shared memory array
    __shared__ particle_t shared_particles[MAX_PARTICLES_PER_BLOCK];

    // 2D thread indexing
    int local_tid = threadIdx.x + threadIdx.y * blockDim.x;
    int totalThreads = blockDim.x * blockDim.y;

    // Load current bin
    if (bin_count > MAX_PARTICLES_PER_BLOCK) {
        if (local_tid == 0) {
            printf("Error: Too many particles in bin (%d). Increase MAX_PARTICLES_PER_BLOCK.\n", bin_count);
        }
        return;
    }
    for (int i = local_tid; i < bin_count; i += totalThreads) {
        shared_particles[i] = particles[bins_gpu[bin_start_index + i]];
    }

    // Load neighbors
    int loaded_count = bin_count;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            // Skip same bin
            if (dx == 0 && dy == 0) continue;

            int nx = bin_x + dx;
            int ny = bin_y + dy;
            // Bounds check
            if (nx < 0 || nx >= num_bins_x || ny < 0 || ny >= num_bins_y) {
                continue;
            }

            int neighbor_index = nx + num_bins_x * ny;
            int nb_start = bin_indices_gpu[neighbor_index];
            int nb_end   = bin_indices_gpu[neighbor_index + 1];
            int nb_count = nb_end - nb_start;

            // Check capacity
            int new_count = loaded_count + nb_count;
            if (new_count > MAX_PARTICLES_PER_BLOCK) {
                if (local_tid == 0) {
                    printf("Error: Too many total particles (%d) in bin + neighbors.\n", new_count);
                }
                return;
            }

            // Strided load of neighbor bin
            for (int j = local_tid; j < nb_count; j += totalThreads) {
                shared_particles[loaded_count + j] = particles[bins_gpu[nb_start + j]];
            }
            loaded_count += nb_count;
        }
    }
    __syncthreads();

    // Compute forces only for current bin's particles
    for (int i = local_tid; i < bin_count; i += totalThreads) {
        particle_t& p = shared_particles[i];
        // Zero out accelerations locally
        p.ax = 0.0;
        p.ay = 0.0;
        for (int j = 0; j < loaded_count; j++) {
            if (i == j) continue; 
            apply_force_gpu(p, shared_particles[j]);
        }
    }
    __syncthreads();

    // Write updated accelerations for current bin to global memory
    for (int i = local_tid; i < bin_count; i += totalThreads) {
        int global_idx = bins_gpu[bin_start_index + i];
        particles[global_idx].ax = shared_particles[i].ax;
        particles[global_idx].ay = shared_particles[i].ay;
    }
}

//-------------------------------------------------------------------------------
// Initialization & single-step simulation
//-------------------------------------------------------------------------------
void init_simulation(particle_t* parts, int num_parts, double size) {
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    total_num_threads = blks * NUM_THREADS;

    bin_size = cutoff;
    num_bins_x = size / bin_size + 1;
    num_bins_y = size / bin_size + 1;
    num_bins   = num_bins_x * num_bins_y;

    cudaMalloc((void **) &bin_indices_gpu,  sizeof(int) * (num_bins + 1));
    cudaMalloc((void **) &bin_counters_gpu, sizeof(int) * (num_bins + 1));
    cudaMalloc((void **) &bins_gpu,         sizeof(int) * num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // 1. Reset accelerations
    initialize_accelerations<<<blks, NUM_THREADS>>>(parts, num_parts);

    // 2. Zero out bin structures
    zero_out_bin_counts<<<blks, NUM_THREADS>>>(bin_indices_gpu, bin_counters_gpu, num_bins, total_num_threads);
    cudaDeviceSynchronize();

    // 3. Count how many particles go in each bin
    compute_bin_counts_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, total_num_threads, bin_size, num_bins_x, bin_indices_gpu);
    cudaDeviceSynchronize();

    // 4. Prefix sum for bin indices
    thrust::device_ptr<int> d_bin_ptr(bin_indices_gpu);
    thrust::exclusive_scan(d_bin_ptr, d_bin_ptr + num_bins + 1, d_bin_ptr);

    // 5. Fill bins_gpu
    compute_bins_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, total_num_threads, bin_size, num_bins_x,
                                           bins_gpu, bin_counters_gpu, bin_indices_gpu);
    cudaDeviceSynchronize();

    // 6. Compute forces in unified kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_bins_x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_bins_y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    compute_forces_unified<<<numBlocks, threadsPerBlock>>>(parts, bins_gpu, bin_indices_gpu, 
                                                           num_bins_x, num_bins_y, bin_size);
    cudaDeviceSynchronize();

    // 7. Move particles (integrate velocities & positions)
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize();
}
