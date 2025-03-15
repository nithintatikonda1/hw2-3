#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
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
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
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

    // Set # particles currently in bin to 0 and set bin starting indices to 0
    for (int i = tid; i < num_bins + 1; i += total_num_threads) {
        bin_indices_gpu[i] = 0;
        bin_counters_gpu[i] = 0;
    }
}

__global__ void compute_bin_counts_gpu(particle_t* particles, int num_parts, int total_num_threads, double bin_size, int num_bins_x, int* bin_indices_gpu) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Compute bin counts
    for (int i = tid; i < num_parts; i += total_num_threads) {
        int bin_index = get_bin_index_gpu(particles[i], bin_size, num_bins_x);
        atomicAdd(&(bin_indices_gpu[bin_index]), 1);
    }

}

__global__ void compute_bins_gpu(particle_t* particles, int num_parts, int total_num_threads, double bin_size, int num_bins_x, int* bins_gpu, int* bin_counters_gpu) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Compute place particles in bin
    for (int i = tid; i < num_parts; i += total_num_threads) {
        int bin_start_index = get_bin_index_gpu(particles[i], bin_size, num_bins_x);
        int count = atomicAdd(&bin_counters_gpu[bin_start_index], 1);
        int particle_index = bin_start_index + count;
        bins_gpu[particle_index] = i; 
    }

}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int total_num_threads, double bin_size, int num_bins_x, int num_bins_y, int* bins_gpu, int*bin_indices_gpu) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread gets assigned particles
    for (int i = tid; i < num_parts; i += total_num_threads) {
        particle_t& particle = particles[i];

        // zero out accelerations
        particle.ax = 0;
        particle.ay = 0;

        // Compute bin indices
        int bin_x = particle.x / bin_size;
        int bin_y = particle.y / bin_size;
        int bin_index = get_bin_index_gpu(particle, bin_size, num_bins_x);

        // Explore neighboring bins
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int neighbor_bin_x = bin_x + dx;
                int neighbor_bin_y = bin_y + dy;

                // Skip if bins are out of bounds
                if (neighbor_bin_x < 0 || neighbor_bin_x >= num_bins_x ||
                    neighbor_bin_y < 0 || neighbor_bin_y >= num_bins_y) {
                        continue;
                }

                // Compute forces for all neighbors in bin.
                int neighbor_bin_index = neighbor_bin_x + num_bins_x * neighbor_bin_y;
                int bin_start_index = bin_indices_gpu[neighbor_bin_index];
                int bin_end_index = bin_indices_gpu[neighbor_bin_index + 1];
                for (int i = bin_start_index; i < bin_end_index; i++) {
                    int neighbor_index = bins_gpu[i];
                    apply_force_gpu(particle, particles[neighbor_index]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    total_num_threads = blks * NUM_THREADS;

    bin_size = cutoff;
    num_bins_x = size / bin_size + 1;
    num_bins_y = size / bin_size + 1;
    num_bins = num_bins_x * num_bins_y;

    cudaMalloc((void **)&bin_indices_gpu, sizeof(int) * num_bins + 1);
    cudaMalloc((void **)&bin_counters_gpu, sizeof(int) * num_bins + 1);
    cudaMalloc((void **)&bins_gpu, sizeof(int) * num_parts);

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Find the number of particles in each bin
    zero_out_bin_counts<<<blks, NUM_THREADS>>>(bin_indices_gpu, bin_counters_gpu, num_bins, total_num_threads);
    compute_bin_counts_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, total_num_threads, bin_size, num_bins_x, bin_indices_gpu);

    // Compute starting indices of bins
    thrust::device_ptr<int> bin_counts_gpu_ptr(bin_indices_gpu);
    thrust::exclusive_scan(bin_counts_gpu_ptr, bin_counts_gpu_ptr + num_bins + 1, bin_counts_gpu_ptr);

    // Add particles to bins
    compute_bins_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, total_num_threads, bin_size, num_bins_x, bins_gpu, bin_counters_gpu);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, total_num_threads, bin_size, num_bins_x, num_bins_y, bins_gpu, bin_indices_gpu);


    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
