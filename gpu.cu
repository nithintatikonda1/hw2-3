#include "common.h"
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

// Enable benchmarking to measure kernel execution times
// #define BENCHMARK 1

#define NUM_THREADS 256
// Constants for A100 GPU
#define NUM_SM              108
#define THREADS_PER_BLOCK   1024
#define MAX_WARPS_PER_BLOCK 64
#define THREADS_PER_WARP    32
// #define SHARED_MEM_SIZE_LIMIT  // 164 KB

// Put any static global variables here that you will use throughout the simulation.
int blks;              // Number of blocks for CUDA kernel launches
int tiled_blocks;      // Number of blocks for tiled approach
size_t shared_mem_size; // Size of shared memory for caching particles in this tile
int* bin_indices_gpu;  // Device array storing the number of particles in each bin (after scan this
                       // will be converted to the starting index of each bin)
int* bin_counters_gpu; // Device array storing count of particles in each bin while computing bin
                       // counts
int* bins_gpu;         // Device array storing particles grouped by bin
double bin_size;       // Size of each spatial bin for neighbor search
int num_bins_x;        // Number of bins in x dimension
int num_bins_y;        // Number of bins in y dimension
int num_bins;          // Total number of bins (num_bins_x * num_bins_y)
int total_num_threads; // Total number of CUDA threads across all blocks

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    /*
     * Compute and apply the force between two particles by calculating the repulsive force between
     * two particles based on their relative positions on the GPU.
     *
     * Parameters:
     *   particle: Reference to the particle to update forces for
     *   neighbor: Reference to the neighboring particle
     */
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
    /*
     * Compute the bin index for a particle based on its position and the bin size on the GPU.
     *
     * Parameters:
     *   particle: Reference to the particle to get the bin index for
     *   bin_size: The size of each bin in the grid
     *   num_bins_x: The number of bins in the x direction
     *
     * Returns:
     *   The bin index for the particle
     */
    int x = particle.x / bin_size;
    int y = particle.y / bin_size;

    return x + num_bins_x * y;
}

__global__ void zero_out_bin_counts(int* bin_indices_gpu, int* bin_counters_gpu, int num_bins,
                                    int total_num_threads) {
    /*
     * Zero out the bin counts and set the bin starting indices to 0 on the GPU.
     *
     * Parameters:
     *   bin_indices_gpu: Array storing the number of particles in each bin (after scan this will be
     * converted to the starting index of each bin) bin_counters_gpu: Array storing the count of
     * particles in each bin while computing bin counts num_bins: Total number of bins
     *   total_num_threads: Total number of threads being used
     */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Set # particles currently in bin to 0 and set bin starting indices to 0
    #pragma unroll 4
    for (int i = tid; i < num_bins + 1; i += total_num_threads) {
        bin_indices_gpu[i] = 0;
        bin_counters_gpu[i] = 0;
    }
}

__global__ void compute_bin_counts_gpu(particle_t* particles, int num_parts, int total_num_threads,
                                       double bin_size, int num_bins_x, int* bin_indices_gpu) {
    /*
     * Compute the number of particles in each bin on the GPU.
     *
     * Parameters:
     *   particles: Array of particles
     *   num_parts: Total number of particles
     *   total_num_threads: Total number of threads being used
     *   bin_size: The size of each bin in the grid
     *   num_bins_x: The number of bins in the x direction
     *   bin_indices_gpu: Array storing the number of particles in each bin (after scan this will be
     * converted to the starting index of each bin)
     */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Compute bin counts
    #pragma unroll 2
    for (int i = tid; i < num_parts; i += total_num_threads) {
        int bin_index = get_bin_index_gpu(particles[i], bin_size, num_bins_x);
        atomicAdd(&(bin_indices_gpu[bin_index]), 1);
    }
}

__global__ void compute_bins_gpu(particle_t* particles, int num_parts, int total_num_threads,
                                 double bin_size, int num_bins_x, int* bins_gpu,
                                 int* bin_counters_gpu, int* bin_indices_gpu) {
    /*
     * Place particles into their corresponding bins on the GPU.
     *
     * Parameters:
     *   particles: Array of particles
     *   num_parts: Total number of particles
     *   total_num_threads: Total number of threads being used
     *   bin_size: The size of each bin in the grid
     *   num_bins_x: The number of bins in the x direction
     *   bins_gpu: Array to store particle indices for each bin
     *   bin_counters_gpu: Array to track count of particles added to each bin
     *   bin_indices_gpu: Array storing the starting index for each bin
     */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

        // Compute place particles in bin
    #pragma unroll 2
    for (int i = tid; i < num_parts; i += total_num_threads) {
        int bin_index = get_bin_index_gpu(particles[i], bin_size, num_bins_x);
        int count = atomicAdd(&bin_counters_gpu[bin_index],
                              1); // Amount of particles that have already been added to the bin
        int particle_index = bin_indices_gpu[bin_index] + count;
        bins_gpu[particle_index] = i;
    }
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, double bin_size,
                                   int num_bins_x, int num_bins_y, int* bins_gpu,
                                   int* bin_indices_gpu) {
    // Get thread ID
    int local_tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int particle_idx = block_start + local_tid;

    // Shared memory for caching particles in this tile
    extern __shared__ particle_t shared_particles[];

    // Collaboratively load the entire tile of particles into shared memory
    if (particle_idx < num_parts) {
        shared_particles[local_tid] = particles[particle_idx];
    }

    __syncthreads(); // Make sure all particles are loaded

    // Each thread processes exactly one particle (tiled approach)
    if (particle_idx < num_parts) {
        particle_t& my_particle = shared_particles[local_tid];

        // Zero out accelerations
        double ax = 0;
        double ay = 0;

        // Compute bin indices for this particle
        int bin_x = my_particle.x / bin_size;
        int bin_y = my_particle.y / bin_size;

        // Explore neighboring bins
        for (int dy = -1; dy <= 1; dy++) {
            #pragma unroll 3
            for (int dx = -1; dx <= 1; dx++) {
                int neighbor_bin_x = bin_x + dx;
                int neighbor_bin_y = bin_y + dy;

                // Skip if bins are out of bounds (using mask to avoid divergence)
                bool valid_bin = (neighbor_bin_x >= 0 && neighbor_bin_x < num_bins_x &&
                                  neighbor_bin_y >= 0 && neighbor_bin_y < num_bins_y);

                // Early exit without branching (for entire warp)
                if (__all_sync(__activemask(), !valid_bin))
                    continue;

                // Only process valid bins (threads for invalid bins will just execute without
                // effect)
                int neighbor_bin_index =
                    valid_bin ? (neighbor_bin_x + num_bins_x * neighbor_bin_y) : 0;

                int bin_start = valid_bin ? bin_indices_gpu[neighbor_bin_index] : 0;
                int bin_end = valid_bin ? bin_indices_gpu[neighbor_bin_index + 1] : 0;

                // Process particles in neighboring bin
                #pragma unroll 4
                for (int j = bin_start; j < bin_end; j++) {
                    int neighbor_idx = bins_gpu[j];

                    // Check if neighbor is in our tile (shared memory)
                    bool in_tile = (neighbor_idx >= block_start) &&
                                   (neighbor_idx < block_start + blockDim.x) &&
                                   (neighbor_idx < num_parts);

                    double nx, ny;
                    if (in_tile) { // Shared memory
                        particle_t& neighbor = shared_particles[neighbor_idx - block_start];
                        nx = neighbor.x;
                        ny = neighbor.y;
                    } else { // Global memory
                        particle_t& neighbor = particles[neighbor_idx];
                        nx = neighbor.x;
                        ny = neighbor.y;
                    }

                    // Calculate force using masked operations to avoid divergence
                    double dx = nx - my_particle.x;
                    double dy = ny - my_particle.y;
                    double r2 = dx * dx + dy * dy;

                    // Instead of branch: if (r2 > cutoff * cutoff) continue;
                    double mask = (r2 <= cutoff * cutoff) ? 1.0 : 0.0;
                    // Skip self interactions
                    mask *= (r2 >= 1e-9) ? 1.0 : 0.0;

                    // Apply minimum r2 without branching
                    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
                    double r = sqrt(r2);

                    // Calculate coefficient - multiply by mask to zero out particles beyond cutoff
                    double coef = mask * (1 - cutoff / r) / r2 / mass;

                    // Update accelerations
                    ax += coef * dx;
                    ay += coef * dy;
                }
            }
        }

        // Write back acceleration values
        particles[particle_idx].ax = ax;
        particles[particle_idx].ay = ay;
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
    tiled_blocks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    shared_mem_size = NUM_THREADS * sizeof(particle_t);

    bin_size = cutoff;
    num_bins_x = size / bin_size + 1;
    num_bins_y = size / bin_size + 1;
    num_bins = num_bins_x * num_bins_y;

    cudaMalloc((void**)&bin_indices_gpu, sizeof(int) * (num_bins + 1));
    cudaMalloc((void**)&bin_counters_gpu, sizeof(int) * (num_bins + 1));
    cudaMalloc((void**)&bins_gpu, sizeof(int) * num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
#ifdef BENCHMARK
    cudaEvent_t start1, stop1, start2, stop2, start3, stop3, start4, stop4;
    float binning_time, scan_time, assignment_time, compute_time;
#endif

#ifdef BENCHMARK
    // Create timing events
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventCreate(&start4);
    cudaEventCreate(&stop4);
#endif

// Find the number of particles in each bin
#ifdef BENCHMARK
    cudaEventRecord(start1, 0);
#endif
    zero_out_bin_counts<<<blks, NUM_THREADS>>>(bin_indices_gpu, bin_counters_gpu, num_bins,
                                               total_num_threads);
    compute_bin_counts_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, total_num_threads, bin_size,
                                                  num_bins_x, bin_indices_gpu);
#ifdef BENCHMARK
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&binning_time, start1, stop1);
#endif

// Compute starting indices of bins
#ifdef BENCHMARK
    cudaEventRecord(start2, 0);
#endif
    thrust::device_ptr<int> bin_counts_gpu_ptr(bin_indices_gpu);
    thrust::exclusive_scan(
        bin_counts_gpu_ptr, bin_counts_gpu_ptr + num_bins + 1,
        bin_counts_gpu_ptr); // This will convert bin_indices_gpu from the number of particles in
                             // each bin to the starting index of each bin
#ifdef BENCHMARK
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&scan_time, start2, stop2);
#endif

// Add particles to bins
#ifdef BENCHMARK
    cudaEventRecord(start3, 0);
#endif
    compute_bins_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, total_num_threads, bin_size,
                                            num_bins_x, bins_gpu, bin_counters_gpu,
                                            bin_indices_gpu);
#ifdef BENCHMARK
    cudaEventRecord(stop3, 0);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&assignment_time, start3, stop3);
#endif

// Compute forces and move particles
#ifdef BENCHMARK
    cudaEventRecord(start4, 0);
#endif
    compute_forces_gpu<<<tiled_blocks, NUM_THREADS, shared_mem_size>>>(
        parts, num_parts, bin_size, num_bins_x, num_bins_y, bins_gpu, bin_indices_gpu);
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
#ifdef BENCHMARK
    cudaEventRecord(stop4, 0);
    cudaEventSynchronize(stop4);
    cudaEventElapsedTime(&compute_time, start4, stop4);

    printf("Binning time: %f ms\n", binning_time);
    printf("Scan time: %f ms\n", scan_time);
    printf("Assignment time: %f ms\n", assignment_time);
    printf("Compute and move time: %f ms\n", compute_time);
    printf("Total time: %f ms\n", binning_time + scan_time + assignment_time + compute_time);
    printf("--------------------------------\n");

    // Cleanup timing events
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    cudaEventDestroy(start4);
    cudaEventDestroy(stop4);
#endif
}
