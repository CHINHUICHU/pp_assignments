// nbody_hip.cpp
#include <hip/hip_runtime.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <omp.h>
#include <algorithm>

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
__device__ __host__ double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << hipGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ---------- I/O helpers ----------
void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<int>& is_device) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    is_device.resize(n);
    for (int i = 0; i < n; i++) {
        std::string type;
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type;
        is_device[i] = (type == "device") ? 1 : 0;
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

// ---------- Kernels ----------
__global__ void compute_forces_kernel(int n, const double* qx, const double* qy, const double* qz,
                                       const double* m, double* ax, double* ay, double* az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double ax_i = 0.0, ay_i = 0.0, az_i = 0.0;
    double qx_i = qx[i], qy_i = qy[i], qz_i = qz[i];

    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        double dx = qx[j] - qx_i;
        double dy = qy[j] - qy_i;
        double dz = qz[j] - qz_i;
        double dist_sq = dx * dx + dy * dy + dz * dz + param::eps * param::eps;
        double dist3 = dist_sq * sqrt(dist_sq);
        double mj = m[j];
        ax_i += param::G * mj * dx / dist3;
        ay_i += param::G * mj * dy / dist3;
        az_i += param::G * mj * dz / dist3;
    }

    ax[i] = ax_i;
    ay[i] = ay_i;
    az[i] = az_i;
}

__global__ void update_velocities_kernel(int n, double* vx, double* vy, double* vz,
                                          const double* ax, const double* ay, const double* az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vx[i] += ax[i] * param::dt;
    vy[i] += ay[i] * param::dt;
    vz[i] += az[i] * param::dt;
}

__global__ void update_positions_kernel(int n, double* qx, double* qy, double* qz,
                                         const double* vx, const double* vy, const double* vz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    qx[i] += vx[i] * param::dt;
    qy[i] += vy[i] * param::dt;
    qz[i] += vz[i] * param::dt;
}

// Fused baseline
__global__ void fused_forces_and_velocities_kernel(int n, const double* qx, const double* qy, const double* qz,
                                                     double* vx, double* vy, double* vz,
                                                     const double* m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double qx_i = qx[i], qy_i = qy[i], qz_i = qz[i];
    double vx_i = vx[i], vy_i = vy[i], vz_i = vz[i];

    double ax_i = 0.0, ay_i = 0.0, az_i = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        double dx = qx[j] - qx_i;
        double dy = qy[j] - qy_i;
        double dz = qz[j] - qz_i;
        double dist_sq = dx * dx + dy * dy + dz * dz + param::eps * param::eps;
        double dist3 = dist_sq * sqrt(dist_sq);
        double mj = m[j];
        ax_i += param::G * mj * dx / dist3;
        ay_i += param::G * mj * dy / dist3;
        az_i += param::G * mj * dz / dist3;
    }

    vx_i += ax_i * param::dt;
    vy_i += ay_i * param::dt;
    vz_i += az_i * param::dt;

    vx[i] = vx_i;
    vy[i] = vy_i;
    vz[i] = vz_i;
}

// Optimized fused kernel with shared memory tiling
// Optimizations: padded shared memory to avoid bank conflicts, masking to eliminate warp divergence
__global__ void fused_forces_and_velocities_kernel_optimized(int n, const double* qx, const double* qy, const double* qz,
                                                               double* vx, double* vy, double* vz,
                                                               const double* m) {
    extern __shared__ double shared_mem[];

    // Add padding to avoid bank conflicts (AMD GPUs have 32 banks, 64-byte cache lines)
    const int tile_size = blockDim.x;
    const int padded_size = tile_size + 1;  // Padding to avoid bank conflicts
    double* s_qx = &shared_mem[0];
    double* s_qy = &shared_mem[padded_size];
    double* s_qz = &shared_mem[2 * padded_size];
    double* s_m = &shared_mem[3 * padded_size];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (i >= n) return;

    double qx_i = qx[i];
    double qy_i = qy[i];
    double qz_i = qz[i];
    double vx_i = vx[i];
    double vy_i = vy[i];
    double vz_i = vz[i];

    double ax_i = 0.0;
    double ay_i = 0.0;
    double az_i = 0.0;

    int num_tiles = (n + tile_size - 1) / tile_size;
    for (int tile = 0; tile < num_tiles; tile++) {
        int j_base = tile * tile_size;
        int j = j_base + tid;

        if (j < n) {
            s_qx[tid] = qx[j];
            s_qy[tid] = qy[j];
            s_qz[tid] = qz[j];
            s_m[tid] = m[j];
        }

        __syncthreads();

        int tile_end = min(tile_size, n - j_base);
        #pragma unroll 8
        for (int jj = 0; jj < tile_end; jj++) {
            int j_global = j_base + jj;

            // Eliminate warp divergence: compute unconditionally and mask the result
            double qx_j = s_qx[jj];
            double qy_j = s_qy[jj];
            double qz_j = s_qz[jj];
            double m_j = s_m[jj];

            double dx = qx_j - qx_i;
            double dy = qy_j - qy_i;
            double dz = qz_j - qz_i;
            double dist_sq = dx * dx + dy * dy + dz * dz + param::eps * param::eps;
            double dist3 = dist_sq * sqrt(dist_sq);
            double inv_dist3 = 1.0 / dist3;
            double factor = param::G * m_j * inv_dist3;

            // Mask out self-interaction (avoids branch divergence)
            double mask = (j_global != i) ? 1.0 : 0.0;
            ax_i += mask * factor * dx;
            ay_i += mask * factor * dy;
            az_i += mask * factor * dz;
        }

        __syncthreads();
    }

    vx_i += ax_i * param::dt;
    vy_i += ay_i * param::dt;
    vz_i += az_i * param::dt;

    vx[i] = vx_i;
    vy[i] = vy_i;
    vz[i] = vz_i;
}

__global__ void update_device_masses_kernel(int n, double* m, const double* m0,
                                             const int* is_device, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (is_device[i]) {
        m[i] = param::gravity_device_mass(m0[i], step * param::dt);
    } else {
        m[i] = m0[i];
    }
}

// Atomic min for double using compare-and-swap
__device__ void atomicMinDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        unsigned long long newval = __double_as_longlong(fmin(val, __longlong_as_double(assumed)));
        old = atomicCAS(address_as_ull, assumed, newval);
    } while (assumed != old);
}

__global__ void compute_distance_kernel(const double* qx, const double* qy, const double* qz,
                                         int planet, int asteroid, double* dist) {
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    *dist = sqrt(dx * dx + dy * dy + dz * dz);
}

__global__ void compute_and_track_min_distance_kernel(const double* qx, const double* qy, const double* qz,
                                                       int planet, int asteroid, double* min_dist) {
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    double dist = sqrt(dx * dx + dy * dy + dz * dz);
    atomicMinDouble(min_dist, dist);
}

// Fused kernel: update positions and track minimum distance (for Problem 1)
__global__ void update_positions_and_track_min_dist_kernel(int n, double* qx, double* qy, double* qz,
                                                             const double* vx, const double* vy, const double* vz,
                                                             int planet, int asteroid, double* min_dist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // All threads update positions
    if (i < n) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }

    // Thread 0 computes and updates minimum distance
    // Use threadfence to ensure position updates are visible
    if (i == 0) {
        __threadfence_system();
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        atomicMinDouble(min_dist, dist);
    }
}

__global__ void check_collision_kernel(const double* qx, const double* qy, const double* qz,
                                        int planet, int asteroid, double planet_radius_sq, int* collision_flag) {
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    double dist_sq = dx * dx + dy * dy + dz * dz;
    if (dist_sq < planet_radius_sq) {
        atomicExch(collision_flag, 1);
    }
}

__global__ void check_missile_hits_kernel(int n, const double* qx, const double* qy, const double* qz,
                                           const int* is_device, int planet, int step,
                                           int* device_hit_times, int* newly_hit_devices, int* num_newly_hit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!is_device[i]) return;
    if (device_hit_times[i] != -1) return; // Already hit

    double qx_planet = qx[planet];
    double qy_planet = qy[planet];
    double qz_planet = qz[planet];
    double qx_device = qx[i];
    double qy_device = qy[i];
    double qz_device = qz[i];

    double dx = qx_planet - qx_device;
    double dy = qy_planet - qy_device;
    double dz = qz_planet - qz_device;
    double device_dist = sqrt(dx * dx + dy * dy + dz * dz);

    double missile_dist = step * param::dt * param::missile_speed;

    if (missile_dist > device_dist) {
        device_hit_times[i] = step;
        int idx = atomicAdd(num_newly_hit, 1);
        newly_hit_devices[idx] = i;
    }
}

// ---------- GPUState ----------
struct GPUState {
    double *d_qx, *d_qy, *d_qz;
    double *d_vx, *d_vy, *d_vz;
    double *d_m, *d_m0;
    double *d_ax, *d_ay, *d_az;
    int *d_is_device;
    double *d_dist;
    int *d_device_hit_times;
    int *d_newly_hit_devices;
    int *d_num_newly_hit;
    double *d_min_dist;
    int *d_collision_flag;
    int device_id;

    GPUState(int dev_id) : device_id(dev_id) {
        d_qx = d_qy = d_qz = nullptr;
        d_vx = d_vy = d_vz = nullptr;
        d_m = d_m0 = nullptr;
        d_ax = d_ay = d_az = nullptr;
        d_is_device = nullptr;
        d_dist = nullptr;
        d_device_hit_times = nullptr;
        d_newly_hit_devices = nullptr;
        d_num_newly_hit = nullptr;
        d_min_dist = nullptr;
        d_collision_flag = nullptr;
    }

    void allocate(int n) {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMalloc(&d_qx, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qy, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qz, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vx, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vy, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vz, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_m, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_m0, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_ax, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_ay, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_az, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_is_device, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_dist, sizeof(double)));
        HIP_CHECK(hipMalloc(&d_device_hit_times, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_newly_hit_devices, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_num_newly_hit, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_min_dist, sizeof(double)));
        HIP_CHECK(hipMalloc(&d_collision_flag, sizeof(int)));
    }

    void free_memory() {
        HIP_CHECK(hipSetDevice(device_id));
        if (d_qx) HIP_CHECK(hipFree(d_qx));
        if (d_qy) HIP_CHECK(hipFree(d_qy));
        if (d_qz) HIP_CHECK(hipFree(d_qz));
        if (d_vx) HIP_CHECK(hipFree(d_vx));
        if (d_vy) HIP_CHECK(hipFree(d_vy));
        if (d_vz) HIP_CHECK(hipFree(d_vz));
        if (d_m) HIP_CHECK(hipFree(d_m));
        if (d_m0) HIP_CHECK(hipFree(d_m0));
        if (d_ax) HIP_CHECK(hipFree(d_ax));
        if (d_ay) HIP_CHECK(hipFree(d_ay));
        if (d_az) HIP_CHECK(hipFree(d_az));
        if (d_is_device) HIP_CHECK(hipFree(d_is_device));
        if (d_dist) HIP_CHECK(hipFree(d_dist));
        if (d_device_hit_times) HIP_CHECK(hipFree(d_device_hit_times));
        if (d_newly_hit_devices) HIP_CHECK(hipFree(d_newly_hit_devices));
        if (d_num_newly_hit) HIP_CHECK(hipFree(d_num_newly_hit));
        if (d_min_dist) HIP_CHECK(hipFree(d_min_dist));
        if (d_collision_flag) HIP_CHECK(hipFree(d_collision_flag));
    }
};

// Checkpoint
struct Checkpoint {
    std::vector<double> qx, qy, qz;
    std::vector<double> vx, vy, vz;
    std::vector<double> m;
    int step;

    Checkpoint(int n) : qx(n), qy(n), qz(n), vx(n), vy(n), vz(n), m(n), step(0) {}
};

// ---------- main ----------
int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> is_device;

    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);

    int blockSize = 256;

    int numSMs = 0;
    hipError_t attr_err = hipDeviceGetAttribute(&numSMs, hipDeviceAttributeMultiprocessorCount, 0);
    if (attr_err != hipSuccess || numSMs <= 0) {
        numSMs = 80; // fallback
    }

    int numBlocks = (n + blockSize - 1) / blockSize;

    if (numBlocks < numSMs) {
        blockSize = std::min(256, std::max(32, n / numSMs));
        if (blockSize >= 256) blockSize = 256;
        else if (blockSize >= 128) blockSize = 128;
        else if (blockSize >= 64) blockSize = 64;
        else blockSize = 32;
        numBlocks = (n + blockSize - 1) / blockSize;
    }

    // Allocate padded shared memory to avoid bank conflicts
    size_t sharedMemSize = 4 * (blockSize + 1) * sizeof(double);

    double min_dist = std::numeric_limits<double>::infinity();
    int hit_time_step = -2;
    std::vector<int> device_hit_times(n, -1);
    std::vector<Checkpoint> device_checkpoints;
    std::vector<int> device_indices;

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            GPUState gpu0(0);
            gpu0.allocate(n);

            std::vector<double> m_problem1 = m;
            for (int i = 0; i < n; i++) {
                if (is_device[i]) m_problem1[i] = 0;
            }

            HIP_CHECK(hipSetDevice(0));
            HIP_CHECK(hipMemcpy(gpu0.d_qx, qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu0.d_qy, qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu0.d_qz, qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu0.d_vx, vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu0.d_vy, vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu0.d_vz, vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu0.d_m, m_problem1.data(), n * sizeof(double), hipMemcpyHostToDevice));

            double init_min = std::numeric_limits<double>::infinity();
            HIP_CHECK(hipMemcpy(gpu0.d_min_dist, &init_min, sizeof(double), hipMemcpyHostToDevice));

            // Compute initial distance at step 0
            compute_and_track_min_distance_kernel<<<1, 1>>>(gpu0.d_qx, gpu0.d_qy, gpu0.d_qz, planet, asteroid, gpu0.d_min_dist);
            HIP_CHECK(hipGetLastError());

            for (int step = 0; step < param::n_steps; step++) {
                hipLaunchKernelGGL((fused_forces_and_velocities_kernel_optimized), dim3(numBlocks), dim3(blockSize), sharedMemSize, 0,
                                   n, gpu0.d_qx, gpu0.d_qy, gpu0.d_qz, gpu0.d_vx, gpu0.d_vy, gpu0.d_vz, gpu0.d_m);
                HIP_CHECK(hipGetLastError());

                // Fused: update positions and track minimum distance
                update_positions_and_track_min_dist_kernel<<<numBlocks, blockSize>>>(
                    n, gpu0.d_qx, gpu0.d_qy, gpu0.d_qz, gpu0.d_vx, gpu0.d_vy, gpu0.d_vz,
                    planet, asteroid, gpu0.d_min_dist);
                HIP_CHECK(hipGetLastError());
            }

            HIP_CHECK(hipDeviceSynchronize());

            double local_min_dist;
            HIP_CHECK(hipMemcpy(&local_min_dist, gpu0.d_min_dist, sizeof(double), hipMemcpyDeviceToHost));
            min_dist = local_min_dist;

            gpu0.free_memory();
        }

        #pragma omp section
        {
            GPUState gpu1(1);
            gpu1.allocate(n);

            HIP_CHECK(hipSetDevice(1));
            HIP_CHECK(hipMemcpy(gpu1.d_qx, qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu1.d_qy, qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu1.d_qz, qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu1.d_vx, vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu1.d_vy, vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu1.d_vz, vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu1.d_m0, m.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu1.d_is_device, is_device.data(), n * sizeof(int), hipMemcpyHostToDevice));

            std::vector<int> init_hit_times(n, -1);
            HIP_CHECK(hipMemcpy(gpu1.d_device_hit_times, init_hit_times.data(), n * sizeof(int), hipMemcpyHostToDevice));

            int zero_flag = 0;
            HIP_CHECK(hipMemcpy(gpu1.d_collision_flag, &zero_flag, sizeof(int), hipMemcpyHostToDevice));

            int local_hit_time_step = -2;
            double planet_radius_sq = param::planet_radius * param::planet_radius;
            std::vector<int> local_device_hit_times(n, -1);
            std::vector<Checkpoint> local_checkpoints;
            std::vector<int> local_indices;

            for (int step = 0; step <= param::n_steps; step++) {
                if (step > 0) {
                    update_device_masses_kernel<<<numBlocks, blockSize>>>(n, gpu1.d_m, gpu1.d_m0, gpu1.d_is_device, step);
                    HIP_CHECK(hipGetLastError());

                    hipLaunchKernelGGL((fused_forces_and_velocities_kernel_optimized), dim3(numBlocks), dim3(blockSize), sharedMemSize, 0,
                                       n, gpu1.d_qx, gpu1.d_qy, gpu1.d_qz, gpu1.d_vx, gpu1.d_vy, gpu1.d_vz, gpu1.d_m);
                    HIP_CHECK(hipGetLastError());

                    update_positions_kernel<<<numBlocks, blockSize>>>(n, gpu1.d_qx, gpu1.d_qy, gpu1.d_qz, gpu1.d_vx, gpu1.d_vy, gpu1.d_vz);
                    HIP_CHECK(hipGetLastError());
                }

                int zero = 0;
                HIP_CHECK(hipMemcpy(gpu1.d_num_newly_hit, &zero, sizeof(int), hipMemcpyHostToDevice));

                check_missile_hits_kernel<<<numBlocks, blockSize>>>(n, gpu1.d_qx, gpu1.d_qy, gpu1.d_qz, gpu1.d_is_device,
                    planet, step, gpu1.d_device_hit_times, gpu1.d_newly_hit_devices, gpu1.d_num_newly_hit);
                HIP_CHECK(hipGetLastError());

                check_collision_kernel<<<1, 1>>>(gpu1.d_qx, gpu1.d_qy, gpu1.d_qz, planet, asteroid, planet_radius_sq, gpu1.d_collision_flag);
                HIP_CHECK(hipGetLastError());

                int collision_flag = 0;
                HIP_CHECK(hipMemcpy(&collision_flag, gpu1.d_collision_flag, sizeof(int), hipMemcpyDeviceToHost));

                if (collision_flag) {
                    local_hit_time_step = step;
                    HIP_CHECK(hipMemcpy(local_device_hit_times.data(), gpu1.d_device_hit_times, n * sizeof(int), hipMemcpyDeviceToHost));

                    // If collision happened, we break — user logic notes that we may need checkpoints earlier
                    break;
                }

                int num_newly_hit = 0;
                HIP_CHECK(hipMemcpy(&num_newly_hit, gpu1.d_num_newly_hit, sizeof(int), hipMemcpyDeviceToHost));

                if (num_newly_hit > 0) {
                    std::vector<int> newly_hit_devices(num_newly_hit);
                    HIP_CHECK(hipMemcpy(newly_hit_devices.data(), gpu1.d_newly_hit_devices,
                                         num_newly_hit * sizeof(int), hipMemcpyDeviceToHost));

                    for (int idx = 0; idx < num_newly_hit; idx++) {
                        int device_i = newly_hit_devices[idx];
                        local_device_hit_times[device_i] = step;

                        Checkpoint chk(n);
                        chk.step = step;
                        HIP_CHECK(hipMemcpy(chk.qx.data(), gpu1.d_qx, n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(chk.qy.data(), gpu1.d_qy, n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(chk.qz.data(), gpu1.d_qz, n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(chk.vx.data(), gpu1.d_vx, n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(chk.vy.data(), gpu1.d_vy, n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(chk.vz.data(), gpu1.d_vz, n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(chk.m.data(), gpu1.d_m, n * sizeof(double), hipMemcpyDeviceToHost));

                        local_checkpoints.push_back(chk);
                        local_indices.push_back(device_i);
                    }
                }
            }

            HIP_CHECK(hipDeviceSynchronize());

            #pragma omp critical
            {
                hit_time_step = local_hit_time_step;
                device_hit_times = local_device_hit_times;
                device_checkpoints = std::move(local_checkpoints);
                device_indices = std::move(local_indices);
            }

            gpu1.free_memory();
        }
    }

    // Problem 3: test devices using both GPUs with HIP streams for concurrent testing
    int gravity_device_id = -1;
    double missile_cost = std::numeric_limits<double>::infinity();
    double planet_radius_sq = param::planet_radius * param::planet_radius;

    // Number of concurrent device tests per GPU
    const int NUM_STREAMS = 4;

    #pragma omp parallel num_threads(2)
    {
        int thread_id = omp_get_thread_num();
        int gpu_id = thread_id;
        HIP_CHECK(hipSetDevice(gpu_id));

        // Create streams and GPU states for concurrent testing
        hipStream_t streams[NUM_STREAMS];
        GPUState* gpu_states[NUM_STREAMS];

        for (int s = 0; s < NUM_STREAMS; s++) {
            HIP_CHECK(hipStreamCreate(&streams[s]));
            gpu_states[s] = new GPUState(gpu_id);
            gpu_states[s]->allocate(n);

            // Copy static data (m0, is_device) to each GPU state
            HIP_CHECK(hipMemcpy(gpu_states[s]->d_m0, m.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu_states[s]->d_is_device, is_device.data(), n * sizeof(int), hipMemcpyHostToDevice));
        }

        int local_best_device = -1;
        double local_best_cost = std::numeric_limits<double>::infinity();

        // Process devices in batches of NUM_STREAMS
        // Each GPU handles devices with indices: thread_id, thread_id+2, thread_id+4, ...
        for (size_t batch_start = thread_id; batch_start < device_checkpoints.size(); batch_start += NUM_STREAMS * 2) {
            // Track which streams are active in this batch
            int active_streams = 0;
            int stream_device_idx[NUM_STREAMS];
            int stream_device_id[NUM_STREAMS];
            int stream_hit_time[NUM_STREAMS];

            // Launch device tests on available streams
            for (int s = 0; s < NUM_STREAMS; s++) {
                size_t idx = batch_start + s * 2;
                if (idx >= device_checkpoints.size()) break;

                int device_i = device_indices[idx];
                int hit_time = device_hit_times[device_i];

                // Skip if device hit after or at collision time
                if (hit_time_step >= 0 && hit_time >= hit_time_step) continue;

                // Store info for this stream
                stream_device_idx[active_streams] = idx;
                stream_device_id[active_streams] = device_i;
                stream_hit_time[active_streams] = hit_time;

                // Load checkpoint state to GPU on this stream
                Checkpoint& chk = device_checkpoints[idx];
                GPUState* gpu = gpu_states[active_streams];
                hipStream_t stream = streams[active_streams];

                HIP_CHECK(hipMemcpyAsync(gpu->d_qx, chk.qx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(gpu->d_qy, chk.qy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(gpu->d_qz, chk.qz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(gpu->d_vx, chk.vx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(gpu->d_vy, chk.vy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(gpu->d_vz, chk.vz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(gpu->d_m, chk.m.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));

                // Set device mass to 0 (destroy it)
                double zero_mass = 0.0;
                HIP_CHECK(hipMemcpyAsync(gpu->d_m + device_i, &zero_mass, sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(gpu->d_m0 + device_i, &zero_mass, sizeof(double), hipMemcpyHostToDevice, stream));

                active_streams++;
            }

            // Run simulations concurrently on all active streams with collision checking
            int check_until = (hit_time_step >= 0) ? hit_time_step + 1000 : param::n_steps;
            check_until = std::min(check_until, param::n_steps);

            // Initialize collision flags for each stream
            std::vector<int> collision_detected(active_streams, 0);

            for (int s = 0; s < active_streams; s++) {
                GPUState* gpu = gpu_states[s];
                hipStream_t stream = streams[s];
                int device_i = stream_device_id[s];
                int hit_time = stream_hit_time[s];

                // Initialize collision flag on GPU
                HIP_CHECK(hipMemcpyAsync(gpu->d_collision_flag, &collision_detected[s], sizeof(int), hipMemcpyHostToDevice, stream));

                // Run simulation from checkpoint and check collision at each step
                for (int step = hit_time + 1; step <= check_until; step++) {
                    update_device_masses_kernel<<<numBlocks, blockSize, 0, stream>>>(
                        n, gpu->d_m, gpu->d_m0, gpu->d_is_device, step);

                    hipLaunchKernelGGL((fused_forces_and_velocities_kernel_optimized),
                                       dim3(numBlocks), dim3(blockSize), sharedMemSize, stream,
                                       n, gpu->d_qx, gpu->d_qy, gpu->d_qz, gpu->d_vx, gpu->d_vy, gpu->d_vz, gpu->d_m);

                    update_positions_kernel<<<numBlocks, blockSize, 0, stream>>>(
                        n, gpu->d_qx, gpu->d_qy, gpu->d_qz, gpu->d_vx, gpu->d_vy, gpu->d_vz);

                    // Check collision on GPU
                    check_collision_kernel<<<1, 1, 0, stream>>>(
                        gpu->d_qx, gpu->d_qy, gpu->d_qz, planet, asteroid, planet_radius_sq, gpu->d_collision_flag);
                }

                // Copy collision flag back after simulation completes
                HIP_CHECK(hipMemcpyAsync(&collision_detected[s], gpu->d_collision_flag, sizeof(int), hipMemcpyDeviceToHost, stream));
            }

            // Synchronize all streams before checking results
            for (int s = 0; s < active_streams; s++) {
                HIP_CHECK(hipStreamSynchronize(streams[s]));
            }

            // Check results for all tested devices in this batch
            for (int s = 0; s < active_streams; s++) {
                int device_i = stream_device_id[s];
                int hit_time = stream_hit_time[s];

                // Collision prevented if collision flag is 0
                bool collision_prevented = (collision_detected[s] == 0);

                // Update best cost if collision prevented
                if (collision_prevented) {
                    double cost = param::get_missile_cost(hit_time * param::dt);
                    if (cost < local_best_cost) {
                        local_best_cost = cost;
                        local_best_device = device_i;
                    }
                }

                // Restore original mass value for next iteration
                GPUState* gpu = gpu_states[s];
                HIP_CHECK(hipMemcpy(gpu->d_m0 + device_i, &m[device_i], sizeof(double), hipMemcpyHostToDevice));
            }
        }

        // Cleanup
        for (int s = 0; s < NUM_STREAMS; s++) {
            gpu_states[s]->free_memory();
            delete gpu_states[s];
            HIP_CHECK(hipStreamDestroy(streams[s]));
        }

        #pragma omp critical
        {
            if (local_best_cost < missile_cost) {
                missile_cost = local_best_cost;
                gravity_device_id = local_best_device;
            }
        }
    }

    if (gravity_device_id == -1) missile_cost = -1;

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);

    return 0;
}
