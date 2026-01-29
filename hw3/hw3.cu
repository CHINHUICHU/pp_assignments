#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <lodepng.h>

// Mandelbulb renderer parameters mirrored from the CPU implementation.
struct RenderParams {
    int AA;
    double power;
    int md_iter;
    int ray_step;
    int shadow_step;
    double step_limiter;
    double ray_multiplier;
    double bailout;
    double eps;
    double FOV;
    double far_plane;
};

constexpr double PI = 3.1415926535897932384626433832795;

#define CUDA_CHECK(call)                                                                \
    do {                                                                                \
        cudaError_t _err = (call);                                                      \
        if (_err != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(_err));                                          \
            std::exit(EXIT_FAILURE);                                                    \
        }                                                                               \
    } while (0)

// --- simple double3 helpers ------------------------------------------------------------
__host__ __device__ inline double3 make_double3_scalar(double s) {
    return make_double3(s, s, s);
}

__host__ __device__ inline double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline double3 operator*(const double3& a, double b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline double3 operator*(double b, const double3& a) {
    return a * b;
}

__host__ __device__ inline double3 operator/(const double3& a, double b) {
    double inv = 1.0 / b;
    return make_double3(a.x * inv, a.y * inv, a.z * inv);
}

__host__ __device__ inline double3 operator*(const double3& a, const double3& b) {
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline double3& operator+=(double3& a, const double3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline double3& operator*=(double3& a, const double3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

__host__ __device__ inline double3& operator/=(double3& a, double b) {
    double inv = 1.0 / b;
    a.x *= inv;
    a.y *= inv;
    a.z *= inv;
    return a;
}

__host__ __device__ inline double dot(const double3& a, const double3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline double3 cross(const double3& a, const double3& b) {
    return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}

__host__ __device__ inline double length(const double3& v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ inline double3 normalize(const double3& v) {
    double len = length(v);
    if (len > 0.0) return v / len;
    return make_double3(0.0, 0.0, 0.0);
}

__host__ __device__ inline double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__host__ __device__ inline double3 clampVec(const double3& v, double lo, double hi) {
    return make_double3(clampd(v.x, lo, hi), clampd(v.y, lo, hi), clampd(v.z, lo, hi));
}

__host__ __device__ inline double3 powVec(const double3& base, const double3& exp) {
    return make_double3(pow(clampd(base.x, 0.0, 1e9), exp.x),
                        pow(clampd(base.y, 0.0, 1e9), exp.y),
                        pow(clampd(base.z, 0.0, 1e9), exp.z));
}

__host__ __device__ inline double3 cosVec(const double3& v) {
    return make_double3(cos(v.x), cos(v.y), cos(v.z));
}

__host__ __device__ inline float3 make_float3_scalar(float s) {
    return make_float3(s, s, s);
}

__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline float3 operator*(float b, const float3& a) {
    return a * b;
}

__host__ __device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    if (len > 0.0f) return v * (1.0f / len);
    return make_float3(0.0f, 0.0f, 0.0f);
}

__host__ __device__ inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__host__ __device__ inline float3 clampVec(const float3& v, float lo, float hi) {
    return make_float3(clampf(v.x, lo, hi), clampf(v.y, lo, hi), clampf(v.z, lo, hi));
}

__host__ __device__ inline float3 powVec(const float3& base, const float3& exp) {
    return make_float3(powf(clampf(base.x, 0.0f, 1e9f), exp.x),
                       powf(clampf(base.y, 0.0f, 1e9f), exp.y),
                       powf(clampf(base.z, 0.0f, 1e9f), exp.z));
}

__host__ __device__ inline float3 powVec(const float3& base, float exp) {
    return make_float3(powf(clampf(base.x, 0.0f, 1e9f), exp),
                       powf(clampf(base.y, 0.0f, 1e9f), exp),
                       powf(clampf(base.z, 0.0f, 1e9f), exp));
}

__host__ __device__ inline float3 cosVec(const float3& v) {
    return make_float3(cosf(v.x), cosf(v.y), cosf(v.z));
}

__device__ inline float3 pal(float t, const float3& a, const float3& b, const float3& c,
                             const float3& d) {
    float3 term = c * t + d;
    return a + b * cosVec(term * (2.0f * static_cast<float>(PI)));
}

// --- distance estimator and helpers ----------------------------------------------------
__device__ inline double mandelbulbDEInternal(double3 p, double& trap,
                                              const RenderParams& params, int maxIter) {
    double3 v = p;
    double dr = 1.0;
    double r = length(v);
    trap = r;

    float power_f = static_cast<float>(params.power);
    for (int i = 0; i < maxIter; ++i) {
        float vx_f = static_cast<float>(v.x);
        float vy_f = static_cast<float>(v.y);
        float vz_f = static_cast<float>(v.z);
        float theta_f = atan2f(vy_f, vx_f);
        float r_f = static_cast<float>(r);
        float invR_f = r_f > 0.0f ? 1.0f / r_f : 0.0f;
        float phi_f = asinf(clampf(vz_f * invR_f, -1.0f, 1.0f));

        float sinTheta_f, cosTheta_f;
        sincosf(theta_f * power_f, &sinTheta_f, &cosTheta_f);
        float sinPhi_f, cosPhi_f;
        sincosf(phi_f * power_f, &sinPhi_f, &cosPhi_f);

        double sinTheta = static_cast<double>(sinTheta_f);
        double cosTheta = static_cast<double>(cosTheta_f);
        double sinPhi = static_cast<double>(sinPhi_f);
        double cosPhi = static_cast<double>(cosPhi_f);

        float zr_f = powf(r_f, power_f);
        double zr = static_cast<double>(zr_f);
        double zrMinus = (r > 0.0) ? static_cast<double>(zr_f * invR_f) : 0.0;
        dr = params.power * zrMinus * dr + 1.0;

        v = p + zr * make_double3(cosTheta * cosPhi, sinTheta * cosPhi, -sinPhi);
        trap = fmin(trap, r);
        r = length(v);
        if (r > params.bailout) break;
    }
    return 0.5 * log(r) * r / dr;
}

__device__ inline double mandelbulbDE(double3 p, double& trap, const RenderParams& params) {
    return mandelbulbDEInternal(p, trap, params, params.md_iter);
}

__device__ inline double mapInternal(double3 p, double& trap, int& ID,
                                     const RenderParams& params, int maxIter) {
    double c = cos(PI * 0.5);
    double s = sin(PI * 0.5);
    double3 rp = make_double3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
    ID = 1;
    return mandelbulbDEInternal(rp, trap, params, maxIter);
}

__device__ inline double map(double3 p, double& trap, int& ID, const RenderParams& params) {
    return mapInternal(p, trap, ID, params, params.md_iter);
}

__device__ inline double map(double3 p, const RenderParams& params) {
    double trap;
    int id;
    return mapInternal(p, trap, id, params, params.md_iter);
}

__device__ inline double mapWithIter(double3 p, const RenderParams& params, int maxIter) {
    double trap;
    int id;
    return mapInternal(p, trap, id, params, maxIter);
}

__device__ inline double3 pal(double t, const double3& a, const double3& b, const double3& c,
                              const double3& d) {
    double3 term = c * t + d;
    return a + b * cosVec(term * (2.0 * PI));
}

__device__ inline double softshadow(double3 ro, double3 rd, double k,
                                    const RenderParams& params) {
    double res = 1.0;
    double t = 0.0;
    for (int i = 0; i < params.shadow_step; ++i) {
        double h = map(ro + rd * t, params);
        res = fmin(res, k * h / t);
        if (res < 0.02) return 0.02;
        t += clampd(h, 0.001, params.step_limiter);
    }
    return clampd(res, 0.02, 1.0);
}

__device__ inline double3 calcNor(double3 p, double baseDist, const RenderParams& params) {
    double3 ex = make_double3(params.eps, 0.0, 0.0);
    double3 ey = make_double3(0.0, params.eps, 0.0);
    double3 ez = make_double3(0.0, 0.0, params.eps);

    int normalIter = params.md_iter > 12 ? params.md_iter / 2 : params.md_iter;
    double dx =
        mapWithIter(p + ex, params, normalIter) - mapWithIter(p - ex, params, normalIter);
    double dy =
        mapWithIter(p + ey, params, normalIter) - mapWithIter(p - ey, params, normalIter);
    double dz =
        mapWithIter(p + ez, params, normalIter) - mapWithIter(p - ez, params, normalIter);

    return normalize(make_double3(dx, dy, dz));
}

__device__ inline double trace(double3 ro, double3 rd, double& trap, int& ID,
                               const RenderParams& params) {
    double t = 0.0;
    double len = 0.0;
    for (int i = 0; i < params.ray_step; ++i) {
        len = map(ro + rd * t, trap, ID, params);
        if (fabs(len) < params.eps || t > params.far_plane) break;
        t += len * params.ray_multiplier;
    }
    return t < params.far_plane ? t : -1.0;
}

// --- kernels ---------------------------------------------------------------------------
constexpr int kSampleMinBlocks = 7;
constexpr int kSampleBlockSize = 128;

__global__ __launch_bounds__(kSampleBlockSize, kSampleMinBlocks)
void traceKernel(double* __restrict__ distances, double* __restrict__ traps,
                 double3* __restrict__ directions, unsigned char* __restrict__ hits, int width,
                 int height, double3 camera_pos, double3 cf, double3 cs, double3 cu,
                 RenderParams params) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int samplesPerPixel = params.AA * params.AA;
    size_t totalSamples = static_cast<size_t>(width) * height * samplesPerPixel;
    if (idx >= totalSamples) return;

    size_t pixelIndex = idx / samplesPerPixel;
    int sampleIndex = static_cast<int>(idx % samplesPerPixel);
    int px = static_cast<int>(pixelIndex % width);
    int py = static_cast<int>(pixelIndex / width);
    int m = sampleIndex / params.AA;
    int n = sampleIndex % params.AA;

    double invAA = 1.0 / static_cast<double>(params.AA);
    double sampleX = static_cast<double>(px) + static_cast<double>(m) * invAA;
    double sampleY = static_cast<double>(py) + static_cast<double>(n) * invAA;

    double iResX = static_cast<double>(width);
    double iResY = static_cast<double>(height);

    double uvx = (-iResX + 2.0 * sampleX) / iResY;
    double uvy = (-iResY + 2.0 * sampleY) / iResY;
    uvy = -uvy;

    double3 rd = normalize(cs * uvx + cu * uvy + cf * params.FOV);

    double trap = 0.0;
    int objID = 0;
    double dist = trace(camera_pos, rd, trap, objID, params);

    distances[idx] = dist;
    traps[idx] = trap;
    directions[idx] = rd;
    hits[idx] = (dist >= 0.0) ? 1 : 0;
}

__global__ __launch_bounds__(kSampleBlockSize, kSampleMinBlocks)
void shadeKernel(const double* __restrict__ distances, const double* __restrict__ traps,
                 const double3* __restrict__ directions, const unsigned char* __restrict__ hits,
                 float3* __restrict__ sampleColors, int width, int height, double3 camera_pos,
                 double3 sd, double3 sc, RenderParams params) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int samplesPerPixel = params.AA * params.AA;
    size_t totalSamples = static_cast<size_t>(width) * height * samplesPerPixel;
    if (idx >= totalSamples) return;

    if (hits[idx] == 0) {
        sampleColors[idx] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    const double dist = distances[idx];
    const float trap_f = static_cast<float>(traps[idx]);
    float3 sd_f = make_float3(static_cast<float>(sd.x), static_cast<float>(sd.y),
                              static_cast<float>(sd.z));
    float3 sc_f = make_float3(static_cast<float>(sc.x), static_cast<float>(sc.y),
                              static_cast<float>(sc.z));

    float3 nr_f;
    float3 rd_f;
    double shadow;
    {
        double3 rd = directions[idx];
        double3 pos = camera_pos + rd * dist;
        double3 nr = calcNor(pos, dist, params);
        shadow = softshadow(pos + nr * 0.001, sd, 16.0, params);
        rd_f = make_float3(static_cast<float>(rd.x), static_cast<float>(rd.y),
                           static_cast<float>(rd.z));
        nr_f = make_float3(static_cast<float>(nr.x), static_cast<float>(nr.y),
                           static_cast<float>(nr.z));
    }

    float3 hal = normalize(sd_f - rd_f);

    float3 col = pal(trap_f - 0.4f, make_float3_scalar(0.5f), make_float3_scalar(0.5f),
                     make_float3(1.0f, 1.0f, 1.0f), make_float3(0.0f, 0.1f, 0.2f));

    float3 ambc = make_float3_scalar(0.3f);
    constexpr float gloss = 32.0f;
    float logTrap = logf(fmaxf(trap_f, 1e-8f));
    float amb =
        (0.7f + 0.3f * nr_f.y) * (0.2f + 0.8f * clampf(0.05f * logTrap, 0.0f, 1.0f));
    float dif = clampf(dot(sd_f, nr_f), 0.0f, 1.0f) * static_cast<float>(shadow);
    float spe = powf(clampf(dot(nr_f, hal), 0.0f, 1.0f), gloss) * dif;

    float3 lin = make_float3(0.0f, 0.0f, 0.0f);
    lin += ambc * (0.05f + 0.95f * amb);
    lin += sc_f * (dif * 0.8f);

    col *= lin;
    col = powVec(col, make_float3(0.7f, 0.9f, 1.0f));
    col += make_float3_scalar(spe * 0.8f);
    col = clampVec(powVec(col, 0.4545f), 0.0f, 1.0f);

    sampleColors[idx] = col;
}

__global__ void finalizeKernel(unsigned char* image, const float3* sampleColors, int width,
                               int height, int samplesPerPixel) {
    size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalPixels = static_cast<size_t>(width) * height;
    if (pixelIndex >= totalPixels) return;

    size_t base = pixelIndex * samplesPerPixel;
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < samplesPerPixel; ++i) {
        float3 sample = sampleColors[base + i];
        sum.x += sample.x;
        sum.y += sample.y;
        sum.z += sample.z;
    }
    float invSamples = 1.0f / static_cast<float>(samplesPerPixel);
    float3 avg = make_float3(sum.x * invSamples, sum.y * invSamples, sum.z * invSamples);

    size_t idx = pixelIndex * 4;
    image[idx + 0] = static_cast<unsigned char>(fminf(fmaxf(avg.x, 0.0f), 1.0f) * 255.0f);
    image[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(avg.y, 0.0f), 1.0f) * 255.0f);
    image[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(avg.z, 0.0f), 1.0f) * 255.0f);
    image[idx + 3] = 255;
}

// --- host utilities -------------------------------------------------------------------
void write_png(const char* filename, const std::vector<unsigned char>& image, unsigned w,
               unsigned h) {
    unsigned error = lodepng_encode32_file(filename, image.data(), w, h);
    if (error) {
        fprintf(stderr, "png error %u: %s\n", error, lodepng_error_text(error));
    }
}

int main(int argc, char** argv) {
    if (argc != 10) {
        fprintf(stderr, "Usage: %s [x1 y1 z1] [x2 y2 z2] [width] [height] [output]\n", argv[0]);
        return EXIT_FAILURE;
    }

    double3 camera_pos = make_double3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    double3 target_pos = make_double3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    int width = atoi(argv[7]);
    int height = atoi(argv[8]);

    RenderParams params{};
    params.AA = 3;
    params.power = 8.0;
    params.md_iter = 24;
    params.ray_step = 10000;
    params.shadow_step = 1500;
    params.step_limiter = 0.2;
    params.ray_multiplier = 0.1;
    params.bailout = 2.0;
    params.eps = 0.0005;
    params.FOV = 1.5;
    params.far_plane = 100.0;

    double3 cf = normalize(target_pos - camera_pos);
    double3 cs = normalize(cross(cf, make_double3(0.0, 1.0, 0.0)));
    double3 cu = normalize(cross(cs, cf));
    double3 sd = normalize(camera_pos);
    double3 sc = make_double3(1.0, 0.9, 0.717);

    size_t totalPixels = static_cast<size_t>(width) * height;
    int samplesPerPixel = params.AA * params.AA;
    size_t totalSamples = totalPixels * samplesPerPixel;

    size_t imageBytes = totalPixels * 4;
    std::vector<unsigned char> hostImage(imageBytes);

    unsigned char* deviceImage = nullptr;
    double* distances = nullptr;
    double* traps = nullptr;
    double3* directions = nullptr;
    unsigned char* hits = nullptr;
    float3* sampleColors = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceImage, imageBytes));
    CUDA_CHECK(cudaMalloc(&distances, totalSamples * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&traps, totalSamples * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&directions, totalSamples * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&hits, totalSamples * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&sampleColors, totalSamples * sizeof(float3)));

    dim3 sampleBlock(kSampleBlockSize);
    dim3 sampleGrid((totalSamples + sampleBlock.x - 1) / sampleBlock.x);

    traceKernel<<<sampleGrid, sampleBlock>>>(distances, traps, directions, hits, width, height,
                                             camera_pos, cf, cs, cu, params);
    CUDA_CHECK(cudaGetLastError());

    shadeKernel<<<sampleGrid, sampleBlock>>>(distances, traps, directions, hits, sampleColors,
                                             width, height, camera_pos, sd, sc, params);
    CUDA_CHECK(cudaGetLastError());

    dim3 pixelBlock(128);
    dim3 pixelGrid((totalPixels + pixelBlock.x - 1) / pixelBlock.x);
    finalizeKernel<<<pixelGrid, pixelBlock>>>(deviceImage, sampleColors, width, height,
                                              samplesPerPixel);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(hostImage.data(), deviceImage, imageBytes, cudaMemcpyDeviceToHost));
    write_png(argv[9], hostImage, static_cast<unsigned>(width), static_cast<unsigned>(height));

    CUDA_CHECK(cudaFree(sampleColors));
    CUDA_CHECK(cudaFree(hits));
    CUDA_CHECK(cudaFree(directions));
    CUDA_CHECK(cudaFree(traps));
    CUDA_CHECK(cudaFree(distances));
    CUDA_CHECK(cudaFree(deviceImage));
    return EXIT_SUCCESS;
}
