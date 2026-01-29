//***********************************************************************************
// 2018.04.01 created by Zexlus1126
//
//    Example 002
// This is a simple demonstration on calculating merkle root from merkle branch 
// and solving a block (#286819) which the information is downloaded from Block Explorer 
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <cstring>

#include <cassert>
#include <omp.h>
#include <sys/time.h>

#include "sha256.h"

// Timing utility
double get_wall_time() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}

////////////////////////   Block   /////////////////////

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;


////////////////////////   Utils   ///////////////////////

// Fast hex lookup table for optimized hex string parsing
// Pre-initialized at compile time for O(1) hex digit conversion
static const unsigned char HEX_LOOKUP[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 0-15
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 16-31
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 32-47
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0,  // 48-63 ('0'-'9' at 48-57)
    0, 10,11,12,13,14,15, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 64-79 ('A'-'F' at 65-70)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 80-95
    0, 10,11,12,13,14,15, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 96-111 ('a'-'f' at 97-102)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 112-127
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 128-143
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 144-159
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 160-175
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 176-191
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 192-207
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 208-223
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 224-239
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   // 240-255
};

//convert one hex-codec char to binary
unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
            return c-'0';
    }
    return 0;
}


// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len/2-1;

    for(; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

// Optimized version using lookup table (2-3× faster than decode())
// Inline for maximum performance in hot paths
inline void convert_string_to_little_endian_bytes_fast(unsigned char* out, const char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t b = string_len/2 - 1;
    for(size_t s = 0; s < string_len; s += 2, --b)
    {
        out[b] = (HEX_LOOKUP[(unsigned char)in[s]] << 4) |
                  HEX_LOOKUP[(unsigned char)in[s+1]];
    }
}

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char* hex, size_t len)
{
    for(int i=0;i<len;++i)
    {
        printf("%02x", hex[i]);
    }
}


// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i=len-1;i>=0;--i)
    {
        printf("%02x", hex[i]);
    }
}

int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

// Device version for GPU kernels - optimized with word-level comparison
// Compares 8 bytes at a time (uint64_t) instead of byte-by-byte for 8× speedup
__device__ int little_endian_bit_comparison_device(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // For 32-byte SHA-256 hashes: compare 4 chunks of 8 bytes each
    // This reduces iterations from 32→4, improving performance in critical mining path

    // Cast to uint64_t for 64-bit word comparisons (8 bytes at a time)
    // Start from highest address since little-endian comparison goes from MSB to LSB
    const uint64_t *a64 = (const uint64_t*)(a + byte_len - 8);
    const uint64_t *b64 = (const uint64_t*)(b + byte_len - 8);

    // Compare 4 words (32 bytes / 8 bytes per word = 4 iterations)
    // Unroll loop for better instruction-level parallelism
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        if (a64[-i] < b64[-i])
            return -1;
        else if (a64[-i] > b64[-i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp)
{

    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Hash   ///////////////////////

// Host version (CPU)
void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}

// Device version for GPU kernels
__device__ void double_sha256_device(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}


////////////////////   Merkle Root   /////////////////////


// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    double t_start = get_wall_time();

    size_t total_count = count; // merkle branch
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    double t_alloc = get_wall_time();

    // copy each branch to the list
    // Use fast hex parsing (2-3× faster than decode())
    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        //convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes_fast(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count*32;

    double t_parse = get_wall_time();

    // calculate merkle root (sequential - OpenMP had race conditions)
    int level = 0;
    while(total_count > 1)
    {
        double t_level_start = get_wall_time();

        // hash each pair
        int i;

        if(total_count % 2 == 1)  //odd,
        {
            memcpy(list[total_count], list[total_count-1], 32);
        }

        // Sequential processing to avoid race conditions
        // OpenMP was causing non-deterministic merkle roots
        int j;
        for(i=0, j=0;i<total_count;i+=2, ++j)
        {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256*)list[j], list[i], 64);
        }

        total_count = j;

        // Disabled for performance profiling
        // double t_level_end = get_wall_time();
        // if (level == 0) {
        //     printf("  Level %d: %d nodes, %.3f ms\n", level, count, (t_level_end - t_level_start) * 1000);
        // }
        level++;
    }

    double t_hash = get_wall_time();

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;

    double t_end = get_wall_time();

    // Disabled for performance profiling
    // printf("Merkle timing (tx=%d): alloc=%.3fms, parse=%.3fms, hash=%.3fms, total=%.3fms\n",
    //        count,
    //        (t_alloc - t_start) * 1000,
    //        (t_parse - t_alloc) * 1000,
    //        (t_hash - t_parse) * 1000,
    //        (t_end - t_start) * 1000);
}


////////////////////   GPU Mining Kernel   /////////////////////

// Constant memory for target value (read by all threads, never changes)
// Using constant memory provides broadcast and caching for uniform reads
__constant__ unsigned char c_target[32];

// Constant memory for SHA-256 midstate (first 64 bytes of block header pre-computed)
// This optimization avoids recomputing SHA-256 state for constant portion of block
__constant__ SHA256 c_midstate;

// Constant memory for last 12 bytes of block header (before nonce)
// Block structure: [64 bytes pre-computed] + [12 bytes constant] + [4 bytes nonce]
__constant__ unsigned char c_last_12_bytes[12];

// CUDA kernel for parallel nonce search with midstate optimization
// Each thread tests different nonce values using grid-stride loop
__global__ void mine_kernel(
    unsigned int *result_nonce,     // Output: the found nonce
    int *found_flag                 // Atomic flag for early termination
)
{
    // Shared memory for faster access (copied from constant memory once per block)
    // Shared memory is faster than constant memory for repeated non-uniform access
    __shared__ unsigned char s_target[32];
    __shared__ SHA256 s_midstate;
    __shared__ unsigned char s_last_12_bytes[12];

    // Cooperatively load constant memory -> shared memory
    // Use all threads in block for parallel loading
    int tid = threadIdx.x;

    // Load target array (32 bytes) - use multiple threads
    if (tid < 32)
    {
        s_target[tid] = c_target[tid];
    }

    // Load midstate (32 bytes for SHA256.b or 8 words for SHA256.h)
    // Copy as bytes for simplicity
    if (tid < 32)
    {
        s_midstate.b[tid] = c_midstate.b[tid];
    }

    // Load last_12_bytes (12 bytes)
    if (tid < 12)
    {
        s_last_12_bytes[tid] = c_last_12_bytes[tid];
    }

    // Synchronize to ensure all shared memory is loaded before any thread uses it
    __syncthreads();

    // Calculate unique starting nonce for this thread
    unsigned int nonce = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    // Loop counter for periodic flag checks (same for all threads - no divergence)
    unsigned int iteration = 0;

    // Grid-stride loop: each thread tests nonces at regular intervals
    // Thread 0: tests 0, stride, 2*stride, ...
    // Thread 1: tests 1, stride+1, 2*stride+1, ...
    for (; nonce <= 0xFFFFFFFF; nonce += stride, ++iteration)
    {
        // Warp-level early exit optimization (Priority 10 from checklist)
        // Only lane 0 of each warp reads global flag, then broadcasts to other lanes
        // This reduces global memory traffic from 32 reads/warp to 1 read/warp
        if ((iteration & 0x3FF) == 0) {  // Check every 1024 iterations
            int lane_id = threadIdx.x & 31;  // Lane ID within warp (0-31)
            int should_exit;

            if (lane_id == 0) {
                // Only lane 0 reads the global flag
                should_exit = *found_flag;
            }

            // Broadcast lane 0's value to all lanes in the warp using shuffle
            should_exit = __shfl_sync(0xFFFFFFFF, should_exit, 0);

            if (should_exit) return;
        }

        // ===== First SHA-256: Use midstate optimization =====
        // Copy pre-computed midstate from SHARED memory (faster than constant memory)
        SHA256 hash1 = s_midstate;

        // Build last 16 bytes: [12 constant bytes] + [4-byte nonce]
        unsigned char last_16_bytes[16];

        // Copy the 12 constant bytes from SHARED memory (bytes 64-75 of block header)
        for (int i = 0; i < 12; i++)
        {
            last_16_bytes[i] = s_last_12_bytes[i];
        }

        // Add nonce as last 4 bytes (little-endian)
        last_16_bytes[12] = (nonce      ) & 0xFF;
        last_16_bytes[13] = (nonce >> 8 ) & 0xFF;
        last_16_bytes[14] = (nonce >> 16) & 0xFF;
        last_16_bytes[15] = (nonce >> 24) & 0xFF;

        // Finalize first SHA-256 (processes last 16 bytes + padding)
        sha256_finalize_80(&hash1, last_16_bytes);

        // ===== Second SHA-256: Normal hash of 32-byte result =====
        SHA256 hash2;
        sha256(&hash2, hash1.b, 32);

        // Check if hash is less than target (valid proof-of-work)
        // Using s_target from SHARED memory (faster than constant memory)
        if (little_endian_bit_comparison_device(hash2.b, s_target, 32) < 0)
        {
            // Found valid nonce! Use atomic to ensure we record the smallest valid nonce
            atomicMin(result_nonce, nonce);
            *found_flag = 1;
            return;
        }

        // Check for overflow to prevent infinite loop
        if (nonce > 0xFFFFFFFF - stride) break;
    }
}


void solve(FILE *fin, FILE *fout, cudaStream_t stream = 0, int blockSize = 256, int gridSize = 8192)
{
    double t_solve_start = get_wall_time();

    // **** read data *****
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);
    // printf("start hashing (tx=%d)", tx);  // Disabled for performance profiling

    double t_read = get_wall_time();

    raw_merkle_branch = new char [tx * 65];
    merkle_branch = new char *[tx];
    for(int i=0;i<tx;++i)
    {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    double t_read_tx = get_wall_time();

    // **** calculate merkle root ****

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    double t_merkle_end = get_wall_time();

    // Disabled for performance profiling
    // printf("merkle root(little): ");
    // print_hex(merkle_root, 32);
    // printf("\n");
    // printf("merkle root(big):    ");
    // print_hex_inverse(merkle_root, 32);
    // printf("\n");

    // **** solve block ****
    // printf("Block info (big): \n");
    // printf("  version:  %s\n", version);
    // printf("  pervhash: %s\n", prevhash);
    // printf("  merkleroot: "); print_hex_inverse(merkle_root, 32); printf("\n");
    // printf("  nbits:    %s\n", nbits);
    // printf("  ntime:    %s\n", ntime);
    // printf("  nonce:    ???\n\n");

    HashBlock block;

    // convert to byte array in little-endian
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash,                  prevhash,    64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits,   nbits,     8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime,   ntime,     8);
    block.nonce = 0;
    
    
    // ********** calculate target value *********
    // calculate target value from encoded difficulty which is encoded on "nbits"
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    
    // little-endian
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));
    
    
    // Disabled for performance profiling
    // printf("Target value (big): ");
    // print_hex_inverse(target_hex, 32);
    // printf("\n");


    // ********** find nonce using GPU **************

    // ===== Pre-compute SHA-256 midstate for first 64 bytes =====
    // This is the Bitcoin mining optimization: only nonce changes, so we can
    // pre-compute the SHA-256 state after processing the first 64 bytes
    SHA256 midstate;
    sha256_init(&midstate);
    sha256_update_64(&midstate, (unsigned char*)&block);

    // printf("Midstate computed (first 64 bytes of block header)\n");  // Disabled for profiling

    // Extract last 12 bytes of block (before nonce): bytes 64-75
    // Block layout: version(4) + prevhash(32) + merkle_root(32) + ntime(4) + nbits(4) + nonce(4)
    // Bytes 64-75: last 12 bytes = merkle_root[28-31] + ntime[0-3] + nbits[0-3]
    unsigned char last_12_bytes[12];
    memcpy(last_12_bytes, ((unsigned char*)&block) + 64, 12);

    // Copy pre-computed data to constant memory (async for stream support)
    cudaMemcpyToSymbolAsync(c_target, target_hex, 32, 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_midstate, &midstate, sizeof(SHA256), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_last_12_bytes, last_12_bytes, 12, 0, cudaMemcpyHostToDevice, stream);

    // Allocate device memory for result and flag only
    unsigned int *d_result_nonce;
    int *d_found_flag;

    cudaMalloc(&d_result_nonce, sizeof(unsigned int));
    cudaMalloc(&d_found_flag, sizeof(int));

    // Initialize result values on host
    unsigned int h_result_nonce = 0xFFFFFFFF;  // Initialize to max value
    int h_found_flag = 0;

    // Copy data to device
    cudaMemcpy(d_result_nonce, &h_result_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_flag, &h_found_flag, sizeof(int), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters (passed as function parameters)
    // blockSize = threads per block
    // gridSize = number of blocks

    // Disabled for performance profiling
    // printf("Launching GPU kernel with %d blocks x %d threads = %d total threads\n",
    //        gridSize, blockSize, gridSize * blockSize);
    // printf("Using midstate optimization (pre-computed first 64 bytes)\n");

    double t_gpu_start = get_wall_time();

    // Launch the mining kernel with midstate optimization (async on stream)
    mine_kernel<<<gridSize, blockSize, 0, stream>>>(d_result_nonce, d_found_flag);

    // For streams: DON'T synchronize here - let kernel run async
    // Caller will synchronize when needed
    // (For non-stream mode, we still need to wait)
    if (stream == 0) {
        // Default stream (0) is synchronous, wait for completion
        cudaDeviceSynchronize();
    }

    double t_gpu_end = get_wall_time();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // For stream mode: synchronize before copying results
    // This ensures kernel has finished before we read results
    if (stream != 0) {
        cudaStreamSynchronize(stream);
    }

    // Copy result back to host
    cudaMemcpy(&h_result_nonce, d_result_nonce, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory (target in constant memory doesn't need freeing)
    cudaFree(d_result_nonce);
    cudaFree(d_found_flag);

    // Set the found nonce
    _Pragma("GCC diagnostic push")
_Pragma("GCC diagnostic ignored \"-Wvoid-pointer-to-int-cast\"")
    block.nonce = h_result_nonce;
_Pragma("GCC diagnostic pop")


    // Verify the solution by computing the hash
    SHA256 sha256_ctx;
    double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));

    // Disabled for performance profiling
    // if (h_found_flag) {
    //     printf("Found Solution!!\n");
    //     printf("hash #%10u (big): ", block.nonce);
    //     print_hex_inverse(sha256_ctx.b, 32);
    //     printf("\n\n");
    // } else {
    //     printf("No solution found in nonce space\n");
    // }

    // print result
    // printf("hash(little): ");
    // print_hex(sha256_ctx.b, 32);
    // printf("\n");
    // printf("hash(big):    ");
    // print_hex_inverse(sha256_ctx.b, 32);
    // printf("\n\n");

    for(int i=0;i<4;++i)
    {
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    double t_solve_end = get_wall_time();

    // Profiling output (enable for performance analysis)
    #ifdef ENABLE_PROFILING
    printf("=== DETAILED PROFILING ===\n");
    printf("  Read file I/O:      %8.3f ms  (%5.2f%%)\n",
           (t_read - t_solve_start) * 1000,
           100.0 * (t_read - t_solve_start) / (t_solve_end - t_solve_start));
    printf("  Read tx hashes:     %8.3f ms  (%5.2f%%)\n",
           (t_read_tx - t_read) * 1000,
           100.0 * (t_read_tx - t_read) / (t_solve_end - t_solve_start));
    printf("  Merkle root calc:   %8.3f ms  (%5.2f%%)\n",
           (t_merkle_end - t_read_tx) * 1000,
           100.0 * (t_merkle_end - t_read_tx) / (t_solve_end - t_solve_start));
    printf("  CPU preprocessing:  %8.3f ms  (%5.2f%%)\n",
           (t_gpu_start - t_merkle_end) * 1000,
           100.0 * (t_gpu_start - t_merkle_end) / (t_solve_end - t_solve_start));
    printf("  GPU kernel:         %8.3f ms  (%5.2f%%) *** CRITICAL PATH ***\n",
           (t_gpu_end - t_gpu_start) * 1000,
           100.0 * (t_gpu_end - t_gpu_start) / (t_solve_end - t_solve_start));
    printf("  GPU->CPU transfer:  %8.3f ms  (%5.2f%%)\n",
           (t_solve_end - t_gpu_end) * 1000,
           100.0 * (t_solve_end - t_gpu_end) / (t_solve_end - t_solve_start));
    printf("  ---\n");
    printf("  TOTAL solve():      %8.3f ms  (100.00%%)\n",
           (t_solve_end - t_solve_start) * 1000);
    printf("==========================\n\n");
    #endif

    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: cuda_miner <in> <out> [blockSize] [gridSize]\n");
        fprintf(stderr, "  blockSize: threads per block (default: adaptive)\n");
        fprintf(stderr, "  gridSize: number of blocks (default: 2048)\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Adaptive mode (no blockSize specified):\n");
        fprintf(stderr, "  - Multi-block cases: 256×2048 (optimal for case01: 9.45s)\n");
        fprintf(stderr, "  - Single-block cases: 128×2048 (optimal for case02: 5.51s)\n");
        return 1;
    }
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    // First, read the number of blocks to determine optimal configuration
    int totalblock;
    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    // Parse optional blockSize and gridSize arguments
    // If not provided, use adaptive selection based on test case characteristics
    int blockSize = -1;  // -1 indicates adaptive mode
    int gridSize = -1;

    if (argc >= 4) {
        blockSize = atoi(argv[3]);
    }
    if (argc >= 5) {
        gridSize = atoi(argv[4]);
    }

    // Adaptive configuration based on experimental results:
    // - Multi-block cases (totalblock > 1): BlockSize=256 optimal
    //   Example: case01 (4 blocks) → 256×2048 = 9.451s (best)
    // - Single-block cases (totalblock == 1): BlockSize=128 optimal
    //   Example: case02 (1 block) → 128×2048 = 5.513s (best)
    if (blockSize == -1) {
        if (totalblock > 1) {
            blockSize = 256;  // Multi-block: larger block size for sustained computation
            gridSize = (gridSize == -1) ? 2048 : gridSize;
        } else {
            blockSize = 128;  // Single-block: smaller block size for lower overhead
            gridSize = (gridSize == -1) ? 2048 : gridSize;
        }
    } else {
        // Manual override provided, use default gridSize if not specified
        gridSize = (gridSize == -1) ? 2048 : gridSize;
    }

    // printf("=== CONFIGURATION ===\n");
    // printf("Test case: %d block(s)\n", totalblock);
    // printf("Block size (threads/block): %d\n", blockSize);
    // printf("Grid size (blocks): %d\n", gridSize);
    // printf("Total threads: %d\n", blockSize * gridSize);
    // printf("Mode: %s\n", (argc >= 4) ? "Manual" : "Adaptive");
    // printf("====================\n\n");

    // Create CUDA streams for pipelining multi-block processing
    // Use 2 streams for double-buffering: while GPU processes block N on stream 0,
    // CPU can prepare block N+1 which will run on stream 1
    const int NUM_STREAMS = 2;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // printf("=== CUDA STREAMS ENABLED ===\n");
    // printf("Using %d streams for pipelined multi-block processing\n", NUM_STREAMS);
    // printf("This allows GPU kernel for block N to overlap with CPU prep for block N+1\n");
    // printf("============================\n\n");

    // Process blocks with stream pipelining
    // Strategy: Launch block i on stream i%NUM_STREAMS
    // Before launching block i, synchronize stream i%NUM_STREAMS to ensure
    // previous block using that stream has completed
    for(int i=0;i<totalblock;++i)
    {
        // Use round-robin stream assignment
        cudaStream_t stream = streams[i % NUM_STREAMS];

        // If this is not the first use of this stream, wait for previous block to finish
        // This ensures: kernel for block i-NUM_STREAMS has finished before we reuse the stream
        // Overlap: while we prepare block i, block i-1 may still be running on a different stream
        if (i >= NUM_STREAMS) {
            cudaStreamSynchronize(stream);
        }

        solve(fin, fout, stream, blockSize, gridSize);
    }

    // Final synchronization: wait for all remaining kernels to complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}

