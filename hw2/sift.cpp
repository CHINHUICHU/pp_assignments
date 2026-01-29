#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <tuple>
#include <cassert>
#include <omp.h>
#include <mpi.h>

#include "sift.hpp"
#include "image.hpp"



namespace {

constexpr int HALO_TAG_UP = 0;    // message sent to the rank above (smaller index)
constexpr int HALO_TAG_DOWN = 1;  // message sent to the rank below (larger index)


// ============================================================================
// 2D Block Partition Structures
// ============================================================================

struct Block2DPartition {
    int row_start, row_end;  // [start, end)
    int col_start, col_end;  // [start, end)
    int rows, cols;
    int proc_row, proc_col;  // Position in process grid
};

struct ProcessGrid {
    int rows, cols;  // Process grid dimensions
    int rank;
    int size;
    MPI_Comm cart_comm;  // Cartesian communicator
    
    // Neighbor ranks (MPI_PROC_NULL if at boundary)
    int north, south, east, west;
    int north_east, north_west, south_east, south_west;
};

// Create 2D Cartesian process grid
ProcessGrid create_process_grid_2d(int world_size, int world_rank) {
    ProcessGrid grid;
    grid.size = world_size;
    grid.rank = world_rank;
    
    // Find optimal grid dimensions (as square as possible)
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    grid.rows = dims[0];
    grid.cols = dims[1];
    
    // Create Cartesian communicator
    int periods[2] = {0, 0};  // Non-periodic (no wraparound)
    int reorder = 0;  // Keep original rank ordering
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid.cart_comm);
    
    // Get neighbors
    MPI_Cart_shift(grid.cart_comm, 0, 1, &grid.north, &grid.south);
    MPI_Cart_shift(grid.cart_comm, 1, 1, &grid.west, &grid.east);
    
    // Calculate diagonal neighbors manually
    int coords[2];
    MPI_Cart_coords(grid.cart_comm, world_rank, 2, coords);
    
    auto get_neighbor = [&](int row_offset, int col_offset) -> int {
        int new_coords[2] = {coords[0] + row_offset, coords[1] + col_offset};
        if (new_coords[0] < 0 || new_coords[0] >= grid.rows ||
            new_coords[1] < 0 || new_coords[1] >= grid.cols) {
            return MPI_PROC_NULL;
        }
        int rank;
        MPI_Cart_rank(grid.cart_comm, new_coords, &rank);
        return rank;
    };
    
    grid.north_west = get_neighbor(-1, -1);
    grid.north_east = get_neighbor(-1, +1);
    grid.south_west = get_neighbor(+1, -1);
    grid.south_east = get_neighbor(+1, +1);
    
    return grid;
}

// Compute 2D block partition for current process
Block2DPartition compute_2d_block_partition(int height, int width, 
                                            const ProcessGrid& grid) {
    Block2DPartition part;

    int coords[2];
    MPI_Cart_coords(grid.cart_comm, grid.rank, 2, coords);
    part.proc_row = coords[0];
    part.proc_col = coords[1];
    
    // Distribute rows
    int base_rows = height / grid.rows;
    int row_remainder = height % grid.rows;
    part.rows = base_rows + (part.proc_row < row_remainder ? 1 : 0);
    part.row_start = part.proc_row * base_rows + std::min(part.proc_row, row_remainder);
    part.row_end = part.row_start + part.rows;
    
    // Distribute columns
    int base_cols = width / grid.cols;
    int col_remainder = width % grid.cols;
    part.cols = base_cols + (part.proc_col < col_remainder ? 1 : 0);
    part.col_start = part.proc_col * base_cols + std::min(part.proc_col, col_remainder);
    part.col_end = part.col_start + part.cols;

    return part;
}

Block2DPartition compute_2d_block_partition_for_rank(int height, int width,
                                                     const ProcessGrid& grid,
                                                     int target_rank) {
    Block2DPartition part;

    int coords[2] = {0, 0};
    MPI_Cart_coords(grid.cart_comm, target_rank, 2, coords);
    part.proc_row = coords[0];
    part.proc_col = coords[1];

    int base_rows = height / grid.rows;
    int row_remainder = height % grid.rows;
    part.rows = base_rows + (part.proc_row < row_remainder ? 1 : 0);
    part.row_start = part.proc_row * base_rows + std::min(part.proc_row, row_remainder);
    part.row_end = part.row_start + part.rows;

    int base_cols = width / grid.cols;
    int col_remainder = width % grid.cols;
    part.cols = base_cols + (part.proc_col < col_remainder ? 1 : 0);
    part.col_start = part.proc_col * base_cols + std::min(part.proc_col, col_remainder);
    part.col_end = part.col_start + part.cols;

    return part;
}

// ============================================================================
// 2D Data Distribution Helpers
// ============================================================================

// Pack 2D block from global image
void pack_2d_block(const float* global_data, int global_width, int global_height,
                   float* local_block, const Block2DPartition& part, int channels) {
    for (int y = part.row_start; y < part.row_end; y++) {
        for (int x = part.col_start; x < part.col_end; x++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = (y * global_width + x) * channels + c;
                int dst_idx = ((y - part.row_start) * part.cols + 
                              (x - part.col_start)) * channels + c;
                local_block[dst_idx] = global_data[src_idx];
            }
        }
    }
}

// Unpack 2D block to global image
void unpack_2d_block(const float* local_block, const Block2DPartition& part,
                     float* global_data, int global_width, int global_height, int channels) {
    for (int y = part.row_start; y < part.row_end; y++) {
        for (int x = part.col_start; x < part.col_end; x++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = ((y - part.row_start) * part.cols + 
                              (x - part.col_start)) * channels + c;
                int dst_idx = (y * global_width + x) * channels + c;
                global_data[dst_idx] = local_block[src_idx];
            }
        }
    }
}

// Scatter 2D blocks from rank 0 to all processes
void scatter_2d_blocks(const float* global_data, int global_width, int global_height,
                       float* local_block, const Block2DPartition& my_part,
                       const ProcessGrid& grid, int channels, int world_rank) {
    int local_size = my_part.rows * my_part.cols * channels;
    MPI_Comm comm = grid.cart_comm;
    if (comm == MPI_COMM_NULL) {
        comm = MPI_COMM_WORLD;
    }

    std::vector<int> sendcounts(grid.size, 0);
    std::vector<int> sdispls(grid.size, 0);
    std::vector<MPI_Datatype> sendtypes(grid.size, MPI_DATATYPE_NULL);
    std::vector<int> recvcounts(grid.size, 0);
    std::vector<int> rdispls(grid.size, 0);
    std::vector<MPI_Datatype> recvtypes(grid.size, MPI_DATATYPE_NULL);
    std::vector<MPI_Datatype> committed_types;
    committed_types.reserve(grid.size);

    recvtypes[0] = MPI_FLOAT;
    if (local_size > 0) {
        recvcounts[0] = local_size;
    }

    if (world_rank == 0 && global_data != nullptr) {
        for (int rank = 0; rank < grid.size; rank++) {
            Block2DPartition part = compute_2d_block_partition_for_rank(global_height, global_width,
                                                                        grid, rank);
            int block_size = part.rows * part.cols * channels;
            if (block_size == 0) {
                continue;
            }

            int sizes[3] = {global_height, global_width, channels};
            int subsizes[3] = {part.rows, part.cols, channels};
            int starts[3] = {part.row_start, part.col_start, 0};

            MPI_Datatype subarray_type;
            MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                     MPI_FLOAT, &subarray_type);
            MPI_Type_commit(&subarray_type);

            sendcounts[rank] = 1;
            sendtypes[rank] = subarray_type;
            committed_types.push_back(subarray_type);
        }
    }

    MPI_Alltoallw(global_data,
                  sendcounts.data(), sdispls.data(), sendtypes.data(),
                  local_block,
                  recvcounts.data(), rdispls.data(), recvtypes.data(),
                  comm);

    for (MPI_Datatype type : committed_types) {
        MPI_Type_free(&type);
    }
}

// Gather 2D blocks from all processes to rank 0
void gather_2d_blocks(const float* local_block, const Block2DPartition& my_part,
                      float* global_data, int global_width, int global_height,
                      const ProcessGrid& grid, int channels, int world_rank) {
    int local_size = my_part.rows * my_part.cols * channels;
    MPI_Comm comm = grid.cart_comm;
    if (comm == MPI_COMM_NULL) {
        comm = MPI_COMM_WORLD;
    }

    std::vector<int> sendcounts(grid.size, 0);
    std::vector<int> sdispls(grid.size, 0);
    std::vector<MPI_Datatype> sendtypes(grid.size, MPI_DATATYPE_NULL);
    std::vector<int> recvcounts(grid.size, 0);
    std::vector<int> rdispls(grid.size, 0);
    std::vector<MPI_Datatype> recvtypes(grid.size, MPI_DATATYPE_NULL);
    std::vector<MPI_Datatype> committed_types;
    committed_types.reserve(grid.size);

    sendtypes[0] = MPI_FLOAT;
    if (local_size > 0) {
        sendcounts[0] = local_size;
    }

    if (world_rank == 0 && global_data != nullptr) {
        for (int rank = 0; rank < grid.size; rank++) {
            Block2DPartition part = compute_2d_block_partition_for_rank(global_height, global_width,
                                                                        grid, rank);
            int block_size = part.rows * part.cols * channels;
            if (block_size == 0) {
                continue;
            }

            int sizes[3] = {global_height, global_width, channels};
            int subsizes[3] = {part.rows, part.cols, channels};
            int starts[3] = {part.row_start, part.col_start, 0};

            MPI_Datatype subarray_type;
            MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                     MPI_FLOAT, &subarray_type);
            MPI_Type_commit(&subarray_type);

            recvcounts[rank] = 1;
            recvtypes[rank] = subarray_type;
            committed_types.push_back(subarray_type);
        }
    }

    MPI_Alltoallw(local_block,
                  sendcounts.data(), sdispls.data(), sendtypes.data(),
                  global_data,
                  recvcounts.data(), rdispls.data(), recvtypes.data(),
                  comm);

    for (MPI_Datatype type : committed_types) {
        MPI_Type_free(&type);
    }
}

// ============================================================================
// 2D Halo Exchange
// ============================================================================

struct HaloExchangeState {
    int radius = 0;
    int channels = 0;
    int local_rows = 0;
    int local_cols = 0;
    int row_stride = 0;
    int col_height = 0;
    int row_len = 0;
    int north = MPI_PROC_NULL;
    int south = MPI_PROC_NULL;
    int west = MPI_PROC_NULL;
    int east = MPI_PROC_NULL;
    float* base_ptr = nullptr;
    float* interior_ptr = nullptr;
    std::vector<float> send_north;
    std::vector<float> send_south;
    std::vector<float> send_west;
    std::vector<float> send_east;
    std::vector<float> recv_north;
    std::vector<float> recv_south;
    std::vector<float> recv_west;
    std::vector<float> recv_east;
    std::array<MPI_Request, 8> requests{};
    int request_count = 0;
};

HaloExchangeState begin_halo_exchange(std::vector<float>& local_with_halo,
                                      int local_rows, int local_cols, int radius,
                                      int channels, const ProcessGrid& grid) {
    HaloExchangeState state;
    state.radius = radius;
    state.channels = channels;
    state.local_rows = local_rows;
    state.local_cols = local_cols;

    if (radius == 0 || local_rows == 0 || local_cols == 0) {
        return state;
    }

    state.row_stride = (local_cols + 2 * radius) * channels;
    state.col_height = local_rows + 2 * radius;
    state.row_len = local_cols * channels;
    state.base_ptr = local_with_halo.data();
    state.interior_ptr = state.base_ptr + radius * state.row_stride + radius * channels;
    state.north = grid.north;
    state.south = grid.south;
    state.west = grid.west;
    state.east = grid.east;

    state.send_north.assign(radius * state.row_len, 0.f);
    state.send_south.assign(radius * state.row_len, 0.f);
    state.recv_north.assign(radius * state.row_len, 0.f);
    state.recv_south.assign(radius * state.row_len, 0.f);

    state.send_west.assign(state.col_height * radius * channels, 0.f);
    state.send_east.assign(state.col_height * radius * channels, 0.f);
    state.recv_west.assign(state.col_height * radius * channels, 0.f);
    state.recv_east.assign(state.col_height * radius * channels, 0.f);

    if (state.north != MPI_PROC_NULL) {
        MPI_Irecv(state.recv_north.data(), radius * state.row_len, MPI_FLOAT,
                  state.north, 1, grid.cart_comm, &state.requests[state.request_count++]);
    }
    if (state.south != MPI_PROC_NULL) {
        MPI_Irecv(state.recv_south.data(), radius * state.row_len, MPI_FLOAT,
                  state.south, 0, grid.cart_comm, &state.requests[state.request_count++]);
    }
    if (state.west != MPI_PROC_NULL) {
        MPI_Irecv(state.recv_west.data(), state.col_height * radius * channels, MPI_FLOAT,
                  state.west, 3, grid.cart_comm, &state.requests[state.request_count++]);
    }
    if (state.east != MPI_PROC_NULL) {
        MPI_Irecv(state.recv_east.data(), state.col_height * radius * channels, MPI_FLOAT,
                  state.east, 2, grid.cart_comm, &state.requests[state.request_count++]);
    }

    for (int r = 0; r < radius; r++) {
        int src_row = std::clamp(r, 0, std::max(local_rows - 1, 0));
        std::copy_n(state.interior_ptr + src_row * state.row_stride, state.row_len,
                    state.send_north.data() + r * state.row_len);

        int south_row = local_rows - radius + r;
        south_row = std::clamp(south_row, 0, std::max(local_rows - 1, 0));
        std::copy_n(state.interior_ptr + south_row * state.row_stride, state.row_len,
                    state.send_south.data() + r * state.row_len);
    }

    for (int y = 0; y < state.col_height; y++) {
        float* row_ptr = state.base_ptr + y * state.row_stride;
        for (int x = 0; x < radius; x++) {
            int west_col = radius + std::min(x, local_cols - 1);
            int east_start = radius + std::max(0, local_cols - radius);
            int east_col = std::min(east_start + x, radius + local_cols - 1);
            for (int c = 0; c < channels; c++) {
                state.send_west[(y * radius + x) * channels + c] =
                    row_ptr[west_col * channels + c];
                state.send_east[(y * radius + x) * channels + c] =
                    row_ptr[east_col * channels + c];
            }
        }
    }

    if (state.north != MPI_PROC_NULL) {
        MPI_Isend(state.send_north.data(), radius * state.row_len, MPI_FLOAT,
                  state.north, 0, grid.cart_comm, &state.requests[state.request_count++]);
    }
    if (state.south != MPI_PROC_NULL) {
        MPI_Isend(state.send_south.data(), radius * state.row_len, MPI_FLOAT,
                  state.south, 1, grid.cart_comm, &state.requests[state.request_count++]);
    }
    if (state.west != MPI_PROC_NULL) {
        MPI_Isend(state.send_west.data(), state.col_height * radius * channels, MPI_FLOAT,
                  state.west, 2, grid.cart_comm, &state.requests[state.request_count++]);
    }
    if (state.east != MPI_PROC_NULL) {
        MPI_Isend(state.send_east.data(), state.col_height * radius * channels, MPI_FLOAT,
                  state.east, 3, grid.cart_comm, &state.requests[state.request_count++]);
    }

    return state;
}

void finalize_halo_exchange(HaloExchangeState& state) {
    if (state.radius == 0 || state.local_rows == 0 || state.local_cols == 0) {
        return;
    }

    if (state.request_count > 0) {
        MPI_Waitall(state.request_count, state.requests.data(), MPI_STATUSES_IGNORE);
    }

    if (state.north != MPI_PROC_NULL) {
        for (int r = 0; r < state.radius; r++) {
            float* dest = state.base_ptr + r * state.row_stride + state.radius * state.channels;
            std::copy_n(state.recv_north.data() + r * state.row_len, state.row_len, dest);
        }
    } else {
        for (int r = 0; r < state.radius; r++) {
            float* dest = state.base_ptr + r * state.row_stride + state.radius * state.channels;
            std::copy_n(state.interior_ptr, state.row_len, dest);
        }
    }

    if (state.south != MPI_PROC_NULL) {
        for (int r = 0; r < state.radius; r++) {
            float* dest = state.interior_ptr + state.local_rows * state.row_stride + r * state.row_stride;
            std::copy_n(state.recv_south.data() + r * state.row_len, state.row_len, dest);
        }
    } else {
        const float* last_row = state.interior_ptr + (state.local_rows - 1) * state.row_stride;
        for (int r = 0; r < state.radius; r++) {
            float* dest = state.interior_ptr + state.local_rows * state.row_stride + r * state.row_stride;
            std::copy_n(last_row, state.row_len, dest);
        }
    }

    if (state.west != MPI_PROC_NULL) {
        for (int y = 0; y < state.col_height; y++) {
            float* row_ptr = state.base_ptr + y * state.row_stride;
            for (int x = 0; x < state.radius; x++) {
                for (int c = 0; c < state.channels; c++) {
                    row_ptr[x * state.channels + c] =
                        state.recv_west[(y * state.radius + x) * state.channels + c];
                }
            }
        }
    } else {
        for (int y = 0; y < state.col_height; y++) {
            float* row_ptr = state.base_ptr + y * state.row_stride;
            for (int x = 0; x < state.radius; x++) {
                for (int c = 0; c < state.channels; c++) {
                    row_ptr[x * state.channels + c] =
                        row_ptr[state.radius * state.channels + c];
                }
            }
        }
    }

    if (state.east != MPI_PROC_NULL) {
        for (int y = 0; y < state.col_height; y++) {
            float* row_ptr = state.base_ptr + y * state.row_stride;
            for (int x = 0; x < state.radius; x++) {
                for (int c = 0; c < state.channels; c++) {
                    row_ptr[(state.radius + state.local_cols + x) * state.channels + c] =
                        state.recv_east[(y * state.radius + x) * state.channels + c];
                }
            }
        }
    } else {
        for (int y = 0; y < state.col_height; y++) {
            float* row_ptr = state.base_ptr + y * state.row_stride;
            for (int x = 0; x < state.radius; x++) {
                for (int c = 0; c < state.channels; c++) {
                    row_ptr[(state.radius + state.local_cols + x) * state.channels + c] =
                        row_ptr[(state.radius + state.local_cols - 1) * state.channels + c];
                }
            }
        }
    }
}

// ============================================================================
// 2D Gaussian Blur
// ============================================================================

void gaussian_blur_region(const std::vector<float>& block_with_halo,
                          int local_rows, int local_cols, int channels,
                          const std::vector<float>& kernel, int radius,
                          float* output,
                          int y_start, int y_end,
                          int x_start, int x_end) {
    assert(channels == 1);
    if (y_start >= y_end || x_start >= x_end) {
        return;
    }

    int padded_cols = local_cols + 2 * radius;
    int row_pitch = padded_cols * channels;
    int kernel_size = static_cast<int>(kernel.size());
    if (kernel_size == 0) {
        return;
    }

    const float* block_ptr = block_with_halo.data();

    #pragma omp parallel if ((y_end - y_start) * (x_end - x_start) > 0)
    {
        std::vector<float> vertical_buffer(padded_cols, 0.f);

        #pragma omp for schedule(static)
        for (int y = y_start; y < y_end; y++) {
            const float* row_base = block_ptr + y * row_pitch;

            for (int col = 0; col < padded_cols; col++) {
                float sum = 0.f;
                const float* col_ptr = row_base + col;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < kernel_size; k++) {
                    sum += col_ptr[k * row_pitch] * kernel[k];
                }
                vertical_buffer[col] = sum;
            }

            float* out_row = output + y * local_cols;
            for (int x = x_start; x < x_end; x++) {
                float sum = 0.f;
                int center = x + radius;
                const float* horiz_ptr = vertical_buffer.data() + center - radius;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < kernel_size; k++) {
                    sum += horiz_ptr[k] * kernel[k];
                }
                out_row[x] = sum;
            }
        }
    }
}

void gaussian_blur_full(const std::vector<float>& block_with_halo,
                        int local_rows, int local_cols, int channels,
                        const std::vector<float>& kernel, int radius,
                        float* output) {
    gaussian_blur_region(block_with_halo, local_rows, local_cols, channels,
                         kernel, radius, output, 0, local_rows, 0, local_cols);
}

// ============================================================================
// Modified generate_gaussian_pyramid_parallel with 2D partitioning
// ============================================================================

ScaleSpacePyramid generate_gaussian_pyramid_parallel(const Image& img, float sigma_min,
                                                     int num_octaves, int scales_per_octave)
{
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ProcessGrid grid = create_process_grid_2d(world_size, world_rank);

    float base_sigma = sigma_min / MIN_PIX_DIST;
    float sigma_diff = std::sqrt(base_sigma * base_sigma - 1.0f);

    Image base_img;
    int width = 0;
    int height = 0;
    int channels = 0;
    if (world_rank == 0) {
        Image doubled = img.resize(img.width * 2, img.height * 2, Interpolation::BILINEAR);
        base_img = gaussian_blur(doubled, sigma_diff);
        width = base_img.width;
        height = base_img.height;
        channels = base_img.channels;
    }

    int dims[3] = {width, height, channels};
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    width = dims[0];
    height = dims[1];
    channels = dims[2];

    if (world_rank != 0) {
        base_img = Image(width, height, channels);
    }

    ScaleSpacePyramid pyramid = {
        num_octaves,
        scales_per_octave + 3,
        std::vector<std::vector<Image>>(num_octaves)
    };

    int levels_per_octave = scales_per_octave + 3;
    std::vector<float> sigma_vals(levels_per_octave);
    sigma_vals[0] = base_sigma;
    float k = std::pow(2.f, 1.f / scales_per_octave);
    for (int i = 1; i < levels_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i - 1);
        float sigma_total = k * sigma_prev;
        sigma_vals[i] = std::sqrt(sigma_total * sigma_total - sigma_prev * sigma_prev);
    }

    for (int octave = 0; octave < num_octaves; octave++) {
        Block2DPartition my_part = compute_2d_block_partition(height, width, grid);
        int local_size = my_part.rows * my_part.cols * channels;
        std::vector<float> local_block(local_size > 0 ? local_size : 0);

        scatter_2d_blocks(world_rank == 0 ? base_img.data : nullptr,
                          width, height,
                          local_size > 0 ? local_block.data() : nullptr,
                          my_part, grid, channels, world_rank);

        if (world_rank == 0) {
            pyramid.octaves[octave].reserve(levels_per_octave);
            pyramid.octaves[octave].push_back(base_img);
        }

        std::vector<int> kernel_sizes(levels_per_octave, 0);
        std::vector<int> radii(levels_per_octave, 0);
        std::vector<std::vector<float>> kernels(levels_per_octave);

        for (int scale = 1; scale < levels_per_octave; scale++) {
            if (world_rank == 0) {
                int kernel_size = std::ceil(6 * sigma_vals[scale]);
                if (kernel_size % 2 == 0) {
                    kernel_size++;
                }
                kernel_sizes[scale] = kernel_size;
                radii[scale] = kernel_size / 2;

                std::vector<float> kernel(kernel_size);
                float kernel_sum = 0.f;
                for (int t = -radii[scale]; t <= radii[scale]; t++) {
                    float val = std::exp(-(t * t) / (2.f * sigma_vals[scale] * sigma_vals[scale]));
                    kernel[t + radii[scale]] = val;
                    kernel_sum += val;
                }
                for (float& val : kernel) {
                    val /= kernel_sum;
                }
                kernels[scale] = std::move(kernel);
            }

            MPI_Bcast(&kernel_sizes[scale], 1, MPI_INT, 0, grid.cart_comm);
            MPI_Bcast(&radii[scale], 1, MPI_INT, 0, grid.cart_comm);

            if (world_rank != 0) {
                kernels[scale].resize(kernel_sizes[scale]);
            }
            MPI_Bcast(kernels[scale].data(), kernel_sizes[scale], MPI_FLOAT, 0, grid.cart_comm);
        }

        for (int scale = 1; scale < levels_per_octave; scale++) {
            int radius = radii[scale];
            const std::vector<float>& kernel = kernels[scale];

            std::vector<float> blurred_block(local_size > 0 ? local_size : 0);
            if (local_size > 0) {
                int halo_rows = my_part.rows + 2 * radius;
                int halo_cols = my_part.cols + 2 * radius;
                std::vector<float> local_with_halo(halo_rows * halo_cols * channels, 0.f);

                for (int y = 0; y < my_part.rows; y++) {
                    for (int x = 0; x < my_part.cols; x++) {
                        for (int c = 0; c < channels; c++) {
                            int src_idx = (y * my_part.cols + x) * channels + c;
                            int dst_idx = ((y + radius) * halo_cols + (x + radius)) * channels + c;
                            local_with_halo[dst_idx] = local_block[src_idx];
                        }
                    }
                }

                HaloExchangeState halo_state = begin_halo_exchange(local_with_halo,
                                                                    my_part.rows,
                                                                    my_part.cols,
                                                                    radius, channels, grid);

                float* output_ptr = blurred_block.data();
                bool halos_finalized = false;

                if (radius == 0) {
                    gaussian_blur_full(local_with_halo, my_part.rows, my_part.cols,
                                       channels, kernel, radius, output_ptr);
                    finalize_halo_exchange(halo_state);
                    halos_finalized = true;
                } else {
                    int inner_y_start = std::min(radius, my_part.rows);
                    int inner_y_end = std::max(my_part.rows - radius, inner_y_start);
                    int inner_x_start = std::min(radius, my_part.cols);
                    int inner_x_end = std::max(my_part.cols - radius, inner_x_start);

                    if (inner_y_start < inner_y_end && inner_x_start < inner_x_end) {
                        gaussian_blur_region(local_with_halo, my_part.rows, my_part.cols,
                                             channels, kernel, radius, output_ptr,
                                             inner_y_start, inner_y_end,
                                             inner_x_start, inner_x_end);
                    }

                    finalize_halo_exchange(halo_state);
                    halos_finalized = true;

                    int top_end = std::min(radius, my_part.rows);
                    if (top_end > 0) {
                        gaussian_blur_region(local_with_halo, my_part.rows, my_part.cols,
                                             channels, kernel, radius, output_ptr,
                                             0, top_end, 0, my_part.cols);
                    }

                    int bottom_start = std::max(my_part.rows - radius, top_end);
                    if (bottom_start < my_part.rows) {
                        gaussian_blur_region(local_with_halo, my_part.rows, my_part.cols,
                                             channels, kernel, radius, output_ptr,
                                             bottom_start, my_part.rows,
                                             0, my_part.cols);
                    }

                    int middle_start = top_end;
                    int middle_end = bottom_start;
                    if (middle_start < middle_end) {
                        int left_end = std::min(radius, my_part.cols);
                        if (left_end > 0) {
                            gaussian_blur_region(local_with_halo, my_part.rows, my_part.cols,
                                                 channels, kernel, radius, output_ptr,
                                                 middle_start, middle_end,
                                                 0, left_end);
                        }

                        int right_start = std::max(my_part.cols - radius, left_end);
                        if (right_start < my_part.cols) {
                            gaussian_blur_region(local_with_halo, my_part.rows, my_part.cols,
                                                 channels, kernel, radius, output_ptr,
                                                 middle_start, middle_end,
                                                 right_start, my_part.cols);
                        }
                    }
                }

                if (!halos_finalized) {
                    finalize_halo_exchange(halo_state);
                }
            }

            Image blurred;
            if (world_rank == 0) {
                blurred = Image(width, height, channels);
            }

            gather_2d_blocks(local_size > 0 ? blurred_block.data() : nullptr,
                             my_part,
                             world_rank == 0 ? blurred.data : nullptr,
                             width, height,
                             grid, channels, world_rank);

            if (world_rank == 0) {
                pyramid.octaves[octave].push_back(blurred);
                base_img = blurred;
            }

            if (local_size > 0) {
                local_block.swap(blurred_block);
            }
        }

        if (world_rank == 0) {
            const Image& next_base = pyramid.octaves[octave][levels_per_octave - 3];
            base_img = next_base.resize(next_base.width / 2, next_base.height / 2,
                                        Interpolation::NEAREST);
            width = base_img.width;
            height = base_img.height;
            channels = base_img.channels;
        }

        int new_dims[3] = {width, height, channels};
        MPI_Bcast(new_dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
        width = new_dims[0];
        height = new_dims[1];
        channels = new_dims[2];

        if (octave + 1 < num_octaves && world_rank != 0) {
            if (width > 0 && height > 0 && channels > 0) {
                base_img = Image(width, height, channels);
            } else {
                base_img = Image();
            }
        }
    }

    if (grid.cart_comm != MPI_COMM_WORLD && grid.cart_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&grid.cart_comm);
    }

    if (world_rank != 0) {
        return {};
    }
    return pyramid;
}

} // namespace

ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
                                            int num_octaves, int scales_per_octave)
{
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size > 1) {
        return generate_gaussian_pyramid_parallel(img, sigma_min, num_octaves, scales_per_octave);
    }

    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for bluring
    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }

    // create a scale space pyramid of gaussian images
    // images in each octave are half the size of images in the previous one
    ScaleSpacePyramid pyramid = {
        num_octaves,
        imgs_per_octave,
        std::vector<std::vector<Image>>(num_octaves)
    };
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(std::move(base_img));
        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = pyramid.octaves[i].back();
            pyramid.octaves[i].push_back(gaussian_blur(prev_img, sigma_vals[j]));
        }
        // prepare base image for next octave
        const Image& next_base_img = pyramid.octaves[i][imgs_per_octave-3];
        base_img = next_base_img.resize(next_base_img.width/2, next_base_img.height/2,
                                        Interpolation::NEAREST);
    }
    return pyramid;
}

// generate pyramid of difference of gaussians (DoG) images
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid)
{
    ScaleSpacePyramid dog_pyramid = {
        img_pyramid.num_octaves,
        img_pyramid.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(img_pyramid.num_octaves)
    };
    for (int octave = 0; octave < dog_pyramid.num_octaves; octave++) {
        dog_pyramid.octaves[octave].reserve(dog_pyramid.imgs_per_octave);
        const auto& gaussian_octave = img_pyramid.octaves[octave];
        for (int scale = 1; scale < img_pyramid.imgs_per_octave; scale++) {
            const Image& high = gaussian_octave[scale];
            const Image& low = gaussian_octave[scale - 1];
            Image diff(high.width, high.height, high.channels);
            const int plane_size = high.width * high.height;
            for (int c = 0; c < high.channels; c++) {
                const float* high_chan = high.channel_data(c);
                const float* low_chan = low.channel_data(c);
                float* dst_chan = diff.channel_data(c);
                for (int idx = 0; idx < plane_size; idx++) {
                    dst_chan[idx] = high_chan[idx] - low_chan[idx];
                }
            }
            dog_pyramid.octaves[octave].push_back(std::move(diff));
        }
    }
    return dog_pyramid;
}

bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    const int width = img.width;
    const int height = img.height;
    const float* img_plane = img.channel_data(0);
    const float* prev_plane = prev.channel_data(0);
    const float* next_plane = next.channel_data(0);

    auto clamp_index = [&](int xx, int yy) -> int {
        xx = std::clamp(xx, 0, width - 1);
        yy = std::clamp(yy, 0, height - 1);
        return yy * width + xx;
    };

    bool is_min = true, is_max = true;
    float val = img_plane[clamp_index(x, y)];

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int idx = clamp_index(x + dx, y + dy);
            float neighbor_prev = prev_plane[idx];
            if (neighbor_prev > val) is_max = false;
            if (neighbor_prev < val) is_min = false;

            float neighbor_next = next_plane[idx];
            if (neighbor_next > val) is_max = false;
            if (neighbor_next < val) is_min = false;

            float neighbor_curr = img_plane[idx];
            if (neighbor_curr > val) is_max = false;
            if (neighbor_curr < val) is_min = false;

            if (!is_min && !is_max) return false;
        }
    }
    return true;
}

// fit a quadratic near the discrete extremum,
// update the keypoint (interpolated) extremum value
// and return offsets of the interpolated extremum from the discrete extremum
std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
                                              const std::vector<Image>& octave,
                                              int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    const int width = img.width;
    const int height = img.height;
    const float* img_plane = img.channel_data(0);
    const float* prev_plane = prev.channel_data(0);
    const float* next_plane = next.channel_data(0);

    auto sample = [&](const float* plane, int xx, int yy) -> float {
        xx = std::clamp(xx, 0, width - 1);
        yy = std::clamp(yy, 0, height - 1);
        return plane[yy * width + xx];
    };

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    g1 = (sample(next_plane, x, y) - sample(prev_plane, x, y)) * 0.5f;
    g2 = (sample(img_plane, x + 1, y) - sample(img_plane, x - 1, y)) * 0.5f;
    g3 = (sample(img_plane, x, y + 1) - sample(img_plane, x, y - 1)) * 0.5f;

    h11 = sample(next_plane, x, y) + sample(prev_plane, x, y) - 2.0f * sample(img_plane, x, y);
    h22 = sample(img_plane, x + 1, y) + sample(img_plane, x - 1, y) - 2.0f * sample(img_plane, x, y);
    h33 = sample(img_plane, x, y + 1) + sample(img_plane, x, y - 1) - 2.0f * sample(img_plane, x, y);
    h12 = (sample(next_plane, x + 1, y) - sample(next_plane, x - 1, y)
          -sample(prev_plane, x + 1, y) + sample(prev_plane, x - 1, y)) * 0.25f;
    h13 = (sample(next_plane, x, y + 1) - sample(next_plane, x, y - 1)
          -sample(prev_plane, x, y + 1) + sample(prev_plane, x, y - 1)) * 0.25f;
    h23 = (sample(img_plane, x + 1, y + 1) - sample(img_plane, x + 1, y - 1)
          -sample(img_plane, x - 1, y + 1) + sample(img_plane, x - 1, y - 1)) * 0.25f;
    
    // invert hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h13*h23 - h13*h13*h22;
    hinv11 = (h22*h33 - h23*h23) / det;
    hinv12 = (h13*h23 - h12*h33) / det;
    hinv13 = (h12*h23 - h13*h22) / det;
    hinv22 = (h11*h33 - h13*h13) / det;
    hinv23 = (h12*h13 - h11*h23) / det;
    hinv33 = (h11*h22 - h12*h12) / det;

    // find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11*g1 - hinv12*g2 - hinv13*g3;
    float offset_x = -hinv12*g1 - hinv22*g2 - hinv23*g3;
    float offset_y = -hinv13*g1 - hinv23*g3 - hinv33*g3;

    float interpolated_extrema_val = sample(img_plane, x, y)
                                   + 0.5f*(g1*offset_s + g2*offset_x + g3*offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return {offset_s, offset_x, offset_y};
}

bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh=C_EDGE)
{
    const Image& img = octave[kp.scale];
    const int width = img.width;
    const int height = img.height;
    const float* plane = img.channel_data(0);
    auto sample = [&](int xx, int yy) -> float {
        xx = std::clamp(xx, 0, width - 1);
        yy = std::clamp(yy, 0, height - 1);
        return plane[yy * width + xx];
    };

    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = sample(x + 1, y) + sample(x - 1, y) - 2.0f * sample(x, y);
    h22 = sample(x, y + 1) + sample(x, y - 1) - 2.0f * sample(x, y);
    h12 = (sample(x + 1, y + 1) - sample(x + 1, y - 1)
          -sample(x - 1, y + 1) + sample(x - 1, y - 1)) * 0.25f;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
}

void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
                                   float sigma_min=SIGMA_MIN,
                                   float min_pix_dist=MIN_PIX_DIST, int n_spo=N_SPO)
{
    kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s+kp.scale)/n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x+kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y+kp.j);
}

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS) {
        auto [offset_s, offset_x, offset_y] = fit_quadratic(kp, octave, kp.scale);

        float max_offset = std::max({std::abs(offset_s),
                                     std::abs(offset_x),
                                     std::abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, float contrast_thresh,
                                     float edge_thresh)
{
    std::vector<Keypoint> keypoints;
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        const std::vector<Image>& octave = dog_pyramid.octaves[i];
        for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
            const Image& img = octave[j];
            const int width = img.width;
            const int height = img.height;
            const float* plane = img.channel_data(0);
            for (int y = 1; y < height-1; y++) {
                const int row = y * width;
                for (int x = 1; x < width-1; x++) {
                    float val = plane[row + x];
                    if (std::abs(val) < 0.8f * contrast_thresh) {
                        continue;
                    }
                    if (point_is_extremum(octave, j, x, y)) {
                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh,
                                                                      edge_thresh);
                        if (kp_is_valid) {
                            keypoints.push_back(kp);
                        }
                    }
                }
            }
        }
    }
    return keypoints;
}

// calculate x and y derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    for (int octave = 0; octave < pyramid.num_octaves; octave++) {
        grad_pyramid.octaves[octave].reserve(grad_pyramid.imgs_per_octave);
        const auto& gaussian_octave = pyramid.octaves[octave];
        for (int scale = 0; scale < pyramid.imgs_per_octave; scale++) {
            const Image& src = gaussian_octave[scale];
            Image grad(src.width, src.height, 2);
            const float* src_plane = src.channel_data(0);
            float* gx_plane = grad.channel_data(0);
            float* gy_plane = grad.channel_data(1);
            const int width = src.width;
            const int height = src.height;
            for (int y = 1; y < height - 1; y++) {
                const int row = y * width;
                for (int x = 1; x < width - 1; x++) {
                    const int idx = row + x;
                    gx_plane[idx] = 0.5f * (src_plane[idx + 1] - src_plane[idx - 1]);
                    gy_plane[idx] = 0.5f * (src_plane[idx + width] - src_plane[idx - width]);
                }
            }
            grad_pyramid.octaves[octave].push_back(std::move(grad));
        }
    }
    return grad_pyramid;
}

// convolve 6x with box filter
void smooth_histogram(float hist[N_BINS])
{
    float tmp_hist[N_BINS];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < N_BINS; j++) {
            int prev_idx = (j-1+N_BINS)%N_BINS;
            int next_idx = (j+1)%N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        for (int j = 0; j < N_BINS; j++) {
            hist[j] = tmp_hist[j];
        }
    }
}

std::vector<float> find_keypoint_orientations(Keypoint& kp, 
                                              const ScaleSpacePyramid& grad_pyramid,
                                              float lambda_ori, float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_grad.width-kp.x,
                                           pix_dist*img_grad.height-kp.y});
    if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp.sigma) {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    int x_start = std::round((kp.x - patch_radius)/pix_dist);
    int x_end = std::round((kp.x + patch_radius)/pix_dist);
    int y_start = std::round((kp.y - patch_radius)/pix_dist);
    int y_end = std::round((kp.y + patch_radius)/pix_dist);

    // accumulate gradients in orientation histogram
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            gx = img_grad.get_pixel(x, y, 0);
            gy = img_grad.get_pixel(x, y, 1);
            grad_norm = std::sqrt(gx*gx + gy*gy);
            weight = std::exp(-(std::pow(x*pix_dist-kp.x, 2)+std::pow(y*pix_dist-kp.y, 2))
                              /(2*patch_sigma*patch_sigma));
            theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
            bin = (int)std::round(N_BINS/(2*M_PI)*theta) % N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    std::vector<float> orientations;
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j-1+N_BINS)%N_BINS], next = hist[(j+1)%N_BINS];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}

void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
        if (std::abs(x_i-x) > 2*lambda_desc/N_HIST)
            continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
            if (std::abs(y_j-y) > 2*lambda_desc/N_HIST)
                continue;
            
            float hist_weight = (1 - N_HIST*0.5/lambda_desc*std::abs(x_i-x))
                               *(1 - N_HIST*0.5/lambda_desc*std::abs(y_j-y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2*M_PI*(k-1)/N_ORI;
                float theta_diff = std::fmod(theta_k-theta_mn+2*M_PI, 2*M_PI);
                if (std::abs(theta_diff) >= 2*M_PI/N_ORI)
                    continue;
                float bin_weight = 1 - N_ORI*0.5/M_PI*std::abs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    float norm = 0;
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    float norm2 = 0;
    for (int i = 0; i < size; i++) {
        hist[i] = std::min(hist[i], 0.2f*norm);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    for (int i = 0; i < size; i++) {
        float val = std::floor(512*hist[i]/norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {0};

    //find start and end coords for loops over image patch
    float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST+1.)/N_HIST;
    int x_start = std::round((kp.x-half_size) / pix_dist);
    int x_end = std::round((kp.x+half_size) / pix_dist);
    int y_start = std::round((kp.y-half_size) / pix_dist);
    int y_end = std::round((kp.y+half_size) / pix_dist);

    float cos_t = std::cos(theta), sin_t = std::sin(theta);
    float patch_sigma = lambda_desc * kp.sigma;
    //accumulate samples into histograms
    for (int m = x_start; m <= x_end; m++) {
        for (int n = y_start; n <= y_end; n++) {
            // find normalized coords w.r.t. kp position and reference orientation
            float x = ((m*pix_dist - kp.x)*cos_t
                      +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
            float y = (-(m*pix_dist - kp.x)*sin_t
                       +(n*pix_dist - kp.y)*cos_t) / kp.sigma;

            // verify (x, y) is inside the description patch
            if (std::max(std::abs(x), std::abs(y)) > lambda_desc*(N_HIST+1.)/N_HIST)
                continue;

            float gx = img_grad.get_pixel(m, n, 0), gy = img_grad.get_pixel(m, n, 1);
            float theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
            float grad_norm = std::sqrt(gx*gx + gy*gy);
            float weight = std::exp(-(std::pow(m*pix_dist-kp.x, 2)+std::pow(n*pix_dist-kp.y, 2))
                                    /(2*patch_sigma*patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
}

std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min,
                                                     int num_octaves, int scales_per_octave, 
                                                     float contrast_thresh, float edge_thresh, 
                                                     float lambda_ori, float lambda_desc)
{
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    assert(img.channels == 1 || img.channels == 3);

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    if (world_size > 1 && world_rank != 0) {
        return {};
    }
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    std::vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    
    std::vector<Keypoint> kps;

    #pragma omp parallel
    {
        std::vector<Keypoint> kps_private;
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < tmp_kps.size(); i++) {
            Keypoint& kp_tmp = tmp_kps[i];
            std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid,
                                                                         lambda_ori, lambda_desc);
            for (float theta : orientations) {
                Keypoint kp = kp_tmp;
                compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
                kps_private.push_back(kp);
            }
        }
        #pragma omp critical
        kps.insert(kps.end(), kps_private.begin(), kps_private.end());
    }

    return kps;
}

float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b)
{
    float dist = 0;
    for (int i = 0; i < 128; i++) {
        int di = (int)a[i] - b[i];
        dist += di * di;
    }
    return std::sqrt(dist);
}

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps)
{
    Image res(img);
    if (img.channels == 1) {
        res = grayscale_to_rgb(res);
    }
    for (auto& kp : kps) {
        draw_point(res, kp.x, kp.y, 5);
    }
    return res;
}
