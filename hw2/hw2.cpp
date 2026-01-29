#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <mpi.h>

#include "image.hpp"
#include "sift.hpp"


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 4) {
        std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        MPI_Finalize();
        return 1;
    }

    std::string input_img = argv[1];
    std::string output_img = argv[2];
    std::string output_txt = argv[3];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Image img;
    int width = 0;
    int height = 0;
    int channels = 0;
    int gaussian_octaves = N_OCT;
    int scales_per_octave = N_SPO;

    if (world_rank == 0) {
        img = Image(input_img);
        img = img.channels == 1 ? img : rgb_to_grayscale(img);
        width = img.width;
        height = img.height;
        channels = img.channels;
    }

    int dims[3] = {width, height, channels};
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    width = dims[0];
    height = dims[1];
    channels = dims[2];

    const int pixel_count = width * height * channels;
    if (world_rank != 0 && pixel_count > 0)
        img = Image(width, height, channels);
    if (pixel_count > 0)
        MPI_Bcast(img.data, pixel_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int config[2] = {gaussian_octaves, scales_per_octave};
    MPI_Bcast(config, 2, MPI_INT, 0, MPI_COMM_WORLD);
    gaussian_octaves = config[0];
    scales_per_octave = config[1];

    std::vector<Keypoint> kps = find_keypoints_and_descriptors(img);


    /////////////////////////////////////////////////////////////
    // The following code is for the validation
    // You can not change the logic of the following code, because it is used for judge system
    if (world_rank == 0) {
        std::ofstream ofs(output_txt);
        if (!ofs) {
            std::cerr << "Failed to open " << output_txt << " for writing.\n";
        } else {
            ofs << kps.size() << "\n";
            for (const auto& kp : kps) {
                ofs << kp.i << " " << kp.j << " " << kp.octave << " " << kp.scale << " ";
                for (size_t i = 0; i < kp.descriptor.size(); ++i) {
                    ofs << " " << static_cast<int>(kp.descriptor[i]);
                }
                ofs << "\n";
            }
            ofs.close();
        }

        Image result = draw_keypoints(img, kps);
        result.save(output_img);
    }
    /////////////////////////////////////////////////////////////

    auto end = std::chrono::high_resolution_clock::now();
    if (world_rank == 0) {
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Execution time: " << duration.count() << " ms\n";
        std::cout << "Found " << kps.size() << " keypoints.\n";
    }

    MPI_Finalize();
    return 0;
}
