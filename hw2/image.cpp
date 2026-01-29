#include <cmath>
#include <iostream>
#include <cassert>
#include <utility>
#include <vector>

#include "image.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image::Image(std::string file_path)
{
    unsigned char *img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        const char *error_msg = stbi_failure_reason();
        std::cerr << "Failed to load image: " << file_path.c_str() << "\n";
        std::cerr << "Error msg (stb_image): " << error_msg << "\n";
        std::exit(1);
    }

    int input_channels = channels;
    if (input_channels != 1 && input_channels != 3 && input_channels != 4) {
        std::cerr << "Unsupported channel count: " << input_channels << " in " << file_path << "\n";
        stbi_image_free(img_data);
        std::exit(1);
    }

    channels = (input_channels == 4) ? 3 : input_channels;
    size = width * height * channels;
    data = new float[size];

    for (int y = 0; y < height; y++) {
        int src_row = y * width * input_channels;
        int dst_row = y * width;
        for (int x = 0; x < width; x++) {
            int src_pixel = src_row + x * input_channels;
            int dst_pixel = dst_row + x;
            for (int c = 0; c < channels; c++) {
                int src_idx = src_pixel + c;
                int dst_idx = c * height * width + dst_pixel;
                data[dst_idx] = img_data[src_idx] / 255.f;
            }
        }
    }
    stbi_image_free(img_data);
}

Image::Image(int w, int h, int c)
    :width {w},
     height {h},
     channels {c},
     size {w*h*c},
     data {new float[w*h*c]()}
{
}

Image::Image()
    :width {0},
     height {0},
     channels {0},
     size {0},
     data {nullptr} 
{
}

Image::~Image()
{
    delete[] this->data;
}

Image::Image(const Image& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {new float[other.size]}
{
    //std::cout << "copy constructor\n";
    for (int i = 0; i < size; i++)
        data[i] = other.data[i];
}

Image& Image::operator=(const Image& other)
{
    if (this != &other) {
        delete[] data;
        //std::cout << "copy assignment\n";
        width = other.width;
        height = other.height;
        channels = other.channels;
        size = other.size;
        data = new float[other.size];
        for (int i = 0; i < other.size; i++)
            data[i] = other.data[i];
    }
    return *this;
}

Image::Image(Image&& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {other.data}
{
    //std::cout << "move constructor\n";
    other.data = nullptr;
    other.size = 0;
}

Image& Image::operator=(Image&& other)
{
    //std::cout << "move assignment\n";
    delete[] data;
    data = other.data;
    width = other.width;
    height = other.height;
    channels = other.channels;
    size = other.size;

    other.data = nullptr;
    other.size = 0;
    return *this;
}

//save image as jpg file
bool Image::save(std::string file_path)
{
    unsigned char *out_data = new unsigned char[width*height*channels]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int dst_idx = y*width*channels + x*channels + c;
                int src_idx = c*height*width + y*width + x;
                out_data[dst_idx] = std::roundf(data[src_idx] * 255.);
            }
        }
    }
    bool success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    if (!success)
        std::cerr << "Failed to save image: " << file_path << "\n";

    delete[] out_data;
    return true;
}

void Image::set_pixel(int x, int y, int c, float val)
{
    if (x >= width || x < 0 || y >= height || y < 0 || c >= channels || c < 0) {
        std::cerr << "set_pixel() error: Index out of bounds.\n";
        std::exit(1);
    }
    data[c*width*height + y*width + x] = val;
}

void Image::clamp()
{
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        float val = data[i];
        val = (val > 1.0) ? 1.0 : val;
        val = (val < 0.0) ? 0.0 : val;
        data[i] = val;
    }
}

//map coordinate from 0-current_max range to 0-new_max range
float map_coordinate(float new_max, float current_max, float coord)
{
    float a = new_max / current_max;
    float b = -0.5 + a*0.5;
    return a*coord + b;
}

Image Image::resize(int new_w, int new_h, Interpolation method) const
{
    Image resized(new_w, new_h, this->channels);
    float value = 0;
    for (int x = 0; x < new_w; x++) {
        for (int y = 0; y < new_h; y++) {
            for (int c = 0; c < resized.channels; c++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                if (method == Interpolation::BILINEAR)
                    value = bilinear_interpolate(*this, old_x, old_y, c);
                else if (method == Interpolation::NEAREST)
                    value = nn_interpolate(*this, old_x, old_y, c);
                resized.set_pixel(x, y, c, value);
            }
        }
    }
    return resized;
}

float bilinear_interpolate(const Image& img, float x, float y, int c)
{
    float p1, p2, p3, p4, q1, q2;
    float x_floor = std::floor(x), y_floor = std::floor(y);
    float x_ceil = x_floor + 1, y_ceil = y_floor + 1;
    p1 = img.get_pixel(x_floor, y_floor, c);
    p2 = img.get_pixel(x_ceil, y_floor, c);
    p3 = img.get_pixel(x_floor, y_ceil, c);
    p4 = img.get_pixel(x_ceil, y_ceil, c);
    q1 = (y_ceil-y)*p1 + (y-y_floor)*p3;
    q2 = (y_ceil-y)*p2 + (y-y_floor)*p4;
    return (x_ceil-x)*q1 + (x-x_floor)*q2;
}

float nn_interpolate(const Image& img, float x, float y, int c)
{
    return img.get_pixel(std::round(x), std::round(y), c);
}

Image rgb_to_grayscale(const Image& img)
{
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float red, green, blue;
            red = img.get_pixel(x, y, 0);
            green = img.get_pixel(x, y, 1);
            blue = img.get_pixel(x, y, 2);
            gray.set_pixel(x, y, 0, 0.299*red + 0.587*green + 0.114*blue);
        }
    }
    return gray;
}

Image grayscale_to_rgb(const Image& img)
{
    assert(img.channels == 1);
    Image rgb(img.width, img.height, 3);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float gray_val = img.get_pixel(x, y, 0);
            rgb.set_pixel(x, y, 0, gray_val);
            rgb.set_pixel(x, y, 1, gray_val);
            rgb.set_pixel(x, y, 2, gray_val);
        }
    }
    return rgb;
}

// separable 2D gaussian blur for 1 channel image
Image gaussian_blur(const Image& img, float sigma)
{
    assert(img.channels == 1);

    int size = std::ceil(6 * sigma);
    if (size % 2 == 0)
        size++;
    int center = size / 2;

    std::vector<float> kernel(size);
    float kernel_sum = 0.f;
    for (int k = 0; k < size; k++) {
        int offset = k - center;
        float val = std::exp(-(offset * offset) / (2.f * sigma * sigma));
        kernel[k] = val;
        kernel_sum += val;
    }
    for (float& val : kernel)
        val /= kernel_sum;

    Image tmp(img.width, img.height, 1);
    Image filtered(img.width, img.height, 1);

    const float* src = img.channel_data(0);
    float* tmp_plane = tmp.channel_data(0);
    float* dst_plane = filtered.channel_data(0);
    const int width = img.width;
    const int height = img.height;

    // convolve vertical
    for (int y = 0; y < height; y++) {
        int row_offset = y * width;
        for (int x = 0; x < width; x++) {
            float sum = 0.f;
            for (int k = 0; k < size; k++) {
                int yy = y + (k - center);
                if (yy < 0)
                    yy = 0;
                else if (yy >= height)
                    yy = height - 1;
                sum += src[yy * width + x] * kernel[k];
            }
            tmp_plane[row_offset + x] = sum;
        }
    }

    // convolve horizontal
    for (int y = 0; y < height; y++) {
        int row_offset = y * width;
        const float* tmp_row = tmp_plane + row_offset;
        float* dst_row = dst_plane + row_offset;
        for (int x = 0; x < width; x++) {
            float sum = 0.f;
            for (int k = 0; k < size; k++) {
                int xx = x + (k - center);
                if (xx < 0)
                    xx = 0;
                else if (xx >= width)
                    xx = width - 1;
                sum += tmp_row[xx] * kernel[k];
            }
            dst_row[x] = sum;
        }
    }

    return filtered;
}

void draw_point(Image& img, int x, int y, int size)
{
    for (int i = x-size/2; i <= x+size/2; i++) {
        for (int j = y-size/2; j <= y+size/2; j++) {
            if (i < 0 || i >= img.width) continue;
            if (j < 0 || j >= img.height) continue;
            if (std::abs(i-x) + std::abs(j-y) > size/2) continue;
            if (img.channels == 3) {
                img.set_pixel(i, j, 0, 1.f);
                img.set_pixel(i, j, 1, 0.f);
                img.set_pixel(i, j, 2, 0.f);
            } else {
                img.set_pixel(i, j, 0, 1.f);
            }
        }
    }
}

void draw_line(Image& img, int x1, int y1, int x2, int y2)
{
    if (x2 < x1) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    int dx = x2 - x1, dy = y2 - y1;
    for (int x = x1; x < x2; x++) {
        int y = y1 + dy*(x-x1)/dx;
        if (img.channels == 3) {
            img.set_pixel(x, y, 0, 0.f);
            img.set_pixel(x, y, 1, 1.f);
            img.set_pixel(x, y, 2, 0.f);
        } else {
            img.set_pixel(x, y, 0, 1.f);
        }
    }
}
