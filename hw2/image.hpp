#ifndef IMAGE_H
#define IMAGE_H
#include <string>

enum Interpolation {BILINEAR, NEAREST};

struct Image {
    explicit Image(std::string file_path);
    Image(int w, int h, int c);
    Image();
    ~Image();
    Image(const Image& other);
    Image& operator=(const Image& other);
    Image(Image&& other);
    Image& operator=(Image&& other);
    int width;
    int height;
    int channels;
    int size;
    float *data;
    bool save(std::string file_path);
    void set_pixel(int x, int y, int c, float val);
    float get_pixel(int x, int y, int c) const;
    void clamp();
    Image resize(int new_w, int new_h, Interpolation method = BILINEAR) const;
    float* channel_data(int c);
    const float* channel_data(int c) const;
};


float bilinear_interpolate(const Image& img, float x, float y, int c);
float nn_interpolate(const Image& img, float x, float y, int c);

Image rgb_to_grayscale(const Image& img);
Image grayscale_to_rgb(const Image& img);

Image gaussian_blur(const Image& img, float sigma);

void draw_point(Image& img, int x, int y, int size=3);
void draw_line(Image& img, int x1, int y1, int x2, int y2);


inline float Image::get_pixel(int x, int y, int c) const
{
    if (x < 0)
        x = 0;
    else if (x >= width)
        x = width - 1;

    if (y < 0)
        y = 0;
    else if (y >= height)
        y = height - 1;

    return data[c*width*height + y*width + x];
}

inline float* Image::channel_data(int c)
{
    return data + c*width*height;
}

inline const float* Image::channel_data(int c) const
{
    return data + c*width*height;
}

#endif
