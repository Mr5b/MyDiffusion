#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <cstring>   // memcpy
#include <algorithm> // std::min
#include <cassert>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
 
namespace MyTensor::Core
{
 
using namespace MNN::Express;


static stbir_pixel_layout get_pixel_layout(int channels)
{
    switch (channels)
    {
        case 1: return STBIR_1CHANNEL;
        case 2: return STBIR_2CHANNEL;
        case 3: return STBIR_RGB;
        case 4: return STBIR_RGBA;
        default: return STBIR_RGB;
    }
}

bool load_and_resize_image
(
    const char* filename,
    int target_width,
    int target_height,
    unsigned char* output_data,
    int* out_channels
)
{
    int orig_w, orig_h, orig_comp;
    if (!stbi_info(filename, &orig_w, &orig_h, &orig_comp))
    {
        fprintf(stderr, "");
        return false;
    }

    if (orig_comp < 1 || orig_comp > 4)
    {
        fprintf(stderr, "%d\n", orig_comp);
        return false;
    }

    int loaded_w, loaded_h, loaded_comp;
    unsigned char* input_data =
        stbi_load
        (
            filename,
            &loaded_w,
            &loaded_h,
            &loaded_comp,
            0
        );
    if (!input_data)
    {
        fprintf
        (
            stderr, ""
        );
        return false;
    }

    if (loaded_w != orig_w || loaded_h != orig_h || loaded_comp != orig_comp)
    {
        fprintf(stderr, "");
        orig_w = loaded_w;
        orig_h = loaded_h;
        orig_comp = loaded_comp;
    }

    bool success = false;
    if (orig_w == target_width && orig_h == target_height)
    {
        size_t total_bytes = orig_w * orig_h * orig_comp;
        std::memcpy(output_data, input_data, total_bytes);
        success = true;
    }
    else
    {
        stbir_pixel_layout layout = get_pixel_layout(orig_comp);
        success = stbir_resize_uint8_linear
        (
            input_data, orig_w, orig_h, 0,
            output_data, target_width, target_height, 0,
            layout
        );
        if (!success)
        {
            fprintf(stderr, "");
        }
    }

    stbi_image_free(input_data);

    if (success && out_channels)
    {
        *out_channels = orig_comp;
    }
    return success;
}

inline VARP load_from_image
(
    const char* filename
)
{
    int width, height, channels;
    
    if (!stbi_info(filename, &width, &height, &channels))
    {
        throw std::runtime_error("图片信息查询失败");
    }
    
    
    
    VARP dst = _Const(nullptr, {1, height, width, channels}, NHWC, halide_type_of<uint8_t>());
    unsigned char* data_ptr = dst->writeMap<unsigned char>();
    
    bool success =
    load_and_resize_image
    (
        filename,
        width,
        height,
        data_ptr,
        nullptr
    );
    
    if (!success) throw std::runtime_error("失败");
    
    return dst;
}

inline VARP load_from_image
(
    const char* filename,
    int target_width,
    int target_height
)
{
    int width, height, channels;
    
    if (!stbi_info(filename, &width, &height, &channels))
    {
        throw std::runtime_error("图片信息查询失败");
    }
    
    
    
    VARP dst = _Const(nullptr, {1, target_height, target_width, channels}, NHWC, halide_type_of<uint8_t>());
    uint8_t* data_ptr = dst->writeMap<uint8_t>();
    
    bool success =
    load_and_resize_image
    (
        filename,
        target_width,
        target_height,
        data_ptr,
        nullptr
    );
    
    if (!success) throw std::runtime_error("失败");
    
    return dst;
}

}//namespace MyTensor::Core

#endif