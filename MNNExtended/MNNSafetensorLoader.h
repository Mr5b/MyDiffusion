#ifndef MNN_SAFETENSOR_LOADER_H
#define MNN_SAFETENSOR_LOADER_H

#include "MNNExternalTensor.h"
#include "safetensors_metadata.hpp"
#include "mymodule_assert.h"

/*inline DataType getDataType(std::string& dtype)
{
    if (dtype == "F32")
    {
        return DataType::kFloat32;
    }
    else if (dtype == "F16")
    {
        return DataType::kFloat16;
    }
    else
    {
        return DataType::Undefined;
    }
    
}*/

namespace DonNotKnowHowToNameIt
{

using namespace MNN::Express;
using namespace MyTensor::Utils;
using namespace MyTensor::Utils::safetensors;


inline halide_type_t stringToHalideType(const std::string& dtype)
{
    if (dtype == "F32" || dtype == "Float32")
    {
        return halide_type_t(halide_type_float, 32);
    }
    else if (dtype == "F16" || dtype == "Float16")
    {
        return halide_type_t(halide_type_float, 16);
    }
    else if (dtype == "BF16" || dtype == "BFloat16")
    {
        return halide_type_t(halide_type_bfloat, 16);
    }
    else if (dtype == "I32" || dtype == "Int32")
    {
        return halide_type_t(halide_type_int, 32);
    }
    else if (dtype == "I64" || dtype == "Int64")
    {
        return halide_type_t(halide_type_int, 64);
    }
    else if (dtype == "I16" || dtype == "Int16")
    {
        return halide_type_t(halide_type_int, 16);
    }
    else if (dtype == "I8" || dtype == "Int8")
    {
        return halide_type_t(halide_type_int, 8);
    }
    else if (dtype == "U8" || dtype == "UInt8")
    {
        return halide_type_t(halide_type_uint, 8);
    }
    else if (dtype == "U16" || dtype == "UInt16")
    {
        return halide_type_t(halide_type_uint, 16);
    }
    else if (dtype == "U32" || dtype == "UInt32")
    {
        return halide_type_t(halide_type_uint, 32);
    }
    else if (dtype == "U64" || dtype == "UInt64")
    {
        return halide_type_t(halide_type_uint, 64);
    }
    else if (dtype == "BOOL" || dtype == "bool")
    {
        return halide_type_t(halide_type_uint, 8);
    }
    else
    {
        throw std::invalid_argument("Unsupported safetensors dtype: " + dtype);
    }
}

class SafetensorLoader
{
public:
    enum class ShapeMode
    {
        STRICT,
        WARNING,
        LOOSE
    };

    SafetensorLoader(const std::string& file_path) :
        handle_(FileHandleCache::get_handle(file_path, true))
    {
        if (!metadata_.load_from_file(file_path))
        {
            throw std::runtime_error("Load");
        }
    }
    
    bool contains(const std::string& key) const
    {
        return
            metadata_.get_tensor_metadata(key) != nullptr;
    }
    
    
    
    /*Tensor get_tensor(const std::string& key) const
    {
        const TensorMetadata* tensor_metadata = metadata_.get_tensor_metadata(key);
        std::string dtype_str = tensor_metadata->dtype;
        DataType tensor_dtype = getDataType(dtype_str);
        size_t tensor_size = tensor_metadata->data_offsets[1] - tensor_metadata->data_offsets[0];
        size_t tensor_start = metadata_.get_data_start_offset() + tensor_metadata->data_offsets[0];
        
        return
            map_from_file
            (
                handle_,
                tensor_start,
                tensor_size,
                tensor_dtype,
                tensor_metadata->shape
            );
    }*/
    
    /*void get_tensor(Tensor& tensor, const std::string& key, bool keep_shape = false) const
    {
        const TensorMetadata* tensor_metadata = metadata_.get_tensor_metadata(key);
        size_t tensor_start = metadata_.get_data_start_offset() + tensor_metadata->data_offsets[0];
        std::string dtype_str = tensor_metadata->dtype;
        DataType tensor_dtype = getDataType(dtype_str);
        
        if (keep_shape)
        {
            if (tensor.sizes().size() != tensor_metadata->shape.size()) throw std::runtime_error("乱来");
            for (int64_t i = 0, n = tensor.sizes().size(); i < n; i++)
            {
                if (tensor.sizes()[i] != tensor_metadata->shape[i]) throw std::runtime_error("乱来");
            }
            
            tensor.map_from_file_
            (
                handle_,
                tensor_start,
                tensor_dtype
            );
            return;
        }
        
        
        size_t tensor_size = tensor_metadata->data_offsets[1] - tensor_metadata->data_offsets[0];
        
        tensor.map_from_file_
        (
            handle_,
            tensor_start,
            tensor_size,
            tensor_dtype,
            tensor_metadata->shape
        );
    }*/
    
    
    std::pair<MNN::Express::EXPRP, std::shared_ptr<MyTensor::Utils::FileMapping>> get_tensor
    (
        const std::string& key,
        Dimensionformat format = Dimensionformat::NCHW,
        INTS expected_shape = {},
        ShapeMode shape_mode = ShapeMode::LOOSE
    ) const
    {
        const TensorMetadata* tensor_metadata = metadata_.get_tensor_metadata(key);
        size_t tensor_start = metadata_.get_data_start_offset() + tensor_metadata->data_offsets[0];
        std::string dtype_str = tensor_metadata->dtype;
        halide_type_t tensor_dtype = stringToHalideType(dtype_str);
        
        
        
        size_t tensor_size = tensor_metadata->data_offsets[1] - tensor_metadata->data_offsets[0];
        
        INTS shape;
        int numel = 1;
        shape.reserve(tensor_metadata->shape.size());
        for (auto dim : tensor_metadata->shape)
        {
            numel *= dim;
            shape.emplace_back(int(dim));
        }
        
        Variable::Info info;
        info.order = format;
        info.type = tensor_dtype;
        
        int expected_numel = 1;
        if (!expected_shape.empty())
        {
            
            for (auto dim : expected_shape) expected_numel *= dim;
            if (expected_numel != numel) throw std::runtime_error("Shape mismatch: " + key);
            
            if (shape_mode != ShapeMode::LOOSE)
            {
                if
                (
                    shape.size() != expected_shape.size() ||
                    !std::equal
                    (
                        shape.begin(),
                        shape.end(),
                        expected_shape.begin()
                    )
                )
                {
                    if (shape_mode == ShapeMode::STRICT)
                    {
                        throw std::runtime_error("Shape mismatch: " + key);
                    }
                    else
                    {
                        std::cout << "Warning: Shape mismatch: " << key << std::endl;
                    }
                }
            }
            info.dim = std::move(expected_shape);
        }
        else
        {
            info.dim = std::move(shape);
        }
        
        
        /*tensor.map_from_file_
        (
            handle_,
            tensor_start,
            tensor_size,
            tensor_dtype,
            tensor_metadata->shape
        );*/
        
        
        
        std::shared_ptr<MyTensor::Utils::FileMapping> mapping_ =
            std::make_unique<FileMapping>(handle_, tensor_start, tensor_size, false);
        
        if (info.type == halide_type_t(halide_type_float, 16))
        {
            info.type = halide_type_t(halide_type_float, 32);
            
            auto old_ptr = reinterpret_cast<MyTensor::Utils::Float16*>(mapping_->data());
            /*auto temp_handle = GlobalTemporaryFiles::get_handle();
            size_t start = temp_handle->size();
            temp_handle->truncate(start + tensor_size);
            auto temp_mapping = GlobalTemporaryFiles::get_mapping();
            float* start_ptr = (float*)((char*)(temp_mapping->data()) + start);*/
            
            //std::vector<float> data(expected_numel);
            std::string file_path = "TemporaryFile/" + key + "_TemporaryFile";
            std::shared_ptr<FileMapping> mapping =
                std::make_shared<TemporaryFileMapping>(file_path, expected_numel*sizeof(float));
                
            float* temp_ptr = reinterpret_cast<float*>(mapping->data());
                
            for (int i = 0; i < expected_numel; i++)
            {
                temp_ptr[i] = static_cast<float>(old_ptr[i]);
            }
            //std::cout << "float16" << std::endl;    
            //return Expr::create(std::move(info), data_ptr, VARP::CONSTANT, Expr::MemoryType::REF);
            return
                std::make_pair
                (
                    Expr::create(std::move(info), temp_ptr, VARP::CONSTANT, Expr::MemoryType::COPY),
                    nullptr
                );
        }
        
        return
            std::make_pair
            (
                Expr::create(std::move(info), mapping_->data(), VARP::CONSTANT, Expr::MemoryType::REF),
                mapping_
            );
    }
    
    
    std::shared_ptr<SharedFileHandle> get_shared_file_handle() const
    {
        return handle_;
    }
    
    const TensorMetadata* get_tensor_metadata(const std::string& key)
    {
        return metadata_.get_tensor_metadata(key);
    }
    
private:
    std::shared_ptr<SharedFileHandle> handle_;
    SafetensorsMetadata metadata_;
};

}//namespace DonNotKnowHowToNameIt

#endif