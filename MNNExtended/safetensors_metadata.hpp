//metadata者，模型之户籍也。
//此文件不涉Tensor，不沾内存，可独善其身。
//元数据乃Tensor立身之本，不可不察。

#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

//#include <sys/stat.h>
namespace MyTensor::Utils
{

namespace safetensors
{

using json = nlohmann::json;

class TensorMetadata
{
public:
    std::string dtype;
    std::vector<int64_t> shape;
    std::array<size_t, 2> data_offsets;
    
    TensorMetadata
    (
        const std::string& dtype, 
        const std::vector<int64_t>& shape,
        const std::array<size_t, 2>& offsets
    ) :
    dtype(dtype), shape(shape), data_offsets(offsets) {}
    
    json to_json() const
    {
        return
        {
            {"dtype", dtype},
            {"shape", shape},
            {"data_offsets", data_offsets}
        };
    }
    
    void print() const
    {
        std::cout << "  dtype: " << dtype << "\n";
        std::cout << "  shape: [";
        for (size_t i = 0, nd = shape.size(); i < nd; i++)
        {
            if (i > 0) std::cout << ", ";
            std::cout << shape[i];
        }
        std::cout << "]\n";
        std::cout << "  data_offsets: [" << data_offsets[0] 
                  << ", " << data_offsets[1] << "]\n";
    }
};

class SafetensorsMetadata
{
private:
    std::unordered_map<std::string, std::string> global_metadata;
    std::unordered_map<std::string, TensorMetadata> tensors_metadata;
    size_t data_start_offset;
    bool loaded = false;
    
    void parse_metadata(const json& j)
    {
        if (j.contains("__metadata__"))
        {
            const auto& meta = j["__metadata__"];
            for (auto it = meta.begin(); it != meta.end(); ++it)
            {
                global_metadata[it.key()] = it.value().get<std::string>();
            }
        }
        
        for (auto it = j.begin(); it != j.end(); it++)
        {
            const std::string& key = it.key();
            if (key == "__metadata__") continue;
            
            const json& tensor_json = it.value();
            TensorMetadata meta
            (
                tensor_json["dtype"].get<std::string>(),
                tensor_json["shape"].get<std::vector<int64_t>>(),
                tensor_json["data_offsets"].get<std::array<size_t, 2>>()
            );
            
            tensors_metadata.emplace(key, std::move(meta));
        }
    }
    
    static uint64_t swap_endian(uint64_t value)
    {
        return ((value & 0xFF00000000000000) >> 56) |
               ((value & 0x00FF000000000000) >> 40) |
               ((value & 0x0000FF0000000000) >> 24) |
               ((value & 0x000000FF00000000) >> 8)  |
               ((value & 0x00000000FF000000) << 8)  |
               ((value & 0x0000000000FF0000) << 24) |
               ((value & 0x000000000000FF00) << 40) |
               ((value & 0x00000000000000FF) << 56);
    }

public:
    bool load_from_file(const std::string& file_path)
    {
        std::ifstream file(file_path, std::ios::binary);
        if (!file)
        {
            std::cerr << "Error opening file: " << file_path << std::endl;
            return false;
        }
        
        uint64_t header_size;
        file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        data_start_offset = 8 + header_size;
        //std::cout << "header_size: " << header_size << std::endl;
        
        bool is_big_endian = (header_size > 0xFFFFFFFF);
        if (is_big_endian)
        {
            header_size = swap_endian(header_size);
            /*((header_size & 0xFF00000000000000) >> 56) |
                          ((header_size & 0x00FF000000000000) >> 40) |
                          ((header_size & 0x0000FF0000000000) >> 24) |
                          ((header_size & 0x000000FF00000000) >> 8) |
                          ((header_size & 0x00000000FF000000) << 8) |
                          ((header_size & 0x0000000000FF0000) << 24) |
                          ((header_size & 0x000000000000FF00) << 40) |
                          ((header_size & 0x00000000000000FF) << 56);*/
        }
        
        std::string header_json(header_size, '\0');
        file.read(&header_json[0], header_size);
        
        size_t real_size = header_size;
        while (real_size > 0 && header_json[real_size - 1] == ' ')
        {
            real_size--;
        }
        header_json.resize(real_size);
        
        try
        {
            json j = json::parse(header_json);
            parse_metadata(j);
            loaded = true;
            return true;
        }
        catch (const json::parse_error& e)
        {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            return false;
        }
    }
    
    size_t get_data_start_offset() const
    {
        return data_start_offset;
    }
    
    /*bool load_from_fd(int fd)
    {
        if (fd < 0)
        {
            std::cerr << "Invalid file descriptor" << std::endl;
            return false;
        }
        
        struct stat sb;
        if (fstat(fd, &sb) == -1)
        {
            std::cerr << "Failed to get file size" << std::endl;
            return false;
        }
        size_t file_size = sb.st_size;
        
        uint64_t header_size;
        if (read(fd, &header_size, sizeof(header_size)) != sizeof(header_size))
        {
            std::cerr << "Failed to read header size" << std::endl;
            return false;
        }
        
        bool is_big_endian = (header_size > 0xFFFFFFFF);
        if (is_big_endian)
        {
            header_size = swap_endian(header_size);
        }
        
        if (8 + header_size > file_size)
        {
            std::cerr << "Invalid header size: " << header_size 
                      << ", file size: " << file_size << std::endl;
            return false;
        }
        
        std::string header_json(header_size, '\0');
        if (read(fd, &header_json[0], header_size) != static_cast<ssize_t>(header_size))
        {
            std::cerr << "Failed to read JSON header" << std::endl;
            return false;
        }
        
        size_t real_size = header_size;
        while (real_size > 0 && header_json[real_size - 1] == ' ')
        {
            real_size--;
        }
        header_json.resize(real_size);
        
        try
        {
            json j = json::parse(header_json);
            parse_metadata(j);
            loaded = true;
            return true;
        }
        catch (const json::parse_error& e)
        {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            return false;
        }
    }*/
    
    /*bool load_from_file(const std::string& file_path)
    {
        int fd = open(file_path.c_str(), O_RDONLY);
        if (fd == -1)
        {
            std::cerr << "Error opening file: " << file_path << std::endl;
            return false;
        }
        
        bool success = load_from_fd(fd);
        close(fd);
        return success;
    }*/
    
    /*void add_tensor_metadata
    (
        const std::string& name, 
        const TensorMetadata& metadata
    )
    {
        tensors_metadata[name] = metadata;
    }*/
    
    void add_tensor_metadata
    (
        const std::string& name,
        const TensorMetadata& metadata
    )
    {
    // 使用 emplace 直接构造元素
        tensors_metadata.emplace(name, metadata);
    
    // 或者使用 try_emplace(C++17)
    // tensors_metadata.try_emplace(name, metadata);
    }
    
    void set_global_metadata(const std::unordered_map<std::string, std::string>& metadata)
    {
        global_metadata = metadata;
    }
    
    const TensorMetadata* get_tensor_metadata(const std::string& name) const
    {
        auto it = tensors_metadata.find(name);
        return (it != tensors_metadata.end()) ? &it->second : nullptr;
    }
    
    const std::unordered_map<std::string, TensorMetadata>& get_all_tensors() const 
    {
        return tensors_metadata;
    }
    
    const std::unordered_map<std::string, std::string>& get_global_metadata() const
    {
        return global_metadata;
    }
    
    // 生成符合 safetensors 标准的 JSON 头部
    std::string generate_header() const
    {
        json j;
        
        // 添加全局元数据
        if (!global_metadata.empty())
        {
            json meta_obj;
            for (const auto& [key, value] : global_metadata)
            {
                meta_obj[key] = value;
            }
            j["__metadata__"] = meta_obj;
        }
        
        for (const auto& [name, metadata] : tensors_metadata)
        {
            j[name] = metadata.to_json();
        }
        
        return j.dump();
    }
    
    /*void print_tensor_metadata(const std::string& name) const
    {
        auto it = tensors_metadata.find(name);
        if (it == tensors_metadata.end())
        {
            std::cout << "Tensor '" << name << "' not found\n";
            return;
        }
        
        std::cout << "Tensor: " << name << "\n";
        it->second.print();
    }
    

    void print_all_metadata() const
    {
        if (!global_metadata.empty())
        {
            std::cout << "Global Metadata:\n";
            for (const auto& [key, value] : global_metadata)
            {
                std::cout << "  " << key << ": " << value << "\n";
            }
        }
        
        std::cout << "\nTensors (" << tensors_metadata.size() << "):\n";
        for (const auto& [name, metadata] : tensors_metadata)
        {
            std::cout << "Tensor: " << name << "\n";
            metadata.print();
            std::cout << "\n";
        }
    }*/
    
    void print_tensor_metadata
    (
        const std::string& name, 
        const std::vector<std::string>& fields = {"dtype", "shape", "data_offsets"}
    ) const
    {
        auto it = tensors_metadata.find(name);
        if (it == tensors_metadata.end())
        {
            std::cout << "Tensor '" << name << "' not found\n";
            return;
        }
        
        const TensorMetadata& meta = it->second;
        std::cout << "Tensor: " << name << "\n";
        
        for (const auto& field : fields)
        {
            if (field == "dtype")
            {
                std::cout << "  dtype: " << meta.dtype << "\n";
            } else if (field == "shape")
            {
                std::cout << "  shape: [";
                for (size_t i = 0, nd = meta.shape.size(); i < nd; ++i)
                {
                    if (i > 0) std::cout << ", ";
                    std::cout << meta.shape[i];
                }
                std::cout << "]\n";
            }
            else if (field == "data_offsets")
            {
                std::cout << "  data_offsets: [" << meta.data_offsets[0] 
                          << ", " << meta.data_offsets[1] << "]\n";
            }
        }
    }
    
    // 打印所有 tensor 元数据（支持选择性输出）
    void print_all_metadata
    (
        const std::vector<std::string>& fields = {"dtype", "shape", "data_offsets"}
    ) const
    {
        // 打印全局元数据
        if (!global_metadata.empty())
        {
            std::cout << "Global Metadata:\n";
            for (const auto& [key, value] : global_metadata)
            {
                std::cout << "  " << key << ": " << value << "\n";
            }
        }
        
        // 打印 tensor 元数据
        std::cout << "\nTensors (" << tensors_metadata.size() << "):\n";
        for (const auto& [name, metadata] : tensors_metadata)
        {
            std::cout << "Tensor: " << name << "\n";
            
            for (const auto& field : fields)
            {
                if (field == "dtype")
                {
                    std::cout << "  dtype: " << metadata.dtype << "\n";
                }
                else if (field == "shape")
                {
                    std::cout << "  shape: [";
                    for (size_t i = 0, nd = metadata.shape.size(); i < nd; ++i)
                    {
                        if (i > 0) std::cout << ", ";
                        std::cout << metadata.shape[i];
                    }
                    std::cout << "]\n";
                }
                else if (field == "data_offsets")
                {
                    std::cout << "  data_offsets: [" << metadata.data_offsets[0] 
                              << ", " << metadata.data_offsets[1] << "]\n";
                }
            }
            std::cout << "\n";
        }
    }
    
    bool is_loaded() const { return loaded; }
};

} // namespace safetensors

} // namespace MyTensor::Utils