#ifndef SHARED_FILE_HANDLE_H
#define SHARED_FILE_HANDLE_H

#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <system_error>
#include <stdexcept>
#include <cstdint>
#include <functional>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#endif

namespace MyTensor::Utils
{

class SharedFileHandle : public std::enable_shared_from_this<SharedFileHandle>
{
public:
    static std::shared_ptr<SharedFileHandle> create
    (
        const std::string& file_path, 
        bool writable,
        bool create_if_not_exists = false
    )
    {
        return
        std::shared_ptr<SharedFileHandle>
        (
            new SharedFileHandle(file_path, writable, create_if_not_exists)
        );
    }

    ~SharedFileHandle()
    {
        #ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE)
        {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
        #else
        if (fd_ != -1)
        {
            close(fd_);
            fd_ = -1;
        }
        #endif
    }
    

    intptr_t native_handle() const
    {
        #ifdef _WIN32
        return reinterpret_cast<intptr_t>(handle_);
        #else
        return static_cast<intptr_t>(fd_);
        #endif
    }
    
    uint64_t size() const
    {
        #ifdef _WIN32
        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(handle_, &file_size))
        {
            throw std::system_error(GetLastError(), std::system_category(), "GetFileSizeEx failed");
        }
        return static_cast<uint64_t>(file_size.QuadPart);
        #else
        struct stat st;
        if (fstat(fd_, &st) == -1)
        {
            throw std::system_error(errno, std::system_category(), "fstat failed");
        }
        return static_cast<uint64_t>(st.st_size);
        #endif
    }
    
    void truncate(uint64_t new_size)
    {
        #ifdef _WIN32
        LARGE_INTEGER new_size_li;
        new_size_li.QuadPart = new_size;
    
        if (!SetFilePointerEx(handle_, new_size_li, NULL, FILE_BEGIN))
        {
            throw std::system_error(GetLastError(), std::system_category(), "SetFilePointerEx failed");
        }
    
        if (!SetEndOfFile(handle_))
        {
            throw std::system_error(GetLastError(), std::system_category(), "SetEndOfFile failed");
        }
        #else
        if (ftruncate(fd_, static_cast<off_t>(new_size)) == -1)
        {
            throw std::system_error(errno, std::system_category(), "ftruncate failed");
        }
        #endif
    }
    
    const std::string& file_path() const { return file_path_; }
    
    bool is_writable() const { return writable_; }
    
private:
    SharedFileHandle
    (
        const std::string& file_path,
        bool writable,
        bool create_if_not_exists
    ) : file_path_(file_path), writable_(writable)
    {
        #ifdef _WIN32
        DWORD desired_access = writable ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ;
        DWORD creation_disposition = create_if_not_exists ? OPEN_ALWAYS : OPEN_EXISTING;

        handle_ =
        CreateFileA
        (
            file_path.c_str(),
            desired_access,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            NULL,
            creation_disposition,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );

        if (handle_ == INVALID_HANDLE_VALUE)
        {
            throw std::system_error(GetLastError(), std::system_category(), "Failed to open file: " + file_path);
        }
        #else
        int flags = writable ? O_RDWR : O_RDONLY;
        if (create_if_not_exists)
        {
            flags |= O_CREAT;
        }

        fd_ = open(file_path.c_str(), flags, 0666);
        if (fd_ == -1)
        {
            throw std::system_error(errno, std::system_category(), "Failed to open file: " + file_path);
        }
        #endif
    }
    
    #ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    #else
    int fd_ = -1;
    #endif
    
    std::string file_path_;
    bool writable_;
};

class FileHandleCache
{
public:
    FileHandleCache(const FileHandleCache&) = delete;
    FileHandleCache& operator=(const FileHandleCache&) = delete;
   
    static FileHandleCache& instance()
    {
        static FileHandleCache instance;
        return instance;
    }
    
    std::shared_ptr<SharedFileHandle> getHandle
    (
        const std::string& file_path, 
        bool writable,
        bool create_if_not_exists = false
    )
    {
        std::lock_guard<std::mutex> lock(mutex_);
    
        std::string key = file_path + (writable ? ":rw" : ":ro");
    
        auto it = cache_.find(key);
        if (it != cache_.end())
        {
            if (auto handle = it->second.lock())
            {
                return handle;
            }
        }
    
        auto handle = SharedFileHandle::create(file_path, writable, create_if_not_exists);
        cache_[key] = handle;
        return handle;
    }

    
    void cleanup()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = cache_.begin(); it != cache_.end(); )
        {
            if (it->second.expired())
            {
                it = cache_.erase(it);
            } 
            else 
            {
                ++it;
            }
        }
    }
    
    static std::shared_ptr<SharedFileHandle> get_handle
    (
        const std::string& file_path, 
        bool writable,
        bool create_if_not_exists = false
    )
    {
        return instance().getHandle(file_path, writable, create_if_not_exists);
    }

private:
    FileHandleCache() = default;
    
    std::mutex mutex_;
    std::unordered_map<std::string, std::weak_ptr<SharedFileHandle>> cache_;
};

}//namespace MyTensor::Utils

#endif // SHARED_FILE_HANDLE_H