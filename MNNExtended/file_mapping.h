#ifndef FILE_MAPPING_H
#define FILE_MAPPING_H

#include "shared_file_handle.h"
#include <cstdint>
#include <system_error>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace MyTensor::Utils
{

#if defined(_WIN32)
inline size_t get_page_size()
{
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    return sys_info.dwPageSize;
}
#else
inline size_t get_page_size()
{
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
}
#endif

inline size_t get_align_offset(size_t offset, size_t page_size)
{
    return (offset / page_size) * page_size;
}

class FileMapping
{
public:
    FileMapping
    (
        std::shared_ptr<SharedFileHandle> file_handle,
        size_t offset,
        size_t size,
        bool writable
    ) : file_handle_(file_handle), size_(size), writable_(writable), original_offset_(offset)
    {
        // 计算对齐偏移
        size_t align_offset = get_align_offset(offset, get_page_size());
        start_offset_ = offset - align_offset;
        offset = align_offset;
        size += start_offset_;

        #ifdef _WIN32
        DWORD protect = writable ? PAGE_READWRITE : PAGE_READONLY;
        mapping_handle_ = CreateFileMappingA
        (
            reinterpret_cast<HANDLE>(file_handle->native_handle()),
            NULL,
            protect,
            0, 0,
            NULL
        );

        if (!mapping_handle_)
        {
            throw std::system_error(GetLastError(), std::system_category(), "CreateFileMapping failed");
        }

        DWORD access = writable ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ;
        data_ = MapViewOfFile
        (
            mapping_handle_,
            access,
            static_cast<DWORD>(offset >> 32),
            static_cast<DWORD>(offset & 0xFFFFFFFF),
            static_cast<SIZE_T>(size)
        );

        if (!data_)
        {
            CloseHandle(mapping_handle_);
            mapping_handle_ = INVALID_HANDLE_VALUE;
            throw std::system_error(GetLastError(), std::system_category(), "MapViewOfFile failed");
        }
        #else
        int prot = PROT_READ;
        if (writable)
        {
            prot |= PROT_WRITE;
        }

        data_ = mmap
        (
            nullptr,
            size,
            prot,
            MAP_SHARED,
            static_cast<int>(file_handle->native_handle()),
            static_cast<off_t>(offset)
        );

        if (data_ == MAP_FAILED)
        {
            data_ = nullptr;
            throw std::system_error(errno, std::system_category(), "mmap failed");
        }
        #endif
    }

    ~FileMapping()
    {
        unmap();
    }

    void* base_data() const { return data_; }
    void* data() const { return (void*)((char*)(data_) + start_offset_); }

    size_t size() const { return size_; }

    void flush(bool async = false)
    {
        if (!data_ || !writable_) return;

        #ifdef _WIN32
        if (!FlushViewOfFile(data_, static_cast<SIZE_T>(size_ + start_offset_)))
        {
            throw std::system_error(GetLastError(), std::system_category(), "FlushViewOfFile failed");
        }

        if (!async && !FlushFileBuffers(reinterpret_cast<HANDLE>(file_handle_->native_handle())))
        {
            throw std::system_error(GetLastError(), std::system_category(), "FlushFileBuffers failed");
        }
        #else
        int flags = async ? MS_ASYNC : MS_SYNC;
        if (msync(data_, (size_ + start_offset_), flags) == -1)
        {
            throw std::system_error(errno, std::system_category(), "msync failed");
        }
        #endif
    }

    void unmap()
    {
        if (!data_) return;

        #ifdef _WIN32
        if (data_)
        {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }

        if (mapping_handle_ != INVALID_HANDLE_VALUE)
        {
            CloseHandle(mapping_handle_);
            mapping_handle_ = INVALID_HANDLE_VALUE;
        }
        #else
        if (data_)
        {
            munmap(data_, size_ + start_offset_);
            data_ = nullptr;
        }
        #endif
    }

    void resize(size_t new_size)
    {
        if (new_size == size_) return;

        unmap();

        size_t align_offset = get_align_offset(original_offset_, get_page_size());
        start_offset_ = original_offset_ - align_offset;
        size_t aligned_total_size = new_size + start_offset_;

        #ifdef _WIN32
        DWORD protect = writable_ ? PAGE_READWRITE : PAGE_READONLY;
        mapping_handle_ = CreateFileMappingA
        (
            reinterpret_cast<HANDLE>(file_handle_->native_handle()),
            NULL,
            protect,
            0, 0,
            NULL
        );

        if (!mapping_handle_)
        {
            throw std::system_error(GetLastError(), std::system_category(), "CreateFileMapping failed in resize");
        }

        DWORD access = writable_ ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ;
        data_ = MapViewOfFile
        (
            mapping_handle_,
            access,
            static_cast<DWORD>(align_offset >> 32),
            static_cast<DWORD>(align_offset & 0xFFFFFFFF),
            static_cast<SIZE_T>(aligned_total_size)
        );

        if (!data_)
        {
            CloseHandle(mapping_handle_);
            mapping_handle_ = INVALID_HANDLE_VALUE;
            throw std::system_error(GetLastError(), std::system_category(), "MapViewOfFile failed in resize");
        }
        #else
        int prot = PROT_READ;
        if (writable_)
            prot |= PROT_WRITE;

        data_ = mmap
        (
            nullptr,
            aligned_total_size,
            prot,
            MAP_SHARED,
            static_cast<int>(file_handle_->native_handle()),
            static_cast<off_t>(align_offset)
        );

        if (data_ == MAP_FAILED)
        {
            data_ = nullptr;
            throw std::system_error(errno, std::system_category(), "mmap failed in resize");
        }
        #endif

        size_ = new_size;
    }

protected:
    void* data_ = nullptr;
    size_t start_offset_;
    size_t size_ = 0;
    bool writable_ = false;
    size_t original_offset_;
    std::shared_ptr<SharedFileHandle> file_handle_;

    #ifdef _WIN32
    HANDLE mapping_handle_ = INVALID_HANDLE_VALUE;
    #endif
};

class TemporaryFileMapping : public MyTensor::Utils::FileMapping
{
public:
    TemporaryFileMapping(std::string path, size_t size) :
        FileMapping
        (
            SharedFileHandle::create(size, path),
            0, size, true
        )
    {}
    
    ~TemporaryFileMapping()
    {
        std::remove(file_handle_->file_path().c_str());
    }
};

} // namespace MyTensor::Utils

#endif // FILE_MAPPING_H