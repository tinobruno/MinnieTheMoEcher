#pragma once

#include <string>
#include <cstdint>
#include <cstddef>
#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <intrin.h>
    #pragma comment(lib, "ws2_32.lib")
    #pragma comment(lib, "crypt32.lib")
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <immintrin.h>
#endif

namespace moecher {
namespace platform {

// ── CPU Intrinsics ──────────────────────────────────────────────────────────
inline int popcount32(uint32_t x) {
#if defined(_WIN32) || defined(_WIN64)
    return (int)__popcnt(x);
#else
    return __builtin_popcount(x);
#endif
}

inline int popcount64(uint64_t x) {
#if defined(_WIN32) || defined(_WIN64)
    return (int)__popcnt64(x);
#else
    return __builtin_popcountll(x);
#endif
}

// ── Cross-Platform Memory Mapped File (for dense weights) ───────────────────
class MemoryMappedFile {
public:
    MemoryMappedFile() : data_(nullptr), size_(0)
#if defined(_WIN32) || defined(_WIN64)
        , file_handle_(INVALID_HANDLE_VALUE), mapping_handle_(NULL)
#else
        , fd_(-1)
#endif
    {}

    ~MemoryMappedFile() {
        close();
    }

    bool open_read(const std::string& path) {
        close();

#if defined(_WIN32) || defined(_WIN64)
        file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                   NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            return false;
        }

        LARGE_INTEGER file_size_large;
        if (!GetFileSizeEx(file_handle_, &file_size_large)) {
            close();
            return false;
        }
        size_ = static_cast<size_t>(file_size_large.QuadPart);

        mapping_handle_ = CreateFileMappingA(file_handle_, NULL, PAGE_READONLY, 0, 0, NULL);
        if (mapping_handle_ == NULL) {
            close();
            return false;
        }

        data_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, size_);
        if (data_ == nullptr) {
            close();
            return false;
        }
        return true;
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            return false;
        }

        struct stat st;
        if (fstat(fd_, &st) != 0) {
            close();
            return false;
        }
        size_ = static_cast<size_t>(st.st_size);

        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            close();
            return false;
        }
        return true;
#endif
    }

    void close() {
#if defined(_WIN32) || defined(_WIN64)
        if (data_ != nullptr) {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (mapping_handle_ != NULL) {
            CloseHandle(mapping_handle_);
            mapping_handle_ = NULL;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (data_ != nullptr && size_ > 0) {
            munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        size_ = 0;
    }

    void* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }

private:
    void* data_;
    size_t size_;

#if defined(_WIN32) || defined(_WIN64)
    HANDLE file_handle_;
    HANDLE mapping_handle_;
#else
    int fd_;
#endif
};

// ── Direct File Handle (Thread-Safe Asynchronous / Direct I/O) ──────────────
class DirectFileHandle {
public:
    DirectFileHandle()
#if defined(_WIN32) || defined(_WIN64)
        : handle_(INVALID_HANDLE_VALUE), is_direct_(false)
#else
        : fd_(-1), is_direct_(false)
#endif
    {}

    ~DirectFileHandle() {
        close();
    }

    bool open_read(const std::string& path, bool use_direct_io = true) {
        close();

#if defined(_WIN32) || defined(_WIN64)
        DWORD flags = FILE_ATTRIBUTE_READONLY | FILE_FLAG_SEQUENTIAL_SCAN;
        if (use_direct_io) {
            flags |= FILE_FLAG_NO_BUFFERING;
        }

        handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              NULL, OPEN_EXISTING, flags, NULL);

        if (handle_ == INVALID_HANDLE_VALUE && use_direct_io) {
            // Fallback to standard buffered read if NO_BUFFERING fails
            flags = FILE_ATTRIBUTE_READONLY | FILE_FLAG_SEQUENTIAL_SCAN;
            handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                  NULL, OPEN_EXISTING, flags, NULL);
            use_direct_io = false;
        }

        if (handle_ == INVALID_HANDLE_VALUE) {
            return false;
        }
        is_direct_ = use_direct_io;
        return true;
#else
        if (use_direct_io) {
            fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd_ >= 0) {
                is_direct_ = true;
                return true;
            }
        }
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ >= 0) {
            is_direct_ = false;
            return true;
        }
        return false;
#endif
    }

    int64_t pread_exact(void* dst, size_t count, uint64_t offset) {
#if defined(_WIN32) || defined(_WIN64)
        if (handle_ == INVALID_HANDLE_VALUE) return -1;

        OVERLAPPED ov = {0};
        ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
        ov.OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFF);

        DWORD bytes_read = 0;
        if (!ReadFile(handle_, dst, static_cast<DWORD>(count), &bytes_read, &ov)) {
            DWORD err = GetLastError();
            if (err != ERROR_HANDLE_EOF) {
                return -1;
            }
        }
        return static_cast<int64_t>(bytes_read);
#else
        if (fd_ < 0) return -1;
        return ::pread(fd_, dst, count, offset);
#endif
    }

    void close() {
#if defined(_WIN32) || defined(_WIN64)
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        is_direct_ = false;
    }

    bool is_open() const {
#if defined(_WIN32) || defined(_WIN64)
        return handle_ != INVALID_HANDLE_VALUE;
#else
        return fd_ >= 0;
#endif
    }

    bool is_direct() const { return is_direct_; }

private:
#if defined(_WIN32) || defined(_WIN64)
    HANDLE handle_;
#else
    int fd_;
#endif
    bool is_direct_;
};

} // namespace platform
} // namespace moecher
