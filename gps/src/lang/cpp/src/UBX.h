// define the _iterator

#ifndef __UBX_H__
#define __UBX_H__

#include <stddef.h>

template<class T>
class _iterator {
public:
    _iterator<T>(char* data, size_t size) :size(size), data(data), i(0) {}
    _iterator<T>(T& msg, size_t size) :size(size), data((char*)(&msg)), i(0) {}
    bool end() const {
        return i >= size;
    }
    void next() {
        i += sizeof(T);
    }
    T& operator*() const { return *((T*)(data+i)); }
    T* operator->() const { return (T*)(data+i); }
    operator char*() const { return data+i; }
private:
    size_t i, size;
    char* data;
};

struct Message {
};

#endif // #define __UBX_H__
