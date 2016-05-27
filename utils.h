//
// Created by 吴先 on 16/5/27.
//

#ifndef SGD_UTILS_H
#define SGD_UTILS_H


#include <cmath>
#include <vector>
#include <unordered_map>
#include "limonp/Logging.hpp"

double sigmoid(float z) {
    return 1.0 / (1 + exp(-z));
}

template <typename T>
T dot_product(const std::vector<T>& a, const std::vector<T>& b) {
    XCHECK(a.size() == b.size());
    T ret = 0;
    for (size_t i = 0; i < a.size(); i++)
        ret += a[i] * b[i];
    return ret;
}

template <typename K, typename T>
T dot_product(const std::unordered_map<K, T>& a, const std::unordered_map<K, T>& b) {
    T ret = 0;
    for (auto p : a)
        if (b.find(p.first) != b.end())
            ret += p.second * b.at(p.first);
    return ret;
}

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = (size_t) (snprintf(nullptr, 0, format.c_str(), args ... ) + 1); // Extra space for '\0'
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}


#endif //SGD_UTILS_H
