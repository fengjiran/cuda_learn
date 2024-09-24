//
// Created by richard on 9/22/24.
//

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 */

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#include <cuda_runtime.h>

using uint = unsigned int;
using ushort = unsigned short;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////
inline float fminf(float a, float b) {
    return a < b ? a : b;
}

inline float fmaxf(float a, float b) {
    return a > b ? a : b;
}

inline int max(int a, int b) {
    return a > b ? a : b;
}

inline int min(int a, int b) {
    return a < b ? a : b;
}

inline float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ int2 make_int2(const int s) {
    return make_int2(s, s);
}

inline __host__ __device__ int2 make_int2(const int3& a) {
    return make_int2(a.x, a.y);
}

inline __host__ __device__ int2 make_int2(const uint2& a) {
    return make_int2(static_cast<int>(a.x), static_cast<int>(a.y));
}

inline __host__ __device__ int2 make_int2(const float2& a) {
    return make_int2(static_cast<int>(a.x), static_cast<int>(a.y));
}

inline __host__ __device__ uint2 make_uint2(const uint s) {
    return make_uint2(s, s);
}

inline __host__ __device__ uint2 make_uint2(const uint3& a) {
    return make_uint2(a.x, a.y);
}

inline __host__ __device__ uint2 make_uint2(const int2& a) {
    return make_uint2(static_cast<uint>(a.x), static_cast<uint>(a.y));
}

inline __host__ __device__ int3 make_int3(const int s) {
    return make_int3(s, s, s);
}

inline __host__ __device__ int3 make_int3(const int2& a) {
    return make_int3(a.x, a.y, 0);
}

inline __host__ __device__ int3 make_int3(const int2& a, const int s) {
    return make_int3(a.x, a.y, s);
}

inline __host__ __device__ int3 make_int3(const uint3& a) {
    return make_int3(static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z));
}

inline __host__ __device__ int3 make_int3(const float3& a) {
    return make_int3(static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z));
}

inline __host__ __device__ uint3 make_uint3(const uint s) {
    return make_uint3(s, s, s);
}

inline __host__ __device__ uint3 make_uint3(const uint2& a) {
    return make_uint3(a.x, a.y, 0);
}

inline __host__ __device__ uint3 make_uint3(const uint2& a, const uint s) {
    return make_uint3(a.x, a.y, s);
}

inline __host__ __device__ uint3 make_uint3(const uint4& a) {
    return make_uint3(a.x, a.y, a.z);
}

inline __host__ __device__ uint3 make_uint3(const int3& a) {
    return make_uint3(static_cast<uint>(a.x), static_cast<uint>(a.y), static_cast<uint>(a.z));
}

inline __host__ __device__ int4 make_int4(const int s) {
    return make_int4(s, s, s, s);
}

inline __host__ __device__ int4 make_int4(const int3& a) {
    return make_int4(a.x, a.y, a.z, 0);
}

inline __host__ __device__ int4 make_int4(const int3& a, const int w) {
    return make_int4(a.x, a.y, a.z, w);
}

inline __host__ __device__ int4 make_int4(const uint4& a) {
    return make_int4(static_cast<int>(a.x), static_cast<int>(a.y),
                     static_cast<int>(a.z), static_cast<int>(a.w));
}

inline __host__ __device__ int4 make_int4(const float4& a) {
    return make_int4(static_cast<int>(a.x), static_cast<int>(a.y),
                     static_cast<int>(a.z), static_cast<int>(a.w));
}

inline __host__ __device__ uint4 make_uint4(const uint s) {
    return make_uint4(s, s, s, s);
}

inline __host__ __device__ uint4 make_uint4(const uint3& a) {
    return make_uint4(a.x, a.y, a.z, 0);
}

inline __host__ __device__ uint4 make_uint4(const uint3& a, const uint w) {
    return make_uint4(a.x, a.y, a.z, w);
}

inline __host__ __device__ uint4 make_uint4(const int4& a) {
    return make_uint4(static_cast<uint>(a.x), static_cast<uint>(a.y),
                      static_cast<uint>(a.z), static_cast<uint>(a.w));
}

inline __host__ __device__ float2 make_float2(const float s) {
    return make_float2(s, s);
}

inline __host__ __device__ float2 make_float2(const float3& a) {
    return make_float2(a.x, a.y);
}

inline __host__ __device__ float2 make_float2(const int2& a) {
    return make_float2(static_cast<float>(a.x), static_cast<float>(a.y));
}

inline __host__ __device__ float2 make_float2(const uint2& a) {
    return make_float2(static_cast<float>(a.x), static_cast<float>(a.y));
}


inline __host__ __device__ float3 make_float3(const float s) {
    return make_float3(s, s, s);
}

inline __host__ __device__ float3 make_float3(const float2& a) {
    return make_float3(a.x, a.y, 0.0f);
}

inline __host__ __device__ float3 make_float3(const float2& a, const float s) {
    return make_float3(a.x, a.y, s);
}

inline __host__ __device__ float3 make_float3(const float4& a) {
    return make_float3(a.x, a.y, a.z);
}

inline __host__ __device__ float3 make_float3(const int3& a) {
    return make_float3(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z));
}

inline __host__ __device__ float3 make_float3(const uint3& a) {
    return make_float3(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z));
}

inline __host__ __device__ float4 make_float4(const float s) {
    return make_float4(s, s, s, s);
}

inline __host__ __device__ float4 make_float4(const float3& a) {
    return make_float4(a.x, a.y, a.z, 0.0f);
}

inline __host__ __device__ float4 make_float4(const float3& a, const float w) {
    return make_float4(a.x, a.y, a.z, w);
}

inline __host__ __device__ float4 make_float4(const int4& a) {
    return make_float4(static_cast<float>(a.x), static_cast<float>(a.y),
                       static_cast<float>(a.z), static_cast<float>(a.w));
}

inline __host__ __device__ float4 make_float4(const uint4& a) {
    return make_float4(static_cast<float>(a.x), static_cast<float>(a.y),
                       static_cast<float>(a.z), static_cast<float>(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ int2 operator-(const int2& a) {
    return make_int2(-a.x, -a.y);
}

inline __host__ __device__ int3 operator-(const int3& a) {
    return make_int3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ int4 operator-(const int4& a) {
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

inline __host__ __device__ float2 operator-(const float2& a) {
    return make_float2(-a.x, -a.y);
}

inline __host__ __device__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float4 operator-(const float4& a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ float2 operator+(const float b, const float2& a) {
    return make_float2(a.x + b, a.y + b);
}

inline __host__ __device__ void operator+=(float2& a, const float b) {
    a.x += b;
    a.y += b;
}

inline __host__ __device__ int2 operator+(const int2& a, const int2& b) {
    return make_int2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(int2& a, const int2& b) {
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ int2 operator+(const int2& a, const int b) {
    return make_int2(a.x + b, a.y + b);
}

inline __host__ __device__ int2 operator+(const int b, const int2& a) {
    return make_int2(a.x + b, a.y + b);
}

inline __host__ __device__ void operator+=(int2& a, const int b) {
    a.x += b;
    a.y += b;
}

inline __host__ __device__ uint2 operator+(const uint2& a, const uint2& b) {
    return make_uint2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(uint2& a, const uint2& b) {
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ uint2 operator+(const uint2& a, const uint b) {
    return make_uint2(a.x + b, a.y + b);
}

inline __host__ __device__ uint2 operator+(const uint b, const uint2& a) {
    return make_uint2(a.x + b, a.y + b);
}

inline __host__ __device__ void operator+=(uint2& a, const uint b) {
    a.x += b;
    a.y += b;
}

inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __host__ __device__ float3 operator+(const float3& a, const float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ void operator+=(float3& a, const float b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(const int3& a, const int3& b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator+=(int3& a, const int3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __host__ __device__ int3 operator+(const int3& a, const int b) {
    return make_int3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ void operator+=(int3& a, const int b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ uint3 operator+(const uint3& a, const uint3& b) {
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator+=(uint3& a, const uint3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __host__ __device__ uint3 operator+(const uint3& a, const uint b) {
    return make_uint3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ void operator+=(uint3& a, const uint b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(const int b, const int3& a) {
    return make_int3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ uint3 operator+(const uint b, const uint3& a) {
    return make_uint3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float3 operator+(const float b, const float3& a) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __host__ __device__ void operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __host__ __device__ float4 operator+(const float4& a, const float b) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline __host__ __device__ float4 operator+(const float b, const float4& a) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline __host__ __device__ void operator+=(float4& a, const float b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ int4 operator+(const int4& a, const int4& b) {
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __host__ __device__ void operator+=(int4& a, const int4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __host__ __device__ int4 operator+(const int4& a, const int b) {
    return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline __host__ __device__ int4 operator+(const int b, const int4& a) {
    return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline __host__ __device__ void operator+=(int4& a, const int b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ uint4 operator+(const uint4& a, const uint4& b) {
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __host__ __device__ void operator+=(uint4& a, const uint4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __host__ __device__ uint4 operator+(const uint4& a, const uint b) {
    return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline __host__ __device__ uint4 operator+(const uint b, const uint4& a) {
    return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline __host__ __device__ void operator+=(uint4& a, const uint b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ void operator-=(float2& a, const float2& b) {
    a.x -= b.x;
    a.y -= b.y;
}

inline __host__ __device__ float2 operator-(const float2& a, const float b) {
    return make_float2(a.x - b, a.y - b);
}

inline __host__ __device__ float2 operator-(const float b, const float2& a) {
    return make_float2(b - a.x, b - a.y);
}

inline __host__ __device__ void operator-=(float2& a, const float b) {
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ int2 operator-(const int2& a, const int2& b) {
    return make_int2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ void operator-=(int2& a, const int2& b) {
    a.x -= b.x;
    a.y -= b.y;
}

inline __host__ __device__ int2 operator-(const int2& a, const int b) {
    return make_int2(a.x - b, a.y - b);
}

inline __host__ __device__ int2 operator-(const int b, const int2& a) {
    return make_int2(b - a.x, b - a.y);
}

inline __host__ __device__ void operator-=(int2& a, const int b) {
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ uint2 operator-(const uint2& a, const uint2& b) {
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2& a, const uint2& b) {
    a.x -= b.x;
    a.y -= b.y;
}

inline __host__ __device__ uint2 operator-(const uint2& a, const uint& b) {
    return make_uint2(a.x - b, a.y - b);
}

inline __host__ __device__ uint2 operator-(const uint b, const uint2& a) {
    return make_uint2(b - a.x, b - a.y);
}

inline __host__ __device__ void operator-=(uint2& a, const uint b) {
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

inline __host__ __device__ float3 operator-(const float3& a, const float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

inline __host__ __device__ float3 operator-(const float b, const float3& a) {
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3& a, const float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ int3 operator-(const int3& a, const int3& b) {
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(int3& a, const int3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

inline __host__ __device__ int3 operator-(const int3& a, const int b) {
    return make_int3(a.x - b, a.y - b, a.z - b);
}

inline __host__ __device__ int3 operator-(const int b, const int3& a) {
    return make_int3(b - a.x, b - a.y, b - a.z);
}

inline __host__ __device__ void operator-=(int3& a, const int b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ uint3 operator-(const uint3& a, const uint3& b) {
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(uint3& a, const uint3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

inline __host__ __device__ uint3 operator-(const uint3& a, const uint b) {
    return make_uint3(a.x - b, a.y - b, a.z - b);
}

inline __host__ __device__ uint3 operator-(const uint b, const uint3& a) {
    return make_uint3(b - a.x, b - a.y, b - a.z);
}

inline __host__ __device__ void operator-=(uint3& a, const uint b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __host__ __device__ void operator-=(float4& a, const float4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

inline __host__ __device__ float4 operator-(const float4& a, const float b) {
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

inline __host__ __device__ void operator-=(float4& a, const float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ int4 operator-(const int4& a, const int4& b) {
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __host__ __device__ void operator-=(int4& a, const int4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

inline __host__ __device__ int4 operator-(const int4& a, const int b) {
    return make_int4(a.x - b, a.y - b, a.z - b, a.w - b);
}

inline __host__ __device__ int4 operator-(const int b, const int4& a) {
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}

inline __host__ __device__ void operator-=(int4& a, const int b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ uint4 operator-(const uint4& a, const uint4& b) {
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __host__ __device__ void operator-=(uint4& a, const uint4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

inline __host__ __device__ uint4 operator-(const uint4& a, const uint b) {
    return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b);
}

inline __host__ __device__ uint4 operator-(uint b, const uint4& a) {
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}

inline __host__ __device__ void operator-=(uint4& a, const uint b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ void operator*=(float2& a, const float2& b) {
    a.x *= b.x;
    a.y *= b.y;
}

inline __host__ __device__ float2 operator*(const float2& a, const float b) {
    return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ float2 operator*(const float b, const float2& a) {
    return make_float2(b * a.x, b * a.y);
}

inline __host__ __device__ void operator*=(float2& a, const float b) {
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ int2 operator*(const int2& a, const int2& b) {
    return make_int2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ void operator*=(int2& a, const int2& b) {
    a.x *= b.x;
    a.y *= b.y;
}

inline __host__ __device__ int2 operator*(const int2& a, const int b) {
    return make_int2(a.x * b, a.y * b);
}

inline __host__ __device__ int2 operator*(const int b, const int2& a) {
    return make_int2(b * a.x, b * a.y);
}

inline __host__ __device__ void operator*=(int2& a, const int b) {
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ uint2 operator*(const uint2& a, const uint2& b) {
    return make_uint2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ void operator*=(uint2& a, const uint2& b) {
    a.x *= b.x;
    a.y *= b.y;
}

inline __host__ __device__ uint2 operator*(const uint2& a, const uint b) {
    return make_uint2(a.x * b, a.y * b);
}

inline __host__ __device__ uint2 operator*(const uint b, const uint2& a) {
    return make_uint2(b * a.x, b * a.y);
}

inline __host__ __device__ void operator*=(uint2& a, const uint b) {
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ void operator*=(float3& a, const float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

inline __host__ __device__ float3 operator*(const float3& a, const float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(const float b, const float3& a) {
    return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ void operator*=(float3& a, const float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ int3 operator*(const int3& a, const int3& b) {
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ void operator*=(int3& a, const int3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

inline __host__ __device__ int3 operator*(const int3& a, const int b) {
    return make_int3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ int3 operator*(const int b, const int3& a) {
    return make_int3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ void operator*=(int3& a, const int b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ uint3 operator*(const uint3& a, const uint3& b) {
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ void operator*=(uint3& a, const uint3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

inline __host__ __device__ uint3 operator*(const uint3& a, const uint b) {
    return make_uint3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ uint3 operator*(const uint b, const uint3& a) {
    return make_uint3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ void operator*=(uint3& a, const uint b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float4 operator*(const float4& a, const float4& b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __host__ __device__ void operator*=(float4& a, const float4& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

inline __host__ __device__ float4 operator*(const float4& a, const float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __host__ __device__ float4 operator*(const float b, const float4& a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

inline __host__ __device__ void operator*=(float4& a, const float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ int4 operator*(const int4& a, const int4& b) {
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __host__ __device__ void operator*=(int4& a, const int4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

inline __host__ __device__ int4 operator*(const int4& a, const int b) {
    return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __host__ __device__ int4 operator*(const int b, const int4& a) {
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}

inline __host__ __device__ void operator*=(int4& a, const int b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ uint4 operator*(const uint4& a, const uint4& b) {
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __host__ __device__ void operator*=(uint4& a, const uint4& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

inline __host__ __device__ uint4 operator*(const uint4& a, const uint b) {
    return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __host__ __device__ uint4 operator*(const uint b, const uint4& a) {
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}

inline __host__ __device__ void operator*=(uint4& a, const uint b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 operator/(const float2& a, const float2& b) {
    return make_float2(a.x / b.x, a.y / b.y);
}

inline __host__ __device__ void operator/=(float2& a, const float2& b) {
    a.x /= b.x;
    a.y /= b.y;
}

inline __host__ __device__ float2 operator/(const float2& a, const float b) {
    return make_float2(a.x / b, a.y / b);
}

inline __host__ __device__ void operator/=(float2& a, const float b) {
    a.x /= b;
    a.y /= b;
}

inline __host__ __device__ float2 operator/(const float b, const float2& a) {
    return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ void operator/=(float3& a, const float3& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

inline __host__ __device__ float3 operator/(const float3& a, const float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ void operator/=(float3& a, const float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

inline __host__ __device__ float3 operator/(const float b, const float3& a) {
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(const float4& a, const float4& b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __host__ __device__ void operator/=(float4& a, const float4& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}

inline __host__ __device__ float4 operator/(const float4& a, const float b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

inline __host__ __device__ void operator/=(float4& a, const float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

inline __host__ __device__ float4 operator/(float b, const float4& a) {
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 fminf(const float2& a, const float2& b) {
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

inline __host__ __device__ float3 fminf(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

inline __host__ __device__ float4 fminf(const float4& a, const float4& b) {
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

inline __host__ __device__ int2 min(const int2& a, const int2& b) {
    return make_int2(min(a.x, b.x), min(a.y, b.y));
}

inline __host__ __device__ int3 min(const int3& a, const int3& b) {
    return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline __host__ __device__ int4 min(const int4& a, const int4& b) {
    return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

inline __host__ __device__ uint2 min(const uint2& a, const uint2& b) {
    return make_uint2(min(a.x, b.x), min(a.y, b.y));
}

inline __host__ __device__ uint3 min(const uint3& a, const uint3& b) {
    return make_uint3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline __host__ __device__ uint4 min(const uint4& a, const uint4& b) {
    return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 fmaxf(const float2& a, const float2& b) {
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

inline __host__ __device__ float3 fmaxf(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline __host__ __device__ float4 fmaxf(const float4& a, const float4& b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

inline __host__ __device__ int2 max(const int2& a, const int2& b) {
    return make_int2(max(a.x, b.x), max(a.y, b.y));
}

inline __host__ __device__ int3 max(const int3& a, const int3& b) {
    return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

inline __host__ __device__ int4 max(const int4& a, const int4& b) {
    return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

inline __host__ __device__ uint2 max(const uint2& a, const uint2& b) {
    return make_uint2(max(a.x, b.x), max(a.y, b.y));
}

inline __host__ __device__ uint3 max(const uint3& a, const uint3& b) {
    return make_uint3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

inline __host__ __device__ uint4 max(const uint4& a, const uint4& b) {
    return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float lerp(const float a, const float b, const float t) {
    return a + t * (b - a);
}

inline __device__ __host__ float2 lerp(const float2& a, const float2& b, const float t) {
    return a + t * (b - a);
}

inline __device__ __host__ float3 lerp(const float3& a, const float3& b, const float t) {
    return a + t * (b - a);
}

inline __device__ __host__ float4 lerp(const float4& a, const float4& b, const float t) {
    return a + t * (b - a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////
inline __device__ __host__ float clamp(const float f, const float a, const float b) {
    return fmaxf(a, fminf(f, b));
}

inline __device__ __host__ int clamp(const int f, const int a, const int b) {
    return max(a, min(f, b));
}

inline __device__ __host__ uint clamp(const uint f, const uint a, const uint b) {
    return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(const float2& v, const float a, const float b) {
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline __device__ __host__ float2 clamp(const float2& v, const float2& a, const float2& b) {
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

inline __device__ __host__ float3 clamp(const float3& v, const float a, const float b) {
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ float3 clamp(const float3& v, const float3& a, const float3& b) {
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

inline __device__ __host__ float4 clamp(const float4& v, const float a, const float b) {
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline __device__ __host__ float4 clamp(const float4& v, const float4& a, const float4& b) {
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(const int2& v, const int a, const int b) {
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline __device__ __host__ int2 clamp(const int2& v, const int2& a, const int2& b) {
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

inline __device__ __host__ int3 clamp(const int3& v, const int a, const int b) {
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ int3 clamp(const int3& v, const int3& a, const int3& b) {
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(const int4& v, const int a, const int b) {
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline __device__ __host__ int4 clamp(const int4& v, const int4& a, const int4& b) {
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(const uint2& v, const uint a, const uint b) {
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline __device__ __host__ uint2 clamp(const uint2& v, const uint2& a, const uint2& b) {
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

inline __device__ __host__ uint3 clamp(const uint3& v, const uint a, const uint b) {
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ uint3 clamp(const uint3& v, const uint3& a, const uint3& b) {
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

inline __device__ __host__ uint4 clamp(const uint4& v, const uint a, const uint b) {
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline __device__ __host__ uint4 clamp(const uint4& v, const uint4& a, const uint4& b) {
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float dot(const float2& a, const float2& b) {
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ int dot(const int2& a, const int2& b) {
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ int dot(const int3& a, const int3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ int dot(const int4& a, const int4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(const uint2& a, const uint2& b) {
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ uint dot(const uint3& a, const uint3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ uint dot(const uint4& a, const uint4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float length(const float2& v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float length(const float4& v) {
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 normalize(const float2& v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline __host__ __device__ float3 normalize(const float3& v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline __host__ __device__ float4 normalize(const float4& v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 floorf(const float2& v) {
    return make_float2(floorf(v.x), floorf(v.y));
}

inline __host__ __device__ float3 floorf(const float3& v) {
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

inline __host__ __device__ float4 floorf(const float4& v) {
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float fracf(const float v) {
    return v - floorf(v);
}

inline __host__ __device__ float2 fracf(const float2& v) {
    return make_float2(fracf(v.x), fracf(v.y));
}

inline __host__ __device__ float3 fracf(const float3& v) {
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}

inline __host__ __device__ float4 fracf(const float4& v) {
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 fmodf(const float2& a, const float2& b) {
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}

inline __host__ __device__ float3 fmodf(const float3& a, const float3& b) {
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}

inline __host__ __device__ float4 fmodf(const float4& a, const float4& b) {
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 fabs(const float2& v) {
    return make_float2(fabs(v.x), fabs(v.y));
}

inline __host__ __device__ float3 fabs(const float3& v) {
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

inline __host__ __device__ float4 fabs(const float4& v) {
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline __host__ __device__ int2 abs(const int2& v) {
    return make_int2(abs(v.x), abs(v.y));
}

inline __host__ __device__ int3 abs(const int3& v) {
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}

inline __host__ __device__ int4 abs(const int4& v) {
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float3 reflect(const float3& i, const float3& n) {
    return i - 2.0f * n * dot(n, i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////
inline __device__ __host__ float smoothstep(const float a, const float b, const float x) {
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (3.0f - (2.0f * y)));
}

inline __device__ __host__ float2 smoothstep(const float2& a, const float2& b, const float2& x) {
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (make_float2(3.0f) - (make_float2(2.0f) * y)));
}

inline __device__ __host__ float3 smoothstep(const float3& a, const float3& b, const float3& x) {
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (make_float3(3.0f) - (make_float3(2.0f) * y)));
}

inline __device__ __host__ float4 smoothstep(const float4& a, const float4& b, const float4& x) {
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (make_float4(3.0f) - (make_float4(2.0f) * y)));
}

#endif//HELPER_MATH_H
