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

inline __host__ __device__ int2 make_int2(const int3 a) {
    return make_int2(a.x, a.y);
}

inline __host__ __device__ int2 make_int2(const uint2 a) {
    return make_int2(static_cast<int>(a.x), static_cast<int>(a.y));
}

inline __host__ __device__ int2 make_int2(const float2 a) {
    return make_int2(static_cast<int>(a.x), static_cast<int>(a.y));
}

inline __host__ __device__ uint2 make_uint2(const uint s) {
    return make_uint2(s, s);
}

inline __host__ __device__ uint2 make_uint2(const uint3 a) {
    return make_uint2(a.x, a.y);
}

inline __host__ __device__ uint2 make_uint2(const int2 a) {
    return make_uint2(static_cast<uint>(a.x), static_cast<uint>(a.y));
}

inline __host__ __device__ int3 make_int3(const int s) {
    return make_int3(s, s, s);
}

inline __host__ __device__ int3 make_int3(const int2 a) {
    return make_int3(a.x, a.y, 0);
}

inline __host__ __device__ int3 make_int3(const int2 a, const int s) {
    return make_int3(a.x, a.y, s);
}

inline __host__ __device__ int3 make_int3(const uint3 a) {
    return make_int3(static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z));
}

inline __host__ __device__ int3 make_int3(const float3 a) {
    return make_int3(static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z));
}

inline __host__ __device__ uint3 make_uint3(const uint s) {
    return make_uint3(s, s, s);
}

inline __host__ __device__ uint3 make_uint3(const uint2 a) {
    return make_uint3(a.x, a.y, 0);
}

inline __host__ __device__ uint3 make_uint3(const uint2 a, const uint s) {
    return make_uint3(a.x, a.y, s);
}

inline __host__ __device__ uint3 make_uint3(const uint4 a) {
    return make_uint3(a.x, a.y, a.z);
}

inline __host__ __device__ uint3 make_uint3(const int3 a) {
    return make_uint3(static_cast<uint>(a.x), static_cast<uint>(a.y), static_cast<uint>(a.z));
}

inline __host__ __device__ int4 make_int4(const int s) {
    return make_int4(s, s, s, s);
}

inline __host__ __device__ int4 make_int4(const int3 a) {
    return make_int4(a.x, a.y, a.z, 0);
}

inline __host__ __device__ int4 make_int4(const int3 a, const int w) {
    return make_int4(a.x, a.y, a.z, w);
}

inline __host__ __device__ int4 make_int4(const uint4 a) {
    return make_int4(static_cast<int>(a.x), static_cast<int>(a.y),
                     static_cast<int>(a.z), static_cast<int>(a.w));
}

inline __host__ __device__ int4 make_int4(const float4 a) {
    return make_int4(static_cast<int>(a.x), static_cast<int>(a.y),
                     static_cast<int>(a.z), static_cast<int>(a.w));
}

inline __host__ __device__ uint4 make_uint4(const uint s) {
    return make_uint4(s, s, s, s);
}

inline __host__ __device__ uint4 make_uint4(const uint3 a) {
    return make_uint4(a.x, a.y, a.z, 0);
}

inline __host__ __device__ uint4 make_uint4(const uint3 a, const uint w) {
    return make_uint4(a.x, a.y, a.z, w);
}

inline __host__ __device__ uint4 make_uint4(const int4 a) {
    return make_uint4(static_cast<uint>(a.x), static_cast<uint>(a.y),
                      static_cast<uint>(a.z), static_cast<uint>(a.w));
}

inline __host__ __device__ float2 make_float2(const float s) {
    return make_float2(s, s);
}

inline __host__ __device__ float2 make_float2(const float3 a) {
    return make_float2(a.x, a.y);
}

inline __host__ __device__ float2 make_float2(const int2 a) {
    return make_float2(static_cast<float>(a.x), static_cast<float>(a.y));
}

inline __host__ __device__ float2 make_float2(const uint2 a) {
    return make_float2(static_cast<float>(a.x), static_cast<float>(a.y));
}


inline __host__ __device__ float3 make_float3(const float s) {
    return make_float3(s, s, s);
}

inline __host__ __device__ float3 make_float3(const float2 a) {
    return make_float3(a.x, a.y, 0.0f);
}

inline __host__ __device__ float3 make_float3(const float2 a, const float s) {
    return make_float3(a.x, a.y, s);
}

inline __host__ __device__ float3 make_float3(const float4 a) {
    return make_float3(a.x, a.y, a.z);
}

inline __host__ __device__ float3 make_float3(const int3 a) {
    return make_float3(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z));
}

inline __host__ __device__ float3 make_float3(const uint3 a) {
    return make_float3(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z));
}

inline __host__ __device__ float4 make_float4(const float s) {
    return make_float4(s, s, s, s);
}

inline __host__ __device__ float4 make_float4(const float3 a) {
    return make_float4(a.x, a.y, a.z, 0.0f);
}

inline __host__ __device__ float4 make_float4(const float3 a, const float w) {
    return make_float4(a.x, a.y, a.z, w);
}

inline __host__ __device__ float4 make_float4(const int4 a) {
    return make_float4(static_cast<float>(a.x), static_cast<float>(a.y),
                       static_cast<float>(a.z), static_cast<float>(a.w));
}

inline __host__ __device__ float4 make_float4(const uint4 a) {
    return make_float4(static_cast<float>(a.x), static_cast<float>(a.y),
                       static_cast<float>(a.z), static_cast<float>(a.w));
}


#endif//HELPER_MATH_H
