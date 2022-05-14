#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "cuda_err_check.hpp"

inline float goodAngle(float angle);

__host__ __device__ struct Point {
    float x, y;
    Point() : x(0), y(0) {}
    Point(float x, float y) : x(x), y(y) {}

    Point operator-(const Point& p) const {
        return Point(x - p.x, y - p.y);
    }

    float get_angle() const {
        return atan2f(y, x);
    }
};

typedef std::vector<Point> Points;

// 如果地图是动态的，则需要根据此函数进行update（覆盖原来的constant mem）
__host__ void updateMap(const float* const host_segs, size_t byte_num);

__host__ void rayTrace(const Points& all_segs, const Eigen::Vector3f& now_pos, const Eigen::Vector3f& lidar_param, std::vector<float>& ranges);

// 预处理模块，进行back culling以及frustum culling
// 由于所有面片, 激光雷达参数等都在constant memory中 故要传入的input不多
// 输出：每个segment四个值（start_id, end_id, distance, normal_angle）以及此segment是否valid（flags）
// 之后的rayTraceKernel使用global memory中的segment四参数进行运算 GPU不允许直接的global memory to constant memory
// TODO: 使用constant memory先写出初版，之后再考虑用texture memory替换
__global__ void preProcess(short start_id, short end_id, short num_segs, bool* flags);

__global__ void rayTraceKernel(int num_segs, float* ranges);