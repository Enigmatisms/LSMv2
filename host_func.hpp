#pragma once
#include <iostream>
#include "ray_tracer.hpp"

typedef std::vector<Eigen::Vector2d> Mesh;
typedef std::vector<Mesh> Meshes;

__host__ void rayTraceRenderCpp(const Meshes& meshes, const Eigen::Vector3d& lidar_param, const Eigen::Vector3d& pose, std::vector<float>& range);

// 暂时不知道Rust需要什么样的接口才能调用
__host__ void rayTraceRenderRust();
