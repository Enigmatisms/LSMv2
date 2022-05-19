#pragma once
#include <iostream>
#include "ray_tracer.hpp"


struct Vec2 {
    float x;
    float y;
};

struct Vec3 {
    float x;
    float y;
    float z;
};

typedef Vec2* MeshPtr;
typedef const Vec2* const MeshConstPtr;

__host__ void intializeFixed(int num_ray);
__host__ void deallocateDevice();
__host__ void deallocateFixed();

__host__ void unwrapMeshes(MeshConstPtr meshes, bool initialized = false);

__host__ void rayTraceRenderCpp(Vec3 lidar_param, Vec2 pose, float angle, int ray_num, float* range);
