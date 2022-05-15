#include "ray_tracer.hpp"
#include "cuda_err_check.hpp"
#include <sm_60_atomic_functions.h>

constexpr int FLOAT_2048 = 0xc5000000;      // when convert to float directly, this is equal to 2048.0
constexpr float _M_2PI = 2.f * M_PIf32;

// 4 * 2048 (for mesh segments) and 6 for obs(x, y, theta) and lidar angle (min, max, inc), 1 (interpreted as int) for lidar ray number
__constant__ float const_mem[8200];       
float* const pose_ptr = &const_mem[8192];
float* const lidar_ptr = &const_mem[8195];

__device__ __forceinline__ int floatToOrderedInt( const float floatVal ) {
    const int intVal = __float_as_int( floatVal );
    return (intVal >= 0 ) ? intVal ^ 0x80000000 : intVal ^ 0xFFFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat( const int intVal ) {
    return __int_as_float( (intVal >= 0) ? intVal ^ 0xFFFFFFFF : intVal ^ 0x80000000);
}

float goodAngle(float angle) {
    if (angle > M_PIf32) 
        return angle - _M_2PI;
    else if (angle < -M_PIf32)
        return angle + _M_2PI;
    return angle;
}

/// block与thread的设置？由于每次是对面片进行处理，故仍然是分组处理
/// 面片数量 >> 7 为block数量，每个block至多处理128个面片（在少面片时效率不一定特别高）
/// sids, eids, angles, dists的大小为总面片数量
__global__ void preProcess(
    short* const sids, short* const eids, float* const angles, float* const dists,
    bool* const flags, short actual_num, short start_id, short end_id, short num_segs
) {
	const short offset = blockIdx.x << 7, seg_offset = offset << 2;
    const short tid = threadIdx.x, seg_base = seg_offset + tid;
    const Point sp(const_mem[seg_base], const_mem[seg_base + 1]), ep(const_mem[seg_base + 2], const_mem[seg_base + 3]), obs(pose_ptr[0], pose_ptr[1]);
    const Point sp_dir = sp - obs, ep_dir = ep - obs;
    const float sp_dir_angle = goodAngle(sp_dir.get_angle()), 
        ep_dir_angle = goodAngle(ep_dir.get_angle()), 
        normal_angle = goodAngle((ep - sp).get_angle() + M_PI_2f32);
    const float distance = sp_dir.norm() * cosf(normal_angle - sp_dir_angle);
    dists[seg_base] = distance;
    flags[seg_base] = (distance < 0);
    angles[seg_base] = normal_angle;
    const float cur_amin = goodAngle(pose_ptr[2] + lidar_ptr[0]), cur_amax = goodAngle(pose_ptr[2] + lidar_ptr[1]), ainc = lidar_ptr[2]; 
    const float sp_dangle = (sp_dir_angle - cur_amin), ep_dangle = (ep_dir_angle - cur_amin);
    sids[seg_base] = static_cast<short>(ceilf((sp_dangle + _M_2PI * (sp_dangle < 0)) / ainc));
    eids[seg_base] = static_cast<short>(floor((ep_dangle + _M_2PI * (ep_dangle < 0)) / ainc));
    __syncthreads();
}
