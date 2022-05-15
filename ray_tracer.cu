#include "ray_tracer.hpp"
#include "cuda_err_check.hpp"
#include <sm_60_atomic_functions.h>
#define DEPTH_DIV_NUM 360

constexpr int FLOAT_2048 = 0xc5000000;      // when convert to float directly, this is equal to 2048.0
constexpr float _M_2PI = 2.f * M_PIf32;

// 4 * 2048 (for mesh segments) and 6 for obs(x, y, theta) and lidar angle (min, max, inc), 1 (interpreted as int) for lidar ray number
__constant__ float const_mem[8192];       

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
    bool* const flags, short ray_num, short start_id, short end_id, short num_segs, 
    const Vec3& lidar_param, const Vec3& pose
) {
    // no need to use shared memory
	const short offset = blockIdx.x << 7, seg_offset = offset << 2;
    const short tid = threadIdx.x, seg_base = seg_offset + tid;
    const Point sp(const_mem[seg_base], const_mem[seg_base + 1]), ep(const_mem[seg_base + 2], const_mem[seg_base + 3]), obs(pose.x, pose.y);
    const Point sp_dir = sp - obs, ep_dir = ep - obs;
    const float sp_dir_angle = goodAngle(sp_dir.get_angle()), 
        ep_dir_angle = goodAngle(ep_dir.get_angle()), 
        normal_angle = goodAngle((ep - sp).get_angle() + M_PI_2f32);
    const float distance = sp_dir.norm() * cosf(normal_angle - sp_dir_angle);
    dists[seg_base] = distance;
    angles[seg_base] = normal_angle;
    const float cur_amin = goodAngle(pose.z + lidar_param.x), ainc = lidar_param.z; 
    const float sp_dangle = (sp_dir_angle - cur_amin), ep_dangle = (ep_dir_angle - cur_amin);
    const short tmp_sid = static_cast<short>(ceilf((sp_dangle + _M_2PI * (sp_dangle < 0)) / ainc)),
        tmp_eid = static_cast<short>(floor((ep_dangle + _M_2PI * (ep_dangle < 0)) / ainc));
    flags[seg_base] = (distance < 0) && ((tmp_eid < ray_num) || (tmp_sid < ray_num));               // back culling + at least either sp or ep is in range
    sids[seg_base] = tmp_sid;
    eids[seg_base] = max(tmp_eid, ray_num - 1);
    // 此处需要考虑与起始点终止点的关系
    // 由于默认激光雷达角度范围至少大于180度，故不会出现start end id均不在范围的情况
    // TODO: ep < sp 是可能的，但是需要特殊处理（sp超出范围，ep在范围内也按照ep < sp处理）
    __syncthreads();
}

/**
 * 预处理结束之后，输出: sids, eids, angles, dists, flags
 * 此后分block处理，每个block将与自己相关的信息保存到shared 线程数量为？线程数量是一个比较大的问题，一个block的线程数量不能超过1024，用满不是很好，256左右就已经够了吧
 * 个人不使用分区标号（比较繁琐，warp divergence无法避免），直接遍历所有的valid面片
 * 深度图分区渲染
 */
// 可以尝试采用双分区方法 --- 每个grid的x方向分为8份（面片分8份），y则是角度分区，则最后将输出到8个单线深度图中，最后将8个单线深度图组合在一起
__global__ void rayTraceKernel(
    short* const sids, short* const eids, float* const angles, float* const dists, bool* const flags,
    short num_segs, short block_seg_num, float* const ranges, const Vec3& lidar_param, const Vec3& pose
) {
    extern __shared__ float local_segements[]; 
    // local segements的大小是 (4B * 4 * len(angles)) / 8 + DEPTH_DIV_NUM * 4B (深度图分区) + (1B * len(flags) / 8) + padding
    const short seg_sid = blockIdx.x * num_segs, seg_eid = seg_sid + block_seg_num - 1;
    const short range_base = blockIdx.y * DEPTH_DIV_NUM, rimg_id = range_base + threadIdx.x;
    int* const range_ptr = (int*)&local_segements[num_segs << 2];           // 此部分是输出
    bool* const flag_ptr = (bool*)&local_segements[num_segs << 2 + DEPTH_DIV_NUM];   
    // 可能有严重的warp divergence
    for (short i = 0; i < 16; i++) {
        const short local_i = threadIdx.x + i * DEPTH_DIV_NUM, global_i = seg_sid + local_i;
        if (global_i > seg_eid) break;     // warp divergence
        flag_ptr[local_i] = flags[global_i];
        if (flag_ptr[local_i] ==  false) continue;
        local_segements[local_i] = sids[global_i]; 
        local_segements[local_i + 1] = eids[global_i]; 
        local_segements[local_i + 2] = angles[global_i]; 
        local_segements[local_i + 3] = dists[global_i]; 
    }
    range_ptr[threadIdx.x] = FLOAT_2048;
    __syncthreads();
    for (short seg_id = seg_sid, i = 0; i < block_seg_num; i++) {         // traverse all segs
        seg_id = seg_sid + i;
        // 当前线程的角度id是rimg_id，如果此id小于start_id 或是大于 end_id 则都跳过
        if (flag_ptr[seg_id] == false || rimg_id < local_segements[seg_id] || rimg_id > local_segements[seg_id + 1]) continue;
        float local_range = cosf(local_segements[seg_id + 2] - (lidar_param.x + pose.z + lidar_param.z * rimg_id)) * local_segements[seg_id + 3];
        const int range_int = floatToOrderedInt(local_range);
        atomicMin(range_ptr + threadIdx.x, range_int);
    }
    // 处理结束，将shared memory复制到global memory
    __syncthreads();
    ranges[rimg_id] = orderedIntToFloat(range_ptr[threadIdx.x]);
}

// 将8个单线深度图合并为一个
__global__ void getMininumRangeKernel(const float* const oct_ranges, float* const output, int range_num) {
    const int range_base = DEPTH_DIV_NUM * blockIdx.x + threadIdx.x;
    float min_range = 1e9;
    for (int i = 0; i < 8; i++) {
        min_range = min(min_range, oct_ranges[range_base + i * range_num]);
    }
    output[range_base] = min_range;
}
