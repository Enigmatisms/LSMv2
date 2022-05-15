#include <numeric>
#include "host_func.hpp"

float* all_segments = nullptr;

__host__ __forceinline__ void intializePinned() {
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &all_segments, 8192 * sizeof(float)));
}

__host__ __forceinline__ void deallocatePinned() {
    CUDA_CHECK_RETURN(cudaFreeHost(all_segments));
}

__host__ void rayTraceRenderCpp(const Meshes& meshes, const Eigen::Vector3d& lidar_param, const Eigen::Vector3d& pose, std::vector<float>& range) {
    // 首先将meshs转为面片
    size_t total_seg_num = std::accumulate(meshes.begin(), meshes.end(), 0, [&](size_t sm, const Mesh& m) {return sm + m.size() - 1;});
    
}
