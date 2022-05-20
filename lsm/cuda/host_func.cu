#include "ray_tracer.hpp"

typedef Vec2* MeshPtr;
typedef const Vec2* const MeshConstPtr;

short *sid_ptr = nullptr, *eid_ptr = nullptr;
float *all_segments = nullptr, *angles_ptr = nullptr, *dists_ptr = nullptr, *final_dense_ranges, *final_sparse_ranges, *oct_ranges;
bool *flag_ptr = nullptr;
int total_seg_num = 0;

// export
extern  "C" {
void intializeFixed(int num_ray) {
    printf("Memory allocated, %d\n", num_ray);
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &all_segments, 8192 * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &final_dense_ranges, num_ray * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &final_sparse_ranges, num_ray / 3 * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &oct_ranges, (num_ray << 3) * sizeof(float)));
}

// export
void deallocateFixed() {
    printf("Memory deallocated.\n");
    CUDA_CHECK_RETURN(cudaFreeHost(all_segments));
    CUDA_CHECK_RETURN(cudaFree(final_dense_ranges));
    CUDA_CHECK_RETURN(cudaFree(final_sparse_ranges));
    CUDA_CHECK_RETURN(cudaFree(oct_ranges));
}

// export
void deallocateDevice() {
    CUDA_CHECK_RETURN(cudaFree(sid_ptr));
    CUDA_CHECK_RETURN(cudaFree(eid_ptr));
    CUDA_CHECK_RETURN(cudaFree(angles_ptr));
    CUDA_CHECK_RETURN(cudaFree(dists_ptr));
    CUDA_CHECK_RETURN(cudaFree(flag_ptr));
}

// export
void unwrapMeshes(MeshConstPtr meshes, int seg_num, bool initialized) {
    size_t mesh_point_cnt = 0;
    for (int i = 0; i < seg_num; i++) {
        int seg_point_base = i << 1;
        const Vec2& start = meshes[seg_point_base], end = meshes[seg_point_base + 1];
        all_segments[mesh_point_cnt++] = start.x;
        all_segments[mesh_point_cnt++] = start.y;
        all_segments[mesh_point_cnt++] = end.x;
        all_segments[mesh_point_cnt++] = end.y;
    }
    updateSegments(all_segments, mesh_point_cnt << 2);
    if (initialized == true)
        deallocateDevice();

    CUDA_CHECK_RETURN(cudaMalloc((void **) &sid_ptr, seg_num * sizeof(short)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &eid_ptr, seg_num * sizeof(short)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &angles_ptr, seg_num * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dists_ptr, seg_num * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &flag_ptr, seg_num * sizeof(bool)));
    total_seg_num = seg_num;
}

// export
void rayTraceRender(const Vec3& lidar_param, const Vec3& pose, int ray_num, float* range) {
    // 对于静态地图而言，由于场景无需频繁update，unwrapMeshes函数调用频率低，则可以省略内存allocation操作
    const int lidar_ray_blocks = ray_num / DEPTH_DIV_NUM;
    const short num_blocks = static_cast<short>(ceilf(total_seg_num / 256.f));          // 面片数 / 128
    preProcess<<<num_blocks, 256>>>(sid_ptr, eid_ptr, angles_ptr, dists_ptr, flag_ptr, ray_num, 
                total_seg_num, lidar_param.x, lidar_param.z, pose.x, pose.y, pose.z);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cudaStream_t streams[8];
    for (short i = 0; i < 8; i++)
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
    const short segment_per_block = static_cast<short>(ceil(0.125f * total_seg_num)),
                last_block_seg_num = short(total_seg_num) - 7 * segment_per_block;
    const size_t size_t_sblock = static_cast<size_t>(segment_per_block);
    const size_t shared_mem_size = (size_t_sblock * 13) + (DEPTH_DIV_NUM << 2) + 4 - size_t_sblock % 4;     // 13 = 8 + 4 + 1 = (4B angles 4B dists, 2B * 2 ids, 1B flags)
    for (int i = 0; i < 8; i++) {
        // 最后由于bool是单字节的类型，需要padding到4的整数倍字节数
        // local segments大小应该是 4B * (angles + dists) / 8 + DEPTH_DIV_NUM * 4B (深度图分区) + 2B * (sids + eids) / 8 + (1B * len(flags) / 8) + padding
        rayTraceKernel<<<lidar_ray_blocks, DEPTH_DIV_NUM, shared_mem_size, streams[i]>>>(
            sid_ptr, eid_ptr, angles_ptr, dists_ptr, flag_ptr, i, segment_per_block, 
            ((i < 7) ? segment_per_block : last_block_seg_num), &oct_ranges[i * ray_num], lidar_param.x, lidar_param.z, pose.z
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++)
        cudaStreamDestroy(streams[i]);
    getMininumRangeKernel<<<lidar_ray_blocks, DEPTH_DIV_NUM>>>(oct_ranges, final_dense_ranges, ray_num);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    sparsifyScan<<<lidar_ray_blocks, DEPTH_DIV_NUM / 3>>>(final_dense_ranges, final_sparse_ranges);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // 注意range的大小应该提前确定
    CUDA_CHECK_RETURN(cudaMemcpy(range, final_sparse_ranges, sizeof(float) * ray_num / 3, cudaMemcpyDeviceToHost));
}
}