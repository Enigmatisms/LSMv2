#include "ray_tracer.hpp"
#include "cuda_err_check.hpp"
#include <sm_60_atomic_functions.h>

constexpr int FLOAT_2048 = 0xc5000000;      // when convert to float directly, this is equal to 2048.0
constexpr float _M_PI = M_PI;
constexpr float _M_2PI = 2 * _M_PI;

// 4 * 2048 (for mesh segments) and 6 for obs(x, y, theta) and lidar angle (min, max, inc), 1 (interpreted as int) for lidar ray number
__constant__ float const_mem[8200];       

__device__ __forceinline__ int floatToOrderedInt( const float floatVal ) {
    const int intVal = __float_as_int( floatVal );
    return (intVal >= 0 ) ? intVal ^ 0x80000000 : intVal ^ 0xFFFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat( const int intVal ) {
    return __int_as_float( (intVal >= 0) ? intVal ^ 0xFFFFFFFF : intVal ^ 0x80000000);
}