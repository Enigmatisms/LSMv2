extern crate libc;

#[repr(C)]
pub struct Vec3_cuda {
    pub x: libc::c_float,
    pub y: libc::c_float,
    pub z: libc::c_float
}

#[repr(C)]
pub struct Vec2_cuda {
    pub x: libc::c_float,
    pub y: libc::c_float
}

#[link(name = "cuda_helper", kind = "static")]
extern {
    pub fn deallocateFixed();
    pub fn deallocateDevice();
    pub fn intializeFixed(num_ray: libc::c_int);
    pub fn unwrapMeshes(segments: *const Vec2_cuda, total_seg_num: libc::c_int, initialized: bool);
    pub fn rayTraceRender(lidar_param: &Vec3_cuda, pose: &Vec3_cuda, ray_num: libc::c_int, noise_k: libc::c_float, range: *mut libc::c_float);
}