extern crate libc;

#[repr(C)]
pub struct Vec3 {
    x: libc::c_float,
    y: libc::c_float,
    z: libc::c_float
}

#[repr(C)]
pub struct Vec2 {
    x: libc::c_float,
    y: libc::c_float
}

#[link(name = "cuda_helper", kind = "static")]
extern {
    pub fn deallocateFixed();
    pub fn deallocateDevice();
    pub fn intializeFixed(num_ray: libc::c_int);
    pub fn unwrapMeshes(segments: *const Vec2, total_seg_num: libc::c_int, initialized: bool);
    pub fn rayTraceRender(lidar_param: &Vec3, pose: &Vec2, angle: libc::c_float, ray_num: libc::c_int, range: *mut libc::c_float);
}

// __host__ void intializeFixed(int num_ray);
// __host__ void deallocateDevice();
// __host__ void deallocateFixed();

// __host__ void unwrapMeshes(MeshConstPtr meshes, int total_seg_num, bool initialized = false);

// __host__ void rayTraceRenderCpp(Vec3 lidar_param, Vec2 pose, float angle, int ray_num, float* range);

fn main() {
    println!("Hello, world!");
    unsafe{
        intializeFixed(2880);
        println!("Test whether functions are called.");
        deallocateFixed();
    }
}
