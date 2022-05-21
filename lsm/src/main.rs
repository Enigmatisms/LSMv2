mod cuda_helper;
mod map_io;
mod viz;
mod ctrl;
mod utils;

// pub fn deallocateFixed();
// pub fn deallocateDevice();
// pub fn intializeFixed(num_ray: libc::c_int);
// pub fn unwrapMeshes(segments: *const Vec2_cuda, total_seg_num: libc::c_int, initialized: bool);
// pub fn rayTraceRender(lidar_param: &Vec3_cuda, pose: &Vec2_cuda, angle: libc::c_float, ray_num: libc::c_int, range: *mut libc::c_float);

fn main() {
    unsafe {
        cuda_helper::intializeFixed(2880);
    }
    nannou::app(viz::model).update(viz::update).run();
}
