/**
 * Collision detection for scanner
 */

 // 基于划线算法的障碍物碰撞检测
 use array2d::Array2D;
 use nannou::prelude::*;
 use crate::model::Model;

 // 根据点链返回grid边界，返回（grid起始坐标（左下角）以及grid数量上限（根据实际范围，左右各扩展两个grid））
pub fn get_bounds(_model: &Model) -> (f32, f32, f32, f32) {
    let mut offset_x: f32 = 1e5;
    let mut offset_y: f32 = 1e5;
    let mut x_blocks: f32 = 1e-5;
    let mut y_blocks: f32 = 1e-5;
    for mesh in _model.map_points.iter() {
        for point in mesh.iter() {
            if point.x < offset_x {
                offset_x = point.x;
            }
            if point.x > x_blocks {
                x_blocks = point.x;
            }
            if point.y < offset_y {
                offset_y = point.y;
            }
            if point.y > y_blocks {
                y_blocks = point.y;
            }
        }
    }
    x_blocks = ((x_blocks - offset_x) / _model.grid_size).ceil() + 4.;
    y_blocks = ((y_blocks - offset_y) / _model.grid_size).ceil() + 4.;
    offset_x -= 2. * _model.grid_size;
    offset_y -= 2. * _model.grid_size;
    (offset_x, offset_y, x_blocks, y_blocks)
} 

// 划线算法，需要在model中新增：Array2D（格点占用情况），对于交点恰好在grid角点上的情况，直接处理，不用存储
pub fn line_drawing(occ_grid: &mut Array2D<i32>, meshes: &Vec<Vec<Point2>>, off_x: f32, off_y: f32) {
    for mesh in meshes.iter() {
        let mesh_len = mesh.len();
        for i in 0..mesh_len {
            let sp = &mesh[i];
            let ep = &mesh[(i + 1) % mesh_len];             // 返回初始点
            let dir = *ep - *sp;
            // 判定角度方向（主y还是主x）
            // 求解初始交点
            // 为了简洁起见，这里实现一个x，y方向统一的函数，根据传入的值不同处理不同的方向
        }
    }
}