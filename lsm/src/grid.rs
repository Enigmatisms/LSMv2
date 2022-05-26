/**
 * Collision detection for scanner
 */

use array2d::Array2D;
use nannou::prelude::*;

const MAGIC_MAX: usize = 2147483647;
static SUR_VECS: [(f32, f32); 4] = [(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)];
static SUR_OFFS: [(i32, i32); 4] = [(0, 0), (0, -1), (-1, -1), (-1, 0)];

 // 根据点链返回grid边界，返回（grid起始坐标（左下角）以及grid数量上限（根据实际范围，左右各扩展两个grid））
pub fn get_bounds(map_points: &Vec<Vec<Point2>>, grid_size: f32) -> (f32, f32, f32, f32) {
    let mut offset_x: f32 = 1e5;
    let mut offset_y: f32 = 1e5;
    let mut x_blocks: f32 = 1e-5;
    let mut y_blocks: f32 = 1e-5;
    for mesh in map_points.iter() {
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
    x_blocks = ((x_blocks - offset_x) / grid_size).ceil() + 4.;
    y_blocks = ((y_blocks - offset_y) / grid_size).ceil() + 4.;
    offset_x -= 2. * grid_size;
    offset_y -= 2. * grid_size;
    (offset_x, offset_y, x_blocks, y_blocks)
} 

// 划线算法，需要在model中新增：Array2D（格点占用情况），对于交点恰好在grid角点上的情况，直接处理，不用存储
pub fn line_drawing(occ_grid: &mut Array2D<i32>, meshes: &Vec<Vec<Point2>>, off_x: f32, off_y: f32, grid_size: f32) {
    for (polygon_id, mesh) in meshes.iter().enumerate() {
        let mesh_len = mesh.len();
        for i in 0..mesh_len {
            let sp = &mesh[i];
            let ep = &mesh[(i + 1) % mesh_len];             // 返回初始点
            let dir = *ep - *sp;
            let normal = (pt2(-dir.y, dir.x)).normalize();
            let x_main = dir.x.abs() > dir.y.abs();
            if x_main {
                let start_m = (sp.x - off_x) / grid_size;
                let end_m = (ep.x - off_x) / grid_size;
                let k = dir.y / dir.x;
                let offset_s = (sp.y - off_y) / grid_size;
                single_line(occ_grid, &normal, start_m, end_m, k, offset_s, polygon_id as i32, true);
            } else {
                let start_m = (sp.y - off_y) / grid_size;
                let end_m = (ep.y - off_y) / grid_size;
                let k = dir.x / dir.y;
                let offset_s = (sp.x - off_x) / grid_size;
                single_line(occ_grid, &normal, start_m, end_m, k, offset_s, polygon_id as i32, false);
            }
        }
    }
}

// ==================== private functions =========================
// @param dir_m 主轴递增？递减？
// @param start_m 主轴的起始位置, end_m 主轴结束位置
// @param offset_s 为副轴的起始位置
fn single_line(occ_grid: &mut Array2D<i32>, normal: &Point2, start_m: f32, end_m: f32, k: f32, offset_s: f32, polygon_id: i32, x_main: bool) {
    let s_m = start_m.ceil() as usize;
    let e_m = end_m.ceil() as usize;
    fill_start(occ_grid, start_m, offset_s, x_main, polygon_id);            // in case that the start is not filled
    if s_m == e_m {
        return;
    }
    let range = match s_m < e_m {
        true => {s_m..e_m},
        false => {e_m..s_m}
    };
    let mut last_v_s: usize = MAGIC_MAX;
    for v_m in range {
        let val_s: f32 = ((v_m as f32) - start_m) * k + offset_s;
        let vs_f32: f32 = val_s.round();
        let vs = val_s.floor() as usize;
        if (val_s - vs_f32).abs() < 1e-5 {          // 过栅格交点
            fill_surrounding(occ_grid, normal, v_m, vs_f32 as usize, polygon_id, x_main);
        } else {
            let now_fill = get_row_col_id(v_m, vs, x_main);
            let extra_fill = get_row_col_id(v_m - 1, vs, x_main);
            let now_val = &mut occ_grid[(now_fill.0, now_fill.1)];
            fill_grid(now_val, polygon_id);
            if vs != last_v_s {         // 根据dir方向填充两个grid
                let extra_val = &mut occ_grid[(extra_fill.0, extra_fill.1)];
                fill_grid(extra_val, polygon_id);
            }
        }
        last_v_s = vs;
    }
}

pub fn collision_detection(occ_grid: &Array2D<i32>, meshes: &Vec<Vec<Point2>>, x: f32, y: f32, grid_size: f32, offset: &Point2) -> bool {
    let grid_x = ((x - offset.x) / grid_size) as usize;
    let grid_y = ((y - offset.y) / grid_size) as usize;
    let mesh_id = occ_grid[(grid_y, grid_x)];
    if mesh_id != -1 {
        if mesh_id == -2 {return false;}
        let mesh = &meshes[mesh_id as usize];
        let mesh_len = mesh.len();
        for i in 0..mesh_len {
            let sp = &mesh[i];
            let ep = &mesh[(i + 1) % mesh_len];             // 返回初始点
        }
    }
    return false;
}

#[inline(always)]
fn get_row_col_id(main_id: usize, sec_id: usize, x_main: bool) -> (usize, usize) {
    if x_main {
        return (sec_id, main_id);
    } else {
        return (main_id, sec_id);
    }
}

#[inline(always)]
fn fill_grid(to_fill: &mut i32, polygon_id: i32) {
    if *to_fill == -1 {
        *to_fill = polygon_id;
    } else if *to_fill != polygon_id {
        *to_fill = -2;
    }
}

#[inline(always)]
fn fill_start(occ_grid: &mut Array2D<i32>, start_m: f32, start_s: f32, x_main: bool, polygon_id: i32) {
    let now_fill = get_row_col_id(start_m.floor() as usize, start_s.floor() as usize, x_main);
    let start_grid = &mut occ_grid[(now_fill.0, now_fill.1)];
    fill_grid(start_grid, polygon_id);
}

// normal is normalized
fn fill_surrounding(occ_grid: &mut Array2D<i32>, normal: &Point2, main_id: usize, sec_id: usize, polygon_id: i32, x_main: bool) {
    let row_col_id = get_row_col_id(main_id, sec_id, x_main); 
    for i in 0..4 {
        let vec = &SUR_VECS[i];
        let offset = &SUR_OFFS[i];
        let center_vec = pt2(vec.0, vec.1);
        let dot_prod = center_vec.dot(*normal);
        if dot_prod > -0.4999 {
            let now_fill = &mut occ_grid[(row_col_id.0 + offset.0 as usize, row_col_id.1 + offset.1 as usize)];
            fill_grid(now_fill, polygon_id);
        }
    }
}