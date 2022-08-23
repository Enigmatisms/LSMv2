use nannou::prelude::*;
use array2d::Array2D;

use super::gui;
use super::ctrl;
use super::cuda_helper;
use super::model::Model;
use super::grid::collision_detection;

use crate::utils::plot;
use crate::utils::utils;
use crate::utils::map_io;

const BOUNDARIES: [(f32, f32); 4] = [(-1e6, -1e6), (1e6, -1e6), (1e6, 1e6), (-1e6, 1e6)];
const BOUNDARY_IDS: [i8; 4] = [3, 0, 0, -3];

pub fn model(app: &App) -> Model {
    let config: map_io::Config = map_io::read_config("../config/simulator_config.json");

    let window_id = app
        .new_window()
        .event(event)
        .key_pressed(ctrl::key_pressed)
        .key_released(ctrl::key_released)
        .raw_event(raw_window_event)
        .mouse_moved(ctrl::mouse_moved)
        .mouse_pressed(ctrl::mouse_pressed)
        .mouse_released(ctrl::mouse_released)
        .mouse_wheel(ctrl::mouse_wheel)
        .size(config.screen.width, config.screen.height)
        .view(view)
        .build()
        .unwrap();

    app.set_exit_on_escape(false);
    let meshes: map_io::Meshes = map_io::parse_map_file(config.map_path.as_str()).unwrap();

    let lidar_param = cuda_helper::Vec3_cuda{x: config.lidar.amin, y: config.lidar.amax, z:config.lidar.ainc};
    let ray_num = map_io::get_ray_num(&lidar_param);

    let mut total_pt_num = 0;
    initialize_cuda_end(&meshes, ray_num, &mut total_pt_num, false);
    Model::new(app, window_id, &config, meshes, lidar_param, ray_num, total_pt_num)
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

pub fn initialize_cuda_end(new_pts: &map_io::Meshes, ray_num: usize, total_pt_num: &mut usize, initialized: bool) {
    let mut seg_point_arr: Vec<cuda_helper::Vec2_cuda> = Vec::new();
    let seg_num = map_io::meshes_to_segments(&new_pts, &mut seg_point_arr);
    let mut points: Vec<cuda_helper::Vec2_cuda> = Vec::new();
    let mut next_ids: Vec<i8> = Vec::new();
    for mesh in new_pts.iter() {
        for pt in mesh.iter() {
            points.push(cuda_helper::Vec2_cuda{x: pt.x, y: pt.y});
        }
        let length = mesh.len();
        let offset: i8 = (length as i8) - 1;
        let mut ids: Vec<i8> = vec![0; length];
        ids[0] = offset;
        ids[length - 1] = -offset;
        next_ids.extend(ids.into_iter());
    }
    for i in 0..4 {                                                 // add boundaries
        let (x, y) = BOUNDARIES[i];
        points.push(cuda_helper::Vec2_cuda{x: x, y: y});
        next_ids.push(BOUNDARY_IDS[i]);
    }
    *total_pt_num = points.len();
    let point_num = *total_pt_num as libc::c_int;
    unsafe {
        if initialized == false {
            cuda_helper::intializeFixed(ray_num as libc::c_int);
        }
        cuda_helper::unwrapMeshes(seg_point_arr.as_ptr(), seg_num as libc::c_int, initialized);
        cuda_helper::updatePointInfo(points.as_ptr(), next_ids.as_ptr(), point_num, initialized);
    }
} 

pub fn update(_app: &App, _model: &mut Model, _update: Update) {
    gui::update_gui(_app, _model, &_update);
    if let Some(result) = _model.timer_event.check_buffer() {
        _model.timer_event.item = result;
    }
    static mut LOCAL_INT: f32 = 0.0;
    static mut LOCAL_DIFF: f32 = 0.0;
    if _model.initialized == false {return;}
    let sina = _model.pose.z.sin();
    let cosa = _model.pose.z.cos();
    let tmp_x = _model.pose.x +_model.velo.x * cosa - _model.velo.y * sina; 
    let tmp_y = _model.pose.y +_model.velo.x * sina + _model.velo.y * cosa; 
    if collision_detection(
        &_model.occ_grid, &_model.map_points, tmp_x, _model.pose.y, _model.grid_size, &_model.grid_specs
    ) {
        _model.pose.x = tmp_x;
    }
    if collision_detection(
        &_model.occ_grid, &_model.map_points, _model.pose.x, tmp_y, _model.grid_size, &_model.grid_specs
    ) {
        _model.pose.y = tmp_y;
    }

    let mouse = plot::local_mouse_position(_app, &_model.wtrans);
    let dir = mouse - pt2(_model.pose.x, _model.pose.y);
    let target_angle = dir.y.atan2(dir.x);
    let diff = utils::good_angle(target_angle - _model.pose.z);
    unsafe {
        LOCAL_INT += diff;
        let kd_val = diff - LOCAL_DIFF;
        LOCAL_DIFF = diff;
        _model.pose.z += _model.pid.x * diff + _model.pid.y * LOCAL_INT + _model.pid.z * kd_val;
        _model.pose.z = utils::good_angle(_model.pose.z);
        let pose = cuda_helper::Vec3_cuda {x:_model.pose.x, y:_model.pose.y, z:_model.pose.z};
        cuda_helper::rayTraceRender(&_model.lidar_param, &pose, _model.ray_num as i32, _model.lidar_noise, _model.ranges.as_mut_ptr());
    }
    if _model.caster.radius > 100.0 {
        let mut valid_pnum = 0;
        let mut raw_pts: Vec<cuda_helper::Vec2_cuda> = vec![cuda_helper::Vec2_cuda{x: 0., y:0.}; _model.caster.total_pt_num << 1];
        let pose = cuda_helper::Vec3_cuda {x:_model.pose.x, y:_model.pose.y, z:_model.pose.z};
        unsafe {
            cuda_helper::shadowCasting(&pose, raw_pts.as_mut_ptr(), &mut valid_pnum);
        }
        _model.caster.viz_pts.clear();
        for i in 0..(valid_pnum as usize) {
            let pt = &raw_pts[i];
            _model.caster.viz_pts.push(pt2(pt.x, pt.y));
        }
    }
}

fn event(_app: &App, _model: &mut Model, _event: WindowEvent) {}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = plot::window_transform(app.draw(), &model.wtrans);

    if model.plot_config.draw_grid == true {
        let win = app.main_window().rect();
        plot::draw_grid(&draw, &win, model.plot_config.grid_step, 1.0, &model.color.grid_color, model.plot_config.grid_alpha);
        plot::draw_grid(&draw, &win, model.plot_config.grid_step / 5., 0.5, &model.color.grid_color, model.plot_config.grid_alpha);
    }
    let (bg_r, bg_g, bg_b) = model.color.bg_color;
    draw.background().rgba(bg_r, bg_g, bg_b, 1.0);
    let (r, g, b, a) = model.color.shape_color;
    for mesh in model.map_points.iter() {
        let points = (0..mesh.len()).map(|i| {
            mesh[i]
        });
        draw.polygon()
            .rgba(r, g, b, a)
            .points(points);
    }

    draw.ellipse()
        .w(15.)
        .h(15.)
        .x(model.pose.x)
        .y(model.pose.y)
        .color(STEELBLUE);
    
    if model.caster.radius > 100.0 {
        let viz_pts = &model.caster.viz_pts;
        let shadow_pts = viz_pts.iter().map(|point| {
            let tex_coords = [(point.x - model.pose.x) / model.caster.radius + 0.5, 0.5 + (model.pose.y - point.y) / model.caster.radius];
            (*point, tex_coords)
        });
        draw.polygon()
            .points_textured(&model.caster.texture, shadow_pts);
    }

    if model.initialized == true {
        visualize_rays(&draw, &model.ranges, &model.pose, &model.lidar_param, &model.color.lidar_color, &model.ray_num / 3);
        // draw_occ_grids(&draw, &model.occ_grid, model.grid_specs.0, model.grid_specs.1, model.grid_size);
    }
        
    let start_pos = pt2(model.pose.x, model.pose.y);
    let dir = plot::local_mouse_position(app, &model.wtrans) - start_pos;
    let norm = (dir.x * dir.x + dir.y * dir.y + 1e-5).sqrt();
    draw.arrow()
        .start(start_pos)
        .end(start_pos + dir * 40. / norm)
        .weight(2.)
        .color(MEDIUMSPRINGGREEN);

    // Write the result of our drawing to the window's frame.
    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}

fn visualize_rays(
    draw: &Draw, ranges: &Vec<libc::c_float>, pose: &Point3, 
    lidar_param: &cuda_helper::Vec3_cuda, color: &[f32; 4], ray_num: usize) {
    let cur_angle_min = pose.z + lidar_param.x + lidar_param.z;
    for i in 0..ray_num {
        let r = ranges[i];
        // if r > 1e5 {continue;}
        let cur_angle = cur_angle_min + lidar_param.z * 3. * (i as f32);
        let dir = pt2( cur_angle.cos(), cur_angle.sin());
        let start_p = pt2(pose.x, pose.y);
        let end_p = start_p + dir * r;
        draw.line()
            .start(start_p)
            .end(end_p)
            .weight(1.)
            .rgba(color[0], color[1], color[2], color[3]);
    }
}

// debug occupancy grid visualization
#[allow(dead_code)]
fn draw_occ_grids(draw: &Draw, occ_grid: &Array2D<i32>, off_x: f32, off_y: f32, grid_size: f32) {
    let map_rows = occ_grid.column_len();
    let map_cols = occ_grid.row_len();
    for i in 0..map_rows {
        for j in 0..map_cols {
            let color = match occ_grid[(i, j)] {
                -2 => {(1., 0., 0.)},
                -1 => {(0., 0., 0.)},
                _ => {(0., 0., 1.)}
            };
            draw.rect()
                .rgba(color.0, color.1, color.2, 0.2)
                .w_h(grid_size, grid_size)
                .x_y(off_x + grid_size * (j as f32 + 0.5), off_y + grid_size * (i as f32 + 0.5));
        }
    } 
}
