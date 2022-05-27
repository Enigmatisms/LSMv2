use nannou::prelude::*;
use array2d::Array2D;
use super::cuda_helper;
use super::map_io;
use super::ctrl;
use super::utils;
use super::model::Model;
use super::grid::collision_detection;


fn local_mouse_position(_app: &App, _model: & Model) -> Point2 {
    let mut mouse = _app.mouse.position();
    mouse -= _model.wtrans.t;
    let rotation_inv= utils::get_rotation(&-_model.wtrans.rot);
    mouse = rotation_inv.mul_vec2(mouse);
    mouse / _model.wtrans.scale
}

pub fn model(app: &App) -> Model {
    let config: map_io::Config = map_io::read_config("../config/config.json");

    let window_id = app
        .new_window()
        .event(event)
        .key_pressed(ctrl::key_pressed)
        .key_released(ctrl::key_released)
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
    let mut seg_point_arr: Vec<cuda_helper::Vec2_cuda> = Vec::new();
    let seg_num = map_io::meshes_to_segments(&meshes, &mut seg_point_arr);
    unsafe {
        cuda_helper::intializeFixed(ray_num as libc::c_int);
        cuda_helper::unwrapMeshes(seg_point_arr.as_ptr(), seg_num as libc::c_int, false);
    }

    Model::new(window_id, &config, meshes, lidar_param, ray_num)
}

pub fn update(_app: &App, _model: &mut Model, _: Update) {
    static mut LOCAL_INT: f32 = 0.0;
    static mut LOCAL_DIFF: f32 = 0.0;
    if _model.initialized == false {return;}
    let sina = _model.pose.z.sin();
    let cosa = _model.pose.z.cos();
    let tmp_x = _model.pose.x +_model.velo.x * cosa - _model.velo.y * sina; 
    let tmp_y = _model.pose.y +_model.velo.x * sina + _model.velo.y * cosa; 
    let half_width = _model.wctrl.win_w / 2.;
    let half_height = _model.wctrl.win_h / 2.;
    if tmp_x > -half_width && tmp_x < half_width {
        if collision_detection(
            &_model.occ_grid, &_model.map_points, tmp_x, _model.pose.y, 
            _model.grid_size, &pt2(_model.grid_specs.0, _model.grid_specs.1)
        ) {
            _model.pose.x = tmp_x;
        }
    }
    if tmp_y > -half_height && tmp_y < half_height {
        if collision_detection(
            &_model.occ_grid, &_model.map_points, _model.pose.x, tmp_y, 
            _model.grid_size, &pt2(_model.grid_specs.0, _model.grid_specs.1)
        ) {
            _model.pose.y = tmp_y;
        }
    }

    let mouse = local_mouse_position(_app, _model);
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
}

pub fn event(_app: &App, _model: &mut Model, event: WindowEvent) {}

pub fn view(app: &App, model: &Model, frame: Frame) {
    let mut draw = app.draw();

    draw = draw
        .x_y(model.wtrans.t.x, model.wtrans.t.y)
        .rotate(model.wtrans.rot)
        .scale_x(model.wtrans.scale)
        .scale_y(model.wtrans.scale);

    if model.plot_config.draw_grid == true {
        let win = app.main_window().rect();
        draw_grid(&draw, &win, model.plot_config.grid_step, 1.0, 0.01);
        draw_grid(&draw, &win, model.plot_config.grid_step / 5., 0.5, 0.01);
    }

    draw.background().color(BLACK);
    for mesh in model.map_points.iter() {
        let points = (0..mesh.len()).map(|i| {
            mesh[i]
        });
        draw.polygon()
            .color(WHITE)
            .points(points);
    }

    draw.ellipse()
        .w(15.)
        .h(15.)
        .x(model.pose.x)
        .y(model.pose.y)
        .color(STEELBLUE);

    if model.initialized == true {
        visualize_rays(&draw, &model.ranges, &model.pose, &model.lidar_param, model.ray_num / 3);
        // draw_occ_grids(&draw, &model.occ_grid, model.grid_specs.0, model.grid_specs.1, model.grid_size);
    }
        
    let start_pos = pt2(model.pose.x, model.pose.y);
    let dir = local_mouse_position(app, model) - start_pos;
    let norm = (dir.x * dir.x + dir.y * dir.y + 1e-5).sqrt();
    draw.arrow()
        .start(start_pos)
        .end(start_pos + dir * 40. / norm)
        .weight(2.)
        .color(MEDIUMSPRINGGREEN);

    // Write the result of our drawing to the window's frame.
    draw.to_frame(app, &frame).unwrap();
}

fn visualize_rays(draw: &Draw, ranges: &Vec<libc::c_float>, pose: &Point3, lidar_param: &cuda_helper::Vec3_cuda, ray_num: usize) {
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
            .color(RED);
    }
}


fn draw_grid(draw: &Draw, win: &Rect, step: f32, weight: f32, alpha: f32) {
    let step_by = || (0..).map(|i| i as f32 * step);
    let r_iter = step_by().take_while(|&f| f < win.right());
    let l_iter = step_by().map(|f| -f).take_while(|&f| f > win.left());
    let x_iter = r_iter.chain(l_iter);
    for x in x_iter {
        draw.line()
            .weight(weight)
            .rgba(1., 1., 1., alpha)
            .points(pt2(x, win.bottom()), pt2(x, win.top()));
        }
        let t_iter = step_by().take_while(|&f| f < win.top());
    let b_iter = step_by().map(|f| -f).take_while(|&f| f > win.bottom());
    let y_iter = t_iter.chain(b_iter);
    for y in y_iter {
        draw.line()
            .weight(weight)
            .rgba(1., 1., 1., alpha)
            .points(pt2(win.left(), y), pt2(win.right(), y));
    }
}

// debug occupancy grid visualization
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
