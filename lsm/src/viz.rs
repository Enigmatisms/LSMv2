use nannou::prelude::*;
use crate::cuda_helper;
use crate::map_io;
use crate::ctrl;
use crate::utils;

pub struct WindowCtrl {
    window_id: WindowId,
    win_w: u32,
    win_h: u32,
    pub exit_func: fn(app: &App)
}

pub struct Model {
    map_points: Vec<Vec<Point2>>,
    pub wctrl: WindowCtrl,
    pub pose: Point3,
    pub velo: Point3,
    pub pid: Point3,
    pub velo_max: Point2,
    pub mouse_pos: Point2,

    lidar_param: cuda_helper::Vec3_cuda,
    ray_num: usize,
    ranges: Vec<libc::c_float>,
    pub initialized: bool
}

fn exit(app: &App) {
    unsafe {
        cuda_helper::deallocateFixed();
        cuda_helper::deallocateDevice();
    }
    app.quit();
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
    println!("File path: {}", config.map_path);
    let meshes: map_io::Meshes = map_io::parse_map_file(config.map_path.as_str()).unwrap();
    let mut seg_point_arr: Vec<cuda_helper::Vec2_cuda> = Vec::new();
    let seg_num = map_io::meshes_to_segments(&meshes, &mut seg_point_arr);
    let lidar_param = cuda_helper::Vec3_cuda{x: config.lidar.amin, y: config.lidar.amax, z:config.lidar.ainc};
    let ray_num = map_io::get_ray_num(&lidar_param);

    unsafe {
        cuda_helper::intializeFixed(ray_num as libc::c_int);
        cuda_helper::unwrapMeshes(seg_point_arr.as_ptr(), seg_num as libc::c_int, false);
    }

    Model {
        map_points: meshes, 
        wctrl: WindowCtrl {
            window_id: window_id,
            win_w: 1200, win_h: 900,
            exit_func: exit,
        },
        pose: pt3(0., 0., 0.),
        velo: pt3(0., 0., 0.),
        pid: pt3(config.robot.pid_kp, config.robot.pid_ki, config.robot.pid_kd),
        velo_max: pt2(config.robot.t_vel, config.robot.r_vel),
        mouse_pos: pt2(0., 0.),
        lidar_param: lidar_param,
        ray_num: ray_num,
        ranges: vec![0.; ray_num],
        initialized: false
    }
}

pub fn update(_app: &App, _model: &mut Model, _: Update) {
    static mut LOCAL_INT: f32 = 0.0;
    static mut LOCAL_DIFF: f32 = 0.0;
    if _model.initialized == false {return;}
    let tmp_x = _model.pose.x +_model.velo.x; 
    let tmp_y = _model.pose.y +_model.velo.y; 
    if tmp_x > -600. && tmp_x < 600. {
        _model.pose.x = tmp_x;
    }
    if tmp_y > -450. && tmp_y < 450. {
        _model.pose.y = tmp_y;
    }

    let dir = _model.mouse_pos - pt2(_model.pose.x, _model.pose.y);
    let target_angle = dir.y.atan2(dir.x);
    let diff = utils::good_angle(target_angle - _model.pose.z);
    unsafe {
        LOCAL_INT += diff;
        let kd_val = diff - LOCAL_DIFF;
        LOCAL_DIFF = diff;
        _model.pose.z += _model.pid.x * diff + _model.pid.y * LOCAL_INT + _model.pid.z * kd_val;
        
        let pose = cuda_helper::Vec3_cuda {x:_model.pose.x, y:_model.pose.y, z:_model.pose.z};
        cuda_helper::rayTraceRender(&_model.lidar_param, &pose, _model.ray_num as i32, _model.ranges.as_mut_ptr());
    }
}

pub fn event(_app: &App, _model: &mut Model, event: WindowEvent) {}

pub fn view(app: &App, model: &Model, frame: Frame) {
    // Begin drawing
    let draw = app.draw();

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

    if model.initialized == false {
        let start_pos = pt2(model.pose.x, model.pose.y);
        let dir = model.mouse_pos - start_pos;
        let norm = (dir.x * dir.x + dir.y * dir.y + 1e-5).sqrt();
        draw.arrow()
            .start(start_pos)
            .end(start_pos + dir * 50. / norm)
            .weight(2.)
            .color(MEDIUMSPRINGGREEN);
    } else {
        visualize_rays(&draw, &model.ranges, &model.pose, &model.lidar_param, model.ray_num / 3);
    }


    // Write the result of our drawing to the window's frame.
    draw.to_frame(app, &frame).unwrap();
}

fn visualize_rays(draw: &Draw, ranges: &Vec<libc::c_float>, pose: &Point3, lidar_param: &cuda_helper::Vec3_cuda, ray_num: usize) {
    let cur_angle_min = pose.z + lidar_param.x + lidar_param.z;
    for i in 0..ray_num {
        let r = ranges[i];
        if r > 1e5 {continue;}
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
