use nannou::prelude::*;
use nannou::event::Key;
use crate::cuda_helper;
use crate::map_io;

pub struct WindowCtrl {
    pub window_id: WindowId,
    pub exit_func: fn(app: &App),
    pub win_w: u32,
    pub win_h: u32
}

pub struct Model {
    pub map_points: Vec<Vec<Point2>>,
    pub wctrl: WindowCtrl,
    pub pose: Point3,
    pub velo: Point3,
    pub lidar_param: cuda_helper::Vec3_cuda,
    pub ray_num: usize,
    pub ranges: [libc::c_float; 3600]
}

fn exit(app: &App) {
    unsafe {
        cuda_helper::deallocateFixed();
        cuda_helper::deallocateDevice();
    }
    app.quit();
}

pub fn model(app: &App) -> Model {
    let window_id = app
        .new_window()
        .event(event)
        .size(1200, 900)
        .view(view)
        .build()
        .unwrap();
    app.set_exit_on_escape(false);

    let meshes: map_io::Meshes = map_io::parse_map_file("../maps/test.txt").unwrap();
    let mut seg_point_arr: Vec<cuda_helper::Vec2_cuda> = Vec::new();
    let seg_num = map_io::meshes_to_segments(&meshes, &mut seg_point_arr);
    unsafe {
        cuda_helper::unwrapMeshes(seg_point_arr.as_ptr(), seg_num as libc::c_int, false);
    }

    match app.window(window_id) {
        Some(window) => {
            let size_tuple = window.inner_size_pixels();
            println!("Size of current window: {}, {}", size_tuple.0, size_tuple.1);
        },
        None => {},
    }

    let lidar_param = cuda_helper::Vec3_cuda{x:-2.093667881871531, y:2.093667881871531, z:0.001454441043328608};
    let ray_num = map_io::get_ray_num(&lidar_param);
    Model {
        map_points: meshes, 
        wctrl: WindowCtrl {
            window_id: window_id,
            exit_func: exit,
            win_w: 1200, win_h: 900
        },
        pose: pt3(0., 0., 0.),
        velo: pt3(0., 0., 0.),
        lidar_param: lidar_param,
        ray_num: ray_num,
        ranges: [0.; 3600]
    }
}

pub fn update(_app: &App, _model: &mut Model, _: Update) {
    let tmp_x = _model.pose.x +_model.velo.x; 
    let tmp_y = _model.pose.y +_model.velo.y; 
    if tmp_x > -600. && tmp_x < 600. {
        _model.pose.x = tmp_x;
    }
    if tmp_y > -450. && tmp_y < 450. {
        _model.pose.y = tmp_y;
    }
    unsafe {
        let pose = cuda_helper::Vec3_cuda {x:_model.pose.x, y:_model.pose.y, z:_model.pose.z};
        cuda_helper::rayTraceRender(&_model.lidar_param, &pose, _model.ray_num as i32, _model.ranges.as_mut_ptr());
    }
}

pub fn event(_app: &App, _model: &mut Model, event: WindowEvent) {
    match event {
        KeyPressed(key) => {
            match key {
                Key::W => {_model.velo.y = 3.0;},
                Key::A => {_model.velo.x = -3.0;},
                Key::S => {_model.velo.y = -3.0;},
                Key::D => {_model.velo.x = 3.0;},
                Key::Escape => {
                    (_model.wctrl.exit_func)(_app);
                },
                _ => {},
            }
        },
        KeyReleased(key) => {
            match key {
                Key::W => {_model.velo.y = 0.0;},
                Key::A => {_model.velo.x = 0.0;},
                Key::S => {_model.velo.y = 0.0;},
                Key::D => {_model.velo.x = 0.0;},
                _ => {},
            }
        }
        _ => {}
    }
}

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

    visualize_rays(&draw, &model.ranges, &model.pose, &model.lidar_param, model.ray_num / 3);

    // Write the result of our drawing to the window's frame.
    draw.to_frame(app, &frame).unwrap();
}

fn visualize_rays(draw: &Draw, ranges: &[f32], pose: &Point3, lidar_param: &cuda_helper::Vec3_cuda, ray_num: usize) {
    let cur_angle_min = pose.z + lidar_param.x + lidar_param.z;
    for i in 0..ray_num {
        let r = ranges[i];
        // print!("{}, ", r);
        let cur_angle = cur_angle_min + lidar_param.z * 3. * (i as f32);
        let dir = pt2( cur_angle.cos(), cur_angle.sin());
        let start_p = pt2(pose.x, pose.y);
        let end_p = start_p + dir * ranges[i];
        draw.line()
            .start(start_p)
            .end(end_p)
            .weight(1.)
            .color(RED);
    } 
    // print!("\n");
}
