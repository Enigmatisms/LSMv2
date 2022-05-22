use nannou::prelude::*;
use crate::cuda_helper;
use crate::map_io;

pub struct WindowCtrl {
    pub window_id: WindowId,
    pub win_w: f32,
    pub win_h: f32,
    pub exit_func: fn(app: &App)
}

pub struct WindowTransform {
    pub t: Point2,
    pub t_start: Point2,
    pub rot: f32,
    pub rot_start: f32,
    pub t_set: bool,
    pub r_set: bool,
    pub scale: f32
}

pub struct PlotConfig {
    pub draw_grid: bool,
    pub grid_step: f32,
}

pub struct Model {
    pub map_points: Vec<Vec<Point2>>,
    pub plot_config: PlotConfig,
    pub wctrl: WindowCtrl,
    pub wtrans: WindowTransform,
    pub pose: Point3,
    pub velo: Point3,
    pub pid: Point3,
    pub velo_max: Point2,

    pub lidar_param: cuda_helper::Vec3_cuda,
    pub lidar_noise: libc::c_float,
    pub ray_num: usize,
    pub ranges: Vec<libc::c_float>,
    pub initialized: bool
}

impl Model {
    pub fn new(window_id:  WindowId, config: &map_io::Config, meshes: map_io::Meshes, lidar_param: cuda_helper::Vec3_cuda, ray_num: usize) -> Model {
        Model {
            map_points: meshes, 
            plot_config: PlotConfig {
                draw_grid: false, grid_step: 100.0,
            },
            wctrl: WindowCtrl {
                window_id: window_id,
                win_w: config.screen.width as f32, win_h: config.screen.height as f32,
                exit_func: exit,
            },
            wtrans: WindowTransform {
                t: pt2(0.0, 0.0), t_start: pt2(0.0, 0.0),
                rot: 0., rot_start: 0., t_set: true, r_set: true, scale: 1.0,
            },
            pose: pt3(0., 0., 0.),
            velo: pt3(0., 0., 0.),
            pid: pt3(config.robot.pid_kp, config.robot.pid_ki, config.robot.pid_kd),
            velo_max: pt2(config.robot.t_vel, config.robot.r_vel),
            lidar_param: lidar_param,
            lidar_noise: config.lidar.noise_k,
            ray_num: ray_num,
            ranges: vec![0.; ray_num],
            initialized: false
        }
    }
}

fn exit(app: &App) {
    unsafe {
        cuda_helper::deallocateFixed();
        cuda_helper::deallocateDevice();
    }
    app.quit();
}