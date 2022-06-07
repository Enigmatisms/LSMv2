use nannou::prelude::*;
use nannou_egui::{Egui, egui::Rect};

use array2d::Array2D;
use super::grid;
use super::cuda_helper;

use crate::utils::map_io;
use crate::utils::color::EditorColor;
use crate::utils::structs::*;
use crate::utils::async_timer as at;

pub struct Model {
    pub map_points: Vec<Vec<Point2>>,
    pub grid_specs: (f32, f32, f32, f32),
    pub occ_grid: Array2D<i32>,
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
    pub initialized: bool,
    pub grid_size: f32,

    pub color: EditorColor,
    pub egui: Egui,
    pub egui_rect: Rect,
    pub key_stat: KeyStatus,
    pub trajectory: Trajectory,
    pub timer_event: at::AsyncTimerEvent<String>,
}

impl Model {
    pub fn new(app: &App, window_id:  WindowId, config: &map_io::Config, meshes: map_io::Meshes, lidar_param: cuda_helper::Vec3_cuda, ray_num: usize) -> Model {
        let grid_specs = grid::get_bounds(&meshes, config.grid_size);
        let mut occ_grid = Array2D::filled_with(-1, grid_specs.3 as usize, grid_specs.2 as usize);
        grid::line_drawing(&mut occ_grid, &meshes, grid_specs.0, grid_specs.1, config.grid_size);
        Model {
            map_points: meshes, 
            occ_grid: occ_grid,
            grid_specs: grid_specs,
            plot_config: PlotConfig {
                draw_grid: false, grid_step: 100.0, grid_alpha: 0.01
            },
            wctrl: WindowCtrl::new(window_id, config.screen.width as f32, config.screen.height as f32, exit),
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
            initialized: false,
            grid_size: config.grid_size,
            color: EditorColor::new(),
            egui: Egui::from_window(&app.window(window_id).unwrap()),
            egui_rect: Rect::from_x_y_ranges(0.0..=1.0, 0.0..=1.0),
            key_stat: KeyStatus{ctrl_pressed: false},
            trajectory: Trajectory::new(),
            timer_event: at::AsyncTimerEvent::new(3)
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