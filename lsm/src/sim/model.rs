use nannou::prelude::*;
use nannou_egui::Egui;

use array2d::Array2D;
use super::grid;
use super::cuda_helper;

use crate::utils::map_io;
use crate::utils::structs::*;
use crate::utils::async_timer as at;
use crate::utils::color::EditorColor;
use std::f32::consts::PI;

#[derive(Default)]
pub struct StringConfig {
    pub map_name: String,
    pub config_output: String,
}

impl StringConfig {
    pub fn new(map_path: &String) -> StringConfig {
        StringConfig {
            map_name: map_path.clone(),
            config_output: String::from(""),
        }
    }
}

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
    pub inside_gui: bool,
    pub key_stat: KeyStatus,
    pub trajectory: Trajectory,
    pub timer_event: at::AsyncTimerEvent<String>,
    pub str_config: StringConfig
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
            plot_config: PlotConfig::new(),
            wctrl: WindowCtrl::new(window_id, config.screen.width as f32, config.screen.height as f32, exit),
            wtrans: WindowTransform::new(),
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
            key_stat: KeyStatus{ctrl_pressed: false},
            trajectory: Trajectory::new(),                                      // TODO: trajectory recorder
            timer_event: at::AsyncTimerEvent::new(3),
            inside_gui: false,
            str_config: StringConfig::new(&config.map_path),
        }
    }

    pub fn reload_config(
        config: &map_io::Config, win_w: &mut f32, win_h: &mut f32, pid: &mut Point3,
        lidar_color: &mut [f32; 4], velo_max: &mut Point2, lidar_noise: &mut f32
    ) {
        pid.x = config.robot.pid_kp;
        pid.y = config.robot.pid_ki;
        pid.z = config.robot.pid_kd;
        velo_max.x = config.robot.t_vel;
        velo_max.y = config.robot.r_vel;
        lidar_color[0] = config.lidar.lidar_r;
        lidar_color[1] = config.lidar.lidar_g;
        lidar_color[2] = config.lidar.lidar_b;
        lidar_color[3] = config.lidar.lidar_a;
        *win_w = config.screen.width as f32; 
        *win_h = config.screen.height as f32; 
        *lidar_noise = config.lidar.noise_k;
    }

    pub fn output_config(
        lidar_param: &cuda_helper::Vec3_cuda, velo_max: &Point2, pid: &Point3, map_name: &String,
        lidar_c: &[f32;4], lidar_noise: &f32, win_w: &f32, win_h: &f32, grid_size: &f32
    ) -> map_io::Config {
        map_io::Config {
            lidar: map_io::LidarConfig {
                amin: lidar_param.x / PI * 180.,
                amax: lidar_param.y / PI * 180.,
                ainc: lidar_param.z / PI * 180.,
                noise_k: *lidar_noise,
                lidar_r: lidar_c[0],
                lidar_g: lidar_c[1],
                lidar_b: lidar_c[2],
                lidar_a: lidar_c[3],
            },
            robot: map_io::ScannerConfig {
                t_vel: velo_max.x, r_vel: velo_max.y,
                pid_kp: pid.x, pid_ki: pid.y, pid_kd: pid.z,
            },
            screen: map_io::ScreenConfig {
                width: *win_w as u32, height: *win_h as u32
            },
            map_path: map_name.clone(),
            grid_size: *grid_size
        }
    }
 
    pub fn recompute_grid(grid_specs: &mut (f32, f32, f32, f32), occ_grid: &mut Array2D<i32>, meshes: &map_io::Meshes, grid_size: f32) {
        *grid_specs = grid::get_bounds(&meshes, grid_size);
        *occ_grid = Array2D::filled_with(-1, grid_specs.3 as usize, grid_specs.2 as usize);
        grid::line_drawing(occ_grid, meshes, grid_specs.0, grid_specs.1, grid_size);
    }
}

fn exit(app: &App) {
    unsafe {
        cuda_helper::deallocateFixed();
        cuda_helper::deallocateDevice();
    }
    app.quit();
}