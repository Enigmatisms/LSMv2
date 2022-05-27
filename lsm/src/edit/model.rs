use nannou::prelude::*;

use crate::utils::map_io;
use crate::utils::structs::{PlotConfig, WindowCtrl, WindowTransform};

// TODO: 个人希望，此处存储点使用链表，则结果的存储使用链表的链表
pub struct Model {
    pub map_points: Vec<Vec<Point2>>,
    pub plot_config: PlotConfig,
    pub wctrl: WindowCtrl,
    pub wtrans: WindowTransform,
    pub scrn_mov: bool
}

impl Model {
    pub fn new(window_id:  WindowId, config: &map_io::Config, meshes: map_io::Meshes) -> Model {
        Model {
            map_points: meshes, 
            plot_config: PlotConfig {
                draw_grid: false, grid_step: 100.0,
            },
            wctrl: WindowCtrl {
                window_id: window_id,
                win_w: config.screen.width as f32, win_h: config.screen.height as f32,
                exit_func: placeholder,
            },
            wtrans: WindowTransform {
                t: pt2(0.0, 0.0), t_start: pt2(0.0, 0.0),
                rot: 0., rot_start: 0., t_set: true, r_set: true, scale: 1.0,
            },
            scrn_mov: false
        }
    }
}

fn placeholder(_app: &App) {}
