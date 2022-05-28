use nannou::prelude::*;
use super::mesh::Chain;

use crate::utils::map_io;
use crate::utils::structs::{PlotConfig, WindowCtrl, WindowTransform};

pub struct Selection {
    pub selected: Vec<Vec<usize>>,
    pub bl: Point2,
    pub tr: Point2,
    pub key_pressed: bool
}

// TODO: 个人希望，此处存储点使用链表，则结果的存储使用链表的链表
pub struct Model {
    pub map_points: Vec<Chain>,
    pub select: Selection,
    pub plot_config: PlotConfig,
    pub wctrl: WindowCtrl,
    pub wtrans: WindowTransform,
    pub scrn_mov: bool,
    pub obj_mov: bool,
    pub mouse_moving_object: bool
}

impl Model {
    pub fn new(window_id:  WindowId, config: &map_io::Config) -> Model {
        Model {
            map_points: vec![Chain::new()], 
            select: Selection {
                selected: Vec::new(),
                bl: pt2(0., 0.), tr: pt2(0., 0.),
                key_pressed: false
            },
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
            scrn_mov: false,
            obj_mov: false,
            mouse_moving_object: false
        }
    }
}

fn exit(app: &App) {
    // TODO: 保存
    app.quit();
}