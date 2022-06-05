use nannou::prelude::*;
use nannou_egui::{Egui, egui::Rect};

use super::mesh::Chain;
use crate::utils::map_io;
use crate::utils::color::EditorColor;
use crate::utils::structs::*;
use crate::utils::async_timer as at;

pub struct Selection {
    pub selected: Vec<Vec<usize>>,
    pub bl: Point2,
    pub tr: Point2,
    pub key_pressed: bool
}

impl Selection {
    pub fn new() -> Selection {
        Selection { 
            selected: Vec::new(),
            bl: pt2(0., 0.), tr: pt2(0., 0.),
            key_pressed: false
        }
    }
}

// TODO: 个人希望，此处存储点使用链表，则结果的存储使用链表的链表
pub struct Model {
    pub map_points: Vec<Chain>,
    pub select: Selection,
    pub plot_config: PlotConfig,
    pub wctrl: WindowCtrl,
    pub wtrans: WindowTransform,
    pub key_stat: KeyStatus,
    pub trajectory: Trajectory,
    pub color: EditorColor,
    pub egui: Egui,
    pub egui_rect: Rect,
    pub saved_file_name: String,
    pub scrn_mov: bool,
    pub obj_mov: bool,
    pub mouse_moving_object: bool,
    pub timer_event: at::AsyncTimerEvent<String>
}

impl Model {
    pub fn new(app: &App, window_id: WindowId, config: &map_io::Config) -> Model {
        let window = app.window(window_id).unwrap();
        let egui = Egui::from_window(&window);
        Model {
            map_points: vec![Chain::new()], 
            select: Selection::new(),
            plot_config: PlotConfig::new(),
            wctrl: WindowCtrl::new(window_id, config.screen.width as f32, config.screen.height as f32, exit),
            wtrans: WindowTransform::new(),
            key_stat: KeyStatus{ctrl_pressed: false},
            trajectory: Trajectory::new(),
            color: EditorColor::new(),
            egui: egui,
            egui_rect: Rect::from_x_y_ranges(0.0..=1.0, 0.0..=1.0),
            saved_file_name: String::from(""),
            scrn_mov: false,
            obj_mov: false,
            mouse_moving_object: false,
            timer_event: at::AsyncTimerEvent::new(3)
        }
    }

    #[inline(always)]
    pub fn cursor_in_gui(&self, w_h: &(f32, f32), pt: &Point2) -> bool {
        let local_pt = pt2(pt.x + 0.5 * w_h.0, 0.5 * w_h.1 - pt.y);
        (local_pt.x > self.egui_rect.left()) && (local_pt.y + 30. > self.egui_rect.top()) && (local_pt.x < self.egui_rect.right()) && (local_pt.y - 240. < self.egui_rect.top())
    }
}

fn exit(app: &App) {
    // TODO: 保存
    app.quit();
}
