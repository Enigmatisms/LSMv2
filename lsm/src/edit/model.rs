use nannou::prelude::*;
use nannou_egui::Egui;

use super::mesh::Chain;
use super::addm::AdditionalMode;
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

#[derive(PartialEq, Eq)]
pub enum DrawState {
    Arbitrary,
    Straight,
    Rect,
}

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
    pub inside_gui: bool,
    pub saved_file_name: String,
    pub scrn_mov: bool,
    pub obj_mov: bool,
    pub mouse_moving_object: bool,
    pub draw_state: DrawState,
    pub timer_event: at::AsyncTimerEvent<String>,
    pub add_drawer: AdditionalMode
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
            inside_gui: false,
            saved_file_name: String::from(""),
            scrn_mov: false,
            obj_mov: false,
            mouse_moving_object: false,
            draw_state: DrawState::Arbitrary,
            timer_event: at::AsyncTimerEvent::new(3),
            add_drawer: AdditionalMode::new()
        }
    }
}

fn exit(app: &App) {
    // TODO: 保存
    app.quit();
}
