use nannou::prelude::*;

pub struct WindowCtrl {
    pub window_id: WindowId,
    pub win_w: f32,
    pub win_h: f32,
    pub gui_visible: bool,
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
    pub grid_alpha: f32
}

pub struct KeyStatus {
    pub ctrl_pressed: bool,
}

pub struct Trajectory {
    pub traj: Vec<Point2>,
    pub is_visible: bool,
    pub alpha: f32,
}

impl WindowCtrl {
    pub fn new(win_id: WindowId, win_w: f32, win_h: f32, exit_f: fn(app: &App)) -> WindowCtrl {
        WindowCtrl {window_id: win_id, win_w: win_w, win_h: win_h, gui_visible: true, exit_func: exit_f}
    }

    pub fn switch_gui_visibility(&mut self) {
        self.gui_visible = !self.gui_visible;
        println!("Visibility toggled.");
    }
}

impl Trajectory {
    pub fn new() -> Trajectory {
        Trajectory {traj: Vec::new(), is_visible: false, alpha: 0.0}
    }
}
