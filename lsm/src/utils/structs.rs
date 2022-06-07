use nannou::prelude::*;
use std::time::Instant;

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

pub struct TimerEvent {
    to_display: String,
    saved_time: Instant,
    recent_saved: bool,
}

impl TimerEvent {
    pub fn new() -> TimerEvent {
        TimerEvent {saved_time: Instant::now(), recent_saved: false, to_display: String::from("...")}
    }
    #[inline(always)]
    pub fn time_elapsed(&self) -> f32 {
        self.saved_time.elapsed().as_secs_f32()
    }

    #[inline(always)]
    pub fn reset_time(&mut self, string: &str) {
        self.recent_saved = true;
        self.saved_time = Instant::now();
        self.to_display = String::from(string);
    }

    #[inline(always)]
    pub fn is_recent_saved(&self) -> bool {
        self.recent_saved
    }
    
    #[inline(always)]
    pub fn save_expire(&mut self) {
        self.recent_saved = false;
        self.to_display = String::from("...");
    }

    #[inline(always)]
    pub fn str_to_display(&self) -> &str {
        self.to_display.as_str()
    }
}

impl WindowCtrl {
    pub fn new(win_id: WindowId, win_w: f32, win_h: f32, exit_f: fn(app: &App)) -> WindowCtrl {
        WindowCtrl {window_id: win_id, win_w: win_w, win_h: win_h, gui_visible: true, exit_func: exit_f}
    }

    pub fn switch_gui_visibility(&mut self) {
        self.gui_visible = !self.gui_visible;
    }
}

impl WindowTransform {
    pub fn new() -> WindowTransform{
        WindowTransform {
            t: pt2(0.0, 0.0), t_start: pt2(0.0, 0.0),
            rot: 0., rot_start: 0., t_set: true, r_set: true, scale: 1.0,
        }
    }
    
    #[inline(always)]
    pub fn clear_offset(&mut self) {
        self.rot = 0.;
        self.t = pt2(0., 0.);
    }
}

impl PlotConfig {
    pub fn new() -> PlotConfig {
        PlotConfig {
            draw_grid: false, grid_step: 100.0, grid_alpha: 0.01
        }
    }
}

impl Trajectory {
    pub fn new() -> Trajectory {
        Trajectory {traj: Vec::new(), is_visible: true, alpha: 0.35}
    }
}
