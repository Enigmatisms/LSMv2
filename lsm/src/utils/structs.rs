use nannou::prelude::*;

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
    pub grid_alpha: f32
}
