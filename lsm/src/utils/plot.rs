// universal plotting for editor, visualizer and simulator
use nannou::prelude::*;
use chrono::prelude::*;
use std::path::Path;
use super::utils;
use super::structs::WindowTransform;

static SIGN_OFFSET: [(f32, f32); 4] = [(-1., 1.), (1., 1.), (1., -1.), (-1., -1.)];

#[inline(always)]
pub fn window_transform(draw: Draw, wint: &WindowTransform) -> Draw {
    draw
        .x_y(wint.t.x, wint.t.y)
        .rotate(wint.rot)
        .scale_x(wint.scale)
        .scale_y(wint.scale)
}

pub fn plot_trajectory(draw: &Draw, trajectory: &Vec<Point2>, rgb: &(f32, f32, f32), alpha: f32) {
    let pt2draw = (0..trajectory.len()).map(|i| {trajectory[i]});
    draw.polyline()
        .weight(1.5)
        .points(pt2draw)
        .rgba(rgb.0, rgb.1, rgb.2, alpha);
}

pub fn draw_grid(draw: &Draw, win: &Rect, step: f32, weight: f32, color: &(f32, f32, f32), alpha: f32) {
    let step_by = || (0..).map(|i| i as f32 * step);
    let r_iter = step_by().take_while(|&f| f < win.right());
    let l_iter = step_by().map(|f| -f).take_while(|&f| f > win.left());
    let x_iter = r_iter.chain(l_iter);
    for x in x_iter {
        draw.line()
            .weight(weight)
            .rgba(color.0, color.1, color.2, alpha)
            .points(pt2(x, win.bottom()), pt2(x, win.top()));
        }
        let t_iter = step_by().take_while(|&f| f < win.top());
    let b_iter = step_by().map(|f| -f).take_while(|&f| f > win.bottom());
    let y_iter = t_iter.chain(b_iter);
    for y in y_iter {
        draw.line()
            .weight(weight)
            .rgba(color.0, color.1, color.2, alpha)
            .points(pt2(win.left(), y), pt2(win.right(), y));
    }
}

pub fn draw_points(draw: &Draw, pts: &Vec<Point2>, rgba: &(f32, f32, f32, f32)) {
    for pt in pts.iter() {
        draw.ellipse()
            .w_h(5., 5.)
            .x_y(pt.x, pt.y)
            .rgba(rgba.0, rgba.1, rgba.2, rgba.3);
    }
}

pub fn draw_frame(app: &App, draw: &Draw, wint: &WindowTransform) {
    let rect = app.main_window().rect();
    let pts = (0..4).map(|i| {
        let (sign_x, sign_y) = SIGN_OFFSET[i];
        localized_position(&pt2(rect.w() / 2. * sign_x, rect.h() / 2. * sign_y), wint)
    });
    draw.polyline()
        .weight(2.)
        .rgb(0.5, 0.5, 0.5)
        .points_closed(pts);
}

pub fn local_mouse_position(_app: &App, wint: &WindowTransform) -> Point2 {
    let mouse = _app.mouse.position();
    localized_position(&mouse, wint)
}

#[inline(always)]
pub fn localized_position(pt: &Point2, wint: &WindowTransform) -> Point2 {
    let mut res = *pt - wint.t;
    let rotation_inv= utils::get_rotation(&-wint.rot);
    res = rotation_inv.mul_vec2(res);
    res / wint.scale
}

#[inline(always)]
pub fn to_screen(pt: &Point2, wtrans: &WindowTransform) -> Point2 {
    let rotation = utils::get_rotation(&wtrans.rot);
    wtrans.scale * rotation.mul_vec2(*pt) + wtrans.t
}

pub fn take_snapshot(window: &Window) {
    let timestamp = (std::time::SystemTime::now()).duration_since(std::time::UNIX_EPOCH).unwrap();
    let time_sec = timestamp.as_secs() as i64;
    let naive = NaiveDateTime::from_timestamp(time_sec, 0);
    let datetime: DateTime<Utc> = DateTime::from_utc(naive, Utc);
    let date_path = datetime.format("../screenshots/screenshot-%Y-%m-%d-%H-%M-%S.png").to_string();
    if Path::new("../screenshots/").exists() {
        let _ = std::fs::create_dir("../screenshots/");
    }
    window.capture_frame(date_path);
}