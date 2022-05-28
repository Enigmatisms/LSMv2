// universal plotting for editor, visualizer and simulator
use nannou::prelude::*;
use super::utils;
use super::structs::WindowTransform;

#[inline(always)]
pub fn window_transform(draw: Draw, wint: &WindowTransform) -> Draw {
    draw
        .x_y(wint.t.x, wint.t.y)
        .rotate(wint.rot)
        .scale_x(wint.scale)
        .scale_y(wint.scale)
}

pub fn draw_grid(draw: &Draw, win: &Rect, step: f32, weight: f32, alpha: f32) {
    let step_by = || (0..).map(|i| i as f32 * step);
    let r_iter = step_by().take_while(|&f| f < win.right());
    let l_iter = step_by().map(|f| -f).take_while(|&f| f > win.left());
    let x_iter = r_iter.chain(l_iter);
    for x in x_iter {
        draw.line()
            .weight(weight)
            .rgba(1., 1., 1., alpha)
            .points(pt2(x, win.bottom()), pt2(x, win.top()));
        }
        let t_iter = step_by().take_while(|&f| f < win.top());
    let b_iter = step_by().map(|f| -f).take_while(|&f| f > win.bottom());
    let y_iter = t_iter.chain(b_iter);
    for y in y_iter {
        draw.line()
            .weight(weight)
            .rgba(1., 1., 1., alpha)
            .points(pt2(win.left(), y), pt2(win.right(), y));
    }
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