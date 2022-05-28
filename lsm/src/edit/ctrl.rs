use nannou::prelude::*;
use super::model::Model;
use crate::utils::utils;

pub fn key_pressed(_app: &App, _model: &mut Model, _key: Key) {
    match _key {
        // _model.scrn and _model.obj are mutex to be true
        Key::LShift => {
            _model.scrn_mov = true;
            _model.obj_mov = false;
        },
        Key::M => {
            _model.obj_mov = true;
            _model.scrn_mov = false;
        },
        _ => {},
    }
}

pub fn key_released(_app: &App, _model: &mut Model, _key: Key) {
    match _key {
        // _model.scrn and _model.obj are mutex to be true
        Key::LShift => {
            _model.scrn_mov = false;
        },
        Key::M => {
            _model.obj_mov = false;
        },
        Key::Delete => {
            let mut remove_stack: Vec<usize> = Vec::new();
            for pts in _model.select.selected.iter() {
                let mesh_id = *pts.first().unwrap();
                if _model.map_points[mesh_id].batch_remove(&pts[1..]) == false {
                    remove_stack.push(mesh_id);
                }
            }
            while remove_stack.is_empty() == false {
                if let Some(mesh_id) = remove_stack.pop() {
                    _model.map_points.remove(mesh_id);
                } else {
                    break;
                }
            }
        },
        _ => {},
    }
}

// initial position selection
pub fn mouse_pressed(_app: &App, _model: &mut Model, _button: MouseButton) {
    let point = _app.mouse.position();
    if _model.scrn_mov == true {
        match _button {
            MouseButton::Middle => {
                _model.wtrans.t_start = point;
                _model.wtrans.t_set = false;
            },
            MouseButton::Left => {
                _model.wtrans.rot_start = point.y.atan2(point.x);
                _model.wtrans.r_set = false;
            },
            _ => {}
        }
    } else if _model.obj_mov == true {
        if _button == MouseButton::Left {
            _model.wtrans.t_start = point;
            _model.mouse_moving_object = true;
        }
    } else {

    }
}

// mouse release will determine the initial angle
pub fn mouse_released(_app: &App, _model: &mut Model, _button: MouseButton) {
    let now_pos = _app.mouse.position();
    if _model.scrn_mov == false {
        match _button {
            MouseButton::Left => {},
            _ => {},
        }
    } else {
        if _model.scrn_mov == true {
            match _button {
                MouseButton::Middle => {
                    _model.wtrans.t += now_pos - _model.wtrans.t_start;
                    _model.wtrans.t_set = true;
                },
                MouseButton::Left => {
                    let delta_angle = utils::good_angle(now_pos.y.atan2(now_pos.x) - _model.wtrans.rot_start);
                    _model.wtrans.rot = utils::good_angle(delta_angle + _model.wtrans.rot);
                    _model.wtrans.r_set = true;
                    _model.wtrans.t = utils::get_rotation(&delta_angle).mul_vec2(_model.wtrans.t);
                },
                MouseButton::Right => {
                    _model.wtrans.t = pt2(0., 0.);
                    _model.wtrans.rot = 0.;
                },
                _ => {}
            }
        } else if _model.obj_mov == true {
            if _button == MouseButton::Left {
                _model.mouse_moving_object = false;
            }
        }
    }
}

// pid angle control
pub fn mouse_moved(_app: &App, _model: &mut Model, _pos: Point2) {
    let point = _app.mouse.position();
    if _model.scrn_mov == true {
        if _model.wtrans.t_set == false {
            _model.wtrans.t += point - _model.wtrans.t_start;
            _model.wtrans.t_start = point;
        }
        if _model.wtrans.r_set == false {
            let current_angle = point.y.atan2(point.x);
            let delta_angle = utils::good_angle(current_angle - _model.wtrans.rot_start);
            _model.wtrans.rot_start = current_angle;
            _model.wtrans.rot = utils::good_angle(delta_angle + _model.wtrans.rot);
            _model.wtrans.t = utils::get_rotation(&delta_angle).mul_vec2(_model.wtrans.t);
        }
    } else if _model.obj_mov == true && _model.mouse_moving_object == true {
        let delta_t = utils::get_rotation(&-_model.wtrans.rot).mul_vec2(point - _model.wtrans.t_start) / _model.wtrans.scale;
        for pts in _model.select.selected.iter() {
            let mesh_id = *pts.first().unwrap();
            _model.map_points[mesh_id].translate(&delta_t, &pts[1..]);
        }
    }
}

// change velocity
pub fn mouse_wheel(_app: &App, _model: &mut Model, _dt: MouseScrollDelta, _phase: TouchPhase) {
    match _dt {
        MouseScrollDelta::LineDelta(_, iy) => {
            _model.wtrans.scale = (_model.wtrans.scale + iy * 0.05).min(2.).max(0.5);
        },
        _ => {
            println!("Mouse scroll data returned type: PixelDelta, which is not implemented.");
        }
    }
}
