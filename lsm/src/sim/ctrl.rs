use nannou::prelude::*;
use super::model::Model;
use crate::utils::utils;
use crate::utils::plot;
use crate::utils::consts::*;

pub fn key_pressed(_app: &App, _model: &mut Model, _key: Key) {
    match _key {
        Key::W => {_model.velo.x = _model.velo_max.x;},
        Key::A => {_model.velo.y = _model.velo_max.x;},
        Key::S => {
            if _model.key_stat.ctrl_pressed == false {
                _model.velo.x = -_model.velo_max.x;
            } else {
                let config_to_save = Model::output_config(
                    &_model.lidar_param, &_model.velo_max, &_model.pid, &_model.str_config.map_name, &_model.color.lidar_color, 
                    &_model.lidar_noise, &_model.wctrl.win_w, &_model.wctrl.win_h, &_model.grid_size
                );
                config_to_save.write_to_file(&mut _model.str_config.config_output);
                _model.timer_event.activate(String::from(NULL_STR), String::from(CONFIG_SAVE_STRING));
            }
        },
        Key::D => {_model.velo.y = -_model.velo_max.x;},
        Key::Escape => {
            (_model.wctrl.exit_func)(_app);
        },
        Key::LControl => {_model.key_stat.ctrl_pressed = true;}
        Key::P => {
            if _model.key_stat.ctrl_pressed == true {
                plot::take_snapshot(&_app.main_window());
                _model.timer_event.activate(String::from(NULL_STR), String::from(SNAPSHOT_STRING));
            }
        },
        _ => {},
    }
}

pub fn key_released(_app: &App, _model: &mut Model, _key: Key) {
    match _key {
        Key::W => {_model.velo.x = 0.0;},
        Key::A => {_model.velo.y = 0.0;},
        Key::S => {_model.velo.x = 0.0;},
        Key::D => {_model.velo.y = 0.0;},
        Key::P => {
            if _model.key_stat.ctrl_pressed == false {
                _model.plot_config.draw_grid = !_model.plot_config.draw_grid;
            }
        },
        Key::LControl => {_model.key_stat.ctrl_pressed = false;},
        _ => {},
    }
}

// initial position selection
pub fn mouse_pressed(_app: &App, _model: &mut Model, _button: MouseButton) {
    let point = _app.mouse.position();
    if _model.inside_gui == true {
        return;
    }
    if _model.initialized == false {
        match _button {
            MouseButton::Left => {
                _model.pose.x = point.x;
                _model.pose.y = point.y;
            },
            _ => {},
        }
    } else {
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
    }
}

// mouse release will determine the initial angle
pub fn mouse_released(_app: &App, _model: &mut Model, _button: MouseButton) {
    let now_pos = _app.mouse.position();
    if _model.inside_gui == true {
        return;
    }
    if _model.initialized == false {
        match _button {
            MouseButton::Left => {
                let dir = now_pos - pt2(_model.pose.x, _model.pose.y);
                _model.pose.z = dir.y.atan2(dir.x);
                _model.initialized = true; 
            },
            _ => {},
        }
    } else {
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
    }
}

// pid angle control
pub fn mouse_moved(_app: &App, _model: &mut Model, _pos: Point2) {
    if _model.wtrans.t_set == false {
        _model.wtrans.t += _pos - _model.wtrans.t_start;
        _model.wtrans.t_start = _pos;
    }
    if _model.wtrans.r_set == false {
        let current_angle = _pos.y.atan2(_pos.x);
        let delta_angle = utils::good_angle(current_angle - _model.wtrans.rot_start);
        _model.wtrans.rot_start = current_angle;
        _model.wtrans.rot = utils::good_angle(delta_angle + _model.wtrans.rot);
        _model.wtrans.t = utils::get_rotation(&delta_angle).mul_vec2(_model.wtrans.t);
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
