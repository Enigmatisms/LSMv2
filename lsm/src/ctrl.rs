use nannou::prelude::*;
use crate::viz::Model;

pub fn key_pressed(_app: &App, _model: &mut Model, _key: Key) {
    match _key {
        Key::W => {_model.velo.y = _model.velo_max.x;},
        Key::A => {_model.velo.x = -_model.velo_max.x;},
        Key::S => {_model.velo.y = -_model.velo_max.x;},
        Key::D => {_model.velo.x = _model.velo_max.x;},
        Key::Escape => {
            (_model.wctrl.exit_func)(_app);
        },
        _ => {},
    }
}

pub fn key_released(_app: &App, _model: &mut Model, _key: Key) {
    match _key {
        Key::W => {_model.velo.y = 0.0;},
        Key::A => {_model.velo.x = 0.0;},
        Key::S => {_model.velo.y = 0.0;},
        Key::D => {_model.velo.x = 0.0;},
        _ => {},
    }
}

// initial position selection
pub fn mouse_pressed(_app: &App, _model: &mut Model, _button: MouseButton) {
    if _model.initialized == false {
        match _button {
            MouseButton::Left => {
                _model.pose.x = _model.mouse_pos.x;
                _model.pose.y = _model.mouse_pos.y;
            },
            _ => {},
        }
    }
}

// mouse release will determine the initial angle
pub fn mouse_released(_app: &App, _model: &mut Model, _button: MouseButton) {
    if _model.initialized == false {
        match _button {
            MouseButton::Left => {
                let dir = _model.mouse_pos - pt2(_model.pose.x, _model.pose.y);
                _model.pose.z = dir.y.atan2(dir.x);
                _model.initialized = true; 
            },
            _ => {},
        }
    }
}

// pid angle control
pub fn mouse_moved(_app: &App, _model: &mut Model, _pos: Point2) {
    _model.mouse_pos = _pos;
}

// change velocity
pub fn mouse_wheel(_app: &App, _model: &mut Model, _dt: MouseScrollDelta, _phase: TouchPhase) {}
