use nannou::prelude::*;
use super::model::Model;
use super::mesh::Chain;
use crate::utils::utils;
use crate::utils::plot;
use crate::utils::map_io::save_to_file;
use crate::utils::structs::WindowTransform;

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
        Key::LControl => {
            _model.key_stat.ctrl_pressed = true;
        },
        Key::S => {
            if _model.key_stat.ctrl_pressed == true {
                _model.saved_file_name = save_to_file(&_model.map_points, &_model.saved_file_name);
                _model.timer_event.activate(String::from("..."));
                _model.timer_event.item = String::from(">>> Map file saved <<<");
            }
        }
        Key::P => {
            if _model.key_stat.ctrl_pressed == true {
                plot::take_snapshot(&_app.main_window());
                _model.timer_event.activate(String::from("..."));
                _model.timer_event.item = String::from(">>> Screenshot saved <<<");
            }
        },
        _ => {},
    }
}

pub fn key_released(_app: &App, _model: &mut Model, _key: Key) {
    match _key {
        Key::M => {                     // exiting vertex moving
            _model.obj_mov = false;
        },
        Key::P => {                     // switch grid drawing mode
            if  _model.key_stat.ctrl_pressed == false {
                _model.plot_config.draw_grid = !_model.plot_config.draw_grid;
            } 
        },
        Key::H => {                     // Set screen offsets to zero
            clear_offset(&mut _model.wtrans);
        },
        Key::V => {
            _model.wctrl.switch_gui_visibility();
        }
        Key::Return => {                // push valid point vector
            let last_vec = _model.map_points.last().unwrap();
            if last_vec.len() > 2 {
                _model.map_points.push(Chain::new());
            }
        },
        Key::LShift => {                // shifting screen
            _model.scrn_mov = false;
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
            if _model.map_points.is_empty() {
                _model.map_points.push(Chain::new());
            }
            _model.select.selected.clear();
        },
        Key::Escape => {
            (_model.wctrl.exit_func)(_app);
        }
        Key::LControl => {
            _model.key_stat.ctrl_pressed = false;
        }
        _ => {},
    }
}

// initial position selection
pub fn mouse_pressed(_app: &App, _model: &mut Model, _button: MouseButton) {
    let point = _app.mouse.position();
    if _model.cursor_in_gui(&_app.main_window().rect().w_h(), &point) {
        return;
    }
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
        if _model.select.selected.is_empty() == false {
            _model.select.selected.clear();
            return;
        }
        match _button {
            MouseButton::Left => {
                // 增加新的点
                let last_vec = _model.map_points.last_mut().unwrap();
                last_vec.push(plot::localized_position(&point, &_model.wtrans));
            },
            MouseButton::Right => {
                _model.select.key_pressed = true;
                _model.select.bl = plot::localized_position(&point, &_model.wtrans);
                _model.select.tr = _model.select.bl;
            },
            _ => {}
        }
    }
}

// mouse release will determine the initial angle
pub fn mouse_released(_app: &App, _model: &mut Model, _button: MouseButton) {
    let now_pos = _app.mouse.position();
    if _model.scrn_mov == true {                    // 画幅移动---视角变换模式
        match _button {
            MouseButton::Middle => {
                // _model.wtrans.t += now_pos - _model.wtrans.t_start;
                _model.wtrans.t_set = true;
            },
            MouseButton::Left => {
                _model.wtrans.r_set = true;
            },
            MouseButton::Right => {
                _model.wtrans.t = pt2(0., 0.);
                _model.wtrans.rot = 0.;
            },
            _ => {}
        }
    } else if _model.obj_mov == true {              // 选中物体移动模式
        if _button == MouseButton::Left {
            _model.mouse_moving_object = false;
        }
    } else {
        if _model.select.key_pressed == true {      // 框选完成，计算选框以及所有被选中点
            calc_select_box(&now_pos, _model);
            _model.select.key_pressed = false;
            // push all the related points to _model.select
            for (chain_id, chain) in _model.map_points.iter().enumerate() {
                if chain.intersect(&_model.select.bl, &_model.select.tr) == true {
                    let mut select_ids: Vec<usize> = Vec::new();
                    select_ids.push(chain_id);
                    for (pt_id, pt) in chain.points.iter().enumerate() {
                        if point_in_box(pt, &_model.select.bl, &_model.select.tr) {
                            select_ids.push(pt_id);
                        }
                    }
                    if select_ids.len() > 1 {
                        _model.select.selected.push(select_ids);
                    }
                }
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
        _model.wtrans.t_start = point;
    } else {
        if _model.select.key_pressed == true {
            _model.select.tr = plot::localized_position(&point, &_model.wtrans);
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

pub fn clear_offset(wtrans: &mut WindowTransform) {
    wtrans.rot = 0.;
    wtrans.t = pt2(0., 0.);
}

fn calc_select_box(point: &Point2, model: &mut Model) {
    let screen_pos = plot::localized_position(point, &model.wtrans);
    let tmp_bl = screen_pos.min(model.select.bl);
    let tmp_tr = screen_pos.max(model.select.bl);
    model.select.bl = tmp_bl;
    model.select.tr = tmp_tr;
}

#[inline(always)]
fn point_in_box(pt: &Point2, bl: &Point2, tr: &Point2) -> bool {
    (pt.x > bl.x) && (pt.y > bl.y) && (pt.x < tr.x) && (pt.y < tr.y)
}
