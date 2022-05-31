use nannou::prelude::*;
use super::gui;
use super::ctrl;
use super::model::Model;
use super::mesh::screen_bounds;

use crate::utils::map_io;
use crate::utils::plot;

pub fn model(app: &App) -> Model {
    let config: map_io::Config = map_io::read_config("../config/editor_config.json");

    let window_id = app
        .new_window()
        .event(event)
        .key_pressed(ctrl::key_pressed)
        .key_released(ctrl::key_released)
        .raw_event(raw_window_event)
        .mouse_moved(ctrl::mouse_moved)
        .mouse_pressed(ctrl::mouse_pressed)
        .mouse_released(ctrl::mouse_released)
        .mouse_wheel(ctrl::mouse_wheel)
        .size(config.screen.width, config.screen.height)
        .view(view)
        .build()
        .unwrap();

    app.set_exit_on_escape(false);
    Model::new(app, window_id, &config)
}

pub fn update(_app: &App, _model: &mut Model, _update: Update) {
    gui::update_gui(_app, _model, &_update);
    if _model.timer_event.is_recent_saved() == true {
        if _model.timer_event.time_elapsed() > 3. {
            _model.timer_event.save_expire();
        }
    }
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn event(_app: &App, _model: &mut Model, _event: WindowEvent) {}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = plot::window_transform(app.draw(), &model.wtrans);

    if model.plot_config.draw_grid == true {
        let win = app.main_window().rect();
        let bounds = screen_bounds(&model.map_points, &win, model.plot_config.grid_step);
        plot::draw_grid(&draw, &bounds, model.plot_config.grid_step, 1.0, model.plot_config.grid_alpha);
        plot::draw_grid(&draw, &bounds, model.plot_config.grid_step / 5., 0.5, model.plot_config.grid_alpha);
    }

    draw.background().rgb(model.color.bg_color.0, model.color.bg_color.1, model.color.bg_color.2);
    plot_unfinished(&draw, model);
    plot_finished(&draw, model);
    draw_selected_points(&draw, model);

    if model.select.key_pressed {
        let center = (model.select.bl + model.select.tr) / 2.;
        let half_diff = (model.select.tr - model.select.bl).abs();
        draw.rect()
            .x_y(center.x, center.y)
            .w_h(half_diff.x, half_diff.y)
            .rgba(model.color.select_box.0, model.color.select_box.1, model.color.select_box.2, model.color.select_box.3);
    }

    if model.trajectory.is_visible && (model.trajectory.traj.is_empty() == false) {
        plot::plot_trajectory(&draw, &model.trajectory.traj, &model.color.traj_color, model.trajectory.alpha);
    }
    if model.key_stat.ctrl_pressed == true {
        plot::draw_frame(app, &draw, &model.wtrans);
    }

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}

fn plot_unfinished(draw: &Draw,  model: &Model) {
    let pt_arr = model.map_points.last().unwrap();
    let pt2draw = (0..pt_arr.len()).map(|i| {pt_arr.points[i]});
    draw.polyline()
        .weight(2.0)
        .points(pt2draw)
        .rgba(model.color.line_color.0, model.color.line_color.1, model.color.line_color.2, model.color.line_color.3);
   plot::draw_points(draw, &pt_arr.points, &model.color.unfinished_pt);
}

fn plot_finished(draw: &Draw,  model: &Model) {
    let max_size = model.map_points.len() - 1;
    for i in 0..max_size {
        let pt_arr = &model.map_points[i];
        let pt2draw = (0..pt_arr.len()).map(|i| {pt_arr.points[i]});
        draw.polygon()
            .points(pt2draw)
            .rgba(model.color.shape_color.0, model.color.shape_color.1, model.color.shape_color.2, model.color.shape_color.3);
        plot::draw_points(draw, &pt_arr.points, &model.color.finished_pt);
    }
}

fn draw_selected_points(draw: &Draw, model: &Model) {
    for ids in model.select.selected.iter() {
        let mesh_id = *ids.first().unwrap();
        let mesh = &model.map_points[mesh_id].points;
        let pt_ids = &ids[1..];
        for pt_id in pt_ids {
            let pt = &mesh[*pt_id];
            draw.ellipse()
                .w_h(8., 8.)
                .x_y(pt.x, pt.y)
                .rgba(model.color.selected_pt.0, model.color.selected_pt.1, model.color.selected_pt.2, model.color.selected_pt.3);
        }
    }
}