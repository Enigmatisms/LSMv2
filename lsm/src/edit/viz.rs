use nannou::prelude::*;
use super::gui;
use super::ctrl;
use super::model::Model;
use super::mesh::screen_bounds;

use crate::utils::map_io;
use crate::utils::plot;

static NAVY_BLUE: (f32, f32, f32, f32) = (0.301961, 0.298039, 0.490196, 0.8);
static MILKY_WHITE: (f32, f32, f32, f32) = (0.913725, 0.835294, 0.792157, 0.9);
static MY_GREY: (f32, f32, f32, f32) = (0.803922, 0.760784, 0.682353, 1.0);
static LIGHT_BLUE: (f32, f32, f32, f32) = (0.600000, 0.768627, 0.784314, 0.8);
static AQUATIC: (f32, f32, f32, f32) = (0.129412, 0.333333, 0.803922, 0.1);
static GEN_RED: (f32, f32, f32, f32) = (1.000000, 0.094118, 0.094118, 1.0);

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
    gui::update_gui(_model, &_update);
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

    draw.background().color(BLACK);
    plot_unfinished(&draw, model);
    plot_finished(&draw, model);
    draw_selected_points(&draw, model);

    if model.select.key_pressed {
        let center = (model.select.bl + model.select.tr) / 2.;
        let half_diff = (model.select.tr - model.select.bl).abs();
        draw.rect()
            .x_y(center.x, center.y)
            .w_h(half_diff.x, half_diff.y)
            .rgba(AQUATIC.0, AQUATIC.1, AQUATIC.2, AQUATIC.3);
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
        .rgba(MILKY_WHITE.0, MILKY_WHITE.1, MILKY_WHITE.2, MILKY_WHITE.3);
    draw_points(draw, &pt_arr.points, &LIGHT_BLUE);
}

fn plot_finished(draw: &Draw,  model: &Model) {
    let max_size = model.map_points.len() - 1;
    for i in 0..max_size {
        let pt_arr = &model.map_points[i];
        let pt2draw = (0..pt_arr.len()).map(|i| {pt_arr.points[i]});
        draw.polygon()
            .points(pt2draw)
            .rgba(MY_GREY.0, MY_GREY.1, MY_GREY.2, MY_GREY.3);
        draw_points(draw, &pt_arr.points, &NAVY_BLUE);
    }
}

fn draw_points(draw: &Draw, pts: &Vec<Point2>, rgba: &(f32, f32, f32, f32)) {
    for pt in pts.iter() {
        draw.ellipse()
            .w_h(5., 5.)
            .x_y(pt.x, pt.y)
            .rgba(rgba.0, rgba.1, rgba.2, rgba.3);
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
                .rgba(GEN_RED.0, GEN_RED.1, GEN_RED.2, GEN_RED.3);
        }
    }
}