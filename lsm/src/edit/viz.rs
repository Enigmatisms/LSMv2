use nannou::prelude::*;
use super::ctrl;
use super::model::Model;

use crate::utils::map_io;
use crate::utils::plot;


pub fn model(app: &App) -> Model {
    let config: map_io::Config = map_io::read_config("../config/editor_config.json");

    let window_id = app
        .new_window()
        .event(event)
        .key_pressed(ctrl::key_pressed)
        .key_released(ctrl::key_released)
        .mouse_moved(ctrl::mouse_moved)
        .mouse_pressed(ctrl::mouse_pressed)
        .mouse_released(ctrl::mouse_released)
        .mouse_wheel(ctrl::mouse_wheel)
        .size(config.screen.width, config.screen.height)
        .view(view)
        .build()
        .unwrap();

    app.set_exit_on_escape(false);
    let meshes: map_io::Meshes = map_io::parse_map_file(config.map_path.as_str()).unwrap();
    Model::new(window_id, &config, meshes)
}

pub fn update(_app: &App, _model: &mut Model, _: Update) {
    
}

pub fn event(_app: &App, _model: &mut Model, _event: WindowEvent) {}

pub fn view(app: &App, model: &Model, frame: Frame) {
    let draw = plot::window_transform(app.draw(), &model.wtrans);

    if model.plot_config.draw_grid == true {
        let win = app.main_window().rect();
        plot::draw_grid(&draw, &win, model.plot_config.grid_step, 1.0, 0.01);
        plot::draw_grid(&draw, &win, model.plot_config.grid_step / 5., 0.5, 0.01);
    }

    draw.background().color(BLACK);
    for mesh in model.map_points.iter() {
        let points = (0..mesh.len()).map(|i| {
            mesh[i]
        });
        draw.polygon()
            .color(WHITE)
            .points(points);
    }

    draw.to_frame(app, &frame).unwrap();
}
