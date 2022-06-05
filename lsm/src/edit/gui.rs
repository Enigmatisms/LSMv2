use nannou::prelude::*;
use nannou_egui::{self, egui};

use super::model::Model;
use super::ctrl::clear_offset;
use super::mesh::from_raw_points;
use crate::utils::plot::take_snapshot;
use crate::utils::map_io::{save_to_file, load_traj_file, load_map_file};

static SAVED_STRING: &str = ">>> Map file saved <<<";
static SNAPSHOT_STRING: &str = ">>> Screenshot saved <<<";

pub fn update_gui(app: &App, model: &mut Model, update: &Update) {
    let Model {
        ref mut map_points,
        ref mut wctrl,
        ref mut saved_file_name,
        ref mut plot_config,
        ref mut wtrans,
        ref mut egui,
        ref mut trajectory,
        ref mut scrn_mov,
        ref mut obj_mov,
        ref mut egui_rect,
        ref mut timer_event,
        ref mut color,
        ref mut key_stat,
        ..
    } = *model;
    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();
    let window = egui::Window::new("Editor configuration");
    let window = window.open(&mut wctrl.gui_visible);
    window.show(&ctx, |ui| {
        *egui_rect = ui.clip_rect();
        egui::Grid::new("switch_grid")
            .num_columns(4)
            .spacing([12.0, 5.0])
            .striped(true)
        .show(ui, |ui| {
            let mut activity_changed = false;
            ui.label("Move screen");
            activity_changed |= ui.add(toggle(scrn_mov)).changed();
            
            ui.label("Move point");
            activity_changed |= ui.add(toggle(obj_mov)).changed();
            ui.end_row();

            ui.label("Draw grid");
            ui.add(toggle(&mut plot_config.draw_grid));

            ui.label("Show trajectory");
            ui.add(toggle(&mut trajectory.is_visible));
            ui.end_row();

            ui.label("Night mode");
            if ui.add(toggle(&mut color.night_mode)).changed() {
                color.switch_mode();
            }

            ui.label("Ctrl pressed");
            ui.add(toggle(&mut key_stat.ctrl_pressed));
            ui.end_row();

            if activity_changed == true {
                update_status(scrn_mov, obj_mov);
            }
        });

        egui::Grid::new("slide_bars")
            .num_columns(2)
            .spacing([24.0, 5.0])
            .striped(true)
        .show(ui, |ui| {
            ui.label("Grid size");
            ui.add(egui::Slider::new(&mut plot_config.grid_step, 20.0..=200.0));
            ui.end_row();

            ui.label("Grid alpha");
            ui.add(egui::Slider::new(&mut plot_config.grid_alpha, 0.001..=0.05));
            ui.end_row();

            ui.label("Canvas scale");
            ui.add(egui::Slider::new(&mut wtrans.scale, 0.5..=2.0));
            ui.end_row();

            ui.label("Trajectory alpha");
            ui.add(egui::Slider::new(&mut trajectory.alpha, 0.001..=1.0));
            ui.end_row();
            
            
        // });
        // egui::Grid::new("buttons")
        //     .num_columns(2)
        //     .spacing([24.0, 5.0])
        // .show(ui, |ui| {
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                if ui.button("Centering").clicked() {
                    clear_offset(wtrans);
                }
            }); 
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                if ui.button("Take screenshot").clicked() {
                    take_snapshot(&app.main_window());
                    timer_event.activate(String::from("..."));
                    timer_event.item = String::from(SNAPSHOT_STRING);
                }
            });
            ui.end_row();
            
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                if ui.button("Save map").clicked() {
                    *saved_file_name = save_to_file(map_points, saved_file_name);
                    timer_event.activate(String::from("..."));
                    timer_event.item = String::from(SAVED_STRING);
                }
            });
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                if ui.button("Save as...").clicked() {
                    *saved_file_name = save_to_file(map_points, &String::from(""));
                    timer_event.activate(String::from("..."));
                    timer_event.item = String::from(SAVED_STRING);
                }
            });
            ui.end_row();

            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                if ui.button("Load map").clicked() {
                    let mut raw_points: Vec<Vec<Point2>> = Vec::new();
                    load_map_file(&mut raw_points);
                    *map_points = from_raw_points(&raw_points);
                }
            });
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                if ui.button("Load trajectory").clicked() {
                    load_traj_file(&mut trajectory.traj);
                }
            });
            ui.end_row();
        });
        ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
            ui.label(timer_event.item.as_str());
        });
    });
}



fn update_status(scrn_mov: &mut bool, obj_mov: &mut bool) {
    *obj_mov &= !*scrn_mov;
    *scrn_mov &= !*obj_mov;
}

// use egui;
/// example from https://github.com/emilk/egui
/// I modified the code so it doesn't depend egui but nannou_egui,
/// The main difference is ``` rust
///  ui.visible() && rect.intersects(ui.clip_rect())        // in this repo
///  ui.is_rect_visible()                                   // in the original repo
/// ```
/// ## Example:
/// ``` rust
/// toggle_ui(ui, &mut my_bool);
/// ```
pub fn toggle_ui(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(2.0, 1.0);
    let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed(); // report back that the value changed
    }

    response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, *on, ""));
    if ui.visible() && rect.intersects(ui.clip_rect()) {
        let how_on = ui.ctx().animate_bool(response.id, *on);
        let visuals = ui.style().interact_selectable(&response, *on);
        let rect = rect.expand(visuals.expansion);
        let radius = 0.5 * rect.height();
        ui.painter()
            .rect(rect, radius, visuals.bg_fill, visuals.bg_stroke);
        let circle_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
        let center = egui::pos2(circle_x, rect.center().y);
        ui.painter()
            .circle(center, 0.75 * radius, visuals.bg_fill, visuals.fg_stroke);
    }
    response
}

#[allow(dead_code)]
fn toggle_ui_compact(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(2.0, 1.0);
    let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, *on, ""));

    if ui.visible() && rect.intersects(ui.clip_rect()) {
        let how_on = ui.ctx().animate_bool(response.id, *on);
        let visuals = ui.style().interact_selectable(&response, *on);
        let rect = rect.expand(visuals.expansion);
        let radius = 0.5 * rect.height();
        ui.painter()
            .rect(rect, radius, visuals.bg_fill, visuals.bg_stroke);
        let circle_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
        let center = egui::pos2(circle_x, rect.center().y);
        ui.painter()
            .circle(center, 0.75 * radius, visuals.bg_fill, visuals.fg_stroke);
    }
    response
}

// A wrapper that allows the more idiomatic usage pattern: `ui.add(toggle(&mut my_bool))`
/// iOS-style toggle switch.
///
/// ## Example:
/// ``` ignore
/// ui.add(toggle(&mut my_bool));
/// ```
pub fn toggle(on: &mut bool) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| toggle_ui(ui, on)
}
