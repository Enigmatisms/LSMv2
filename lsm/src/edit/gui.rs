use nannou::prelude::*;
use nannou_egui::{self, egui};

use super::model::{Model, DrawState};
use super::mesh::from_raw_points;
use crate::utils::toggle::toggle;
use crate::utils::plot::take_snapshot;
use crate::utils::map_io::{save_to_file, load_traj_file, load_map_file};
use crate::utils::consts::*;

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
        ref mut inside_gui,
        ref mut draw_state,
        ref mut timer_event,
        ref mut color,
        ref mut key_stat,
        ref mut add_drawer,
        ..
    } = *model;
    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();
    let window = egui::Window::new("Editor configuration").default_width(270.);
    let window = window.open(&mut wctrl.gui_visible);
    window.show(&ctx, |ui| {
        
        egui::Grid::new("switch_grid")
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

            ui.label("Trajectory");
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
            
            let mut add_drawer_toggled = false;
            ui.label("Drawing mode");
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                add_drawer_toggled |= ui.selectable_value(draw_state, DrawState::Arbitrary, "Normal").changed();
            });
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                add_drawer_toggled |= ui.selectable_value(draw_state, DrawState::Straight, "Line").changed();
            });
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                add_drawer_toggled |= ui.selectable_value(draw_state, DrawState::Rect, "Rect").changed();
            });
            if add_drawer_toggled == true {
                add_drawer.update_last(map_points);
            }
            ui.end_row();
        });

        egui::Grid::new("slide_bars")
            .striped(true)
        .show(ui, |ui| {
            ui.label("Grid size");
            ui.add(egui::Slider::new(&mut plot_config.grid_step, 20.0..=200.0));
            ui.end_row();

            ui.label("Grid alpha");
            ui.add(egui::Slider::new(&mut plot_config.grid_alpha, 0.001..=0.1));
            ui.end_row();

            
            ui.label("Trajectory alpha");
            ui.add(egui::Slider::new(&mut trajectory.alpha, 0.001..=1.0));
            ui.end_row();

            ui.label("Canvas zoom scale");
            ui.add(egui::Slider::new(&mut wtrans.scale, 0.5..=2.0));
            ui.end_row();
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Centering view").clicked() {
                    wtrans.clear_offset();
                }
            });
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Take screenshot").clicked() {
                    take_snapshot(&app.main_window());
                    timer_event.activate(String::from(NULL_STR), String::from(SNAPSHOT_STRING));
                }
            });

            ui.end_row();
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Save map file").clicked() {
                    *saved_file_name = save_to_file(map_points, saved_file_name);
                    timer_event.activate(String::from(NULL_STR), String::from(SAVED_STRING));
                }
            });
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Save as file ...").clicked() {
                    *saved_file_name = save_to_file(map_points, &String::from(""));
                    timer_event.activate(String::from(NULL_STR), String::from(SAVED_STRING));
                }
            });
            ui.end_row();

            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Load map file").clicked() {
                    let mut raw_points: Vec<Vec<Point2>> = Vec::new();
                    load_map_file(&mut raw_points);
                    *map_points = from_raw_points(&raw_points);
                }
            });
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Load trajectory").clicked() {
                    load_traj_file(&mut trajectory.traj);
                }
            });
            ui.end_row();
        });

        ui.horizontal(|ui| {
            ui.centered_and_justified(|ui| {
                ui.label(timer_event.item.as_str());
            });
        });

        *inside_gui = ui.ctx().is_pointer_over_area();
    });
}



fn update_status(scrn_mov: &mut bool, obj_mov: &mut bool) {
    *obj_mov &= !*scrn_mov;
    *scrn_mov &= !*obj_mov;
}
