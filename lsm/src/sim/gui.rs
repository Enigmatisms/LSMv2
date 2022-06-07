use nannou::prelude::*;
use nannou_egui::{self, egui};

use super::model::Model;
use crate::utils::toggle::toggle;
use crate::utils::plot::take_snapshot;
use crate::utils::map_io::load_map_file;

pub fn update_gui(app: &App, model: &mut Model, update: &Update) {
    let Model {
        ref mut map_points,
        ref mut wctrl,
        ref mut plot_config,
        ref mut wtrans,
        ref mut egui,
        ref mut egui_rect,
        ref mut key_stat,
        ref mut color,
        ref mut trajectory,
        ref mut timer_event,
        ..
    } = *model;
    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();
    let window = egui::Window::new("Editor configuration").default_width(270.);
    let window = window.open(&mut wctrl.gui_visible);
    window.show(&ctx, |ui| {
        *egui_rect = ui.clip_rect();
        
        egui::Grid::new("switch_grid")
            .striped(true)
        .show(ui, |ui| {
            ui.label("Trajectory");
            ui.add(toggle(&mut trajectory.is_visible));
            ui.label("Ctrl pressed");
            ui.add(toggle(&mut key_stat.ctrl_pressed));
            ui.end_row();

            ui.label("Draw grid");
            ui.add(toggle(&mut plot_config.draw_grid));
            ui.label("Night mode");
            if ui.add(toggle(&mut color.night_mode)).changed() {
                color.switch_mode();
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
                    clear_offset(wtrans);
                }
            });
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Take screenshot").clicked() {
                    take_snapshot(&app.main_window());
                    timer_event.activate(String::from("..."));
                    timer_event.item = String::from(SNAPSHOT_STRING);
                }
            });

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
    });
}
