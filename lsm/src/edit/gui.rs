use nannou::prelude::*;
use nannou_egui::{self, egui};
use std::io::Write;

use std::fs;
use super::model::Model;
use super::mesh::Chain;
use super::ctrl::clear_offset;

pub fn update_gui(model: &mut Model, update: &Update) {
    let Model {
        ref map_points,
        ref mut plot_config,
        ref mut wtrans,
        ref mut egui,
        ref mut scrn_mov,
        ref mut obj_mov,
        ref mut egui_rect,
        ..
    } = *model;
    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();
    egui::Window::new("Editor configuration").show(&ctx, |ui| {
        *egui_rect = ui.clip_rect();
        egui::Grid::new("my_grid")
            .num_columns(2)
            .spacing([25.0, 5.0])
            .striped(true)
        .show(ui, |ui| {
            let mut changed = false;
            ui.label("Move screen");
            changed |= ui.add(toggle(scrn_mov)).changed();
            ui.end_row();
            
            ui.label("Move point");
            changed |= ui.add(toggle(obj_mov)).changed();
            ui.end_row();

            ui.label("Draw grid");
            ui.add(toggle(&mut plot_config.draw_grid));
            ui.end_row();

            ui.label("Grid size");
            ui.add(egui::Slider::new(&mut plot_config.grid_step, 20.0..=200.0));
            ui.end_row();

            ui.label("Grid alpha");
            ui.add(egui::Slider::new(&mut plot_config.grid_alpha, 0.001..=0.05));
            ui.end_row();

            ui.label("Canvas scale");
            ui.add(egui::Slider::new(&mut wtrans.scale, 0.5..=2.0));
            ui.end_row();
            
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                if ui.button("Centering").clicked() {
                    clear_offset(wtrans);
                }
            }); 
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                if ui.button("Save to file").clicked() {
                    save_to_file(map_points);
                }
            });
            ui.end_row();
            
            if changed == true {
                update_status(scrn_mov, obj_mov);
            }
        });
    });
}

fn save_to_file(map_points: &Vec<Chain>) {
    let path = rfd::FileDialog::new()
        .set_file_name("../maps/new_map.txt")
        .set_directory(".")
        .save_file();
        
    if let Some(file) = path {
        let path_res = file.as_os_str().to_str().take().unwrap();
        let mut file = std::fs::File::create(path_res).expect("Failed to create file.");
        for chain in map_points.iter() {
            if chain.len() <= 2 {
                continue;
            }
            write!(file, "{} ", chain.len()).expect("Failed to write to file.");
            for pt in chain.points.iter() {
                write!(file, "{} {} ", pt.x, pt.y).expect("Failed to write to file.");
            }
            write!(file, "\n").expect("Failed to write to file.");
        }
    } else {
        println!("Failed to open file.");
    }
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
