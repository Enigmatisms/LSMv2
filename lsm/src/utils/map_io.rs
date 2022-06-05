use std::fs;
use serde::Deserialize;
use nannou::prelude::*;
use std::io::Write;
use std::io::{prelude::*, BufReader};
use std::f32::consts::PI;
use crate::edit::mesh::Chain;
use crate::sim::cuda_helper::{Vec2_cuda, Vec3_cuda, self};

pub type Mesh = Vec<Point2>;
pub type Meshes = Vec<Mesh>;

#[derive(Deserialize)]
pub struct LidarConfig {
    pub amin: f32,
    pub amax: f32,
    pub ainc: f32,
    pub noise_k: f32
}

#[derive(Deserialize)]
pub struct ScannerConfig {
    pub t_vel: f32,
    pub r_vel: f32,
    pub pid_kp: f32,
    pub pid_ki: f32,
    pub pid_kd: f32,
}

#[derive(Deserialize)]
pub struct ScreenConfig {
    pub width: u32,
    pub height: u32
}

#[derive(Deserialize)]
pub struct Config {
    pub lidar: LidarConfig,
    pub robot: ScannerConfig,
    pub screen: ScreenConfig,
    pub map_path: String,
    pub grid_size: f32
}

/// TODO: offset 600 and 450 needs to be removed
pub fn parse_map_file<T>(filepath: T) -> Option<Meshes> where T: AsRef<std::path::Path> {
    if let Some(all_lines) = read_lines(filepath) {
        let mut result: Meshes = Vec::new();
        for line in all_lines.iter() {
            let str_vec: Vec<&str> = line.split(" ").collect();
            let point_num =  str_vec[0].parse::<usize>().unwrap() + 1;
            let mut mesh: Mesh = Vec::new();
            for i in 1..point_num {
                let str1 = str_vec[(i << 1) - 1];
                let str2 = str_vec[i << 1];
                if str1.is_empty() == true {
                    break;
                } else {
                    mesh.push(pt2(
                        str1.parse::<f32>().unwrap() - 600.,
                        str2.parse::<f32>().unwrap() - 450.
                    ));
                }
            }
            result.push(mesh);
        }
        return Some(result);
    } else {
        return None;
    }
}

pub fn meshes_to_segments(meshes: &Meshes, segments: &mut Vec<cuda_helper::Vec2_cuda>) -> usize {
    let mut ptr: usize = 0;
    for mesh in meshes.iter() {
        let first = &mesh[0];
        segments.push(Vec2_cuda {x: first.x, y: first.y});
        for i in 1..(mesh.len()) {
            let current = &mesh[i];
            segments.push(Vec2_cuda {x: current.x, y: current.y});
            segments.push(Vec2_cuda {x: current.x, y: current.y});
        }
        segments.push(Vec2_cuda {x: first.x, y: first.y});
        ptr += mesh.len();
    }
    ptr
}

#[inline(always)]
pub fn get_ray_num(lidar_param: &Vec3_cuda) -> usize {
    (((lidar_param.y - lidar_param.x) / lidar_param.z).round() as usize) + 1
}

pub fn read_config(file_path: &str) -> Config  {
    let file: fs::File = fs::File::open(file_path).ok().unwrap();
    let reader = BufReader::new(file);
    let mut config: Config = serde_json::from_reader(reader).ok().unwrap();
    config.lidar.amin = config.lidar.amin * PI / 180.;
    config.lidar.amax = config.lidar.amax * PI / 180.;
    config.lidar.ainc = config.lidar.ainc * PI / 360.;
    config.lidar.amin += config.lidar.ainc / 2.0;
    config.lidar.amax -= config.lidar.ainc / 2.0;
    config
}

pub fn save_to_file(map_points: &Vec<Chain>, file_name: &String) -> String {
    let mut _path_res= String::new();
    if file_name.len() == 0 {
        let path = rfd::FileDialog::new()
            .set_file_name("../maps/new_map.txt")
            .set_directory(".")
            .save_file();
        if let Some(file) = path {
            _path_res = String::from(file.as_os_str().to_str().unwrap());
        } else {
            println!("Failed to open file.");
            return String::new();
        }
    } else {
        _path_res = file_name.clone();
    }
    let mut file = std::fs::File::create(_path_res.as_str()).expect("Failed to create file.");
    if map_points.len() <= 1 {
        return _path_res;
    }
    let map_size = map_points.len() - 1;
    for i in 0..map_size {          // the last one (not completed) will not be stored
        let chain = &map_points[i];
        if chain.len() <= 2 {
            continue;
        }
        write!(file, "{} ", chain.len()).expect("Failed to write to file.");
        for pt in chain.points.iter() {
            write!(file, "{} {} ", pt.x, pt.y).expect("Failed to write to file.");
        }
        write!(file, "\n").expect("Failed to write to file.");
    }
    return _path_res;
}

pub fn load_traj_file(tral_points: &mut Vec<Point2>) {
    let path = rfd::FileDialog::new()
        .set_file_name("../maps/hfps0.lgp")
        .set_directory(".")
        .pick_file();
    if let Some(path_res) = path {
        *tral_points = parse_traj_file(path_res).unwrap();
    } else {
        tral_points.clear();
    }
}

pub fn load_map_file(map_points: &mut Meshes) {
    let path = rfd::FileDialog::new()
        .set_file_name("../maps/standard0.lgp")
        .set_directory(".")
        .pick_file();
    if let Some(path_res) = path {
        *map_points = parse_map_file(path_res).unwrap();
    } else {
        map_points.clear();
    }
}

// ========== privates ==========
fn read_lines<T>(filepath: T) -> Option<Vec<String>> where T: AsRef<std::path::Path> {
    if let Ok(file) = fs::File::open(filepath) {
        let reader = BufReader::new(file);
        let mut result_vec: Vec<String> = Vec::new();
        for line in reader.lines() {
            if let Ok(line_inner) = line {
                result_vec.push(line_inner);
            } else {
                return None;
            }
        }
        return Some(result_vec);
    }
    return None;
}

/// TODO: offset 600 and 450 needs to be removed
fn parse_traj_file<T>(filepath: T) -> Option<Mesh> where T: AsRef<std::path::Path> {
    if let Some(all_lines) = read_lines(filepath) {
        let mut result: Mesh = Vec::new();
        for i in 2..all_lines.len() {
            let line = &all_lines[i];
            let str_vec: Vec<&str> = line.split(" ").collect();
            result.push(pt2(
                str_vec[1].parse::<f32>().unwrap() - 600.,
                str_vec[2].parse::<f32>().unwrap() - 450.
            ));
        }
        return Some(result);
    } else {
        return None;
    }
}
