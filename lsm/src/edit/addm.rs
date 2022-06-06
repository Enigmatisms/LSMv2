use nannou::prelude::*;
use super::mesh::Chain;
use crate::utils::structs::WindowTransform;
use crate::utils::plot::{to_screen, localized_position};

const PI_4: f32 = std::f32::consts::PI / 4.;
const PI_34: f32 = PI_4 * 3.;

const MOVEMENT: [(f32, f32); 4] = [(1., 0.), (0., 1.0), (-1.0, 0.), (0., -1.)];

pub struct AdditionalMode {
    last_p: Point2,
    last_set: bool,
    display_pts: Vec<Point2>
}

impl AdditionalMode {
    pub fn new() -> AdditionalMode {
        AdditionalMode{last_p: pt2(0., 0.), last_set: false, display_pts: Vec::new()}
    }

    pub fn get_points2draw(&self) -> impl Iterator<Item = Point2> + '_ {
        (0..self.display_pts.len()).map(|i| {self.display_pts[i]})
    }

    #[inline(always)]
    pub fn is_last_set(&self) -> bool {
        self.last_set
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.display_pts.len()
    }

    #[inline(always)]
    pub fn clear_last_flag(&mut self) {
        self.last_set = false;
    }

    #[inline(always)]
    pub fn iter(&self) -> core::slice::Iter<Point2>{
        self.display_pts.iter()
    }

    pub fn update_last(&mut self, map_points: &Vec<Chain>) {
        if map_points.len() == 0 {
            self.last_set = false;
            return;
        } else {
            if let Some(&inner) = map_points.last().as_ref() {
                if inner.len() == 0 {
                    self.last_set = false;
                    return;
                }
                let last_p_ref = inner.points.last().unwrap();
                self.last_p.x = last_p_ref.x;
                self.last_p.y = last_p_ref.y;
                self.last_set = true;
            }
        }
    }

    // input cur_p is current screen space point, output canvas frame point
    pub fn straight_point(&mut self, cur_p: &Point2, wtrans: &WindowTransform) {
        if self.last_set == false {return;}
        let last_scrn_p = to_screen(&self.last_p, wtrans);
        let delta_p = *cur_p - last_scrn_p;
        let axis_angle = delta_p.y.atan2(delta_p.x).abs();
        let mut output_pt = *cur_p;
        if axis_angle >= PI_4 && axis_angle < PI_34 {
            output_pt.x = last_scrn_p.x;
        } else {
            output_pt.y = last_scrn_p.y;
        }
        self.display_pts.clear();
        self.display_pts.push(self.last_p);
        self.display_pts.push(localized_position(&output_pt, wtrans));
    }

    // input cur_p is current screen space point, output canvas frame point
    pub fn get_rectangle(&mut self, cur_p: &Point2, wtrans: &WindowTransform) {
        if self.last_set == false {return;}
        let last_scrn_p = to_screen(&self.last_p, wtrans);
        let w_h = *cur_p - last_scrn_p;
        let mut iterator: Box<dyn Iterator<Item = &(f32, f32)>> = Box::new(MOVEMENT.iter().rev().cycle().skip(2));
        if ((w_h.x.to_bits() ^ w_h.y.to_bits()) & 0x80000000) == 0x0 {        // w_h.x and w_h.y is of different sign
            iterator = Box::new(MOVEMENT.iter());
        }
        self.display_pts.clear();
        self.display_pts.push(self.last_p);
        let mut point = last_scrn_p;
        for _ in 0..3 {
            let (dx, dy) = *iterator.next().unwrap();
            point += pt2(w_h.x * dx, w_h.y * dy);
            self.display_pts.push(localized_position(&point, wtrans));
        }
    }
}