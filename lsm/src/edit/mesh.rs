use nannou::prelude::*;

pub struct Chain {
    pub points: Vec<Point2>,
    pub bl: Point2,
    pub tr: Point2
}

impl Chain {
    pub fn new() -> Chain {
        Chain {points: Vec::new(), bl: pt2(1e6, 1e6), tr: pt2(-1e6, -1e6)}
    }

    pub fn intersect(&self, s_bl: &Point2, s_tr: &Point2) -> bool {
        !((s_tr.x <= self.bl.x) || (s_tr.y <= self.bl.y) || (s_bl.x >= self.tr.x) || (s_bl.y >= self.tr.y))
    }

    pub fn push(&mut self, new_p: Point2) {
        self.update_bounds(new_p);
        self.points.push(new_p);
    }

    pub fn batch_remove(&mut self, rm: &[usize]) -> bool {
        for id in rm.iter().rev() {
            self.points.remove(*id);
        }
        if self.points.len() < 3 {
            return false;
        }
        self.bl = pt2(1e6, 1e6);
        self.tr = pt2(-1e6, -1e6);
        for i in 0..self.points.len() {
            self.update_bounds(self.points[i]);
        }
        return true;
    }

    pub fn translate(&mut self, t: &Point2, ids: &[usize]) {
        for id in ids.iter() {
            self.points[*id] += *t;
            self.bl = self.bl.min(self.points[*id]);
            self.tr = self.tr.max(self.points[*id]);
        }
    }

    pub fn copy(&mut self, ids: &[usize]) -> Chain {
        let mut new_ch = Chain::new();
        for id in ids.iter() {
            new_ch.push(self.points[*id]);
        } 
        new_ch
    }

    
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.points.len()
    }
    
    #[inline(always)]
    fn update_bounds(&mut self, new_p: Point2) {
        self.bl = new_p.min(self.bl);
        self.tr = new_p.max(self.tr);
    }
}

pub fn from_raw_points(raw_points:&Vec<Vec<Point2>>) -> Vec<Chain> {
    let mut result: Vec<Chain> = Vec::new();
    for mesh in raw_points.iter() {
        let mut chain: Chain = Chain::new();
        for pt in mesh.iter() {
            chain.push(*pt);
        }
        result.push(chain);
    }
    result.push(Chain::new());
    result
}

pub fn screen_bounds(meshes: &Vec<Chain>, win: &Rect,  grid_size: f32) -> Rect<f32>  {
    let mut bl = pt2(1e6, 1e6);
    let mut tr = pt2(-1e6, -1e6);
    for chain in meshes.iter() {
        bl = bl.min(chain.bl);
        tr = tr.max(chain.tr);
    }
    bl -= pt2(grid_size, grid_size);
    tr += pt2(grid_size, grid_size);
    bl = bl.min(win.bottom_left());
    tr = tr.max(win.top_right());
    Rect::from_corner_points(bl.to_array(), tr.to_array())
}
