use nannou::prelude::*;

pub struct Chain {
    pub points: Vec<Point2>,
    pub tl: Point2,
    pub br: Point2
}

impl Chain {
    pub fn new() -> Chain {
        Chain {points: Vec::new(), tl: pt2(1e6, 1e6), br: pt2(-1e6, -1e6)}
    }

    pub fn intersect(&self, stl: &Point2, sbr: &Point2) -> bool {
        !((sbr.x <= self.tl.x) || (sbr.y <= self.tl.y) || (stl.x >= self.br.x) || (stl.y >= self.br.y))
    }

    pub fn push(&mut self, new_p: Point2) {
        self.update_bounds(new_p);
        self.points.push(new_p);
    }

    pub fn batch_remove(&mut self, rm: &[usize]) -> bool {
        for id in rm.iter().rev() {
            self.points.remove(*id);
        }
        if self.points.is_empty() {
            return false;
        }
        self.tl = pt2(1e6, 1e6);
        self.br = pt2(-1e6, -1e6);
        for i in 0..self.points.len() {
            self.update_bounds(self.points[i]);
        }
        return true;
    }

    pub fn translate(&mut self, t: &Point2, ids: &[usize]) {
        for id in ids.iter() {
            self.points[*id] += *t;
        }
        self.tl += *t;
        self.br += *t;
    }

    pub fn copy(&mut self, ids: &[usize]) -> Chain {
        let mut new_ch = Chain::new();
        for id in ids.iter() {
            new_ch.push(self.points[*id]);
        } 
        new_ch
    }

    #[inline(always)]
    fn update_bounds(&mut self, new_p: Point2) {
        if new_p.x < self.tl.x {
            self.tl.x = new_p.x;
        }
        if new_p.x > self.br.x {
            self.br.x = new_p.x;
        }
        if new_p.y < self.tl.y {
            self.tl.y = new_p.y;
        }
        if new_p.y > self.br.y {
            self.br.y = new_p.y;
        }
    }
}

