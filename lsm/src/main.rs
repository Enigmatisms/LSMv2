mod cuda_helper;
mod map_io;
mod viz;
mod ctrl;
mod utils;
mod model;
mod grid;

fn main() {
    nannou::app(viz::model).update(viz::update).run();
}
