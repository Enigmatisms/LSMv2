// use 
use lsm::sim_viz;

fn main() {
    nannou::app(sim_viz::model).update(sim_viz::update).run();
}
