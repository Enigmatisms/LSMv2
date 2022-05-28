use lsm::edit::viz;

/**
 * 框选操作使用栅格不合适，个人认为，每个Vec<Point2>需要维护一个BoundingBox
 * 判定选框是否在boundingBox内部：可以快速判断：
 *  选框br.x < boundingBox.tl.x, br.y < boundingBox.tl.y
 *  选框tl.x > boundingBox.br.x, br.y > boundingBox.br.y
 * 则相交的选框对应的Vec：遍历内部所有点，在选框内部则保存id
 */

fn main() {
    nannou::app(viz::model).update(viz::update).run();
}

