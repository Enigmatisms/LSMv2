#[inline(always)]
pub fn good_angle(angle: f32) -> f32 {
    if angle > std::f32::consts::PI {
        return angle - std::f32::consts::PI * 2.;
    } else if angle < -std::f32::consts::PI {
        return angle + std::f32::consts::PI * 2.;
    }
    angle
}
