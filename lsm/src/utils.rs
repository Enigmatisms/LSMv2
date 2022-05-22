use nannou::glam::Mat2;

#[inline(always)]
pub fn good_angle(angle: f32) -> f32 {
    if angle > std::f32::consts::PI {
        return angle - std::f32::consts::PI * 2.;
    } else if angle < -std::f32::consts::PI {
        return angle + std::f32::consts::PI * 2.;
    }
    angle
}

#[inline(always)]
pub fn get_rotation(angle: &f32) -> Mat2 {
    let cosa = angle.cos();
    let sina = angle.sin();
    Mat2::from_cols_array(&[cosa, sina, -sina, cosa])
}

