
type Color3 = (f32, f32, f32);
type Color4 = (f32, f32, f32, f32);
pub struct EditorColor {
    pub bg_color: Color3,
    pub traj_color: Color3,
    pub grid_color: Color3,
    pub selected_pt: Color4,
    pub unfinished_pt: Color4,
    pub finished_pt: Color4,
    pub line_color: Color4,
    pub shape_color: Color4,
    pub select_box: Color4,
    pub night_mode: bool
}

impl EditorColor {
    pub fn new() -> EditorColor{
        EditorColor {
            traj_color: (0., 1., 0.),
            bg_color: (0., 0., 0.),
            grid_color: (1., 1., 1.),
            selected_pt: (1.000000, 0.094118, 0.094118, 1.0),
            unfinished_pt: (0.301961, 0.298039, 0.490196, 0.8),
            finished_pt: (0.301961, 0.298039, 0.490196, 0.8),
            line_color: (0.913725, 0.835294, 0.792157, 0.9),
            shape_color: (0.803922, 0.760784, 0.682353, 1.0),
            select_box: (0.129412, 0.333333, 0.803922, 0.1),
            night_mode: true
        }
    }

    pub fn switch_mode(&mut self){
        if self.night_mode == true {
            self.traj_color = (0., 1., 0.);
            self.bg_color = (0., 0., 0.);
            self.grid_color = (1., 1., 1.);

            self.selected_pt = (1.000000, 0.094118, 0.094118, 1.0);
            self.unfinished_pt = (0.301961, 0.298039, 0.490196, 0.8);
            self.finished_pt = (0.301961, 0.298039, 0.490196, 0.8);
            self.line_color = (0.913725, 0.835294, 0.792157, 0.9);
            self.shape_color = (0.803922, 0.760784, 0.682353, 1.0);
            self.select_box = (0.129412, 0.333333, 0.803922, 0.1);
        } else {
            self.traj_color = (0., 0.5, 0.);
            self.bg_color = (1., 1., 1.);
            self.grid_color = (0., 0., 0.);

            self.selected_pt = (0.700000, 0.000000, 0.000000, 1.0);
            self.unfinished_pt = (0.160784, 0.203922, 0.384314, 0.8);
            self.finished_pt = (0.566667, 0.543137, 0.472549, 0.9);
            self.line_color = (0.082353, 0.074510, 0.235294, 0.9);
            self.shape_color = (0.058824, 0.054902, 0.054902, 1.0);
            self.select_box = (0.129412, 0.333333, 0.803922, 0.3);
        }
    }
}