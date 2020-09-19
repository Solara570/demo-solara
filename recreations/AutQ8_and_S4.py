from big_ol_pile_of_manim_imports import *

class AutQ8andS4CommonScene(ThreeDScene):
    CONFIG = {
        "default_camera_orientation" : {
            "phi" : 60 * DEGREES,
            "theta" : -60 * DEGREES,
            "distance" : 50,
        },
        "sep_line_center" : RIGHT_SIDE/2,
        "ref_line_center" : LEFT_SIDE/2,
        "colors" : [PINK, BLUE, RED, MAROON_B, GREEN, TEAL],
        "directions" : [IN, OUT, LEFT, RIGHT, UP, DOWN],
        "max_norm" : 3.5,
        "rot_axes" : [UP, LEFT, UR, OUT+DR, IN+LEFT, DR],
        "rot_angles" : [PI/2, PI, PI, PI*2/3, PI, PI],
    }
    def setup(self):
        sep_line = Line(TOP, BOTTOM, stroke_width = 5, stroke_color = WHITE)
        sep_line.move_to(self.sep_line_center)
        title = TexMobject("\\mathrm{Aut}(Q_8)", "\\cong", "\\text{正方体旋转群}")
        title.next_to(sep_line.get_top(), DOWN, index_of_submobject_to_align = 1)
        sur_rect = SurroundingRectangle(
            title, fill_opacity = 1, fill_color = BLACK, stroke_width = 5, stroke_color = WHITE, buff = 0.25,
        )
        ref_line = Line(TOP, BOTTOM, stroke_width = 1, stroke_color = GREY)
        ref_line.move_to(self.ref_line_center)
        self.add_fixed_in_frame_mobjects(sep_line, sur_rect, title)
        self.add_fixed_in_frame_mobjects(ref_line)


class Q8Automorphism3DPart(AutQ8andS4CommonScene):
    CONFIG = {
        "sphere_radius" : 2,
        "line_config" : {
            "stroke_width" : 2,
            "stroke_color" : GREY,
            "shade_in_3d" : True,
        },
        "axis_config" : {
            "stroke_width" : 5,
            "stroke_color" : YELLOW,
            "shade_in_3d" : True,
        },
        "text_buff_factor" : 1.3,
    }
    def construct(self):
        self.set_camera_orientation(**self.default_camera_orientation)
        sphere = Sphere(
            radius = self.sphere_radius,
            resolution = (24, 48), stroke_width = 1,
            fill_opacity = 0.4,
            stroke_opacity = 0.5,
        )
        sphere.set_color_by_gradient(BLUE, GREY, PINK)
        inner_lines = ThreeDVMobject(*[
            Line(self.sphere_radius*direction, -self.sphere_radius*direction, **self.line_config)
            for direction in [RIGHT, UP, OUT]
        ])
        outer_lines = ThreeDVMobject(*[
            Line(self.sphere_radius*direction, self.max_norm*direction, **self.line_config)
            for direction in [RIGHT, LEFT, UP, DOWN, OUT, IN]
        ])
        negk_dot, k_dot, negi_dot, i_dot, j_dot, negj_dot = dots = VGroup(*[
            Sphere(radius = 0.08).set_color(color).move_to(self.sphere_radius * direction)
            for color, direction in zip(self.colors, self.directions)
        ])
        negk, k, negi, i, j, negj = texts = ThreeDVMobject(*[
            SingleStringTexMobject(char, color = color, shade_in_3d = True)
            for char, color in zip(["-k", "k", "-i", "i", "j", "-j"], self.colors)
        ])
        one_dot = Sphere(radius = 0.08).set_color(GREY).move_to(ORIGIN)
        one = ThreeDVMobject(TexMobject("1", color = GREY, shade_in_3d = True))
        one.next_to(one_dot, IN)
        for text in texts.submobjects+[one]:
            text.rotate(PI/2, axis = RIGHT)
        # ......
        texts[0].add_updater(lambda m: m.move_to(dots[0].get_center() * self.text_buff_factor))
        texts[1].add_updater(lambda m: m.move_to(dots[1].get_center() * self.text_buff_factor))
        texts[2].add_updater(lambda m: m.move_to(dots[2].get_center() * self.text_buff_factor))
        texts[3].add_updater(lambda m: m.move_to(dots[3].get_center() * self.text_buff_factor))
        texts[4].add_updater(lambda m: m.move_to(dots[4].get_center() * self.text_buff_factor))
        texts[5].add_updater(lambda m: m.move_to(dots[5].get_center() * self.text_buff_factor))
        group = ThreeDVMobject(VGroup(inner_lines, sphere, dots, outer_lines))
        self.add(group, texts, one_dot, one)
        for axis_direction, angle in zip(self.rot_axes, self.rot_angles):
            norm_l = normalize(axis_direction)
            inner_dist = norm_l*self.sphere_radius
            max_dist = norm_l*self.max_norm
            inner_axis = Line(inner_dist, -inner_dist, **self.axis_config)
            outer_axis_1 = Line(inner_dist, max_dist, **self.axis_config)
            outer_axis_2 = Line(-inner_dist, -max_dist, **self.axis_config)
            axis = ThreeDVMobject(inner_axis, outer_axis_1, outer_axis_2)
            self.play(FadeIn(axis), Animation(group), run_time = 0.5)
            self.wait(0.5)
            self.play(Rotating(group, axis = norm_l, radians = angle), rate_func = smooth, run_time = 2.5)
            self.play(FadeOut(axis), Animation(group), run_time = 0.5)
            self.wait(1)


class Q8Automorphism2DPart(Scene):
    def construct(self):
        texts = VGroup(*[
            TexMobject(t, "\\mapsto", t)
            for t in ("1", "-1", "i", "-i", "j", "-j", "k", "-k")
        ])
        texts.arrange_submobjects(DOWN)
        # This part shouldn't be hard-coded, but honestly, it's much faster to work with.
        for text, color in zip(texts, [GREY, GREY, MAROON_B, RED, GREEN, TEAL, BLUE, PINK]):
            text[::2].set_color(color)
        self.add(texts)
        self.positions = [text[-1].get_left() for text in texts[2:]]
        for new_indices in [
            (4, 5, 2, 3, 1, 0), (4, 5, 3, 2, 0, 1), (3, 2, 4, 5, 1, 0),
            (5, 4, 0, 1, 3, 2), (3, 2, 1, 0, 5, 4), (0, 1, 2, 3, 4, 5)
            ]:
            self.wait(0.5 + 0.5)
            sources = VGroup(*[text[-1] for text in texts[2:]])
            target_positions = [self.positions[i] for i in new_indices]
            for mob, target_position in zip(sources, target_positions):
                mob.generate_target()
                mob.target.next_to(target_position, RIGHT, buff = 0)
            self.play(
                AnimationGroup(*[MoveToTarget(mob, path_arc = PI/2.) for mob in sources]),
                rate_func = smooth, run_time = 2.5
            )
            self.wait(0.5 + 1)


class Q8RepresentationRemark(Scene):
    def construct(self):
        remark = TextMobject("注：该球面为实部为0的四元数的三维球极投影")
        self.add(remark)
        self.wait()


class CubeRotationScene(AutQ8andS4CommonScene):
    CONFIG = {
        "sep_line_center" : LEFT_SIDE/2,
        "ref_line_center" : RIGHT_SIDE/2,
        "side_length" : 3,
        "axis_config" : {
            "stroke_width" : 5,
            "stroke_color" : YELLOW,
            "shade_in_3d" : True,
        },
        "text_buff_factor" : 1.2,
    }
    def construct(self):
        self.set_camera_orientation(**self.default_camera_orientation)
        cube = Cube(side_length = self.side_length)
        for face, color in zip(cube.submobjects, self.colors):
            face.set_fill(color = color, opacity = 0.9)
            face.set_stroke(color = GREY, width = 3)
        group = cube
        self.add(group)
        for axis_direction, angle in zip(self.rot_axes, self.rot_angles):
            inner_dist = axis_direction*self.side_length/2.
            max_dist = axis_direction*self.max_norm
            inner_axis = Line(inner_dist, -inner_dist, **self.axis_config)
            outer_axis_1 = Line(inner_dist, max_dist, **self.axis_config)
            outer_axis_2 = Line(-inner_dist, -max_dist, **self.axis_config)
            axis = ThreeDVMobject(inner_axis, outer_axis_1, outer_axis_2)
            self.play(FadeIn(axis), Animation(group), run_time = 0.5)
            self.wait(0.5)
            self.play(Rotating(group, axis = axis_direction, radians = angle), rate_func = smooth, run_time = 2.5)
            self.play(FadeOut(axis), Animation(group), run_time = 0.5)
            self.wait(1)


