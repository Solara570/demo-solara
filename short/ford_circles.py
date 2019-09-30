#coding=utf-8

import math
from manimlib.constants import *
from manimlib.animation.creation import ShowCreation
from manimlib.animation.fading import FadeOut
from manimlib.animation.transform import ApplyMethod, ReplacementTransform, Restore
from manimlib.mobject.types.vectorized_mobject import VMobject, VGroup
from manimlib.mobject.svg.tex_mobject import TexMobject, TextMobject
from manimlib.mobject.geometry import Circle, Rectangle, Arrow
from manimlib.mobject.coordinate_systems import Axes
from manimlib.scene.scene import Scene


def is_coprime(p, q):
    return math.gcd(p, q) == 1

def get_coprime_numers_by_denom(q):
    return [0, 1] if q == 1 else [p for p in range(1, q) if is_coprime(p, q)]

def get_stroke_width_by_height(height, thres = 1):
    return 1 if height > thres else height


class AssembledFraction(VMobject):
    CONFIG = {
        "stroke_width": 0,
        "fill_opacity": 1.0,
    }
    def __init__(self, p, q, **kwargs):
        self.p = str(p)
        self.q = str(q)
        super(AssembledFraction, self).__init__(**kwargs)

    def generate_points(self):
        numer = TexMobject(self.p)
        denom = TexMobject(self.q)
        line = Rectangle(height = 0.02)
        line.set_width(max(numer.get_width(), denom.get_width()) * 1.1, stretch = True)
        self.add(numer, line, denom)
        self.arrange_submobjects(DOWN, buff = 0.15)
        self.numer = numer
        self.line = line
        self.denom = denom


class ZoomInOnFordCircles(Scene):
    CONFIG = {
        "q_max" : 100,
        "label_height_factor" : 0.5,
        "axes_center": 2.5*DOWN,
        "axes_config": {
            "x_min": -1,
            "x_max": 2,
            "number_line_config": {
                "unit_size" : 1,
                "color": LIGHT_GREY,
                "include_tip" : False,
                "tick_size": 0,
                "numbers_with_elongated_ticks" : [],
                "exclude_zero_from_default_numbers": False,
            },
        },
        "circle_config": {
            "stroke_width" : 1,
            "stroke_color" : BLUE,
        },
    }
    def construct(self):
        self.setup_axes()
        self.setup_circles_and_labels()
        self.add_remark()
        self.show_different_levels_of_zooming()

    def setup_axes(self):
        axes = Axes(**self.axes_config)
        axes.center()
        axes.shift(self.axes_center)
        axes.x_axis.set_stroke(width = 1)
        axes.y_axis.set_stroke(width = 0)
        self.add_foreground_mobjects(axes)
        self.axes = axes

    def setup_circles_and_labels(self):
        circles = VGroup()
        labels = VGroup()
        for q in range(1, self.q_max+1):
            for p in get_coprime_numers_by_denom(q):
                circle = self.generate_circle_by_fraction(p, q)
                circle.add_updater(
                    lambda m: m.set_stroke(width = get_stroke_width_by_height(m.get_height()))
                )
                # label = TexMobject("\\dfrac{%d}{%d}" % (p, q))
                label = AssembledFraction(p, q)
                label.set_height(1)
                label.set_height(circle.get_height() * self.label_height_factor)
                label.move_to(circle.get_center())
                circles.add(circle)
                labels.add(label)
        self.add(circles, labels)
        self.circles = circles
        self.labels = labels

    def add_remark(self):
        nl_text = TextMobject("数轴")
        nl_arrow = Arrow(ORIGIN, UP).match_height(nl_text)
        nl_remark = VGroup(nl_arrow, nl_text)
        nl_remark.scale(0.8)
        nl_remark.set_color(LIGHT_GREY)
        nl_remark.arrange_submobjects(RIGHT, buff = 0.1)
        nl_remark.next_to(self.axes.coords_to_point(0, 0), DOWN, buff = 0.1)
        nl_remark.to_edge(LEFT, buff = 0.15)
        frac_remark = TextMobject("圆内分数为圆心横坐标")
        frac_remark.scale(0.6)
        frac_remark.to_corner(DL, buff = 0.15)
        farey_sum_remark = TexMobject(
            "\\text{Farey Sum: }", "\\dfrac{a}{b} \\oplus \\dfrac{c}{d}", "=", "\\dfrac{a+c}{b+d}"
        )
        farey_sum_remark[1].set_color(YELLOW)
        farey_sum_remark[-1].set_color(PINK)
        farey_sum_remark.to_corner(DR, buff = 0.15)
        self.add(nl_remark, frac_remark, farey_sum_remark)

    def show_different_levels_of_zooming(self):
        self.zoom_in_on(1/2., 6)
        self.wait()
        self.acl = VGroup(self.axes, self.circles, self.labels)
        self.acl.save_state()
        # First Zoom
        self.play_farey_sum_animation(0, 1, 1, 1)
        self.wait()
        # Second Zoom
        self.play_zooming_animation(1/np.sqrt(2), 9, run_time = 2)
        self.play_farey_sum_animation(2, 3, 3, 4)
        self.wait()
        # Third Zoom
        self.play_zooming_animation(0.73, 5, run_time = 2)
        self.play_farey_sum_animation(5, 7, 8, 11)
        self.wait()
        # Fourth Zoom
        self.play_zooming_animation(0.74, 5, run_time = 2)
        self.play_farey_sum_animation(11, 15, 14, 19)
        self.play_farey_sum_animation(14, 19, 17, 23)
        self.wait()
        # Reset
        self.play(Restore(self.acl), lag_ratio = 0, run_time = 4)
        self.wait()

    def generate_circle_by_fraction(self, p, q):
        radius = 1./(2 * q**2)
        center = self.axes.coords_to_point(p/q, radius)
        circle = Circle(radius = radius, **self.circle_config)
        circle.rotate(-PI/2.)
        circle.move_to(center)
        return circle

    def zoom_in_on(self, center, scaling_factor, update_circles = True):
        VGroup(self.axes, self.circles, self.labels).scale(
            scaling_factor, about_point = self.axes.coords_to_point(center, 0)
        )
        if update_circles:
            for circle in self.circles:
                circle.update(0)

    def play_zooming_animation(self, center, scaling_factor, **kwargs):
        self.play(
            ApplyMethod(
                VGroup(self.axes, self.circles, self.labels).scale, scaling_factor,
                {"about_point" : self.axes.coords_to_point(center, 0)}
            ),
            lag_ratio = 0, **kwargs,
        )

    def get_farey_sum_key_mobjects(self, p1, q1, p2, q2):
        assert (q1 + q2 <= self.q_max)
        c1 = self.get_circle_by_fraction(p1, q1).deepcopy()
        c2 = self.get_circle_by_fraction(p2, q2).deepcopy()
        c3 = self.get_circle_by_fraction(p1+p2, q1+q2).deepcopy()
        l1 = self.get_label_by_fraction(p1, q1).deepcopy()
        l2 = self.get_label_by_fraction(p2, q2).deepcopy()
        l3 = self.get_label_by_fraction(p1+p2, q1+q2).deepcopy()
        for c, color in zip([c1, c2, c3], [YELLOW, YELLOW, PINK]):
            c.clear_updaters()
            c.set_stroke(width = 5)
            c.set_color(color)
        return c1, c2, c3, l1, l2, l3

    def play_farey_sum_animation(self, p1, q1, p2, q2):
        c1, c2, c3, l1, l2, l3 = self.get_farey_sum_key_mobjects(p1, q1, p2, q2)
        l3.set_color(PINK)
        self.wait()
        self.play(
            ShowCreation(c1), ApplyMethod(l1.set_color, YELLOW),
            ShowCreation(c2), ApplyMethod(l2.set_color, YELLOW),
        )
        self.play(
            ReplacementTransform(l1.numer.deepcopy(), l3.numer),
            ReplacementTransform(l1.line.deepcopy(), l3.line),
            ReplacementTransform(l1.denom.deepcopy(), l3.denom),
            ReplacementTransform(l2.numer.deepcopy(), l3.numer),
            ReplacementTransform(l2.line.deepcopy(), l3.line),
            ReplacementTransform(l2.denom.deepcopy(), l3.denom),
            ReplacementTransform(c1.deepcopy(), c3),
            ReplacementTransform(c2.deepcopy(), c3),
        )
        self.wait()
        self.play(FadeOut(VGroup(c1, c2, c3, l1, l2, l3)))

    def get_circle_by_fraction(self, p, q, thres = 1e-6):
        x = p/q
        for circle in self.circles:
            cx = self.axes.point_to_coords(circle.get_center())[0]
            if np.abs(x - cx) < thres:
                return circle
        return VMobject()

    def get_label_by_fraction(self, p, q, thres = 1e-6):
        x = p/q
        for label in self.labels:
            lx = self.axes.point_to_coords(label.get_center())[0]
            if np.abs(x - lx) < thres:
                return label
        return VMobject()

