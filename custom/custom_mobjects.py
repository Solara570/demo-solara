import numpy as np

from helpers import *
from mobject import Mobject
from mobject.vectorized_mobject import *
from mobject.point_cloud_mobject import *
from mobject.svg_mobject import *
from mobject.tex_mobject import *

from animation.animation import Animation
from animation.transform import *
from animation.simple_animations import *
from animation.playground import *

from topics.geometry import *
from topics.objects import *
from topics.number_line import *
from topics.three_dimensions import *
from topics.common_scenes import *

from scene import Scene
from camera import Camera

# self.skip_animations
# self.force_skipping()
# self.revert_to_original_skipping_status()


## Some handmade control buttons
class Button(VMobject):
    CONFIG = {
        "color" : YELLOW,
        "inner_radius" : 2,
        "outer_radius" : 2.5,
        "circle_stroke_width" : 30,
    }
    def generate_points(self):
        ring = Annulus(
            inner_radius = self.inner_radius,
            outer_radius = self.outer_radius,
            fill_color = self.color
        )
        symbol = self.generate_symbol()
        self.add(VGroup(ring, symbol))

    def generate_symbol(self):
        raise Exception("Not Implemented")

class PauseButton(Button):
    def generate_symbol(self):
        symbol = VGroup(*[
            Rectangle(
                length = 2, width = 0.5, stroke_width = 0,
                fill_color = self.color, fill_opacity = 1,
            )
            for i in range(2)
        ])
        symbol.arrange_submobjects(RIGHT, buff = 0.5)
        symbol.scale_to_fit_height(self.inner_radius)
        return symbol

class PlayButton(Button):
    def generate_symbol(self):
        symbol = RegularPolygon(
            n = 3, stroke_width = 0,
            fill_color = self.color, fill_opacity = 1,
        )
        symbol.scale_to_fit_height(self.inner_radius)
        return symbol

class SkipButton(Button):
    def generate_symbol(self):
        symbol = VGroup(*[
            RegularPolygon(
                n = 3, stroke_width = 0,
                fill_color = self.color, fill_opacity = 1
            )
            for i in range(2)
        ])
        symbol.arrange_submobjects(RIGHT, buff = 0)
        symbol.scale_to_fit_height(self.inner_radius * 0.7)
        return symbol

class RewindButton(Button):
    def generate_symbol(self):
        symbol = VGroup(*[
            RegularPolygon(
                n = 3, stroke_width = 0, start_angle = np.pi,
                fill_color = self.color, fill_opacity = 1
            )
            for i in range(2)
        ])
        symbol.arrange_submobjects(RIGHT, buff = 0)
        symbol.scale_to_fit_height(self.inner_radius * 0.7)
        return symbol

class StopButton(Button):
    def generate_symbol(self):
        symbol = Square(stroke_width = 0, fill_color = self.color, fill_opacity = 1)
        symbol.scale_to_fit_height(self.inner_radius * 0.8)
        return symbol


## AngleArc class
# Useful when adding an angle indicator with a label.
class AngleArc(VMobject):
    CONFIG = {
        "radius" : 0.5,
        "start_angle" : 0,
        "angle_text" : "",
        "arc_color" : WHITE,
        "text_color" : WHITE,
    }
    def __init__(self, *angle_pts, **kwargs):
        digest_config(self, kwargs, locals())
        VMobject.__init__(self, **kwargs)
        start_angle, delta_angle = self.process_angle_info()
        tip_point = self.get_tip_point()
        self.arc = Arc(
            delta_angle, start_angle = start_angle,
            radius = self.radius, color = self.arc_color
        )
        self.text = TexMobject(self.angle_text).highlight(self.text_color)
        self.adjust_text()
        angle_and_text = VGroup(self.arc, self.text)
        angle_and_text.shift(tip_point)
        self.add(angle_and_text)

    def process_angle_info(self):
        pts = self.angle_pts
        assert len(pts) == 3
        if isinstance(pts[0], Mobject):
            A, B, C = [mob.get_center() for mob in pts]
        else:
            A, B, C = pts
        start_angle = self.compute_angle(B, C)
        final_angle = self.compute_angle(B, A)
        if final_angle < start_angle:
            final_angle, start_angle = start_angle, final_angle
        delta_angle = self.mod_angle(final_angle - start_angle)
        if delta_angle > np.pi:
            delta_angle -= TAU
        return start_angle, delta_angle

    def compute_angle(self, start_pt, final_pt):
        diff = final_pt - start_pt
        num = diff[0] + diff[1] * 1j
        return np.angle(num)

    def mod_angle(self, angle):
        return (angle - np.pi) % TAU + np.pi

    def get_tip_point(self):
        pt = self.angle_pts[1]
        if isinstance(pt, Mobject):
            return pt.get_center()
        else:
            return pt

    def adjust_text(self):
        # Scaling
        self.text.scale(self.radius / 0.5)
        # Positioning
        mid_pt = self.arc.point_from_proportion(0.5)
        shift_vec = mid_pt * 1.5
        self.text.shift(shift_vec)
        return self
