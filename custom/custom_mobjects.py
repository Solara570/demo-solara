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
    }
    def generate_points(self):
        self.ring = Annulus(
            inner_radius = self.inner_radius,
            outer_radius = self.outer_radius,
            fill_color = self.color
        )
        self.symbol = self.generate_symbol()
        self.add(VGroup(self.ring, self.symbol))

    def generate_symbol(self):
        # raise Exception("Not Implemented")
        return VMobject()

    def get_ring(self):
        return self.ring

    def get_symbol(self):
        return self.symbol


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
        "angle_text" : "",
        "minor_arc" : True,
        "scale_tex" : True,
        "arc_config" : {"radius" : 0.4, "color" : WHITE},
        "tex_config" : {"fill_color" : WHITE},
    }
    def __init__(self, *angle_pts, **kwargs):
        VMobject.__init__(self, **kwargs) 
        self.angle_pts = self.process_angle_pts(angle_pts)
        self.start_angle, self.delta_angle = self.compute_angles()
        arc = Arc(
            self.delta_angle, start_angle = self.start_angle,
            **self.arc_config
        )
        tex = TexMobject(self.angle_text, **self.tex_config)
        self.adjust_tex_to_arc(tex, arc)
        self.arc_and_tex = VGroup(arc, tex)
        vertex = self.get_vertex()
        self.arc_and_tex.shift(vertex)
        self.add(self.arc_and_tex)

    def get_vertex(self):
        return self.angle_pts[1]

    def get_arc(self):
        return self.arc_and_tex[0]

    def get_tex(self):
        return self.arc_and_tex[1]

    # Misc
    def process_angle_pts(self, angle_pts):
        assert len(angle_pts) == 3
        def process(obj):
            return (obj.get_center() if isinstance(obj, Mobject) else obj)
        return map(process, angle_pts)

    def compute_angles(self):
        A, B, C = self.angle_pts
        start_angle = self.compute_angle(B, C)
        final_angle = self.compute_angle(B, A)
        if final_angle < start_angle:
            final_angle, start_angle = start_angle, final_angle
        delta_angle = self.compute_delta_angle(start_angle, final_angle, self.minor_arc)
        return start_angle, delta_angle

    def compute_angle(self, start_pt, final_pt):
        diff = final_pt - start_pt
        num = diff[0] + diff[1] * 1j
        return np.angle(num)

    def compute_delta_angle(self, start_angle, final_angle, minor_arc):
        angle = (final_angle - start_angle) % TAU
        if ((angle < np.pi) ^ minor_arc):
            angle -= TAU
        return angle

    def adjust_tex_to_arc(self, tex, arc):
        # Scaling
        if self.scale_tex:
            tex.scale(arc.radius / 0.5)
        # Positioning
        mid_pt = arc.point_from_proportion(0.5)
        shift_vec = mid_pt * 1.5
        tex.shift(shift_vec)








