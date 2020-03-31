#coding=utf-8

################################################################################################
#  A 3-part series on circle inversion, Descartes' theorem along with its variants, and more!  #
#                                                                                              #
#    Part 1: An Introduction to Circle Inversion   - https://zhuanlan.zhihu.com/p/86644341     #
#    Part 2: Four Circles & Descartes' Theorem (1) - https://zhuanlan.zhihu.com/p/105819963    #
#    Part 3: Four Circles & Descartes' Theorem (2) - https://zhuanlan.zhihu.com/p/106874090    #
################################################################################################

import numpy as np
import itertools as it
from manimlib.constants import *
from manimlib.utils.color import *
from manimlib.utils.space_ops import *
from manimlib.utils.simple_functions import *
from manimlib.animation.composition import AnimationGroup
from manimlib.animation.creation import ShowCreation, Write, DrawBorderThenFill
from manimlib.animation.fading import FadeOut, FadeInFromDown
from manimlib.animation.transform import Transform, ReplacementTransform, MoveToTarget, ApplyMethod
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.coordinate_systems import Axes, NumberPlane, ThreeDAxes
from manimlib.mobject.geometry import Circle, Line, Dot, SmallDot, Square, Polygon, RegularPolygon, \
                                      Arrow, Sector, Vector
from manimlib.mobject.numbers import DecimalNumber
from manimlib.mobject.value_tracker import ValueTracker
from manimlib.mobject.shape_matchers import BackgroundRectangle, SurroundingRectangle
from manimlib.mobject.three_dimensions import Sphere
from manimlib.mobject.svg.brace import Brace
from manimlib.mobject.svg.tex_mobject import TexMobject, TextMobject
from manimlib.mobject.types.vectorized_mobject import VMobject, VGroup, VectorizedPoint, DashedVMobject
from manimlib.scene.scene import Scene
from manimlib.scene.three_d_scene import ThreeDScene

from short.apollonian_gasket import calc_centers_by_radii, calc_new_agc_info, AGCircle, \
                                    ApollonianGasket, ApollonianGasketScene
from short.ford_circles import get_coprime_numers_by_denom, get_stroke_width_by_height, \
                               AssembledFraction, ZoomInOnFordCircles


#####
## Constants
MAX_NORM = 1e2
CB_DARK  = "#825201"
CB_LIGHT = "#B69B4C"


#####
## General Methods
def complex_inversion(z, z0, r):
    return z0 + np.conjugate(r**2 / (z-z0))

def R3_inversion(point, inv_center, radius):
    z = R3_to_complex(point)
    z0 = R3_to_complex(inv_center)
    w = complex_inversion(z, z0, radius)
    return complex_to_R3(w)

def inversion(point, inv_center, radius):
    # Just a rename
    return R3_inversion(point, inv_center, radius)

def is_close_in_R3(p1, p2, thres = 1e-6):
    """Check if two points are close in R^3."""
    return np.linalg.norm(p1 - p2) < thres

def is_close(z1, z2, thres = 1e-6):
    """Check if two complex numbers are close to each other."""
    return np.abs(z1 - z2) < thres

def get_tangent_point(c1, c2, thres = 1e-4):
    """Return the tangency point of circles 'c1' and 'c2'."""
    p1 = c1.get_center()
    p2 = c2.get_center()
    r1 = c1.get_height() / 2
    r2 = c2.get_height() / 2
    d = get_norm(p2 - p1)
    if is_close(d, r1-r2, thres):
        return p1 + r1*normalize(p2-p1)
    elif is_close(d, r2-r1, thres):
        return p2 + r2*normalize(p1-p2)
    elif is_close(d, r1+r2, thres):
        return (r1*p2+r2*p1) / (r1+r2)
    else:
        raise Exception("These two circles aren't tangent.")

def get_para_and_perp_components(point, lp1, lp2):
    v = lp2 - point
    v0 = lp2 - lp1
    v_para = fdiv(np.dot(v, v0), np.dot(v0, v0)) * v0
    v_perp = v - v_para
    return v_para, v_perp

def distance_to_the_line(point, lp1, lp2):
    """Return the distance from 'point' to the line given by 'lp1' and 'lp2'."""
    v_para, v_perp = get_para_and_perp_components(point, lp1, lp2)
    return np.linalg.norm(v_perp)

def is_on_the_line(point, lp1, lp2, thres = 1e-6):
    """Check if 'point' is on the line given by two points 'lp1' and 'lp2'."""
    return is_close(distance_to_the_line(point, lp1, lp2), thres)

def get_random_vector(max_step):
    """Return a random vector with a maximum length of 'max_step'."""
    return max_step*np.random.random() * rotate_vector(RIGHT, TAU*np.random.random())

def get_nearest_int(num):
    return int(np.round(num, 0))

def solve_quadratic_equation(a, b, c):
    delta = b**2 - 4*a*c
    x1 = (-b-np.sqrt(delta)) /(2*a)
    x2 = (-b+np.sqrt(delta)) /(2*a)
    print(a, b, c, x1, x2)
    return x1, x2

def get_next_terms(k1, k2, k3):
    """Return two adjacent terms in the loxodromic sequence."""
    b = -2*(k1+k2+k3)
    c = 2*(k1**2+k2**2+k3**2) - (k1+k2+k3)**2
    return list(map(get_nearest_int, solve_quadratic_equation(1, b, c)))

def get_sequence_string(arr):
    arr_copy = list(map(str, arr))
    arr_copy.insert(0, "...")
    arr_copy.append("...")
    return ", ".join(arr_copy)


#####
## Mobjects
class FineCircle(Circle):
    CONFIG = {
        # In manim, circles are approximated by multiple cubic Beziers,
        # so it's necessary to increase the number of components for
        # high-precision calculations.
        "num_components": 100,
    }


class ExtendedLine(Line):
    def __init__(self, sp, ep, n = 10, **kwargs):
        unit_vec = normalize(ep - sp)
        new_sp = sp - n * unit_vec
        new_ep = ep + n * unit_vec
        Line.__init__(self, new_sp, new_ep, **kwargs)


class DotLabel(VMobject):
    CONFIG = {
        "position" : UP,
        "label_buff" : 0.25,
    }
    def __init__(self, label_text, dot, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.dot = dot
        label = TexMobject(label_text, **kwargs)
        if self.position is not None:
            label.add_updater(
                lambda l: l.next_to(self.dot.get_center(), self.position, buff = self.label_buff)
            )
        self.add(label)

    def set_label(self, label):
        label.next_to(self.dot.get_center())


class TwoDotsSegment(Line):
    def __init__(self, dot_1, dot_2, **kwargs):
        self.dot_1 = dot_1
        self.dot_2 = dot_2
        sp, ep = self.get_dots_centers()
        Line.__init__(self, start = sp, end = ep, **kwargs)
        self.add_updater(self.set_start_and_end)

    def get_dots_centers(self):
        return self.dot_1.get_center(), self.dot_2.get_center()

    def set_start_and_end(self, line_mob):
        sp, ep = self.get_dots_centers()
        line_mob.put_start_and_end_on(sp, ep)


class LengthLabel(DecimalNumber):
    CONFIG = {
        "num_decimal_places" : 3,
        "label_height" : 0.3,
        "label_buff" : 0.3,
        "offset" : 0,
        "is_on_opposite_side" : False,
    }
    def __init__(self, line_mob, **kwargs):
        DecimalNumber.__init__(self, **kwargs)
        self.line_mob = line_mob
        self.add_updater(self.set_label)

    def set_label(self, label):
        label.set_value(self.line_mob.get_length())
        label.set_height(self.label_height)
        label.rotate(self.line_mob.get_angle())
        side_factor = -1 if self.is_on_opposite_side else 1
        label.move_to(
            self.line_mob.get_center() \
            + self.line_mob.get_vector() / 2 * self.offset \
            + side_factor * rotate_vector(self.line_mob.get_unit_vector(), PI/2) * self.label_buff
        )

    def set_offset(self, offset):
        self.offset = offset
        return self

    def switch_side(self):
        self.is_on_opposite_side = not self.is_on_opposite_side
        return self


class ManyDotsPolygon(VMobject):
    def __init__(self, *dots, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.dots = dots
        dots_centers = self.get_dots_centers()
        polygon = Polygon(*dots_centers, **kwargs)
        polygon.add_updater(self.set_vertices)
        self.add(polygon)

    def get_dots_centers(self):
        return [dot.get_center() for dot in self.dots]

    def set_vertices(self, polygon_mob):
        vertices = self.get_dots_centers()
        polygon_mob.set_points_as_corners([*vertices, vertices[0]])


class AngleIndicator(VMobject):
    CONFIG = {
        "color" : RED,
        "radius" : 0.2,
        "fill_opacity" : 0.6,
        "is_minor_arc" : True,
    }
    def __init__(self, dot_A, dot_C, dot_B, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.dot_A = dot_A
        self.dot_C = dot_C
        self.dot_B = dot_B
        sector = Sector()
        sector.add_updater(self.set_sector)
        self.add(sector)
        self.sector = sector

    def get_point_center(self, point_or_mob):
        if isinstance(point_or_mob, Mobject):
            return point_or_mob.get_center()
        else:
            return point_or_mob

    def get_point_centers(self):
        return tuple(map(self.get_point_center, [self.dot_A, self.dot_C, self.dot_B]))

    def set_sector(self, mob):
        pt_A, pt_C, pt_B = self.get_point_centers()
        start_angle, angle = self.get_angles()
        outer_radius = min([self.radius, get_norm(pt_C - pt_A)/2, get_norm(pt_C - pt_B)/2])
        new_sector = Sector(
            start_angle = start_angle, angle = angle, outer_radius = outer_radius,
            color = self.color, fill_opacity = self.fill_opacity, stroke_width = 0
        )
        new_sector.move_arc_center_to(self.get_point_center(self.dot_C))
        mob.become(new_sector)
        
    def get_angles(self):
        pt_A, pt_C, pt_B = self.get_point_centers()
        start_angle = angle_of_vector(pt_A - pt_C)
        end_angle = angle_of_vector(pt_B - pt_C)
        angle = (end_angle - start_angle) % TAU
        if self.is_minor_arc and angle > PI:
            angle -= TAU
        return start_angle, angle


class RightAngleIndicator(VMobject):
    CONFIG = {
        "color" : WHITE,
        "side_length" : 0.2,
        "line_width" : 1,
        "square_opacity" : 0.5,
    }
    def __init__(self, dot_A, dot_C, dot_B, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.dot_A = dot_A
        self.dot_C = dot_C
        self.dot_B = dot_B
        line = VMobject(stroke_width = self.line_width, fill_opacity = 0)
        square = VMobject(stroke_width = 0, fill_color = self.color, fill_opacity = self.square_opacity)
        line.add_updater(self.set_line)
        square.add_updater(self.set_square)
        self.add(square, line)
        self.line = line
        self.square = square

    def get_point_center(self, point_or_mob):
        if isinstance(point_or_mob, Mobject):
            return point_or_mob.get_center()
        else:
            return point_or_mob

    def get_point_centers(self):
        return tuple(map(self.get_point_center, [self.dot_A, self.dot_C, self.dot_B]))

    def get_norm_vectors(self):
        pt_A, pt_C, pt_B = self.get_point_centers()
        norm_vec_CA = normalize(pt_A - pt_C)
        norm_vec_CB = normalize(pt_B - pt_C)
        return norm_vec_CA, norm_vec_CB

    def get_corner_points(self):
        pt_A, pt_C, pt_B = self.get_point_centers()
        norm_vec_CA, norm_vec_CB = self.get_norm_vectors()
        side_length = min([self.side_length, get_norm(pt_A - pt_C)/2, get_norm(pt_B - pt_C)/2])
        return (
            pt_C,
            pt_C + norm_vec_CA * side_length,
            pt_C + norm_vec_CA * side_length + norm_vec_CB * side_length,
            pt_C + norm_vec_CB * side_length
        )

    def set_line(self, line_mob):
        p, q, r, s = self.get_corner_points()
        line_mob.set_points_as_corners([q, r, s])

    def set_square(self, square_mob):
        p, q, r, s = self.get_corner_points()
        square_mob.set_points_as_corners([p, q, r, s, p])


class InversedDot(VMobject):
    CONFIG = {
        "color" : PINK,
        "stroke_width" : 3,
        "fill_opacity" : 1,
        "is_hollow" : True,
        "center_color" : BLACK,
    }
    def __init__(self, orig_dot, circle, **kwargs):
        self.orig_dot = orig_dot
        self.circle = circle
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        if self.is_hollow:
            self.fill_color = self.center_color
        else:
            self.fill_color = self.color
            self.stroke_width = 0
        inv_dot = Dot(ORIGIN, color = self.color)
        self.inv_dot = inv_dot
        self.add(inv_dot)
        self.add_updater_to_inversed_dot()

    def add_updater_to_inversed_dot(self):
        self.inv_dot.add_updater(self.move_inversed_dot)

    def move_inversed_dot(self, inv_dot):
        point = self.orig_dot.get_center()
        inv_center = self.circle.get_center()
        radius = self.circle.get_height() / 2.
        if is_close_in_R3(point, inv_center):
            pass
        else:
            inv_dot.move_to(inversion(point, inv_center, radius))


class InversedVMobject(VMobject):
    CONFIG = {
        "is_analytical" : True,
        "match_original_style" : False,
        "use_dashed_vmob" : True,
        "dashed_vmob_config": {
            "num_dashes" : 50,
            "positive_space_ratio" : 0.6,
        },
    }
    def __init__(self, orig_vmob, circle, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.orig_vmob = orig_vmob
        self.circle = circle
        self.orig_vmob_type = "Others"
        self.initialize_orig_vmob_type()
        self.add_updater_to_inversed_vmobject()

    def add_updater_to_inversed_vmobject(self):
        self.add_updater(self.set_inversed_vmobject)

    def initialize_orig_vmob_type(self):
        if isinstance(self.orig_vmob, Line):
            self.orig_vmob_type = "Line"
        elif isinstance(self.orig_vmob, Circle):
            self.orig_vmob_type = "Circle"
        else:
            self.orig_vmob_type = "Others"

    def set_orig_vmob_type(self, orig_vmob_type):
        self.orig_vmob_type = orig_vmob_type

    def set_inversed_vmobject(self, inv_vmob):
        inv_center = self.circle.get_center()
        radius = self.circle.get_height() / 2.
        if self.is_analytical and self.orig_vmob_type == "Line":
            # If it's a line...
            lp1, lp2 = self.orig_vmob.get_start_and_end()
            if is_on_the_line(inv_center, lp1, lp2):
                # If it's a line passing through the inversion center,
                # then the inversion is just the line itself.
                temp_vmob = ExtendedLine(lp1, lp2)
            else:
                # If it's a line NOT through the inversion center,
                # then the inversion is a circle passing through the inversion center.
                v_para, v_perp = get_para_and_perp_components(inv_center, lp1, lp2)
                d = distance_to_the_line(inv_center, lp1, lp2)
                # d = np.linalg.norm(v_perp)
                inv_vmob_radius = fdiv(radius**2, 2*d)
                closepoint = inv_center + v_perp
                inv_vmob_closepoint = inversion(closepoint, inv_center, radius)
                inv_vmob_center = (inv_center + inv_vmob_closepoint) / 2.
                temp_vmob = FineCircle(radius = inv_vmob_radius)
                temp_vmob.move_to(inv_vmob_center)
        elif self.is_analytical and self.orig_vmob_type == "Circle":
        # If it's a circle...
            orig_vmob_center = self.orig_vmob.get_center()
            orig_vmob_radius = self.orig_vmob.get_height() / 2.
            center_vec = orig_vmob_center - inv_center
            d = get_norm(center_vec)
            if is_close(orig_vmob_radius, d):
                # If it's a circle passing through the inversion center,
                # then the inversion is a line perps to the line through the circle centers.
                foot = inv_center + fdiv(radius**2, 2*d) * normalize(center_vec)
                lp1 = foot + rotate_vector(center_vec, PI/2)
                lp2 = foot + rotate_vector(center_vec, -PI/2)
                temp_vmob = ExtendedLine(lp1, lp2)
            else:
                # If it's a circle NOT through the inversion center,
                # then the inversion is a circle NOT through the inversion center.
                dp1 = orig_vmob_center - orig_vmob_radius * normalize(center_vec)
                dp2 = orig_vmob_center + orig_vmob_radius * normalize(center_vec)
                inv_dp1 = inversion(dp1, inv_center, radius)
                inv_dp2 = inversion(dp2, inv_center, radius)
                inv_vmob_radius = get_norm(inv_dp2 - inv_dp1) / 2.
                inv_vmob_center = (inv_dp2 + inv_dp1) / 2.
                temp_vmob = FineCircle(radius = inv_vmob_radius)
                temp_vmob.move_to(inv_vmob_center)
        else:
            temp_vmob = self.orig_vmob.copy()
            temp_vmob.apply_function(lambda p: inversion(p, inv_center, radius))
        if self.use_dashed_vmob:
            temp_vmob = DashedVMobject(temp_vmob, **self.dashed_vmob_config)
        inv_vmob.become(temp_vmob)
        if self.match_original_style:
            inv_vmob.match_style(self.orig_vmob)


class FourCirclesNormalForm(VMobject):
    CONFIG = {
        "circle_colors" : [MAROON_B, RED, GREEN, BLUE],
        "r" : 1.2,
        "l" : 9,
        "use_dashed_vmob" : True,
        "dashed_vmob_config" : {
            "num_dashes" : 30,
            "positive_space_ratio" : 0.6,
        }
    }
    def __init__(self, **kwargs):
        VMobject.__init__(self, **kwargs)
        c1 = Circle(radius = self.r, **kwargs).shift(self.r*LEFT)
        c2 = Circle(radius = self.r, **kwargs).shift(self.r*RIGHT)
        c3 = Line(self.l*LEFT, self.l*RIGHT, **kwargs).shift(self.r*DOWN)
        c4 = Line(self.l*LEFT, self.l*RIGHT, **kwargs).shift(self.r*UP)
        for mob, color in zip([c1, c2, c3, c4], self.circle_colors):
            mob.set_color(color)
            if self.use_dashed_vmob:
                self.add(DashedVMobject(mob, **self.dashed_vmob_config))
            else:
                self.add(mob)


class DescartesFourCircles(VMobject):
    CONFIG = {
        "outer_circle_index" : None,
        "orig_circle_color" : BLUE,
        "new_circle_color" : YELLOW,
        "show_new_circles" : True,
        "show_new_circles_centers" : False,
    }
    def __init__(self, ccdot1, ccdot2, ccdot3, **kwargs):
        self.ccdot1 = ccdot1
        self.ccdot2 = ccdot2
        self.ccdot3 = ccdot3
        VMobject.__init__(self, **kwargs)
        self.add_orig_circles()
        self.add_orig_circles_updaters()
        self.generate_new_circles()
        if self.show_new_circles:
            self.add_new_circles()
        if self.show_new_circles_centers:
            self.add_new_circles_centers()
            
    def add_orig_circles(self):
        self.c1, self.c2, self.c3 = self.cs = VGroup(*[
            Circle(arc_center = cc, radius = r, color = self.orig_circle_color)
            for cc, r in zip(self.get_orig_circle_centers(), self.calc_radii_by_centers())
        ])
        self.add(self.cs)

    def add_orig_circles_updaters(self):
        def get_center(k):
            return self.get_orig_circle_centers()[k]
        def get_abs_radius(k):
            return np.abs(self.calc_radii_by_centers()[k])
        # Since enumerate() won't work here (seriously?),
        # I have to use a much more direct approach - list them all.
        self.c1.add_updater(lambda c: c.move_to(get_center(0)))
        self.c1.add_updater(lambda c: c.set_height(2*get_abs_radius(0)))
        self.c2.add_updater(lambda c: c.move_to(get_center(1)))
        self.c2.add_updater(lambda c: c.set_height(2*get_abs_radius(1)))
        self.c3.add_updater(lambda c: c.move_to(get_center(2)))
        self.c3.add_updater(lambda c: c.set_height(2*get_abs_radius(2)))

    def get_orig_circles(self):
        return self.cs

    def get_orig_circle_centers(self):
        return [dot.get_center() for dot in (self.ccdot1, self.ccdot2, self.ccdot3)]

    def get_orig_circle_radii(self):
        return self.calc_radii_by_centers()

    def get_orig_circle_curvatures(self):
        return [fdiv(1, radius) for radius in self.calc_radii_by_centers()]

    def calc_radii_by_centers(self):
        p1, p2, p3 = self.get_orig_circle_centers()
        d12 = get_norm(p2 - p1)
        d23 = get_norm(p3 - p2)
        d13 = get_norm(p3 - p1)
        sum_r = (d12 + d23 + d13) / 2.
        if self.outer_circle_index == 1:
            # If circle 1 contains other two circles...
            return [-sum_r, sum_r-d12, sum_r-d13]
        elif self.outer_circle_index == 2:
            # If circle 2 contains other two circles...
            return [sum_r-d12, -sum_r, sum_r-d23]
        elif self.outer_circle_index == 3:
            # If circle 3 contains other two circles...
            return [sum_r-d13, sum_r-d23, -sum_r]
        else:
            return [sum_r-d23, sum_r-d13, sum_r-d12]

    def generate_new_circles(self):
        self.c4_1, self.c4_2 = self.new_circles = VGroup(*[
            Circle(arc_center = new_cc, radius = new_r, color = self.new_circle_color)
            for new_cc, new_r in self.calc_new_circles_centers_and_radii()
        ])
        self.generate_new_circles_centers()
        self.add_new_circles_updaters()

    def calc_new_circles_centers_and_radii(self):
        k1, k2, k3 = self.get_orig_circle_curvatures()
        z1, z2, z3 = map(R3_to_complex, self.get_orig_circle_centers())
        # Calculate the curvatures of new circles
        sum_k = k1 + k2 + k3
        sum_k2 = k1**2 + k2**2 + k3**2
        sum_k_cycle_prod = k1*k2 + k2*k3 + k3*k1
        b = (-2)*sum_k
        c = sum_k2 - 2*sum_k_cycle_prod
        delta = b**2 - 4*c
        k4_1 = (-b + np.sqrt(delta)) / 2
        k4_2 = (-b - np.sqrt(delta)) / 2
        # Calculate the centers of new circles
        # arxiv.org/abs/math/0101066v1 - Eqn 2.3
        sum_kz = k1*z1 + k2*z2 + k3*z3
        sum_k2z = k1**2 * z1 + k2**2 * z2 + k3**2 * z3
        coeff_1 = (sum_k - k4_1) * k4_1
        const_1 = 2 * sum_k2z - (sum_k + k4_1) * sum_kz
        z4_1 = const_1 / coeff_1
        coeff_2 = (sum_k - k4_2) * k4_2
        const_2 = 2 * sum_k2z - (sum_k + k4_2) * sum_kz
        z4_2 = const_2 / coeff_2
        return [[complex_to_R3(z4_1), fdiv(1, k4_1)], [complex_to_R3(z4_2), fdiv(1, k4_2)]]

    def generate_new_circles_centers(self):
        ccdot4_1 = Dot(color = self.new_circle_color)
        ccdot4_1.add_updater(lambda m: m.move_to(self.c4_1.get_center()))
        ccdot4_2 = Dot(color = self.new_circle_color)
        ccdot4_2.add_updater(lambda m: m.move_to(self.c4_2.get_center()))
        self.ccdot4_1 = ccdot4_1
        self.ccdot4_2 = ccdot4_2

    def add_new_circles_updaters(self):
        def get_new_center(k):
            return self.calc_new_circles_centers_and_radii()[k][0]
        def get_abs_new_radius(k):
            return np.abs(self.calc_new_circles_centers_and_radii()[k][1])
        # Since enumerate() won't work here (seriously?),
        # I have to use a much more direct approach - list them all.
        self.c4_1.add_updater(lambda c: c.move_to(get_new_center(0)))
        self.c4_1.add_updater(lambda c: c.set_height(2*get_abs_new_radius(0)))
        self.c4_2.add_updater(lambda c: c.move_to(get_new_center(1)))
        self.c4_2.add_updater(lambda c: c.set_height(2*get_abs_new_radius(1)))

    def add_new_circles(self):
        if not hasattr(self, "new_circles"):
            self.new_circles = generate_new_circles()
        self.add(self.new_circles)

    def get_new_circles(self):
        if not hasattr(self, "new_circles"):
            self.new_circles = generate_new_circles()
        return self.new_circles

    def add_new_circles_centers(self):
        self.add(self.ccdot4_1, self.ccdot4_2)

    def remove_new_circles_center(self):
        self.remove(self.ccdot4_1, self.ccdot4_2)



#####
## Inversion Introduction Scenes
class ConceptsInInversion(Scene):
    CONFIG = {
        "color_circle" : YELLOW,
        "color_radius" : RED,
        "color_P" : WHITE,
    }
    def construct(self):
        self.add_backgrounds()
        self.move_around_point_P()

    def add_backgrounds(self):
        circle_O = Circle(radius = 3.5, color = self.color_circle)
        circle_O.shift(3*LEFT)
        remark_circle = TextMobject("反演圆", color = self.color_circle)
        remark_circle.next_to(circle_O.get_bottom(), UP)
        dot_O = Dot(circle_O.get_center(), color = self.color_circle)
        label_O = DotLabel("O", dot_O, color = self.color_circle, position = DOWN)
        remark_O = TextMobject("反演中心", color = self.color_circle)
        remark_O.next_to(label_O, LEFT, buff = 0.15)
        radius = Line(circle_O.get_center(), circle_O.get_left())
        label_radius = TexMobject("R").scale(0.8)
        remark_radius = TextMobject("反演幂").scale(0.8)
        brace_radius = Brace(radius, UP)
        brace_radius.put_at_tip(label_radius)
        remark_radius.next_to(label_radius, LEFT, buff = 0.15)
        group_radius = VGroup(radius, label_radius, brace_radius, remark_radius)
        group_radius.set_color(self.color_radius)
        group_radius.rotate(-PI/12, about_point = dot_O.get_center())
        def_inversion = TextMobject("反演变换：$P \\mapsto P'$")
        rlt_inversion = TexMobject("|OP| \\times |OP'|=", "R^2")
        rlt_inversion.next_to(def_inversion, DOWN, aligned_edge = RIGHT)
        rlt_inversion[-1].set_color(self.color_radius)
        remarks = VGroup(def_inversion, rlt_inversion)
        remarks.to_corner(DR)
        dot_P = Dot(LEFT, color = self.color_P)
        label_P = DotLabel("P", dot_P, color = self.color_P, position = DL, label_buff = 0.2)
        dot_Pi = InversedDot(dot_P, circle_O, color = self.color_P)
        label_Pi = DotLabel("P'", dot_Pi, color = self.color_P, position = DR, label_buff = 0.2)
        line_OP = TwoDotsSegment(dot_O, dot_P, stroke_width = 2)
        line_OPi = TwoDotsSegment(dot_O, dot_Pi, stroke_width = 2)
        self.add(remarks)
        self.add(group_radius)
        self.add(circle_O, dot_O, label_O, remark_O, remark_circle)
        self.add(dot_P, dot_Pi, label_P, label_Pi, line_OP, line_OPi)
        self.circle_O = circle_O
        self.dot_P = dot_P

    def move_around_point_P(self):
        self.dot_P.save_state()
        for dx, dy in [(-0.2, 0.3), (0.1, -0.4), (4, 0.3), (1, 1)]:
            vec = np.array([dx, dy, 0])
            self.play(self.dot_P.shift, vec, run_time = 1)
            self.wait()
        self.play(self.dot_P.move_to, self.circle_O.get_right())
        self.wait()
        self.play(self.dot_P.restore, run_time = 1)
        self.wait()


class InversionExamples(Scene):
    CONFIG = {
        "color_circle" : YELLOW,
    }
    def construct(self):
        circle_O = Circle(radius = 3.5, color = self.color_circle)
        circle_O.shift(3*LEFT)
        remark_circle = TextMobject("反演圆", color = self.color_circle)
        remark_circle.next_to(circle_O.get_bottom(), UP)
        dot_O = Dot(circle_O.get_center(), color = self.color_circle)
        label_O = DotLabel("O", dot_O, color = self.color_circle, position = DOWN)
        init_shape = Square(side_length = 1.2, color = BLUE).rotate(TAU/13)
        init_shape.next_to(circle_O.get_right(), LEFT, buff = 0.5)
        init_shape.save_state()
        inv_shape = InversedVMobject(init_shape, circle_O, use_dashed_vmob = False)
        new_shapes = [
            RegularPolygon(n = 6, start_angle = PI/7, color = PINK).scale(0.8),
            TexMobject("42", color = RED).scale(2.5).rotate(-PI/9),
            TexMobject("\\pi", color = MAROON_B).scale(5).rotate(PI/15),
        ]

        self.add(circle_O, remark_circle, dot_O, label_O)
        self.add(init_shape, inv_shape)
        for new_shape in new_shapes:
            # new_shape.set_color(BLUE)
            new_shape.next_to(circle_O.get_right(), LEFT, buff = 0.6)
            self.play(Transform(init_shape, new_shape), run_time = 1)
            self.wait()
            init_shape.generate_target()
            init_shape.target.become(new_shape)
            init_shape.target.shift(get_random_vector(0.5))
            random_angle = 0.5*np.random.random()
            init_shape.target.rotate(random_angle)
            self.play(MoveToTarget(init_shape, path_arc = random_angle, run_time = 1)),
            self.wait()
        self.play(ApplyMethod(init_shape.restore))
        self.wait()


class LineToLineInversion(Scene):
    CONFIG = {
        "color_circle" : YELLOW,
        "color_orig" : BLUE,
        "color_inv" : RED,
    }
    def construct(self):
        self.add_backgrounds()
        self.show_line_to_line_inversion()

    def add_backgrounds(self):
        circle_O = Circle(radius = 2.5, color = self.color_circle)
        remark_circle = TextMobject("反演圆", color = self.color_circle)
        remark_circle.next_to(circle_O.get_bottom(), UP)
        dot_O = Dot(circle_O.get_center(), color = self.color_circle)
        label_O = DotLabel("O", dot_O, color = self.color_circle, position = DOWN)
        conclusion = TextMobject("经过反演中心的直线", "$\\mapsto$", "经过反演中心的直线")
        conclusion.scale(0.8)
        conclusion[0].set_color(self.color_orig)
        conclusion[2].set_color(self.color_inv)
        conclusion.to_corner(DR)
        self.add(circle_O, remark_circle, dot_O, label_O)
        self.add(conclusion)
        self.circle_O = circle_O
    
    def show_line_to_line_inversion(self):
        angle_tracker = ValueTracker(-PI/11)
        position_tracker = ValueTracker(1.4)
        angle_tracker.save_state()
        position_tracker.save_state()
        orig_line = ExtendedLine(LEFT, RIGHT, color = self.color_orig, stroke_width = 8)
        orig_line.add_updater(lambda m: m.rotate(angle_tracker.get_value() - m.get_angle()))
        inv_line = ExtendedLine(LEFT, RIGHT, color = self.color_inv, stroke_width = 4)
        inv_line.add_updater(lambda m: m.rotate(angle_tracker.get_value() - m.get_angle()))
        dot_P = Dot(color = self.color_orig)
        dot_P.add_updater(
            lambda m: m.move_to(
                position_tracker.get_value() * rotate_vector(RIGHT, angle_tracker.get_value())
            )
        )
        dot_Pi = InversedDot(dot_P, self.circle_O, is_hollow = False, color = self.color_inv)
        label_P = DotLabel("P", dot_P, position = DOWN, color = self.color_orig)
        label_Pi = DotLabel("P'", dot_Pi, position = DOWN, color = self.color_inv)
        
        def get_lb():
            return LEFT_SIDE + UP * LEFT_SIDE[0] * np.tan(angle_tracker.get_value())
        def get_rb():
            return RIGHT_SIDE + UP * RIGHT_SIDE[0] * np.tan(angle_tracker.get_value())
        def is_oolb(m):
            return m.get_right()[0] < LEFT_SIDE[0]
        def is_oorb(m):
            return m.get_left()[0] > RIGHT_SIDE[0]

        oolb_arrow = Arrow(ORIGIN, LEFT, color = self.color_inv).scale(2)
        oolb_arrow.add_updater(lambda m: m.set_angle(angle_tracker.get_value() + PI))
        oolb_arrow.add_updater(lambda m: m.next_to(get_lb(), DOWN, aligned_edge = LEFT, buff = 0.2))
        oorb_arrow = Arrow(ORIGIN, RIGHT, color = self.color_inv).scale(2)
        oorb_arrow.add_updater(lambda m: m.set_angle(angle_tracker.get_value()))
        oorb_arrow.add_updater(lambda m: m.next_to(get_rb(), DOWN, aligned_edge = RIGHT, buff = 0.2))
        oolb_label = TexMobject("P'", color = self.color_inv, background_stroke_width = 0)
        oolb_label.add_updater(lambda m: m.next_to(oolb_arrow, DOWN, buff = 0.2))
        oorb_label = TexMobject("P'", color = self.color_inv, background_stroke_width = 0)
        oorb_label.add_updater(lambda m: m.next_to(oorb_arrow, DOWN, buff = 0.2))
        oolb_group = VGroup(oolb_arrow, oolb_label)
        oorb_group = VGroup(oorb_arrow, oorb_label)
        oolb_group.add_updater(lambda m: m.set_fill(opacity = 1 if is_oolb(label_Pi) else 0))
        oolb_group.add_updater(lambda m: m.set_stroke(opacity = 1 if is_oolb(label_Pi) else 0))
        oorb_group.add_updater(lambda m: m.set_fill(opacity = 1 if is_oorb(label_Pi) else 0))
        oorb_group.add_updater(lambda m: m.set_stroke(opacity = 1 if is_oorb(label_Pi) else 0))

        self.add(orig_line, inv_line, dot_P, dot_Pi, label_P, label_Pi)
        self.add(oolb_group, oorb_group)
        for d_position, d_angle in [(2, 0), (1, PI/10), (-5, 0), (-3, -PI/7), (4, PI/11)]:
            self.play(
                ApplyMethod(position_tracker.increment_value, d_position),
                ApplyMethod(angle_tracker.increment_value, d_angle),
                run_time = 2,
            )
            self.wait()
        self.play(
            ApplyMethod(angle_tracker.restore),
            ApplyMethod(position_tracker.restore),
            run_time = 2,
        )
        self.wait()


class LineToCircleInversion(Scene):
    CONFIG = {
        "color_circle" : YELLOW,
        "color_orig" : BLUE,
        "color_inv" : RED,
        "line_config" : {
            "stroke_width" : 2,
            "color" : WHITE,
        },
    }
    def construct(self):
        self.add_backgrounds()
        self.add_shapes()
        self.show_line_to_circle_inversion()

    def add_backgrounds(self):
        circle_O = Circle(radius = 3, color = self.color_circle)
        circle_O.shift(3*LEFT+0.5*UP)
        remark_circle = TextMobject("反演圆", color = self.color_circle)
        remark_circle.next_to(circle_O.get_bottom(), UP)
        dot_O = Dot(circle_O.get_center(), color = self.color_circle)
        label_O = DotLabel("O", dot_O, color = self.color_circle, position = DOWN)
        conclusion1 = TextMobject("不经过反演中心的直线", "$\\mapsto$", "经过反演中心的圆")
        conclusion1[0].set_color(self.color_orig)
        conclusion1[-1].set_color(self.color_inv)
        conclusion2 = TextMobject("经过反演中心的圆", "$\\mapsto$", "不经过反演中心的直线")
        conclusion2[0].set_color(self.color_inv)
        conclusion2[-1].set_color(self.color_orig)
        conclusions = VGroup(conclusion1, conclusion2)
        for c in conclusions:
            c.scale(0.8)
        conclusions.arrange_submobjects(DOWN, index_of_submobject_to_align = 1)
        conclusions.to_corner(DR)
        bg_rect = BackgroundRectangle(conclusions)
        self.add(circle_O, remark_circle)
        self.add_foreground_mobjects(dot_O, label_O, bg_rect, conclusions)
        self.dot_O = dot_O
        self.circle_O = circle_O
        self.conclusions = conclusions
        self.bg_rect = bg_rect

    def add_shapes(self):
        position_tracker = ValueTracker(2)
        line_angle_tracker = ValueTracker(PI*9/19)
        circle_angle_tracker = ValueTracker(PI/5)
        line = ExtendedLine(LEFT, RIGHT, color = self.color_orig)
        line.add_updater(lambda m: m.move_to(position_tracker.get_value() * RIGHT))
        line.add_updater(lambda m: m.rotate(line_angle_tracker.get_value() - m.get_angle()))
        inv_line = InversedVMobject(line, self.circle_O, use_dashed_vmob = False, color = self.color_inv)
        inv_line_center = SmallDot(color = self.color_inv)
        inv_line_center.add_updater(lambda m: m.move_to(inv_line.get_center()))
        dot_Ai = Dot(color = self.color_inv)
        dot_Ai.add_updater(
            lambda m: m.move_to(inv_line.get_center() * 2 - self.circle_O.get_center())
        )
        dot_Pi = Dot(color = self.color_inv)
        dot_Pi.add_updater(
            lambda m: m.move_to(
                inv_line.get_center() \
                + rotate_vector(
                    inv_line.get_center() - self.circle_O.get_center(),
                    circle_angle_tracker.get_value()
                )
            )
        )
        dot_P = InversedDot(dot_Pi, self.circle_O, is_hollow = False, color = self.color_orig)
        dot_A = InversedDot(dot_Ai, self.circle_O, is_hollow = False, color = self.color_orig)
        line_OA, line_OAi, line_OP, line_OPi, line_AP, line_AiPi = aux_lines = VGroup(*[
            TwoDotsSegment(pt_1, pt_2, **self.line_config)
            for pt_1, pt_2 in [
                (self.dot_O, dot_A), (self.dot_O, dot_Ai),
                (self.dot_O, dot_P), (self.dot_O, dot_Pi),
                (dot_A, dot_P), (dot_Ai, dot_Pi)
            ]
        ])
        ai_AiOPi = AngleIndicator(dot_Ai, self.dot_O, dot_Pi, color = MAROON_B, radius = 0.8)
        rtai_OAP = RightAngleIndicator(self.dot_O, dot_A, dot_P)
        rtai_OPiAi = RightAngleIndicator(self.dot_O, dot_Pi, dot_Ai)
        label_P = TexMobject("P", color = self.color_orig)
        label_Pi = TexMobject("P'", color = self.color_inv)
        label_A = TexMobject("A", color = self.color_orig)
        label_Ai = TexMobject("A'", color = self.color_inv)
        label_A.add_updater(
            lambda m: m.move_to(
                dot_A.get_center() + 0.3 * normalize(dot_A.get_center() - self.dot_O.get_center())
            )
        )
        label_P.add_updater(
            lambda m: m.move_to(
                dot_P.get_center() + 0.3 * normalize(dot_A.get_center() - self.dot_O.get_center())
            )
        )
        label_Ai.add_updater(
            lambda m: m.move_to(
                dot_Ai.get_center() + 0.4 * rotate_vector(
                    normalize(dot_Ai.get_center() - inv_line_center.get_center()), -PI/4
                )
            )
        )
        label_Pi.add_updater(
            lambda m: m.move_to(
                dot_Pi.get_center() + 0.4 * normalize(dot_Pi.get_center() - inv_line_center.get_center())
            )
        )

        def get_ub():
            return line.get_center() + TOP + RIGHT * TOP[1] / np.tan(line_angle_tracker.get_value())
        def get_bb():
            return line.get_center() + BOTTOM + RIGHT * BOTTOM[1] / np.tan(line_angle_tracker.get_value())
        def is_ooub(m):
            return m.get_bottom()[1] > TOP[1]
        def is_oobb(m):
            return m.get_top()[1] < BOTTOM[1]
        ooub_arrow = Arrow(ORIGIN, LEFT, color = self.color_orig).scale(2)
        ooub_arrow.add_updater(lambda m: m.set_angle(line_angle_tracker.get_value()))
        ooub_arrow.add_updater(lambda m: m.next_to(get_ub(), RIGHT, aligned_edge = TOP, buff = 0.2))
        oobb_arrow = Arrow(ORIGIN, RIGHT, color = self.color_orig).scale(2)
        oobb_arrow.add_updater(lambda m: m.set_angle(line_angle_tracker.get_value() + PI))
        oobb_arrow.add_updater(lambda m: m.next_to(get_bb(), RIGHT, aligned_edge = BOTTOM, buff = 0.2))
        oolb_label = TexMobject("P", color = self.color_orig, background_stroke_width = 0)
        oolb_label.add_updater(lambda m: m.next_to(ooub_arrow, RIGHT, buff = 0.2))
        oorb_label = TexMobject("P", color = self.color_orig, background_stroke_width = 0)
        oorb_label.add_updater(lambda m: m.next_to(oobb_arrow, RIGHT, buff = 0.2))
        ooub_group = VGroup(ooub_arrow, oolb_label)
        oobb_group = VGroup(oobb_arrow, oorb_label)
        ooub_group.add_updater(lambda m: m.set_fill(opacity = 1 if is_ooub(label_P) else 0))
        ooub_group.add_updater(lambda m: m.set_stroke(opacity = 1 if is_ooub(label_P) else 0))
        oobb_group.add_updater(lambda m: m.set_fill(opacity = 1 if is_oobb(label_P) else 0))
        oobb_group.add_updater(lambda m: m.set_stroke(opacity = 1 if is_oobb(label_P) else 0))

        self.add(line, inv_line)
        self.add(dot_A, dot_P, dot_Ai, dot_Pi)
        self.add(label_P, label_Pi, label_A, label_Ai)
        self.add(aux_lines)
        self.add(ai_AiOPi, rtai_OAP, rtai_OPiAi)
        self.add(ooub_group, oobb_group)

        self.position_tracker = position_tracker
        self.line_angle_tracker = line_angle_tracker
        self.circle_angle_tracker = circle_angle_tracker

    def show_line_to_circle_inversion(self):
        play_args = [
            [0, PI/12, 0, 2],
            [0, 0, PI*7/5, 4],
            [-2, PI/8, -PI/5, 3],
            [0, 0, PI*19/10, 6],
            [1.5, -PI/7, PI*2/5, 4],
        ]
        restore_arg = [
            -sum([arg[k] for arg in play_args])
            for k in range(len(play_args[0]))
        ]
        restore_arg[1] = (restore_arg[1] + PI) % (2*PI) - PI
        restore_arg[2] = (restore_arg[2] + PI) % (2*PI) - PI
        restore_arg[-1] = 3
        play_args.append(restore_arg)
        for d_center, d_line_angle, d_circle_angle, run_time in play_args:
            self.play(
                ApplyMethod(self.position_tracker.increment_value, d_center),
                ApplyMethod(self.line_angle_tracker.increment_value, d_line_angle),
                ApplyMethod(self.circle_angle_tracker.increment_value, d_circle_angle),
                run_time = run_time,
            )
            self.wait()


class InversionCreateSimilarTriangles(Scene):
    CONFIG = {
        "random_seed" : 5+7-0,
        "num_of_nudges" : 5,
        "max_step" : 1,
        "color_A" : RED,
        "color_B" : BLUE,
        "color_combined" : MAROON_B,
        "color_circle": YELLOW,
    }
    def construct(self):
        self.add_remark()
        self.show_figure_animation()

    def add_remark(self):
        cond_1 = TexMobject("{|OP|", "\\over", "|OQ|}", "=", "{|OQ'|", "\\over", "|OP'|}")
        cond_2 = TexMobject("\\angle POQ", "=", "\\angle Q'OP'")
        conds = VGroup(cond_1, cond_2)
        conds.arrange_submobjects(DOWN, buff = 0.5)
        conds_rect = SurroundingRectangle(conds, color = WHITE)
        arrow = TexMobject("\\Downarrow")
        arrow.next_to(conds_rect, DOWN)
        concl = TexMobject("\\triangle OPQ", "\\sim", "\\triangle OQ'P'")
        concl.next_to(arrow, DOWN)
        for mob in (cond_1[0], cond_1[2], concl[0]):
            mob.set_color(self.color_A)
        for mob in (cond_1[-1], cond_1[-3], concl[-1]):
            mob.set_color(self.color_B)
        for mob in (cond_2[0], cond_2[-1]):
            mob.set_color(self.color_combined)
        remark = VGroup(conds, conds_rect, arrow, concl)
        remark.to_corner(DR)
        self.add(remark)

    def show_figure_animation(self):
        circle = Circle(radius = 3, color = self.color_circle)
        circle.move_to(3.5*LEFT)
        dot_O = Dot(color = self.color_combined)
        dot_O.add_updater(lambda m: m.move_to(circle.get_center()))
        dot_P = Dot(point = 1.2*UP+LEFT, color = self.color_A)
        dot_Q = Dot(point = 0.5*DOWN+1.9*LEFT, color = self.color_A)
        dot_Pi = InversedDot(dot_P, circle, is_hollow = False, color = self.color_B)
        dot_Qi = InversedDot(dot_Q, circle, is_hollow = False, color = self.color_B)
        triangle_OPQ = ManyDotsPolygon(
            dot_O, dot_P, dot_Q, color = self.color_A,
            stroke_width = 5, fill_opacity = 0.4
        )
        triangle_OPiQi = ManyDotsPolygon(
            dot_O, dot_Pi, dot_Qi, color = self.color_B,
            stroke_width = 2, fill_opacity = 0.3
        )
        label_O, label_P, label_Pi, label_Q, label_Qi = (
            DotLabel(
                text, dot, color = color, position = position,
                background_stroke_width = 5,
            ).scale(0.8)
            for text, dot, color, position in zip(
                ["O", "P", "P'", "Q", "Q'"],
                [dot_O, dot_P, dot_Pi, dot_Q, dot_Qi],
                [self.color_combined, self.color_A, self.color_B, self.color_A, self.color_B],
                [LEFT, UP, UP, DOWN, DOWN]
            )
        )
        self.add(dot_O, dot_P, dot_Q, dot_Pi, dot_Qi)
        self.add(circle, triangle_OPQ, triangle_OPiQi)
        self.add(label_O, label_P, label_Pi, label_Q, label_Qi)
        dot_P.save_state()
        dot_Q.save_state()
        for k in range(self.num_of_nudges):
            nudge_P = get_random_vector(self.max_step)
            nudge_Q = get_random_vector(self.max_step)
            self.play(
                ApplyMethod(dot_P.shift, nudge_P),
                ApplyMethod(dot_Q.shift, nudge_Q),
                run_time = 2
            )
            self.wait()
        self.play(dot_P.restore, dot_Q.restore, run_time = 2)
        self.wait()


class CircleToCircleInversionProof(Scene):
    CONFIG = {
        "color_O" : YELLOW,
        "color_A" : RED,
        "color_B" : BLUE,
        "color_combined" : MAROON_B,
        "label_buff" : 0.1,
        "label_scaling_factor" : 0.75,
        "line_config" : {
            "stroke_width" : 2,
            "color" : WHITE,
        },
    }
    def construct(self):
        self.add_backgrounds()
        self.show_left_and_right_points()
        self.show_random_point()
        self.show_similar_triangles()
        self.show_complementary_property()
        self.show_inversion_result()

    def add_backgrounds(self):
        circle_O = Circle(radius = 3.2, color = self.color_O)
        circle_O.shift(3.5*LEFT)
        dot_O = Dot(circle_O.get_center(), color = self.color_O)
        remark_O = TextMobject("反演圆", color = YELLOW)
        remark_O.next_to(circle_O.get_bottom(), UP, buff = 0.4)
        circle_C = Circle(radius = 0.8, stroke_width = 2)
        circle_C.next_to(circle_O.get_right(), LEFT, buff = 0.5)
        dot_C = Dot(circle_C.get_center())
        label_O, label_C = (
            DotLabel(
                text, dot, color = color, position = DOWN, label_buff = self.label_buff
            ).scale(self.label_scaling_factor)
            for text, dot, color in zip(["O", "C"], [dot_O, dot_C], [self.color_O, WHITE])
        )
        for orig_mob in (circle_C, dot_C, label_C):
            orig_mob.set_sheen_direction(RIGHT)
            orig_mob.set_color([self.color_A, self.color_B])
        inv_circle_template = InversedVMobject(circle_C, circle_O, use_dashed_vmob = False)
        inv_circle = Circle(radius = inv_circle_template.get_width()/2)
        inv_circle.move_to(inv_circle_template.get_center())
        inv_circle.set_sheen_direction(LEFT)
        inv_circle.set_color([self.color_A, self.color_B])
        self.add(circle_O, dot_O, circle_C, dot_C)
        self.add(label_O, label_C)
        self.add(remark_O)
        self.wait()

        self.circle_O = circle_O
        self.dot_O = dot_O
        self.remark_O = remark_O
        self.circle_C = circle_C
        self.dot_C = dot_C
        self.inv_circle = inv_circle

    def show_left_and_right_points(self):
        dot_A = Dot(color = self.color_A)
        dot_A.move_to(self.circle_C.get_left())
        dot_B = Dot(color = self.color_B)
        dot_B.move_to(self.circle_C.get_right())
        dot_Ai = InversedDot(dot_A, self.circle_O, is_hollow = False, color = self.color_A)
        dot_Bi = InversedDot(dot_B, self.circle_O, is_hollow = False, color = self.color_B)
        dot_Q = Dot((dot_Ai.get_center() + dot_Bi.get_center()) / 2)
        line_OB = Line(self.dot_O.get_center(), dot_B.get_center(), **self.line_config)
        line_OAi = Line(self.dot_O.get_center(), dot_Ai.get_center(), **self.line_config)
        label_A, label_Ai, label_B, label_Bi = (
            DotLabel(
                text, dot, color = color, position = position, label_buff = self.label_buff
            ).scale(self.label_scaling_factor)
            for text, dot, color, position in zip(
                ["A", "A'", "B", "B'"],
                [dot_A, dot_Ai, dot_B, dot_Bi],
                [self.color_A, self.color_A, self.color_B, self.color_B],
                [DL, DR, DR, DL]
            )
        )
        remark_AB = TextMobject("圆心连线 \\\\ 的交点...").scale(0.6)
        remark_AB.next_to(VGroup(dot_A, dot_B), DOWN, buff = 1)
        arrows_AB = VGroup(*[
            Arrow(remark_AB.get_critical_point(direction), dot, buff = 0.1)
            for direction, dot in zip([UL, UR], [dot_A, dot_B])
        ])
        remark_AiBi = TextMobject("...以及它们的反点").scale(0.8)
        remark_AiBi.next_to(VGroup(dot_Ai, dot_Bi), DOWN, buff = 1)
        arrows_AiBi = VGroup(*[
            Arrow(remark_AiBi.get_critical_point(direction), dot, buff = 0.1)
            for direction, dot in zip([UR, UL], [dot_Ai, dot_Bi])
        ])
        self.play(ShowCreation(line_OB))
        self.play(Write(dot_A), Write(dot_B), Write(label_A), Write(label_B))
        self.wait()
        self.play(Write(remark_AB), ShowCreation(arrows_AB))
        self.wait()
        self.play(
            ReplacementTransform(dot_A.deepcopy(), dot_Ai),
            ReplacementTransform(dot_B.deepcopy(), dot_Bi),
        )
        self.play(Write(label_Ai), Write(label_Bi))
        self.wait()
        self.play(
            ReplacementTransform(remark_AB, remark_AiBi),
            ReplacementTransform(arrows_AB, arrows_AiBi)
        )
        self.play(ReplacementTransform(line_OB, line_OAi))
        self.play(FadeOut(VGroup(remark_AiBi, arrows_AiBi)))
        self.wait()

        self.dot_A = dot_A
        self.dot_Ai = dot_Ai
        self.dot_B = dot_B
        self.dot_Bi = dot_Bi
        self.dot_Q = dot_Q
        self.line_OAi = line_OAi
        self.dots_AB = VGroup(dot_A, dot_Ai, dot_B, dot_Bi)
        self.labels_AB = VGroup(label_A, label_Ai, label_B, label_Bi)

    def show_random_point(self):
        angle_tracker = ValueTracker(PI/3)
        dot_P = Dot()
        dot_P.add_updater(
            lambda m: m.move_to(
                self.circle_C.point_at_angle(angle_tracker.get_value() % TAU)
            )
        )
        dot_P.add_updater(
            lambda m: m.set_color(
                interpolate_color(
                    self.color_A, self.color_B,
                    (dot_P.get_center()[0] - self.dot_A.get_center()[0]) / (self.dot_B.get_center()[0] - self.dot_A.get_center()[0])
                )
            )
        )
        label_P = DotLabel("P", dot_P, position = None)
        label_P.scale(0.8)
        label_P.add_updater(lambda m: m.set_color(dot_P.get_color()))
        label_P.add_updater(
            lambda m: m.move_to(dot_P.get_center() * 1.4 - self.dot_C.get_center() * 0.4)
        )
        arrow_P = Vector(DR, buff = 0, color = WHITE).scale(0.5)
        arrow_P.add_updater(lambda m: m.next_to(dot_P, UL, buff = 0.1))
        remark_P = TextMobject("圆上任意一点...").scale(0.75)
        remark_P.add_updater(lambda m: m.next_to(arrow_P, UL, buff = 0.1))
        dot_Pi = InversedDot(dot_P, self.circle_O, is_hollow = False)
        dot_Pi.add_updater(lambda m: m.set_color(dot_P.get_color()))
        label_Pi = DotLabel("P'", dot_Pi, position = None)
        label_Pi.scale(0.8)
        label_Pi.add_updater(lambda m: m.set_color(dot_Pi.get_color()))
        label_Pi.add_updater(
            lambda m: m.move_to(dot_Pi.get_center() * 1.1 - self.inv_circle.get_center() * 0.1)
        )
        arrow_Pi = Vector(DL, buff = 0, color = WHITE).scale(0.5)
        arrow_Pi.add_updater(lambda m: m.next_to(dot_Pi, UR, buff = 0.1))
        remark_Pi = TextMobject("...以及它的反点").scale(0.75)
        remark_Pi.add_updater(lambda m: m.next_to(arrow_Pi, UR, buff = 0.1))
        line_OP, line_OPi, line_AP, line_AiPi, line_BP, line_BiPi = aux_lines = VGroup(*[
            TwoDotsSegment(pt_1, pt_2, **self.line_config)
            for pt_1, pt_2 in [
                (self.dot_O, dot_P), (self.dot_O, dot_Pi), (self.dot_A, dot_P),
                (self.dot_Ai, dot_Pi), (self.dot_B, dot_P), (self.dot_Bi, dot_Pi)
            ]
        ])
        rtai_APB = RightAngleIndicator(self.dot_A, dot_P, self.dot_B)
        rtai_BiPiAi = RightAngleIndicator(self.dot_Bi, dot_Pi, self.dot_Ai, side_length = 0.5)
        self.play(Write(dot_P), Write(label_P))
        self.play(ShowCreation(arrow_P), Write(remark_P))
        self.play(Write(line_AP), Write(line_BP))
        self.play(ShowCreation(rtai_APB))
        self.wait()
        self.play(ReplacementTransform(dot_P.deepcopy(), dot_Pi))
        self.play(Write(label_Pi))
        self.play(
            ReplacementTransform(arrow_P.deepcopy(), arrow_Pi),
            ReplacementTransform(remark_P.deepcopy(), remark_Pi),
        )
        self.play(angle_tracker.increment_value, PI/6, run_time = 2)
        self.play(FadeOut(VGroup(arrow_P, remark_P, arrow_Pi, remark_Pi)))
        self.wait()
        self.play(Write(VGroup(line_OP, line_OPi, line_AiPi, line_BiPi)))
        self.wait()

        self.dot_P = dot_P
        self.dot_Pi = dot_Pi
        self.rtai_APB = rtai_APB
        self.rtai_BiPiAi = rtai_BiPiAi
        self.angle_tracker = angle_tracker
        self.aux_lines = aux_lines
        self.dots_P = VGroup(dot_P, dot_Pi)
        self.labels_P = VGroup(label_P, label_Pi)
        self.rtais = VGroup(self.rtai_APB, self.rtai_BiPiAi)

    def show_similar_triangles(self):
        ai_OAP = AngleIndicator(self.dot_O, self.dot_A, self.dot_P, radius = 0.3, color = self.color_A)
        ai_OBP = AngleIndicator(self.dot_O, self.dot_B, self.dot_P, radius = 0.4, color = self.color_B)
        ai_OPiAi = AngleIndicator(self.dot_O, self.dot_Pi, self.dot_Ai, radius = 0.3, color = self.color_A)
        ai_OPiBi = AngleIndicator(self.dot_O, self.dot_Pi, self.dot_Bi, radius = 0.4, color = self.color_B)
        triangle_OAP, triangle_OPiAi, triangle_OBP, triangle_OPiBi = [
            ManyDotsPolygon(
                pt_1, pt_2, pt_3, color = self.color_combined,
                stroke_width = 0, fill_opacity = 0.4
            )
            for pt_1, pt_2, pt_3 in (
                (self.dot_O, self.dot_A, self.dot_P),
                (self.dot_O, self.dot_Pi, self.dot_Ai),
                (self.dot_O, self.dot_B, self.dot_P),
                (self.dot_O, self.dot_Pi, self.dot_Bi),
            )
        ]
        remark_sim_A = TexMobject("\\triangle OAP", "\\sim", "\\triangle OP'A'")
        remark_sim_B = TexMobject("\\triangle OBP", "\\sim", "\\triangle OP'B'")
        remark_arrow = TexMobject("\\Downarrow")
        remark_angle_A = TexMobject("\\angle OAP", "=", "\\angle OP'A'")
        remark_angle_B = TexMobject("\\angle OBP", "=", "\\angle OP'B'")
        remarks_A = VGroup(remark_sim_A, remark_arrow, remark_angle_A)
        remarks_B = VGroup(remark_sim_B, remark_arrow, remark_angle_B)
        remarks_A.arrange_submobjects(DOWN)
        remarks_A.next_to(self.dot_Q, DOWN, buff = 1)
        remark_sim_B.move_to(remark_sim_A.get_center())
        remark_angle_B.move_to(remark_angle_A.get_center())
        for remark, color in ([remark_sim_A, self.color_combined], [remark_sim_B, self.color_combined], \
                              [remark_angle_A, self.color_A], [remark_angle_B, self.color_B]):
            remark[0].set_color(color)
            remark[-1].set_color(color)
        self.play(Write(remark_sim_A))
        self.play(FadeInFromDown(VGroup(remark_arrow, remark_angle_A)))
        self.wait()
        self.play(ShowCreation(triangle_OAP), ShowCreation(ai_OAP))
        self.wait()
        self.play(
            ReplacementTransform(triangle_OAP, triangle_OPiAi),
            ReplacementTransform(ai_OAP.deepcopy(), ai_OPiAi),
        )
        self.play(FadeOut(triangle_OPiAi))
        self.wait()
        self.play(ReplacementTransform(remarks_A, remarks_B))
        self.wait()
        self.play(ShowCreation(triangle_OBP), ShowCreation(ai_OBP))
        self.wait()
        self.play(
            ReplacementTransform(triangle_OBP, triangle_OPiBi),
            ReplacementTransform(ai_OBP.deepcopy(), ai_OPiBi),
        )
        self.play(FadeOut(remarks_B), FadeOut(triangle_OPiBi))
        self.wait()

        self.ai_OAP = ai_OAP
        self.ai_OBP = ai_OBP
        self.ai_OPiAi = ai_OPiAi
        self.ai_OPiBi = ai_OPiBi
        self.ais = VGroup(ai_OAP, ai_OBP, ai_OPiAi, ai_OPiBi)

    def show_complementary_property(self):
        ai_OAP_copy = self.ai_OAP.deepcopy()
        ai_OBP_copy = self.ai_OBP.deepcopy()
        rtai_APB_copy = self.rtai_APB.deepcopy()
        for ai_copy in (ai_OAP_copy, ai_OBP_copy, rtai_APB_copy):
            ai_copy.clear_updaters()
        comp_prop = VGroup(ai_OAP_copy, TexMobject("="), ai_OBP_copy, TexMobject("+"), rtai_APB_copy)
        comp_prop.arrange_submobjects(RIGHT)
        comp_prop.scale(1.2)
        comp_prop.next_to(self.circle_O.get_top(), DOWN, buff = 1)
        self.play(
            ReplacementTransform(self.ai_OAP.deepcopy(), ai_OAP_copy),
            ReplacementTransform(self.ai_OBP.deepcopy(), ai_OBP_copy),
            ReplacementTransform(self.rtai_APB.deepcopy(), rtai_APB_copy),
        )
        self.play(Write(comp_prop[1]), Write(comp_prop[3]))
        self.wait()
        self.play(ReplacementTransform(rtai_APB_copy.deepcopy(), self.rtai_BiPiAi))
        self.wait()
        for ai in self.ais:
            ai.clear_updaters()
        self.play(
            FadeOut(comp_prop),
            FadeOut(self.ais),
            FadeOut(self.labels_AB), FadeOut(self.labels_P),
        )
        self.wait()

    def show_inversion_result(self):
        inv_circle_copy = self.inv_circle.deepcopy()
        self.play(self.angle_tracker.set_value, PI, run_time = 2)
        self.wait()
        def update_inv_circle(inv_circle):
            angle = self.angle_tracker.get_value()
            if (angle <= -PI) or (angle > PI):
                alpha = 1
            else:
                QPi = self.dot_Pi.get_center() - self.dot_Q.get_center()
                QAi = self.dot_Ai.get_center() - self.dot_Q.get_center()
                theta = angle_between(QPi, QAi)
                if self.dot_Pi.get_center()[1] < self.dot_Q.get_center()[1]:
                    theta = 2*PI - theta
                alpha = theta / (2*PI)
            inv_circle.become(inv_circle_copy.get_subcurve(0, alpha))
        self.inv_circle.add_updater(update_inv_circle)
        self.add(self.inv_circle)
        self.play(
            ApplyMethod(self.angle_tracker.increment_value, -2*PI),
            run_time = 5,
        )
        self.inv_circle.clear_updaters()
        for line in self.aux_lines:
            line.clear_updaters()
        self.play(
            FadeOut(self.dots_AB), FadeOut(self.dots_P), FadeOut(self.rtais),
            FadeOut(self.line_OAi), FadeOut(self.aux_lines)
        )
        self.wait()
        color_template = Square(
            stroke_width = 0, fill_opacity = 1, fill_color = [self.color_A, self.color_B]
        )
        conclusion = TextMobject("不经过反演中心的圆", "$\\mapsto$", "不经过反演中心的圆")
        conclusion.scale(0.8)
        conclusion[0].set_color_by_gradient(self.color_A, self.color_B)
        conclusion[2].set_color_by_gradient(self.color_B, self.color_A)
        conclusion.to_corner(DR)
        self.play(Write(conclusion))
        self.wait(3)
        self.play(FadeOut(conclusion), FadeOut(self.inv_circle))
        self.wait()


class ConcentricPropertyDoesNotHold(Scene):
    def setup(self):
        N = 8
        self.circle_radii = [0.9-0.1*k for k in range(N)]
        self.dot_radii = [0.08-0.005*k for k in range(N)]
        self.circle_colors = color_gradient([BLUE, GREEN, RED], N)

    def construct(self):
        orig_circles = VGroup(*[
            Circle(radius = radius, stroke_width = 1.5,color = color)
            for radius, color in zip(self.circle_radii, self.circle_colors)]
        )
        orig_circles.shift(2*LEFT+0.5*DOWN)
        orig_circles_centers = VGroup(*[
            Dot(circle.get_center(), radius = radius, color = color)
            for circle, radius, color in zip(orig_circles, self.dot_radii, self.circle_colors)
        ])
        # Dot(orig_circles.get_center())
        circle = Circle(radius = 3, color = YELLOW)
        circle.shift(3.8*LEFT+0.5*DOWN)
        circle_center = Dot(circle.get_center(), color = YELLOW)
        inv_circles = VGroup(*[
            InversedVMobject(orig_circle, circle).clear_updaters().set_color(color)
            for orig_circle, color in zip(orig_circles, self.circle_colors)
        ])
        inv_circles_centers = VGroup(*[
            Dot(inv_circle.get_center(), color = color)
            for inv_circle, color in zip(inv_circles, self.circle_colors)
        ])

        circle_text = TextMobject("反演圆", color = YELLOW)
        circle_text.next_to(circle.get_bottom(), UP, buff = 0.4)
        orig_circles_text = TextMobject("同心的圆", color = WHITE)
        orig_circles_text.next_to(orig_circles, UP)
        orig_circles_text.to_edge(UP, buff = 0.4)
        inv_circles_text = TextMobject("不同心的像", color = WHITE)
        inv_circles_text.next_to(inv_circles, UP)
        inv_circles_text.to_edge(UP, buff = 0.4)
        arrow = Arrow(orig_circles_text.get_right(), inv_circles_text.get_left())

        self.add(circle, circle_center)
        self.add(orig_circles, orig_circles_centers)
        self.add(inv_circles, inv_circles_centers)
        self.add(circle_text, orig_circles_text, inv_circles_text, arrow)
        self.wait()


class DemonstratePtolemyInequality(Scene):
    CONFIG = {
        "R" : 2.7,
        "angle_A" : -PI*2/3,
        "angle_B" : PI*4/5,
        "angle_D" : -PI/5,
        "radius_C" : 3.2,
        "angle_C" : PI/5,
    }
    def construct(self):
        radius_tracker = ValueTracker(self.radius_C)
        angle_tracker = ValueTracker(self.angle_C)
        circle = Circle(radius = self.R, color = WHITE, stroke_width = 1)
        circle.shift(DOWN)
        dashed_circle = DashedVMobject(circle, num_dashes = 100, positive_space_ratio = 0.5)
        dot_A, dot_B, dot_C, dot_D = dots = VGroup(*[
            Dot(circle.point_at_angle(angle % TAU), color = WHITE)
            for angle in (self.angle_A, self.angle_B, self.angle_C, self.angle_D)
        ])
        dot_C.add_updater(
            lambda m: m.move_to(
                circle.get_center() + radius_tracker.get_value() * \
                rotate_vector(RIGHT, angle_tracker.get_value())
            )
        )
        dot_labels = VGroup(*[
            DotLabel(text, dot, position = position, label_buff = 0.1)
            for text, dot, position in zip(
                ["A", "B", "C", "D"], dots, [DL, UL, UR, DR]
            )
        ])
        lines = VGroup(*[
            TwoDotsSegment(dot_1, dot_2)
            for dot_1, dot_2 in (
                [dot_B, dot_A], [dot_A, dot_C], [dot_A, dot_D],
                [dot_B, dot_C], [dot_B, dot_D], [dot_C, dot_D],
            )
        ])
        length_labels = VGroup(*[LengthLabel(line) for line in lines])
        length_labels[0].switch_side()
        length_labels[2].switch_side()
        length_labels[1].set_offset(-0.4)
        length_labels[-2].set_offset(-0.4)

        def get_sums():
            AB, AC, AD, BC, BD, CD = [line.get_length() for line in lines]
            sum_lhs = AB * CD + AD * BC
            sum_rhs = AC * BD
            return sum_lhs, sum_rhs
        relation_eq = TexMobject(
            "|AB| \\cdot |CD| + |AD| \\cdot |BC|", "=", "|AC| \\cdot |BD|",
            background_stroke_width = 0,
        )
        relation_neq = TexMobject(
            "|AB| \\cdot |CD| + |AD| \\cdot |BC|", ">", "|AC| \\cdot |BD|",
            background_stroke_width = 0,
        )
        relation_eq[1].set_color(GREEN)
        relation_neq[1].set_color(RED)
        relation_eq.to_edge(UP, buff = 1.2)
        for eq_mob, neq_mob in zip(relation_eq, relation_neq):
            neq_mob.move_to(eq_mob.get_center())
        lhs, eq_sign, rhs = relation_eq
        neq_sign = relation_neq[1]
        label_lhs = DecimalNumber(num_decimal_places = 4, show_ellipsis = True)
        label_rhs = DecimalNumber(num_decimal_places = 4, show_ellipsis = True)
        label_lhs.add_updater(lambda m: m.set_value(get_sums()[0]))
        label_rhs.add_updater(lambda m: m.set_value(get_sums()[1]))
        brace_lhs = Brace(lhs, UP, buff = 0.1)
        brace_rhs = Brace(rhs, UP, buff = 0.1)
        brace_lhs.put_at_tip(label_lhs)
        brace_rhs.put_at_tip(label_rhs)

        def get_indication_color(thres = 1e-2):
            return GREEN if is_close(radius_tracker.get_value(), self.R, thres = thres) else RED
        def get_indication_opacity(thres = 1e-2):
            return 0 if is_close(radius_tracker.get_value(), self.R, thres = thres) else 1
        figure_group = VGroup(dashed_circle, dots, lines, length_labels, dot_labels)
        figure_group.add_updater(lambda m: m.set_color(get_indication_color()))
        relation_group = VGroup(lhs, eq_sign, rhs, neq_sign, brace_lhs, brace_rhs, label_lhs, label_rhs)
        label_lhs.add_updater(lambda m: m.set_color(get_indication_color()))
        label_rhs.add_updater(lambda m: m.set_color(get_indication_color()))
        eq_sign.add_updater(lambda m: m.set_opacity(1 - get_indication_opacity()))
        neq_sign.add_updater(lambda m: m.set_opacity(get_indication_opacity()))
        self.add(figure_group)
        self.add(relation_group)

        deltas = [
            (0.5, -0.1), (0, -0.4), (-1, 0.3), (0, 0.4),
            (-1, 0), (0.3, -0.2), (0.7, -0.3),
        ]
        radius_tracker.save_state()
        angle_tracker.save_state()
        for d_radius, d_angle in deltas:
            self.play(
                ApplyMethod(radius_tracker.increment_value, d_radius),
                ApplyMethod(angle_tracker.increment_value, d_angle),
                run_time = 2,
            )
            self.wait()
        self.play(
            ApplyMethod(radius_tracker.restore),
            ApplyMethod(angle_tracker.restore),
            run_time = 2,
        )
        self.wait()


class PtolemyInversionFigure(Scene):
    CONFIG = {
        "R" : 3.8,
        "r" : 1.3,
        "angle_A" : PI,
        "angle_B" : PI/3,
        "angle_C" : -PI/9,
        "angle_D" : -PI*2/7,
        "color_circle" : YELLOW,
        "color_ABD" : BLUE,
    }
    def construct(self):
        circle_ABD = Circle(radius = self.r, color = self.color_ABD, stroke_width = 3)
        circle_ABD.shift(0.2*LEFT)
        dot_A, dot_B, dot_C, dot_D = dots = VGroup(*[
            Dot(circle_ABD.point_at_angle(angle % TAU), color = WHITE)
            for angle in (self.angle_A, self.angle_B, self.angle_C, self.angle_D)
        ])
        dot_A.set_color(self.color_circle)
        dot_C.shift(0.4*RIGHT)
        circle = Circle(radius = self.R, color = self.color_circle, stroke_width = 5)
        circle.move_to(dot_A.get_center())
        remark_circle = TextMobject("反演圆", color = self.color_circle)
        remark_circle.next_to(circle.get_bottom(), UP)
        label_A, label_B, label_C, label_D = dot_labels = VGroup(*[
            DotLabel(text, dot, position = position, label_buff = 0.2)
            for text, dot, position in zip(
                ["A", "B", "C", "D"], dots, [DL, UP, DOWN, DOWN]
            )
        ])
        label_A.set_color(self.color_circle)
        dot_Bi, dot_Ci, dot_Di = inv_dots = VGroup(*[
            InversedDot(dot, circle, is_hollow = False, color = WHITE)
            for dot in (dot_B, dot_C, dot_D)
        ])
        label_Bi, label_Ci, label_Di = inv_dot_labels = VGroup(*[
            DotLabel(text, dot, position = RIGHT, label_buff = 0.2)
            for text, dot in zip(["B'", "C'", "D'"], [dot_Bi, dot_Ci, dot_Di])
        ])
        lines = VGroup(*[
            TwoDotsSegment(dot_1, dot_2, stroke_width = 1)
            for dot_1, dot_2 in (
                [dot_A, dot_B], [dot_A, dot_C], [dot_A, dot_D],
                [dot_B, dot_C], [dot_B, dot_D], [dot_C, dot_D],
                [dot_A, dot_Bi], [dot_A, dot_Ci], [dot_A, dot_Di],
                [dot_Bi, dot_Ci], [dot_Bi, dot_Di], [dot_Ci, dot_Di],
            )
        ])
        inv_circle_ABD = InversedVMobject(circle_ABD, circle, use_dashed_vmob = False)
        inv_circle_ABD.add_updater(lambda m: m.set_color(self.color_ABD))
        inv_circle_ABD.add_updater(lambda m: m.set_stroke(width = 2))
        self.add(circle, remark_circle, circle_ABD, inv_circle_ABD)
        self.add(dots, dot_labels, inv_dots, inv_dot_labels, lines)
        self.add()
        self.wait()


#####
## Inversion Advanced P1 Scenes
class KissingCirclesPuzzle(Scene):
    def construct(self):
        self.show_figure()
        self.show_question()

    def show_figure(self):
        type_text_1 = TextMobject("外切-外切-外切")
        type_text_2 = TextMobject("内切-内切-外切")
        type_text_1.move_to(LEFT_SIDE/2)
        type_text_2.move_to(RIGHT_SIDE/2)
        type_text_1.to_edge(DOWN)
        type_text_2.to_edge(DOWN)
        dot_l1, dot_l2, dot_l3 = dots_l = VGroup(*[
            VectorizedPoint(np.array([coords[0], coords[1], 0]), color = BLUE)
            for coords in [(-3.9, 1.5), (-4.9, 0.0), (-2.8, -1.0)]
        ])
        dot_r1, dot_r2, dot_r3 = dots_r = VGroup(*[
            VectorizedPoint(np.array([coords[0], coords[1], 0]), color = BLUE)
            for coords in [(4.6, 0.3), (3.9, 0.6), (3.5, 1.6)]
        ])
        dfc_l = DescartesFourCircles(*dots_l, show_new_circles = False)
        dfc_r = DescartesFourCircles(*dots_r, show_new_circles = False, outer_circle_index = 2)
        for dfc in [dfc_l, dfc_r]:
            for mob in dfc.get_orig_circles():
                mob.set_stroke(width = 2, color = BLUE)
        self.add(type_text_1, type_text_2)
        self.add(dfc_l, dfc_r)
        self.dfc_l = dfc_l
        self.dfc_r = dfc_r
        self.dots_l = dots_l
        self.dots_r = dots_r

    def show_question(self):
        question = TextMobject("能否添加第四个圆，使之与其他三个圆都相切？")
        question.to_edge(UP, buff = 0.2)
        self.add(question)
        self.wait()

        
class KissingCirclesSimplified(Scene):
    def construct(self):
        line1 = ExtendedLine(UL, UR)
        line2 = ExtendedLine(DL, DR)
        center_circle = Circle(radius = 1)
        figure_group = VGroup(line1, line2, center_circle)
        for mob in figure_group:
            mob.set_stroke(width = 2, color = BLUE)
        question = TextMobject("能否添加第四个“圆”，使之与其他三个“圆”都相切？")
        question.next_to(figure_group, UP, buff = 0.5)
        group = VGroup(question, figure_group)
        group.move_to(ORIGIN)
        self.add(group)
        self.wait()


class KissingCirclesSimplifiedAnswer(Scene):
    def construct(self):
        line1 = ExtendedLine(UL, UR, stroke_width = 2, color = BLUE)
        line2 = ExtendedLine(DL, DR, stroke_width = 2, color = BLUE)
        center_circle = Circle(radius = 1, stroke_width = 2, color = BLUE)
        new_circles = VGroup(*[
            Circle(radius = 1, color = color, fill_opacity = 0.1, stroke_width = 5) \
            .next_to(center_circle, direction, buff = 0)
            for direction, color in zip([LEFT, RIGHT], [RED, ORANGE])
        ])
        numbers = VGroup(*[
            TexMobject(f"{num}", color = circle.get_color()).move_to(circle.get_center())
            for num, circle in zip(["1", "2"], new_circles)
        ])
        group = VGroup(line1, line2, center_circle, new_circles, numbers)
        group.move_to(ORIGIN)
        self.add(group)
        self.wait()


class KissingCirclesSimplifiedExplanation(Scene):
    CONFIG = {
        "dashed_vmob_config" : {
            "num_dashes" : 30,
            "positive_space_ratio" : 0.6,
        },
        "line_colors" : [GREEN, BLUE],
        "center_color" : MAROON_B,
        "circle_colors" : [RED, ORANGE],
    }
    def construct(self):
        self.add_backgrounds()
        self.show_process()

    def add_backgrounds(self):
        N = 5
        line1 = Line(UP + N*LEFT, UP + N*RIGHT, stroke_width = 2, color = self.line_colors[0])
        line2 = Line(DOWN + N*LEFT, DOWN + N*RIGHT, stroke_width = 2, color = self.line_colors[1])
        center_circle = FineCircle(radius = 1, stroke_width = 2, color = self.center_color)
        new_circle1 = FineCircle(radius = 1, stroke_width = 5, color = self.circle_colors[0])
        new_circle1.next_to(center_circle, LEFT, buff = 0)
        new_circle2 = FineCircle(radius = 1, stroke_width = 5, color = self.circle_colors[1])
        new_circle2.next_to(center_circle, RIGHT, buff = 0)
        inv_old_group = VGroup(line1, line2, center_circle)
        inv_new_group = VGroup(new_circle1, new_circle2)
        inv_group = VGroup(inv_old_group, inv_new_group)
        inv_group.rotate(-PI*2/5)
        inv_group.shift(3*RIGHT)
        circle = FineCircle(radius = 3.5, color = YELLOW)
        circle.shift(2*LEFT)
        circle_center = Dot(circle.get_center(), color = YELLOW)
        remark_circle = TextMobject("反演圆", color = YELLOW)
        remark_circle.next_to(circle.get_bottom(), UP)
        remark_center = VGroup(*[
            Arrow(DL, UR, color = YELLOW, buff = 0).scale(0.3),
            TextMobject("反演中心", color = YELLOW).scale(0.8),
        ])
        remark_center.arrange_submobjects(DL, buff = 0)
        remark_center.next_to(circle_center, DL, buff = 0.1)
        orig_old_group = VGroup(*[
            InversedVMobject(mob, circle, use_dashed_vmob = False, match_original_style = True)
            for mob in inv_old_group
        ])
        orig_new_group = VGroup(*[
            InversedVMobject(mob, circle, use_dashed_vmob = False, match_original_style = True)
            for mob in inv_new_group
        ])
        for mob in orig_old_group:
            mob.clear_updaters()
            mob.set_stroke(width = 2)
        for mob in orig_new_group:
            mob.clear_updaters()
            mob.set_stroke(width = 5)
            mob.set_fill(opacity = 0.1)
        self.add(orig_old_group)
        self.add(circle, circle_center, remark_circle, remark_center)
        self.circle = circle
        self.inv_old_group = inv_old_group
        self.inv_new_group = inv_new_group
        self.orig_old_group = orig_old_group
        self.orig_new_group = orig_new_group
    
    def show_process(self):
        dashed_inv_old_group = VGroup(*[
            DashedVMobject(mob, **self.dashed_vmob_config)
            for mob in self.inv_old_group
        ])
        dashed_inv_new_group = VGroup(*[
            DashedVMobject(mob, **self.dashed_vmob_config)
            for mob in self.inv_new_group
        ])
        self.play(ShowCreation(dashed_inv_old_group, lag_ratio = 0.05), run_time = 3)
        self.wait()
        dashed_copys = VGroup(*[dashed_inv_old_group[-1].deepcopy() for k in range(2)])
        dashed_copys.generate_target()
        for mob_copy, mob_template in zip(dashed_copys.target, dashed_inv_new_group):
            mob_copy.match_style(mob_template)
            mob_copy.move_to(mob_template.get_center())
        self.play(MoveToTarget(dashed_copys), run_time = 3)
        self.remove(dashed_copys)
        self.add(dashed_inv_new_group)
        self.wait()
        self.play(DrawBorderThenFill(self.orig_new_group), run_time = 3)
        self.wait(2)
        self.play(
            FadeOut(dashed_inv_new_group),
            FadeOut(dashed_inv_old_group),
            FadeOut(self.orig_new_group),
        )
        self.wait()


class DifferentTangentTypesWithSameConclusion(KissingCirclesPuzzle):
    CONFIG = {
        "random_seed" : 570,
        "num_of_nudges" : 5, 
        "max_step" : 0.5,
        "color_1" : ORANGE,
        "color_2" : RED,
    }
    def construct(self):
        super().show_figure()
        self.dots_l.save_state()
        self.dots_r.save_state()
        for dfc in [self.dfc_l, self.dfc_r]:
            dfc.add_new_circles()
            dfc.get_orig_circles().set_stroke(width = 2)
            c4_1, c4_2 = dfc.get_new_circles()
            c4_1.set_color(self.color_1)
            c4_2.set_color(self.color_2)
        self.add(self.dfc_l, self.dfc_r)
        for k in range(self.num_of_nudges):
            for dot in it.chain(self.dots_l, self.dots_r):
                dot.generate_target()
                dot.target.shift(get_random_vector(self.max_step))
            anims = AnimationGroup(*[
                MoveToTarget(dot, path_arc = PI/3., run_time = 1.5)
                for dot in it.chain(self.dots_l, self.dots_r)
            ], run_time = 2)
            self.play(anims)
            self.wait()
        self.play(self.dots_l.restore, self.dots_r.restore, run_time = 1.5)


class LineToCircleInversionRevisited(LineToCircleInversion):
    def construct(self):
        super().construct()
        self.remove_conclusions()
        self.add_explanation()

    def remove_conclusions(self):
        self.remove(self.bg_rect)
        self.remove(self.conclusions)

    def add_explanation(self):
        radius = Line(
            self.circle_O.get_left(), self.circle_O.get_center(),
            color = self.color_circle, stroke_width = 1,
        )
        radius_text = TexMobject("R", color = self.color_circle)
        radius_text.next_to(radius, UP, buff = 0.1)
        radius_group = VGroup(radius, radius_text)
        radius_group.rotate(-PI/12, about_point = self.circle_O.get_center())
        remark_length = TexMobject("|OA| = d", "\\Downarrow", "|OA'| = \dfrac{R^2}{d}")
        remark_length.arrange_submobjects(DOWN)
        remark_length.scale(1.2)
        remark_length[0].set_color(self.color_orig)
        remark_length[-1].set_color(self.color_inv)
        remark_length.to_edge(RIGHT)
        self.add(radius_group, remark_length)
        self.wait()


class CircleToCircleInversionRevisited(CircleToCircleInversionProof):
    def construct(self):
        super().add_backgrounds()
        super().show_left_and_right_points()
        super().show_random_point()
        super().show_similar_triangles()
        self.arrange_elements()
        self.add_explanation()

    def arrange_elements(self):
        self.angle_tracker.set_value(PI/3)
        self.remove(self.remark_O)
        self.remove(self.ai_OAP, self.ai_OBP, self.ai_OPiAi, self.ai_OPiBi)
        self.add(self.inv_circle)
        self.add(self.dots_P, self.labels_P)
        self.add(self.dots_AB, self.labels_AB)
        self.add(self.aux_lines, self.rtais)
        dot_I = Dot(self.inv_circle.get_center())
        label_I = DotLabel("I", dot_I, position = DOWN, label_buff = 0.15).scale(0.8)
        for mob in (dot_I, label_I):
            mob.set_sheen_direction(RIGHT)
            mob.set_color([self.color_B, self.color_A])
        remark_I = TextMobject("反形的圆心（并非$C$的反点！）")
        remark_I.scale(0.5)
        remark_I.next_to(label_I, DOWN, buff = 0.1)
        self.add(dot_I, label_I, remark_I)

    def add_explanation(self):
        for circle, color, text, angle in zip(
                [self.circle_O, self.circle_C], [self.color_O, MAROON_B],
                ["R", "r"], [-PI/12, PI/3]
            ):
            radius = Line(
                circle.get_left(), circle.get_center(),
                color = color, stroke_width = 1,
            )
            radius_text = TexMobject(text, color = color)
            radius_text.next_to(radius, UP, buff = 0.1)
            radius_group = VGroup(radius, radius_text)
            radius_group.rotate(angle, about_point = circle.get_center())
            self.add(radius_group)
        remark_length_A = TexMobject("|OA| = d-r", "\\Rightarrow", "|OA'| = \dfrac{R^2}{d-r}")
        remark_length_B = TexMobject("|OB| = d+r", "\\Rightarrow", "|OB'| = \dfrac{R^2}{d+r}")
        remark_length_A[0].set_color(self.color_A)
        remark_length_A[-1].set_color(self.color_A)
        remark_length_B[0].set_color(self.color_B)
        remark_length_B[-1].set_color(self.color_B)
        length_group = VGroup(remark_length_A, remark_length_B)
        length_group.arrange_submobjects(DOWN, buff = 0.4)
        brace = Brace(length_group, RIGHT)
        arrow = TexMobject("\\Rightarrow")
        remarks = VGroup(
            TexMobject("|A'B'| = \\dfrac{2 R^2 r}{|d^2-r^2|}"),
            TexMobject("|OI| = \\dfrac{R^2 d}{|d^2-r^2|}")
        )
        remarks.arrange_submobjects(DOWN, aligned_edge = LEFT)
        remarks.set_color(MAROON_B)
        result_group = VGroup(brace, arrow, remarks)
        result_group.arrange_submobjects(RIGHT)
        result_group.next_to(length_group, RIGHT)
        remark_group = VGroup(length_group, result_group)
        remark_group.center().to_edge(DOWN, buff = 0.2)
        bg_rect = BackgroundRectangle(remark_group, fill_opacity = 0.9)
        self.add(bg_rect, remark_group)
        self.wait()


class DescartesTheoremExamples(Scene):
    CONFIG = {
        "circle_colors" : [MAROON_B, RED, GREEN, BLUE],
        "curvs_outer" : [3, 6, 7, 34],
        "curvs_inner" : [10, 15, 19, -6],
    }
    def setup(self):
        self.text_color_map = dict(
            zip(["{k_1}", "{k_2}", "{k_3}", "{k_4}"], self.circle_colors)
        )

    def construct(self):
        self.add_title()
        self.add_outer_dfc()
        self.add_inner_dfc()

    def add_title(self):
        title = TexMobject(
            "\\left(", "{k_1}", "+", "{k_2}", "+", "{k_3}", "+", "{k_4}", "\\right) ^2",
            "= 2 \\left(", "{k_1}","^2 +","{k_2}","^2 +","{k_3}","^2 +","{k_4}","^2", "\\right)"
        )
        title.set_color_by_tex_to_color_map(self.text_color_map)
        title.scale(1.2)
        title.to_edge(UP, buff = 0.2)
        self.add(title)

    def add_outer_dfc(self):
        r1, r2, r3, r4 = [1./curv for curv in self.curvs_outer]
        p1, p2, p3 = [
            VectorizedPoint(center)
            for center in calc_centers_by_radii(r1, r2, r3, init_angle = PI*2/3)
        ]
        outer_dfc = DescartesFourCircles(p1, p2, p3, show_new_circles = False)
        c1, c2, c3 = outer_dfc.get_orig_circles()
        c4 = outer_dfc.get_new_circles()[0]
        outer_circles = VGroup(c1, c2, c3, c4)
        outer_circles.clear_updaters()
        outer_circles.set_height(5.5)
        outer_circles.to_corner(DL)
        texts = VGroup(*[
            TexMobject(f"k_{num}", "=", f"{curv}") \
            .scale(0.8) \
            .move_to(circle.get_center())
            for num, curv, circle in zip(range(1, 5), self.curvs_outer, outer_circles)
        ])
        for circle, text, color in zip(outer_circles, texts, self.circle_colors):
            circle.set_color(color)
            text.set_color(color)
        texts[-1].shift(2.5*RIGHT+1.2*UP)
        arrow = Arrow(
            texts[-1].get_bottom(), outer_circles[-1].get_right(),
            path_arc = -PI*2/3, buff = 0.1,
        ).set_color(self.circle_colors[-1])
        outer_group = VGroup(outer_circles, texts, arrow)
        self.add(outer_group)

    def add_inner_dfc(self):
        r1, r2, r3, r4 = [1./curv for curv in self.curvs_inner]
        p1, p2, p3 = [
            VectorizedPoint(center)
            for center in calc_centers_by_radii(r1, r2, r3, init_angle = -PI/7)
        ]
        inner_dfc = DescartesFourCircles(p1, p2, p3, show_new_circles = False)
        c1, c2, c3 = inner_dfc.get_orig_circles()
        c4 = inner_dfc.get_new_circles()[1]
        inner_circles = VGroup(c1, c2, c3, c4)
        inner_circles.clear_updaters()
        inner_circles.set_height(5.5)
        inner_circles.to_corner(DR)
        inner_texts = VGroup(*[
            TexMobject(f"k_{num}", "=", f"{curv}") \
            .scale(0.8) \
            .move_to(circle.get_center())
            for num, curv, circle in zip(range(1, 5), self.curvs_inner, inner_circles)
        ])
        for circle, text, color in zip(inner_circles, inner_texts, self.circle_colors):
            circle.set_color(color)
            text.set_color(color)
        inner_texts[-1].shift(2.8*LEFT+2.7*UP)
        inner_arrow = Arrow(
            inner_texts[-1].get_critical_point(DOWN),
            inner_texts[-1].get_critical_point(DOWN)+0.7*DR,
            buff = 0.1,
        ).set_color(self.circle_colors[-1])
        inner_group = VGroup(inner_circles, inner_texts, inner_arrow)
        self.add(inner_group)
        self.wait()
        self.inner_circles = inner_circles
        self.inner_texts = inner_texts
        self.inner_arrow = inner_arrow


class DFCInversionProofP1(DescartesTheoremExamples):
    CONFIG = {
        "remark_scale_text" : "示意图，图像并非真实比例",
        "orig_label_texts" : ["C_1", "C_2", "C_3", "C_4"],
        "inv_label_texts" : ["C_1'", "C_2'", "C_3'", "C_4'"],
    }
    def construct(self):
        super().add_inner_dfc()
        self.arrange_elements()
        self.add_labels()
        self.add_inversion_center()
        self.add_mapsto_symbol()
        self.add_not_to_scale_remark()
        self.wait()

    def arrange_elements(self):
        self.remove(self.inner_texts, self.inner_arrow)
        self.inner_circles.center().shift(4*UP)
        normal_form = FourCirclesNormalForm()
        normal_form.shift(4*DOWN)
        self.add(normal_form)
        self.normal_form = normal_form

    def add_labels(self):
        orig_labels = VGroup()
        for n, (circle, text) in enumerate(zip(self.inner_circles, self.orig_label_texts)):
            label = TexMobject(text).scale(1.2)
            label.set_color(circle.get_color())
            label.move_to(circle.get_center())
            orig_labels.add(label)
        inv_labels = VGroup()
        for n, (circle, text) in enumerate(zip(self.normal_form, self.inv_label_texts)):
            label = TexMobject(text).scale(1.2)
            label.set_color(circle.get_color())
            label.move_to(circle.get_center())
            inv_labels.add(label)
        c1, c2, c3, c4 = self.inner_circles
        l1, l2, l3, l4 = orig_labels
        c1i, c2i, c3i, c4i = self.normal_form
        l1i, l2i, l3i, l4i = inv_labels
        l4.next_to(c4.get_bottom(), UP, buff = 0.3)
        l3i.next_to(c3i, DOWN).to_edge(RIGHT)
        l4i.next_to(c4i, UP).to_edge(RIGHT)
        self.add(orig_labels, inv_labels)
        self.orig_labels = orig_labels
        self.inv_labels = inv_labels

    def add_inversion_center(self):
        c1, c2, c3, c4 = self.inner_circles
        inv_center = get_tangent_point(c3, c4)
        dot_O = Dot(inv_center, color = YELLOW)
        label_O = TexMobject("O", color = YELLOW).next_to(dot_O, UP)
        remark_O = TextMobject("反演中心", color = YELLOW)
        remark_O.next_to(dot_O, RIGHT, buff = 1.5)
        arrow_O = Arrow(remark_O.get_left(), dot_O.get_right(), color = YELLOW, buff = 0.2)
        orig_center_group = VGroup(dot_O, label_O, remark_O, arrow_O)
        inv_dot_O = VectorizedPoint()
        inv_dot_O.next_to(self.normal_form[-1], UP, buff = 1.4)
        inv_dot_O.shift(2*RIGHT)
        inv_center_group = orig_center_group.deepcopy()
        inv_center_group.shift(inv_dot_O.get_center() - dot_O.get_center())
        self.add(orig_center_group, inv_center_group)
        self.orig_center_group = orig_center_group
        self.inv_center_group = inv_center_group

    def add_mapsto_symbol(self):
        mapsto = TexMobject("\\mapsto")
        mapsto.rotate(-PI/2)
        mapsto.scale(2.5)
        mapsto.next_to(self.inner_circles, DOWN)
        remark_mapsto = TextMobject("反演变换")
        remark_mapsto.next_to(mapsto, LEFT)
        self.add(mapsto, remark_mapsto)

    def add_not_to_scale_remark(self):
        remark_scale = TextMobject("（" + self.remark_scale_text + "）")
        remark_scale.scale(0.75)
        remark_scale.next_to(6.5*DL, RIGHT, buff = 0)
        self.add(remark_scale)


class DFCInversionProofP2(DFCInversionProofP1):
    CONFIG = {
        "remark_scale_text" : "示意图，反演圆未标出，且图像并非真实比例",
        "inv_label_texts" : ["C_1'", "C_2'", "C_3':y=-1", "C_4':y=1"],
        "inv_center_coord_text" : "(x_0, y_0) \\, (y_0>1)",
        "circle_center_coord_texts" : ["(-1,0)", "(1,0)"],
    }
    def construct(self):
        super().construct()
        self.change_center_remarks()
        self.add_coord_system()
        self.change_inv_labels()
        self.wait()

    def change_center_remarks(self):
        for center_group in (self.orig_center_group, self.inv_center_group):
            dot, label, remark, arrow = center_group
            self.remove(remark, arrow)
            if center_group is self.inv_center_group:
                coord = TexMobject(self.inv_center_coord_text)
                coord.next_to(dot, RIGHT)
                coord.set_color(dot.get_color())
                self.add(coord)

    def add_coord_system(self):
        c1, c2, c3, c4 = self.normal_form
        center_point = (c1.get_center() + c2.get_center()) / 2
        unit_size = c1.get_height()/2
        coord_system = Axes(
            center_point = center_point,
            number_line_config = {"unit_size" : unit_size},
            y_min = -1.8, y_max = 2.8,
        )
        self.add(coord_system)
        self.coord_system = coord_system

    def change_inv_labels(self):
        l1i, l2i, l3i, l4i = self.inv_labels
        for label, x_coord, coord_text in zip([l1i, l2i], [-1, 1], self.circle_center_coord_texts):
            center = self.coord_system.c2p(x_coord, 0)
            label.next_to(center, UP)
            dot_i = Dot(center, radius = 0.1).set_color(label.get_color())
            coord_i = TexMobject(coord_text).set_color(label.get_color()).next_to(center, DOWN)
            self.add(dot_i, coord_i)


#####
## Inversion Advanced P2 Scenes
class ApollonianGasketConstruction(ApollonianGasketScene):
    CONFIG = {
        "max_iter" : 8,
        "curvatures" : [2, 2, 3],
        "init_angle" : 0,
        "curv_thres" : 30000,
        "ag_config": {
            "agc_config" : {
                "radius_thres" : 1e-3,
                "circle_color" : BLUE,
                "label_color" : WHITE,
            },
        },
        "color_curr" : YELLOW,
        "wait_time" : 2,
    }
    def construct(self):
        r1, r2, r3 = [1./curv for curv in self.curvatures]
        p1, p2, p3 = calc_centers_by_radii(r1, r2, r3, init_angle = self.init_angle)
        agc1 = AGCircle(p1, r1, parents = None, **self.ag_config["agc_config"])
        agc2 = AGCircle(p2, r2, parents = None, **self.ag_config["agc_config"])
        agc3 = AGCircle(p3, r3, parents = None, **self.ag_config["agc_config"])
        remark = TextMobject("（圆内数字为该圆的曲率）")
        remark.scale(0.75).to_corner(DL)
        self.add(remark)
        for k in range(self.max_iter):
            agcs_copy = [agc.deepcopy() for agc in (agc1, agc2, agc3)]
            ag = ApollonianGasket(
                *agcs_copy, num_iter = k,
                curv_thres = self.curv_thres, **self.ag_config
            )
            iter_num = VGroup(
                TextMobject("迭代次数："), TexMobject(f"{k}")
            ).arrange_submobjects(RIGHT).scale(1.5)
            iter_num.to_edge(LEFT, buff = 1)
            ag.scale(3.8)
            ag.shift(np.array([0, 3.8, 0]) - ag.get_top() + 3*RIGHT)
            VGroup(*ag.agc_list[-1]).set_color(self.color_curr)
            self.add(ag, iter_num)
            self.wait(self.wait_time)
            if k != self.max_iter-1:
                self.remove(ag, iter_num)
            

class ApollonianGasketExample1(Scene):
    CONFIG = {
        "max_iter" : 20,
        "curvatures" : [3, 6, 7],
        "curvature_texts" : [-2, 3, 6, 7],
        "init_angle" : 0,
        "curv_thres" : 4000,
        "ag_config": {
            "agc_config" : {
                "radius_thres" : 1e-3,
                "circle_color" : BLUE,
                "label_color" : WHITE,
            },
        },
        "ag_scaling_factor" : 5.2,
    }
    def construct(self):
        r1, r2, r3 = [1./curv for curv in self.curvatures]
        p1, p2, p3 = calc_centers_by_radii(r1, r2, r3, init_angle = self.init_angle)
        agc1 = AGCircle(p1, r1, parents = None, **self.ag_config["agc_config"])
        agc2 = AGCircle(p2, r2, parents = None, **self.ag_config["agc_config"])
        agc3 = AGCircle(p3, r3, parents = None, **self.ag_config["agc_config"])
        ag_seed = ApollonianGasket(
            *[agc.deepcopy() for agc in (agc1, agc2, agc3)],
            num_iter = 0, curv_thres = self.curv_thres, **self.ag_config
        )
        ag_result = ApollonianGasket(
            *[agc.deepcopy() for agc in (agc1, agc2, agc3)],
            num_iter = self.max_iter, curv_thres = self.curv_thres, **self.ag_config
        )
        ag_seed_center = ag_seed[0][0].get_right()
        ag_result_center = ag_result[0][0].get_right()
        arrow = Arrow(LEFT, RIGHT)
        figure_group = VGroup(ag_seed, ag_result, arrow)
        for ag, center, direction in zip(
            [ag_seed, ag_result], [ag_seed_center, ag_result_center], [4*LEFT, 4*RIGHT]):
            ag.scale(self.ag_scaling_factor)
            ag.shift(direction - center)
        figure_group.shift(DOWN)
        k1, k2, k3, k4 = list(map(str, self.curvature_texts))
        title = TexMobject(
            f"({k1}+{k2}+{k3}+{k4})^2 = 2\\left[({k1})^2+{k2}^2+{k3}^2+{k4}^2 \\right]"
        )
        title.set_width(13)
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.add(figure_group, title)
        self.wait()


class ApollonianGasketExample2(ApollonianGasketExample1):
    CONFIG = {
        "max_iter" : 20,
        "curvatures" : [5, 8, 12],
        "curvature_texts" : [-3, 5, 8, 12],
        "curv_thres" : 5000,
        "ag_config": {
            "agc_config" : {
                "radius_thres" : 5e-4,
                "circle_color" : BLUE,
                "label_color" : WHITE,
            },
        },
        "ag_scaling_factor" : 8,
    }


class LoxodromicSpiralInTangentCircles(Scene):
    CONFIG = {
        "max_iter" : 20,
        "agc_config" : {
            "radius_thres" : 1,
            "circle_color" : BLUE,
            "label_color" : WHITE,
        },
        "curve_config" : {
            "color" : YELLOW,
            "stroke_width" : 2,
        },
        "alpha" : 0.6,
        "dashed_line_config" : {
            "color" : GREY,
            "stroke_width" : 0.5,
            "num_dashes" : 200,
            "positive_space_ratio" : 0.6,
        }
    }
    def construct(self):
        self.generate_circles()
        self.generate_curves()
        self.generate_labels()
        self.generate_lines()
        self.add_elements()
        self.zooming_in()

    def generate_circles(self):
        agcm2 = AGCircle(2/3.*UP, 1/3., **self.agc_config)
        agcm1 = AGCircle(RIGHT/2, 1/2., **self.agc_config)
        agczr = AGCircle(ORIGIN, -1, **self.agc_config)
        agcp1 = AGCircle(LEFT/2, 1/2., **self.agc_config)
        agcp2 = AGCircle(2/3.*DOWN, 1/3., **self.agc_config)
        agc_list = [agcm2, agcm1, agczr, agcp1, agcp2]
        for n in range(self.max_iter):
            A, B, C, known_agc = agc_list[:4]
            agc_m_k, agc_m_c = calc_new_agc_info(A, B, C, known_agc = known_agc)
            agc_m = AGCircle(agc_m_c, 1./agc_m_k, parents = (A, B, C), **self.agc_config)
            known_agc, C, B, A = agc_list[-4:]
            agc_p_k, agc_p_c = calc_new_agc_info(C, B, A, known_agc = known_agc)
            agc_p = AGCircle(agc_p_c, 1./agc_p_k, parents = (C, B, A), **self.agc_config)
            agc_list.insert(0, agc_m)
            agc_list.append(agc_p)
        agc_group = VGroup(*agc_list)
        agc_group.set_height(7.8)
        self.agc_list = agc_list
        self.agc_group = agc_group

    def generate_curves(self):
        agc_ps = self.agc_list[-self.max_iter-4:]
        agc_ps_points = []
        loxo_curve_p_solid = VMobject(**self.curve_config)
        for k in range(len(agc_ps)-2):
            if k != 0:
                c1, c2, c3 = agc_ps[k], agc_ps[k+1], agc_ps[k+2]
                pt1 = get_tangent_point(c1, c2)
                pt2 = get_tangent_point(c2, c3)
                p = c2.get_center()
                if k != 1:
                    agc_ps_points.extend(
                        [pt1, p*(1-self.alpha)+pt1*self.alpha, p*(1-self.alpha)+pt2*self.alpha, pt2]
                    )
                else:
                    agc_ps_points.extend(
                        [pt1, p*0.7+pt1*0.3, p*0.6+pt2*0.4, pt2]
                    )
            else:
                c1, c2 = agc_ps[1], agc_ps[2]
                pt = get_tangent_point(c1, c2)
                agc_ps_points.extend([8*LEFT, 7*LEFT, 6*LEFT, pt])
        loxo_curve_p_solid.append_points(agc_ps_points)
        loxo_curve_m_solid = loxo_curve_p_solid.deepcopy()
        loxo_curve_m_solid.rotate(PI, about_point = self.agc_group.get_center())
        self.loxo_curve_p_solid = loxo_curve_p_solid
        self.loxo_curve_m_solid = loxo_curve_m_solid
    
    def generate_labels(self):
        labels = VGroup(*[
            TexMobject("C_{%d}" % num, background_stroke_width = 0)
            for num in range(-self.max_iter-2, self.max_iter+3)
        ])
        for label, circle in zip(labels, self.agc_group):
            label.set_height(circle.get_height()*0.15)
            label.move_to(circle.get_center())
        label_c0 = labels[self.max_iter+2]
        label_c0.set_height(0.8)
        label_c0.next_to(self.agc_group.get_critical_point(UL), DR, buff = 0.1)
        self.labels = labels

    def generate_lines(self):
        agc_ps = self.agc_list[-self.max_iter-2:]
        line_p_solid = VMobject(**self.dashed_line_config)
        line_p_solid_corners = [8*LEFT]
        for circle in agc_ps:
            line_p_solid_corners.append(circle.get_center())
        line_p_solid.set_points_as_corners(line_p_solid_corners)
        line_m_solid = line_p_solid.deepcopy()
        line_m_solid.rotate(PI, about_point = self.agc_group.get_center())
        self.line_p_solid = line_p_solid
        self.line_m_solid = line_m_solid

    def add_elements(self):
        figure = VGroup(
            self.agc_group, self.loxo_curve_p_solid, self.loxo_curve_m_solid,
            self.line_p_solid, self.line_m_solid, self.labels,
        )
        self.add(figure)
        self.figure = figure

    def zooming_in(self):
        self.figure.save_state()
        self.wait(0.5)
        self.play(
            ApplyMethod(self.figure.shift, -self.agc_group[-1].get_center()),
            run_time = 2,
        )
        self.wait()
        for k in range(10):
            self.play(
                ApplyMethod(self.figure.scale, 2.5, {"about_point" : self.agc_group[-1].get_center()}),
                run_time = 2,
            )
        self.wait()
        self.play(self.figure.restore, run_time = 15)
        self.wait(2)


class ShowFordCircles(ZoomInOnFordCircles):
    CONFIG = {
        "q_max" : 30,
    }
    def construct(self):
        self.setup_axes()
        self.setup_circles_and_labels()
        self.add_remarks()
        self.first_zoom_in()
        self.wait()

    def first_zoom_in(self):
        self.zoom_in_on(1/2., 6)

    def add_remarks(self):
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
        self.add(nl_remark, frac_remark)


class ShowFordCirclesDetails(ShowFordCircles):
    CONFIG = {
        "q_max" : 100,
    }
    def construct(self):
        super().construct()
        self.further_zoom_in()

    def setup_circles_and_labels(self):
        circles = VGroup()
        labels = VGroup()
        for q in range(1, self.q_max+1):
            for p in get_coprime_numers_by_denom(q):
                if (q <= 40) or (0.6 <= p/q <= 0.8):
                    circle = self.generate_circle_by_fraction(p, q)
                    circle.add_updater(
                        lambda m: m.set_stroke(width = get_stroke_width_by_height(m.get_height()))
                    )
                    label = AssembledFraction(p, q)
                    label.set_height(circle.get_height() * self.label_height_factor)
                    label.move_to(circle.get_center())
                    circles.add(circle)
                    labels.add(label)
        self.add(circles, labels)
        self.circles = circles
        self.labels = labels

    def further_zoom_in(self):
        self.acl = VGroup(self.axes, self.circles, self.labels)
        self.acl.save_state()
        self.wait(0.5)
        self.play_zooming_animation(1/np.sqrt(2), 9, run_time = 5)
        self.wait()
        self.play_zooming_animation(0.73, 5, run_time = 4)
        self.wait()
        self.play_zooming_animation(0.74, 5, run_time = 4)
        self.wait()
        self.play(self.acl.restore, run_time = 5)
        self.wait(2)


class ProveFordCirclesPropertiesP1(Scene):
    CONFIG = {
        "c1_frac" : [2, 3],
        "c2_frac" : [3, 4],
        "c3_frac" : [5, 7],
        "circle_config" : {"stroke_color" : BLUE, "stroke_width" : 2,},
        "line_config" : {"stroke_color" : GREY, "stroke_width" : 2,},
        "aux_line_config" : {"stroke_color" : GREY, "stroke_width" : 0.8,},
        "polygon_config" : {"fill_color" : GREY, "fill_opacity" : 0.4, "stroke_width" : 0,},
    }
    def setup(self):
        a, b = self.c1_frac
        c, d = self.c2_frac
        p, q = self.c3_frac
        r1 = 1/(2*b**2)
        r2 = 1/(2*d**2)
        r3 = 1/(2*q**2)
        c1_center = a/b*RIGHT + r1*UP
        c2_center = c/d*RIGHT + r2*UP
        c3_center = p/q*RIGHT + r3*UP
        c1 = Circle(arc_center = c1_center, radius = r1, **self.circle_config)
        c2 = Circle(arc_center = c2_center, radius = r2, **self.circle_config)
        c3 = Circle(arc_center = c3_center, radius = r3, **self.circle_config)
        c1_dot = SmallDot(color = GREY)
        c1_dot.add_updater(lambda m: m.move_to(c1.get_center()))
        c2_dot = SmallDot(color = GREY)
        c2_dot.add_updater(lambda m: m.move_to(c2.get_center()))
        c3_dot = SmallDot(color = GREY)
        c3_dot.add_updater(lambda m: m.move_to(c3.get_center()))
        line = Line(
            2*c1.get_bottom()-c2.get_bottom(),
            2*c2.get_bottom()-c1.get_bottom(),
            **self.line_config
        )
        VGroup(c1, c2, c3, line).set_height(6).center().to_edge(UP)
        aux_line_1 = Line(c1.get_center(), c1.get_bottom(), **self.aux_line_config)
        aux_line_2 = Line(c2.get_center(), c2.get_bottom(), **self.aux_line_config)
        aux_line_3 = Line(c1.get_center(), c2.get_center(), **self.aux_line_config)
        aux_line_4 = Line(c1.get_bottom(), c2.get_bottom(), **self.aux_line_config) \
                     .shift(c2.get_height()/2*UP)
        polygon = Polygon(
            c1.get_center(), c2.get_center(), aux_line_4.get_start_and_end()[0],
            **self.polygon_config,
        )
        l1 = TexMobject("\\dfrac{a}{b}").next_to(c1, DOWN)
        l2 = TexMobject("\\dfrac{c}{d}").next_to(c2, DOWN)
        l3 = TexMobject("\\dfrac{a+c}{b+d}").next_to(c3, DOWN)
        self.orig_group = VGroup(c1, c2, line, c1_dot, c2_dot, l1, l2)
        self.aux_group = VGroup(aux_line_1, aux_line_2, aux_line_3, aux_line_4, polygon)
        self.new_group = VGroup(c3, c3_dot, l3)
    
    def construct(self):
        self.add(self.orig_group, self.aux_group)
        self.wait()


class ProveFordCirclesPropertiesP2(ProveFordCirclesPropertiesP1):
    def construct(self):
        self.add(self.orig_group, self.new_group)
        self.wait()


class ShowFordCirclesFareySum(ZoomInOnFordCircles):
    pass
    # A rename, that's it.


class DFCInversionProofP3(DFCInversionProofP2):
    CONFIG = {
        "remark_scale_text" : "示意图，反演圆未标出，且图像并非真实比例",
        "inv_label_texts" : ["C_1'", "C_2'", "C_3':\\mathrm{Im}(z)=-1", "C_4':\\mathrm{Im}(z)=1"],
        "inv_center_coord_text" : "z_0 = x_0+iy_0\\, (y_0>1)",
        "circle_center_coord_texts" : ["-1", "1"],
    }
    def construct(self):
        super().construct()
        self.wait()

    def add_coord_system(self):
        c1, c2, c3, c4 = self.normal_form
        center_point = (c1.get_center() + c2.get_center()) / 2
        unit_size = c1.get_height()/2
        coord_system = NumberPlane(
            center_point = center_point,
            number_line_config = {"unit_size" : unit_size},
            y_min = -3, y_max = 3,
            background_line_style = {
                "stroke_color" : GREY,
                "stroke_width" : 1.5,
                "stroke_opacity" : 0.8,
            },
        )
        aux_coord_system = Axes(
            center_point = center_point,
            number_line_config = {"unit_size" : unit_size},
            y_min = -3, y_max = 3,
            stroke_opacity = 0.8,
        )
        self.add(coord_system, aux_coord_system)
        self.coord_system = coord_system


class NormalFormIn3D(ThreeDScene):
    CONFIG = {
        "axis_unit_size" : 1.5,
        "axis_min" : -1.5,
        "axis_max" : 2.8,
        "resolution" : (60, 120),
        "plane_colors" : [GREEN, BLUE],
        "sphere_colors" : [MAROON_B, RED, PINK],
    }
    def construct(self):
        self.add_3d_stuff()
        self.add_2d_stuff()

    def add_3d_stuff(self):
        self.set_camera_orientation(theta = 70 * DEGREES, phi = 50 * DEGREES)
        axes = ThreeDAxes(
            x_min = self.axis_min, x_max = self.axis_max,
            y_min = self.axis_min, y_max = self.axis_max,
            z_min = self.axis_min, z_max = self.axis_max,
            number_line_config = {"unit_size" : self.axis_unit_size},
        )
        sphere_centers = [
            axis.number_to_point(1)
            for axis in [axes.x_axis, axes.y_axis, axes.z_axis]
        ]
        radius = 1/np.sqrt(2) * self.axis_unit_size
        sphere_dots = VGroup(*[
            Sphere(
                radius = 0.08, resolution = self.resolution,
                fill_opacity = 1, stroke_width = 0,
            ).move_to(sphere_center).set_color(color)
            for sphere_center, color in zip(sphere_centers, self.sphere_colors)
        ])
        spheres = VGroup(*[
            Sphere(
                radius = radius, resolution = self.resolution,
                fill_opacity = 0.6, stroke_width = 0.5,
            ).move_to(sphere_center).set_color(color)
            for sphere_center, color in zip(sphere_centers, self.sphere_colors)
        ])
        planes = VGroup(*[
            VGroup(*[
                Square(
                    side_length = 1, fill_opacity = fill_opacity,
                    stroke_color = GREY, stroke_width = 0.3, stroke_opacity = 0.2,
                )
                for k in range(n**2)
            ]).arrange_in_grid(n, n, buff = 0) \
            .apply_matrix(z_to_vector([1, 1, 1])) \
            .move_to(np.average(sphere_centers)) \
            .shift(radius * normalize(direction)) \
            .set_color(color)
            for n, fill_opacity, direction, color in zip(
                [7, 8], [0.2, 0.3], [np.ones(3), -np.ones(3)], self.plane_colors,
            )
        ])
        figure_group = VGroup(axes, planes, sphere_dots, spheres)
        figure_group.shift(RIGHT*2+0.5*OUT)
        self.add(figure_group)
        self.add(axes)
        self.add(planes)
        self.add(sphere_dots, spheres)

    def add_2d_stuff(self):
        sphere_remarks = VGroup(*[
            TextMobject(
                "球：圆心为" + f"$({int(x)},{int(y)},{int(z)})$" + \
                "，半径为" + "$\\dfrac{1}{\\sqrt{2}}$"
            ).set_color(color)
            for (x, y, z), color in zip([RIGHT, UP, OUT], self.sphere_colors)
        ]).arrange_submobjects(DOWN)
        plane_remarks = VGroup(*[
            TexMobject(
                "\\text{平面：}" + "x+y+z=1" + sign + "\\dfrac{\\sqrt{3}}{\\sqrt{2}"
            ).set_color(color)
            for sign, color in zip(["+", "-"], self.plane_colors)
        ]).arrange_submobjects(DOWN)
        remarks = VGroup(sphere_remarks, plane_remarks)
        remarks.arrange_submobjects(DOWN, aligned_edge = LEFT)
        remarks.scale(0.8)
        remarks.to_corner(DR)
        self.add_fixed_in_frame_mobjects(remarks)
        self.wait()


#####
## Banner
class Banner_Intro(Scene):
    CONFIG = {
        "circle_color" : YELLOW,
        "text_color" : BLUE,
        "inv_text_color" : BLUE,
        "circle_center" : 0.8*UP,
        "circle_radius" : 3,
        "grid_side_length" : 0.5,
        "x_range" : 300,
        "y_range" : 300,
        "dist_thres" : 300,
    }
    def construct(self):
        circle = Circle(color = self.circle_color, radius = self.circle_radius, stroke_width = 5)
        circle.move_to(self.circle_center)
        dot = SmallDot(self.circle_center, color = self.circle_color)
        text = TextMobject("Inversion", color = self.text_color, background_stroke_width = 3)
        text.rotate(PI/2.)
        text.move_to(0.4*RIGHT)
        text.apply_complex_function(np.exp)
        text.rotate(-PI/2.)
        text.scale(1.5)
        text.move_to(0.9*DOWN)
        inv_text = InversedVMobject(text, circle, use_dashed_vmob = False)
        inv_text.suspend_updating()
        inv_text.set_background_stroke(color = "#303030", width = 3)
        inv_text.set_stroke(width = 0)
        inv_text.set_fill(color = self.inv_text_color, opacity = 0.5)
        grid = VGroup(*[
            Square(
                side_length = self.grid_side_length,
                stroke_width = 0, fill_opacity = 0.3,
                fill_color = CB_DARK if (i+j)%2==0 else CB_LIGHT
            ).move_to(self.circle_center + (i*RIGHT+j*UP)*self.grid_side_length)
            for i in range(-self.x_range, self.x_range+1, 1)
            for j in range(-self.y_range, self.y_range+1, 1)
            if np.sqrt(i**2+j**2) * self.grid_side_length < self.dist_thres
        ])
        for square in grid:
            if is_close_in_R3(square.get_center(), self.circle_center):
                grid.remove(square)
        inv_grid = InversedVMobject(grid, circle, use_dashed_vmob = False)
        self.add(inv_grid, circle, dot, text, inv_text)
        self.wait()


class Banner_AdvancedP1(ApollonianGasketScene):
    CONFIG = {
        "curvatures" : [570, 968, 1112],
        "init_angle" : PI/7,
        "num_iter" : 20,
        "curv_thres" : 1e6,
        "ag_config" : {
            "agc_config" : {
                "radius_thres" : 5e-6,
                "circle_color" : YELLOW,
                "label_color" : WHITE,
            },
        },
        "part_text" : "上篇",
    }
    def construct(self):
        super().construct()
        ag = self.ag
        ag.set_height(7)
        circle_myst = ag.agc_list[0][0]
        label_myst = circle_myst.label
        label_question = TexMobject("???")
        label_question.match_height(label_myst)
        label_question.move_to(label_myst)
        self.remove(label_myst)
        self.add(label_question)
        part = TextMobject(self.part_text)
        part.to_corner(DR)
        self.add(part)


class Banner_AdvancedP2(Banner_AdvancedP1):
    CONFIG = {
        "part_text" : "下篇",
    }



