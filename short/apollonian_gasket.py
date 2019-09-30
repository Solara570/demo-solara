#coding=utf-8

import math
from manimlib.constants import *
from manimlib.utils.space_ops import rotate_vector, get_norm, R3_to_complex, complex_to_R3
from manimlib.mobject.types.vectorized_mobject import VMobject, VGroup
from manimlib.mobject.svg.tex_mobject import TexMobject
from manimlib.mobject.geometry import Circle
from manimlib.scene.scene import Scene

def calc_radii_by_centers(p1, p2, p3):
    r12 = get_norm(p1 - p2)
    r23 = get_norm(p2 - p3)
    r13 = get_norm(p1 - p3)
    sum_123 = (r12 + r23 + r13) / 2
    r1 = sum_123 - r23
    r2 = sum_123 - r13
    r3 = sum_123 - r12
    return r1, r2, r3

def calc_centers_by_radii(r1, r2, r3, init_angle = 0):
    if init_angle is None:
        init_angle = TAU * np.random.random_sample()
    r12 = r1 + r2
    r23 = r2 + r3
    r13 = r1 + r3
    cos_theta = (r12**2 + r13**2 - r23**2) / (2*r12*r13)
    theta = math.acos(cos_theta)
    p1 = ORIGIN
    p2 = p1 + rotate_vector(RIGHT, init_angle) * r12
    p3 = p1 + rotate_vector(RIGHT * r13, init_angle + theta)
    return p1, p2, p3

def calc_new_agc_info(agc1, agc2, agc3, known_agc = None):
    k1, k2, k3 = [agc.get_curvature() for agc in (agc1, agc2, agc3)]
    z1, z2, z3 = [agc.get_complex_center() for agc in (agc1, agc2, agc3)]
    sum_k = k1 + k2 + k3
    sum_kz = k1*z1 + k2*z2 + k3*z3
    if known_agc is not None:
        kc = known_agc.get_curvature()
        zc = known_agc.get_complex_center()
        k4 = 2*sum_k - kc
        z4 = (2*sum_kz - kc*zc) / k4
        return k4, complex_to_R3(z4)
    else:
        # Calculate the curvatures of new circles
        sum_k2 = k1**2 + k2**2 + k3**2
        sum_k_cycle_prod = k1*k2 + k2*k3 + k3*k1
        b = (-2)*sum_k
        c = sum_k2 - 2*sum_k_cycle_prod
        delta = b**2 - 4*c
        k4_1 = (-b + np.sqrt(delta)) / 2
        k4_2 = (-b - np.sqrt(delta)) / 2
        # Calculate the centers of new circles
        # arxiv.org/abs/math/0101066v1 - Eqn 2.3
        sum_k2z = k1**2 * z1 + k2**2 * z2 + k3**2 * z3
        coeff_1 = (sum_k - k4_1) * k4_1
        const_1 = 2 * sum_k2z - (sum_k + k4_1) * sum_kz
        z4_1 = const_1 / coeff_1
        coeff_2 = (sum_k - k4_2) * k4_2
        const_2 = 2 * sum_k2z - (sum_k + k4_2) * sum_kz
        z4_2 = const_2 / coeff_2
        return [(k4_1, complex_to_R3(z4_1)), (k4_2, complex_to_R3(z4_2))]

class AGCircle(VMobject):
    CONFIG = {
        "radius_thres" : 1e-3,
        "circle_color" : BLUE,
        "label_color" : WHITE,
    }
    def __init__(self, center, radius, parents = None, **kwargs):
        super(AGCircle, self).__init__(**kwargs)
        self.center = center
        self.radius = radius
        self.parents = parents
        self.add_circle()
        if self.radius > self.radius_thres:
            self.add_label()

    def add_circle(self):
        circle = Circle(
            radius = np.abs(self.radius),
            stroke_color = self.circle_color, stroke_width = 1,
        )
        circle.move_to(self.center)
        self.add(circle)
        self.circle = circle

    def add_label(self):
        circle = self.circle
        label = TexMobject(
            "%d" % int(np.round(1./self.radius)),
            background_stroke_width = 0,
        )
        h_factor = circle.get_width() * 0.6 / label.get_width()
        v_factor = circle.get_height() * 0.5 / label.get_height()
        factor = np.min([h_factor, v_factor])
        label.scale(factor)
        label.move_to(self.center)
        self.add(label)
        self.label = label

    def get_curvature(self):
        return 1./self.radius

    def get_parents(self):
        return self.parents

    def get_R3_center(self):
        return self.center

    def get_complex_center(self):
        return R3_to_complex(self.center)


class ApollonianGasket(VMobject):
    CONFIG = {
        "num_iter" : 1,
        "curv_thres" : 10000,
        "agc_config" : {
            "circle_color" : RED,
            "label_color" : WHITE,
        }
    }
    def __init__(self, agc1, agc2, agc3, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.agc_list = [(agc1, agc2, agc3)]
        for n in range(self.num_iter+1):
            self.add(VGroup(*self.agc_list[-1]))
            if n != self.num_iter:
                self.update_agc_list(n)

    def update_agc_list(self, n):
        if n == 0:
            # First iteration
            agc1, agc2, agc3 = self.agc_list[-1]
            [(k4_1, p4_1), (k4_2, p4_2)] = calc_new_agc_info(agc1, agc2, agc3)
            new_agcs = (
                AGCircle(p4_1, 1./k4_1, parents = self.agc_list[-1], **self.agc_config), 
                AGCircle(p4_2, 1./k4_2, parents = self.agc_list[-1], **self.agc_config)
            )
            self.agc_list.append(new_agcs)
        else:
            new_agcs = []
            for curr_agc in self.agc_list[-1]:
                agc1, agc2, agc3 = curr_agc.get_parents()
                for A, B, C in [
                    (agc1, agc2, agc3), (agc2, agc3, agc1), (agc3, agc1, agc2)
                ]:
                    k4, p4 = calc_new_agc_info(curr_agc, A, B, known_agc = C)
                    if k4 < self.curv_thres:
                        new_agcs.append(
                            AGCircle(p4, 1./k4, parents = (curr_agc, A, B), **self.agc_config)
                        )
            self.agc_list.append(tuple(new_agcs))


class ApollonianGasketScene(Scene):
    CONFIG = {
        "random_seed" : None,
        "curvatures" : [2, 2, 3],
        "init_angle" : None,
        "num_iter" : 15,
        "curv_thres" : 2000,
        "ag_config" : {
            "agc_config" : {
                "circle_color" : RED,
                "label_color" : WHITE,
            }
        }
    }
    def construct(self):
        r1, r2, r3 = [1./curv for curv in self.curvatures]
        p1, p2, p3 = calc_centers_by_radii(r1, r2, r3, init_angle = self.init_angle)
        agc1 = AGCircle(p1, r1, parents = None, **self.ag_config["agc_config"])
        agc2 = AGCircle(p2, r2, parents = None, **self.ag_config["agc_config"])
        agc3 = AGCircle(p3, r3, parents = None, **self.ag_config["agc_config"])
        ag = ApollonianGasket(
            agc1, agc2, agc3, num_iter = self.num_iter,
            curv_thres = self.curv_thres, **self.ag_config
        )
        ag.center()
        ag.set_height(7.8)
        self.add(ag)
        self.wait()


class ApollonianGasket_2_2_3(ApollonianGasketScene):
    CONFIG = {
        "curvatures" : [2, 2, 3],
        "init_angle" : 0,
        "ag_config" : {
            "agc_config" : {
                "circle_color" : BLUE,
                "label_color" : WHITE,
            }
        }
    }

class ApollonianGasket_3_6_7(ApollonianGasketScene):
    CONFIG = {
        "curvatures" : [3, 6, 7],
        "init_angle" : None,
    }

class ApollonianGasket_8_9_9(ApollonianGasketScene):
    CONFIG = {
        "curvatures" : [9, 9, 8],
        "init_angle" : PI,
        "ag_config" : {
            "agc_config" : {
                "circle_color" : GREEN,
                "label_color" : WHITE,
            }
        }
    }

class ApollonianGasket_5_8_8(ApollonianGasketScene):
    CONFIG = {
        "curvatures" : [8, 8, 5],
        "init_angle" : -PI/2.,
        "ag_config" : {
            "agc_config" : {
                "circle_color" : YELLOW,
                "label_color" : WHITE,
            }
        }
    }

class ApollonianGasket_10_15_19(ApollonianGasketScene):
    CONFIG = {
        "curvatures" : [10, 15, 19],
        "init_angle" : None,
        "ag_config" : {
            "agc_config" : {
                "circle_color" : PINK,
                "label_color" : WHITE,
            }
        }
    }

class ApollonianGasket_11_14_15(ApollonianGasketScene):
    CONFIG = {
        "curvatures" : [11, 14, 15],
        "init_angle" : None,
        "ag_config" : {
            "agc_config" : {
                "circle_color" : YELLOW,
                "label_color" : WHITE,
            }
        }
    }

class ApollonianGasket_12_17_20(ApollonianGasketScene):
    CONFIG = {
        "curvatures" : [12, 17, 20],
        "init_angle" : None,
        "ag_config" : {
            "agc_config" : {
                "circle_color" : BLUE,
                "label_color" : WHITE,
            }
        }
    }

class ApollonianGasket_14_26_27(ApollonianGasketScene):
    CONFIG = {
        "curvatures" : [14, 26, 27],
        "init_angle" : None,
        "ag_config" : {
            "agc_config" : {
                "circle_color" : RED,
                "label_color" : WHITE,
            }
        }
    }

class ApollonianGasket_13_21_24(ApollonianGasketScene):
    CONFIG = {
        "curvatures" : [13, 21, 24],
        "init_angle" : None,
        "ag_config" : {
            "agc_config" : {
                "circle_color" : TEAL,
                "label_color" : WHITE,
            }
        }
    }

