#coding=utf-8

import numpy as np
from helpers import *
from mobject import *
from mobject.tex_mobject import *
from topics.geometry import *
from scene import Scene
from camera import Camera

class LightQuantumThumbnail(Scene):
    def construct(self):
        # Add text
        light = TextMobject("光", stroke_width = 3, stroke_color = YELLOW)
        quantum = TextMobject("量子")
        texts = VGroup(*[light, quantum])
        texts.scale(3.5)
        texts.arrange_submobjects(RIGHT, buff = 1)
        texts.to_edge(UP, buff = 2)
        self.add(texts)

        # Add those shiny lines
        radius = 2
        angles = [k*np.pi/6. for k in range(12)]
        for rm_angle in [0, np.pi]:
            angles.remove(rm_angle)
        for angle in angles:
            inner_point = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            outer_point = 1.2 * (1 + pow(np.cos(angle), 2) / 10.) * inner_point
            line = Line(
                inner_point, outer_point,
                stroke_color = YELLOW,
                stroke_width = 8,
            )
            line.next_to(
                light.get_center(),
                direction = rotate_vector(RIGHT, angle),
                buff = 1.05 + pow(np.cos(angle), 2) / 6.,
            )
            self.add(line)
