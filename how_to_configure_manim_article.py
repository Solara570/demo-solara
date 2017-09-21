#coding=utf-8

from helpers import *
import numpy as np

from animation.transform import *
from animation.simple_animations import *
from mobject.svg_mobject import *
from mobject.tex_mobject import *
from scene import Scene
from camera import Camera


class ConfigManimThumbnail(Scene):
    def construct(self):
        config = TextMobject("如何配置", "m", "anim", "?", \
            arg_separator = "")
        config[1].highlight(BLUE)
        config[2].highlight(YELLOW)
        config.scale(3)

        full_name = TextMobject("(", "m", "ath", " anim", "ator)", \
            arg_separator = "")
        full_name[1].highlight(BLUE)
        full_name[3].highlight(YELLOW)
        manim = VGroup(*config[1:3])
        full_name.scale_to_fit_width(manim.get_width())
        full_name.next_to(manim, DOWN)

        gear = SVGMobject(file_name = "gear")
        gear.rotate(np.pi/5.)
        gear.fade(0.9)
        gear.scale(4)
        gear.move_to(manim.get_center())

        self.add(*[gear, config, full_name])