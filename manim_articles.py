#coding=utf-8

from helpers import *
import numpy as np

from mobject.svg_mobject import *
from mobject.tex_mobject import *
from scene import Scene
from camera import Camera


class ConfigManimThumbnail(Scene):
    CONFIG = {
        "verb"         : "配置",
        "svg_filename" : "hammer_and_wrench",
        "bg_angle"     : 0,
    }
    def construct(self):
        config = TextMobject("如何", self.verb, "m", "anim", "?", \
            arg_separator = "")
        config[2].highlight(BLUE)
        config[3].highlight(YELLOW)
        config.scale(3)

        full_name = TextMobject("(", "m", "ath", " anim", "ator)", \
            arg_separator = "")
        full_name[1].highlight(BLUE)
        full_name[3].highlight(YELLOW)
        manim = VGroup(*config[2:4])
        full_name.scale_to_fit_width(manim.get_width())
        full_name.next_to(manim, DOWN)

        words = VGroup(config, full_name)
        rect = BackgroundRectangle(words).scale_to_fit_width(SPACE_WIDTH * 2)

        bg_object = SVGMobject(file_name = self.svg_filename)
        bg_object.rotate(self.bg_angle)
        bg_object.fade(0.7)
        bg_object.scale_to_fit_height(SPACE_HEIGHT * 2)
        bg_object.move_to(manim.get_center())

        self.add(*[bg_object, rect, config, full_name])


class RunManimThumbnail(ConfigManimThumbnail):
    CONFIG = {
        "verb"         : "使用",
        "svg_filename" : "gear",
        "bg_angle"     : np.pi/5,
    }