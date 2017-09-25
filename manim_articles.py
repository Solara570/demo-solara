#coding=utf-8

from helpers import *
import numpy as np

from animation.simple_animations import *
from animation.transform import *
from mobject.svg_mobject import *
from mobject.tex_mobject import *
from scene import Scene
from camera import Camera


# Part 1: https://www.bilibili.com/read/cv17444
class ConfigManimBanner(Scene):
    CONFIG = {
        "verb"         : "配置",
        "svg_filename" : "hammer_and_wrench",
        "author_colors": [ORANGE, GREEN],
        "bg_angle"     : 0,
    }
    def construct(self):
        question = TextMobject("如何", self.verb, "m", "anim", "?", \
            arg_separator = "")
        question.scale(3)

        full_name = TextMobject("(", "m", "ath", " anim", "ator)", \
            arg_separator = "")
        manim = VGroup(*question[2:4])
        full_name.scale_to_fit_width(manim.get_width())
        full_name.next_to(manim, DOWN)

        for word in (question, full_name):
            word.highlight_by_tex("m", BLUE)
            word.highlight_by_tex("anim", YELLOW)

        words = VGroup(question, full_name)
        rect = BackgroundRectangle(words).scale_to_fit_width(SPACE_WIDTH * 2)

        author = TextMobject("@Solara570")
        author.scale(1.5)
        author.gradient_highlight(*self.author_colors)
        author.to_corner(LEFT+DOWN)

        bg_object = SVGMobject(file_name = self.svg_filename)
        bg_object.rotate(self.bg_angle)
        bg_object.fade(0.7)
        bg_object.scale_to_fit_height(SPACE_HEIGHT * 2)
        bg_object.move_to(manim.get_center())

        self.add(*[bg_object, rect, author, question, full_name])


# Part 2: https://www.bilibili.com/read/cv19963
class RunManimBanner(ConfigManimBanner):
    CONFIG = {
        "verb"         : "使用",
        "svg_filename" : "gear",
        "author_colors": [PURPLE, WHITE],
        "bg_angle"     : np.pi/5,
    }


class Manim(Scene):
    def construct(self):
        text = TextMobject("manim")
        self.play(Write(text))
        self.play(ApplyPointwiseFunction(
            lambda (x, y, z) : complex_to_R3(np.exp(complex(x, y))),
            text
        ))
        self.dither()



