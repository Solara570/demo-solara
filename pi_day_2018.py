#!/usr/bin/env python
#coding=utf-8

from mobject.tex_mobject import *
from topics.geometry import *
from custom.custom_mobjects import *
from scene import Scene
from camera import Camera


def wallis_numer(n):
    return (n + 1) if n % 2 == 1 else n

def wallis_denom(n):
    return n if n % 2 == 1 else (n + 1)

def a(n):
    assert n >= 0
    return 1. if n == 0 else a(n-1) * float(2*n-1)/(2*n)


class WallisRectangles(VMobject):
    CONFIG = {
        "rect_colors" : [
            "#FF0000", "#FF8000", "#FFFF00", "#80FF00", "#00FF00",
            "#00FFFF", "#0080FF", "#0000FF", "#8000FF", "#FF00FF",
        ],
        "height" : 6,
    }
    def __init__(self, order = 570, **kwargs):
        digest_config(self, kwargs, locals())
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        self.adjust_colors()
        self.generate_rects()
        self.add(self.rects)
        self.scale_to_fit_height(self.height)
        self.center()

    def adjust_colors(self):
        if len(self.rect_colors) != self.order:
            self.rect_colors = color_gradient(self.rect_colors, self.order)

    def generate_rects(self):
        self.rects = VGroup()
        for p in range(0, self.order):
            self.rects.add(VGroup())
            for q in range(self.order - p):
                rect = self.generate_rectangle(p, q)
                if p == 0 and q == 0 :
                    pass
                elif p == 0 and q > 0:
                    align_rect = self.get_rectangle(p, q-1)
                    rect.next_to(align_rect, RIGHT, buff = 0)
                elif p > 0 and q == 0:
                    align_rect = self.get_rectangle(p-1, q)
                    rect.next_to(align_rect, UP, buff = 0)
                else:
                    align_rect = self.get_rectangle(p-1, q-1)
                    rect.next_to(align_rect, UP+RIGHT, buff = 0)
                self.rects[-1].add(rect)

    def generate_rectangle(self, i, j):
        height, width = self.get_side_lengths(i, j)
        fill_color = self.get_fill_color(i, j)
        rect = Rectangle(
            width = width, height = height,
            stroke_color = WHITE, stroke_width = 0,
            fill_color = fill_color, fill_opacity = 1
        )
        return rect

    def get_rectangle(self, i, j):
        return self.rects[i][j]

    def get_rectangles(self):
        return self.rects

    def get_fill_color(self, i, j):
        return self.rect_colors[i+j]    

    def get_side_lengths(self, i, j):
        return [a(i), a(j)]

    def get_order(self):
        return self.order

    def get_top_left(self):
        return np.array([self.get_left()[0], self.get_top()[1], 0])

    def get_bottom_left(self):
        return np.array([self.get_left()[0], self.get_bottom()[1], 0])

    def get_bottom_right(self):
        return np.array([self.get_right()[0], self.get_bottom()[1], 0])


class DeriveWallisProduct(Scene):
    def construct(self):
        self.add_figure()
        self.add_words()

    def add_figure(self):
        # Main figure
        wallis_rects = WallisRectangles(order = 10)
        order = wallis_rects.get_order()

        # Variables and braces on the left
        left_rects = [
            wallis_rects.get_rectangle(k, 0)
            for k in [0, 1, 2, 3, -1]
        ]
        left_braces = [
            Brace(rect, direction = LEFT, buff = 0.1)
            for rect in left_rects
        ]
        left_lengths = [
            TexMobject(symbol)
            for symbol in ["a_0", "a_1", "a_2", "a_3", "a_{n-1}"]
        ]
        left_lengths[-1].scale(0.7)
        for brace, length in zip(left_braces, left_lengths):
            brace.put_at_tip(length, buff = 0.15)
        left_vdots = TexMobject("\\vdots")
        left_vdots.scale(2)
        left_vdots.move_to(
            (left_braces[-1].get_center() + left_braces[-2].get_center())/2
        )

        # Variables and braces on the bottom
        down_rects = [
            wallis_rects.get_rectangle(0, k)
            for k in [0, 1, 2, 3, -1]
        ]
        down_braces = [
            Brace(rect, direction = DOWN, buff = 0.1)
            for rect in down_rects
        ]
        down_lengths = [
            TexMobject(symbol)
            for symbol in ["a_0", "a_1", "a_2", "a_3", "a_{n-1}"]
        ]
        down_lengths[-1].scale(0.7)
        for brace, length in zip(down_braces, down_lengths):
            brace.put_at_tip(length, buff = 0.15)
        down_cdots = TexMobject("\\cdots")
        down_cdots.scale(2)
        down_cdots.move_to(
            (down_braces[-1].get_center() + down_braces[-2].get_center())/2
        )

        # The quarter circle
        quarter_circle = Sector(
            outer_radius = wallis_rects.get_height(),
            stroke_color = GREY, stroke_width = 5.70,
            fill_opacity = 0,
        )
        quarter_circle.move_arc_center_to(wallis_rects.get_bottom_left())
        
        # Add everthing
        figure_group = VGroup(
            wallis_rects,
            VGroup(*left_braces), VGroup(*left_lengths), left_vdots,
            VGroup(*down_braces), VGroup(*down_lengths), down_cdots,
            quarter_circle,
        )
        figure_group.center().to_edge(LEFT, buff = 0.15)
        self.add(figure_group)

    def add_words(self):
        # Wallis product
        product = TexMobject(*[
            ["\\text{Wallis公式：}"] + [
                "{%d \\over %d} \\," % (wallis_numer(n), wallis_denom(n))
                for n in range(1, 8)
            ] + ["\\cdots = {\\pi \\over 2}"]
        ]).highlight(YELLOW)
        rect = SurroundingRectangle(product, color = YELLOW, buff = 0.25)
        wallis_product = VGroup(product, rect)
        wallis_product.scale_to_fit_width(6)

        # All those steps
        nums = [
            TextMobject("%d. " % k)
            for k in [1, 2, 3, 4]
        ]
        words = [
            TextMobject(word)
            for word in [
                "构造合适的矩形边长",
                "同种颜色的矩形的面积之和恒为1",
                "整个图形又像一个${1 \\over 4}$圆，半径是",
                "比较${1 \\over 4}$圆与矩形的面积",
            ]
        ]
        formulae = [
            TextMobject(formula)
            for formula in [
                "$a_0 = 1,\\, a_n = {1 \\over 2} \cdot {3 \\over 4} \cdots {2n-1 \\over 2n} (n \geq 1)$",
                "$a_0 a_n + a_1 a_{n-1} + \\cdots + a_n a_0 = 1$",
                "$\\begin{aligned} \
                r_n & = a_0 + a_1 + \\cdots + a_{n-1} \\\\ \
                    & = \\textstyle{3 \\over 2} \cdot {5 \\over 4} \cdots {2n-1 \\over 2n-2} \
                    \\quad (n \geq 2) \
                \\end{aligned}$",
                "${1 \\over 4} \\pi {r_n}^2 \\approx n \\quad \\Rightarrow \\quad \\text{Wallis公式}$"
            ]
        ]

        steps = VGroup()
        for num, word, formula in zip(nums, words, formulae):
            num.next_to(word, LEFT)
            formula.next_to(word, DOWN, aligned_edge = LEFT)
            steps.add(VGroup(num, word, formula))
        steps.arrange_submobjects(DOWN, buff = 0.6, aligned_edge = LEFT)
        steps.scale_to_fit_width(6)
        steps.next_to(wallis_product, DOWN)
        VGroup(wallis_product, steps).center().to_edge(RIGHT, buff = 0.15)

        # Sep line and QED
        sep_line = DashedLine(2*TOP, 2*BOTTOM, color = GREY, buff = 0.5)
        sep_line.next_to(steps, LEFT)
        qed = QEDSymbol(height = 0.570 / 2)
        qed.next_to(steps[-1][-1][-1], RIGHT, aligned_edge = DOWN)

        # Add everything
        self.add(wallis_product, steps, sep_line, qed)




