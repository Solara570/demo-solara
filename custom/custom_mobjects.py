#coding=utf-8

import random
import itertools as it

from manimlib.constants import *
from manimlib.mobject.types.vectorized_mobject import VMobject, VGroup
from manimlib.mobject.svg.tex_mobject import TexMobject
from manimlib.mobject.geometry import Rectangle, Square, Annulus, RegularPolygon
from manimlib.once_useful_constructs.fractals import fractalify
from manimlib.utils.space_ops import get_norm

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
        symbol.set_height(self.inner_radius)
        return symbol


class PlayButton(Button):
    def generate_symbol(self):
        symbol = RegularPolygon(
            n = 3, stroke_width = 0,
            fill_color = self.color, fill_opacity = 1,
        )
        symbol.set_height(self.inner_radius)
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
        symbol.set_height(self.inner_radius * 0.7)
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
        symbol.set_height(self.inner_radius * 0.7)
        return symbol


class StopButton(Button):
    def generate_symbol(self):
        symbol = Square(stroke_width = 0, fill_color = self.color, fill_opacity = 1)
        symbol.set_height(self.inner_radius * 0.8)
        return symbol


class TickButton(Button):
    CONFIG = {
        "color" : GREEN,
    }
    def generate_symbol(self):
        symbol = TexMobject("\\checkmark")
        symbol.set_color(self.color)
        symbol.set_height(self.inner_radius)
        return symbol


## Danger Sign
class HollowTriangle(VMobject):
    CONFIG = {
        "inner_height" : 3.5,
        "outer_height" : 5,
        "color" : RED,
        "fill_opacity" : 1,
        "stroke_width" : 0,
        "mark_paths_closed" : False,
        "propagate_style_to_family" : True,
    }
    def generate_points(self):
        self.points = []
        inner_tri = RegularPolygon(n = 3, start_angle = np.pi/2)
        outer_tri = RegularPolygon(n = 3, start_angle = np.pi/2)
        inner_tri.flip()
        inner_tri.set_height(self.inner_height, about_point = ORIGIN)
        outer_tri.set_height(self.outer_height, about_point = ORIGIN)
        self.points = outer_tri.points
        self.add_subpath(inner_tri.points)


class DangerSign(VMobject):
    CONFIG = {
        "color" : RED,
        "triangle_config" : {},
    }
    def generate_points(self):
        hollow_tri = HollowTriangle(**self.triangle_config)
        bang = TexMobject("!")
        bang.set_height(hollow_tri.inner_height * 0.7)
        bang.move_to(hollow_tri.get_center_of_mass())
        self.add(hollow_tri, bang)
        self.set_color(self.color)


## QED symbols
# Used when try to claim something proved/"proved".
class QEDSymbol(VMobject):
    CONFIG = {
        "color" : WHITE,
        "height" : 0.5,
    }
    def generate_points(self):
        qed = Square(fill_color = self.color, fill_opacity = 1, stroke_width = 0)
        self.add(qed)
        self.set_height(self.height)


class FakeQEDSymbol(VMobject):
    CONFIG = {
        "jagged_percentage" : 0.02,
        "order" : 1,
        "qed_config" : {},
    }
    def generate_points(self):
        fake_qed = fractalify(
            QEDSymbol(**self.qed_config),
            order = self.order, dimension = 1+self.jagged_percentage
        )
        self.add(fake_qed)


## Sudoku board
# A byproduct during the experiment. Not finalized.
class SudokuBoard(VGroup):
    CONFIG = {
        "n" : 3,
        "height" : 6,
    }
    def generate_points(self):
        n = self.n
        self.small_squares = VGroup(*[
            Square(side_length = 1, stroke_color = GREY, stroke_width = 3)
            for k in range(n**4)
        ])
        self.small_squares.arrange_submobjects_in_grid(n**2, n**2, buff = 0)
        self.big_squares = VGroup(*[
            Square(side_length = self.n, stroke_color = WHITE, stroke_width = 8)
            for k in range(n**2)
        ])
        self.big_squares.arrange_submobjects_in_grid(n, n, buff = 0)
        self.entries = dict()
        self.add(self.small_squares, self.big_squares)
        self.center()
        self.set_height(self.height)

    def add_entry(self, x, y, entry):
        # x: row index (up->down, 0->8); y: column index (left->right, 0->8)
        text = TexMobject(str(entry))
        square = self.get_square(x, y)
        fit_mobject_in(text, square)
        self.entries[(x, y)] = text
        self.add(text)

    def add_entries(self, entries):
        for (x, y), entry in entries:
            self.add_entry(x, y, entry)

    def get_square(self, x, y):
        square_index = (self.n**2) * x + y
        return self.small_squares[square_index]

    def get_square_critical_point(self, x, y, direction):
        square = self.get_square(x, y)
        return square.get_critical_point(direction)

    def get_entry(self, x, y):
        return self.entries[(x, y)]

    def get_elimination_arrow(self, x1, y1, x2, y2, direction, **arrow_kwargs):
        nudge = self.get_square(0, 0).get_height() * 0.05 * (-direction)
        start_point = self.get_square_critical_point(x1, y1, direction) + nudge
        end_point = self.get_square_critical_point(x2, y2, direction) + nudge
        arrow = Arrow(start_point, end_point, buff = 0, **arrow_kwargs)
        return arrow



