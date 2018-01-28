#!/usr/bin/env python
#coding=utf-8

from active_projects.fourier import *


class Thumbnail(FourierMachineScene):
    CONFIG = {
        "func_color" : BLUE,
        "fourier_color" : YELLOW,
    }
    def construct(self):
        self.generate_func_graphs()
        self.generate_wound_up_graphs()
        self.generate_title()
        self.add_mobs()

    def generate_func_graphs(self):
        def func(t):
            return np.cos(2*TAU*t) + np.cos(3*TAU*t) + np.cos(5*t)
        fourier = get_fourier_transform(func, -5, 5)
        func_graph = FunctionGraph(func, x_min = -5, x_max = 5)
        func_graph.highlight(self.func_color)
        fourier_graph = FunctionGraph(fourier, x_min = 0, x_max = 6)
        fourier_graph.highlight(self.fourier_color)
        for g in func_graph, fourier_graph:
            g.stretch_to_fit_height(2)
            g.stretch_to_fit_width(10)
            g.set_stroke(width = 8)
        self.func_graph = func_graph
        self.fourier_graph = fourier_graph

    def generate_wound_up_graphs(self):
        self.time_axes = NumberPlane()
        title_func = lambda t : np.cos(2*TAU*t) + np.cos(3*TAU*t) + 2
        intval = 2E-2 / 3.
        wound_up_graphs = VGroup()
        for freq in np.linspace(2 - 3*intval, 2 + 3*intval, 7):
            title_graph = FunctionGraph(title_func, x_min = 0, x_max = 10)
            title_fourier_graph = self.get_polarized_mobject(title_graph, freq = freq)
            title_fourier_graph.rotate((freq - 2) * 5 * TAU)
            wound_up_graphs.add(title_fourier_graph)
        wound_up_graphs.gradient_highlight(*[self.func_color, self.fourier_color])
        wound_up_graphs.arrange_submobjects(RIGHT, buff = 2)
        wound_up_graphs.scale_to_fit_width(2 * SPACE_WIDTH - 1)
        self.wound_up_graphs = wound_up_graphs

    def generate_title(self):
        title = TextMobject("形象展示 \\\\ 傅里叶变换")
        title.highlight(YELLOW)
        title.set_stroke(RED, 1)
        title.scale(2)
        title.add_background_rectangle()
        self.title = title

    def add_mobs(self):
        VGroup(
            self.func_graph, self.title, self.fourier_graph
        ).arrange_submobjects(DOWN, buff = 0.3)
        self.add(
            self.func_graph, self.wound_up_graphs,
            self.fourier_graph, self.title
        )





