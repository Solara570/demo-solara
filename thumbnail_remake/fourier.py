#!/usr/bin/env python
#coding=utf-8

from active_projects.fourier import *


class Thumbnail(Scene):
    def construct(self):
        title = TextMobject("形象展示 \\\\ 傅里叶变换")
        title.highlight(YELLOW)
        title.set_stroke(RED, 1)
        title.scale(2)
        title.add_background_rectangle()

        def func(t):
            return np.cos(2*TAU*t) + np.cos(3*TAU*t) + np.cos(5*t)
        fourier = get_fourier_transform(func, -5, 5)

        graph = FunctionGraph(func, x_min = -5, x_max = 5)
        graph.highlight(BLUE)
        fourier_graph = FunctionGraph(fourier, x_min = 0, x_max = 6)
        fourier_graph.highlight(RED_C)
        for g in graph, fourier_graph:
            g.stretch_to_fit_height(2)
            g.stretch_to_fit_width(10)
            g.set_stroke(width = 8)
        arrow = Vector(
            2.5*DOWN, 
            rectangular_stem_width = 0.2,
            tip_length = 0.5,
            color = WHITE
        )
        title_group = VGroup(arrow.copy(), title, arrow.copy())
        title_group.arrange_submobjects(RIGHT, buff = 0.5)

        group = VGroup(graph, title_group, fourier_graph)
        group.arrange_submobjects(DOWN)
        self.add(group)



