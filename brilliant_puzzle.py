#!/usr/bin/env python
#coding=utf-8

import numpy as np

from helpers import *
from mobject import Mobject
from mobject.vectorized_mobject import *
from mobject.point_cloud_mobject import *
from mobject.svg_mobject import *
from mobject.tex_mobject import *

from custom.custom_animations import *
from custom.custom_mobjects import *

from animation.animation import Animation
from animation.transform import *
from animation.simple_animations import *
from animation.playground import *

from topics.geometry import *
from topics.objects import *
from topics.number_line import *
from topics.three_dimensions import *
from topics.common_scenes import *

from scene import Scene
from camera import Camera

# self.skip_animations
# self.force_skipping()
# self.revert_to_original_skipping_status()

#####

LEFT_MID  = (LEFT_SIDE  + ORIGIN) / 2
RIGHT_MID = (RIGHT_SIDE + ORIGIN) / 2


class IntroduceBrilliant(Scene):
    def construct(self):
        title = TextMobject("Brilliant.org")
        title.to_edge(UP)
        rect = ScreenRectangle(height = 6)
        rect.next_to(title, DOWN)
        self.add(title)
        self.play(ShowCreation(rect))
        self.wait(1)


class QuestionScene(Scene):
    CONFIG = {
        "animate_construction" : True,
        "show_question" : True,
        "circle_radius" : 2.5,
        "circle_position" : DOWN / 3.,
        "n_sides" : 11,
        "polygon_color" : PURPLE,
        "line_colors" : [RED, YELLOW],
        "question_width" : 13,
    }

    def setup(self):
        # Change color settings
        if len(self.line_colors) < self.n_sides - 1:
            new_colors = color_gradient(self.line_colors, self.n_sides - 1)
            self.line_colors = new_colors

    def construct(self):
        if not self.animate_construction:
            self.force_skipping()
        self.show_circle_and_polygon()
        self.add_dots_and_labels()
        self.show_lines()
        self.write_question()
        self.revert_to_original_skipping_status()
        self.wait()

    def show_circle_and_polygon(self):
        circle = Circle(
            color = WHITE, start_angle = np.pi/2., radius = self.circle_radius
        )
        polygon = RegularPolygon(
            n = self.n_sides, start_angle = np.pi/2., color = self.polygon_color
        )
        polygon.scale(self.circle_radius)
        polygon.move_to(circle, aligned_edge = UP)
        VGroup(circle, polygon).shift(self.circle_position)

        self.play(ShowCreation(circle))
        self.wait()
        self.play(ShowCreation(polygon))
        self.wait()

        self.circle = circle
        self.polygon = polygon

    def add_dots_and_labels(self):
        dots = self.generate_dots()
        self.dots = dots

        labels = self.generate_dot_labels()
        self.get_labels_update(dots, labels).update(0)
        self.play(
            DrawBorderThenFill(dots),
            Write(labels),
            run_time = 2
        )
        self.wait()
        self.labels = labels

    def show_lines(self):
        a = self.vertices[0]
        other_vertices = self.vertices[1:]
        lines = VGroup()
        for vertex in other_vertices:
            line = Line(a, vertex, stroke_width = 10)
            lines.add(line)
        lines.gradient_highlight(*self.line_colors)

        self.play(
            ShowCreation(lines),
            Animation(self.dots),
            run_time = 2
        )
        self.wait()

        self.lines = lines

    def write_question(self):
        text = VGroup(*[
            TexMobject("\\left| %s \\right|" % ("A" + chr(ord("B") + i/2)))
            if i % 2 == 0
            else TexMobject("\\times")
            for i in range(2 * self.n_sides - 3)
        ])
        text.add(TexMobject("=?"))
        text.arrange_submobjects(RIGHT, buff = 0.1)
        text.scale_to_fit_width(self.question_width)
        text.move_to(self.circle.get_center())
        text.to_edge(UP)
        bg_rect = BackgroundRectangle(text, buff = 0.1)

        lengths = VGroup(*text[::2])
        for i, length in enumerate(lengths):
            length.highlight(self.line_colors[i])
        symbols = VGroup(*text[1::2])

        if self.show_question:
            dummy_mobs = VGroup(*[
                Dot(radius = 0).move_to(self.lines[i])
                for i in range(len(self.lines))
            ])
            dummy_mobs.gradient_highlight(*self.line_colors)
            self.play(
                FadeIn(bg_rect),
                ReplacementTransform(
                    dummy_mobs, lengths,
                    submobject_mode = "smoothed_lagged_start",
                ),
                run_time = 2,
            )
            self.wait()
            self.play(Write(symbols))
            self.wait()

        equation = VGroup(*[bg_rect, text])
        self.equation = equation

    #####

    def generate_dots(self):
        dots = VGroup(*[
            Dot(color = WHITE).move_to(vertex)
            for vertex in self.get_polygon_vertices()
        ])
        return dots

    def generate_dot_labels(self):
        texts = VGroup(*[
            TexMobject("%s" % chr(ord('A') + i))
            for i in range(len(self.dots))
        ])
        bg_rects = VGroup(*[
            BackgroundRectangle(text, buff = 0.05)
            for text in texts
        ])
        dot_labels = VGroup(*[
            VGroup(*[bg_rects[i], texts[i]])
            for i in range(len(texts))
        ])
        return dot_labels

    def get_figure(self):
        group = VGroup(*[
            self.circle, self.polygon, self.lines, self.dots
        ])
        return group

    def get_lengths_labels_in_equation(self):
        return VGroup(*self.equation[1][::2])

    def get_polygon_vertices(self):
        self.vertices = self.polygon.get_vertices()[:-1]
        return self.vertices

    def get_labels_update(self, dots, labels):
        def update_labels(self):
            for dot, label in zip(dots, labels):
                label.move_to(dot)
                vect = dot.get_center() - self.get_center()
                vect /= np.linalg.norm(vect)
                label.shift(0.4 * vect)
            return labels
        return UpdateFromFunc(labels, update_labels)


class SimpleCasesScene(QuestionScene):
    CONFIG = {
        "animate_construction" : False,
        "n_sides" : 3,
        "question_width" : 3,
        "circle_position" : DOWN / 3. + 3 * RIGHT,
        "results" : [
            ("正三边形", "3"), ("正四边形", "4"),
            ("正五边形", "???"), ("正六边形", "6"),
        ],
        # Status:
        #  1: Highlighted (lit);    0: Inactive (dim);
        # -1: Not shown        ; else: Normal
        "result_status" : (1, -1, -1, -1),
        "highlight_color" : BLUE,
        "row_buff" : 0.2,
        "column_buff" : 1,
    }
    def construct(self):
        self.add_table()
        QuestionScene.construct(self)

    def add_table(self):
        title = VGroup(*[
            TextMobject("正多边形"),
            TextMobject("乘积")
        ])
        title.arrange_submobjects(RIGHT, buff = self.column_buff)

        table_contents = self.generate_contents(self.results)
        self.change_content_status(table_contents, self.result_status)

        sep_line = Line(LEFT, RIGHT)
        sep_line.scale_to_fit_width(title.get_width())

        table = VGroup(title, sep_line, table_contents)
        table.arrange_submobjects(
            DOWN, buff = 1.5 * self.row_buff, aligned_edge = LEFT
        )
        table.to_corner(UP+LEFT)
        self.add(table)
        self.table = table

    def generate_contents(self, results):
        table_contents = VGroup(*[
            VGroup(*[
                TextMobject(text)
                for text in result
            ]).arrange_submobjects(RIGHT, buff = self.column_buff)
            for result in results
        ])
        table_contents.arrange_submobjects(
            DOWN, buff = self.row_buff, aligned_edge = LEFT
        )
        return table_contents

    def change_content_status(self, contents, status):
        for i, status in enumerate(status):
            if status == 1:          # Active / Highlight
                contents[i].highlight(self.highlight_color)
                contents[i][1].fade(1)
            elif status == 0:        # Inactive / Dim
                contents[i].fade(0.8)
            elif status == -1:       # Not Shown
                contents[i].fade(1)
            else:                    # Normal status - do nothing
                pass

    def get_table_contents(self):
        return self.table[2]

    def get_table_keys(self):
        contents = self.get_table_contents()
        table_keys = VGroup(*[content[0] for content in contents])
        return table_keys

    def get_num_characters(self):
        table_keys = self.get_table_keys()
        num_chars = VGroup(*[table_key[1] for table_key in table_keys])
        return num_chars
        
    def get_table_values(self):
        contents = self.get_table_contents()
        table_values = VGroup(*[content[1] for content in contents])
        return table_values

    def get_result_index(self):
        try:
            i = self.result_status.index(1)
        except:
            raise Exception("No desired targets found")
        return i

    def get_result_target(self):
        i = self.get_result_index()
        table_contents = self.get_table_contents()
        target = table_contents[i][1]
        return target

    # A hacky solution to VMobject replacement
    def replace_result_mob(self, mob):
        result_target = self.get_result_target()
        result_target.fade(0)
        self.play(ReplacementTransform(mob, result_target), run_time = 0)

    def transform_result_to_table(self):
        result_target = self.get_result_target().copy()
        result_target.fade(0)
        result_source = self.get_final_result().copy()
        self.play(
            ReplacementTransform(result_source, result_target)
        )
        self.replace_result_mob(result_source)
        self.remove(result_target)
        self.wait()

    def get_final_result(self):
        try:
            return self.final_result
        except:
            raise Exception()
        

class Case3Triangle(SimpleCasesScene):
    CONFIG = {
        "n_sides" : 3,
        "question_width" : 3,
    }
    def construct(self):
        SimpleCasesScene.construct(self)
        self.show_new_constructs()
        self.show_calculation()
        self.transform_result_to_table()
        self.add_caveat()

    def show_new_constructs(self):
        center_dot = Dot().move_to(self.circle.get_center())
        dashed_lines = VGroup(*[
            DashedLine(center_dot.get_center(), vertex_pos).highlight(GREY)
            for vertex_pos in self.get_polygon_vertices()[:2]
        ])
        radius_texts = VGroup(*[
            TexMobject("1").highlight(GREY)
            for k in range(len(dashed_lines)) 
        ])
        for i, text in enumerate(radius_texts):
            line = dashed_lines[i]
            shift_vec = rotate_vector(line.get_vector(), np.pi/2.)
            shift_vec /= np.linalg.norm(shift_vec)
            text.move_to(line)
            text.shift(shift_vec / 4. * (-1)**(i+1))
        self.play(
            ShowCreation(dashed_lines, submobject_mode = "all_at_once"),
            FadeIn(center_dot),
            Write(radius_texts),
            Animation(self.get_figure()),
            run_time = 1,
        )
        self.wait()

        A, B, C = vertices = self.get_polygon_vertices()
        angle_arc = AngleArc(
            *[A, center_dot, B], angle_text = "2\\pi \\over 3",
            arc_config = {"radius" : 0.3, "color" : GREY},
            tex_config = {"fill_color" : GREY},
            tex_shifting_factor = 2
        )

        self.play(Write(angle_arc), run_time = 1)
        self.wait()

        self.radius_texts = radius_texts

    def show_calculation(self):
        formula = TexMobject(
            "\\left| AB \\right|^2", "&=",
            "1", "^2", "+", "1", "^2", "-",
            "2\\cdot", "1", "\\cdot", "1",
            "\\cdot", "\\cos", "{2\\pi \\over 3}",
            "\\\\ &=", "3"
        )

        formula.scale_to_fit_width(6.5)
        formula.next_to(self.table, DOWN, aligned_edge = LEFT, buff = -1)
        self.play(Write(VGroup(*formula[:-2])))
        self.wait()
        self.play(Write(VGroup(*formula[-2:])))
        self.wait()

        self.final_result = formula[-1]
        self.radian = formula[-3]

    def add_caveat(self):
        arrow_end = self.radian.get_bottom()
        arrow_start = arrow_end + (DOWN + LEFT / 2.)
        arrow = Arrow(arrow_start, arrow_end)
        caveat = TextMobject("推广弧度制使用", "从你我做起")
        caveat.arrange_submobjects(DOWN)
        caveat.scale(0.5)
        caveat.next_to(arrow_start, DOWN, buff = 0)
        group = VGroup(caveat, arrow)
        group.highlight(GREY)
        self.play(FadeIn(group), run_time = 0.5)
        self.wait(0.5)


class Case4Square(SimpleCasesScene):
    CONFIG = {
        "result_status" : (0, 1, -1, -1),
        "n_sides" : 4,
        "question_width" : 4.5,
        "has_equation" : True,
        "length_texts" : ["\\sqrt{2}", "2", "\\sqrt{2}"],
        "lengths_buff" : 0,
        "equation_buff" : -0.5,
        "height" : 2.2,
    }

    def construct(self):
        SimpleCasesScene.construct(self)
        if self.has_equation:
            self.generate_equation_texts()
        self.show_all_lengths()
        self.show_calculation()
        self.transform_result_to_table()

    def generate_equation_texts(self):
        self.equation_texts = [
            self.length_texts[i/2] if i % 2 == 0 else "\\times"
            for i in range(2 * self.n_sides - 3)
        ]
        for extra_text in ["=", str(self.n_sides)]:
            self.equation_texts.append(extra_text)

    def show_all_lengths(self):
        equation = VGroup(*[
            TexMobject(text) for text in self.equation_texts
        ])
        equation.arrange_submobjects(RIGHT, buff = 0.2)
        equation.next_to(
            self.table, DOWN, aligned_edge = LEFT, buff = self.equation_buff
        )
        final_result = equation[-1]
        equation_values = VGroup(*equation[:-1:2])

        length_results = VGroup(*[
            TexMobject("\\left|A%s\\right|" % (chr(ord('B') + i)), "=", text)
            for i, text in enumerate(self.length_texts)
        ])
        length_results.arrange_submobjects(DOWN, aligned_edge = LEFT)
        length_results.scale_to_fit_height(self.height)
        length_results.next_to(self.table, DOWN, buff = self.lengths_buff)
        length_results.to_edge(LEFT, buff = 1)

        length_labels_source = self.get_lengths_labels_in_equation().copy()
        length_labels_target = VGroup(*[result[0] for result in length_results])
        fade_parts = VGroup(*[VGroup(*result[:-1]) for result in length_results])
        rem_parts = VGroup(*[VGroup(*result[1:]) for result in length_results])
        length_values = VGroup(*[result[-1] for result in length_results])

        for values in (length_results, equation_values):
            for i, value in enumerate(values):
                value.highlight(self.line_colors[i])
        equation_symbols = VGroup(*equation[1::2]).add(final_result)

        self.play(
            ReplacementTransform(length_labels_source, length_labels_target)
        )
        self.wait()
        self.play(Write(rem_parts))
        self.wait()

        self.fade_parts = fade_parts
        self.length_values = length_values
        self.equation_values = equation_values
        self.equation_symbols = equation_symbols
        self.final_result = final_result

    def show_calculation(self):
        self.play(
            FadeOut(self.fade_parts),
            ReplacementTransform(self.length_values, self.equation_values, path_arc = np.pi/2.),
        )
        self.play(Write(self.equation_symbols))
        self.wait()


class Case5Pentagon(SimpleCasesScene):
    CONFIG = {
        "result_status" : (0, 0, 1, -1),
        "has_equation" : False,
        "n_sides" : 5,
        "question_width" : 6,
    }
    def construct(self):
        SimpleCasesScene.construct(self)
        self.special_values_are_not_common_sense()
        self.transform_result_to_table()
        self.decide_to_skip()

    def special_values_are_not_common_sense(self):
        special_values = TexMobject(
            "\\sin {\\pi \\over 5}", "=",
            "\\sqrt{{5 \\over 8} - {\\sqrt{5} \\over 8}}"
        ).scale(0.7)
        question_marks = TextMobject("???")

        # Yet another hacky solution to bubble placements
        speech = SpeechBubble()
        speech.to_corner(LEFT+DOWN)
        speech.add_content(special_values)
        speech.resize_to_content()

        thought = ThoughtBubble().flip()
        thought.to_corner(LEFT+DOWN).shift(1.2 * LEFT)
        thought.add_content(question_marks)
        thought.resize_to_content()

        for mob in (speech, thought):
            Group(mob, mob.content).shift(0.5 * UP)

        self.play(BubbleCreation(speech))
        self.wait()
        self.play(BubbleCreation(thought))
        self.wait()

        self.final_result = thought.content
        self.speech = speech
        self.thought = thought

    def decide_to_skip(self):
        skip_tex = TextMobject("超纲了 \\\\ 换一个").scale(0.75)
        skip_speech = SpeechBubble(direction = RIGHT)
        skip_speech.add_content(skip_tex)
        skip_speech.resize_to_content()
        skip_speech.move_to(self.thought)
        skip_speech.add_content(skip_tex)
        tip = skip_speech.get_tip()
        self.play(
            BubbleFadeOut(self.thought),
            BubbleGrowFromPoint(
                skip_speech,
                bubble_animation_args = [tip],
                content_animation_args = [tip],
            )
        )
        self.wait()


class Case6Hexagon(Case4Square):
    CONFIG = {
        "result_status" : (0, 0, 0, 1),
        "n_sides" : 6,
        "question_width" : 7.5,
        "length_texts" : ["1", "\\sqrt{3}", "2", "\\sqrt{3}", "1"],
        "height" : 3,
        "lengths_buff" : 0.5,
        "equation_buff" : 0.5,
    }


class SimpleCaseClosure(Case6Hexagon):
    CONFIG = {
        "scheme_height" : 6.5,
        "hypothesis_color" : YELLOW,
        "random_seed" : hash("Solara") % 570,
    }
    def construct(self):
        self.skip_animations = True
        Case6Hexagon.construct(self)
        self.skip_animations = False
        self.focus_on_the_table()
        self.highlight_num_chars()
        self.show_hypothesis()
        self.three_step_plan()

    def focus_on_the_table(self):
        fade_mobs = VGroup(*[
            self.get_figure(), self.labels, self.equation,
            self.equation_values, self.equation_symbols,
        ])

        self.table.generate_target()
        self.table.target.center()
        self.table.target.scale(1.2)
        self.table.target.fade(0)
        self.table.target.highlight(WHITE)

        self.play(FadeOut(fade_mobs))
        self.play(MoveToTarget(self.table))
        self.wait()

    def highlight_num_chars(self):
        chars = self.get_num_characters()
        chars_rect = SurroundingRectangle(chars)
        values = self.get_table_values()
        values_rect = SurroundingRectangle(values)
        for mob, rect in [(chars, chars_rect), (values, values_rect)]:
            self.play(
                mob.highlight, self.hypothesis_color,
                ShowCreationThenDestruction(rect)
            )
        self.wait()

    def show_hypothesis(self):
        hypothesis = TextMobject(
            "猜想：在正", "$n$", "边形中，所求乘积为", "$n$",
            arg_separator = ""
        )
        hypothesis.to_edge(UP)
        hypothesis.highlight_by_tex("$n$", self.hypothesis_color)
        self.play(Write(hypothesis))
        self.wait()

        self.hypothesis = hypothesis

    def three_step_plan(self):
        # Preparation
        sep_line = Line(LEFT, RIGHT)
        sep_line.scale_to_fit_width(2 * SPACE_WIDTH)
        sep_line.next_to(self.hypothesis, DOWN, buff = 0.5)
        self.play(
            ApplyMethod(self.hypothesis.to_corner, LEFT+UP),
            FadeOut(self.table),
        )
        self.play(ShowCreation(sep_line))
        self.wait()

        step1 = TextMobject("1.", "求出每条线段的长度")
        step2 = TextMobject("2.", "求出所有线段的长度的乘积")
        step3 = TextMobject("3.", "验证猜想的正确性")
        steps = VGroup(step1, step2, step3)
        steps.arrange_submobjects(DOWN, aligned_edge = LEFT, buff = 1.5)
        steps.center().to_edge(LEFT)

        step1_remark = TexMobject(*[
            "x", "=\\,?\\quad", "y", "=\\,?\\quad", "z", "=\\,?\\quad", "w", "=\\,?"
        ])
        step2_remark = TexMobject(*["x", "y", "z", "w", "=\\,?"])
        step3_remark = TextMobject("并且推出一个有趣的结论")
        remarks = VGroup(step1_remark, step2_remark, step3_remark)

        for pair in zip(steps, remarks):
            step, remark = pair
            remark.next_to(step[1], DOWN, aligned_edge = LEFT)

        lines = VGroup(*[
            Line(ORIGIN, pt[0] * RIGHT + pt[1] * UP, stroke_width = 10)
            for pt in (np.random.rand(4, 2) - 0.5) * self.scheme_height
        ])
        end_points = [ORIGIN] + [line.get_end() for line in lines]
        end_dots = VGroup(*[Dot(end_point) for end_point in end_points])
        texs = TexMobject("x", "y", "z", "w")
        braces = VGroup(*[
            Brace(
                line,
                direction = rotate_vector(line.get_vector(), np.pi/2),
                width_multiplier = 1,
                buff = 0.05,
            ).put_at_tip(tex, buff = 0.05)
            for k, (line, tex) in enumerate(zip(lines, texs))
        ])
        for mobs in (lines, texs, VGroup(*step1_remark[::2]), VGroup(*step2_remark[:-1])):
            mobs.gradient_highlight(*self.line_colors)
        
        scheme = VGroup(lines, end_dots, texs, braces)
        scheme.shift(4*RIGHT + 0.7*DOWN)

        # 1. Find the lengths
        self.play(Write(step1))
        self.wait()
        self.play(
            DrawBorderThenFill(end_dots, submobject_mode = "all_at_once"),
            run_time = 1
        )
        self.play(
            ShowCreation(lines, submobject_mode = "all_at_once"),
            Animation(end_dots)
        )
        self.wait()
        self.play(
            AnimationGroup(*[GrowFromCenter(brace) for brace in braces]),
            AnimationGroup(*[Write(tex) for tex in texs]),
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                ReplacementTransform(tex.copy(), target)
                for k, (tex, target) in enumerate(zip(texs, step1_remark[::2]))
            ]),
            Write(VGroup(*step1_remark[1::2])),
        )
        self.wait()
        
        # 2. Calculate the product of the lengths
        self.play(Write(step2))
        self.wait()
        self.play(
            ReplacementTransform(VGroup(*step1_remark[::2]), VGroup(*step2_remark[:-1])),
            ReplacementTransform(step1_remark[-1].copy(), step2_remark[-1])
        )
        self.wait()

        # 3. Prove/Disprove the hypothesis
        self.play(Write(step3))
        self.wait()
        self.play(Write(step3_remark))
        self.wait()









