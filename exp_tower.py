#!/usr/bin/env python
#coding=utf-8

import numpy as np

from helpers import *
from mobject import Mobject
from mobject.vectorized_mobject import *
from mobject.tex_mobject import *

from animation.animation import Animation
from animation.transform import *
from animation.simple_animations import *

from custom.custom_mobjects import *

from topics.geometry import *

from scene import Scene
from camera import Camera

# self.skip_animations
# self.force_skipping()
# self.revert_to_original_skipping_status()

#####
## Mobjects

class ExpDots(VMobject):
    CONFIG = {
        "angle" : TAU / 8,
        "gap_buff" : 0.1,
    }
    def generate_points(self):
        dots = TexMobject(list("..."))
        dots.arrange_submobjects(RIGHT, buff = self.gap_buff)
        dots.rotate(self.angle)
        dots.scale(1.5)
        self.add(dots)


class ExpTower(VMobject):
    CONFIG = {
        "order"   : 5,
        "element" : "x",
        "tower_texts" : None,
        "scale_factor" : 0.8,
        "height_stretch_factor" : 0.8,
        "is_infinite"  : True,
        "expdots_config" : {},
    }
    def generate_points(self):
        tower_texts = self.get_tower_texts()
        tower_tex = self.generate_tower_tex_from_texts(tower_texts)
        self.tower_tex = tower_tex
        self.add(tower_tex.center())

    def get_tower_texts(self):
        if not hasattr(self, "tower_texts") or self.tower_texts is None:
            self.tower_texts = [self.element] * self.order
        assert len(self.tower_texts) == self.order
        return self.tower_texts

    def generate_tower_tex_from_texts(self, texts):
        tower_tex = VGroup(*[
            TexMobject(text)
            for text in self.tower_texts
        ])
        if self.is_infinite:
            tower_tex.add(ExpDots(**self.expdots_config))
        for k, part in enumerate(tower_tex):
            part.scale(self.scale_factor**k)
            if k > 0:
                buff = 0.05 / np.sqrt(k)
                part.stretch(self.height_stretch_factor, 1)
                part.next_to(tower_tex[k-1], RIGHT+UP, buff = buff)
        return tower_tex

    def get_tower(self):
        return self.tower_tex

    def get_exponent(self):
        return VGroup(self.tower_tex[1:])

    def get_elements(self):
        if self.is_infinite:
            return VGroup(self.tower_tex[:-1])
        else:
            return self.get_tower()

    def get_tex_by_order(self, order):
        return self.tower_tex[order]

    def get_base(self):
        return self.get_tex_by_order(0)

    def get_expdots(self):
        if self.is_infinite:
            return self.get_tex_by_order(-1)
        else:
            raise Exception("No ExpDots found!")


class CoverRectangle(VMobject):
    CONFIG = {
        "stroke_width" : 5,
        "stroke_color" : YELLOW,
        "fill_color"   : BLACK,
        "fill_opacity" : 0.7,
        "text"         : "",
        "text_color"   : YELLOW,
        "text_height_factor" : 0.6, 
    }
    def __init__(self, covered_mob, **kwargs):
        self.covered_mob = covered_mob
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        rect = SurroundingRectangle(self.covered_mob)
        rect.set_style_data(
            stroke_color = self.stroke_color, stroke_width = self.stroke_width,
            fill_color = self.fill_color, fill_opacity = self.fill_opacity
        )
        text = TextMobject(str(self.text))
        text.highlight(self.text_color)
        text.scale_to_fit_height(rect.get_height() * self.text_height_factor)
        text.move_to(rect)
        self.group = VGroup(rect, text)
        self.add(self.group)

    def get_text_mob(self):
        return self.group[1]


#####
## Scenes

class ExpTowerIntro(Scene):
    def construct(self):
        self.build_the_tower()
        self.build_the_equation()
        self.ask_to_solve()
        self.countdown()
        self.show_the_solution()
        self.fadeout_everthing()

    def build_the_tower(self):
        highest_order = 6
        towers = [
            ExpTower(order = k, is_infinite = False)
            for k in range(1, highest_order + 1)
        ]
        towers.append(ExpTower(order = highest_order, is_infinite = True))
        heights = list(np.linspace(3, 4.5, highest_order + 2))
        for tower, height in zip(towers, heights):
            tower.scale_to_fit_height(height)

        init_tower = towers[0]
        final_tower = towers[-1]
        self.play(Write(init_tower))
        self.wait()
        for k, tower in enumerate(towers):
            if k < len(towers) - 2:
                new_tower = towers[k+1]
                new_exponent = new_tower.get_exponent()
                new_base = new_tower.get_base()
                self.play(
                    ReplacementTransform(tower, new_exponent),
                    Write(new_base),
                )
                self.wait()
            else:
                xs = final_tower.get_elements()
                expdots = final_tower.get_expdots()
                sur_rect = SurroundingRectangle(expdots)
                self.play(
                    ReplacementTransform(tower, xs, run_time = 1),
                    Write(expdots, run_time = 2),
                )
                self.play(
                    Indicate(expdots, run_time = 1),
                    ShowCreationThenDestruction(sur_rect, run_time = 2),
                )
                self.wait()
                break
        self.tower = final_tower
        self.highest_order = highest_order

    def build_the_equation(self):
        l_part = self.tower.copy()
        r_part = TexMobject("=", "2")
        r_part.match_height(l_part.get_base())
        equation = VGroup(l_part, r_part)
        equation.arrange_submobjects(RIGHT, aligned_edge = DOWN)

        self.play(
            ReplacementTransform(self.tower, l_part, run_time = 1),
            Write(r_part, run_time = 2),
        )
        self.equation = equation

    def ask_to_solve(self):
        question = TextMobject("解", "方程（$x>0$）", "：")
        solve = TextMobject("解", "：")
        for tex in (question, solve):
            tex.scale(1.2)
            tex.to_corner(LEFT+UP)
        self.play(Write(question), run_time = 1)
        self.wait()
        self.question = question
        self.solve = solve

    def countdown(self):
        for k in range(5, -1, -1):
            countdown_text = TexMobject(str(k))
            countdown_text.scale(1.5)
            countdown_text.highlight(YELLOW)
            countdown_text.to_corner(RIGHT+UP)
            self.add(countdown_text)
            self.wait()
            self.remove(countdown_text)
    
    def show_the_solution(self):
        # Prepare for the solution
        pre_solve = VGroup(*self.question[::2])
        self.play(
            ReplacementTransform(pre_solve, self.solve),
            FadeOut(self.question[1]),
            run_time = 1,
        )
        self.wait()

        # Manipulate LHS
        old_l_part, r_part = self.equation
        new_l_part = ExpTower(order = self.highest_order+1, is_infinite = True)
        new_l_part.match_height(old_l_part)
        new_l_part.next_to(r_part, LEFT, aligned_edge = DOWN)
        old_rect, new_rect = rects = [
            CoverRectangle(part, text = "2")
            for part in (old_l_part, new_l_part.get_exponent())
        ]
        old_two, new_two = twos = [
            rect.get_text_mob()
            for rect in rects
        ]
        self.play(DrawBorderThenFill(old_rect, run_time = 1))
        self.wait()
        self.play(
            ReplacementTransform(old_l_part, new_l_part),
            ReplacementTransform(old_rect, new_rect),
        )
        self.wait()
        new_equation = VGroup(new_l_part, r_part, new_rect)
        new_equation.generate_target()
        new_equation.target.scale(0.8)
        new_equation.target.shift(UP)
        self.play(MoveToTarget(new_equation))
        self.wait()

        # A little bit clean-up
        source_eq = VGroup(*[
            mob.copy()
            for mob in (new_l_part.get_base(), new_two, r_part[0], r_part[1])
        ])
        source_eq.generate_target()
        target_eq = TexMobject("x", "^2", "=", "2")
        target_eq.scale(3).next_to(source_eq, DOWN, buff = 1)
        for k, (old_part, new_part) in enumerate(zip(source_eq.target, target_eq)):
            old_part.move_to(new_part)
            if k == 1:
                old_part.scale(0.5)
                old_part.shift(RIGHT/4)
        self.play(
            FadeOut(new_rect),
            MoveToTarget(source_eq),
        )
        self.wait()

        # Reveal the final answer
        result = TexMobject("x", "=", "\\sqrt", "2")
        result.scale_to_fit_height(source_eq.get_height() * 0.7)
        result.move_to(source_eq)
        self.play(*[
            ReplacementTransform(source_eq[m], result[n], path_arc = angle)
            for m, n, angle in [(0, 0, 0), (1, 2, -TAU/3), (2, 1, 0), (3, 3, 0)]
        ])
        self.wait()
        qed = QEDSymbol()
        qed.next_to(result, RIGHT, aligned_edge = DOWN, buff = 1.5)
        self.play(FadeIn(qed))
        self.wait()

    def fadeout_everthing(self):
        mobs = VGroup(self.mobjects)
        self.play(FadeOut(mobs), run_time = 1)


class IntuitiveButDangerous(Scene):
    def construct(self):
        self.show_comments()
        self.rewind_and_play_again()

    def show_comments(self):
        height = 1.5
        simple, but, trap = texts = TextMobject("简单直观", "但是", "暗藏陷阱")
        texts.scale(1.5)
        texts.arrange_submobjects(DOWN, buff = 0.8)
        tick = TickButton()
        tick.scale_to_fit_height(height)
        tick.next_to(simple, RIGHT)
        danger = DangerSign()
        danger.scale_to_fit_height(height)
        danger.next_to(trap, LEFT)
        words = [(simple, tick), (but, ), (danger, trap)]
        colors = [GREEN, WHITE, RED]
        for word, color in zip(words, colors):
            VGroup(word).highlight(color)
            self.play(FadeIn(VGroup(word)), run_time = 1)
            self.wait(0.5)
        self.wait()
        self.words = words

    def rewind_and_play_again(self):
        watch_again = TextMobject("再看一遍...")
        watch_again.highlight(YELLOW)
        rewind, play = buttons = VGroup(*[
            Button().get_symbol().highlight(YELLOW).scale_to_fit_height(1)
            for Button in RewindButton, PlayButton
        ])
        group = VGroup(watch_again, rewind)
        group.arrange_submobjects(DOWN, aligned_edge = RIGHT)
        group.to_corner(DOWN+RIGHT)
        play.move_to(rewind, aligned_edge = RIGHT)
        self.play(Write(watch_again), run_time = 0.5)
        self.wait()
        self.add(rewind)
        self.words.reverse()
        for word in self.words:
            self.play(FadeOut(VGroup(word)), run_time = 0.2)
            self.wait(0.1)
        self.remove(rewind)
        self.add(play)
        self.play(FadeOut(watch_again), run_time = 0.5)
        self.wait(0.5)
        self.remove(play)


class Prove2Equals4(Scene):
    CONFIG = {
        "tower_edge_buff" : 0.8,
    }
    def construct(self):
        self.rhs_could_be_any_number()
        self.choose_two_special_numbers()
        self.solve_the_equations()
        self.reveal_2_is_4()

    def rhs_could_be_any_number(self):
        rhs_list = [
            "2",         # The journey starts here.
            "3",         # The first number that pops into your head when you hear "2".
            "\\pi",      # The good ol' pi.
            "\\sqrt{2}", # The (most) famous irrational number.
            "42",        # The answer to everything.
            "17.29",     # TAXI.CAB
            "-1",        # A negative number, because why not?
            "e",         # The (most) famous mathematical constant.
            "\\dfrac{3+\\sqrt{13}}{2}", # The bronze ratio... and a fraction.
            "570",       # 1/3 of the author.
            "\\varphi",  # The golden ratio.
            "g_{64}",    # Gigigigigigigigantic...
            "0",         # And... stop!
            "N",
        ]
        anything_text = TextMobject("等号右边似乎可以放任何数...")
        anything_text.highlight(YELLOW)
        anything_text.to_edge(UP)
        equations = [
            VGroup(ExpTower(order = 5), TexMobject("=", rhs).scale(0.8)).scale(2.5)
            for rhs in rhs_list
        ]
        init_equation = equations[0]
        init_equation.arrange_submobjects(RIGHT, aligned_edge = DOWN)
        init_equation.next_to(anything_text, DOWN, buff = self.tower_edge_buff)
        init_equation.to_edge(LEFT, buff = self.tower_edge_buff)
        for equation in equations:
            equation[0].move_to(init_equation[0])
            equation[1][0].move_to(init_equation[1][0])
            equation[1][1].move_to(init_equation[1][1], aligned_edge = LEFT)

        self.play(FadeIn(init_equation))
        self.wait()
        self.play(Write(anything_text), run_time = 1)
        self.play(
            ShowCreationThenDestruction(SurroundingRectangle(init_equation[1][1])),
            run_time = 2
        )
        self.wait()
        for k, equation in enumerate(equations):
            if k > 0 and k != len(equations)-1:
                equation[0].move_to(init_equation[0])
                equation[1].move_to(init_equation[1], aligned_edge = LEFT)
                self.remove(equations[k-1])
                self.add(equations[k])
                self.wait(1./3)
            elif k == len(equations)-1:
                self.wait()
                self.play(ReplacementTransform(equations[k-1], equations[k]))
                self.wait()

        self.anything_text = anything_text
        self.equation = equations[-1]

    def choose_two_special_numbers(self):
        two_and_four_text = TextMobject("现在选择两个特殊的数...")
        two_and_four_text.highlight(YELLOW)
        two_and_four_text.to_edge(UP)
        self.play(Transform(self.anything_text, two_and_four_text))
        self.wait()

        two_equation = self.equation
        four_equation = two_equation.copy()
        self.play(four_equation.to_edge, RIGHT, self.tower_edge_buff)
        self.wait()

        nums = [2, 4]
        colors = [GREEN, RED]
        equations = [two_equation, four_equation]
        targets = [equation[1][1] for equation in equations]
        two, four = num_texs = [
            TexMobject(str(num)).highlight(color).match_height(target).move_to(target)
            for num, color, equation, target in zip(nums, colors, equations, targets)
        ]

        for N_tex, num_tex in zip(targets, num_texs):
            self.play(Transform(N_tex, num_tex))
            self.wait(0.5)
        self.wait(0.5)

        self.nums = nums
        self.colors = colors
        self.equations = equations
        self.x_towers = VGroup(*[equation[0] for equation in equations])
        
    def solve_the_equations(self):
        rects = VGroup(*[
            CoverRectangle(
                equation[0].get_exponent(), stroke_color = color,
                text = str(num), text_color = color
            )
            for equation, color, num in zip(self.equations, self.colors, self.nums)
        ])
        self.play(DrawBorderThenFill(rects))
        self.wait()

        sps = VGroup()
        for equation, num, color in zip(self.equations, self.nums, self.colors):
            sp = TexMobject("x", "^{%d}" % num, "=", "%d" % num)
            sp[1::2].highlight(color)
            sp.scale(2)
            sp.next_to(equation, DOWN, buff = 1)
            sps.add(sp)

        rss = VGroup()
        for num, sp, color in zip(self.nums, sps, self.colors):
            rs = TexMobject("x", "=", "%d" % num , "^{{1}\\over{%d}}" % num, "=\\sqrt{2}")
            for tex in (rs[2], rs[3][2]):
                tex.highlight(color)
            rs.match_height(sp).move_to(sp)
            rss.add(rs)

        tf_anims = []
        for sp, rs in zip(sps, rss):
            tf_anims.append(ReplacementTransform(sp[0], rs[0]))
            tf_anims.append(ReplacementTransform(sp[1], rs[3][2], path_arc = -TAU/4))
            tf_anims.append(ReplacementTransform(sp[2], rs[1]))
            tf_anims.append(ReplacementTransform(sp[3], rs[2]))
            tf_anims.append(Write(rs[3][:2]))

        self.play(FadeIn(sps))
        self.wait()
        self.play(AnimationGroup(*tf_anims), run_time = 2)
        self.wait()
        self.play(Write(VGroup(*[rs[4:] for rs in rss]), submobject_mode = "all_at_once"))
        self.wait()
        self.play(FadeOut(rects))
        self.wait()

        self.rss = rss

    def reveal_2_is_4(self):
        sqrt2_towers = VGroup(*[
            ExpTower(element = "\\sqrt{2}", order = 5).match_height(x_tower) \
                .move_to(x_tower, aligned_edge = RIGHT)
            for x_tower in self.x_towers
        ])
        self.play(
            Transform(self.x_towers, sqrt2_towers, submobject_mode = "lagged_start"),
            run_time = 2,
        )
        self.wait()
        self.play(FadeOut(self.rss))
        self.wait()

        two_equals_four = TexMobject("2", "=", "4")
        for tex, color in zip(two_equals_four, [GREEN, WHITE, RED]):
            tex.highlight(color)
        two_equals_four.scale(3)
        two_equals_four.to_edge(DOWN, buff = 1)
        sources = VGroup(*[
            self.equations[i][j][k].copy()
            for i, j, k in [(0, 1, 1), (1, 1, 0), (1, 1, 1)]
        ])
        for source, target in zip(sources, two_equals_four):
            self.play(Transform(source, target))
        self.wait()

        fake_qed = FakeQEDSymbol(order = 2, jagged_percentage = 0.3)
        fake_qed.next_to(two_equals_four, RIGHT, aligned_edge = DOWN, buff = 1)
        self.play(FadeIn(fake_qed))
        self.wait()

        issue = TextMobject("思考：\\\\问题在哪？")
        issue.highlight(YELLOW)
        issue.to_corner(RIGHT+DOWN)
        self.play(Write(issue), run_time = 1)
        self.wait(2)


class TheEnd(Scene):
    def construct(self):
        author = TextMobject("@Solara570")
        support = TextMobject("(Powered by @3Blue1Brown)")
        author.scale(1.8)
        support.match_width(author)
        group = VGroup(author, support)
        group.arrange_submobjects(DOWN)
        group.to_corner(RIGHT+DOWN)
        self.play(FadeIn(group))
        self.wait(2)


#####
## Thumbnail

class Thumbnail(Scene):
    def construct(self):
        exp_tower = ExpTower(element = "x", order = 10)
        exp_tower.scale_to_fit_height(6)
        exp_tower.gradient_highlight(YELLOW, BLUE)
        two, equal_sign, four = equation = TexMobject("2", "=", "4")
        two.highlight(GREEN)
        four.highlight(RED)
        equation.scale(10)
        question_mark = TexMobject("?")
        question_mark.scale_to_fit_height(2)
        question_mark.next_to(equal_sign, UP, buff = 0.5)

        notations = VGroup(*[
            TexMobject("{}^{\\infty} x"),
            TexMobject("x \\uparrow \\uparrow \\infty"),
        ])
        for notation, num, direction, angle, color in \
        zip(notations, [two, four], [UP+RIGHT, DOWN+LEFT], [-TAU/15, TAU/24], [YELLOW, BLUE]):
            notation.scale(3)
            notation.rotate(angle)
            notation.next_to(num, direction)
            notation.highlight(color)

        self.add(exp_tower, notations)
        self.add(FullScreenFadeRectangle())
        self.add(equation, question_mark)






