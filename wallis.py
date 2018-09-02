#coding=utf-8

import random
import numpy as np
from scipy.special import comb

from constants import *
from utils.config_ops import *
from utils.rate_functions import *
from utils.space_ops import *

from animation.animation import Animation
from animation.composition import AnimationGroup, Succession
from animation.creation import ShowCreation, Write, FadeIn, FadeOut, DrawBorderThenFill, GrowFromCenter
from animation.transform import Transform, ReplacementTransform, ApplyMethod, MoveToTarget
from mobject.svg.svg_mobject import SVGMobject
from mobject.svg.brace import Brace
from mobject.svg.tex_mobject import TexMobject, TextMobject
from mobject.types.vectorized_mobject import VMobject, VGroup, VectorizedPoint
from mobject.geometry import Dot, Circle, Arrow, Line, DashedLine, Sector
from mobject.frame import FullScreenFadeRectangle
from mobject.shape_matchers import BackgroundRectangle, SurroundingRectangle
from mobject.number_line import NumberLine
from mobject.coordinate_systems import Axes
from scene.scene import Scene

from pi_day_2018 import WallisRectangles

# self.skip_animations
# self.force_skipping()
# self.revert_to_original_skipping_status()


#####
## Functions

def snap_head_and_tail(func, head = 0.1, tail = 0.9):
    return squish_rate_func(func, head, tail)

def central_binomial_coeff(n):
    return int(comb(2*n, n, exact = True))

def wallis_numer(n):
    return (n + 1) if n % 2 == 1 else n

def wallis_denom(n):
    return n if n % 2 == 1 else (n + 1)

def a(n):
    assert n >= 0
    return 1. if n == 0 else a(n-1) * float(2*n-1)/(2*n)

def s(n):
    return 2 * n * a(n)

def random_walk_string(n, char_pool = "UD"):
    walk_string = ""
    for k in range(n):
        random_char = random.choice(char_pool)
        walk_string += random_char
    return walk_string


#####
## Mobjects

class RandomWalk1D(VMobject):
    CONFIG = {
        "up_char" : "U",
        "down_char" : "D",
        "step_size" : 1,
        "up_color" : RED,
        "down_color" : RED,
    }
    def __init__(self, walk_string, **kwargs):
        digest_config(self, kwargs, locals())
        VMobject.__init__(self, **kwargs)
        self.check_string(walk_string)

    def check_string(self, walk_string):
        for char in walk_string:
            if char not in (self.up_char, self.down_char):
                raise Exception("Invalid string!")

    def generate_points(self):
        raise Exception("Not Implemented")

    def generate_vertices_from_string(self, walk_string):
        vertices = [ORIGIN]
        for char in walk_string:
            last_point = vertices[-1]
            next_point = last_point + self.get_direction_from_char(char) * self.step_size
            vertices.append(next_point)
        return vertices

    def get_direction_from_char(self, char):
        return (RIGHT+UP) if char == self.up_char else (RIGHT+DOWN)


class RandomWalk1DLineAndDot(RandomWalk1D):
    CONFIG = {
        "dot_color" : WHITE,
    }
    def generate_points(self):
        self.line_group = VGroup()
        self.dot_group = VGroup()
        vertices = self.generate_vertices_from_string(self.walk_string)
        for k in range(len(vertices) - 1):
            line = Line(
                vertices[k], vertices[k+1],
                color = self.get_mob_color_by_number(k)
            )
            dot = Dot(vertices[k], color = self.dot_color)
            self.line_group.add(line)
            self.dot_group.add(dot)
        self.dot_group.add(Dot(vertices[-1], color = self.dot_color))
        self.add(self.line_group, self.dot_group)
        self.horizontally_center()

    def get_line_by_number(self, n):
        return self.line_group[n]

    def get_dots_by_number(self, n):
        return VGroup(self.dot_group[n:n+2])

    def get_mob_color_by_number(self, n):
        return self.up_color if self.walk_string[n] == self.up_char else self.down_color

    def split_at(self, n):
        extra_dot = self.get_dots_by_number(n)[0].copy()
        # ?????


class RandomWalk1DArrow(RandomWalk1D):
    CONFIG = {
        "up_color" : BLUE,
        "down_color" : RED,
    }
    def generate_points(self):
        self.arrow_group = VGroup()
        vertices = self.generate_vertices_from_string(self.walk_string)
        for k in range(len(vertices) - 1):
            arrow = Arrow(
                vertices[k], vertices[k+1],
                color = self.get_arrow_color_by_number(k), buff = 0
            )    
            self.arrow_group.add(arrow)
        self.add(self.arrow_group)
        self.horizontally_center()

    def get_length(self):
        return len(self.arrow_group)

    def get_arrow_by_number(self, n):
        return self.arrow_group[n]

    def get_arrow_color_by_number(self, n):
        return self.up_color if self.walk_string[n] == self.up_char else self.down_color

    def split_at(self, n):
        if n < 0:
            return VGroup(), self.arrow_group
        else:
            return VGroup(self.arrow_group[:n+1]), VGroup(self.arrow_group[n+1:])

    def get_arrow_starting_point(self, n):
        arrow = self.get_arrow_by_number(n)
        return arrow.get_start()

    def get_arrow_end_point(self, n):
        arrow = self.get_arrow_by_number(n)
        return arrow.get_end()

    def get_starting_point(self):
        return self.get_arrow_starting_point(0)

    def get_end_point(self):
        return self.get_arrow_end_point(-1)

    def move_start_to(self, position):
        self.shift(position - self.get_starting_point())
        return self

    def get_flip_arrows_animation(self, n, color = None):
        arrows = [
            self.get_arrow_by_number(k)
            for k in range(n+1)
        ]
        for arrow in arrows:
            arrow.generate_target()
            arrow.target.rotate(np.pi)
            if color is not None:
                arrow.target.set_color(color)
        return AnimationGroup(*[
            MoveToTarget(arrow)
            for arrow in arrows
        ])


class House(VMobject):
    CONFIG = {
        "color" : DARK_GREY,
        "height" : 3,
    }
    def generate_points(self):
        self.house = SVGMobject(
            file_name = "house", color = self.color, height = self.height
        )
        self.point = VectorizedPoint(self.house.get_bottom())
        self.add(self.house, self.point)

    def get_position(self):
        return self.point.get_center()

    def place_on(self, position):
        self.shift(position - self.get_position())
        return self


class Drunk(VMobject):
    CONFIG = {
        "direction" : RIGHT,
        "color" : YELLOW_E,
        "height" : 1,
    }
    def generate_points(self):
        self.drunk = SVGMobject(
            file_name = "drunk", color = self.color, height = self.height
        )
        self.point = VectorizedPoint(self.drunk.get_bottom())
        self.add(self.drunk, self.point)
        self.rotate(angle_between(RIGHT, self.direction))

    def get_position(self):
        return self.point.get_center()

    def step_on(self, position):
        self.shift(position - self.get_position())
        return self

    def turn_around(self, **kwargs):
        axis = rotate_vector(self.direction, np.pi/2)
        self.rotate(np.pi, axis = axis, **kwargs)
        self.change_direction()
        return self

    def get_direction(self):
        return self.direction

    def change_direction(self):
        self.direction = -self.direction


#####
## Animations

class DrunkMoveToPosition(AnimationGroup):
    def __init__(self, drunk, pos, total_time = 1, turn_ratio = 0.25):
        anims = []
        if self.is_same_direction(drunk, pos):
            anims.append(ApplyMethod(drunk.step_on, pos, run_time = total_time))
        else:
            turn_target = drunk.deepcopy().turn_around()
            move_target = turn_target.deepcopy().step_on(pos)
            ## TODO: This part is currently broken!
            anims.append(
                Succession(
                    ApplyMethod(drunk.turn_around, run_time = turn_ratio * total_time),
                    ApplyMethod(turn_target.step_on, pos, run_time = (1-turn_ratio) * total_time),
                    FadeOut(turn_target, run_time = 0),
                    Transform(drunk, move_target, run_time = 0),
            ))
            drunk.change_direction()
        AnimationGroup.__init__(self, *anims)

    def is_same_direction(self, drunk, pos):
        vec = pos - drunk.get_position()
        if np.dot(vec, drunk.get_direction()) < 0:
            return False
        return True


class DrunkWander(DrunkMoveToPosition):
    def __init__(self, drunk, number, direction = RIGHT, total_time = 1, turn_ratio = 0.25):
        pos = drunk.get_position() + number * direction
        DrunkMoveToPosition.__init__(self, drunk, pos, total_time, turn_ratio)


#####
## Main Scenes

class DrunkWanderIntro(Scene):
    def construct(self):
        # Setup
        line = NumberLine()
        house = House()
        drunk = Drunk(direction = RIGHT)
        house.place_on(ORIGIN)
        drunk.step_on(ORIGIN)
        t_equals = TexMobject("t = ")
        time = TexMobject("0")
        time.next_to(t_equals, RIGHT, buff = 0.15)
        VGroup(t_equals, time).next_to(line, DOWN, buff = 0.5)
        old_drunk = drunk.copy()
        old_time = time.copy()
        self.add(house, line, drunk, t_equals)

        # Start wandering
        for k in range(20):
            new_time = TexMobject("%s" % str(k+1))
            new_time.next_to(t_equals, RIGHT, buff = 0.15)
            self.play(
                DrunkWander(drunk, random.choice([-1, 1]), total_time = 0.5),
                Transform(time, new_time, rate_func = snap_head_and_tail(smooth), run_time = 0.5),
            )
        self.wait()

        # Reset
        self.play(Transform(time, old_time), Transform(drunk, old_drunk))
        self.wait()


class PositionAndTimeAxes(Scene):
    def construct(self):
        # Setup
        pos_axis = NumberLine(x_min = -3.5, x_max = 3.5, color = WHITE)
        pos_axis.rotate(np.pi/2)
        pos_axis.add_tip()
        time_axis = NumberLine(x_min = 0, x_max = 9.5, color = WHITE)
        time_axis.add_tip()
        house = House()
        house.rotate(np.pi/2)
        house.place_on(pos_axis.number_to_point(0))
        drunk = Drunk(direction = UP)
        drunk.step_on(pos_axis.number_to_point(0))
        group = VGroup(house, VGroup(pos_axis, time_axis), drunk)
        group.to_edge(LEFT)
        pos_text = TextMobject("位置")
        pos_text.next_to(pos_axis.number_to_point(3.5), LEFT)
        time_text = TextMobject("时间")
        time_text.next_to(time_axis.number_to_point(9.5), DOWN)
        self.add(group, pos_text, time_text)
        old_drunk = drunk.copy()

        # Start Wandering
        sequence = "UDUDDDUD"
        random_walk = RandomWalk1DArrow(sequence)
        random_walk.move_start_to(pos_axis.number_to_point(0))
        for k, walk_char in enumerate(sequence):
            increment = 1 if walk_char == "U" else -1
            arrow = random_walk.get_arrow_by_number(k)
            self.play(
                DrunkWander(drunk, increment, direction = UP),
                ShowCreation(arrow),
            )
        self.wait()

        # Reset
        self.play(Transform(drunk, old_drunk), FadeOut(random_walk))
        self.wait()


class TwoTypicalPaths(Scene):
    CONFIG = {
        "zero_color" : ORANGE,
        "neg_color" : GREEN,
    }
    def construct(self):
        # Setup
        pos_axis = NumberLine(x_min = -3.5, x_max = 3.5, color = WHITE)
        pos_axis.rotate(np.pi/2)
        pos_axis.add_tip()
        time_axis = NumberLine(x_min = 0, x_max = 9.5, color = WHITE)
        time_axis.add_tip()
        house = House()
        house.rotate(np.pi/2)
        house.place_on(pos_axis.number_to_point(0))
        zero_drunk = Drunk(direction = UP, color = self.zero_color)
        zero_drunk.step_on(pos_axis.number_to_point(0))
        neg_drunk = Drunk(direction = UP, color = self.neg_color)
        neg_drunk.step_on(pos_axis.number_to_point(0))
        group = VGroup(house, VGroup(pos_axis, time_axis), zero_drunk, neg_drunk)
        group.to_edge(LEFT)
        pos_text = TextMobject("位置")
        pos_text.next_to(pos_axis.number_to_point(3.5), LEFT)
        time_text = TextMobject("时间")
        time_text.next_to(time_axis.number_to_point(9.5), DOWN)
        self.add(group, pos_text, time_text)
        old_zero_drunk = zero_drunk.copy()
        old_neg_drunk = neg_drunk.copy()

        # Start Wandering
        zero_sequence = "UUUDDDDU"
        zero_walk = RandomWalk1DArrow(
            zero_sequence, up_color = self.zero_color, down_color = self.zero_color,
        )
        zero_walk.move_start_to(pos_axis.number_to_point(0))
        neg_sequence = "DDUDDUDU"
        neg_walk = RandomWalk1DArrow(
            neg_sequence, up_color = self.neg_color, down_color = self.neg_color,
        )
        neg_walk.move_start_to(pos_axis.number_to_point(0))
        for k, (zero_char, neg_char) in enumerate(zip(zero_sequence, neg_sequence)):
            zero_increment = 1 if zero_char == "U" else -1
            neg_increment = 1 if neg_char == "U" else -1
            zero_arrow = zero_walk.get_arrow_by_number(k)
            neg_arrow = neg_walk.get_arrow_by_number(k)
            self.play(
                DrunkWander(zero_drunk, zero_increment, direction = UP),
                DrunkWander(neg_drunk, neg_increment, direction = UP),
                ShowCreation(zero_arrow),
                ShowCreation(neg_arrow)
            )
        self.wait()

        # Reset
        self.play(
            Transform(zero_drunk, old_zero_drunk),
            Transform(neg_drunk, old_neg_drunk),
            FadeOut(zero_walk),
            FadeOut(neg_walk),
        )
        self.wait()


class SplitPathIntoTwo(Scene):
    def construct(self):
        # Setup
        pos_axis = NumberLine(x_min = -3.5, x_max = 3.5, color = WHITE)
        pos_axis.rotate(np.pi/2)
        pos_axis.add_tip()
        time_axis = NumberLine(x_min = 0, x_max = 9.5, color = WHITE)
        time_axis.add_tip()
        pos_text = TextMobject("位置")
        pos_text.next_to(pos_axis.number_to_point(3.5), LEFT)
        time_text = TextMobject("时间")
        time_text.next_to(time_axis.number_to_point(9.5), DOWN)
        group = VGroup(pos_axis, time_axis, pos_text, time_text)
        group.to_edge(LEFT, buff = 2)
        self.add(group)

        # Some preparations
        sequences = ["UDDDUDUD", "DUUDDDUD", "DDUUUDUD", "UUUDDUDU", "UDDDUDUD"]
        nums = [1, 3, 7, -1, 1]
        walks = [
            RandomWalk1DArrow(sequence).move_start_to(pos_axis.number_to_point(0))
            for sequence in sequences
        ]
        parts = [
            walk.split_at(num)
            for walk, num in zip(walks, nums)
        ]
        split_lines = [
            DashedLine(TOP, BOTTOM, color = GREY) \
            .move_to(time_axis.number_to_point(num+1))
            for num in nums
        ]
        zero_words = [
            TextMobject("“正点到家” \\\\ $P_%s$" % str((num+1)//2)) \
            .next_to(line, LEFT).shift(2.8 * DOWN).set_color(ORANGE)
            for num, line in zip(nums, split_lines)
        ]
        nocross_words = [
            TextMobject("“不经过家门” \\\\ $Q_%s$" % str(4-(num+1)//2)) \
            .next_to(line, RIGHT).shift(2.8 * DOWN).set_color(GREEN)
            for num, line in zip(nums, split_lines)
        ]
        for words, direction in zip([zero_words, nocross_words], [RIGHT, LEFT]):
            for word in words:
                text, symbol = VGroup(word[:-2]), VGroup(word[-2:])
                symbol.scale(1.5)
                symbol.next_to(text, DOWN, aligned_edge = direction)
                word.add_background_rectangle()

        # Demonstrate how to split
        self.add(walks[0], split_lines[0], zero_words[0], nocross_words[0])
        l = len(walks)
        for k in range(l-1):
            cur_walk = walks[k]
            cur_line = split_lines[k]
            cur_zero_word = zero_words[k]
            cur_nocross_word = nocross_words[k]
            next_walk = walks[k+1]
            next_line = split_lines[k+1]
            next_zero_word = zero_words[k+1]
            next_nocross_word = nocross_words[k+1]
            part1, part2 = parts[k]
            cur_walk.save_state()
            self.play(
                part1.set_color, ORANGE,
                part2.set_color, GREEN,
            )
            self.wait()
            self.play(cur_walk.restore)
            self.wait()
            self.play(
                ReplacementTransform(cur_walk, next_walk),
                ReplacementTransform(cur_line, next_line),
                ReplacementTransform(cur_zero_word, next_zero_word),
                ReplacementTransform(cur_nocross_word, next_nocross_word),
            )
            self.wait()


class ZeroAndNonCrossingComparison(Scene):
    def construct(self):
        # Chart on the left
        colors = [WHITE, ORANGE, GREEN]
        titles = VGroup(*[
            TexMobject(text).set_color(color)
            for text, color in zip(["n", "p_n", "q_n"], colors)
        ])
        contents = VGroup(*[
            VGroup(*[
                TexMobject("%d" % num)
                for num in [k, central_binomial_coeff(k), central_binomial_coeff(k)]
            ])
            for k in range(8)
        ])
        titles.arrange_submobjects(RIGHT, buff = 1)
        for num, line in enumerate(contents):
            for k, element in enumerate(line):
                buff = 0.6 + 0.8 * num
                element.next_to(titles[k], DOWN, aligned_edge = LEFT, buff = buff)
                element.set_color(colors[k])
        sep_line = Line(ORIGIN, 4.5*RIGHT, stroke_width = 5)
        sep_line.next_to(titles, DOWN)
        chart = VGroup(titles, contents, sep_line)
        chart.set_height(7)
        chart.center().to_edge(LEFT)
        self.add(chart)

        # Figures on the right
        std_zero_pos_axis = NumberLine(
            x_min = -2, x_max = 2, color = GREY, unit_size = 0.25, tick_size = 0.05
        )
        std_zero_pos_axis.rotate(np.pi/2)
        std_nocross_pos_axis = NumberLine(
            x_min = -4, x_max = 4, color = GREY, unit_size = 0.25, tick_size = 0.05
        )
        std_nocross_pos_axis.rotate(np.pi/2)
        std_time_axis = NumberLine(
            x_min = 0, x_max = 5.5, color = GREY, unit_size = 0.25, tick_size = 0.05
        )
        std_zero_axes = VGroup(std_zero_pos_axis, std_time_axis)
        std_nocross_axes = VGroup(std_nocross_pos_axis, std_time_axis)

        zero_walks = VGroup()
        for sequence in ["UUDD", "UDUD", "UDDU", "DDUU", "DUDU", "DUUD"]:
            axes = std_zero_axes.copy()
            zero_walk = RandomWalk1DArrow(sequence, step_size = 0.25)
            zero_walk.move_start_to(axes[0].number_to_point(0))
            zero_walks.add(VGroup(axes, zero_walk))
        zero_walks.arrange_submobjects_in_grid(2, 3, buff = 0.5)
        zero_rect = SurroundingRectangle(zero_walks, color = ORANGE, buff = 0.4)
        zero_walks.add(zero_rect)

        nocross_walks = VGroup()
        for sequence in ["DDDD", "DDUD", "DDDU", "UUUU", "UUDU", "UUUD"]:
            axes = std_nocross_axes.copy()
            nocross_walk = RandomWalk1DArrow(sequence, step_size = 0.25)
            nocross_walk.move_start_to(axes[0].number_to_point(0))
            nocross_walks.add(VGroup(axes, nocross_walk))
        nocross_walks.arrange_submobjects_in_grid(2, 3, buff = 0.5)
        nocross_rect = SurroundingRectangle(nocross_walks, color = GREEN, buff = 0.4)
        nocross_walks.add(nocross_rect)

        relation = TexMobject("p_2", "=", "q_2", "=", "6")
        relation[0].set_color(ORANGE)
        relation[2].set_color(GREEN)
        relation.scale(1.5)
        figure = VGroup(zero_walks, relation, nocross_walks)
        figure.arrange_submobjects(DOWN)
        figure.set_height(7)
        figure.center().to_edge(RIGHT)

        self.add(figure)
        self.wait()


class BijectionScene(Scene):
    def setup(self):
        pos_axis = NumberLine(x_min = -4.5, x_max = 2.5, color = WHITE)
        pos_axis.rotate(np.pi/2)
        pos_axis.add_tip()
        time_axis = NumberLine(x_min = 0, x_max = 9.5, color = WHITE)
        time_axis.add_tip()
        vec = pos_axis.number_to_point(0) - time_axis.number_to_point(0)
        time_axis.shift(vec)
        pos_text = TextMobject("位置")
        pos_text.next_to(pos_axis.number_to_point(2.5), LEFT)
        time_text = TextMobject("时间")
        time_text.next_to(time_axis.number_to_point(9.5), DOWN)
        axes_group = VGroup(pos_axis, time_axis, pos_text, time_text)
        axes_group.center()

        title_pq = TexMobject("P_4", "570", "Q_4")
        title_pq.scale(1.5)
        title_pq.to_corner(UP+RIGHT)
        for part, color in zip(title_pq, [ORANGE, BLACK, GREEN]):
            part.set_color(color)
        r_arrow = TexMobject("\\rightarrow")
        l_arrow = TexMobject("\\leftarrow")
        for arrow in (r_arrow, l_arrow):
            arrow.scale(1.5)
            arrow.move_to(title_pq[1])
    
        sequence_p, sequence_q = sequences = ["UDUUDUDD", "DDUDDUDD"]
        colors = [ORANGE, GREEN]
        walk_p, walk_q = walks = [
            RandomWalk1DArrow(sequence, up_color = color, down_color = color) \
            .move_start_to(pos_axis.number_to_point(0))
            for sequence, color in zip(sequences, colors)
        ]
        parts_p, parts_q = [
            walk.split_at(3)
            for walk in walks
        ]
        self.axes_group = axes_group
        self.title_pq = title_pq
        self.walk_p = walk_p
        self.walk_q = walk_q
        self.parts_p = parts_p
        self.parts_q = parts_q
        self.r_arrow = r_arrow
        self.l_arrow = l_arrow


class BijectionRulePQ(BijectionScene):
    def construct(self):
        # Setup
        axes_group = self.axes_group
        title_pq = self.title_pq
        walk_p = self.walk_p
        parts_p = self.parts_p
        r_arrow = self.r_arrow
        self.add(axes_group, title_pq, walk_p, r_arrow)
        walk_p_copy = walk_p.copy()

        # P_4 -> Q_4
        steps_pq = VGroup(*[
            TextMobject(text)
            for text in [
                "1. 第一步是沿着正方向走的", "2. 找到第一次到达最大值的时刻",
                "3. 在这个时刻上进行分割", "4. 将第一段水平翻转",
                "5. 拼接两个片段"
            ]
        ])
        for step in steps_pq:
            step.set_color(YELLOW)
            step.add_background_rectangle()
        step1_pq, step2_pq, step3_pq, step4_pq, step5_pq = steps_pq
        
        # 1. Check the first step of the walk
        step1_circle = Circle(color = YELLOW)
        first_arrow = walk_p.get_arrow_by_number(0)
        step1_circle.surround(first_arrow)
        step1_pq.next_to(step1_circle, RIGHT+DOWN)
        self.play(
            ShowCreation(step1_circle),
            Write(step1_pq),
            run_time = 1
        )
        self.wait(1.5)
        self.play(FadeOut(step1_circle), FadeOut(step1_pq))

        # 2. Find the first time it reaches the maximum
        peak = walk_p.get_arrow_end_point(3)
        horiz_line = DashedLine(2.5*LEFT, 2.5*RIGHT, color = YELLOW)
        horiz_line.move_to(peak)
        dot = Dot(color = YELLOW)
        dot.move_to(peak)
        step2_pq.next_to(horiz_line, UP)
        self.play(
            ShowCreation(horiz_line),
            DrawBorderThenFill(dot),
            Write(step2_pq),
            run_time = 1
        )
        self.wait(1.5)
        self.play(FadeOut(horiz_line), FadeOut(step2_pq))

        # 3. Split
        vert_line = DashedLine(2.5*UP, 2.5*DOWN, color = YELLOW)
        vert_line.move_to(peak)
        step3_pq.next_to(vert_line, DOWN)
        left_part_p, right_part_p = parts_p
        self.play(
            ShowCreation(vert_line),
            Write(step3_pq),
            run_time = 1
        )
        self.play(
            FadeOut(dot),
            left_part_p.shift, 0.5*DOWN+0.5*LEFT,
            right_part_p.shift, DOWN+0.5*RIGHT,
        )
        self.wait(1.5)
        self.play(FadeOut(vert_line), FadeOut(step3_pq))

        # 4. Flip the first segment horizontally
        flip_axis = DashedLine(2*UP, 2*DOWN, color = GREY)
        flip_axis.move_to(left_part_p)
        step4_pq.next_to(flip_axis, DOWN)
        self.play(
            ShowCreation(flip_axis),
            Write(step4_pq),
            run_time = 1,
        )
        self.play(
            left_part_p.flip,
            Animation(flip_axis),    
        )
        self.wait(1.5)
        self.play(FadeOut(step4_pq), FadeOut(flip_axis))

        # 5. Put the pieces together
        step5_pq.move_to(dot)
        flip_arrow_anims = walk_p.get_flip_arrows_animation(3, color = GREEN)
        self.play(Write(step5_pq), run_time = 1)
        self.wait(0.5)
        self.play(
            flip_arrow_anims,
            right_part_p.set_color, GREEN)
        self.wait(0.5)
        self.play(
            left_part_p.shift, 1.5*DOWN+0.5*RIGHT,
            right_part_p.shift, 3*DOWN+0.5*LEFT,
            Animation(step5_pq),
        )
        self.wait(0.5)
        self.play(FadeOut(step5_pq))
        self.wait(1.5)

        # Now Reset
        self.play(FadeOut(walk_p))
        self.play(FadeIn(walk_p_copy))
        self.wait()


class BijectionRuleQP(BijectionScene):
    def construct(self):
        # Setup
        axes_group = self.axes_group
        pos_axis, time_axis = axes_group[:2]
        title_pq = self.title_pq
        walk_q = self.walk_q
        parts_q = self.parts_q
        l_arrow = self.l_arrow
        self.add(axes_group, title_pq, walk_q, l_arrow)
        walk_q_copy = walk_q.copy()

        # Q_4 -> P_4
        steps_qp = VGroup(*[
            TextMobject(text)
            for text in [
                "1. 找到路径终点的位置坐标$h$", "2. 找到最晚一次穿过$\\frac{h}{2}$的时刻",
                "3. 在这个时刻上进行分割", "4. 将第一段水平翻转",
                "5. 拼接两个片段"
            ]
        ])
        for step in steps_qp:
            step.set_color(YELLOW)
            step.add_background_rectangle()
        step1_qp, step2_qp, step3_qp, step4_qp, step5_qp = steps_qp

        # 1. Find the endpoint
        step1_qp.next_to(time_axis.number_to_point(4.5), UP)
        end_horiz_line = DashedLine(LEFT_SIDE, RIGHT_SIDE, color = YELLOW)
        end_horiz_line.move_to(pos_axis.number_to_point(-4))
        end_horiz_line.horizontally_center()
        end_brace_line = DashedLine(time_axis.number_to_point(8), walk_q.get_end_point())
        end_brace = Brace(end_brace_line, direction = RIGHT, color = YELLOW)
        h = TexMobject("h").set_color(YELLOW)
        end_brace.put_at_tip(h)
        self.play(
            Write(step1_qp),
            ShowCreation(end_horiz_line),
            run_time = 1
        )
        self.play(
            GrowFromCenter(end_brace),
            GrowFromCenter(h)
        )
        self.wait(1.5)
        self.play(FadeOut(step1_qp))

        # 2. Find the last time it GOES THROUGH half its final value
        half_point = walk_q.get_arrow_end_point(3)
        step2_qp.next_to(time_axis.number_to_point(4.5), UP)
        half_horiz_line = end_horiz_line.copy().shift(2*UP)
        half_brace_line = DashedLine(time_axis.number_to_point(4), half_point)
        half_brace = Brace(half_brace_line, direction = RIGHT, color = YELLOW)
        half_h = TexMobject("\\frac{h}{2}").set_color(YELLOW)
        half_brace.put_at_tip(half_h)
        half_dot = Dot(half_point, color = YELLOW)
        self.play(FadeIn(step2_qp), run_time = 1)
        self.wait(0.5)
        self.play(
            ReplacementTransform(end_brace, half_brace),
            ReplacementTransform(end_horiz_line, half_horiz_line),
            ReplacementTransform(h, half_h[0]),
            Write(half_h[1:]),
        )
        self.play(DrawBorderThenFill(half_dot))
        self.wait(1.5)
        self.play(FadeOut(VGroup(step2_qp, half_horiz_line, half_brace, half_h)))

        # 3. Split 
        vert_line = DashedLine(2.5*UP, 2.5*DOWN, color = YELLOW)
        vert_line.move_to(half_point)
        step3_qp.next_to(vert_line, UP)
        left_part_q, right_part_q = parts_q
        self.play(
            ShowCreation(vert_line),
            Write(step3_qp),
            run_time = 1
        )
        self.play(
            FadeOut(half_dot),
            left_part_q.shift, 0.5*DOWN+0.5*LEFT,
            right_part_q.shift, 0.5*UP+0.5*RIGHT,
        )
        self.wait(1.5)
        self.play(FadeOut(vert_line), FadeOut(step3_qp))

        # 4. Flip the first segment horizontally
        flip_axis = DashedLine(2*UP, 2*DOWN, color = GREY)
        flip_axis.move_to(left_part_q)
        step4_qp.next_to(flip_axis, DOWN)
        self.play(
            ShowCreation(flip_axis),
            Write(step4_qp),
            run_time = 1,
        )
        self.play(
            left_part_q.flip,
            Animation(flip_axis),    
        )
        self.wait(1.5)
        self.play(FadeOut(step4_qp), FadeOut(flip_axis))

        # 5. Put the pieces together
        step5_qp.shift(2.5*DOWN)
        flip_arrow_anims = walk_q.get_flip_arrows_animation(3, color = ORANGE)
        self.play(Write(step5_qp), run_time = 1)
        self.wait(0.5)
        self.play(
            flip_arrow_anims,
            right_part_q.set_color, ORANGE)
        self.wait(0.5)
        self.play(
            left_part_q.shift, 2.5*UP+0.5*RIGHT,
            right_part_q.shift, 3.5*UP+0.5*LEFT,
            Animation(step5_qp),
        )
        self.wait(0.5)
        self.play(FadeOut(step5_qp))
        self.wait(1.5)

        # Now Reset
        self.play(FadeOut(walk_q))
        self.play(FadeIn(walk_q_copy))
        self.wait()


class SimpleCaseWithArea5(Scene):
    def construct(self):
        colors = ["#FF0000", "#FF8000", "#FFFF00", "#00FF00", "#0080FF"]
        wallis_rects_4 = WallisRectangles(
            order = 5,
            rect_colors = colors, 
        )
        vert_lines = VGroup(*[
            Line(3.5*UP, 3.5*DOWN, color = GREY, stroke_width = 3) \
            .next_to(wallis_rects_4.get_rectangle(0, k), direction, buff = 0)
            for k, direction in zip(list(range(5))+[4], [LEFT]*5+[RIGHT])
        ])
        horiz_lines = VGroup(*[
            Line(3.5*LEFT, 3.5*RIGHT, color = GREY, stroke_width = 3) \
            .next_to(wallis_rects_4.get_rectangle(k, 0), direction, buff = 0)
            for k, direction in zip(list(range(5))+[4], [DOWN]*5+[UP])
        ])
        for vert_line in vert_lines:
            vert_line.vertically_center()
        for horiz_line in horiz_lines:
            horiz_line.horizontally_center()
        vert_labels = VGroup(*[
            TexMobject("a_%d" % k) \
            .move_to((vert_lines[k].get_center() + vert_lines[k+1].get_center())/2) \
            .shift(3.5*DOWN)
            for k in range(5)
        ])
        horiz_labels = VGroup(*[
            TexMobject("a_%d" % k) \
            .move_to((horiz_lines[k].get_center() + horiz_lines[k+1].get_center())/2) \
            .shift(3.5*LEFT)
            for k in range(5)
        ])

        area_texs = VGroup()
        factors = [1.25, 1, 0.9, 0.7, 0.6]
        for p in range(5):
            for q in range(5-p):
                rect = wallis_rects_4.get_rectangle(p, q)
                tex = TexMobject("{a_%d} {a_%d}" % (q, p))
                tex.scale(factors[p+q])
                tex.move_to(rect)
                area_texs.add(tex)

        figure = VGroup()
        figure.add(wallis_rects_4, vert_lines, horiz_lines, vert_labels, horiz_labels, area_texs)
        figure.to_edge(LEFT)
        self.add(figure)

        tex_list = VGroup()
        for p in range(5):
            formula_string = (" + ".join([
                "a_%d a_%d" % (q, p-q)
                for q in range(p+1)
            ]) + "=1")
            formula = TexMobject(formula_string)
            tex_list.add(formula)
        # tex_list.add(TexMobject("\\vdots"))
        tex_factors = np.linspace(1, 0.7, 5)
        for tex, color, factor in zip(tex_list, colors, tex_factors):
            tex.set_color(color)
            tex.scale(factor)
        tex_list.arrange_submobjects(DOWN, aligned_edge = LEFT)
        tex_list.to_edge(RIGHT)
        self.add(tex_list)
        self.wait()


class CompareQuarterCircleWithWR570(Scene):
    def construct(self):
        wallis_rects_570 = WallisRectangles(order = 570)
        quarter_circle = Sector(
            outer_radius = wallis_rects_570.get_height(),
            stroke_color = GREY, stroke_width = 5.70,
            fill_opacity = 0,
        )
        quarter_circle.move_arc_center_to(wallis_rects_570.get_bottom_left())
        text = TexMobject("n = 570")
        text.scale(1.5)
        text.move_to(wallis_rects_570.get_top_right())
        self.add(wallis_rects_570, quarter_circle, text)
        self.wait()


class BackToWR5Case(Scene):
    def construct(self):
        wallis_rects_5 = WallisRectangles(
            rect_colors = ["#FF0000", "#FF8000", "#FFFF00", "#00FF00", "#0080FF"],
            order = 5
        )
        quarter_circle = Sector(
            outer_radius = wallis_rects_5.get_height(),
            stroke_color = GREY, stroke_width = 5.70,
            fill_opacity = 0,
        )
        quarter_circle.move_arc_center_to(wallis_rects_5.get_bottom_left())
        text = TexMobject("n = 5")
        text.scale(1.5)
        text.move_to(wallis_rects_5.get_top_right())
        self.add(wallis_rects_5, quarter_circle, text)
        self.wait()


class TweakNumberOfLayers(Scene):
    def construct(self):
        orders = [5, 4, 5, 6, 5]
        colors = ["#FF0000", "#FF8000", "#FFFF00", "#00FF00", "#0080FF", "#FF00FF"]
        texts = VGroup(*[
            TexMobject("n = %d" % order)
            for order in orders
        ])
        wallis_rects = VGroup(*[
            WallisRectangles(
                rect_colors = colors[:][:order], order = order,
                height = s(order) / s(5) * 6.
            )
            for order in orders
        ])
        for wallis_rect in wallis_rects[1:]:
            vec = wallis_rects[0].get_bottom_left() - wallis_rect.get_bottom_left()
            wallis_rect.shift(vec)
        for text in texts:
            text.scale(1.5)
            text.move_to(wallis_rects[0].get_top_right())
        quarter_circle = Sector(
            outer_radius = wallis_rects[0].get_height(),
            stroke_color = GREY, stroke_width = 5.70,
            fill_opacity = 0,
        )
        quarter_circle.move_arc_center_to(wallis_rects[0].get_bottom_left())

        init_wallis_rect = wallis_rects[0]
        init_text = texts[0]
        self.add(init_wallis_rect, init_text, quarter_circle)
        layer_anims = [
            Action(wallis_rects[k].get_layer(num))
            for Action, k, num in zip([FadeOut, FadeIn, FadeIn, FadeOut], [0, 2, 3, 3], [4, 4, 5, 5])
        ]
        for layer_anim, new_text in zip(layer_anims, texts[1:]):
            self.play(
                layer_anim,
                Transform(init_text, new_text),
                Animation(quarter_circle)
            )
            self.wait()


class FormalizeTopRightCorner(Scene):
    def construct(self):
        axes = Axes(
            x_min = -0.5, x_max = 6.5, y_min = -0.5, y_max = 6.5,
            color = GREY, number_line_config = {"tick_size" : 0}
        )
        axes.center()
        wallis_rects_5 = WallisRectangles(
            rect_colors = ["#FF0000", "#FF8000", "#FFFF00", "#00FF00", "#0080FF"],
            order = 5
        )
        wallis_rects_5.move_corner_to(axes.coords_to_point(0, 0))
        quarter_circle = Sector(
            outer_radius = wallis_rects_5.get_height(),
            stroke_color = GREY, stroke_width = 5.70,
            fill_opacity = 0,
        )
        quarter_circle.move_arc_center_to(wallis_rects_5.get_bottom_left())
        self.add(wallis_rects_5, axes, quarter_circle)

        order = wallis_rects_5.get_order()
        inner_corners = [
            rect.get_critical_point(UP+RIGHT)
            for rect in wallis_rects_5.get_layer(3)
        ]
        outer_corners = [
            rect.get_critical_point(UP+RIGHT)
            for rect in wallis_rects_5.get_layer(4)
        ]
        inner_dots = VGroup(*[
            Dot(corner, color = PINK, stroke_width = 2, stroke_color = BLACK)
            for corner in inner_corners
        ])
        outer_dots = VGroup(*[
            Dot(corner, stroke_width = 2, stroke_color = BLACK)
            for corner in outer_corners
        ])
        inner_texts = VGroup(*[
            TexMobject("(s_%d, s_%d)" % (order - k, k)) \
            .scale(0.8) \
            .set_color(PINK) \
            .next_to(dot, LEFT+DOWN, buff = 0.05)
            for k, dot in zip(range(1, len(inner_dots) + 1), inner_dots)
        ])
        outer_texts = VGroup(*[
            TexMobject("(s_%d, s_%d)" % (order + 1 - k, k)) \
            .scale(0.8) \
            .set_color(WHITE) \
            .next_to(dot, RIGHT+UP, buff = 0.05)
            for k, dot in zip(range(1, len(outer_dots) + 1), outer_dots)
        ])
        self.add(inner_dots, outer_dots, inner_texts, outer_texts)
        self.wait()


#####
## Test Scenes

class TestDrunk(Scene):
    CONFIG = {
        "random_seed" : 570,
    }
    def construct(self):
        line = NumberLine()
        self.add(line)
        drunk = Drunk(direction = UP)
        drunk.step_on(ORIGIN)
        self.play(DrawBorderThenFill(drunk))
        self.wait()
        for k in range(10):
            self.play(DrunkWander(drunk, random.choice([-1, 1])))
        self.wait()


class TestRandomWalk(Scene):
    def construct(self):
        # walk_strings = ["UUUDDD", "UUUUUU", "UUDDDD", "UUDDDU"]
        walk_strings = [random_walk_string(10) for k in range(10)]
        # random_walks = [RandomWalk1DLineAndDot(ws, step_size = 0.5) for ws in walk_strings]
        random_walks = [RandomWalk1DArrow(ws, step_size = 0.5) for ws in walk_strings]
        self.add(random_walks[0])
        for walk in random_walks[1:]:
            self.play(Transform(random_walks[0], walk), run_time = 0.5)


class BannerP1(Scene):
    CONFIG = {
        "part"    : "上篇",
        "n_terms" : 7,
        "n_walks" : 20,
        "random_seed" : hash("Solara570") % 2**32-1,
    }
    def construct(self):
        formula = TexMobject(*(
            [
                "{%d \\over %d} \\cdot" % (wallis_numer(n), wallis_denom(n))
                for n in range(1, self.n_terms + 1)
            ] + ["\\cdots"]
        ))
        result = TexMobject("=", "{\\pi \\over 2}")
        result.scale(2)
        pi = result[-1][0]
        pi.set_color(YELLOW)
        circle = Circle(color = YELLOW)
        circle.surround(pi)
        question = TexMobject("?", color = YELLOW)
        question.scale(2).next_to(circle, RIGHT, buff = 0.4)
        result_group = VGroup(result, circle, question)
        result_group.next_to(formula, DOWN)
        group = VGroup(formula, result_group)
        group.center().set_width(10)
        bg_rect = BackgroundRectangle(group, fill_opacity = 0.5, buff = 0.2)

        random_walks = VGroup(*[
            RandomWalk1DArrow(random_walk_string(30), step_size = 0.5)
            for k in range(self.n_walks)
        ])
        for k, walk in enumerate(random_walks):
            walk.shift(random.randrange(-5, 5) * 2 * walk.step_size * UP)
        random_walks.center()

        text = TextMobject(self.part).scale(3).set_color(YELLOW)
        text.to_corner(RIGHT+DOWN)

        self.add(random_walks)
        self.add(FullScreenFadeRectangle())
        self.add(bg_rect, group, text)


class BannerP2(BannerP1):
    CONFIG = {
        "part" : "下篇",
    }


