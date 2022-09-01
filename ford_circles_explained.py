# coding=utf-8

import math
from manimlib import *
from typing import Iterable
from custom.custom_mobjects import PauseButton


# Constants
X_COOR_MASK = [1, 0, 0]
Y_COOR_MASK = [0, 1, 0]
Z_COOR_MASK = [0, 0, 1]


# Functions and usual objects
def get_quotient_and_remainder(dividend, divisor):
    return dividend // divisor, dividend % divisor


def is_coprime(p, q):
    return math.gcd(p, q) == 1


def get_bezout_coefficients(p, q):
    if p < q:
        p, q = q, p
    if p % q == 0:
        return (0, 1)
    else:
        N, r = p // q, p % q
        aN_plus_b, a = get_bezout_coefficients(q, r)
        return a, aN_plus_b - a * N


def get_bezout_coefficients_strings(p, q):
    return tuple(
        map(
            lambda x: f'{x}' if x >= 0 else f'({x})',
            get_bezout_coefficients(p, q)
        )
    )


def get_coprime_numers_by_denom(q):
    return [0, 1] if q == 1 else [p for p in range(1, q) if is_coprime(p, q)]


def get_stroke_width_by_height(height, thres=1):
    return 1.5 if height > thres else height


def get_fraction_opacity_by_height(height, thres=0.5):
    return 1 if height > thres else height / thres


class FareyRational(object):
    def __init__(self, p, q, level=0):
        self.p = p
        self.q = q
        self.level = level

    def value(self):
        return self.p / self.q

    def farey_add(self, other):
        assert (type(self) == type(other))
        new_level = max(self.level, other.level) + 1
        return type(self)(self.p + other.p, self.q + other.q, new_level)

    def is_adjacent(self, other):
        assert (type(self) == type(other))
        return abs(self.p * other.q - self.q * other.p) == 1

    def is_one_level_off(self, other):
        assert (type(self) == type(other))
        return abs(self.level - other.level) == 1

    def __str__(self):
        return f'{self.p}/{self.q} ({self.level})'

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        if type(self) == type(other):
            return self.value() < other.value()
        else:
            return self.value() < other

    def __gt__(self, other):
        if type(self) == type(other):
            return self.value() > other.value()
        else:
            return self.value() > other

    def __eq__(self, other):
        if type(self) == type(other):
            return self.p == other.p and self.q == other.q
        else:
            return self.value() == other


def get_approximation_sequence(target, n_limit=1000, use_tuple=False):
    left_bound = FareyRational(0, 1)
    right_bound = FareyRational(1, 1)
    assert (left_bound < target < right_bound)
    sequence = [left_bound, right_bound]
    counter = 0
    while counter <= n_limit:
        bisect_point = left_bound.farey_add(right_bound)
        sequence.append(bisect_point)
        if bisect_point == target:
            break
        elif bisect_point > target:
            right_bound = bisect_point
        else:
            left_bound = bisect_point
    if use_tuple:
        result = [(rational.p, rational.q) for rational in sequence]
    else:
        result = sequence
    return result


def get_approximation_sequence_with_threshold(target, bound_func, q_limit=500000, use_tuple=False):
    assert (0 < target < 1)
    left = FareyRational(0, 1)
    right = FareyRational(1, 1)
    sequence = [left] if target < 1 / 2 else [right]
    while True:
        mid = left.farey_add(right)
        error = bound_func(mid.q)
        if abs(mid.value() - target) < error:
            sequence.append(mid)
        if mid > target:
            right = mid
        else:
            left = mid
        if mid.q > q_limit:
            break
    if use_tuple:
        result = [(rational.p, rational.q) for rational in sequence]
    else:
        result = sequence
    return result


def get_mediant_sequence_from(init_seq, n_levels, use_tuple=False):
    for level in range(1, n_levels + 1):
        new_seq = []
        for k in range(len(init_seq) - 1):
            new_seq.append(init_seq[k])
            new_seq.append(init_seq[k].farey_add(init_seq[k + 1]))
        new_seq.append(init_seq[-1])
        init_seq = new_seq
    if use_tuple:
        result = [(rational.p, rational.q) for rational in init_seq]
    else:
        result = init_seq
    return result


def get_farey_sequence(n_levels, use_tuple=False):
    init_seq = [FareyRational(0, 1, level=0), FareyRational(1, 1, level=0)]
    return get_mediant_sequence_from(init_seq, n_levels, use_tuple=use_tuple)


def get_stern_brocot_sequence(n_levels, use_tuple=False):
    init_seq = [FareyRational(0, 1, level=0), FareyRational(1, 0, level=0)]
    return get_mediant_sequence_from(init_seq, n_levels, use_tuple=use_tuple)


# Mobjects
class Fraction(VGroup):
    def __init__(self, numer, denom, **kwargs):
        super().__init__(**kwargs)
        numer_tex = Tex(str(numer))
        denom_tex = Tex(str(denom))
        line = Rectangle(
            fill_color=WHITE, fill_opacity=1, stroke_width=0,
            height=0.02,
        )
        line.set_width(
            max(numer_tex.get_width(), denom_tex.get_width()) * 1.2,
            stretch=True,
        )
        self.add(numer_tex, line, denom_tex)
        self.arrange(DOWN, buff=0.15)
        self.numer = numer
        self.denom = denom
        self.numer_tex = numer_tex
        self.line = line
        self.denom_tex = denom_tex

    def get_numerator(self):
        return self.numer

    def get_denominator(self):
        return self.denom

    def get_numerator_tex(self):
        return self.numer

    def get_denominator_tex(self):
        return self.denom

    def get_line_tex(self):
        return self.line


class FordFractalAxis(NumberLine):
    CONFIG = {
        'x_range': [-1, 2, 1],
        'unit_size': 7,
        'include_tick_numbers': True,
        'tick_number_color': GREY_A,
        'tick_number_size': 1,
        'y_offset': -2,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shift(self.y_offset * UP)
        if self.include_tick_numbers:
            self.add_tick_numbers()

    def add_tick_numbers(self):
        x_min, x_max, step = self.x_range
        for x in range(x_min, x_max + 1):
            self.add_tick_number(x)

    def add_tick_number(self, x):
        tex = Tex(str(x))
        tex.set_color(self.tick_number_color)
        tex.scale(self.tick_number_size)
        tex.next_to(self.number_to_point(x), DOWN)
        self.add(tex)


class FordFractalGap(VMobject):
    CONFIG = {
        'gap_style': {
            'color': YELLOW,
            'fill_opacity': 0.5,
            'stroke_width': 2,
        }
    }

    def __init__(self, ford_fractal, *numers_and_denoms, **kwargs):
        super().__init__(**kwargs)
        if len(numers_and_denoms) == 2:
            # Gap between to two circles and the axis
            (p1, q1), (p2, q2) = numers_and_denoms
            axis = ford_fractal.get_axis()
            circle1 = ford_fractal.get_circle(p1, q1).get_circle()
            center1 = circle1.get_center()
            tangent1 = axis.number_to_point(p1 / q1)
            circle2 = ford_fractal.get_circle(p2, q2).get_circle()
            center2 = circle2.get_center()
            tangent2 = axis.number_to_point(p2 / q2)
            polygon = Polygon(center1, center2, tangent2, tangent1)
            clips = (circle1, circle2)
        else:
            # Gap between to three circles
            circles = tuple(
                map(
                    lambda p_and_q: ford_fractal.get_circle(*p_and_q).get_circle(),
                    numers_and_denoms,
                )
            )
            centers = tuple(
                map(
                    lambda circle: circle.get_center(),
                    circles,
                )
            )
            polygon = Polygon(*centers)
            clips = circles
        result = polygon
        for clip in clips:
            result = Difference(result, clip, **self.gap_style)
        self.add(result)


class FordFractal(VGroup):
    CONFIG = {
        'max_denom': 10,
        'max_zoom_level': 27,
        'zoom_places': [],
        'axis_config': {},
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_axis()
        self.add_circles()

    def add_axis(self):
        self.axis = FordFractalAxis(**self.axis_config)
        self.add(self.axis)

    def add_circles(self):
        self.add_basic_circles()
        self.add_extra_circles_for_zooming()

    def add_basic_circles(self):
        self.circle_dict = {}
        for denom in range(1, self.max_denom + 1):
            for numer in get_coprime_numers_by_denom(denom):
                self.add_circle(numer, denom)

    def add_extra_circles_for_zooming(self):
        # Extra circles have to be close enough to one of the zooming places
        diameter_thres = 0.01 / self.max_zoom_level
        denom_thres = int(math.sqrt(1 / diameter_thres)) + 1
        extra_circle_list = set()
        for denom in range(self.max_denom + 1, denom_thres + 1):
            for target in self.zoom_places:
                base = denom * target
                offset = 100 / denom
                lower = max(0, round(base - offset) - 1)
                upper = min(denom, round(base + offset) + 1)
                for numer in get_coprime_numers_by_denom(denom):
                    if lower <= numer <= upper:
                        extra_circle_list.add((numer, denom))
        for (numer, denom) in extra_circle_list:
            self.add_circle(numer, denom)

    def has_circle(self, numer, denom):
        return (numer, denom) in self.circle_dict

    def generate_circle(self, numer, denom):
        circle = SingleFCircle(self.axis, numer, denom)
        self.circle_dict[(numer, denom)] = circle
        return circle

    def add_circle(self, numer, denom):
        if not self.has_circle(numer, denom):
            circle = self.generate_circle(numer, denom)
            self.add(circle)

    def get_circle(self, numer, denom):
        if not self.has_circle(numer, denom):
            raise Exception(f'Circle {numer}/{denom} not found.')
        return self.circle_dict[(numer, denom)]

    def get_circles(self, numer_denom_pairs):
        return VGroup(*[
            self.get_circle(numer, denom)
            for (numer, denom) in numer_denom_pairs
        ])

    def get_all_circles(self):
        return self.get_circles(self.circle_dict)

    def get_numer_denom_pairs(self):
        return self.circle_dict

    def get_axis(self):
        return self.axis

    def zoom_in_on(self, place, zoom_factor, animate=False):
        axis = self.axis
        point = axis.number_to_point(place)
        if animate:
            return ApplyMethod(axis.scale, zoom_factor, {'about_point': point})
        else:
            axis.scale(zoom_factor, about_point=point)
            return self

    def get_tangent_point(self, a, b, c, d):
        discriminant = abs(a * d - b * c)
        if discriminant == 1:
            unit_size = self.get_axis().get_unit_size()
            origin = self.get_axis().number_to_point(0)
            x_coord = (a * b + c * d) / (b**2 + d**2)
            y_coord = 1 / (b**2 + d**2)
            return origin + unit_size * (x_coord * RIGHT + y_coord * UP)
        else:
            raise Exception(f'Two circles {a}/{b} and {c}/{d} are not tangent.')


class FineCircle(Circle):
    CONFIG = {
        'color': BLUE,
        'n_components': 100,
    }

    def __init__(self, **kwargs):
        super().__init__(start_angle=-PI / 2, **kwargs)


class SingleFCircle(VGroup):
    CONFIG = {
        'include_fraction': True,
        'fraction_height_factor': 0.6,
    }

    def __init__(self, axis, numer, denom, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.numer = numer
        self.denom = denom
        # Generate the circle
        circle = self.generate_circle()
        self.add(circle)
        self.circle = circle
        if self.include_fraction:
            fraction = self.generate_fraction()
            self.add(fraction)
            self.fraction = fraction

    def generate_circle(self):
        circle = FineCircle()

        def circle_updater(circle):
            unit_size = self.axis.get_unit_size()
            number = self.numer / self.denom
            diameter = 1 / (self.denom**2)
            circle_height = circle.get_height()
            circle.set_height(unit_size * diameter)
            circle.set_stroke(width=get_stroke_width_by_height(circle_height))
            circle.next_to(self.axis.number_to_point(number), UP, buff=0)
        circle.add_updater(circle_updater)
        return circle

    def generate_fraction(self):
        fraction = Fraction(self.numer, self.denom)

        def fraction_updater(frac):
            circle_height = self.circle.get_height()
            if self.circle.get_height() > 1e-6:
                frac.set_height(circle_height * self.fraction_height_factor)
            frac.set_opacity(get_fraction_opacity_by_height(circle_height))
            frac.move_to(self.circle)
        fraction.add_updater(fraction_updater)
        return fraction

    def get_circle(self):
        return self.circle

    def get_fraction(self):
        return self.fraction


class RemarkText(TexText):
    def __init__(
        self, mob, text,
        direction=DOWN, aligned_edge=LEFT,
        scale_factor=0.6, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        **kwargs
    ):
        super().__init__(text, **kwargs)
        self.scale(scale_factor)
        self.next_to(mob, direction=direction, aligned_edge=aligned_edge, buff=buff)


class CircleCenter(Dot):
    def __init__(self, circle, **kwargs):
        super().__init__(**kwargs)
        self.set_color(circle.get_color())
        self.add_updater(lambda mob: mob.move_to(circle.get_center()))


class CalcForm(VGroup):
    CONFIG = {
        'number_colors': [RED, BLUE, WHITE, GOLD],
    }

    def __init__(self, dividend, divisor, **kwargs):
        super().__init__(**kwargs)
        quotient, remainder = get_quotient_and_remainder(dividend, divisor)
        self.dividend, self.divisor, self.quotient, self.remainder = (dividend, divisor, quotient, remainder)
        for part in self.get_parts_texts():
            self.add(Tex(str(part)))
        self.arrange(RIGHT)
        number_texs = self.get_number_texs()
        for tex, color in zip(number_texs, self.number_colors):
            tex.set_color(color)
        self.dividend_tex, self.divisor_tex, self.quotient_tex, self.remainder_tex = number_texs

    def get_parts_texts(self):
        raise Exception('This should be implemented in subclasses.')

    def get_number_texs(self):
        raise Exception('This should be implemented in subclasses.')

    def get_dividend_tex(self):
        return self.dividend_tex

    def get_divisor_tex(self):
        return self.divisor_tex

    def get_quotient_tex(self):
        return self.quotient_tex

    def get_remainder_tex(self):
        return self.remainder_tex


class DivisionForm(CalcForm):
    def get_parts_texts(self):
        return (
            self.dividend, '\\divisionsymbol', self.divisor, '=',
            self.quotient, '\\cdots', self.remainder,
        )

    def get_number_texs(self):
        return self[::2]


class MultiplicationForm(CalcForm):
    def get_parts_texts(self):
        return (
            self.dividend, '=', self.quotient, '\\times',
            self.divisor, '+', self.remainder,
        )

    def get_number_texs(self):
        return [self[ind] for ind in (0, 4, 2, 6)]


class RemainderForm(CalcForm):
    def get_parts_texts(self):
        return (
            self.remainder, '=', self.dividend, '-',
            self.quotient, '\\times', self.divisor,
        )

    def get_number_texs(self):
        return [self[ind] for ind in (2, 6, 4, 0)]


class CalcList(VGroup):
    CONFIG = {
        'form_type': None,
        'number_colors': [RED, BLUE, GOLD_D, MAROON_B, GREEN, YELLOW_B, PINK, TEAL, ORANGE],
        'align_index': -4,
        'form_buff': 0.3,
    }

    def __init__(self, dividend, divisor, **kwargs):
        super().__init__(**kwargs)
        level = 0
        self.add_inline_form(dividend, divisor, level)
        remainder = dividend % divisor
        while remainder > 0:
            dividend, divisor = divisor, remainder
            level += 1
            self.add_inline_form(dividend, divisor, level)
            remainder = dividend % divisor
        self.arrange(
            DOWN,
            index_of_submobject_to_align=self.align_index,
            buff=self.form_buff,
        )

    def add_inline_form(self, dividend, divisor, level):
        FormType = self.form_type
        colors = self.get_number_colors_set(level)
        inline_form = FormType(dividend, divisor, number_colors=colors)
        self.add(inline_form)

    def get_number_colors_set(self, level):
        colors = self.number_colors
        a, b, c = list(map(lambda ind: ind % len(colors), range(level, level + 3)))
        return [colors[a], colors[b], WHITE, colors[c]]

    def get_inline_form(self, level):
        return self[level]


class DivisionList(CalcList):
    CONFIG = {
        'form_type': DivisionForm,
        'align_index': -4,
        'form_buff': 0.5,
    }

    def get_all_quotients(self):
        return VGroup(*[form.get_quotient_tex() for form in self])


class MultiplicationList(CalcList):
    CONFIG = {
        'form_type': MultiplicationForm,
    }


class RemainderList(CalcList):
    CONFIG = {
        'form_type': RemainderForm,
        'align_index': 1,
        'form_buff': 0.5,
    }


class Reference(VGroup):
    CONFIG = {
        'name_color': BLUE,
        'author_color': BLUE_A,
        'pub_color': GREY_A,
        'doi_color': GREY_A,
        'info_color': GOLD_A,
    }

    def __init__(
        self,
        name='Name',
        authors='Authors',
        pub='Publication',
        doi='doi',
        info='',
        **kwargs,
    ):
        super().__init__(**kwargs)
        texts = [name, authors, pub, doi, info]
        colors = [self.name_color, self.author_color, self.pub_color, self.doi_color, self.info_color]
        scale_factors = [1, 0.7, 0.6, 0.6, 0.7]
        texs = VGroup(*[
            TexText(text, color=color, alignment='').scale(factor)
            for text, color, factor in zip(texts, colors, scale_factors)
        ])
        texs.arrange(DOWN, aligned_edge=LEFT)
        texs[2:-1].shift(0.5 * RIGHT)
        texs.scale(0.8)
        self.add(texs)


class FareyTree(VGroup):
    CONFIG = {
        'n_levels': 5,
        'include_top': True,
        'h_buff': 0.5,
        'v_buff': 1.5,
        'line_config': {
            'color': GREY,
            'stroke_width': 3,
            'buff': 0.5,
        },
        'auto_line_buff': True,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        sequence = self.generate_sequence(self.n_levels)
        if not self.include_top:
            sequence = sequence[1:-1]
        self.sequence = sequence
        # Add fractions
        fractions = VGroup()
        frac_to_tex_dict = {}
        for k, frac in enumerate(sequence):
            tex = Tex(f'{frac.p} \\over {frac.q}')
            tex.shift(k * self.h_buff * RIGHT)
            tex.shift(frac.level * self.v_buff * DOWN)
            fractions.add(tex)
            frac_to_tex_dict[(frac.p, frac.q)] = tex
        fractions.center()
        self.fractions = fractions
        self.frac_to_tex_dict = frac_to_tex_dict
        # Add lines connecting the fractions
        lines = VGroup()
        line_to_fracs_dict = {}
        fracs_to_line_dict = {}
        for (frac_a, frac_b) in it.combinations(sequence, 2):
            if frac_a.is_adjacent(frac_b) and frac_a.is_one_level_off(frac_b):
                line = self.generate_line(frac_a, frac_b)
                lines.add(line)
                top_frac, bottom_frac = self.sort_fracs_by_level(frac_a, frac_b)
                line_to_fracs_dict[line] = ((top_frac.p, top_frac.q), (bottom_frac.p, bottom_frac.q))
                fracs_to_line_dict[((top_frac.p, top_frac.q), (bottom_frac.p, bottom_frac.q))] = line
        self.add(fractions, lines)
        self.lines = lines
        self.line_to_fracs_dict = line_to_fracs_dict
        self.fracs_to_line_dict = fracs_to_line_dict

    def generate_sequence(self, level):
        return get_farey_sequence(level)

    def sort_fracs_by_level(self, frac_a, frac_b):
        return (frac_a, frac_b) if frac_a.level < frac_b.level else (frac_b, frac_a)

    def generate_line(self, frac_a, frac_b):
        top_frac, bottom_frac = self.sort_fracs_by_level(frac_a, frac_b)
        top_tex = self.get_fraction_from_rational(top_frac)
        bottom_tex = self.get_fraction_from_rational(bottom_frac)
        lc = self.line_config
        tc, bc = top_tex.get_center(), bottom_tex.get_center()
        # Line buff tweak
        if self.auto_line_buff:
            abs_distance = get_norm(bc - tc)
            x_diff = abs(top_tex.get_x() - bottom_tex.get_x())
            y_diff = abs(top_tex.get_y() - bottom_tex.get_y())
            if y_diff < 4 * x_diff:
                outer_x_diff = max([0.0, x_diff - top_tex.get_width()])
                x_buff_ratio = min([0.4, 1. - outer_x_diff / x_diff])
                buff = abs_distance * x_buff_ratio
            else:
                outer_y_diff = max([0.0, y_diff - top_tex.get_height()])
                y_buff_ratio = min([0.2, 1. - outer_y_diff / y_diff])
                buff = abs_distance * y_buff_ratio
            lc['buff'] = buff
        return Line(tc, bc, **lc)

    def get_sequence(self):
        return self.sequence

    def get_fraction_from_numbers(self, numer, denom):
        return self.frac_to_tex_dict[(numer, denom)]

    def get_fraction_from_rational(self, rational):
        return self.frac_to_tex_dict[(rational.p, rational.q)]

    def get_subtrees_from_numbers(self, numer, denom):
        left_group, right_group = VGroup(), VGroup()
        sequence = self.get_sequence()
        ind = sequence.index(FareyRational(numer, denom))
        root_frac = sequence[ind]
        for k_left in range(ind - 1, -1, -1):
            left_frac = sequence[k_left]
            if left_frac.level <= root_frac.level:
                break
            mob = self.get_fraction_from_rational(left_frac)
            left_group.add(mob)
        for k_right in range(ind + 1, len(sequence), 1):
            right_frac = sequence[k_right]
            if right_frac.level <= root_frac.level:
                break
            mob = self.get_fraction_from_rational(right_frac)
            right_group.add(mob)
        return left_group, right_group

    def get_all_fractions(self):
        return self.fractions

    def get_line_from_frac_tuples(self, frac_a, frac_b):
        top_frac, bottom_frac = self.sort_fracs_by_level(frac_a, frac_b)
        key = ((top_frac.p, top_frac.q), (bottom_frac.p, bottom_frac.q))
        return self.fracs_to_line_dict[key]

    def get_line_from_frac_component(self, frac):
        for key in self.fracs_to_line_dict:
            for component in key:
                if frac == component:
                    return self.fracs_to_line_dict[key]

    def get_all_lines(self):
        return self.lines

    def get_all_adjacent_lines(self):
        return VGroup(*[
            self.generate_line(frac_a, frac_b)
            for (frac_a, frac_b) in it.combinations(self.sequence, 2)
            if frac_a.is_adjacent(frac_b)
        ])

    def get_fractions_of_level(self, level):
        return tuple(
            filter(
                lambda frac: frac.level == level,
                self.sequence
            )
        )

    def get_fractions_tuples_of_level(self, level):
        return tuple(
            map(
                lambda frac: (frac.p, frac.q),
                filter(
                    lambda frac: frac.level == level,
                    self.sequence
                )
            )
        )

    def get_fraction_mobs_of_level(self, level):
        return VGroup(
            *map(
                lambda frac: self.get_fraction_from_rational(frac),
                self.get_fractions_of_level(level)
            )
        )

    def get_parent_fracs_tuple_of_line(self, line):
        tip_frac_numbers = self.line_to_fracs_dict.get(line, None)
        if tip_frac_numbers:
            return tip_frac_numbers
        else:
            raise Exception('Line is not in the tree.')

    def get_line_mobs_of_level(self, level):
        frac_of_this_level = self.get_fractions_tuples_of_level(level)
        return VGroup(
            *filter(
                lambda line: self.get_parent_fracs_tuple_of_line(line)[0] in frac_of_this_level,
                self.lines
            )
        )


class SternBrocotTree(FareyTree):
    def generate_sequence(self, level):
        return get_stern_brocot_sequence(level)


# Custom Scenes
class ReferenceScene(Scene):
    CONFIG = {
        'pause_time_between_pages': 5,
    }

    def construct(self):
        refs = self.get_references()
        if len(refs) == 0:
            return
        for ref, y_pos in zip(refs, it.cycle([3, -0.5])):
            ref.center().next_to(y_pos * UP, DOWN, buff=0).to_edge(LEFT, buff=1)
        num_of_groups = (len(refs) + 1) // 2
        ref_groups = [
            VGroup(*refs[2 * k:2 * k + 2])
            for k in range(num_of_groups)
        ]
        curr_group = None
        for group in ref_groups:
            if not curr_group:
                self.play(FadeIn(group), run_time=0.5)
            else:
                self.play(FadeOut(curr_group), run_time=0.5)
                self.play(FadeIn(group), run_time=0.5)
            curr_group = group
            self.wait(self.pause_time_between_pages)
        self.play(FadeOut(ref_groups[-1]), run_time=0.5)
        self.wait()

    def get_references(self):
        return ()


# Quadratic field stuff
def get_imag_unit(D=1):
    if D == 1 or D == 2:
        return math.sqrt(D) * 1j
    else:
        return (1 + math.sqrt(D) * 1j) / 2


def get_real_and_imag_part(z, D=1):
    imag_unit = get_imag_unit(D)
    imag_part = z.imag / imag_unit.imag
    real_part = z.real - imag_part * imag_unit.real
    return real_part, imag_part


def quad_field_gcd(a, b, thres=1e-8):
    assert (a.D == b.D)
    D = a.D
    if abs(a) < abs(b):
        a, b = b, a
    return a if (abs(b) < thres) else quad_field_gcd(b, a % b)


class DInteger(object):
    def __init__(self, x, y, D=1):
        """
        The set of algebraic integers Z[sigma] in the complex quadratic field Q(sqrt(-D))
        sigma = sqrt(-D),       D = 1 or 2 mod 4
              = (1+sqrt(-D))/2, D = 3 mod 4
        """
        self.D = D
        self.imag_unit = get_imag_unit(D)
        self.real = x
        self.imag = y

    def __add__(self, di):
        return DInteger(self.real + di.real, self.imag + di.imag, self.D)

    def __sub__(self, di):
        return DInteger(self.real - di.real, self.imag - di.imag, self.D)

    def __mul__(self, di):
        product = self.get_complex_value() * di.get_complex_value()
        real_part, imag_part = get_real_and_imag_part(product, self.D)
        return DInteger(round(real_part), round(imag_part), self.D)

    def __mod__(self, di):
        if self.D in [1, 2]:
            quotient_unit_list = [
                DInteger(1, 0, self.D),
                DInteger(0, 1, self.D),
                DInteger(-1, 0, self.D),
                DInteger(0, -1, self.D),
            ]
        elif self.D in [3, 7, 11]:
            quotient_unit_list = [
                DInteger(1, 0, self.D),
                DInteger(0, 1, self.D),
                DInteger(-1, 0, self.D),
                DInteger(0, -1, self.D),
                DInteger(1, -1, self.D),
                DInteger(-1, 1, self.D),
            ]
        else:
            raise ValueError('Quadratic ring is not Euclidean')
        di_norm = di.norm()
        if di_norm == 0:
            raise ZeroDivisionError
        remainder = self
        while remainder.norm() >= di_norm:
            quotient_unit = min(quotient_unit_list, key=lambda e: (remainder + (e * di)).norm())
            remainder += quotient_unit * di
            while (remainder + (quotient_unit * di)).norm() < remainder.norm():
                remainder += quotient_unit * di
        return remainder

    def __truediv__(self, di):
        quotient = self.get_complex_value() / di.get_complex_value()
        real_part, imag_part = get_real_and_imag_part(quotient, self.D)
        return DInteger(real_part, imag_part, self.D)

    def __floordiv__(self, di):
        remainder = self % di
        quotient = (self - remainder) / di
        return DInteger(round(quotient.real), round(quotient.imag), self.D)

    def __divmod__(self, di):
        return self // di, self % di

    def __pos__(self):
        return self

    def __neg__(self):
        return DInteger(0, 0, self.D) - self

    def __str__(self):
        return f"{self.real}+{self.imag}d"

    def __repr__(self):
        return f"{self.real}+{self.imag}d"

    def norm(self):
        if self.D % 4 != 3:
            return (self.real**2 + self.D * self.imag**2)
        else:
            return (self.real**2 + self.real * self.imag + (self.D + 1) // 4 * self.imag**2)

    def __abs__(self):
        return math.sqrt(self.norm())

    def get_complex_value(self):
        return self.real + self.imag * self.imag_unit


class GaussianInteger(DInteger):
    def __init__(self, x, y):
        DInteger.__init__(self, x, y, D=1)


class HippasusInteger(DInteger):
    def __init__(self, x, y):
        DInteger.__init__(self, x, y, D=2)


class EisensteinInteger(DInteger):
    def __init__(self, x, y):
        DInteger.__init__(self, x, y, D=3)


class KleinianInteger(DInteger):
    def __init__(self, x, y):
        DInteger.__init__(self, x, y, D=7)


class UnnamedInteger(DInteger):
    def __init__(self, x, y):
        DInteger.__init__(self, x, y, D=11)


# Part 1 Scenes
class IntroSceneP1(Scene):
    def construct(self):
        # How to add two fractions the right way
        question = Tex('{1 \\over 2}', '+', '{1 \\over 3}', '=\\,?')
        answer = Tex('{3 \\over 6}', '+', '{2 \\over 6}', '= {5 \\over 6}')
        wrong_answer = Tex('{1 \\over 2}', '\\oplus', '{1 \\over 3}', '= {2 \\over 5}')
        tick = Tex('\\text{\\ding{51}}', color=GREEN)
        cross = Tex('\\text{\\ding{55}}', color=RED)
        for mob in (question, answer, wrong_answer, tick, cross):
            mob.scale(1.5)
            mob.shift(1.5 * UP)
        answer.next_to(question, DOWN, aligned_edge=LEFT, buff=0.8)
        wrong_answer.set_color(RED).move_to(answer)
        tick.next_to(question, RIGHT, buff=1)
        cross.next_to(wrong_answer, RIGHT, buff=0.8)
        self.play(Write(question))
        self.wait()
        self.play(FadeTransform(question[:3].copy(deep=True), answer[:3]))
        self.wait()
        self.play(Write(answer[3:]))
        self.wait()
        self.play(
            FadeOut(question[3:]), FadeOut(answer[:3]),
            answer[3:].animate.next_to(question[:3].get_right(), coor_mask=Y_COOR_MASK).set_color(GREEN),
            question[:3].animate.set_color(GREEN),
        )
        self.wait()
        # It would be hilarious to see denominators and numerators add respectively
        self.play(Write(wrong_answer))
        self.wait()
        self.play(Write(tick), Write(cross))
        self.wait()
        fadeout_mobs = VGroup(question[:3], answer[3:], tick, cross)
        remained_mobs = VGroup(wrong_answer)
        self.play(
            FadeOut(fadeout_mobs),
            remained_mobs.animate.center().shift(2.5 * UP).set_color(GREEN),
            run_time=2,
        )
        self.wait()
        # This weird addition connects to many ideas
        pi_approx = Tex('\\pi \\approx {22 \\over 7}')
        pi_approx.set_color(YELLOW)
        pi_approx.move_to(3 * RIGHT + DOWN)
        ph_rect = ScreenRectangle(height=3)     # This is only a placeholder
        ph_rect.become(Mobject())
        ph_rect.move_to((1 * LEFT + DOWN))
        self.play(Write(ph_rect), Write(pi_approx))
        self.wait()
        hurwitz = Tex('\\left| {p \\over q} - \\alpha \\right| < {1 \\over \\sqrt{5} q^2}')
        contd_frac = Tex('{355 \\over 113} = 3 + \\frac{1}{7 + \\frac{1}{16}}')
        poly_1 = ImageMobject('poly_1.png')
        poly_11 = ImageMobject('poly_11.png')
        ford_spheres_1 = ImageMobject('ford_spheres_1.png')
        idea_group = Group(hurwitz, poly_1, contd_frac, poly_11, ford_spheres_1)
        colors = [BLUE_A, None, MAROON_A, None, None]
        positions = [
            5 * LEFT + 0.5 * UP,
            5.6 * RIGHT + 0.1 * DOWN,
            0.1 * RIGHT + 2.7 * DOWN,
            5.6 * LEFT + 2.3 * DOWN,
            5.4 * RIGHT + 2.6 * DOWN
        ]
        for mob, color, pos in zip(idea_group, colors, positions):
            if color:
                mob.set_color(color)
                mob.scale(random.random() * 0.1 + 0.7)
            else:
                mob.set_height(random.random() * 0.5 + 1.8)
            mob.rotate(random.randint(-19, 19) * DEGREES)
            mob.move_to(pos)
        self.play(
            AnimationGroup(
                *[GrowFromCenter(mob) for mob in idea_group],
                lag_ratio=0.2,
            ),
            run_time=2,
        )
        self.wait()


class FordCirclesIntroP1(Scene):
    def construct(self):
        # Show ford circles' axis
        ff = FordFractal(
            max_denom=30,
            max_zoom_level=60,
            zoom_places=[np.sqrt(2) / 2],
            axis_config={'y_offset': -2.5, },
        )
        ff.suspend_updating()
        ff_axis = ff.get_axis()
        self.play(Write(ff_axis))
        self.wait()
        # Construct circle associated with 1/2
        one_half_circle = Circle(color=YELLOW, start_angle=-PI / 2)
        one_half_circle.move_to(1.5 * RIGHT + 1.5 * UP)
        one_half_tex = Tex('1 \\over 2')
        one_half_tex.scale(1.5).move_to(1.5 * LEFT + 1.5 * UP)
        arrow = Arrow(one_half_tex.get_right(), one_half_circle.get_left(), buff=0.4)
        one_half_tex.move_to(1.5 * UP)
        one_half_pos = ff_axis.number_to_point(1 / 2)
        one_half_tex.generate_target()
        one_half_tex.scale(0).move_to(one_half_pos)
        one_half_circle.generate_target()
        one_half_circle.move_to(1.5 * UP).fade(1)
        self.play(MoveToTarget(one_half_tex), run_time=1.5)
        self.wait()
        self.play(
            one_half_tex.animate.shift(1.5 * LEFT),
            MoveToTarget(one_half_circle),
            GrowFromCenter(arrow),
        )
        self.wait()
        self.play(
            ApplyMethod(
                one_half_circle.next_to, one_half_pos,
                {'direction': UP, 'buff': 0, },
                path_arc=-PI / 3,
            ),
        )
        self.wait()
        one_half_tex_copy = one_half_tex.copy()
        self.play(
            one_half_tex_copy.animate.scale(0.5).next_to(one_half_pos, DOWN),
        )
        self.wait()
        diameter_rule = TexText('直径', '$\\dfrac{1}{2^2}$')
        diameter_rule.scale(1.5).next_to(arrow.get_right())
        diameter_rule[-1][2:-1].set_color(YELLOW)
        true_one_half_circle = ff.get_circle(1, 2).get_circle()
        true_one_half_fraction = ff.get_circle(1, 2).get_fraction()
        self.play(
            Write(diameter_rule),
            one_half_tex[-1][-1].animate.set_color(YELLOW),
        )
        self.wait()
        self.play(Transform(one_half_circle, true_one_half_circle))
        self.wait()
        one_half_tex_copy.generate_target()
        one_half_tex_copy.target[0].become(true_one_half_fraction)
        self.play(MoveToTarget(one_half_tex_copy), run_time=5)
        self.wait()
        # Switch this animated mobject with the real ford circle
        replaced_mobs = VGroup(one_half_tex_copy, one_half_circle)
        self.remove(replaced_mobs)
        self.add(ff.get_circle(1, 2))
        # More examples: 2/3, 1/5 and 0/1
        pairs = [(2, 3), (1, 5), (0, 1)]
        old_diameter_rule = diameter_rule
        old_fraction_tex = one_half_tex
        for pair in pairs:
            numer, denom = pair
            fraction_tex = Tex(f'{numer} \\over {denom}')
            fraction_tex.scale(1.5).next_to(arrow.get_left(), LEFT)
            fraction_tex[-1][-1].set_color(YELLOW),
            diameter_rule = TexText('直径', f'$\\dfrac{{1}}{{{{{denom}}}^2}}$')
            diameter_rule.scale(1.5).next_to(arrow.get_right(), RIGHT)
            diameter_rule[-1][2:-1].set_color(YELLOW)
            self.play(
                FadeTransform(old_fraction_tex, fraction_tex),
                FadeTransform(old_diameter_rule, diameter_rule),
            )
            self.wait()
            true_circle = ff.get_circle(numer, denom)
            true_circle.generate_target()
            true_circle.scale(0).move_to(diameter_rule)
            self.play(MoveToTarget(true_circle))
            self.wait()
            old_fraction_tex = fraction_tex
            old_diameter_rule = diameter_rule
        # Fade out rules and show all circles
        rule_group = VGroup(fraction_tex, arrow, diameter_rule)
        self.play(FadeOut(rule_group))
        self.wait()
        rest_pairs = set(ff.get_numer_denom_pairs()) - set(pairs + [(1, 2)])
        rest_circles = tuple(map(lambda pair: ff.get_circle(*pair), rest_pairs))
        pairs_circles_zip = list(zip(rest_pairs, rest_circles))
        random.shuffle(pairs_circles_zip)
        self.play(
            AnimationGroup(*[
                FadeIn(circle, shift=(DOWN if pair[1] == 1 else UP) / np.sqrt(pair[1]))
                for pair, circle in pairs_circles_zip
            ]),
            lag_ratio=0.8, run_time=5,
        )
        self.wait()
        # Zoom in on a point to show 'tangency is everywhere'
        ff.resume_updating()
        point = ff.zoom_places[0]
        run_time = 2.5
        wait_time = 1.5
        for zoom_factor in (3, 2, 2, 2, 2):
            self.play(ff.zoom_in_on(point, zoom_factor, animate=True), run_time=run_time)
            self.wait(wait_time)


class FordCirclesIntroAddonP1(Scene):
    def construct(self):
        chn_text = TexText('福特圆', color=YELLOW)
        chn_text.to_corner(UL)
        eng_text = RemarkText(chn_text, '(Ford Circles)', color=YELLOW)
        ford_portrait = ImageMobject('Lester_Ford.png', height=3)
        ford_portrait.to_edge(LEFT)
        ford_text = RemarkText(ford_portrait, 'Lester R. Ford', aligned_edge=ORIGIN, scale_factor=0.5)
        self.add(chn_text, eng_text, ford_portrait, ford_text)
        self.wait()


class WhyTangencyEverywhereP1(Scene):
    def construct(self):
        # Transition from the last scene
        axis = Line(LEFT_SIDE, RIGHT_SIDE, color=GREY_B, stroke_width=2)
        axis.move_to(2.5 * DOWN)
        self.add(axis)
        self.wait()

        # Separate, intersect and tangent
        r1 = 2.0
        r2 = 1.5
        color1 = BLUE
        color2 = RED
        # color1 = RED
        # color2 = BLUE
        c1 = FineCircle(radius=r1, fill_opacity=0.1, color=color1)
        c2 = FineCircle(radius=r2, fill_opacity=0.1, color=color2)
        for k, c in enumerate([c1, c2]):
            c.next_to(axis, UP, buff=0)
            c.shift(random.random() * 1.5 * LEFT * (-1)**k)
        self.play(DrawBorderThenFill(c1), DrawBorderThenFill(c2))
        self.wait()
        text_intersect = TexText('相交')
        text_separate = TexText('相离')
        text_tangent = TexText('相切')
        relations = VGroup(text_intersect, text_separate, text_tangent)
        for mob in relations:
            mob.set_color(YELLOW).scale(1.2).move_to(2.5 * UP)
        self.play(Write(text_intersect))
        self.wait()
        self.play(
            c1.animate.shift(LEFT),
            c2.animate.shift(RIGHT),
            FadeTransform(text_intersect, text_separate),
        )
        self.wait()
        dx = 2 * np.sqrt(r1 * r2)
        self.play(
            c1.animate.move_to(dx / 2 * LEFT).next_to(axis, UP, buff=0, coor_mask=Y_COOR_MASK),
            c2.animate.move_to(dx / 2 * RIGHT).next_to(axis, UP, buff=0, coor_mask=Y_COOR_MASK),
            FadeTransform(text_separate, text_tangent),
        )
        self.wait()
        self.play(
            c1.animate.shift(0.7 * LEFT),
            c2.animate.shift(0.7 * RIGHT),
            FadeOut(text_tangent),
        )
        self.wait()

        # Calculate 2 quantities and determine the relation
        cc1, cc2 = CircleCenter(c1), CircleCenter(c2)
        aux_line_config = {'color': GREY_B, 'stroke_width': 2, 'fade': 0.2}
        cc_line, cc1_axis_line, cc2_axis_line, perp_line = \
            line_group = VGroup(*[Line(LEFT, RIGHT, **aux_line_config) for _ in range(4)])
        cc_line.add_updater(
            lambda line: line.put_start_and_end_on(c1.get_center(), c2.get_center())
        )
        cc1_axis_line.add_updater(
            lambda line: line.put_start_and_end_on(c1.get_center(), c1.get_bottom())
        )
        cc2_axis_line.add_updater(
            lambda line: line.put_start_and_end_on(c2.get_center(), c2.get_bottom())
        )
        perp_line.add_updater(
            lambda line: line.put_start_and_end_on(
                c2.get_center(), c2.get_center() + (c2.get_center() - c1.get_center()) * LEFT
            )
        )
        line_group.suspend_updating()
        self.play(
            *[GrowFromCenter(line) for line in line_group[:-1]],
            GrowFromCenter(cc1),
            GrowFromCenter(cc2),
        )
        line_group.resume_updating()
        self.wait()

        distance_text = TexText('圆心距', color=YELLOW)
        radii_sum_text = TexText('半径和', color=YELLOW)
        text_group = VGroup(distance_text, radii_sum_text)
        text_group.arrange(RIGHT, buff=1)
        text_group.move_to(2.5 * UP)
        self.play(GrowFromPoint(distance_text, cc_line.get_center()))
        self.wait()
        self.play(
            GrowFromPoint(radii_sum_text[0][:-1], cc1_axis_line.get_center()),
            GrowFromPoint(radii_sum_text[0][-1], cc2_axis_line.get_center()),
        )
        self.add(radii_sum_text)
        self.wait()

        # Parameters for circles a/b and c/d
        a_over_b = Tex('a \\over b', color=color1)
        diameter_ab = TexText('直径', '$\\dfrac{1}{b^2}$', color=color1)
        radius_ab = TexText('半径', '$\\dfrac{1}{2b^2}$', color=color1)
        c_over_d = Tex('c \\over d', color=color2)
        diameter_cd = TexText('直径', '$\\dfrac{1}{d^2}$', color=color2)
        radius_cd = TexText('半径', '$\\dfrac{1}{2d^2}$', color=color2)
        a_over_b.add_updater(lambda mob: mob.next_to(c1, DOWN))
        c_over_d.add_updater(lambda mob: mob.next_to(c2, DOWN))

        def c1_remark_updater(mob):
            mob.next_to(c1, LEFT).next_to(axis, UP, coor_mask=Y_COOR_MASK)

        def c2_remark_updater(mob):
            mob.next_to(c2, RIGHT).next_to(axis, UP, coor_mask=Y_COOR_MASK)
        diameter_ab.add_updater(c1_remark_updater)
        radius_ab.add_updater(c1_remark_updater)
        diameter_cd.add_updater(c2_remark_updater)
        radius_cd.add_updater(c2_remark_updater)
        circle_remark_group = VGroup(
            a_over_b, diameter_ab, radius_ab, c_over_d, diameter_cd, radius_cd,
        )
        circle_remark_group.suspend_updating()
        for tex in (a_over_b, diameter_ab, c_over_d, diameter_cd):
            self.play(Write(tex))
            self.wait()
        self.play(
            AnimationGroup(*[
                FadeTransform(*pair)
                for pair in [(diameter_ab, radius_ab), (diameter_cd, radius_cd)]
            ])
        )
        self.wait()
        circle_remark_group.resume_updating()

        # Calculate sum of radii
        sum_of_radii = Tex('\\dfrac{1}{2b^2}', '+', '\\dfrac{1}{2d^2}')
        sum_of_radii.next_to(radii_sum_text.get_left(), RIGHT, buff=0)
        self.play(
            AnimationGroup(
                ReplacementTransform(radius_ab[1].copy(), sum_of_radii[0]),
                Write(sum_of_radii[1]),
                ReplacementTransform(radius_cd[1].copy(), sum_of_radii[-1]),
                lag_ratio=0.1,
            ),
            radii_sum_text.animate.shift(UP),
        )
        self.wait()

        # Construct a right triangle
        def triangle_vertices():
            return [c1.get_center(), perp_line.get_left(), c2.get_center(), c1.get_center()]
        right_triangle = Polygon(
            *triangle_vertices(),
            stroke_width=0, fill_color=YELLOW, fill_opacity=0.5,
        )
        right_triangle.add_updater(
            lambda mob: mob.set_points_as_corners(triangle_vertices())
        )
        triangle_group = VGroup(perp_line, right_triangle)
        triangle_group.suspend_updating()
        self.play(
            GrowFromCenter(perp_line),
            DrawBorderThenFill(right_triangle),
        )
        triangle_group.resume_updating()
        self.wait()

        # Calculate the length of the right sides
        fraction_diff = Tex('\\Big|', '\\dfrac{a}{b}', '-', '\\dfrac{c}{d}', '\\Big|')
        radius_diff = Tex('\\Big|', '\\dfrac{1}{2b^2}', '-', '\\dfrac{1}{2d^2}', '\\Big|')
        diffs = VGroup(fraction_diff, radius_diff)
        for diff in diffs:
            diff[1].set_color(color1)
            diff[-2].set_color(color2)
            diff.scale(0.6)
        fraction_diff.add_updater(lambda mob: mob.next_to(right_triangle, DOWN))
        radius_diff.add_updater(lambda mob: mob.next_to(right_triangle, LEFT))
        diff_sources = [(a_over_b[0], c_over_d[0]), (radius_ab[1], radius_cd[1])]
        diffs.suspend_updating()
        for diff, sources in zip(diffs, diff_sources):
            self.play(
                ReplacementTransform(sources[0].copy(), diff[1]),
                Write(diff[::2]),
                ReplacementTransform(sources[1].copy(), diff[-2]),
            )
            self.wait()
        self.add(diffs)
        diffs.resume_updating()

        # And now we have the distance between the circle centers
        distance_between_centers = Tex(
            '\\sqrt{',
            '\\Big(', '\\dfrac{a}{b}', '-', '\\dfrac{c}{d}', '\\Big)',
            '^2 + ',
            '\\Big(', '\\dfrac{1}{2b^2}', '-', '\\dfrac{1}{2d^2}', '\\Big)',
            '^2 }'
        )
        distance_between_centers.next_to(distance_text.get_right(), LEFT, buff=0)
        self.play(
            ReplacementTransform(fraction_diff.copy(), distance_between_centers[1:6]),
            ReplacementTransform(radius_diff.copy(), distance_between_centers[7:12]),
            Write(distance_between_centers[::6]),
            distance_text.animate.shift(UP),
            run_time=3,
        )
        self.wait()

        # What happens if two circles are tangent?
        tangent_text = TexText('相切：', color=YELLOW)
        tangent_text.next_to(distance_text, LEFT)

        def get_position_between(mobject1, mobject2):
            return (mobject1.get_right() + mobject2.get_left()) / 2
        text_equal_sign = Tex('=', color=YELLOW)
        text_equal_sign.move_to(get_position_between(distance_text, radii_sum_text))
        formula_equal_sign = Tex('=')
        formula_equal_sign.move_to(get_position_between(distance_between_centers, sum_of_radii))
        addons = VGroup(tangent_text, text_equal_sign, formula_equal_sign)
        self.play(
            AnimationGroup(*[GrowFromCenter(mob) for mob in addons]),
            c1.animate.shift(0.7 * RIGHT),
            c2.animate.shift(0.7 * LEFT),
        )
        self.wait()

        # Mathematical details
        details_rect = FullScreenFadeRectangle(fill_color=GREY_E, fill_opacity=0.98)
        details_rect.next_to(distance_between_centers, DOWN, coor_mask=Y_COOR_MASK)
        self.play(FadeIn(details_rect))
        self.wait()
        details = VGroup(
            Tex(
                '\\left( \\dfrac{a}{b} - \\dfrac{c}{d} \\right)^2 + \\left( \\dfrac{1}{2b^2} - \\dfrac{1}{2d^2} \\right)^2',
                '=',
                '\\left( \\dfrac{1}{2b^2} + \\dfrac{1}{2d^2} \\right)^2'
            ),
            Tex(
                '\\left( \\dfrac{a}{b} - \\dfrac{c}{d} \\right)^2',
                '=',
                '\\left( \\dfrac{1}{2b^2} + \\dfrac{1}{2d^2} \\right)^2 - \\left( \\dfrac{1}{2b^2} - \\dfrac{1}{2d^2} \\right)^2'
            ),
            Tex(
                '\\dfrac{\\left( ad - bc \\right)^2}{b^2 d^2}',
                '=',
                '4 \\cdot \\dfrac{1}{2b^2} \\cdot \\dfrac{1}{2d^2}'
            ),
            Tex('\\left( ad - bc \\right)^2', '=', '1'),
            Tex('\\left| ad - bc \\right|', '=', '1'),
        )
        details[-1].set_color(YELLOW)
        details.scale(0.7)
        details.arrange(DOWN, index_of_submobject_to_align=1, buff=1)
        details.next_to(details_rect.get_boundary_point(UL), DR)
        self.play(FadeIn(details, lag_ratio=0.5), run_time=3)
        self.wait()
        shortcut_rect = SurroundingRectangle(VGroup(details[1][-1], details[2][-1]), color=MAROON_B)
        shortcut = Tex('(', 'x', ' + ', 'y', ')^2 - (', 'x', ' - ', 'y', ')^2 = 4 ', 'x', 'y')
        shortcut.scale(0.6)
        shortcut.set_color_by_tex_to_color_map({'x': RED, 'y': BLUE})
        shortcut.next_to(shortcut_rect, DOWN, aligned_edge=RIGHT)
        self.play(
            ShowCreation(shortcut_rect),
            Write(shortcut),
            run_time=1,
        )
        self.wait()
        self.play(FadeOut(VGroup(shortcut_rect, shortcut)))
        self.wait()

        # Tangency implies |ad-bc|=1
        result_tangent = VGroup(
            TexText('相切'),
            Tex('\\Downarrow'),
            Tex('\\left| ad - bc \\right|', '=', '1'),
        )
        result_tangent.set_color(YELLOW).arrange(DOWN)
        result_tangent.to_corner(DR, buff=0.7)
        self.play(
            ReplacementTransform(tangent_text[0][:-1].copy(), result_tangent[0][0]),
            GrowFromCenter(result_tangent[1]),
            ReplacementTransform(details[-1].copy(), result_tangent[-1]),
        )
        self.add(result_tangent)
        self.wait()

        # The converse is also true: |ad-bc|=1 implies tangency
        flow_arrow = Arrow(ORIGIN, 5 * UP, color=BLUE)
        flow_arrow.next_to(details, RIGHT)
        self.play(
            GrowArrow(flow_arrow, run_time=1),
            Rotate(result_tangent[1], -PI, run_time=3),
        )
        self.wait()
        self.play(FadeOut(flow_arrow))
        self.wait()

        # Name |ad-bc| as 'discriminant'
        double_arrow = Tex('\\Updownarrow', color=YELLOW)
        double_arrow.move_to(result_tangent[1])
        discriminant_rect = SurroundingRectangle(result_tangent[-1][0], color=BLUE)
        discriminant_text = TexText('判别式', color=BLUE)
        discriminant_text.next_to(discriminant_rect, LEFT)
        self.play(FadeTransform(result_tangent[1], double_arrow))
        result_tangent[1].become(double_arrow)
        self.add(result_tangent)
        self.remove(double_arrow)
        self.wait()
        self.play(ShowCreation(discriminant_rect))
        self.wait()
        self.play(Write(discriminant_text))
        self.wait()
        self.play(FadeOut(VGroup(discriminant_text, discriminant_rect)))
        self.wait()

        # Remove the details and show results
        result_tangent.generate_target()
        result_tangent.target.move_to(ORIGIN).to_edge(UP, buff=0.2)
        self.play(
            FadeOut(
                VGroup(
                    details, distance_between_centers, sum_of_radii,
                    distance_text, radii_sum_text, addons,
                )
            ),
            MoveToTarget(result_tangent),
            run_time=1
        )
        self.play(FadeOut(details_rect))
        self.wait()

        # Same goes for separating and intersecting
        result_separate = VGroup(
            TexText('相离'),
            Tex('\\Updownarrow'),
            Tex('\\left| ad - bc \\right|', '>', '1'),
        )
        result_intersect = VGroup(
            TexText('相交'),
            Tex('\\Updownarrow'),
            Tex('\\left| ad - bc \\right|', '<', '1'),
        )
        for result in (result_separate, result_intersect):
            result.set_color(YELLOW).arrange(DOWN)
            result.move_to(result_tangent)
        result_separate.to_edge(LEFT)
        result_intersect.to_edge(RIGHT)
        self.play(
            c1.animate.shift(0.5 * LEFT),
            c2.animate.shift(0.5 * RIGHT),
            FadeTransform(result_tangent.copy(), result_separate),
        )
        self.wait()
        self.play(
            c1.animate.shift(0.8 * RIGHT),
            c2.animate.shift(0.8 * LEFT),
            FadeTransform(result_tangent.copy(), result_intersect),
        )
        self.wait()

        # Something fishy is going on with the intersecting case
        self.play(FocusOn(result_intersect[-1]))
        self.wait()
        brace = Brace(result_intersect[-1][0])
        integer_remark = TexText('整数')
        full_integer_remark = TexText('(0,1)上的', '整数')
        for remark in (integer_remark, full_integer_remark):
            remark.scale(0.6)
            brace.put_at_tip(remark)
        lines = VGroup(*[
            Line(integer_remark.get_top(), result_intersect[-1][0][ind].get_bottom(), buff=0.2)
            for ind in (1, 2, 4, 5)
        ])
        VGroup(brace, integer_remark, full_integer_remark, lines).set_color(GREY_B)
        self.play(
            AnimationGroup(
                *[GrowFromPoint(line, line.get_start()) for line in lines],
                lag_ratio=0.5,
            ),
            Write(integer_remark)
        )
        self.wait()
        self.play(ReplacementTransform(lines, brace))
        self.wait()
        greater_than_zero = Tex('0<', color=YELLOW)
        greater_than_zero.next_to(result_intersect[-1], LEFT)
        self.play(Write(greater_than_zero))
        self.wait()
        self.play(
            Write(full_integer_remark[0]),
            ReplacementTransform(integer_remark[0], full_integer_remark[1])
        )
        self.add(full_integer_remark)
        self.wait()

        # |ad-bc| has to be an integer between 0 and 1 (exclusive), so it doesn't exist!
        cross = Cross(result_intersect)
        self.play(ShowCreation(cross))
        self.wait()
        self.play(
            FadeOut(VGroup(brace, full_integer_remark, greater_than_zero)),
            result_intersect.animate.set_color(GREY_B).fade(0.8),
            cross.animate.set_color(GREY_B),
        )
        self.wait()

        # Show ford circles once more, when editing the video though....
        pass

        # We also obtained a simple discriminant to determine the relation of two circles.
        # Highlight them once.
        rects = VGroup(*[
            SurroundingRectangle(result, color=YELLOW)
            for result in (result_tangent, result_separate)
        ])
        for rect in rects:
            self.play(
                ShowCreationThenDestruction(rect),
                c1.animate.shift(0.3 * LEFT),
                c2.animate.shift(0.3 * RIGHT),
            )
            self.wait()


class WhatHappenedToTheWeirdAdditionP1(Scene):
    CONFIG = {
        'circle_colors': [RED, BLUE, GREEN],
    }

    def construct(self):
        addition_rule = self.get_farey_addition_formula('a', 'b', 'c', 'd')
        addition_rule.scale(2)
        self.play(FadeIn(addition_rule), run_time=2)
        self.wait()
        self.play(addition_rule.animate.scale(0.5).to_edge(DOWN))
        self.wait()
        self.addition_rule = addition_rule

        # Ford circles, again...
        ff = FordFractal(
            max_denom=50,
            max_zoom_level=100,
            zoom_places=[2 / 9, 7 / 23],
        )
        ff.shift(8 * UP)
        ff.suspend_updating()
        self.play(ff.animate.shift(8 * DOWN), run_time=5)
        ff.resume_updating()
        self.ff = ff
        ff.generate_target()
        self.wait()

        # 0/1 oplus 1/1 equals 1/2
        self.play_farey_addition_animations(0, 1, 1, 1)

        # 1/3 oplus 1/4 equals 2/7
        self.play(ff.zoom_in_on(1 / 4, 7, animate=True), run_time=3)
        self.wait()
        self.play_farey_addition_animations(1, 4, 1, 3)

        # 1/4 oplus 2/7 equals 3/11
        self.play(ff.zoom_in_on(3 / 11, 2, animate=True), run_time=3)
        self.wait()
        self.play_farey_addition_animations(1, 4, 2, 7)

        # 4/15 oplus 3/11 equals 7/26
        self.play(ff.zoom_in_on(7 / 26, 7.5, animate=True), run_time=3)
        self.wait()
        self.play_farey_addition_animations(4, 15, 3, 11)

        # Shift a bit, 3/10 oplus 4/13 equals 7/23
        x_offset = (13 / 43 - 7 / 26) * ff.get_axis().get_unit_size() * LEFT
        ff.suspend_updating()
        self.play(ff.animate.shift(x_offset), run_time=6)
        ff.resume_updating()
        self.wait()
        self.play_farey_addition_animations(3, 10, 4, 13)

    def get_ford_fractal(self):
        return self.ff

    def get_farey_addition_formula(self, a, b, c, d):
        tex = Tex(
            f'\\dfrac{{ {a} }}{{ {b} }}', '\\oplus', f'\\dfrac{{ {c} }}{{ {d} }}',
            '=', f'\\dfrac{{ {a} + {c} }}{{ {b} + {d} }}'
        )
        for ind, color in zip([0, 2, 4], self.circle_colors):
            tex[ind].set_color(color)
        return tex

    def play_farey_addition_animations(self, a, b, c, d):
        """
        Returns a series of animations showing 'a/b oplus c/d equals (a+c)/(b+d)'
        """
        ff = self.get_ford_fractal()
        color1, color2, color3 = self.circle_colors
        pairs = (a, b), (c, d), (a + c, b + d)
        for pair in pairs:
            if not ff.has_circle(*pair):
                ff.add_circle(*pair)
        circle1, circle2, circle3 = circles = VGroup(*[
            ff.get_circle(*pair).get_circle().copy()
            for pair in pairs
        ])
        text1, text2, text3 = texts = VGroup(*[
            ff.get_circle(*pair).get_fraction().copy()
            for pair in pairs
        ])
        for mob in it.chain(circles, texts):
            mob.clear_updaters()
        self.play(
            circle1.animate.set_color(color1).set_fill(opacity=0.2).set_stroke(width=5),
            text1.animate.set_color(color1),
            circle2.animate.set_color(color2).set_fill(opacity=0.2).set_stroke(width=5),
            text2.animate.set_color(color2),
        )
        self.wait()
        formula = self.get_farey_addition_formula(a, b, c, d)
        formula.to_edge(DOWN)
        self.play(
            ReplacementTransform(circle1.copy(), circle3),
            ReplacementTransform(circle2.copy(), circle3),
            ReplacementTransform(text1.copy(), text3),
            ReplacementTransform(text2.copy(), text3),
            circle3.animate.set_color(color3).set_fill(opacity=0.2).set_stroke(width=5),
            text3.animate.set_color(color3),
            FadeTransform(self.addition_rule, formula),
            run_time=2,
        )
        self.wait(3)
        self.play(
            FadeOut(VGroup(circles, texts)),
            FadeTransform(formula, self.addition_rule),
        )
        self.wait()


class OnBezoutsIdentityP1(Scene):
    def construct(self):
        # Relatively prime means gcd is 1
        prime_equiv_text = VGroup(
            Tex('a', '\\text{与}', 'b', '\\text{互素}'),
            Tex('\\Leftrightarrow'),
            Tex('\\text{gcd}', '(', 'a', ',\\,', 'b', ')', '=', '1'),
        )
        VGroup(prime_equiv_text[0][0], prime_equiv_text[-1][2]).set_color(RED)
        VGroup(prime_equiv_text[0][2], prime_equiv_text[-1][4]).set_color(BLUE)
        prime_equiv_text.scale(1.25).arrange(RIGHT, buff=0.5)
        gcd_name_text = prime_equiv_text[-1][0]
        gcd_remark = RemarkText(gcd_name_text, '最大公约数', direction=UP, aligned_edge=ORIGIN, scale_factor=0.5,
                                color=YELLOW)
        for mob in (prime_equiv_text[0], prime_equiv_text[1:]):
            self.play(FadeIn(mob))
            self.wait()
        self.play(Write(gcd_remark), Indicate(gcd_name_text))
        self.wait()

        # For numbers, there's a simple algorithm to compute gcd
        gcd_one_step = Tex('\\text{gcd}', '(', '21', ',\\,', '7', ')', '=', '7')
        gcd_multi_steps = Tex('\\text{gcd}', '(', '66', ',\\,', '26', ')', '=', '2')
        gcd_nums_group = VGroup(gcd_one_step, gcd_multi_steps)
        gcd_nums_group.arrange(RIGHT, buff=3).shift(UP)
        for mob in gcd_nums_group:
            mob[2].set_color(RED)
            mob[4].set_color(BLUE)
        gcd_one_step[-1].set_color(BLUE)
        gcd_multi_steps[-1].set_color(GREEN)
        prime_equiv_group = VGroup(prime_equiv_text, gcd_remark)
        gcd_func_text = prime_equiv_text[-1][:-2]
        self.play(
            FadeOut(prime_equiv_group),
            FadeTransformPieces(gcd_func_text, gcd_one_step[:-2]),
            FadeTransformPieces(gcd_func_text, gcd_multi_steps[:-2]),
            run_time=2,
        )
        self.wait()

        # Euclidean algorithm in action
        number_pairs = [(21, 7), (66, 26)]
        dl_group = VGroup()
        for pair, gcd_num in zip(number_pairs, gcd_nums_group):
            dl = DivisionList(*pair)
            dl.next_to(gcd_num, DOWN, buff=0.5)
            dl_group.add(dl)
        dl_one_step, dl_multi_steps = dl_group
        self.play(*[Write(dl[0]) for dl in dl_group])
        self.wait()

        # One step: gcd is the divisor
        cover_rect_multi = FullScreenFadeRectangle(fill_opacity=0.8).move_to(RIGHT_SIDE)
        self.play(
            FadeIn(cover_rect_multi),
            Indicate(dl_one_step[0].get_remainder_tex()),
        )
        self.wait()
        self.play(
            Write(gcd_one_step[-2]),
            ReplacementTransform(dl_one_step[0].get_divisor_tex().copy()[0], gcd_one_step[-1]),
        )
        self.add(gcd_one_step, dl_one_step)
        self.wait()

        # Multiple steps: gcd is the last division's divisor
        cover_rect_one = FullScreenFadeRectangle(fill_opacity=0.8).move_to(LEFT_SIDE)
        self.play(FadeOut(cover_rect_multi), FadeIn(cover_rect_one))
        self.wait()
        for curr_level in range(1, len(dl_multi_steps)):
            prev_level = curr_level - 1
            prev_form = dl_multi_steps.get_inline_form(prev_level)
            curr_form = dl_multi_steps.get_inline_form(curr_level)
            self.play(
                ReplacementTransform(
                    prev_form.get_divisor_tex().copy(),
                    curr_form.get_dividend_tex(),
                ),
                ReplacementTransform(
                    prev_form.get_remainder_tex().copy(),
                    curr_form.get_divisor_tex(),
                ),
            )
            self.play(
                Write(VGroup(*[curr_form[ind] for ind in (1, 3, 4, 5, 6)]))
            )
            self.wait()
        self.play(
            Write(gcd_multi_steps[-2]),
            ReplacementTransform(dl_multi_steps[-1].get_divisor_tex().copy()[0], gcd_multi_steps[-1]),
        )
        self.add(gcd_multi_steps, dl_multi_steps)
        self.wait()

        # This algorithm has a name
        name_chn = TexText('辗转相除法', color=YELLOW)
        name_chn.scale(1.2).to_edge(UP, buff=0.5)
        name_eng = RemarkText(name_chn, '（欧几里得算法）', RIGHT, DOWN, color=YELLOW)
        self.play(FadeOut(cover_rect_one), Write(name_chn))
        self.wait()
        self.play(Write(name_eng))
        self.wait()

        # Though there's a drawback: quotients are completely lost in the process
        all_quotients = VGroup(*[dl.get_all_quotients() for dl in dl_group])
        all_quotients.save_state()
        self.play(
            *[ShowCreationThenDestructionAround(qs) for qs in all_quotients],
            all_quotients.animate.fade(0.8),
        )
        self.wait()
        self.play(all_quotients.animate.restore())
        self.wait()

        # Multiple-step example: 66 and 26
        fade_mobs = VGroup(name_chn, name_eng, gcd_one_step, dl_one_step)
        self.play(
            FadeOut(fade_mobs, run_time=1),
            ApplyMethod(dl_multi_steps.move_to, 3 * LEFT + UP, run_time=2),
            ApplyMethod(gcd_multi_steps.move_to, 3 * LEFT + 3 * UP, run_time=2),
        )
        self.wait()
        rl_multi_steps = RemainderList(*number_pairs[-1])
        rl_multi_steps.move_to(3 * RIGHT + UP)
        rl_multi_steps.save_state()

        # Transform step by step
        dl_copy = dl_multi_steps.copy()
        self.add(dl_copy)
        dl_multi_steps.save_state()
        dl_multi_steps.fade(0.8)
        old_indices = [[6, 3], [0, 5], [4, 1, 2]]
        new_indices = [[0, 1], [2, 3], [4, 5, 6]]
        run_times = [1.5, 2.5, 2.5]
        for old_ind, new_ind, t in zip(old_indices, new_indices, run_times):
            if isinstance(old_ind, Iterable):
                old_mobs = VGroup(*[form[x] for form in dl_copy for x in old_ind])
                new_mobs = VGroup(*[form[y] for form in rl_multi_steps for y in new_ind])
            else:
                old_mobs = VGroup(*[form[old_ind] for form in dl_copy])
                new_mobs = VGroup(*[form[new_ind] for form in rl_multi_steps])
            self.play(
                ReplacementTransform(old_mobs, new_mobs, path_arc=PI / 3, run_time=t)
            )
            self.wait(0.5)
        self.add(rl_multi_steps)
        self.play(dl_multi_steps.animate.restore())
        self.wait()

        # Start from 2=...
        for line_set in (dl_multi_steps, rl_multi_steps):
            for line in line_set:
                line.save_state()
        dl1, dl2, dl3, dl4 = dl_multi_steps
        rl1, rl2, rl3, rl4 = rl_multi_steps
        prev_lines = [dl1, dl2, dl3, rl1, rl2, rl3]
        linear_comb = rl3.copy()
        self.play(*[line.animate.fade(0.8) for line in prev_lines])
        self.wait()
        gcd_rects = VGroup(*[
            SurroundingRectangle(mob)
            for mob in (dl4.get_divisor_tex(), rl4.get_divisor_tex(), rl3.get_remainder_tex())
        ])
        self.play(*[ShowCreation(rect) for rect in gcd_rects[:-1]])
        self.wait()
        self.play(
            *[ReplacementTransform(rect, gcd_rects[-1]) for rect in gcd_rects[:-1]],
            rl3.animate.restore(),
            *[line.animate.fade(0.8) for line in (dl4, rl4)],
        )
        self.wait()
        linear_comb_pos = 6 * LEFT + DOWN
        linear_comb.next_to(linear_comb_pos, RIGHT, buff=0)
        self.play(
            ReplacementTransform(rl3.copy(), linear_comb),
            rl3.animate.fade(0.8),
            FadeOut(gcd_rects[-1]),
        )
        self.wait()

        # Replace the divisor 12 on the right hand side
        twelve_rects = VGroup(*[
            SurroundingRectangle(mob)
            for mob in (linear_comb.get_divisor_tex(), dl3.get_divisor_tex(), rl2.get_remainder_tex())
        ])
        self.play(
            dl3.animate.restore(),
            *[ShowCreation(rect) for rect in twelve_rects[:-1]],
        )
        self.wait()
        self.play(
            *[ReplacementTransform(rect, twelve_rects[-1]) for rect in twelve_rects[:-1]],
            rl2.animate.restore(),
            dl3.animate.fade(0.8),
        )
        self.wait()
        outer_left_par, outer_right_par = outer_pars = VGroup(Tex('('), Tex(')'))
        outer_left_par.next_to(linear_comb[-2], RIGHT)
        twelve_comp = rl2[2:].copy()
        twelve_comp.generate_target()
        two_comp_step1 = VGroup(
            linear_comb[:-1], outer_left_par, twelve_comp.target, outer_right_par,
        )
        two_comp_step1.arrange(RIGHT).next_to(linear_comb_pos, RIGHT, buff=0)
        self.play(
            FadeTransformPieces(linear_comb.get_divisor_tex()[0], outer_pars),
            rl2.animate.fade(0.8),
            MoveToTarget(twelve_comp),
            FadeOut(twelve_rects),
            run_time=1.5,
        )
        linear_comb[-1].scale(0)
        self.wait()

        # Same goes for 14
        fourteen_rects = VGroup(*[
            SurroundingRectangle(mob)
            for mob in (linear_comb[2], twelve_comp[-1], dl2.get_divisor_tex(), rl1.get_remainder_tex())
        ])
        self.play(
            dl2.animate.restore(),
            *[ShowCreation(rect) for rect in fourteen_rects[:-1]],
        )
        self.wait()
        self.play(
            *[ReplacementTransform(rect, fourteen_rects[-1]) for rect in fourteen_rects[:-1]],
            rl1.animate.restore(),
            dl2.animate.fade(0.8),
        )
        self.wait()
        fourteen_comp1, fourteen_comp2 = [rl1[2:].copy() for _ in range(2)]
        fourteen_tex1 = linear_comb[2]
        fourteen_tex2 = twelve_comp[-1]
        dividend_left_par, dividend_right_par = outer_pars.copy()
        inner_left_par, inner_right_par = outer_pars.copy()
        linear_comb_part1, linear_comb_part2 = linear_comb[:2], linear_comb[3:6]
        twelve_comp_part = twelve_comp[:-1]
        moving_mobs = VGroup(
            fourteen_comp1, fourteen_comp2, linear_comb_part1, linear_comb_part2,
            outer_left_par, outer_right_par, twelve_comp_part,
        )
        for mob in moving_mobs:
            mob.generate_target()
        two_comp_step2 = VGroup(
            linear_comb_part1.target, dividend_left_par, fourteen_comp1.target, dividend_right_par,
            linear_comb_part2.target, outer_left_par.target, twelve_comp_part.target,
            inner_left_par, fourteen_comp2.target, inner_right_par, outer_right_par.target,
        )
        two_comp_step2.arrange(RIGHT).next_to(linear_comb_pos, RIGHT, buff=0)
        self.play(
            FadeTransformPieces(fourteen_tex1[0], VGroup(dividend_left_par, dividend_right_par)),
            FadeTransformPieces(fourteen_tex2[0], VGroup(inner_left_par, inner_right_par)),
            rl1.animate.fade(0.8),
            *[MoveToTarget(mob) for mob in moving_mobs],
            FadeOut(fourteen_rects),
            run_time=2,
        )
        self.wait()

        # Get mobjects in the linear combination result and get the leftmost mobject
        linear_comb_result = VGroup(
            *filter(lambda mob: mob.get_center()[1] < -0.5, it.chain(self.mobjects))
        )
        two_equals_tex = sorted(linear_comb_result.submobjects, key=lambda mob: mob.get_center()[0])[0]

        # Only integers, subtractions and multiplications appear ...
        equal_sign = two_equals_tex[-1]
        only_integers_text = TexText('只有整数', color=YELLOW)
        only_pmm_text = TexText('只有``$-$\'\'和``$\\times$\'\'', color=YELLOW)
        texts = VGroup(only_integers_text, only_pmm_text)
        texts.arrange(RIGHT, buff=1).next_to(linear_comb_result, DOWN, buff=0.5)
        for text in texts:
            self.play(Write(text))
            self.wait()

        # ... which implies a linear combination in the end
        some_comb_text = Tex('=', '\\heartsuit', '\\times', '66', '+', '\\spadesuit', '\\times', '26')
        some_comb_text.next_to(equal_sign, DOWN, aligned_edge=LEFT, buff=0.5)
        a, b = get_bezout_coefficients_strings(*number_pairs[-1])
        true_comb_text = Tex('=', f'{a}', '\\times', '66', '+', f'{b}', '\\times', '26')
        true_comb_text.next_to(some_comb_text.get_left(), RIGHT, buff=0)
        for text in (some_comb_text, true_comb_text):
            text[1::4].set_color(YELLOW)
            text[3].set_color(RED)
            text[-1].set_color(BLUE)
        self.play(
            Write(some_comb_text[0]),
            FadeTransform(texts, some_comb_text[1:])
        )
        self.wait()
        brace = Brace(some_comb_text[1:], DOWN)
        linear_comb_def = TexText('66', '和', '26', '的', '线性组合')
        linear_comb_def.scale(0.75)
        linear_comb_def[0].set_color(RED)
        linear_comb_def[2].set_color(BLUE)
        linear_comb_def[-1].set_color(YELLOW)
        brace.put_at_tip(linear_comb_def)
        self.play(GrowFromCenter(brace), Write(linear_comb_def))
        self.wait()
        self.play(
            VFadeOut(VGroup(brace, linear_comb_def)),
            FadeTransform(some_comb_text, true_comb_text)
        )
        self.wait()

        # That's what we got when keeping the quotients
        comb_multi_steps = Tex('2', '=', f'{a}', '\\times', '66', '+', f'{b}', '\\times', '26')
        comb_multi_steps[0].set_color(two_equals_tex[0].get_color())
        for new_mob, old_mob in zip(comb_multi_steps[2::2], true_comb_text[1::2]):
            new_mob.set_color(old_mob.get_color())
        comb_multi_steps.next_to(rl_multi_steps, UP)
        comb_multi_steps.next_to(gcd_multi_steps, RIGHT, coor_mask=Y_COOR_MASK)
        self.play(
            ReplacementTransform(two_equals_tex[0].copy()[0], comb_multi_steps[0]),
            ReplacementTransform(true_comb_text.copy(), comb_multi_steps[1:]),
            FadeOut(true_comb_text),
            FadeOut(linear_comb_result),
            dl_multi_steps.animate.restore(),
            rl_multi_steps.animate.restore(),
            run_time=2,
        )
        self.wait()

        left_side = VGroup(gcd_multi_steps, dl_multi_steps)
        left_side_rect = SurroundingRectangle(left_side, buff=0.5)
        left_side_text = TexText('欧几里得算法', color=YELLOW)
        left_side_text.next_to(left_side_rect, DOWN)
        left_side_remark = RemarkText(left_side_text, '（求最大公约数）', aligned_edge=ORIGIN, color=YELLOW)
        self.play(
            ShowCreation(left_side_rect),
            Write(left_side_text),
        )
        self.wait()
        self.play(Write(left_side_remark))
        self.wait()

        right_side = VGroup(comb_multi_steps, rl_multi_steps)
        right_side_rect = SurroundingRectangle(right_side, buff=0.5)
        right_side_text = TexText('后续步骤', color=YELLOW)
        right_side_text.next_to(right_side_rect, DOWN)
        right_side_remark = RemarkText(right_side_text, '（求组合方式）', aligned_edge=ORIGIN, color=YELLOW)
        self.play(
            ShowCreation(right_side_rect),
            Write(right_side_text),
        )
        self.wait()
        self.play(Write(right_side_remark))
        self.wait()

        placeholder = Rectangle(width=12, height=2.5)
        placeholder.shift(DOWN)
        full_procedure_rect = SurroundingRectangle(
            VGroup(left_side, right_side, placeholder),
            buff=0.5,
        )
        full_procedure_text = TexText('扩展', '欧几里得算法', color=YELLOW)
        full_procedure_text.next_to(full_procedure_rect, DOWN)
        full_procedure_abbrev = RemarkText(full_procedure_text, '（扩欧算法）', direction=RIGHT, aligned_edge=DOWN,
                                           color=YELLOW)
        self.play(
            ReplacementTransform(left_side_rect, full_procedure_rect),
            ReplacementTransform(right_side_rect, full_procedure_rect),
            ReplacementTransform(
                left_side_text[0], full_procedure_text[1],
                rate_func=squish_rate_func(smooth, 0, 0.7),
                run_time=2,
            ),
            FadeTransform(
                right_side_text, full_procedure_text[0],
                rate_func=squish_rate_func(smooth, 0.3, 1),
                run_time=2,
            ),

            FadeOut(VGroup(left_side_remark, right_side_remark), run_time=0.5),
        )
        self.wait()
        self.play(Write(full_procedure_abbrev))
        self.wait()

        # Few examples should help
        example_pairs = [(50, 9), (21, 7), (98, 21), (570, 393)]
        (gcd_eq, comb_eq, dl, rl) = (gcd_multi_steps, comb_multi_steps, dl_multi_steps, rl_multi_steps)
        for pair in example_pairs:
            p, q = pair
            d = math.gcd(p, q)
            a, b = get_bezout_coefficients_strings(p, q)
            new_dl = DivisionList(p, q)
            new_rl = RemainderList(p, q)
            d_color = new_dl[-1].get_divisor_tex().get_color()
            new_gcd_eq = Tex('\\text{gcd}', '(', f'{p}', ',\\,', f'{q}', ')', '=', f'{d}')
            new_comb_eq = Tex(f'{d}', '=', f'{a}', '\\times', f'{p}', '+', f'{b}', '\\times', f'{q}')
            for p_tex in (new_gcd_eq[2], new_comb_eq[4]):
                p_tex.set_color(RED)
            for q_tex in (new_gcd_eq[4], new_comb_eq[-1]):
                q_tex.set_color(BLUE)
            for d_tex in (new_gcd_eq[-1], new_comb_eq[0]):
                d_tex.set_color(d_color)
            new_comb_eq[2::4].set_color(YELLOW)
            for new_eq, old_eq in zip([new_gcd_eq, new_comb_eq], [gcd_eq, comb_eq]):
                new_eq.move_to(old_eq)
            for new_list, old_list in zip([new_dl, new_rl], [dl, rl]):
                new_list.next_to(old_list.get_top(), DOWN, buff=0)
            old_ee_group = VGroup(gcd_eq, comb_eq, dl, rl)
            new_ee_group = VGroup(new_gcd_eq, new_comb_eq, new_dl, new_rl)
            self.play(
                *[
                    FadeTransform(old_mob, new_mob)
                    for old_mob, new_mob in zip(old_ee_group, new_ee_group)
                ],
                run_time=1
            )
            self.wait(3)
            (gcd_eq, comb_eq, dl, rl) = (new_gcd_eq, new_comb_eq, new_dl, new_rl)

        # Finally, Bezout's identity!
        all_mobs = VGroup(
            gcd_eq, comb_eq, dl, rl, full_procedure_rect,
            full_procedure_text, full_procedure_abbrev,
        )
        bezout_text = TexText('裴蜀定理', color=YELLOW)
        bezout_text.scale(1.2).to_edge(UP)
        bezout_text_remark = RemarkText(bezout_text, "(B\\'ezout's Lemma)", color=YELLOW)
        all_mobs.generate_target()
        all_mobs.target.match_height(bezout_text).move_to(bezout_text).fade(1)
        self.play(
            FadeIn(
                VGroup(bezout_text, bezout_text_remark),
                rate_func=squish_rate_func(smooth, 0.3, 1),
            ),
            MoveToTarget(all_mobs, rate_func=squish_rate_func(smooth, 0, 0.7)),
            run_time=2,
        )
        self.remove(all_mobs)
        self.wait()
        bezout_content = VGroup(
            TexText('考虑整数', '$x$', ', ', '$y$', '以及它们的最大公约数', '$d$', ', 如下两条成立:'),
            TexText('(1) ', '$x$', '和', '$y$', '的某个线性组合等于', '$d$', ';'),
            TexText('(2) ', '$x$', '和', '$y$', '的所有线性组合就是', '$d$', '的所有整数倍.'),
        )
        bezout_content.arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        bezout_content.next_to(bezout_text_remark, DOWN, buff=0.5)
        bezout_content.to_edge(LEFT, buff=1)
        for line in bezout_content:
            line[1].set_color(RED)
            line[3].set_color(BLUE)
            line[5].set_color(MAROON_B)
            self.play(Write(line), run_time=2)
            self.wait()

        # Proof to show
        proof_text = TexText('证:')
        proof_content = VGroup(
            TexText('由扩欧算法, 总是可以找到一组$(a,\\,b) \\in (\\mathbb{Z},\\,\\mathbb{Z})$使得$ax+by=d$, (1)成立.'),
            TexText('$x$和$y$都是$d$的整数倍, 所以$x$和$y$的线性组合是$d$的整数倍.'),
            TexText('而$kd=(ka)x+(kb)y\\,\\,(k \\in \\mathbb{Z})$, 所以$d$的整数倍也是$x$和$y$的线性组合.'),
            TexText('因此$\\{ \\text{$x$和$y$的线性组合} \\} = \\{ \\text{$d$的整数倍} \\}$, (2)成立.')
        )
        qed = Square(side_length=0.25, fill_opacity=1)
        proof_content.arrange(DOWN, aligned_edge=LEFT)
        proof_group = VGroup(proof_text, proof_content)
        proof_group.arrange(RIGHT, aligned_edge=UP, buff=0.3).scale(0.7)
        qed.next_to(proof_content, RIGHT, aligned_edge=DOWN)
        proof_group.add(qed)
        proof_group.move_to(2 * DOWN).to_edge(LEFT, buff=1)
        proof_group.set_color(GREY_B)
        self.play(FadeIn(proof_group))
        self.wait()

        # Fade out everthing
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)
        self.wait()


class BezoutsIdentityAndCoprimeP1(Scene):
    def construct(self):
        # A cycle
        text_positions = [3.5 * LEFT, 2.5 * DOWN, 3.5 * RIGHT, 2.5 * UP]
        arrow_positions = [
            (text_positions[k] + text_positions[(k + 1) % 4]) / 2
            for k in range(4)
        ]
        arrow_angles = [-PI / 4 + PI / 2 * k for k in range(4)]
        texts = VGroup(*[
            Tex(*tex_string).move_to(text_pos)
            for tex_string, text_pos in zip(
                [
                    ('x', '\\text{与}', 'y', '\\text{互素}'),
                    ('\\text{gcd}(', 'x', ',\\,', 'y', ')=1'),
                    ('a', 'x', '+', 'b', 'y', '=1'),
                    ('\\text{gcd}(', 'x', ',\\,', 'y', ')\\text{是1的因子}'),
                ],
                text_positions,
            )
        ])
        for text, ind_x, ind_y in zip(texts, [0, 1, 1, 1], [2, 3, 4, 3]):
            text[ind_x].set_color(RED)
            text[ind_y].set_color(BLUE)
        arrows = VGroup(*[
            Tex('\\Rightarrow', color=YELLOW).scale(2).rotate(angle).move_to(arrow_pos)
            for angle, arrow_pos in zip(arrow_angles, arrow_positions)
        ])
        self.play(FadeIn(texts[0]), run_time=0.5)
        self.wait()
        for k in range(4):
            anims = [FadeInFromPoint(arrows[k], texts[k].get_center())]
            if k <= 2:
                anims.append(FadeInFromPoint(texts[(k + 1) % 4], texts[k].get_center()))
            self.play(*anims)
            self.wait()
            if k == 2:
                text_alt = texts[1].copy()
                text_alt.move_to(texts[-1])
                self.play(FadeTransform(texts[-1], text_alt))
                self.wait()

        # A new way to determine co-primality (...?)
        texts = VGroup(*texts[:-1], text_alt)
        prime_text, comb_text = texts[::2]
        double_arrow = Tex('\\Leftrightarrow').scale(1.5)
        self.play(
            prime_text.animate.next_to(double_arrow, LEFT, buff=0.5),
            comb_text.animate.next_to(double_arrow, RIGHT, buff=0.5),
            GrowFromCenter(double_arrow, rate_func=squish_rate_func(smooth, 0, 0.5)),
            FadeOut(VGroup(texts[1], arrows[:2]), shift=DOWN),
            FadeOut(VGroup(texts[-1], arrows[2:]), shift=UP),
            run_time=2
        )
        self.wait()
        conclusion = VGroup(prime_text, double_arrow, comb_text)
        self.play(conclusion.animate.scale(1.2).to_edge(UP, buff=0.8))
        self.wait()

        underline = Underline(comb_text, color=YELLOW)
        relation_text = TexText('合适的关系')
        relation_text.set_color(YELLOW)
        relation_text.next_to(underline, DOWN, buff=0.5)

        self.play(ShowCreation(underline))
        self.wait()
        self.play(Write(relation_text))
        self.wait()
        self.play(FadeOut(VGroup(underline, relation_text)))
        self.wait()

        # 1st question of 1st IMO (1959) as an example
        frac = Tex('{21n+4', '\\over', '14n+3}', '\\,(n\\text{为自然数})')
        frac.scale(1.2)
        frac[0].set_color(RED)
        frac[2].set_color(BLUE)
        frac_relation = Tex('(-2) \\times (', '21n+4', ')+3 \\times (', '14n+3', ')=1')
        frac_relation[1].set_color(RED)
        frac_relation[3].set_color(BLUE)
        frac_relation.next_to(frac, DOWN, buff=0.8)
        simplest_rect = SurroundingRectangle(frac, buff=0.5)
        simplest_text = TexText('最简分数', color=YELLOW)
        simplest_text.next_to(simplest_rect, DOWN)
        self.play(Write(frac))
        self.wait()
        self.play(
            *[Write(frac_relation[::2])],
            ReplacementTransform(
                frac[0].copy(), frac_relation[1],
                rate_func=squish_rate_func(smooth, 0, 0.4)
            ),
            ReplacementTransform(
                frac[2].copy(), frac_relation[3],
                rate_func=squish_rate_func(smooth, 0.2, 0.6)
            ),
            run_time=3,
        )
        self.add(frac_relation)
        self.wait()
        self.play(
            ShowCreation(simplest_rect),
            FadeTransform(frac_relation, simplest_text),
        )
        self.wait()


class ConjecturesAndProofsAboutFareyMediantP1(WhatHappenedToTheWeirdAdditionP1):
    def construct(self):
        # Match the main scene
        addition_rule = self.get_farey_addition_formula('a', 'b', 'c', 'd')
        addition_rule.to_edge(DOWN)
        self.add(addition_rule)
        self.wait()

        # Show terms
        farey_sum_text = TexText('法里和：')
        farey_sum_text.next_to(addition_rule, LEFT)
        farey_mediant = TexText('法里中项', color=self.circle_colors[-1])
        farey_mediant.scale(0.8)
        farey_mediant.next_to(addition_rule, RIGHT, buff=0.5)
        for text in (farey_sum_text, farey_mediant):
            self.play(Write(text))
            self.wait()
        self.play(FadeOut(farey_mediant))
        self.wait()

        # Show conjecture with a static example
        axis = Line(LEFT_SIDE, RIGHT_SIDE, color=GREY_A, stroke_width=2)
        axis.shift(2 * DOWN)
        x1 = -0.6
        x2 = 3.4
        r1 = 1.6
        r2 = (x2 - x1)**2 / (4 * r1)
        r3 = (1 / (1 / np.sqrt(r1) + 1 / np.sqrt(r2)))**2
        x3 = 2 * np.sqrt(r1 * r3) + x1
        xr_pairs = [(x1, r1), (x2, r2), (x3, r3)]
        fracs = ['\\dfrac{a}{b}', '\\dfrac{c}{d}', '\\dfrac{a+c}{b+d}']
        example_circles = VGroup()
        for (x, r), color, frac in zip(xr_pairs, self.circle_colors, fracs):
            circle = FineCircle(radius=r, color=color, fill_opacity=0.2)
            circle.move_to(x * RIGHT).next_to(axis, UP, buff=0, coor_mask=Y_COOR_MASK)
            text = Tex(frac, color=color)
            text.set_height(circle.get_height() * 0.5).move_to(circle.get_center())
            example_circles.add(VGroup(circle, text))
        self.play(FadeIn(axis))
        self.wait()
        self.play(*[GrowFromCenter(mob) for mob in example_circles[:-1]])
        self.wait()
        self.play(GrowFromCenter(example_circles[-1]))
        self.add(example_circles)
        self.wait()

        # There are 4 conditions that must be satisfied
        BulletedList()
        conditions = VGroup(
            Tex('\\text{1. }', f'{fracs[0]}', '<', f'{fracs[2]}', '<', f'{fracs[1]}'),
            Tex('\\text{2. }', f'\\text{{圆}}{fracs[2]}', '\\text{与}', f'\\text{{圆}}{fracs[0]}', '\\text{相切}'),
            Tex('\\text{3. }', f'\\text{{圆}}{fracs[2]}', '\\text{与}', f'\\text{{圆}}{fracs[1]}', '\\text{相切}'),
            Tex('\\text{4. }', f'{fracs[2]}', '\\text{是最简分数}'),
        )
        cond4_alt = Tex('\\text{4. }', '({a+c})', '\\text{与}', '({b+d})', '\\text{互素}')
        for mob_list in (conditions, VGroup(cond4_alt)):
            for cond in mob_list:
                for tex in cond:
                    if '{b}' in tex.tex_string:
                        tex.set_color(self.circle_colors[0])
                    elif '{d}' in tex.tex_string:
                        tex.set_color(self.circle_colors[1])
                    elif '{a+c}' in tex.tex_string or '{b+d}' in tex.tex_string:
                        tex.set_color(self.circle_colors[2])
            mob_list.scale(0.7)
        conditions.arrange(DOWN, aligned_edge=LEFT).to_corner(UL)
        cond1, cond2, cond3, cond4 = conditions
        circle_ab, circle_cd, circle_mediant = example_circles
        cond4_alt.next_to(cond3, DOWN, aligned_edge=LEFT)
        self.play(
            Indicate(circle_mediant, scale_factor=1),
            Write(cond1),
            run_time=2,
        )
        self.wait()
        self.play(
            Indicate(circle_ab, scale_factor=1),
            Indicate(circle_mediant, scale_factor=1),
            Write(cond2),
            run_time=2,
        )
        self.wait()
        self.play(
            Indicate(circle_cd, scale_factor=1),
            Indicate(circle_mediant, scale_factor=1),
            Write(cond3),
            run_time=2,
        )
        self.wait()
        self.play(
            Indicate(circle_mediant, scale_factor=1),
            Write(cond4),
            run_time=2,
        )
        self.wait()
        self.play(FadeTransform(cond4, cond4_alt))
        conditions = VGroup(cond1, cond2, cond3, cond4_alt)
        self.add(conditions)
        self.wait()

        # Further explanation about the last condition in another scene
        # i.e. Bezout's identity
        last_cond_rect = SurroundingRectangle(cond4_alt)
        self.play(ShowCreationThenDestruction(last_cond_rect), run_time=2)
        self.wait()

        # Start the proof
        pause_button = PauseButton()
        fade_rect = FullScreenFadeRectangle()
        self.play(VFadeIn(pause_button), FadeIn(fade_rect))
        self.wait()
        self.play(VFadeOut(pause_button), FadeOut(fade_rect))
        self.wait()
        other_mobs = VGroup(example_circles, axis, addition_rule, farey_sum_text)
        self.play(FadeOut(other_mobs))
        self.wait()

        # Property-proved ticks
        tick1, tick2, tick3, tick4 = ticks = VGroup(*[
            TexText('\\ding{51}', color=YELLOW)
            for _ in range(4)
        ])
        for tick, cond in zip(ticks, conditions):
            tick.scale(0.75)
            tick.next_to(cond, LEFT, buff=0.1)
            tick.save_state()

        # Property 1
        cond1_copy = cond1[1:].copy()
        conditions.save_state()
        self.play(conditions[1:].animate.fade(0.8))
        self.wait()
        plane = NumberPlane(
            x_range=[0, 8], y_range=[0, 6],
            background_line_style={
                "stroke_color": GREY_D,
                "stroke_width": 2,
                "stroke_opacity": 1,
            },
        )
        plane.next_to(conditions, RIGHT, coor_mask=X_COOR_MASK, buff=1)
        self.play(ShowCreation(plane))
        self.wait()
        pairs = [(5, 1), (2, 4), (7, 5)]
        coord_texts = ['(b,\\,a)', '(d,\\,c)', '(b+d,\\,a+c)']
        arrow_ab, arrow_cd, arrow_mediant = arrows = VGroup(*[
            plane.get_vector(pair, color=color)
            for pair, color in zip(pairs, self.circle_colors)
        ])
        coord_ab, coord_cd, coord_mediant = coords = VGroup(*[
            Tex(text, color=color).scale(0.7).next_to(plane.c2p(*pair), direction)
            for text, color, pair, direction in zip(
                coord_texts, self.circle_colors, pairs, (DR, UL, UP)
            )
        ])
        self.play(
            GrowArrow(arrow_ab), GrowArrow(arrow_cd),
            Write(coord_ab), Write(coord_cd),
        )
        self.wait()
        arrow_ab_copy, arrow_cd_copy = arrows[:-1].copy().fade(0.5)
        self.play(
            arrow_ab_copy.animate.shift(arrow_cd.get_vector()),
            arrow_cd_copy.animate.shift(arrow_ab.get_vector()),
        )
        self.wait()
        self.play(GrowArrow(arrow_mediant), Write(coord_mediant))
        self.wait()
        slope_text = TexText('斜率关系')
        cond1_copy.next_to(plane, LEFT, aligned_edge=DOWN, buff=0.8)
        slope_text.next_to(cond1_copy, UP)
        self.play(Write(slope_text), Write(cond1_copy))
        self.wait()
        cond1_group = VGroup(
            slope_text, cond1_copy, plane, arrows, coords,
            arrow_ab_copy, arrow_cd_copy,
        )
        self.play(
            FadeOut(cond1_group),
            conditions.animate.restore(),
            Write(tick1),
        )
        self.wait()

        # Property 2-3
        self.play(VGroup(conditions[0], tick1).animate.fade(0.8))
        self.wait()
        cond2_arrow = Tex('\\Leftrightarrow')
        cond2_alt = Tex('|b(a+c)-a(b+d)|', '=1')
        cond3_arrow = Tex('\\Leftrightarrow')
        cond3_alt = Tex('|d(a+c)-c(b+d)|', '=1')
        cond23_group = VGroup(cond2_arrow, cond2_alt, cond3_arrow, cond3_alt)
        for mob in cond23_group:
            mob.scale(0.8)
        cond2_arrow.next_to(cond2, RIGHT, buff=0.5)
        cond2_alt.next_to(cond2_arrow, RIGHT, buff=0.5)
        cond3_arrow.next_to(cond3, RIGHT, buff=0.5)
        cond3_alt.next_to(cond3_arrow, RIGHT, buff=0.5)
        self.play(
            *[GrowFromCenter(arrow) for arrow in (cond2_arrow, cond3_arrow)],
            *[Write(alt) for alt in (cond2_alt, cond3_alt)],
        )
        self.wait()

        cond2_proof = Tex(
            '  & |', 'b', '(a+c)', '-', 'a', '(b+d)', '| \\\\',
            '= & |ab+bc-ab-ad| \\\\',
            '= & |ad-bc| \\\\',
            '= & 1'
        )
        cond3_proof = Tex(
            '  & |', 'd', '(a+c)', '-', 'c', '(b+d)', '| \\\\',
            '= & |ad+cd-bc-cd| \\\\',
            '= & |ad-bc| \\\\',
            '= & 1'
        )
        cond23_proofs = VGroup(cond2_proof, cond3_proof)
        cond23_proofs.scale(0.9)
        cond23_proofs.arrange(RIGHT, buff=1).shift(2 * DOWN)
        for mob in (cond2_proof[:7], cond2_proof[7], cond2_proof[8]):
            self.play(Write(mob))
            self.wait()
        self.play(FadeTransform(cond2_proof[:-1].copy(), cond3_proof[:-1]))
        self.wait()
        self.play(*[Indicate(proof[-2]) for proof in (cond2_proof, cond3_proof)])
        self.wait()
        self.play(Write(VGroup(cond2_proof[-1], cond3_proof[-1])))
        self.wait()
        self.play(Write(tick2), Write(tick3))

        # Property 4
        numbers = cond2_proof[2:6:3]
        coeffs = VGroup(cond2_proof[1], cond2_proof[3:5])
        self.play(
            numbers.animate.set_color(GREEN),
            cond2_proof[7:-1].animate.fade(0.8),
            cond3_proof.animate.fade(0.8),
        )
        self.wait()
        self.play(coeffs.animate.set_color(YELLOW))
        self.wait()
        self.play(*[Indicate(coeff) for coeff in coeffs])
        self.wait()
        self.play(Indicate(cond2_proof[-1]))
        self.wait()
        self.play(Write(tick4))
        self.wait()
        cond234_group = VGroup(cond23_group, cond23_proofs)
        self.play(
            FadeOut(cond234_group),
            conditions.animate.restore(),
            tick1.animate.restore(),
        )
        self.add(ticks)
        self.wait()
        self.play(
            conditions.animate.restore(),
            tick1.animate.restore(),
        )
        self.wait()
        self.play(FadeIn(other_mobs))
        self.wait()


class FewWordsBeforeFordSpheresP1(WhatHappenedToTheWeirdAdditionP1):
    def construct(self):
        # Match the last scene
        addition_rule = self.get_farey_addition_formula('a', 'b', 'c', 'd')
        addition_rule.to_edge(DOWN)
        farey_sum_text = TexText('法里和：')
        farey_sum_text.next_to(addition_rule, LEFT)
        self.add(addition_rule, farey_sum_text)
        self.wait()

        # Conclusions in this video
        ff = FordFractal(
            max_denom=50,
            max_zoom_level=108,
            zoom_places=[(np.sqrt(5) - 1) / 2],
        )
        self.add(ff)
        self.wait()
        bg_rect = FullScreenFadeRectangle(fill_opacity=0.9)
        bg_rect.stretch_to_fit_height(FRAME_HEIGHT / 4).to_edge(UP, buff=0)
        conclusions = VGroup(
            Tex('|', 'a', 'd', '-', 'b', 'c', '|=1'),
            Tex('\\Leftrightarrow'),
            Tex('\\text{圆}{a \\over b}', '\\text{与}', '\\text{圆}{c \\over d}', '\\text{相切}'),
            Tex('\\Rightarrow'),
            Tex(
                '\\text{圆}{a+c \\over b+d}', '\\text{与}',
                '\\text{圆}{a \\over b}', ', ', '\\text{圆}{c \\over d}',
                '\\text{都相切}'
            ),
        )
        conclusions.arrange(RIGHT, buff=0.5).set_width(FRAME_WIDTH - 2)
        conclusions.move_to(bg_rect)
        colored_mobs = [
            VGroup(conclusions[0][1], conclusions[0][4], conclusions[2][0], conclusions[4][2]),
            VGroup(conclusions[0][2], conclusions[0][5], conclusions[2][2], conclusions[4][4]),
            VGroup(conclusions[4][0]),
        ]
        for mob, color in zip(colored_mobs, self.circle_colors):
            mob.set_color(color)
        self.play(FadeIn(bg_rect))
        self.wait()
        self.play(FadeIn(conclusions[:-2]))
        self.wait()
        self.play(FadeIn(conclusions[-2:]))
        self.wait()

        # Ford circle animations
        point = ff.zoom_places[0]
        run_time = 2
        wait_time = 1
        for zoom_factor in (4, 3, 3, 2):
            self.play(
                ff.zoom_in_on(point, zoom_factor, animate=True),
                Animation(VGroup(bg_rect, conclusions)),
                run_time=run_time,
            )
            self.wait(wait_time)

        # 2 applications of these ideas in the next video
        self.add(conclusions)
        self.play(bg_rect.animate.stretch_to_fit_height(FRAME_HEIGHT).center())
        self.wait()
        app_text1, app_text2 = app_texts = VGroup(
            TexText('有理数近似'),
            TexText('无理数近似'),
        )
        frame1, frame2 = frames = VGroup(*[
            PictureInPictureFrame().set_height(2.5)
            for _ in range(2)
        ])
        frames.arrange(RIGHT, buff=0.5).shift(DOWN)
        for app_text, frame in zip(app_texts, frames):
            app_text.next_to(frame, UP)
        frame2_content = Tex(
            '\\left| {p \\over q} - \\alpha \\right| < {1 \\over 2q^2}',
            color=MAROON_B,
        )
        frame2_content.scale(1.2).move_to(frame2)
        group1 = VGroup(app_text1, frame1)
        group2 = VGroup(app_text2, VGroup(frame2, frame2_content))
        self.play(
            Write(group1),
            Write(group2),
            run_time=2,
        )
        self.wait()

        ford_spheres_frame = PictureInPictureFrame().set_height(5.5)
        ford_spheres_frame.move_to(VGroup(group1, group2))
        ford_spheres_frame.fade(0.8)
        self.play(FadeIn(ford_spheres_frame))
        self.wait()


class CrucialPropertiesForConstructingP1(Scene):
    def construct(self):
        # Definition of Ford circles
        ford_circles_def = VGroup(
            TexText(
                '对于', '最简分数', '$\\dfrac{p}{q}$',
                ', 在', '数轴', '$\\dfrac{p}{q}$',
                '的位置上放置半径为', '$\\dfrac{1}{2q^2}$', '的',
                '圆', ',',
            ),
            TexText('再将所有的', '圆', '收集起来即可.'),
        )
        ford_circles_def.arrange(DOWN, aligned_edge=LEFT)
        self.play(Write(ford_circles_def), run_time=5)
        self.wait()

        # Looks simple, while it's not.
        simplest_text = ford_circles_def[0][1:3]
        self.play(
            ShowCreationThenDestructionAround(simplest_text),
            simplest_text.animate.set_color(YELLOW),
        )
        self.wait()

        full_screen_rect = FullScreenFadeRectangle(fill_opacity=1)
        foa_text = TexText('算术基本定理', color=YELLOW)
        foa_text.next_to(simplest_text, UP, buff=2)
        foa_content = RemarkText(
            foa_text,
            '每个大于1的整数都能分解为有限个素数的乘积，且分解方式唯一。',
            direction=DOWN, aligned_edge=LEFT,
            scale_factor=0.7, buff=0.5, color=GOLD_A
        )
        self.play(
            FadeIn(full_screen_rect),
            FadeInFromPoint(foa_text, simplest_text.get_center()),
        )
        self.wait()

        # Two examples for unique factorization.
        factor11 = Tex('45', '= 3^2 \\times 5')
        factor12 = Tex('77', '= 7 \\times 11')
        gcd1 = Tex('\\text{gcd}(', '45', ',\\,', '77', ')=1')
        coprime1 = Tex('45', '\\text{与}', '77', '\\text{互素}')
        simp_frac1 = Tex('{45', ' \\over ', '77}', '\\text{是最简分数}')
        example_group1 = VGroup(factor11, factor12, gcd1, coprime1, simp_frac1)

        factor21 = Tex('570', '= 2 \\times 3 \\times 5 \\times 19')
        factor22 = Tex('323', '= 17 \\times 19')
        gcd2 = Tex('\\text{gcd}(', '570', ',\\,', '323', ')=19')
        coprime2 = Tex('570', '\\text{与}', '323', '\\text{不互素}')
        simp_frac2 = Tex('{570', ' \\over ', '323}', '\\text{不是最简分数}')
        example_group2 = VGroup(factor21, factor22, gcd2, coprime2, simp_frac2)

        for group in (example_group1, example_group2):
            f1, f2, g, c, s = group
            f1[0].set_color(RED)
            f2[0].set_color(BLUE)
            for mob, ind_a, ind_b in zip([g, c, s], [1, 0, 0], [3, 2, 2]):
                mob[ind_a].set_color(RED)
                mob[ind_b].set_color(BLUE)
            VGroup(f1, f2).arrange(DOWN, aligned_edge=LEFT)
            VGroup(g, c, s).arrange(DOWN, aligned_edge=LEFT)\
                           .next_to(f2, DOWN, aligned_edge=LEFT, buff=1)
        VGroup(example_group1, example_group2)\
            .arrange(RIGHT, buff=3)\
            .shift(DOWN)\
            .scale(0.9)
        self.play(
            FadeIn(foa_content, shift=0.5 * DOWN),
            run_time=1,
        )
        self.wait()
        for mobs in zip(example_group1[:2], example_group2[:2]):
            self.play(*[Write(factor) for factor in mobs], run_time=1)
            self.wait()
        for mobs in zip(example_group1[2:], example_group2[2:]):
            self.play(*[FadeIn(mob, shift=0.5 * DOWN) for mob in mobs], run_time=1)
            self.wait()

        # Another property is division algorithm, which help simplify any given fraction
        div_text = TexText('带余除法', color=YELLOW)
        div_text.next_to(foa_text, DOWN, buff=1.5, aligned_edge=LEFT)
        div_content = RemarkText(
            div_text, '衍生出辗转相除法，快速求最大公约数并化简分数。',
            direction=DOWN, aligned_edge=LEFT,
            scale_factor=0.7, buff=0.5, color=GOLD_A
        )
        self.play(
            FadeOut(example_group1),
            FadeOut(example_group2[:-1]),
            simp_frac2.animate.next_to(gcd2.get_top(), DOWN, buff=0),
        )
        self.wait()

        question_frac2 = Tex('{???', ' \\over ', '???}', '\\text{是最简分数}')
        result_frac2 = Tex('{30', ' \\over ', '17}', '\\text{是最简分数}')
        for frac in (question_frac2, result_frac2):
            frac.next_to(simp_frac2, DOWN, buff=0.5, aligned_edge=LEFT)
            frac[0].set_color(RED)
            frac[2].set_color(BLUE)
        dl2 = DivisionList(570, 323)
        dl2.next_to(result_frac2, LEFT, aligned_edge=DOWN, buff=1)
        self.play(FadeIn(question_frac2, shift=0.5 * DOWN))
        self.wait()
        self.play(Write(div_text))
        self.wait()
        self.play(
            FadeIn(div_content, shift=0.5 * DOWN),
            run_time=1,
        )
        self.wait()
        self.play(
            AnimationGroup(*[FadeIn(line) for line in dl2], lag_ratio=0.2),
            run_time=3,
        )
        self.wait()
        self.play(FadeTransform(question_frac2, result_frac2))
        self.wait()

        # Are there any other structures have these two properties?
        property_rect = SurroundingRectangle(
            VGroup(foa_text, foa_content, div_text, div_content),
            buff=0.5
        )
        self.play(ShowCreation(property_rect))
        self.wait()


class GaussianIntegersIntroP1(Scene):
    def construct(self):
        # Definition of Gaussian integers
        gaussian_text = TexText('高斯整数', color=YELLOW)
        gaussian_def = Tex('a', '+', 'b', 'i')
        gaussian_def.next_to(gaussian_text, DOWN)
        int_remark = RemarkText(
            gaussian_def[:-1], '整数',
            direction=DOWN, aligned_edge=ORIGIN,
            scale_factor=0.8, buff=0.5, color=GREY,
        )
        lines = VGroup(*[
            Line(mob.get_bottom(), int_remark.get_top(), color=GREY, buff=0.1)
            for mob in gaussian_def[::2]
        ])
        for mob in (gaussian_text, gaussian_def):
            self.play(Write(mob))
            self.wait()
        self.play(ShowCreation(lines), Write(int_remark))
        self.wait()

        # They are grid points with integer coordinates
        gaussian_group = VGroup(gaussian_text, gaussian_def, lines, int_remark)
        gaussian_rect = BackgroundRectangle(gaussian_group, fill_opacity=0.9)
        gaussian_rect.set_height(3, stretch=True).set_width(5, stretch=True)
        gaussian_rect.to_corner(UL, buff=0)
        gaussian_group.generate_target()
        gaussian_group.target.next_to(gaussian_rect.get_bounding_box_point(UL), DR, buff=0.5)
        complex_plane = ComplexPlane(faded_line_ratio=0)
        complex_plane.add_coordinate_labels()
        self.play(
            FadeIn(complex_plane, run_time=2),
            FadeIn(gaussian_rect),
            MoveToTarget(gaussian_group),
        )
        gaussian_group = VGroup(gaussian_rect, *gaussian_group.submobjects)
        self.wait()
        grid_coords_raw = [(x, y) for x in range(-8, 9) for y in range(-5, 6)]
        grid_dict = {}
        for (x, y) in grid_coords_raw:
            norm = x**2 + y**2
            if norm in grid_dict:
                grid_dict[norm].append((x, y))
            else:
                grid_dict[norm] = [(x, y)]
        grid_coords_grouped = list(grid_dict.values())
        grid_coords_grouped.sort(key=lambda l: sum(map(lambda x: x**2, l[0])))
        grid_dots = VGroup(*[
            VGroup(*[
                Dot(color=YELLOW).move_to(complex_plane.c2p(x, y))
                for (x, y) in coords
            ])
            for coords in grid_coords_grouped
        ])
        self.play(
            AnimationGroup(
                *[
                    DrawBorderThenFill(grouped_dot, lag_ratio=0, stroke_color=PINK)
                    for grouped_dot in grid_dots
                ],
                Animation(gaussian_group),
                lag_ratio=0.1,
            ),

            run_time=3,
        )
        self.wait()
        gaussian_zi = Tex('\\mathbb{Z} ', '\\big[', ' i ', '\\big]', color=YELLOW)
        gaussian_zsqrt1 = Tex('\\mathbb{Z} ', '\\Big[', ' \\sqrt{-1} ', '\\Big]', color=YELLOW)
        for z in (gaussian_zi, gaussian_zsqrt1):
            z.next_to(gaussian_text, RIGHT, buff=0.3)
        self.play(ReplacementTransform(grid_dots.copy().fade(1), gaussian_zi), run_time=1)
        self.wait()
        self.play(FadeTransformPieces(gaussian_zi, gaussian_zsqrt1))
        self.wait()

        # Gaussian integers are an extension of integers
        self.play(
            AnimationGroup(
                gaussian_rect.animate.set_height(FRAME_HEIGHT, stretch=True)
                                     .set_width(FRAME_WIDTH, stretch=True)
                                     .center(),
                Animation(gaussian_group[1:]),
                Animation(gaussian_zsqrt1),
            )
        )
        self.wait()
        int_text = TexText('整数', color=GOLD)
        int_z = Tex('\\mathbb{Z}', color=GOLD)
        int_text.next_to(TOP, DR, buff=0.5)
        int_text.shift(RIGHT)
        int_z.next_to(int_text, RIGHT, buff=0.3)
        self.play(Write(int_text), Write(int_z))
        self.wait()

        # Many concepts are analogous: primes, factorization and division algorithm
        gaussian_prime = Tex('\\text{高斯素数：}', ',\\,'.join(['1+i', '3', '3-2i', '\\dots']))
        gaussian_factor1 = Tex('\\text{唯一分解：}', '5+5i', ' = (1+i) \\cdot (2+i) \\cdot (2-i)')
        gaussian_factor2 = Tex('4-2i', ' = (1+i) \\cdot (1-i) \\cdot (2-i)')
        gaussian_div = Tex('\\text{带余除法：}', '(5+5i)', '= i \\cdot ', '(4-2i)', ' + ', '(3+i)')
        gaussian_rem = Tex('|3+i|', ' < ', '|4-2i|')
        int_prime = Tex('\\text{素数：}', ',\\,'.join(['5', '2', '13', '7', '\\dots']))
        int_factor1 = Tex('\\text{唯一分解：}', '70', '= 2 \\times 5 \\times 7')
        int_factor2 = Tex('28', '= 2 \\times 2 \\times 7')
        int_div = Tex('\\text{带余除法：}', '70', '= 2 \\times ', '28', ' + ', '14')
        int_rem = Tex('14', ' < ', '28')
        gaussian_example_group = VGroup(
            gaussian_prime, gaussian_factor1, gaussian_factor2, gaussian_div, gaussian_rem
        )
        int_example_group = VGroup(
            int_prime, int_factor1, int_factor2, int_div, int_rem
        )
        groups = (gaussian_example_group, int_example_group)
        colors = (gaussian_zsqrt1.get_color(), int_z.get_color())
        align_mobs = (gaussian_text, int_text)
        for group, color, align_mob in zip(groups, colors, align_mobs):
            group.scale(0.7).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            group.shift(0.5 * DOWN)
            group.next_to(align_mob, DOWN, aligned_edge=LEFT, coor_mask=X_COOR_MASK)
            group[2].next_to(group[1][1], DOWN, aligned_edge=LEFT, coor_mask=X_COOR_MASK)
            group[-1].next_to(group[-2], DOWN, aligned_edge=RIGHT, buff=0.6)
            group[1][1].set_color(RED)
            group[3][1].set_color(RED)
            group[2][0].set_color(BLUE)
            group[3][-3].set_color(BLUE)
            group[-1][-1].set_color(BLUE)
            group[3][-1].set_color(MAROON_B)
            group[-1][0].set_color(MAROON_B)
            for ind in (0, 1, 3):
                group[ind][0][:-1].set_color(color)
        self.play(*[FadeIn(group[0]) for group in groups])
        self.wait()
        self.play(*[FadeIn(group[1:3]) for group in groups])
        self.wait()
        self.play(*[FadeIn(group[3]) for group in groups])
        self.wait()

        # Slight difference on the remainder
        for group in reversed(groups):
            self.play(
                AnimationGroup(
                    ReplacementTransform(group[-2][-1].copy(), group[-1][0]),
                    Write(group[-1][1]),
                    ReplacementTransform(group[-2][-3].copy(), group[-1][2]),
                    lag_ratio=0.1,
                )
            )
            self.add(group)
            self.wait()

        # As for the division algorithm in Gaussian integers... maybe next time
        tbc_rect = SurroundingRectangle(gaussian_div)
        tbc_arrow = Arrow(
            ORIGIN, 5.5 * LEFT,
            color=GREY_E,
            stroke_width=60,
            tip_width_ratio=2,
            max_width_to_length_ratio=1000,
        )
        tbc_arrow.to_corner(DL, buff=0.5).shift(0.3 * UP)
        tbc_text = TexText('To Be Continued', color=GREY_D)
        tbc_text.set_height(0.3).next_to(tbc_arrow.get_right(), LEFT, buff=0.7)
        tbc_group = VGroup(tbc_arrow, tbc_text)
        tbc_group.fade(0.8)
        self.play(ShowCreation(tbc_rect))
        self.wait()
        self.play(FadeIn(tbc_group, shift=100 * LEFT, rate_func=rush_into))
        self.wait()


class FromCirclesToSpheresP1(Scene):
    def construct(self):
        # Definition of Ford circles... again. Connect to the last scene.
        ford_circles_def = VGroup(
            TexText(
                '对于', '最简分数', '$\\dfrac{p}{q}$',
                ', 在', '数轴', '$\\dfrac{p}{q}$',
                '的位置上放置半径为', '$\\dfrac{1}{2q^2}$', '的',
                '圆', ',',
            ),
            TexText('再将所有的', '圆', '收集起来即可.'),
        )
        ford_circles_def.arrange(DOWN, aligned_edge=LEFT)
        self.play(Write(ford_circles_def), run_time=5)
        self.wait()

        # Change 1: The sphere needs to be put on the 2D complex plane
        axis_text = ford_circles_def[0][4]
        axis_cross = Cross(axis_text)
        plane_text = RemarkText(
            axis_text, '复平面',
            direction=UP, aligned_edge=ORIGIN, scale_factor=0.7,
            color=YELLOW
        )
        pq_complex_remark = TexText('$p,\\,q$都是复数', color=YELLOW)
        pq_complex_remark.next_to(axis_text, UP, buff=0.4)
        self.play(
            Write(pq_complex_remark),
            axis_text.animate.set_color(GREY),
        )
        self.wait()
        self.play(ShowCreation(axis_cross))
        self.wait()
        self.play(FadeTransform(pq_complex_remark, plane_text))
        self.wait()

        # Change 2: The radius has to be real, and the dimension goes from 2 to 3
        circle_texts = VGroup(ford_circles_def[0][-2], ford_circles_def[1][1])
        circle_crosses = VGroup(*[Cross(text) for text in circle_texts])
        radius_text = ford_circles_def[0][-4]
        radius_cross = Cross(radius_text)
        real_radius_remark = TexText('半径是实数', color=YELLOW)
        real_radius_text = Tex('\\dfrac{1}{2|q|^2}', color=YELLOW)
        for rt in (real_radius_remark, real_radius_text):
            rt.next_to(radius_text, UP, buff=0.4)
        dimension_remarks = VGroup(*[
            TexText('维度 \\\\ $+1$', color=YELLOW).next_to(text, DOWN, buff=0.4)
            for text in circle_texts
        ])
        sphere_texts = VGroup(*[
            TexText('球', color=YELLOW).next_to(text, DOWN, buff=0.4)
            for text in circle_texts
        ])
        self.play(
            radius_text.animate.set_color(GREY),
            Write(real_radius_remark),
        )
        self.wait()
        self.play(
            circle_texts.animate.set_color(GREY),
            Write(dimension_remarks),
        )
        self.wait()
        self.play(ShowCreation(circle_crosses), ShowCreation(radius_cross))
        self.wait()
        self.play(
            FadeTransform(real_radius_remark, real_radius_text),
            *[FadeTransform(dimension_remarks[ind], sphere_texts[ind]) for ind in range(2)],
        )
        self.wait()

        # Full definition
        ford_spheres_def = VGroup(
            TexText(
                '对于', '最简分数', '$\\dfrac{p}{q}$',
                ', 在', '复平面', '$\\dfrac{p}{q}$',
                '的位置上放置半径为', '$\\dfrac{1}{2|q|^2}$', '的',
                '球', ',',
            ),
            TexText('再将所有的', '球', '收集起来即可.'),
        )
        ford_spheres_def.arrange(DOWN, aligned_edge=LEFT)
        ford_spheres_def.scale(0.9).to_edge(UP, buff=0.2)
        self.play(
            ReplacementTransform(ford_circles_def[0][:4], ford_spheres_def[0][:4]),
            ReplacementTransform(plane_text[0], ford_spheres_def[0][4]),
            ReplacementTransform(ford_circles_def[0][5:7], ford_spheres_def[0][5:7]),
            ReplacementTransform(real_radius_text[0], ford_spheres_def[0][7]),
            ReplacementTransform(ford_circles_def[0][8::2], ford_spheres_def[0][8::2]),
            ReplacementTransform(sphere_texts[0][0], ford_spheres_def[0][9]),
            ReplacementTransform(ford_circles_def[1][::2], ford_spheres_def[1][::2]),
            ReplacementTransform(sphere_texts[1][0], ford_spheres_def[1][1]),
            FadeOut(
                VGroup(*[
                    ford_circles_def[x][y]
                    for (x, y) in [(0, 4), (0, 7), (0, 9), (1, 1)]
                ]),
                shift=0.4 * UP,
                rate_func=squish_rate_func(rush_into, 0, 0.4),
            ),
            FadeOut(
                VGroup(axis_cross, radius_cross, circle_crosses),
                shift=0.4 * UP,
                rate_func=squish_rate_func(rush_into, 0, 0.4),
            ),
            run_time=2,
        )
        self.add(ford_spheres_def)
        self.wait()

        # An example of (4-2i)/(5+5i)
        frac = Tex('{4-2i', ' \\over ', '5+5i}')
        gcd = VGroup(
            Tex('5+5i', ' = ', 'i \\cdot ', '(4-2i)', ' + ', '(3+i)'),
            Tex('4-2i', ' = ', '(1-i) \\cdot ', '(3+i)', ' + ', '0'),
        )
        gcd.arrange(DOWN, index_of_submobject_to_align=1, buff=0.6)
        simp_frac = Tex('{1-i', ' \\over ', '2+i}')
        for mob in (frac, simp_frac):
            mob.scale(1.5).center()
        gcd.next_to(frac, DOWN, buff=0.7)
        gcd_underline = Underline(gcd[1][-3], color=GREEN)
        number_color_map = {
            '4-2i': RED,
            '5+5i': BLUE,
            '3+i': GREEN,
            '0': MAROON_B,
            '1-i': RED,
            '2+i': BLUE,
        }
        for tex in (*frac, *gcd[0], *gcd[1], *simp_frac):
            for key in number_color_map:
                if key in tex.tex_string:
                    tex.set_color(number_color_map[key])
        gcd[1][2].set_color(WHITE)
        self.play(Write(frac))
        self.wait()
        self.play(
            AnimationGroup(*[FadeIn(line) for line in gcd]),
            lag_ratio=0.3,
        )
        self.wait()
        self.play(ShowCreation(gcd_underline), Indicate(gcd[1][-3]))
        gcd.add(gcd_underline)
        self.wait()
        self.play(
            FadeOutToPoint(gcd, frac.get_center()),
            FadeTransform(frac, simp_frac),
        )
        self.wait()
        def_rect = FullScreenFadeRectangle()
        def_rect.stretch_to_fit_height(ford_spheres_def.get_height() * 1.2)
        def_rect.move_to(ford_spheres_def)
        def_group = VGroup(def_rect, ford_spheres_def)
        self.add(def_group)

        # Show the example sphere
        param = VGroup(
            Tex('\\text{位置：}', '{1-i', ' \\over ', '2+i}'),
            Tex('\\text{半径：}', '{1 \\over 10}'),
        )
        param.arrange(DOWN, aligned_edge=LEFT)
        param.to_corner(DL, buff=0.3)
        param.set_color(YELLOW)
        param_rect = BackgroundRectangle(param, buff=0.3)
        param_rect.stretch_to_fit_width(4)
        plane = ComplexPlane(
            x_range=[-8, 8, 1],
            y_range=[-8, 8, 1],
            axis_config={'unit_size': 2.2, }
        )
        plane.add_coordinate_labels()
        self.play(
            AnimationGroup(
                ShowCreation(plane),
                Animation(def_group),
                FadeIn(param_rect),
                Write(VGroup(*[line[0] for line in param])),
                ReplacementTransform(simp_frac, param[0][1:]),
            ),
            run_time=2,
        )
        self.wait()
        numer, denom = (1 - 1j), (2 + 1j)
        value = numer / denom
        radius = 1 / (2 * abs(denom)**2) * plane.get_x_unit_size()
        pos = plane.number_to_point(value)
        pos_arrow = Arrow(ORIGIN, UR, color=YELLOW)
        pos_arrow.next_to(pos, DL, buff=0.1)
        sphere = Sphere(radius=radius, opacity=0.5, color=YELLOW)
        sphere.next_to(pos, OUT, buff=0)
        self.play(FadeTransform(param[0][1:].copy(), pos_arrow))
        self.wait()
        self.play(Write(param[1][1:]))
        self.wait()
        self.play(
            FadeTransform(param[1][1:].copy(), sphere, rate_func=squish_rate_func(smooth, 0, 0.7)),
            FadeOut(pos_arrow, rate_func=squish_rate_func(smooth, 0.3, 1))
        )
        self.wait()

        # Change to 3D perspective
        new_fractions = [
            (1, 1 + 1j, '1', '1+i'),
            (0, 1, '0', '1'),
            (-1 - 1j, 1, '-1-i', '1'),
            (1 - 1j, 3 + 2j, '1-i', '3+2i'),
            (2 - 3j, 3 + 1j, '2-3i', '3+i'),
        ]
        new_params = [
            VGroup(
                Tex('\\text{位置：}', f'{{ {numer_str} \\over {denom_str} }}'),
                Tex('\\text{半径：}', f'{{ 1 \\over {2*round(abs(denom)**2)} }}'),
            )
            for numer, denom, numer_str, denom_str in new_fractions
        ]
        for new_param in new_params:
            new_param.arrange(DOWN, aligned_edge=LEFT)
            new_param.to_corner(DL, buff=0.3)
            new_param.set_color(YELLOW)
        fixed_mobs = VGroup(
            def_rect, ford_spheres_def, param_rect, param,
            *new_params,
        )
        fixed_mobs.fix_in_frame()
        self.remove(fixed_mobs)
        self.add(fixed_mobs[:4])
        frame = self.camera.frame
        self.camera.max_allowable_norm = FRAME_WIDTH * 10
        self.play(
            frame.animate.set_euler_angles(phi=PI / 3, theta=PI / 6),
            run_time=2,
        )
        self.wait()
        rotation_rate = -0.02
        frame.add_updater(
            lambda mob, dt: mob.increment_theta(dt * rotation_rate)
        )
        for (frac, new_param) in zip(new_fractions, new_params):
            numer, denom, _, _ = frac
            new_value = numer / denom
            new_radius = 1 / (2 * abs(denom)**2) * plane.get_x_unit_size()
            new_pos = plane.number_to_point(new_value)
            new_sphere = Sphere(radius=new_radius, opacity=0.5, color=YELLOW)
            new_sphere.next_to(new_pos, OUT, buff=0)
            self.play(
                FadeIn(new_sphere, shift=IN),
                sphere.animate.set_color(random.choice([RED, ORANGE, GREEN, TEAL, MAROON_B, GOLD])),
                ReplacementTransform(param[0][0], new_param[0][0]),
                FadeTransform(param[0][1:], new_param[0][1:]),
                ReplacementTransform(param[1][0], new_param[1][0]),
                FadeTransform(param[1][1:], new_param[1][1:]),
            )
            self.remove(param)
            self.add(new_param)
            param = new_param
            sphere = new_sphere
            self.wait(3)
        self.wait(20)


class FiveFordSpheresDemonstrationP1(Scene):
    def construct(self):
        # Info
        z_symbols = VGroup(*[
            Tex('\\mathbb{Z}', '\\left[ {\\sqrt{-1}} \\right]'),
            Tex('\\mathbb{Z}', '\\left[ {\\sqrt{-2}} \\right]'),
            Tex('\\mathbb{Z}', '\\left[ {1 + \\sqrt{-3} \\over 2} \\right]'),
            Tex('\\mathbb{Z}', '\\left[ {1 + \\sqrt{-7} \\over 2} \\right]'),
            Tex('\\mathbb{Z}', '\\left[ {1 + \\sqrt{-11} \\over 2} \\right]'),
        ])
        z_defs = VGroup(*[
            Tex('\\left\{ \\left. a + b \\cdot {\\sqrt{-1}} \\right| a,\\,b \\in \\mathbb{Z} \\right\}'),
            Tex('\\left\{ \\left. a + b \\cdot {\\sqrt{-2}} \\right| a,\\,b \\in \\mathbb{Z} \\right\}'),
            Tex('\\left\{ \\left. a + b \\cdot {1 + \\sqrt{-3} \\over 2} \\right| a,\\,b \\in \\mathbb{Z} \\right\}'),
            Tex('\\left\{ \\left. a + b \\cdot {1 + \\sqrt{-7} \\over 2} \\right| a,\\,b \\in \\mathbb{Z} \\right\}'),
            Tex('\\left\{ \\left. a + b \\cdot {1 + \\sqrt{-11} \\over 2} \\right| a,\\,b \\in \\mathbb{Z} \\right\}'),
        ])
        int_chn_names = VGroup(*[
            TexText(name)
            for name in ('高斯整数', '', '艾森斯坦整数', '克莱因整数', '')
        ])
        int_eng_names = VGroup(*[
            TexText(name)
            for name in ('(Gaussian Integers)', '', '(Eisenstein Integers)', '(Kleinian Integers)', '')
        ])
        tangency_images = Group(*[
            ImageMobject(f'poly_{suffix}.png', height=2)
            for suffix in (1, 2, 3, 7, 11)
        ])

        # Arrangement
        mob_sets = (z_symbols, z_defs, int_chn_names, int_eng_names, tangency_images)
        colors = (YELLOW, YELLOW_A, YELLOW, YELLOW_A, None)
        scale_factors = (1, 0.6, 1, 0.7, 1)
        y_positions = (3, 1.8, 0.3, -0.3, -2.5)
        edge_buff = 0.5

        for mob_set, color, scale_factor, y_pos in zip(mob_sets, colors, scale_factors, y_positions):
            for mob in mob_set:
                mob.scale(scale_factor)
                mob.move_to(y_pos * UP)
                mob.to_edge(LEFT, buff=edge_buff)
                if color is not None:
                    mob.set_color(color)

        # Show 5 Ford spheres from imaginary quadratic fields
        group1, group2, group3, group7, group11 = list(map(lambda mobs: Group(*mobs), zip(*mob_sets)))
        self.add(group1)
        self.wait()
        ed_chn_text = TexText('欧几里得整环', color=BLUE)
        ed_eng_text = TexText('(Euclidean Domain)', color=BLUE_A)
        ed_eng_text.scale(0.6)
        ed_texts = VGroup(ed_chn_text, ed_eng_text)
        ed_texts.arrange(DOWN, aligned_edge=LEFT)
        ed_rect = BackgroundRectangle(group1[-1])
        ed_texts.move_to(ed_rect).to_edge(LEFT, buff=edge_buff)
        self.play(FadeIn(ed_rect), Write(ed_texts), run_time=3)
        self.wait()
        self.play(FadeOut(VGroup(ed_rect, ed_texts)))
        self.wait(5)

        self.play(FadeOut(group1), FadeIn(group2), run_time=1)
        self.wait(5)
        self.play(FadeOut(group2), FadeIn(group3), run_time=1)
        self.wait(5)
        self.play(FadeOut(group3), FadeIn(group7), run_time=1)
        self.wait(5)
        self.play(ShowCreationThenDestructionAround(group7[-1]))
        self.wait()
        self.play(FadeOut(group7), FadeIn(group11), run_time=1)
        self.wait(5)


class ReferencesP1(ReferenceScene):
    def get_references(self):
        ford_circles_ref = Reference(
            name='Fractions',
            authors='Lester R. Ford',
            pub='The American Mathematical Monthly, 1938, 45(9): 586-601.',
            doi='https://doi.org/10.2307/2302799',
            info='福特圆的提出，外加它和其他概念的联系。'
        )
        ford_spheres_ref = Reference(
            name='Ford Circles and Spheres',
            authors='Sam Northshield',
            pub='arXiv preprint arXiv:1503.00813, 2015.',
            doi='https://arxiv.org/abs/1503.00813',
            info='内容非常详尽，觉得上面那个读起来太简单的话，可以看这篇。介绍了二维福特圆的3种等价的构造方式，以及把它推广到三维的福特球。文末还有些开放/半开放的相关问题。'
        )
        visual_gaussian_ref = Reference(
            name='A note on geometric representations of Gaussian rational numbers',
            authors='Clifford A. Pickover',
            pub='The Visual Computer, 1997, 3(13): 127-130.',
            doi='https://doi.org/10.2307/2302799',
            info='高斯整数对应的福特球的绘制，可以看看底部的相切结构。',
        )
        real_quad_field_ref = Reference(
            name='A generalization of Apollonian packing of circles',
            authors='Gerhard Guettler and Colin Mallows',
            pub='Journal of Combinatorics, 2010, 1(1): 1-27.',
            doi='https://dx.doi.org/10.4310/JOC.2010.v1.n1.a1',
            info='三维的福特球有很多，但其实二维的福特圆也不止一个。这篇的第9章介绍了一种跟$\\sqrt{2}$有关的福特圆与法里和。'
        )
        return (
            ford_circles_ref, ford_spheres_ref, visual_gaussian_ref, real_quad_field_ref,
        )


class ThumbnailP1(Scene):
    def construct(self):
        zoom_x = 0.65
        ff = FordFractal(
            max_denom=20,
            zoom_places=[zoom_x],
            axis_config={'y_offset': -2, },
        )
        # ff.suspend_updating(recurse=True)
        ff.scale(3, about_point=ff.get_axis().number_to_point(zoom_x))
        self.wait()
        circles = VGroup(*[
            ff.get_circle(*pair)
            for pair in [(1, 2), (2, 3), (3, 5)]
        ])
        colors = [RED, BLUE, GREEN]
        tex = Tex('{1 \\over 2}', '\\oplus', '{2 \\over 3}', '=', '{3 \\over 5}')
        tex.scale(2).to_corner(UR).shift(0.5 * DOWN)
        for c, t, color in zip(circles, tex[::2], colors):
            c.set_color(color)
            c.suspend_updating()
            c.get_circle().set_fill(opacity=0.15).set_stroke(width=5)
            t.set_color(color)
        self.add(ff, tex)
        self.wait()


# Part 2 Scenes
class LastVideoRecapP2(Scene):
    def construct(self):
        title = TexText('上期回顾')
        title.to_edge(UP)
        last_video = PictureInPictureFrame()
        last_video.set_height(6)
        last_video.next_to(title, DOWN)

        coming_up = TexText('接下来的内容')
        coming_up.to_edge(UP)
        farey_text = TexText('法里和', color=YELLOW)
        farey_text.move_to(4 * LEFT + 2.5 * UP)
        farey_content = Tex('{a \\over b}', '\\oplus', '{c \\over d}', '=', '{a+b \\over c+d}')
        for mob, color in zip(farey_content[::2], [RED, BLUE, GREEN]):
            mob.set_color(color)
        farey_content.scale(1.2).next_to(farey_text, DOWN)
        ford_text = TexText('福特圆', color=YELLOW)
        ford_text.move_to(4 * LEFT + 0.5 * DOWN)
        ford_content = FordFractal(axis_config={'x_range': [0, 1, 1], 'include_tick_numbers': False},)
        ford_content.set_height(3)
        ford_content.next_to(ford_text, DOWN)
        diophantine_text = TexText('丢番图逼近', color=YELLOW)
        diophantine_text.move_to(3 * RIGHT + 2.5 * UP)
        diophantine_remark = RemarkText(
            diophantine_text, '（用有理数近似实数）',
            direction=DOWN, scale_factor=0.6, aligned_edge=ORIGIN, color=YELLOW_A,
        )
        diophantine_content = VGroup(
            FareyTree(
                n_levels=4, include_top=False,
                h_buff=0.7, v_buff=1.3, line_config={'stroke_width': 2, },
            ),
            Tex('\\left| {p \\over q} - \\alpha \\right| < {1 \\over 2 q^2}'),
        )
        diophantine_content[0].set_width(6)
        diophantine_content.arrange(DOWN, buff=0.8)
        diophantine_content.next_to(diophantine_remark, DOWN, buff=0.5)
        diophantine_content[1].set_color(MAROON_B)
        last_video_content = VGroup(farey_text, farey_content, ford_text, ford_content)
        this_video_content = VGroup(diophantine_text, diophantine_remark, diophantine_content)
        content_group = VGroup(last_video_content, this_video_content)
        content_group.scale(0.75).move_to(last_video)
        self.play(Write(title), ShowCreation(last_video))
        self.wait()
        self.add(last_video_content)
        self.wait()
        self.play(FadeTransform(title, coming_up))
        self.wait()
        self.wait()
        self.play(
            Write(diophantine_text),
            AnimationGroup(*[GrowFromCenter(mob) for mob in diophantine_content])
        )
        self.wait()
        self.play(Write(diophantine_remark))
        self.wait()


class LayerByLayerConstructionP2(Scene):
    def construct(self):
        # Regular construction: all at once
        ff = FordFractal(max_denom=20, axis_config={'y_offset': -3},)
        axis = ff.get_axis()
        self.play(Write(axis))
        self.wait()
        circles = ff.get_all_circles()
        random.shuffle(circles.submobjects)
        ff.suspend_updating()
        self.play(
            AnimationGroup(*[GrowFromCenter(circle) for circle in circles], lag_ratio=0.005),
            run_time=5,
        )
        ff.resume_updating()
        self.add(ff)
        self.wait()

        # But (arguably) these circles have different layer
        layer_group = VGroup()
        max_denom = ff.max_denom
        for denom in range(1, max_denom // 2 + 1):
            curr_layer_group = VGroup()
            possible_numers = get_coprime_numers_by_denom(denom)
            for numer in possible_numers:
                curr_layer_group.add(ff.get_circle(numer, denom))
            layer_group.add(curr_layer_group)
        self.play(
            AnimationGroup(
                *[Indicate(group, scale_factor=1) for group in layer_group],
                lag_ratio=0.25,
            ),
            run_time=3,
        )
        self.wait()

        # Weird transition...
        ff.suspend_updating()
        self.play(ff.animate.shift(UP))
        ff.resume_updating()
        self.wait()
        # A hacky solution to prevent the axis fades out along with the tick numbers
        ff_copy = FordFractal(max_denom=1, axis_config={'include_tick_numbers': False},)
        self.add(ff_copy)
        ff.suspend_updating()
        self.play(FadeOut(ff))
        self.remove(ff)
        ff = ff_copy
        axis = ff.get_axis()
        self.add(ff)
        self.wait()

        # Alternate construction: layer by layer
        ff.suspend_updating()
        level = 0
        sequence = np.array([(0, 1), (1, 1)])
        first_gap = FordFractalGap(ff, *sequence)
        first_circles = VGroup(*[ff.get_circle(*pair) for pair in sequence])
        for circle, color in zip(first_circles, [RED, BLUE]):
            circle.save_state()
            circle.generate_target()
            circle.target.set_color(color)
            circle.target.get_circle().set_fill(opacity=0.3)
        self.play(*[MoveToTarget(circle) for circle in first_circles])
        self.wait()
        self.play(DrawBorderThenFill(first_gap, stroke_color=YELLOW))
        self.wait()

        # First circle to insert: 1/2, the Farey sum of 0/1 and 1/1
        circle_1_2 = ff.generate_circle(1, 2)
        circle_1_2.suspend_updating()
        circle_1_2.save_state()
        circle_1_2.set_color(GREEN)
        circle_1_2.get_circle().set_fill(opacity=0.3)
        self.play(
            FadeOut(first_gap, run_time=1),
            GrowFromCenter(circle_1_2, run_time=2)
        )
        self.wait()
        first_equation = Tex('{0 \\over 1}', '\\oplus', '{1 \\over 1}', '=', '{1 \\over 2}')
        for tex, color in zip(first_equation[::2], [RED, BLUE, GREEN]):
            tex.set_color(color)
        first_equation.next_to(axis.number_to_point(1 / 2), DOWN, index_of_submobject_to_align=-1)
        tex_0_1, tex_1_1, tex_1_2 = first_equation[::2]
        for tex, circle in zip(first_equation[::2], it.chain(first_circles, [circle_1_2])):
            tex.generate_target()
            tex.target.scale(0.8).next_to(circle, DOWN)
        tex_1_2.target.next_to(circle_1_2, DOWN, buff=0.45)
        self.play(Write(first_equation))
        self.wait()
        self.play(
            MoveToTarget(tex_0_1, path_arc=-PI / 3),
            MoveToTarget(tex_1_1, path_arc=PI / 3),
            MoveToTarget(tex_1_2),
            FadeOut(first_equation[1::2], rate_func=squish_rate_func(smooth, 0, 0.5)),
            run_time=2,
        )
        axis_fractions = VGroup(tex_0_1, tex_1_1, tex_1_2)
        self.add(axis_fractions)
        self.wait()
        self.play(
            AnimationGroup(*[circle.animate.restore() for circle in first_circles]),
            circle_1_2.animate.restore(),
            AnimationGroup(*[tex.animate.set_color(WHITE) for tex in (tex_0_1, tex_1_1, tex_1_2)]),
        )
        self.wait()

        # 1/2 definitely can't fill up the gap, we need another two circles
        # So does every other circles
        circle_mobs = VGroup(circle_1_2)
        for level in range(1, 4):
            curr_seq = get_farey_sequence(level, use_tuple=True)
            new_seq = get_farey_sequence(level + 1, use_tuple=True)

            new_gaps = [
                FordFractalGap(ff, curr_seq[k], curr_seq[k + 1])
                for k in range(len(curr_seq) - 1)
            ]
            new_circles = [
                ff.generate_circle(*pair)
                for pair in new_seq[1::2]
            ]
            new_fractions = [
                Tex(f'{{{pair[0]} \\over {pair[1]}}}').scale(0.8).next_to(circle, DOWN, buff=0.45 + level * 0.2)
                for pair, circle in zip(new_seq[1::2], new_circles)
            ]
            circle_mobs.add(*new_circles)
            axis_fractions.add(*new_fractions)
            for circle in new_circles:
                circle.suspend_updating()
                circle.save_state()
                circle.set_color(GREEN)
                circle.get_circle().set_fill(opacity=0.3)
            self.play(
                AnimationGroup(*[
                    GrowFromEdge(gap, DOWN)
                    for gap, circle in zip(new_gaps, new_circles)
                ])
            )
            self.wait()
            self.play(
                AnimationGroup(*[GrowFromCenter(circle) for circle in new_circles]),
                AnimationGroup(*[FadeOut(gap) for gap in new_gaps]),
                run_time=2,
            )
            self.wait()
            self.play(
                AnimationGroup(*[Write(frac) for frac in new_fractions]),
                AnimationGroup(*[circle.animate.restore() for circle in new_circles]),
            )
            self.wait()
            # Extra animation for 1/3 and 2/3
            if level == 1:
                extra_additions = VGroup(
                    Tex('{0 \\over 1}', '\\oplus', '{1 \\over 2}', '=', '{1 \\over 3}'),
                    Tex('{1 \\over 2}', '\\oplus', '{1 \\over 1}', '=', '{2 \\over 3}'),
                )
                for addition, corner in zip(extra_additions, [DL, DR]):
                    for tex, color in zip(addition[::2], [RED, BLUE, GREEN]):
                        tex.set_color(color)
                    addition.scale(0.8).to_corner(corner)
                for k in range(len(curr_seq) - 1):
                    circle_a = ff.get_circle(*curr_seq[k])
                    circle_b = ff.get_circle(*curr_seq[k + 1])
                    circle_apb = new_circles[k]
                    for circle, color in zip([circle_a, circle_b, circle_apb], [RED, BLUE, GREEN]):
                        circle.generate_target()
                        circle.target.set_color(color)
                        circle.target.get_circle().set_fill(opacity=0.3)
                    self.play(
                        MoveToTarget(circle_a),
                        MoveToTarget(circle_b),
                        MoveToTarget(circle_apb),
                        Indicate(axis_fractions[2 * k], color=RED),
                        Indicate(axis_fractions[2 - k], color=BLUE),
                        Indicate(axis_fractions[3 + k], color=GREEN),
                        FadeIn(extra_additions[k]),
                        rate_func=there_and_back_with_pause,
                        run_time=2,
                    )
                    self.wait()
                self.play(FadeOut(extra_additions))
                self.wait()

        # Collect the fractions
        gained_fractions = axis_fractions.submobjects

        def tex_sort_func(tex_mob):
            numer_str, denom_str = tex_mob.tex_string.split(sep='\\over')
            numer = int(numer_str[1:])
            denom = int(denom_str[:-1])
            return numer / denom
        gained_fractions.sort(key=tex_sort_func)
        arrange_fractions = VGroup(*gained_fractions)
        ff_group = VGroup(ff, circle_mobs)
        ff_group.suspend_updating()
        self.play(
            FadeOut(ff_group),
            arrange_fractions.animate.arrange(RIGHT, buff=0.3, coor_mask=X_COOR_MASK),
            run_time=3,
        )
        self.wait()

        # Arrange the levels and switch scene, as the ford circles are rendering-heavy
        ft = FareyTree(n_levels=4, include_top=True)
        ft.set_height(7.5)
        ft_fractions = ft.get_all_fractions()
        self.play(
            AnimationGroup(*[
                FadeTransform(old_f, new_f)
                for old_f, new_f in zip(axis_fractions, ft_fractions)
            ], lag_ratio=0.01),
            run_time=3,
        )
        self.wait()


class FareyTreeAndItsPropertiesP2(Scene):
    def construct(self):
        n_levels = 4
        ft = FareyTree(n_levels=n_levels, include_top=True)
        ft.set_height(7.5)
        ft_fractions = ft.get_all_fractions()
        self.add(ft_fractions)
        self.wait()

        # Show levels
        level_text_group = VGroup()
        for level in range(n_levels + 1):
            fracs = ft.get_fraction_mobs_of_level(level)
            level_text = TexText(f'第{level}层', color=YELLOW)
            if level == 0:
                fracs_rect = SurroundingRectangle(fracs, buff=0.1)
                fracs_rect.set_width(9, stretch=True)
                anim_list = [ShowCreation(fracs_rect)]
                level_text.next_to(fracs_rect, RIGHT, buff=0.4)
            else:
                fracs_rect.generate_target()
                fracs_rect.target.move_to(fracs)
                anim_list = [MoveToTarget(fracs_rect)]
                level_text.next_to(fracs_rect.target, RIGHT, buff=0.4)
            anim_list.append(Write(level_text))
            self.play(*anim_list)
            level_text_group.add(level_text)
            self.wait()
        self.play(FadeOut(fracs_rect))
        self.wait()

        # Connect 0/1 and 1/2 using Ford circles
        pairs = [(0, 1, 0), (1, 2, 1)]
        frac_01 = ft.get_fraction_from_numbers(*pairs[0][:-1])
        frac_12 = ft.get_fraction_from_numbers(*pairs[1][:-1])
        line_01_12 = ft.get_line_from_frac_tuples(*[FareyRational(*pair) for pair in pairs])
        demo_mobs = (frac_01, frac_12, line_01_12)
        for mob in demo_mobs:
            mob.save_state()
        line_01_12.set_color(YELLOW)
        self.play(
            frac_01.animate.scale(1.2).set_color(RED),
            frac_12.animate.scale(1.2).set_color(BLUE),
        )
        self.wait()
        self.play(GrowFromCenter(line_01_12))
        self.wait()
        self.play(AnimationGroup(*[mob.animate.restore() for mob in demo_mobs]))
        self.wait()

        # Connect 1/1 and 1/2 using discriminant
        rule_dis = VGroup(
            Tex('{a \\over b}', ' \\text{与} ', '{c \\over d}', ' \\text{连线}'),
            Tex('\\Leftrightarrow'),
            Tex('|', 'a', ' \\cdot ', 'd', '-', 'b', ' \\cdot ', 'c', '|=1'),
        )
        demo_dis = VGroup(
            Tex('{1 \\over 2}', ' \\text{与} ', '{1 \\over 1}', ' \\text{连线}'),
            Tex('\\Leftrightarrow'),
            Tex('|', '1', ' \\cdot ', '1', '-', '2', ' \\cdot ', '1', '|=1'),
        )
        dis_rect = FullScreenFadeRectangle(fill_opacity=0.95)
        dis_rect.set_height(5, stretch=True)
        dis_rect.to_edge(DOWN, buff=0)
        for dis in (rule_dis, demo_dis):
            VGroup(dis[0][0], dis[2][1::4]).set_color(RED)
            VGroup(dis[0][2], dis[2][3::4]).set_color(BLUE)
            dis.scale(1.2).arrange(RIGHT, buff=0.4)
            dis.shift(dis_rect.get_center() - dis[1].get_center())
        pairs = [(1, 2, 1), (1, 1, 0)]
        frac_12 = ft.get_fraction_from_numbers(*pairs[0][:-1])
        frac_11 = ft.get_fraction_from_numbers(*pairs[1][:-1])
        line_12_11 = ft.get_line_from_frac_tuples(*[FareyRational(*pair) for pair in pairs])
        demo_mobs = (frac_12, frac_11, line_12_11)
        for mob in demo_mobs:
            mob.save_state()
        line_12_11.set_color(YELLOW)
        self.play(FadeIn(dis_rect), Write(rule_dis))
        self.wait()
        self.play(
            frac_12.animate.scale(1.2).set_color(RED),
            frac_11.animate.scale(1.2).set_color(BLUE),
            FadeTransformPieces(rule_dis, demo_dis),
        )
        self.wait()
        self.play(GrowFromCenter(line_12_11))
        self.wait()
        self.play(
            AnimationGroup(*[mob.animate.restore() for mob in demo_mobs]),
            FadeTransformPieces(demo_dis, rule_dis),
        )
        self.wait()
        self.play(FadeOut(VGroup(dis_rect, rule_dis)))
        self.wait()

        # Show all possible line, but we only need the ones connecting adjacent levels
        all_possible_lines = ft.get_all_adjacent_lines()
        all_possible_lines.set_color(YELLOW)
        random.shuffle(all_possible_lines.submobjects)
        self.play(
            AnimationGroup(*[
                GrowFromCenter(line)
                for line in all_possible_lines
            ], lag_ratio=0.02),
            run_time=3,
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                AnimationGroup(*[
                    Indicate(frac, color=TEAL)
                    for frac in level_fracs
                ])
                for level_fracs in [
                    ft.get_fraction_mobs_of_level(k)
                    for k in range(n_levels + 1)
                ]
            ], lag_ratio=0.5),
            AnimationGroup(*[
                Indicate(level_text, color=TEAL)
                for level_text in level_text_group
            ], lag_ratio=0.5),
            run_time=3,
        )
        self.wait()
        all_real_lines = ft.get_all_lines()
        all_real_lines.set_color(YELLOW)
        self.add(all_real_lines)
        random.shuffle(all_possible_lines.submobjects)
        self.play(
            AnimationGroup(*[
                ShrinkToCenter(line)
                for line in all_possible_lines
            ], lag_ratio=0.02),
            run_time=3,
        )
        self.add(ft)
        self.remove(all_possible_lines)
        self.wait()

        # 0-th level doesn't fit in this tree structure, so drop it
        zeroth_group = VGroup(
            level_text_group[0],
            ft.get_fraction_mobs_of_level(0),
            ft.get_line_mobs_of_level(0),
        )
        self.play(FadeOut(zeroth_group))
        alt_ft = FareyTree(n_levels=n_levels, include_top=False)
        alt_ft.match_width(ft.get_fraction_mobs_of_level(4))
        alt_ft.next_to(ft.get_bottom(), UP, buff=0)
        alt_ft.save_state()
        alt_ft.get_all_lines().set_color(YELLOW)
        self.remove(ft)
        self.add(alt_ft)
        self.wait()
        self.play(
            FadeOut(level_text_group[1:]),
            alt_ft.animate.restore(),
        )
        self.wait()

        # Now we have the Farey Tree
        farey_chn = TexText('法里树', color=YELLOW)
        farey_chn.to_edge(UP)
        farey_eng = RemarkText(farey_chn, '(Farey Tree)', aligned_edge=ORIGIN, color=YELLOW)
        remark = TexText(
            '注：可能更多人会想到Stern-Brocot树（S-B树） \\\\',
            '它和Farey树很像，但是第0层有区别 \\\\',
            'Farey树由${0 \\over 1}$和${1 \\over 1}$生成，只包含$(0,\\,1)$上的有理数 \\\\',
            'S-B树由${0 \\over 1}$和${1 \\over 0}$生成，可以包含所有正有理数',
            alignment='',
        )
        remark.set_color(BLUE_B).scale(0.45)
        remark.to_corner(UL, buff=0.25)
        self.play(
            FadeIn(VGroup(farey_chn, farey_eng), shift=DOWN),
            FadeIn(remark, shift=DOWN),
        )
        self.add(farey_chn, farey_eng)
        self.wait()
        self.play(FadeOut(remark))
        self.wait()

        # Animate its growth from 4 levels to 8 levels
        ft = alt_ft
        h_buff = 0.5
        root = ft.get_fraction_from_numbers(1, 2)
        circle = Circle(stroke_width=5, color=YELLOW)
        circle.surround(root, dim_to_match=1)
        self.play(ShowCreationThenDestruction(circle))
        self.wait()
        for level in range(5, 9):
            self.play(ft.animate.shift(1.6 * UP))
            self.wait()
            trans_ft_config = {'n_levels': level - 1, 'include_top': False, 'h_buff': h_buff * 2}
            trans_ft = FareyTree(**trans_ft_config)
            source_width = trans_ft.get_fraction_mobs_of_level(1).get_width()
            target_width = ft.get_fraction_mobs_of_level(1).get_width()
            trans_ft.scale(target_width / source_width)
            trans_ft.next_to(ft.get_top(), DOWN, buff=0)
            self.play(ReplacementTransform(ft, trans_ft), run_time=3)
            mod_ft_config = {'n_levels': level, 'include_top': False, 'h_buff': h_buff}
            mod_ft = FareyTree(**mod_ft_config)
            source_width = mod_ft.get_fraction_mobs_of_level(1).get_width()
            target_width = trans_ft.get_fraction_mobs_of_level(1).get_width()
            mod_ft.scale(target_width / source_width)
            mod_ft.next_to(trans_ft.get_top(), DOWN, buff=0)
            mod_ft.generate_target()
            last_level_fracs = mod_ft.get_fractions_of_level(level)
            last_level_line_mobs = mod_ft.get_line_mobs_of_level(level - 1)
            for line in last_level_line_mobs:
                parent_frac_tuple = mod_ft.get_parent_fracs_tuple_of_line(line)[0]
                parent_frac_mob = mod_ft.get_fraction_from_numbers(*parent_frac_tuple)
                line.scale(0).move_to(parent_frac_mob)
            for frac in last_level_fracs:
                frac_mob = mod_ft.get_fraction_from_rational(frac)
                line_mob = mod_ft.get_line_from_frac_component((frac.p, frac.q))
                frac_mob.scale(0).move_to(line_mob)
            self.remove(trans_ft)
            self.add(mod_ft)
            self.play(MoveToTarget(mod_ft), run_time=3)
            self.wait()
            ft = mod_ft

        # The name 'Farey' suggests it can be generated using Farey sum
        self.play(FadeOut(ft))
        self.wait()
        n_levels = 4
        ft = FareyTree(n_levels=n_levels, h_buff=0.75)
        self.play(*[Write(mob) for mob in ft.get_fraction_mobs_of_level(0)])
        self.wait()
        for level in range(1, n_levels + 1):
            new_fracs = ft.get_fractions_of_level(level)
            new_lines = ft.get_line_mobs_of_level(level - 1)
            collect_anims = []
            merge_anims = []
            sequence = get_farey_sequence(level, use_tuple=False)
            for new_frac in new_fracs:
                new_frac_mob = ft.get_fraction_from_rational(new_frac)
                ind = sequence.index(new_frac)
                parent_fracs = [sequence[k] for k in (ind - 1, ind + 1)]
                parents = [
                    ft.get_fraction_from_rational(frac)
                    for frac in parent_fracs
                ]
                parent1, parent2 = parents
                parent1_copy, parent2_copy = parent1.copy(), parent2.copy()
                oplus_sign = Tex('\\oplus')
                group = VGroup(parent1_copy, oplus_sign, parent2_copy)
                group.arrange(RIGHT, buff=0.15).move_to(new_frac_mob)
                collect_anims.extend([
                    ReplacementTransform(parent1.copy(), parent1_copy),
                    Write(oplus_sign),
                    ReplacementTransform(parent2.copy(), parent2_copy),
                ])
                merge_anims.append(FadeTransform(group, new_frac_mob))
            self.play(*collect_anims)
            self.wait()
            self.play(
                *merge_anims,
                *[ShowCreation(line) for line in new_lines]
            )
            self.wait()
        # Remove 0-th level after constructing
        zeroth_fracs = ft.get_fraction_mobs_of_level(0)
        zeroth_lines = ft.get_line_mobs_of_level(0)
        self.play(FadeOut(VGroup(zeroth_fracs, zeroth_lines)))
        alt_ft = FareyTree(n_levels=n_levels, include_top=False, h_buff=0.75)
        alt_ft.next_to(ft.get_bottom(), UP, buff=0)
        self.remove(ft)
        self.add(alt_ft)
        self.wait()
        ft = alt_ft

        # But the Farey mediant is right between its parents
        ft.save_state()
        pairs = [(1, 3), (1, 2), (2, 3)]
        for num, pair in enumerate(pairs):
            p, q = pair
            root = ft.get_fraction_from_numbers(p, q)
            left_tree, right_tree = ft.get_subtrees_from_numbers(p, q)
            sep_line = DashedLine(ORIGIN, 5.5 * DOWN)
            sep_line.set_color(GREEN).fade(0.5)
            sep_line.move_to(root).to_edge(DOWN, buff=0.1)
            left_text = Tex('\\square', f' < {{{p} \\over {q}}}', color=RED)
            right_text = Tex('\\square', f' > {{{p} \\over {q}}}', color=BLUE)
            for text, direction in zip([left_text, right_text], [LEFT, RIGHT]):
                text.scale(1.2)
                factor = 2 if q == 2 else 1
                text.move_to(root).shift(2.5 * direction * factor)
            if q - p == 2:
                self.play(
                    root.animate.set_color(GREEN).scale(1.2),
                    ShowCreation(sep_line),
                )
                self.wait()
                self.play(
                    left_tree.animate.set_color(RED),
                    right_tree.animate.set_color(BLUE),
                )
                self.wait()
                self.play(FadeIn(left_text, shift=LEFT))
                self.wait()
                self.play(FadeIn(right_text, shift=RIGHT))
                self.wait()
            else:
                self.play(
                    FadeIn(left_text, shift=LEFT),
                    FadeIn(right_text, shift=RIGHT),
                    ShowCreation(sep_line),
                    root.animate.set_color(GREEN).scale(1.2),
                    left_tree.animate.set_color(RED),
                    right_tree.animate.set_color(BLUE),
                )
                self.wait()
            self.play(
                FadeOut(left_text, shift=RIGHT),
                FadeOut(right_text, shift=LEFT),
                Uncreate(sep_line),
                ft.animate.restore(),
            )
            self.wait()

        # Search 4/7 in this BST, to be continued in the next scene
        self.play(FadeOut(farey_chn), FadeOut(farey_eng))
        self.wait()


class BinarySearchInFareyTreeP2(Scene):
    def construct(self):
        # Connect to the last scene
        n_levels = 4
        hidden_ft = FareyTree(n_levels=n_levels, h_buff=0.75)
        ft = FareyTree(n_levels=n_levels, include_top=False, h_buff=0.75)
        ft.next_to(hidden_ft.get_bottom(), UP, buff=0)
        self.add(ft)
        self.wait()
        left_rect, right_rect = [FullScreenFadeRectangle(fill_opacity=0.85) for k in range(2)]
        for rect, direction in zip([left_rect, right_rect], [LEFT, RIGHT]):
            rect.stretch_to_fit_height((ft.get_top() - BOTTOM)[1] + 0.1)
            rect.stretch_to_fit_width(0)
            rect.to_edge(DOWN, buff=0).to_edge(direction, buff=0)
        self.ft = ft
        self.left_rect = left_rect
        self.right_rect = right_rect

        # Search 4/7 in this BST
        target = Tex('{4 \\over 7}')
        target[-1].set_color(YELLOW)
        target.to_edge(UP)
        left_line, right_line = lines = [DashedLine(ORIGIN, 5.5 * DOWN) for k in range(2)]
        left_bound, right_bound = bounds = [(0, 1), (1, 1)]
        left_text, right_text = texts = [Tex(f'{{{p} \\over {q}}}') for (p, q) in bounds]
        for line, color, direction, text in zip(lines, [RED, BLUE], [LEFT, RIGHT], texts):
            line.set_color(color).fade(0.5)
            line.next_to(ft.get_bounding_box_point(direction), direction, buff=0.4)
            text.set_color(color).move_to(target)
            text.next_to(line, UP, coor_mask=X_COOR_MASK)
        self.play(Write(target))
        self.wait()
        self.play(*[ShowCreation(line) for line in lines])
        # self.add(*lines)
        self.wait()
        for text in texts:
            self.play(Write(text))
            self.wait()

        # Calculate Farey mediant of two bounds
        mediant_circles = VGroup()
        mediant = ft.get_fraction_from_numbers(1, 2)
        mediant.generate_target()
        mediant.target.set_color(GREEN)
        circle = Circle(color=GREEN)
        circle.set_height(mediant.get_height() * 1.25)
        circle.move_to(mediant)
        mediant_circles.add(circle)
        self.play(
            FadeTransform(left_text.copy(), mediant.target),
            FadeTransform(right_text.copy(), mediant.target),
            MoveToTarget(mediant),
            ShowCreation(mediant_circles[-1]),
        )
        self.remove(mediant.target)
        self.wait()

        # 4/7 is a bit larger than 1/2, so we keep the right subtree and replace the left bound
        comparison = VGroup(
            Tex('{1 \\over 2}'), Tex(' < '), Tex('{4 \\over 7}')
        )
        comparison.arrange(RIGHT)
        comparison[0].set_color(GREEN)
        comparison[2].set_color(YELLOW)
        comparison.next_to(target.get_left(), RIGHT, buff=0)
        self.play(
            ReplacementTransform(mediant.copy(), comparison[0]),
            Write(comparison[1]),
            ReplacementTransform(target, comparison[2]),
        )
        self.wait()
        self.play(
            AnimationGroup(
                self.get_cover_lhs_animation(1, 2),
                Animation(left_line),
                FadeOut(comparison[1]),
                mediant_circles[-1].animate.set_color(GREY).fade(0.5),
            )
        )
        self.wait()
        new_left_text, target = comparison[::2]
        self.add(new_left_text, target)
        new_left_text.generate_target()
        new_left_text.target.set_color(RED)
        new_mediant = ft.get_fraction_from_numbers(2, 3)
        self.play(
            mediant.animate.set_color(RED),
            left_line.animate.move_to(mediant, coor_mask=X_COOR_MASK),
            FadeTransform(left_text, new_left_text.target),
            MoveToTarget(new_left_text),
            target.animate.move_to(new_mediant, coor_mask=X_COOR_MASK),
        )
        self.remove(new_left_text.target)
        self.wait()
        left_text = new_left_text

        # Continue on finding Farey mediants and pruning branches
        mediant = new_mediant
        mediant.generate_target()
        mediant.target.set_color(GREEN)
        circle = Circle(color=GREEN)
        circle.set_height(mediant.get_height() * 1.25)
        circle.move_to(mediant)
        mediant_circles.add(circle)
        self.play(
            FadeTransform(left_text.copy(), mediant.target),
            FadeTransform(right_text.copy(), mediant.target),
            MoveToTarget(mediant),
            ShowCreation(mediant_circles[-1]),
        )
        self.remove(mediant.target)
        self.wait()

        # 4/7 is a bit smaller than 2/3, so we keep the left subtree and replace the right bound
        comparison = VGroup(
            Tex('{4 \\over 7}'), Tex(' < '), Tex('{2 \\over 3}')
        )
        comparison.arrange(RIGHT)
        comparison[2].set_color(GREEN)
        comparison[0].set_color(YELLOW)
        comparison.next_to(target.get_right(), LEFT, buff=0)
        self.play(
            ReplacementTransform(mediant.copy(), comparison[2]),
            Write(comparison[1]),
            ReplacementTransform(target, comparison[0]),
        )
        self.wait()
        self.play(
            AnimationGroup(
                self.get_cover_rhs_animation(2, 3),
                Animation(right_line),
                FadeOut(comparison[1]),
                mediant_circles[-1].animate.set_color(GREY).fade(0.5),
            )
        )
        self.wait()
        target, new_right_text = comparison[::2]
        self.add(new_right_text, target)
        new_right_text.generate_target()
        new_right_text.target.set_color(BLUE)
        new_mediant = ft.get_fraction_from_numbers(3, 5)
        self.play(
            mediant.animate.set_color(BLUE),
            right_line.animate.move_to(mediant, coor_mask=X_COOR_MASK),
            FadeTransform(right_text, new_right_text.target),
            MoveToTarget(new_right_text),
            target.animate.move_to(new_mediant, coor_mask=X_COOR_MASK),
        )
        self.remove(new_right_text.target)
        self.wait()
        right_text = new_right_text

        # Repeat this process until we find the target
        mediant = new_mediant
        mediant.generate_target()
        mediant.target.set_color(GREEN)
        circle = Circle(color=GREEN)
        circle.set_height(mediant.get_height() * 1.25)
        circle.move_to(mediant)
        mediant_circles.add(circle)
        self.play(
            FadeTransform(left_text.copy(), mediant.target),
            FadeTransform(right_text.copy(), mediant.target),
            MoveToTarget(mediant),
            ShowCreation(mediant_circles[-1]),
        )
        self.remove(mediant.target)
        self.wait()
        comparison = VGroup(
            Tex('{4 \\over 7}'), Tex(' < '), Tex('{3 \\over 5}')
        )
        comparison.arrange(RIGHT)
        comparison[2].set_color(GREEN)
        comparison[0].set_color(YELLOW)
        comparison.next_to(target.get_right(), LEFT, buff=0)
        self.play(
            ReplacementTransform(mediant.copy(), comparison[2]),
            Write(comparison[1]),
            ReplacementTransform(target, comparison[0]),
        )
        self.wait()
        self.play(
            AnimationGroup(
                self.get_cover_rhs_animation(3, 5),
                Animation(right_line),
                FadeOut(comparison[1]),
                mediant_circles[-1].animate.set_color(GREY).fade(0.5),
            )
        )
        self.wait()
        target, new_right_text = comparison[::2]
        self.add(new_right_text, target)
        new_right_text.generate_target()
        new_right_text.target.set_color(BLUE)
        new_mediant = ft.get_fraction_from_numbers(4, 7)
        self.play(
            mediant.animate.set_color(BLUE),
            right_line.animate.move_to(mediant, coor_mask=X_COOR_MASK),
            FadeTransform(right_text, new_right_text.target),
            MoveToTarget(new_right_text),
            target.animate.move_to(new_mediant, coor_mask=X_COOR_MASK),
        )
        self.remove(new_right_text.target)
        self.wait()
        right_text = new_right_text

        mediant = new_mediant
        mediant.generate_target()
        mediant.target.set_color(GREEN)
        circle = Circle(color=GREEN)
        circle.set_height(mediant.get_height() * 1.25)
        circle.move_to(mediant)
        mediant_circles.add(circle)
        self.play(
            FadeTransform(left_text.copy(), mediant.target),
            FadeTransform(right_text.copy(), mediant.target),
            MoveToTarget(mediant),
            ShowCreation(mediant_circles[-1]),
        )
        self.remove(mediant.target)
        self.wait()

        self.add(mediant_circles)
        self.play(
            Uncreate(left_line), Uncreate(right_line),
            FadeOut(VGroup(left_text, right_text)),
            ft.get_all_fractions().animate.set_color(WHITE),
            mediant.animate.set_color(YELLOW),
            AnimationGroup(
                self.left_rect.animate.set_width(0, stretch=True).to_edge(LEFT, buff=0),
                self.right_rect.animate.set_width(0, stretch=True).to_edge(RIGHT, buff=0),
            ),
            mediant_circles[-1].animate.set_color(YELLOW).fade(0.5),
            Animation(mediant_circles[::2]),
            run_time=3,
        )
        self.wait()

        # Apart from getting to the target, we also obtained an approximate sequence
        self.play(
            AnimationGroup(*[
                Indicate(circle) for circle in mediant_circles
            ], lag_ratio=0.1),
            run_time=2,
        )
        self.wait()
        seq_text = VGroup(
            Tex('{4 \\over 7}', color=YELLOW),
            TexText('的近似序列：', color=YELLOW),
            Tex(
                '\\Big\{',
                '{1 \\over 2}', ',\\,',
                '{2 \\over 3}', ',\\,',
                '{3 \\over 5}', ',\\,',
                '{4 \\over 7}', '\\Big\}',
            ),
        )
        seq_text.arrange(RIGHT)
        seq_text.to_edge(UP)
        self.play(
            ReplacementTransform(target, seq_text[0], rate_func=squish_rate_func(smooth, 0, 0.5)),
            AnimationGroup(*[
                ReplacementTransform(
                    ft.get_fraction_from_numbers(p, q).copy()[0],
                    seq_text[2][ind]
                )
                for (p, q), ind in zip(
                    [(1, 2), (2, 3), (3, 5), (4, 7)],
                    [1, 3, 5, 7],
                )
            ], lag_ratio=0.2),
            Write(seq_text[1], rate_func=squish_rate_func(smooth, 0.2, 0.8)),
            Write(seq_text[2][::2], rate_func=squish_rate_func(smooth, 0.6, 1)),
            run_time=3,
        )
        self.wait()

        necessary_text = TexText('有必要', '吗？', color=BLUE)
        necessary_text.next_to(seq_text, RIGHT, buff=0.8)
        self.play(
            Write(necessary_text),
            ShowCreationThenDestructionAround(
                seq_text[2],
                surrounding_rectangle_config={'color': BLUE}
            )
        )
        self.wait()
        ex_mark = TexText('！', color=BLUE)
        ex_mark.next_to(necessary_text[1].get_left(), RIGHT, buff=0)
        self.play(
            FadeIn(ex_mark, 0.5 * DOWN),
            FadeOut(necessary_text[-1], 0.5 * DOWN),
        )
        self.wait()

    def get_cover_lhs_animation(self, numer, denom, **anim_kwargs):
        frac = self.ft.get_fraction_from_numbers(numer, denom)
        fill_opacity = self.left_rect.get_fill_opacity()
        new_cover_rect = FullScreenFadeRectangle(fill_opacity=fill_opacity)
        new_width = (frac.get_left() - LEFT_SIDE)[0] - 0.05
        new_cover_rect.stretch_to_fit_width(new_width)
        new_cover_rect.stretch_to_fit_height(self.left_rect.get_height())
        new_cover_rect.to_edge(LEFT, buff=0).to_edge(DOWN, buff=0)
        return Transform(self.left_rect, new_cover_rect, **anim_kwargs)

    def get_cover_rhs_animation(self, numer, denom, **anim_kwargs):
        frac = self.ft.get_fraction_from_numbers(numer, denom)
        fill_opacity = self.right_rect.get_fill_opacity()
        new_cover_rect = FullScreenFadeRectangle(fill_opacity=fill_opacity)
        new_width = (RIGHT_SIDE - frac.get_right())[0] - 0.05
        new_cover_rect.stretch_to_fit_width(new_width)
        new_cover_rect.stretch_to_fit_height(self.right_rect.get_height())
        new_cover_rect.to_edge(RIGHT, buff=0).to_edge(DOWN, buff=0)
        return Transform(self.right_rect, new_cover_rect, **anim_kwargs)


class FareyTreeAndItsPropertiesAddonP2(Scene):
    def construct(self):
        # Addon 1: connections correlate to tangencies in the Ford circles
        ff = FordFractal(axis_config={'y_offset': -3.8})
        self.add(ff)
        self.wait()
        circle_01 = ff.get_circle(0, 1)
        circle_12 = ff.get_circle(1, 2)
        circles = (circle_01, circle_12)
        tangent_point = ff.get_tangent_point(0, 1, 1, 2)
        dot = Dot(tangent_point, color=YELLOW)
        for c, color in zip(circles, [RED, BLUE]):
            c.save_state()
            c.generate_target()
            c.target.set_color(color)
            c.target.get_circle().set_fill(opacity=0.3)
        dot.generate_target()
        dot.scale(10).fade(1)
        self.play(AnimationGroup(*[MoveToTarget(c) for c in circles]))
        self.wait()
        self.play(MoveToTarget(dot))
        self.wait()
        self.play(
            AnimationGroup(*([c.animate.restore() for c in circles] + [FadeOut(dot)])),
        )
        self.wait()

        # Addon 2: Farey Tree is an infinite complete binary tree
        self.remove(*self.mobjects)
        self.wait()
        farey_chn = TexText('法里树', color=YELLOW)
        farey_chn.to_edge(UP)
        farey_eng = RemarkText(farey_chn, '(Farey Tree)', aligned_edge=ORIGIN, color=YELLOW)
        self.add(farey_chn, farey_eng)
        self.wait()

        inf_tree = TexText('——无限完全二叉树', color=YELLOW)
        inf_tree.next_to(farey_chn, RIGHT)
        self.play(Write(inf_tree))
        self.wait()
        self.play(FadeOut(inf_tree))
        self.wait()

        # Addon 3: Farey Tree is a BST
        bst = TexText('——二叉搜索树', color=YELLOW)
        bst.next_to(farey_chn, RIGHT)
        self.play(Write(bst))
        self.wait()
        self.play(FadeOut(bst), FadeOut(farey_chn), FadeOut(farey_eng))
        self.wait()

        # Addon 4: choices of branch point
        self.remove(*self.mobjects)
        self.wait()
        title = Tex('\\text{在} \\left( {a \\over b} ,\\, {c \\over d} \\right) \\text{中选取二分点}', color=YELLOW)
        title.scale(1.5)
        texts = VGroup(*[
            TexText(text, color=color)
            for text, color in zip(['法里和', '平均值'], [BLUE, RED])
        ])
        texts.arrange(RIGHT, buff=4)
        formulae = VGroup(
            Tex('{a+c \\over b+d}', color=BLUE),
            Tex('{1 \\over 2} \\left( {a \\over b} + {c \\over d} \\right)', color=RED),
        )
        remarks = VGroup(texts, formulae)
        for text, formula in zip(*remarks):
            formula.next_to(text, DOWN, buff=0.4)
        vs = TexText('vs.', color=GREY)
        vs.move_to(texts)
        remarks.add(vs)
        group = VGroup(title, remarks)
        group.arrange(DOWN, buff=0.8)
        group.set_width(FRAME_WIDTH - 2)
        self.play(FadeIn(group))
        self.wait()
        for formula in remarks[1]:
            self.play(Indicate(formula, color=formula.get_color()))
            self.wait()
        self.play(FadeOut(group))
        self.wait()


class SimpleGearTrainDemo(Scene):
    def construct(self):
        formula = Tex(
            '{\\text{转速}_\\text{A}', ' \\over ', '\\text{转速}_\\text{B}}',
            ' = ',
            '{\\text{齿数}_\\text{B}', ' \\over ', '\\text{齿数}_\\text{A}}',
        )
        for ind, color in zip([0, 6, 2, 4], [RED, RED, BLUE, BLUE]):
            formula[ind].set_color(color)
        formula.to_edge(UP)
        self.play(Write(formula))
        self.wait()

        # Examples with 1:3 and 1:2 gear ratio
        gear_nums1 = VGroup(
            Tex('\\text{齿数}_\\text{A}', ' = ', '10'),
            Tex('\\text{齿数}_\\text{B}', ' = ', '30'),
        )
        gear_nums2 = VGroup(
            Tex('\\text{齿数}_\\text{A}', ' = ', '10'),
            Tex('\\text{齿数}_\\text{B}', ' = ', '20'),
        )
        gear_speed1 = Tex('\\text{转速}_\\text{B}', ' = ', '{1 \\over 3}', '\\text{转速}_\\text{A}')
        gear_speed2 = Tex('\\text{转速}_\\text{B}', ' = ', '{1 \\over 2}', '\\text{转速}_\\text{A}')
        for text, color in zip([gear_nums1, gear_nums2], [BLUE, GREEN]):
            text.arrange(DOWN, aligned_edge=LEFT)
            text[0][::2].set_color(RED)
            text[1][::2].set_color(color)
            text.to_edge(UP)
        for text, color in zip([gear_speed1, gear_speed2], [BLUE, GREEN]):
            text.next_to(gear_nums1, RIGHT, buff=1)
            text[0].set_color(color)
            text[2][0].set_color(RED)
            text[2][-1].set_color(color)
            text[-1].set_color(RED)
        self.play(
            formula.animate.shift(4.5 * LEFT),
            FadeIn(gear_nums1),
        )
        self.wait()
        self.play(FadeIn(gear_speed1, shift=0.5 * RIGHT))
        self.wait()
        formula.generate_target()
        for ind in [2, 4]:
            formula.target[ind].set_color(GREEN)
        self.play(
            MoveToTarget(formula),
            FadeOut(gear_nums1, shift=0.5 * DOWN),
            FadeOut(gear_speed1, shift=0.5 * DOWN),
            FadeIn(gear_nums2, shift=0.5 * DOWN),
            FadeIn(gear_speed2, shift=0.5 * DOWN),
        )
        self.wait()
        self.play(FadeOut(VGroup(formula, gear_nums2, gear_speed2)))
        self.wait()

        # Driving gear info
        gear_red_info = VGroup(
            TexText('主动轮', '：'),
            VGroup(Tex('\\text{齿数}=', '10'), Tex('\\text{周期}=', '1\\text{分钟}')),
        )
        gear_orange_info = VGroup(
            TexText('从动轮', '：'),
            VGroup(Tex('\\text{齿数}=', '600'), Tex('\\text{周期}=', '60\\text{分钟}')),
        )
        gear_groups = (gear_red_info, gear_orange_info)
        for group, color in zip(gear_groups, [RED, GOLD]):
            group[1].arrange(DOWN, aligned_edge=LEFT)
            for mob in (group[0][0], group[1][0][1], group[1][1][1]):
                mob.set_color(color)
            group.arrange(RIGHT, aligned_edge=UP, buff=0.2)
            group.to_corner(UL)
        gear_orange_info.next_to(gear_red_info, DOWN, aligned_edge=LEFT, buff=1)
        gear_red_rect, gear_orange_rect = drive_rects = VGroup(*[
            SurroundingRectangle(group[1][1])
            for group in gear_groups
        ])
        drive_second_text, drive_minute_text = drive_texts = VGroup(*[
            TexText('驱动' + hand, color=YELLOW).next_to(rect, RIGHT)
            for hand, rect in zip(['秒针', '分针'], drive_rects)
        ])
        # Scale a bit to match the Blender animation
        scaling_mobs = VGroup(gear_red_info, gear_orange_info, drive_rects, drive_texts)
        scaling_mobs.scale(0.8).to_corner(UL)
        wrong_rate_remark = VGroup(
            TexText('我知道红色齿轮的周期其实是10秒，'),
            TexText('但是如果真的按照1分钟的周期做动画，'),
            TexText('转动效果就太慢了，先凑合一下吧...'),
        )
        wrong_rate_remark.arrange(DOWN, aligned_edge=LEFT)
        wrong_rate_remark.scale(0.5)
        wrong_rate_remark.next_to(gear_red_info, DOWN, aligned_edge=LEFT, buff=0.6)
        wrong_rate_remark.set_color(BLUE_B)
        self.play(
            FadeIn(VGroup(gear_red_info[0], gear_red_info[1][0]), shift=0.5 * DOWN)
        )
        self.wait()
        self.play(
            FadeIn(gear_red_info[1][1], shift=0.5 * DOWN),
            FadeIn(wrong_rate_remark),
        )
        self.wait()
        self.play(FadeOut(wrong_rate_remark))
        self.wait()
        self.play(
            ShowCreation(gear_red_rect),
            Write(drive_second_text)
        )
        self.wait()
        self.play(
            FadeTransform(gear_red_info[1][1].copy(), gear_orange_info[1][1]),
            FadeTransform(gear_red_rect.copy(), gear_orange_rect),
            FadeTransform(drive_second_text.copy(), drive_minute_text),
        )
        self.wait()
        self.play(
            FadeTransform(gear_red_info[0].copy(), gear_orange_info[0]),
            FadeTransform(gear_red_info[1][0].copy(), gear_orange_info[1][0]),
        )
        self.wait()

        # But this gear is too big to contain in the screen
        too_big_text = TexText('（大到画不下...）', color=BLUE_B)
        too_big_text.scale(0.6).next_to(gear_orange_info[1][0], RIGHT)
        self.play(Write(too_big_text))
        self.wait()
        self.play(FadeOut(too_big_text), FadeOut(scaling_mobs[1:]))
        self.wait()

        # Alternative way: Split up the ratio and use a gear train
        break_sixty = Tex(
            '{1 \\over 60}', ' = ', '{1 \\over 6}', ' \\times ', '{1 \\over 10}',
            color=YELLOW,
        )
        break_sixty.next_to(gear_red_info, DOWN, aligned_edge=LEFT, buff=1)
        gear_train_info = VGroup(
            TexText('红色', '齿轮：', '10', '齿'),
            TexText('橙色', '齿轮：', '60', '齿'),
            TexText('绿色', '齿轮：', '6', '齿'),
            TexText('蓝色', '齿轮：', '60', '齿'),
        )
        gear_train_info.arrange(DOWN, aligned_edge=LEFT)
        braces = VGroup(*[
            Brace(gear_train_info[2 * k:2 * k + 2], direction=RIGHT)
            for k in range(2)
        ])
        brace_texts = VGroup(*[
            Tex('1', ' \\over ', str(k)) for k in (6, 10)
        ])
        for brace, text in zip(braces, brace_texts):
            brace.put_at_tip(text)
        colors = [RED, GOLD, GREEN, BLUE]
        for info, color in zip(gear_train_info, colors):
            info[::2].set_color(color)
        for text, color1, color2 in zip(brace_texts, colors[::2], colors[1::2]):
            text[0].set_color(color1)
            text[-1].set_color(color2)
        gear_train_group = VGroup(gear_train_info, braces, brace_texts)
        gear_train_group.scale(0.8)
        gear_train_group.next_to(break_sixty, DOWN, aligned_edge=LEFT, buff=0.6)
        result = Tex('\\text{蓝色}', '\\text{齿轮：周期}=', '60\\text{分钟}')
        result.scale(0.8)
        result[::2].set_color(BLUE)
        result.next_to(gear_train_group, DOWN, aligned_edge=LEFT, buff=0.6)
        self.play(Write(break_sixty))
        self.wait()
        self.play(
            AnimationGroup(*[
                GrowFromEdge(info, LEFT)
                for info in gear_train_info
            ], lag_ratio=0.2)
        )
        self.wait()
        for brace, text in zip(braces, brace_texts):
            self.play(GrowFromCenter(brace))
            self.wait()
            self.play(Write(text))
            self.wait()
        self.play(Indicate(break_sixty))
        self.wait()
        self.play(FadeIn(result, shift=DOWN))
        self.wait()
        all_mobs = VGroup(gear_red_info, break_sixty, gear_train_group, result)
        self.play(all_mobs.animate.next_to(LEFT_SIDE, LEFT, coor_mask=X_COOR_MASK))
        self.wait()

        # It works because 60 can be broken down into smaller factors like 6 and 10
        break_720 = Tex(
            '{1 \\over 720}', ' = ', '{1 \\over 6}', ' \\times ', '{1 \\over 10}', ' \\times ', '{1 \\over 12}',
        )
        break_720_gear = Tex(
            '{1 \\over 720}', ' = ', '{10 \\over 60}', ' \\times ', '{6 \\over 60}', ' \\times ', '{6 \\over 72}',
        )
        for mob in (break_720, break_720_gear):
            mob.to_edge(UP)
            mob[0].set_color(YELLOW)
        colors = [RED, GOLD, GREEN, BLUE, '#4440D3', PINK]
        for k, mob in enumerate(break_720[2::2]):
            mob[0].set_color(colors[2 * k])
            mob[2:].set_color(colors[2 * k + 1])
        for k, mob in enumerate(break_720_gear[2::2]):
            if k == 0:
                mob[:2].set_color(colors[2 * k])
                mob[3:].set_color(colors[2 * k + 1])
            else:
                mob[0].set_color(colors[2 * k])
                mob[2:].set_color(colors[2 * k + 1])
        self.play(
            AnimationGroup(*[
                GrowFromCenter(mob) for mob in break_720
            ], lag_ratio=0.05)
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                FadeTransform(source, target)
                for source, target in zip(break_720, break_720_gear)
            ], lag_ratio=0.05)
        )
        self.wait()
        self.play(FadeOut(break_720_gear))
        self.wait()

        # Other examples of small factors
        examples = VGroup(
            Tex('1', '\\text{周}=', '7', '\\text{天}', ''),
            Tex('1', '\\text{天}=', '24', '\\text{小时} \\quad\\quad', '(24 = 2^3 \\times 3)'),
            Tex('1', '\\text{小时}=', '60', '\\text{分钟} \\quad', '(60 = 2^2 \\times 3 \\times 5)'),
        )
        examples.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        for mob in examples:
            mob[::2].set_color(YELLOW)
        self.play(
            AnimationGroup(*[
                FadeIn(mob) for mob in examples
            ], lag_ratio=0.4)
        )
        self.wait()
        self.play(FadeOut(examples))
        self.wait()


class ApproximateATrickyRatioP2(Scene):
    def construct(self):
        # Show our task
        drive_gear_info = VGroup(
            TexText('主动轮'),
            Tex('\\text{周期}=', '1\\text{天}')
        )
        passive_gear_info = VGroup(
            TexText('从动轮'),
            Tex('\\text{周期}=', '1\\text{年}')
        )
        gear_info_group = VGroup(drive_gear_info, passive_gear_info)
        for gear_info, color in zip(gear_info_group, [RED, GOLD]):
            gear_info[0].scale(1.25).set_color(color)
            gear_info[1][-1].set_color(color)
            gear_info.arrange(DOWN)
            gear_info.move_to(3 * LEFT if color == RED else 3 * RIGHT)
        self.play(GrowFromCenter(drive_gear_info))
        self.wait()
        self.play(FadeTransform(drive_gear_info.copy(), passive_gear_info))
        self.wait()

        # 1 day is exactly 24 hours, while 1 year is not 365 or 366 days
        drive_line = Underline(drive_gear_info[1][-1], color=RED)
        drive_period = TexText('24小时', color=RED)
        drive_period.scale(0.8).next_to(drive_line, DOWN, buff=0.5)
        passive_line = Underline(passive_gear_info[1][-1], color=GOLD)
        wrong_passive_period = TexText('365天?', ' \\quad ', '366天?', color=GOLD)
        true_passive_period = TexText('365天5小时49分钟', color=GOLD)
        for mob in (wrong_passive_period, true_passive_period):
            mob.scale(0.8).next_to(passive_line, DOWN, buff=0.5)
        self.play(
            Write(drive_line, rate_func=squish_rate_func(smooth, 0, 0.6)),
            GrowFromPoint(
                drive_period, drive_line.get_center(),
                rate_func=squish_rate_func(smooth, 0.4, 1),
            ),
        )
        self.wait()
        self.play(
            Write(passive_line, rate_func=squish_rate_func(smooth, 0, 0.6)),
            AnimationGroup(*[
                GrowFromPoint(mob, passive_line.get_center())
                for mob in wrong_passive_period
            ], lag_ratio=0.2, rate_func=squish_rate_func(smooth, 0.4, 1)),
        )
        self.wait()
        wrong_passive_period.generate_target()
        wrong_passive_period.target.set_color(GREY).fade(0.3)
        self.play(
            MoveToTarget(wrong_passive_period, lag_ratio=0.1)
        )
        self.wait()
        self.play(FadeTransform(wrong_passive_period, true_passive_period))
        self.wait()
        minute_periods = VGroup(*[
            TexText(num, '分钟', color=color).next_to(target_info[1][-1].get_left(), RIGHT, buff=0)
            for num, color, target_info in zip(
                ['1440', '525949'], [RED, GOLD], gear_info_group
            )
        ])
        for text, gear_info, underline in zip(minute_periods, gear_info_group, [drive_line, passive_line]):
            self.play(
                FadeTransform(gear_info[1][-1], text),
                FadeOut(underline),
            )
            self.wait()

        # Objective: construct a 1440/525949 ratio gear train
        arrow = CurvedArrow(
            drive_gear_info[0].get_bounding_box_point(UR),
            passive_gear_info[0].get_bounding_box_point(UL),
            angle=-PI / 2,
        )
        arrow.scale(0.9, about_edge=UP)
        arrow.set_color(YELLOW)
        ratio_target = Tex('\\text{转速} \\times', '{1440', ' \\over ', '525949}')
        ratio_target.next_to(arrow, UP, buff=0.3)
        for mob, color in zip(ratio_target, [WHITE, RED, WHITE, GOLD]):
            mob.set_color(color)
        self.play(ShowCreation(arrow))
        self.wait()
        self.play(
            Write(ratio_target[::2], rate_func=squish_rate_func(smooth, 0.2, 1)),
            AnimationGroup(*[
                FadeTransform(source[0].copy(), target)
                for source, target in zip(minute_periods, ratio_target[1::2])
            ], lag_ratio=0.1, rate_func=squish_rate_func(smooth, 0, 0.8)),
            run_time=2,
        )
        self.wait()
        drive_gear_info = VGroup(
            drive_gear_info[0],
            VGroup(drive_gear_info[1][0], minute_periods[0])
        )
        passive_gear_info = VGroup(
            passive_gear_info[0],
            VGroup(passive_gear_info[1][0], minute_periods[1])
        )
        self.play(
            AnimationGroup(*[
                gear_info[1].animate.next_to(gear_info[0], DOWN)
                for gear_info in (drive_gear_info, passive_gear_info)
            ]),
            FadeOut(VGroup(drive_period, true_passive_period))
        )
        self.wait()

        # But there's a huge problem: 525949 is prime
        prime_arrow = Arrow(ORIGIN, 1.5 * LEFT, color=YELLOW)
        prime_text = TexText('素数!', color=YELLOW)
        prime_arrow.next_to(ratio_target[-1], RIGHT)
        prime_text.next_to(prime_arrow, RIGHT)
        self.play(
            ShowCreationThenDestructionAround(ratio_target[-1]),
            GrowFromPoint(prime_arrow, prime_arrow.get_right()),
            Write(prime_text)
        )
        self.wait()
        self.play(FadeOut(VGroup(prime_arrow, prime_text)))
        self.wait()

        # This forces us to approximate 1440/525949 with another rational,
        # one that's easier for building a gear train
        approx_target = Tex('\\text{近似}', '\\Big(', '\\text{转速} \\times', '{1440', ' \\over ', '525949}', '\\Big)')
        approx_target.to_edge(UP)
        for mob, color in zip(approx_target, [WHITE, WHITE, WHITE, RED, WHITE, GOLD, WHITE]):
            mob.set_color(color)
        self.play(
            Write(approx_target[:2]),
            Write(approx_target[-1]),
            ratio_target.animate.move_to(approx_target[2:-1]),
        )
        self.remove(ratio_target)
        self.add(approx_target)
        self.wait()

        approx_text = Tex('\\text{近似}', '{1440', ' \\over ', '525949}')
        approx_text.to_edge(UP)
        for mob, color in zip(approx_text, [WHITE, RED, WHITE, GOLD]):
            mob.set_color(color)
        self.play(
            AnimationGroup(*[
                FadeOut(approx_target[k])
                for k in (1, 2, 6)
            ]),
            AnimationGroup(*[
                approx_target[i].animate.move_to(approx_text[j])
                for i, j in zip([0, 3, 4, 5], [0, 1, 2, 3])
            ]),
            FadeOut(VGroup(drive_gear_info, passive_gear_info, arrow)),
        )
        self.remove(approx_target)
        self.add(approx_text)
        self.wait()

        brocot_name = TexText('Achille Brocot', color=YELLOW)
        brocot_name.scale(1.2).to_corner(UL)
        brocot_remark = RemarkText(
            brocot_name, '（无照片，只能放个名字...）',
            scale_factor=0.6, color=YELLOW_A
        )
        self.play(
            Write(brocot_name, rate_func=squish_rate_func(smooth, 0, 0.6)),
            FadeIn(brocot_remark, rate_func=squish_rate_func(smooth, 0.4, 1)),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeOut(VGroup(brocot_name, brocot_remark)),
            approx_text.animate.set_color(YELLOW),
        )
        self.wait()


class AnExampleOfApproximationPartAP2(Scene):
    def construct(self):
        # Match the last scene
        title = Tex('\\text{近似}', '{1440', ' \\over ', '525949}', color=YELLOW)
        title.to_edge(UP)
        self.add(title)

        target = FareyRational(1440, 525949)
        # First approximation
        left_bound = FareyRational(0, 1)
        right_bound = FareyRational(1, 1)
        bisect_point = left_bound.farey_add(right_bound)
        seq_text = VGroup(TexText('近似序列：'))
        seq_text.next_to(ORIGIN, LEFT, buff=0).next_to(title, DOWN, buff=0.8, coor_mask=Y_COOR_MASK)
        line_config = {'stroke_width': 3, 'color': GREY}
        number_line = Line(5 * LEFT, 5 * RIGHT, **line_config)
        ticks = VGroup(*[
            Line(0.25 * UP, 0.25 * DOWN, **line_config).move_to(pos)
            for pos in [number_line.get_left(), number_line.get_center(), number_line.get_right()]
        ])
        line_group = VGroup(number_line, *ticks.submobjects)
        line_group.shift(2 * DOWN)
        fractions = VGroup(*[
            Tex(f'{{{fr.p} \\over {fr.q}}}')
            for fr in (left_bound, bisect_point, right_bound)
        ])
        point_names = VGroup(*[
            TexText(text, color=YELLOW)
            for text in ('左端点', '二分点', '右端点')
        ])
        for tick, frac, name in zip(ticks, fractions, point_names):
            frac.next_to(tick, UP, buff=0.3)
            name.next_to(frac, UP, buff=0.3)

        self.play(
            *[GrowFromCenter(line) for line in line_group],
            Write(seq_text),
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                AnimationGroup(Write(frac), GrowFromCenter(name))
                for frac, name in zip(fractions[::2], point_names[::2])
            ])
        )
        self.wait()
        self.play(
            Write(point_names[1]),
            FadeTransform(fractions[0].copy(), fractions[1]),
            FadeTransform(fractions[2].copy(), fractions[1]),
        )
        self.wait()

        # Keep the right section and record the bisection point
        left_section = VGroup(
            Line(5 * LEFT, ORIGIN),
            Tex('{1440 \\over 525949} \\text{在这边}')
        )
        right_section = VGroup(
            Line(ORIGIN, 5 * RIGHT),
            Tex('{1440 \\over 525949} \\text{在这边}')
        )
        for section in (left_section, right_section):
            section[0].set_stroke(width=20)
            section[1].scale(0.75)
            section[1].next_to(section[0], DOWN)
            section.set_color(GREEN)
            section.shift(2 * DOWN)

        self.play(FadeIn(left_section))
        self.wait()

        new_seq_text = seq_text
        new_seq_text.add(fractions[1])
        new_seq_text.generate_target()
        new_seq_text.target.arrange(RIGHT, buff=0.4)
        new_seq_text.target.next_to(seq_text[:-1].get_right(), LEFT, buff=0)

        new_right_fraction = fractions[1].copy()
        self.play(
            MoveToTarget(new_seq_text),
            new_right_fraction.animate.next_to(ticks[2], UP, buff=0.3),
            FadeOut(fractions[2]),
            FadeOut(left_section),
        )
        self.wait()

        # Start from 0/1 and 1/2, continue playing without interference
        left_bound = FareyRational(0, 1)
        right_bound = FareyRational(1, 2)
        fractions = VGroup(fractions[0], VMobject(), new_right_fraction)  # Placeholder
        counter = 0
        while True:
            # Calculate bisection point
            bisect_point = left_bound.farey_add(right_bound)
            bisect_fraction = Tex(f'{{{bisect_point.p} \\over {bisect_point.q}}}')
            # Set run time of each run
            if counter < 5 or counter > 380:
                run_time = 1. / 3
            elif bisect_point.p == 1:
                run_time = 1. / 60
            else:
                run_time = 1. / 15
            # Create new fractions group and arrange it
            fractions = VGroup(fractions[0], bisect_fraction, fractions[2])
            for tick, frac in zip(ticks, fractions):
                frac.next_to(tick, UP, buff=0.3)
            # Show bisecting fraction
            self.play(
                FadeTransform(fractions[0].copy(), fractions[1]),
                FadeTransform(fractions[2].copy(), fractions[1]),
                run_time=run_time,
            )
            # Show which section the target is in
            if bisect_point != target:
                target_section = left_section if bisect_point > target else right_section
                self.play(
                    FadeIn(target_section),
                    run_time=run_time,
                )
            # self.wait()
            # Keep the right section and record the bisection point
            seq_text.add(fractions[1])
            seq_text.generate_target()
            seq_text.target.arrange(RIGHT, buff=0.4)
            seq_text.target.next_to(seq_text[:-1].get_right(), LEFT, buff=0)
            new_bound = fractions[2] if bisect_point > target else fractions[0]
            ref_tick = ticks[2] if bisect_point > target else ticks[0]
            new_bound_fraction = fractions[1].copy()
            if bisect_point != target:
                self.play(
                    MoveToTarget(seq_text),
                    new_bound_fraction.animate.next_to(ref_tick, UP, buff=0.3),
                    FadeOut(new_bound),
                    FadeOut(target_section),
                    run_time=run_time,
                )
            else:
                self.play(
                    MoveToTarget(seq_text),
                    run_time=run_time,
                )
            # self.wait()
            # Truncate seq_text if it's too long
            if len(seq_text) > 12:
                self.remove(seq_text[:-12])
                seq_text = seq_text[-12:]
            # Update left and right bounds, as well as the fractions group placeholder
            # Break if it reaches the target
            if bisect_point > target:
                right_bound = bisect_point
                fractions = VGroup(fractions[0], VMobject(), new_bound_fraction)  # Placeholder
            elif bisect_point < target:
                left_bound = bisect_point
                fractions = VGroup(new_bound_fraction, VMobject(), fractions[2])  # Placeholder
            else:
                break
            counter += 1

        # After that... we switch scene, because rendering this one is already time-consuming
        self.wait()
        self.play(
            seq_text.animate.move_to(DOWN).to_edge(RIGHT),
            FadeOut(VGroup(point_names, fractions[::2], ticks, number_line)),
        )
        self.wait()


class AnExampleOfApproximationPartBP2(Scene):
    def construct(self):
        # Match the last scene
        title = Tex('\\text{近似}', '{1440', ' \\over ', '525949}', color=YELLOW)
        title.to_edge(UP)
        target = FareyRational(1440, 525949)
        approx_seq = get_approximation_sequence(target, use_tuple=True)
        seq_text = VGroup(*[
            Tex(f'{{{p} \\over {q}}}')
            for p, q in approx_seq[-8:]
        ])
        seq_text.arrange(RIGHT, buff=0.4)
        seq_text.move_to(DOWN).to_edge(RIGHT)
        self.add(title, seq_text)

        # we check each approximation from the end to see if it consists of small factors
        factorizations = VGroup(
            Tex('2^3 \\times 7^2 \\over 3 \\times 5^2 \\times 23 \\times 83'),
            Tex('5^2 \\times 17 \\over 2^2 \\times 151 \\times 257'),
            Tex('2 \\times 229 \\over 409^2'),
            Tex('491 \\over 2 \\times 3^7 \\times 41'),
            Tex('13 \\times 73 \\over 5 \\times 181 \\times 383'),
        )
        for f, frac, direction in zip(factorizations, seq_text[-6:-1], it.cycle([UP, DOWN])):
            f.next_to(frac, direction, buff=0.8)
        mark_green_indices = (
            (0, 1, 3, 4, 6, 8, 9, 11, 12, 14, 15),
            (0, 1, 3, 4, 6, 7),
            (0, ),
            (4, 6, 7, 9, 10),
            (0, 1, 3, 4, 6),
        )
        mark_red_indices = (
            (),  # This one is a perfect choice, yay!
            (9, 10, 11, 13, 14, 15),
            (2, 3, 4, 6, 7, 8, 9),
            (0, 1, 2),
            (8, 9, 10, 12, 13, 14),
        )
        for f, red_indices, green_indices in zip(factorizations, mark_red_indices, mark_green_indices):
            for ind in red_indices:
                f[0][ind].set_color(RED)
            for ind in green_indices:
                f[0][ind].set_color(GREEN)

        self.play(seq_text[-1].animate.fade(0.7))
        self.wait()
        for k in range(5):
            curr_frac = seq_text[-(k + 2)]
            curr_factor = factorizations[-(k + 1)]
            self.play(
                GrowFromPoint(curr_factor, curr_frac.get_center()),
                run_time=0.5
            )
            self.wait()
            if k < 4:
                self.play(
                    curr_frac.animate.fade(0.8),
                    curr_factor.animate.fade(0.8),
                    run_time=0.5
                )
                self.wait()
            else:
                self.play(curr_frac.animate.set_color(GREEN))
                self.wait()
                seq_text.remove(curr_frac)
                factorizations.remove(curr_factor)

        best_frac = curr_frac
        best_factor = curr_factor
        other_mobs = VGroup(seq_text, factorizations)
        self.play(FadeOut(other_mobs))
        self.wait()

        # This fraction is very close to the target, while it only has prime factors less than 100
        close_text = Tex(
            '{392 \\over 143175}', ' - ', '{1440', ' \\over ', '525949}',
            '\\approx {1 \\over 10^{10}}',
        )
        close_text.next_to(best_frac.get_left(), RIGHT, buff=0)
        close_text[0].set_color(GREEN)
        close_text[2:-1].set_color(YELLOW)
        target_copy = title[1:].copy()
        target_copy.generate_target()
        target_copy.target.move_to(close_text[2:-1])
        self.play(
            ReplacementTransform(best_frac[0], close_text[0], rate_func=squish_rate_func(smooth, 0, 0.2)),
            Write(close_text[1], rate_func=squish_rate_func(smooth, 0, 0.2)),
            MoveToTarget(target_copy, rate_func=squish_rate_func(smooth, 0, 0.6)),
            Write(close_text[-1], rate_func=squish_rate_func(smooth, 0.5, 1)),
            run_time=2,
        )
        self.remove(target_copy)
        self.add(close_text)
        self.wait()
        self.play(
            AnimationGroup(*[
                Indicate(best_factor[0][ind], color=GREEN)
                for ind in mark_green_indices[0]
            ], lag_ratio=0.05),
            run_time=2
        )
        self.wait()

        # Split into 4 fractions that are suitable for gear train construction
        split_fractions = Tex(
            '{2 \\over 3}', '\\times',
            '{4 \\over 25}', '\\times',
            '{7 \\over 23}', '\\times',
            '{7 \\over 83}',
        )
        expand_fractions = Tex(
            '{20 \\over 30}', '\\times',
            '{8 \\over 50}', '\\times',
            '{14 \\over 46}', '\\times',
            '{7 \\over 83}',
        )
        for fs in (split_fractions, expand_fractions):
            fs.move_to(best_factor)
        self.play(FadeTransform(best_factor, split_fractions))
        self.wait()
        self.play(FadeTransformPieces(split_fractions, expand_fractions))
        self.wait()
        result = Tex(
            '{392 \\over 143175}', ' = ',
            '{20 \\over 30}', '\\times',
            '{8 \\over 50}', '\\times',
            '{14 \\over 46}', '\\times',
            '{7 \\over 83}',
        )
        result.to_corner(DL)
        # Final arrangement
        lhs = close_text[0]
        lhs.generate_target()
        lhs.target.move_to(result[0]).set_color(WHITE)
        rhs = expand_fractions
        rhs.generate_target()
        for source, target in zip(rhs.target, result[2:]):
            source.move_to(target)
        colors = [
            RED, GOLD,
            GREEN, BLUE,
            '#4E5BC3', PINK,
            YELLOW, '#8759C1'
        ]
        start_end_indices = [
            (0, 2), (-2, None),
            (0, 1), (-2, None),
            (0, 2), (-2, None),
            (0, 1), (-2, None)
        ]
        for k, color in enumerate(colors):
            sind, eind = start_end_indices[k]
            mob = rhs.target[k // 2 * 2]
            mob[sind:eind].set_color(color)
        bg_rect = FullScreenRectangle(fill_color='#404040')
        self.play(
            FadeIn(bg_rect),
            AnimationGroup(
                MoveToTarget(lhs),
                MoveToTarget(rhs),
                Write(result[1]),
                lag_ratio=0.1,
            ),
            FadeOut(
                VGroup(title, close_text[1:]),
                rate_func=squish_rate_func(smooth, 0, 0.6),
            ),
            run_time=2,
        )
        self.wait()

        # How well is this approximation?
        closeness_text = Tex('1\\text{天}', '\\rightarrow', '1\\text{年}', '-1.2\\text{秒}')
        closeness_text.next_to(lhs, UP, aligned_edge=LEFT, buff=0.5)
        for mob, color in zip(closeness_text, [colors[0], WHITE, colors[-1], colors[-1]]):
            mob.set_color(color)
        self.play(Write(closeness_text[0]))
        self.wait()
        self.play(
            GrowFromPoint(closeness_text[1], closeness_text[0].get_center()),
            FadeTransform(closeness_text[0].copy(), closeness_text[2])
        )
        self.wait()
        self.play(Write(closeness_text[3]))
        self.wait()


class AnExampleOfApproximationAddonP2(Scene):
    def construct(self):
        # Addon 1: approximation sequence will always end
        # Addon 2: the approximation's denominator is always smaller than the target's
        # Addon 3: how to speed up the process using division algorithm?
        # Addon 5: we only covered rationals
        end_text = TexText('序列必然会终止')
        small_denom_text = TexText('分母总是比目标值的小')
        how_to_speed_up_text = TexText('思考：如何加速这个过程?')
        rational_text = TexText('$\\gets$ 有理数')
        for text in (end_text, small_denom_text, how_to_speed_up_text, rational_text):
            text.to_edge(LEFT)
            text.set_color(BLUE)
            self.play(Write(text))
            self.wait()
            self.play(FadeOut(text))
            self.wait()

        # Addon 4: Stern-Brocot tree
        title = TexText('Stern-Brocot树', color=YELLOW)
        title.to_edge(UP)
        sb_tree = SternBrocotTree()
        sb_tree.set_width(FRAME_WIDTH - 1)
        self.add(title, sb_tree)
        self.wait()


class FromRationalsToIrrationalsP2(Scene):
    def construct(self):
        title = Tex('\\text{近似}', '{\\pi \\over 4}', color=YELLOW)
        title.to_edge(UP)
        self.add(title)
        self.wait()

        approx_text = TexText('近似序列：')
        approx_seq = get_approximation_sequence(PI / 4)
        seq_mob_list = [approx_text]
        for fr in approx_seq[2:]:
            frac = Tex(f'{{{fr.p} \\over {fr.q}}}')
            seq_mob_list.append(frac)
        seq_mob_list.append(Tex('\\cdots'))
        approx_fracs = VGroup(*seq_mob_list)
        approx_fracs.arrange(RIGHT, buff=0.5)
        shift_vec = ORIGIN - approx_fracs[0].get_right()
        approx_fracs.shift(shift_vec)

        def set_opacity_based_on_x(group):
            for mob in group:
                x = mob.get_center()[0] - 4
                if x > 0:
                    opacity = 0
                elif -1 < x < 0:
                    opacity = abs(x)
                else:
                    opacity = 1
                mob.set_opacity(opacity)

        # Use a virtual point to move approx_fracs
        point = VMobject().move_to(approx_fracs.get_left())
        approx_fracs.add_updater(set_opacity_based_on_x)
        approx_fracs.add_updater(lambda group: group.next_to(point, RIGHT, buff=0))
        approx_fracs.suspend_updating()
        self.play(FadeIn(approx_fracs))
        approx_fracs.resume_updating()
        self.wait()
        shift_vec = 3 * RIGHT - approx_fracs.get_right()
        self.play(
            point.animate.shift(shift_vec),
            run_time=15,
        )
        approx_fracs.clear_updaters()
        self.wait()

        # Infinite is easy to grasp, but 'good enough'?
        infinite_text = TexText('无限', color=GREEN)
        cdots_rect = SurroundingRectangle(approx_fracs[-1], buff=0.25, color=GREEN)
        infinite_text.next_to(cdots_rect, DOWN, buff=1)
        good_enough_text = TexText('“足够良好”?', color=RED)
        good_enough_text.next_to(infinite_text, LEFT, buff=0.8)
        self.play(
            ShowCreationThenDestruction(cdots_rect),
            Write(infinite_text),
        )
        self.wait()
        self.play(FadeIn(good_enough_text, shift=DOWN))
        self.wait()
        pi_text = Tex('\\pi')
        pi_copy = title[1][0].copy()
        self.play(
            FadeOut(Group(*self.mobjects)),
            pi_copy.animate.move_to(pi_text).set_color(WHITE),
        )
        self.wait()


class ExplainGoodEnoughWithPiP2(Scene):
    def construct(self):
        # Match the last scene
        pi_text = Tex('\\pi', ' = 3.14159265358979 \\dots')
        pi_text.shift(ORIGIN - pi_text[0].get_center())
        self.add(pi_text[0])
        self.wait()

        # Two famous approximations of pi
        statue_zu = ImageMobject('ZuChongzhi.jpg')
        portrait_archimedes = ImageMobject('Archimedes.png')
        ratio_22_7 = Tex('\\text{约率}', '{22 \\over 7}', ' = 3.14285714285714 \\dots')
        ratio_355_113 = Tex('\\text{密率}', '{355 \\over 113}', ' = 3.14159292035398 \\dots')
        ratios = VGroup(ratio_355_113, ratio_22_7)
        images = Group(statue_zu, portrait_archimedes)
        remarks = VGroup()
        group_list = []
        for ratio, image, direction, text in zip(ratios, images, [UP, DOWN], ['祖冲之', '阿基米德']):
            ratio.next_to(pi_text, direction, index_of_submobject_to_align=-1, aligned_edge=LEFT, buff=1)
            image.set_width(2.5).move_to(4 * LEFT)
            image.next_to(ratio, LEFT, aligned_edge=-direction, coor_mask=Y_COOR_MASK)
            image.shift(-0.5 * direction)
            remark = RemarkText(image, text, aligned_edge=-direction, direction=LEFT, color=BLUE_A, buff=0.4)
            remarks.add(remark)
            group_list.append(Group(ratio[:-1], image, remark))
        for group, direction in zip(group_list[::-1], [DOWN, UP]):
            self.play(
                GrowFromPoint(group[0], pi_text[0].get_center()),
                FadeIn(group[1:], shift=direction),
            )
            self.wait()

        # Show digits of 3 numbers
        number_texts = VGroup(ratio_355_113, pi_text, ratio_22_7)
        self.play(
            AnimationGroup(*[
                AnimationGroup(*[
                    FadeIn(digit, shift=RIGHT)
                    for digit in text[-1]
                ], lag_ratio=0.02)
                for text in number_texts
            ], lag_ratio=0.2),
            FadeOut(Group(images, remarks)),
            run_time=2
        )
        self.wait()

        # Close ratio is closer to pi, as 6 digits are the same
        for text in number_texts:
            text.save_state()
            text.generate_target()
        first_comps = VGroup(ratio_355_113, pi_text)
        second_comps = VGroup(ratio_22_7, pi_text)
        for text in first_comps:
            VGroup(text.target[:-1], text.target[-1][:9]).set_color(GREEN)
        ratio_22_7.target.set_color(GREY)
        same_digits = VGroup(*[
            TexText(text, color=YELLOW).move_to(4 * LEFT).next_to(comp, LEFT, buff=0.8, coor_mask=Y_COOR_MASK)
            for text, comp in zip(['6位小数一致', '2位小数一致'], [first_comps, second_comps])
        ])
        same_6_digits, same_2_digits = same_digits
        self.play(
            *[MoveToTarget(text) for text in number_texts],
            Write(same_6_digits),
        )
        self.wait()
        self.play(Indicate(ratio_355_113[1][-3:], rate_func=there_and_back_with_pause), run_time=2)
        self.wait()

        # Approximate ratio is slightly off
        # Still, 2 digits are identical with denominator as small as 7
        for text in number_texts:
            text.generate_target()
        ratio_355_113.target.set_color(GREY)

        for text in second_comps:
            text.target.set_color(WHITE)
            VGroup(text.target[:-1], text.target[-1][:5]).set_color(GREEN)
        self.play(
            *[MoveToTarget(text) for text in number_texts],
            FadeOut(same_6_digits),
            Write(same_2_digits),
        )
        self.wait()
        self.play(Indicate(ratio_22_7[1][-1], rate_func=there_and_back_with_pause), run_time=2)
        self.wait()

        # Same goes for other irrationals: it's (relatively) easy to approximate with a big denominator
        self.play(
            *[text.animate.restore() for text in number_texts],
            FadeOut(same_2_digits),
        )
        self.wait()
        colors = [BLUE, RED]
        denom_groups = VGroup(*[
            VGroup(
                TexText(text, color=color)
                .move_to(4.5 * LEFT)
                .scale(0.75)
                .next_to(ratio, LEFT, buff=0.8, coor_mask=Y_COOR_MASK)
            )
            for text, color, ratio in zip(
                ['分母大，近似相对容易', '分母小，近似相对困难'], colors, [ratio_355_113, ratio_22_7]
            )
        ])
        for group in denom_groups:
            underline = Underline(group[0][0][-4:-2], color=group[0].get_color())
            group.add(underline)
        for text, denom_ind, color in zip(number_texts[::2], [-3, -1], colors):
            text.generate_target()
            text.target[1][denom_ind:].set_color(color)
        self.play(
            *[MoveToTarget(text) for text in number_texts[::2]],
            *[Write(group) for group in denom_groups],
            run_time=2,
        )
        self.wait()

        # Describe "good enough" with mathematics
        title = TexText('用', '有理数', '“良好”近似', '无理数')
        rational = Tex('{p \\over q}')
        irrational = Tex('\\alpha')
        colors = [TEAL, YELLOW, MAROON_B]
        for num, mob, color in zip([rational, VMobject(), irrational], title[1:], colors):
            num.next_to(mob, DOWN, buff=0.5)
            VGroup(num, mob).set_color(color)
        prev_mobs = VGroup(denom_groups, number_texts)
        prev_mobs.generate_target()
        prev_mobs.target.scale(0.3).center().fade(1)
        curr_mobs = VGroup(title, rational, irrational)
        curr_mobs.center().shift(UP)
        curr_mobs.generate_target()
        curr_mobs.scale(0.3).fade(1).center()
        self.play(
            MoveToTarget(prev_mobs, rate_func=squish_rate_func(smooth, 0, 0.7)),
            MoveToTarget(curr_mobs, rate_func=squish_rate_func(smooth, 0.3, 1)),
            run_time=2,
        )
        self.remove(prev_mobs)
        self.wait()
        self.play(Indicate(title[-2]))
        self.wait()

        formula = Tex('\\bigg| ', '{p \\over q}', ' - ', '\\alpha', ' \\bigg|', ' < ', '\\text{阈值}')
        formula.next_to(title, DOWN, buff=0.8)
        for k, color in zip([1, slice(-2, None), 3], colors):
            formula[k].set_color(color)
        self.play(
            ReplacementTransform(rational[0], formula[1], rate_func=squish_rate_func(smooth, 0, 0.5)),
            ReplacementTransform(irrational[0], formula[3], rate_func=squish_rate_func(smooth, 0.1, 0.6)),
            AnimationGroup(*[
                Write(formula[ind]) for ind in (2, 5, 6, 0, 4)
            ], lag_ratio=0.2, rate_func=squish_rate_func(smooth, 0.4, 1)),
            run_time=2,
        )
        self.add(formula)
        self.wait()

        # q influences the threshold: the bigger the q, the smaller the threshold
        self.play(Indicate(formula[1][-1], scale_factor=1.5, color=BLUE))
        self.wait()
        remark = TexText('$q$', '越大', '，近似', '$\\alpha$', '越容易', '，', '阈值', '应该越小')
        remark.scale(0.8)
        remark.next_to(formula, DOWN, buff=0.8)
        for ind, color in zip([0, 6, 3], colors):
            remark[ind].set_color(color)
        self.play(
            ReplacementTransform(formula[1][-1].copy(), remark[0][0]),
            Write(remark[1]),
        )
        self.wait()
        self.play(
            Write(remark[2]),
            ReplacementTransform(formula[3].copy(), remark[3]),
            Write(remark[4]),
        )
        self.wait()
        self.play(
            Write(remark[5]),
            ReplacementTransform(formula[-1].copy(), remark[6]),
            Write(remark[7]),
        )
        self.add(remark)
        self.wait()

        # So the threshold should be a decreasing function of q
        func_text = Tex('\\text{函数}f(q)', color=colors[1])
        func_text.next_to(formula, RIGHT, buff=0.06)
        self.play(
            FadeOut(remark, run_time=1.5),
            Write(func_text, run_time=1),
        )
        formula.add(func_text)
        self.wait()
        loose_bound, squeeze_bound, right_bound = choices = VGroup(
            Tex('{1 \\over q}'),
            Tex('\\left( {2 \\over 3} \\right)^q'),
            Tex('{1 \\over 2q^2}'),
        )
        choices.set_color(colors[1])
        loose_bound.next_to(formula[-2].get_left(), RIGHT, buff=0)
        scroll_shift = 0.8 * UP
        self.play(
            FadeIn(loose_bound, shift=scroll_shift),
            FadeOut(formula[-2:], shift=scroll_shift),
        )
        self.wait()
        title_group = VGroup(title, formula[:-2], loose_bound)
        title_group.generate_target()
        title_group.target.to_edge(UP)
        VGroup(title_group.target[1:]).shift(0.3 * UP)
        self.play(MoveToTarget(title_group))
        self.wait()

        # Examples with sqrt(2)/2 and pi/6
        sqrt2_group = VGroup(
            Tex('{\\sqrt{2} \\over 2}', '\\text{的}', '\\text{“良好”近似}', '\\text{序列：}'),
        )
        pi_group = VGroup(
            Tex('{\\pi \\over 6}', '\\text{的}', '\\text{“良好”近似}', '\\text{序列：}'),
        )
        def loose_func(q): return 1 / q
        loose_config = {'q_coprime': False, 'q_unique': False, 'width_thres': 9}
        sqrt2_loose_fracs = self.get_approx_mobs(np.sqrt(2) / 2., loose_func, **loose_config)
        sqrt2_group.add(sqrt2_loose_fracs)
        pi_loose_fracs = self.get_approx_mobs(PI / 6, loose_func, **loose_config)
        pi_group.add(pi_loose_fracs)
        examples = VGroup(sqrt2_group, pi_group)
        for group in examples:
            group[1].set_color(colors[0])
            group[0][2].set_color(colors[1])
            group[0][0].set_color(colors[2])
            group.arrange(RIGHT)
        examples.arrange(DOWN, index_of_submobject_to_align=0, aligned_edge=RIGHT, buff=1)
        examples.scale(0.8)
        examples.to_edge(DOWN, buff=1.2).to_edge(LEFT, buff=0.5)
        self.play(
            AnimationGroup(*[
                AnimationGroup(*[
                    Write(mob)
                    for mob in group[0]
                ], lag_ratio=0.3)
                for group in examples
            ])
        )
        self.wait()

        self.play(
            AnimationGroup(*[
                AnimationGroup(*[
                    FadeIn(mob, shift=0.5 * RIGHT)
                    for mob in group[1]
                ], lag_ratio=0.3)
                for group in examples
            ])
        )
        self.wait()

        not_coprime_rect = FullScreenFadeRectangle(fill_opacity=0.9)
        not_coprime_rect.set_height(FRAME_HEIGHT / 2).to_edge(UP, buff=0)
        not_coprime_remark = TexText(
            '你可能注意到序列中不全是最简分数，这里是为了配合鸽巢原理做出的简化。即便把可约分数剔除，两个序列依然无限长，不影响结论。\\\\',
            '剔除可约分数的理由也很简单：既然数一样大，提升分母$q$反而收紧了阈值，无异于自找麻烦。所以在后面的例子中，我们默认近似分数都是最简的。',
            alignment='',
            color=BLUE_A,
        )
        not_coprime_remark.set_width(FRAME_WIDTH - 1)
        not_coprime_remark.move_to(not_coprime_rect)
        not_coprime_group = VGroup(not_coprime_rect, not_coprime_remark)
        self.play(FadeIn(not_coprime_group))
        self.wait()
        self.play(
            AnimationGroup(*[
                AnimationGroup(*[
                    Indicate(mob[-1], color=BLUE)
                    for mob in group[1][:-1]
                ], lag_ratio=0.2)
                for group in examples
            ])
        )
        self.wait()
        self.wait()
        self.play(FadeOut(not_coprime_group))
        self.wait()

        # If the bound is too tight, there won't be enough "good" approximations
        for bound in choices[1:]:
            bound.next_to(loose_bound.get_left(), RIGHT, buff=0)

        def squeeze_func(q): return 1 / (1.5**q)
        squeeze_config = {'q_coprime': True, 'add_cdots': False, 'q_unique': False}
        sqrt2_squeeze_fracs = self.get_approx_mobs(np.sqrt(2) / 2., squeeze_func, **squeeze_config)
        pi_squeeze_fracs = self.get_approx_mobs(PI / 6, squeeze_func, **squeeze_config)
        loose_fracs = (sqrt2_loose_fracs, pi_loose_fracs)
        squeeze_fracs = (sqrt2_squeeze_fracs, pi_squeeze_fracs)
        for fracs, source in zip(squeeze_fracs, loose_fracs):
            fracs.set_color(colors[0])
            fracs.scale(0.8)
            fracs.next_to(source.get_left(), RIGHT, buff=0)
        self.play(
            AnimationGroup(*[
                FadeOut(fracs, shift=RIGHT, rate_func=squish_rate_func(smooth, 0, 0.5))
                for fracs in loose_fracs
            ]),
            AnimationGroup(*[
                AnimationGroup(*[
                    FadeIn(frac, shift=0.5 * RIGHT)
                    for frac in fracs
                ], lag_ratio=0.3)
                for fracs in squeeze_fracs
            ]),
            FadeIn(squeeze_bound, shift=scroll_shift, rate_func=squish_rate_func(smooth, 0, 0.5)),
            FadeOut(loose_bound, shift=scroll_shift, rate_func=squish_rate_func(smooth, 0, 0.5)),
            run_time=2,
        )
        self.wait()

        not_enough_text = TexText('长度有限!', color=RED)
        not_enough_text.scale(1.2)
        not_enough_text.next_to(VGroup(*squeeze_fracs), RIGHT, buff=1.5)
        self.play(Write(not_enough_text))
        self.wait()

        # Two objectives we'd like to achieve
        infinite_text = TexText('序列无限长', color=BLUE)
        threshold_text = TexText('阈值尽量紧', color=BLUE)
        infinite_text.next_to(VGroup(*squeeze_fracs), RIGHT, buff=1.5)
        threshold_text.move_to(infinite_text).next_to(squeeze_bound, RIGHT, coor_mask=Y_COOR_MASK)
        self.play(FadeTransform(not_enough_text, infinite_text))
        self.wait()
        self.play(Write(threshold_text))
        self.wait()

        # We will show that 1/(2q^2) is a suitable threshold
        def right_func(q): return 1 / (2 * q**2)
        right_config = {'q_limit': 500000, 'q_coprime': True, 'add_cdots': True, 'q_unique': True, 'width_thres': 9}
        sqrt2_right_fracs = self.get_approx_mobs(np.sqrt(2) / 2., right_func, **right_config)
        pi_right_fracs = self.get_approx_mobs(PI / 6, right_func, **right_config)
        squeeze_fracs = (sqrt2_squeeze_fracs, pi_squeeze_fracs)
        right_fracs = (sqrt2_right_fracs, pi_right_fracs)
        for fracs, source in zip(right_fracs, loose_fracs):
            fracs.set_color(colors[0])
            fracs.scale(0.8)
            fracs.next_to(source.get_left(), RIGHT, buff=0)
        self.play(
            FadeOut(VGroup(infinite_text, threshold_text), rate_func=squish_rate_func(smooth, 0, 0.5)),
            AnimationGroup(*[
                FadeOut(fracs, shift=RIGHT, rate_func=squish_rate_func(smooth, 0, 0.5))
                for fracs in squeeze_fracs
            ]),
            AnimationGroup(*[
                AnimationGroup(*[
                    FadeIn(frac, shift=0.5 * RIGHT)
                    for frac in fracs
                ], lag_ratio=0.3)
                for fracs in right_fracs
            ]),
            FadeIn(right_bound, shift=scroll_shift, rate_func=squish_rate_func(smooth, 0, 0.5)),
            FadeOut(squeeze_bound, shift=scroll_shift, rate_func=squish_rate_func(smooth, 0, 0.5)),
            run_time=3,
        )
        self.wait()

        # Finally, the statement
        statement = Tex(
            '\\text{对于任一}', '\\text{无理数} \\alpha ',
            '\\text{, 总是存在无穷多个满足下式的}', '\\text{最简分数} {p \\over q}',
        )
        statement[1].set_color(colors[2])
        statement[3].set_color(colors[0])
        statement.to_edge(UP)
        self.play(
            FadeOut(title, run_time=3),
            Write(statement, run_time=10),
        )
        self.wait()

        hint = TexText('（福特圆...?）', color=YELLOW)
        hint.scale(0.6).next_to(right_bound, RIGHT, buff=0.5)
        self.play(Write(hint))
        self.wait()

    def get_approx_sequence(self, target, bound_func, q_limit=100, q_unique=True, q_coprime=False):
        result = []
        for q in range(1, q_limit + 1):
            found_p = False
            error = bound_func(q)
            p_upper, p_lower = q * (target + error), q * (target - error)
            for p in range(int(p_lower) - 1, int(p_upper) + 2):
                if all([
                    not (found_p and q_unique),
                    not q_coprime or is_coprime(p, q),
                    abs(p / q - target) < error,
                ]):
                    pair = (p, q)
                    result.append(pair)
                    found_p = True
        return result

    def get_approx_mobs(self, target, bound_func, q_limit=100, q_unique=True, q_coprime=False, add_cdots=True, width_thres=6, buff=0.5):
        sequence = self.get_approx_sequence(
            target, bound_func,
            q_limit=q_limit, q_unique=q_unique, q_coprime=q_coprime
        )
        result = VGroup()
        for p, q in sequence:
            mob = Tex(f'{{{p}', ' \\over ', f'{q}}}')
            if result.get_width() > width_thres:
                break
            else:
                result.add(mob)
                result.arrange(RIGHT, buff=buff)
        if add_cdots:
            result.add(Tex('\\cdots'))
            result.arrange(RIGHT, buff=buff)
        return result


class FordCirclesAndGoodApproximationsPartAP2(Scene):
    def construct(self):
        # Setup the axis
        ff = FordFractal(
            max_denom=20,
            max_zoom_level=54,
            zoom_places=[],
            axis_config={'y_offset': -0.5}
        )
        axis = ff.get_axis().copy()
        self.play(Write(axis))
        self.wait()

        # We choose a irrational number alpha between 0 and 1
        number = 0.5 + 0.095 * random.random()
        print(number)
        point = axis.number_to_point(number)
        alpha_dot = Dot(point, color=MAROON_B)
        alpha_dot.generate_target()
        alpha_dot.scale(5).fade(1)
        alpha_text = Tex('\\alpha', color=MAROON_B)
        alpha_text.scale(1.2)
        alpha_text.next_to(alpha_dot.target, DOWN)
        self.play(MoveToTarget(alpha_dot))
        self.wait()
        self.play(GrowFromPoint(alpha_text, alpha_dot.get_center()))
        self.wait()

        # Condition reminder
        colors = [TEAL, MAROON_B, YELLOW]
        condition = Tex('\\text{良好近似：}', '\\bigg| ', '{p \\over q}', ' - ', '\\alpha', ' \\bigg|', ' < {1 \\over 2q^2}')
        for ind, color in zip([2, 4, 6], colors):
            condition[ind].set_color(color)
        condition.to_edge(UP)
        self.play(Write(condition))
        self.wait()

        # Draw out the circle and move around a bit
        pos_tracker = ValueTracker(0.42)
        rad_tracker = ValueTracker(0.001)

        def position_updater(mob):
            mob.move_to(axis.number_to_point(pos_tracker.get_value()), coor_mask=X_COOR_MASK)

        def radius_updater(mob):
            unit_size = axis.get_unit_size()
            mob.set_width(2 * rad_tracker.get_value() * unit_size)
        rational_dot = alpha_dot.copy()
        rational_dot.set_color(colors[0])
        rational_dot.add_updater(position_updater)
        rational_circle = Circle(color=colors[2], fill_opacity=0.1)
        rational_circle.add_updater(lambda mob: mob.move_to(rational_dot))
        rational_circle.add_updater(radius_updater)
        rational_dot.generate_target()
        rational_dot.scale(5).fade(1)
        rational_text = Tex('{p \\over q}', color=colors[0])
        rational_text.next_to(rational_dot.target, DOWN)
        self.play(
            MoveToTarget(rational_dot),
            Write(rational_text),
        )
        rational_text.add_updater(lambda mob: mob.next_to(rational_dot, DOWN, coor_mask=X_COOR_MASK))
        self.add(rational_circle, rational_dot)
        self.wait()
        self.play(rad_tracker.animate.set_value(0.3))
        self.wait()

        def good_check():
            lower, upper = rational_circle.get_left()[0], rational_circle.get_right()[0]
            target = alpha_dot.get_center()[0]
            return lower < target < upper

        def is_good_updater(mob):
            mob.next_to(rational_circle, DOWN)
            mob.to_edge(DOWN)
            mob.set_fill(opacity=1 if good_check() else 0)

        def is_bad_updater(mob):
            mob.next_to(rational_circle, DOWN)
            mob.to_edge(DOWN)
            mob.set_fill(opacity=0 if good_check() else 1)

        is_good_text = TexText('是良好近似', color=YELLOW)
        is_good_text.add_updater(is_good_updater)
        is_bad_text = TexText('不是良好近似', color=GREY)
        is_bad_text.add_updater(is_bad_updater)

        self.play(Write(is_good_text))
        self.add(is_bad_text)
        self.wait()

        for pos, rad in ([0.3289, 0.2], [0.2477, 0.12]):
            self.play(
                pos_tracker.animate.set_value(pos),
                rad_tracker.animate.set_value(rad),
            )
            self.wait()

        # We now switch perspective, from checking if the circle covers the dot
        # to checking if the line x=alpha intersects with the circle
        alpha_line = Line(10 * UP, ORIGIN, color=colors[1])
        alpha_line.next_to(point, UP, buff=0)
        self.play(
            condition.animate.to_corner(UL),
            GrowFromPoint(alpha_line, point),
            FadeOut(alpha_text),
            Animation(rational_circle),
            run_time=2,
        )
        self.wait()
        for pos, rad in ([0.6750, 0.20], [0.87, 0.1], [1 / 2, 1 / 8]):
            self.play(
                pos_tracker.animate.set_value(pos),
                rad_tracker.animate.set_value(rad),
            )
            self.wait()

        # Now we can move the circle up freely without influencing the result
        def line_updater(line):
            start = rational_circle.get_center()
            end = axis.number_to_point(pos_tracker.get_value())
            line.put_start_and_end_on(start, end)

        helper_line = Line(color=colors[0], stroke_width=1)
        helper_line.add_updater(line_updater)
        self.add(helper_line)
        unit_size = axis.get_unit_size()
        self.play(rational_dot.animate.shift(3 / 8 * unit_size * UP))
        self.wait()
        self.play(rational_dot.animate.shift(2 / 8 * unit_size * DOWN))
        self.wait()

        # Switch the example with 1/2 circle
        circle_1_2 = ff.get_circle(1, 2)
        for mob in (ff, is_good_text, is_bad_text):
            mob.suspend_updating()
        fadeout_group = VGroup(
            rational_text, rational_dot, rational_circle,
            helper_line, is_good_text, is_bad_text,
        )
        self.play(
            FadeOut(fadeout_group),
            FadeIn(circle_1_2),
        )
        self.wait()

        # Now we complete the Ford circles
        other_keys = list(ff.circle_dict.keys())
        other_keys.remove((1, 2))
        random.shuffle(other_keys)
        other_circles = VGroup(*[
            ff.get_circle(*pair)
            for pair in other_keys
        ])
        title_rect = FullScreenFadeRectangle(fill_opacity=0.9)
        title_rect.set_height(2, stretch=True)
        title_rect.to_edge(UP, buff=0)
        self.play(
            AnimationGroup(
                AnimationGroup(*[
                    GrowFromPoint(circle, circle.get_bottom())
                    for circle in other_circles
                ], lag_ratio=0.005),
                FadeIn(title_rect),
                Animation(condition),
            ),
            run_time=5,
        )
        self.remove(axis)
        self.add(ff, title_rect, condition)
        self.wait()
        ff.suspend_updating()
        ff.generate_target()
        ff.target.shift(2 * DOWN)
        ff.target.axis[1:].fade(1)
        self.play(
            AnimationGroup(
                VGroup(alpha_line, alpha_dot).animate.shift(2 * DOWN),
                MoveToTarget(ff),
                Animation(title_rect),
                Animation(condition),
            ),
        )
        ff.resume_updating()
        self.wait()

        equiv_statement1 = Tex(
            '\\text{直线}', '\\text{穿过}', '{p \\over q}\\text{对应的圆}',
            ' \\Leftrightarrow ', '{p \\over q}', '\\text{是良好近似}',
        )
        equiv_statement2 = Tex(
            '\\text{直线}', '\\text{穿过}', '\\text{无穷多个圆}',
            ' \\Leftrightarrow ', '\\text{有无穷多个良好近似}',
        )
        for statement in (equiv_statement1, equiv_statement2):
            statement.move_to(title_rect)
        for mob, color in zip(equiv_statement1[::2], [colors[1], colors[0], colors[0]]):
            mob.set_color(color)
        equiv_statement2.set_color(colors[2])
        self.play(
            FadeOut(condition, shift=UP),
            FadeIn(equiv_statement1, shift=DOWN),
        )
        self.wait()
        self.play(
            FadeOut(equiv_statement1, shift=UP),
            FadeIn(equiv_statement2, shift=DOWN),
        )
        self.wait()
        self.play(
            FadeOut(title_rect),
            FadeOut(equiv_statement2),
        )
        self.wait()

        # What will happen if we draw the line from the top down?
        self.play(
            Uncreate(alpha_line),
            FadeOut(alpha_dot, shift=DOWN),
        )
        self.wait()


class FordCirclesAndGoodApproximationsPartBP2(Scene):
    def construct(self):
        # Match the last scene
        number = 0.5 + 0.095 * random.random()
        ff = FordFractal(
            max_denom=30,
            max_zoom_level=20000,
            # max_zoom_level=50,
            zoom_places=[number],
            axis_config={'y_offset': -2.5}
        )
        ff.get_axis()[1:].fade(1)
        point = ff.get_axis().number_to_point(number)
        self.add(ff)

        # Setup the line
        line_factor = ValueTracker(0.01)
        line_start, line_end = VectorizedPoint(point + 10 * UP), VectorizedPoint(point)
        alpha_line = Line(color=MAROON_B)

        def alpha_line_updater(line):
            start = line_start.get_center()
            end = line_end.get_center()
            mid = start + (end - start) * line_factor.get_value()
            line.put_start_and_end_on(start, mid)
        alpha_line.add_updater(alpha_line_updater)
        self.add(alpha_line)

        # Make sure the line is ready for the zooming animations
        def get_line_zoom_animation(number, zoom_factor, additional_mobs=()):
            mobs = [line_start, line_end]
            for mob in additional_mobs:
                mobs.append(mob)
            return AnimationGroup(*[
                ApplyMethod(
                    mob.scale, zoom_factor,
                    {'about_point': ff.get_axis().number_to_point(number)}
                )
                for mob in mobs
            ])

        # Approximate the `line_factor`
        def get_line_factors(numer, denom, epsilon=1e-3):
            circle = ff.get_circle(numer, denom)
            assert(circle.get_left()[0] < alpha_line.get_center()[0] < circle.get_right()[0])
            end = line_end.get_center()[1]
            start = line_start.get_center()[1]
            height = abs(start - end)
            rectangle = Rectangle(height=height, width=epsilon)
            rectangle.next_to(point, UP, buff=0)
            intersection = Intersection(rectangle, circle)
            top = intersection.get_top()[1]
            bottom = intersection.get_bottom()[1]
            top_factor = abs(top - start) / height
            bottom_factor = abs(bottom - start) / height
            return (top_factor, bottom_factor)

        # At start, the line encounters two 'guards': 0/1 and 1/1
        top_factor, bottom_factor = get_line_factors(1, 1)
        top_factor -= 0.05
        self.play(line_factor.animate.set_value(top_factor))
        self.wait()
        circles = VGroup(*[
            ff.get_circle(*pair)
            for pair in ([0, 1], [1, 1])
        ])
        for circle in circles:
            circle.save_state()
            circle.generate_target()
            circle.target.set_fill(opacity=0.2)
            circle.target.set_color(YELLOW)
        self.play(
            AnimationGroup(*[MoveToTarget(circle) for circle in circles]),
            run_time=2,
        )
        self.wait()

        # There're few points that the line never crosses
        cross_size = 0.2
        crosses = VGroup(*[
            Cross(Circle(radius=cross_size).move_to(p)).set_stroke(width=15)
            for p in (
                ff.get_tangent_point(0, 1, 1, 1),
                *[circle.get_bottom() for circle in circles],
            )
        ])
        self.play(ShowCreation(crosses[0]))
        self.wait()
        self.play(
            AnimationGroup(*[
                ShowCreation(cross)
                for cross in crosses[1:]
            ]),
        )
        self.wait()

        bottom_factor -= 0.01
        self.play(
            line_factor.animate.set_value(bottom_factor),
            FadeOut(crosses),
        )
        self.wait()

        # And it will enter the gap in between
        gap_config = {
            'gap_style': {
                'color': GREY,
                'fill_opacity': 0.5,
                'stroke_width': 1,
            }
        }
        gap = FordFractalGap(ff, (0, 1), (1, 1), **gap_config)
        self.play(
            *[circle.animate.restore() for circle in circles],
            FadeIn(gap),
        )
        self.wait()

        # Zoom in for a better view
        first_zoom_factor = 1.8
        self.play(
            ff.zoom_in_on(number, first_zoom_factor, animate=True),
            get_line_zoom_animation(number, first_zoom_factor, additional_mobs=(gap, )),
            run_time=3,
        )
        self.wait()

        # Helper methods to add circles in the animations in advance
        def add_circles_in_advance(mid, left, right, n=30):
            chain = get_chain(mid, left, right, n)
            for element in chain:
                ff.add_circle(element.p, element.q)

        # Helper methods for finding a chain
        def get_chain(mid, left, right, n=30):
            result = [mid]
            for k in range(n):
                curr_left, curr_right = result[0], result[-1]
                new_left = curr_left.farey_add(left)
                new_right = curr_right.farey_add(right)
                result.insert(0, new_left)
                result.append(new_right)
            result.insert(0, left)
            result.append(right)
            return result

        # Helper methods for finding the circles in a chain
        def get_chain_circles(mid, left, right, n=30):
            add_circles_in_advance(mid, left, right, n)
            return VGroup(*[
                ff.get_circle(element.p, element.q)
                for element in get_chain(mid, left, right, n)
            ])

        # Helper methods for finding the gaps between circles in a chain
        def get_gaps(mid, left, right, n=30):
            add_circles_in_advance(mid, left, right, n)
            chain = get_chain(mid, left, right, n)
            return VGroup(*[
                FordFractalGap(
                    ff,
                    (chain[k].p, chain[k].q),
                    (chain[k + 1].p, chain[k + 1].q),
                    **gap_config
                )
                for k in range(len(chain) - 1)
            ])

        # First chain demonstration
        max_k = 50
        left, right = FareyRational(0, 1), FareyRational(1, 1)
        mid = left.farey_add(right)
        add_circles_in_advance(mid, left, right, max_k)
        curr_gaps = gap
        curr_additions = VGroup()
        ff.suspend_updating()
        ff.save_state()
        for k in list(range(7)) + [max_k]:
            chain = get_chain(mid, left, right, k)
            new_gaps = get_gaps(mid, left, right, k)
            new_additions = VGroup()
            if k == 0:
                new_additions.add(
                    Tex(
                        f'{{{left.p} \\over {left.q}}}', ' \\oplus ', f'{{{right.p} \\over {right.q}}}', ' = ',
                        f'{{{mid.p} \\over {mid.q}}}',
                    ),
                )
                target = ff.get_circle(mid.p, mid.q).get_center()
                source = new_additions[0][1].get_center()
                new_additions.shift(target - source)
            elif k == max_k:
                pass
            else:
                text_left = Tex(
                    f'{{{left.p} \\over {left.q}}}', ' \\oplus ', f'{{{chain[2].p} \\over {chain[2].q}}}', ' = ',
                    f'{{{chain[1].p} \\over {chain[1].q}}}',
                )
                target_left = ff.get_circle(chain[1].p, chain[1].q).get_center()
                source_left = text_left[1].get_center()
                text_left.shift(target_left - source_left)
                text_right = Tex(
                    f'{{{chain[-3].p} \\over {chain[-3].q}}}', ' \\oplus ', f'{{{right.p} \\over {right.q}}}', ' = ',
                    f'{{{chain[-2].p} \\over {chain[-2].q}}}',
                )
                target_right = ff.get_circle(chain[-2].p, chain[-2].q).get_center()
                source_right = text_right[1].get_center()
                text_right.shift(target_right - source_right)
                new_additions.add(text_left, text_right)
            for addition in new_additions:
                addition.scale(0.8, about_point=addition[1].get_center())
                addition.to_edge(DOWN, buff=0.25)
                addition[-1].set_color(YELLOW)
            inner_chain_circles = get_chain_circles(mid, left, right, k)[1:-1]
            for circle in inner_chain_circles:
                circle.generate_target()
                circle.target.set_color(YELLOW)
                circle.target.get_circle().set_fill(opacity=0.2)
            self.play(
                *[MoveToTarget(circle) for circle in inner_chain_circles],
                FadeOut(curr_additions),
                FadeIn(new_additions),
                FadeOut(curr_gaps),
                FadeIn(new_gaps),
            )
            curr_gaps = new_gaps
            curr_additions = new_additions
            self.wait()

        # The circles filled the (0, 1) interval, so the line has to cross at least one circle
        first_circle_cross_text = TexText('直线', '会穿过', '一个圆')
        first_circle_cross_text.to_edge(DOWN)
        for mob, color in zip(first_circle_cross_text, [MAROON_B, WHITE, YELLOW]):
            mob.set_color(color)
        top_factor, bottom_factor = get_line_factors(1, 2)
        bottom_factor -= 0.005
        mid_factor = (top_factor + bottom_factor) / 2
        self.play(
            Write(first_circle_cross_text),
            line_factor.animate.set_value(mid_factor)
        )
        self.wait()

        # There're infinitely many gaps generated as well
        # The line is bound to cross one of them, and we need to find it
        def find_crossed_gap(gaps):
            line_x = alpha_line.get_x()
            for gap in gaps:
                gap_left, gap_right = gap.get_left()[0], gap.get_right()[0]
                if gap_left < line_x < gap_right:
                    return gap
            raise Exception('Crossed gap not found.')

        crossed_gap = find_crossed_gap(curr_gaps)
        curr_gaps.remove(crossed_gap)
        self.add(curr_gaps, crossed_gap)
        crossed_gap.save_state()
        self.play(
            ff.animate.restore(),
            curr_gaps.animate.set_color(YELLOW),
            crossed_gap.animate.set_color(YELLOW),
            FadeOut(first_circle_cross_text),
        )
        ff.resume_updating()
        self.wait()

        # Add crosses between gaps like last time
        def get_gap_crosses(mid, left, right, n=30, shuffle=True):
            chain = get_chain(mid, left, right, n)
            if shuffle:
                random.shuffle(chain)
            gap_crosses = VGroup()
            for element in chain:
                circle = ff.get_circle(element.p, element.q)
                h = circle.get_height()
                bottom = circle.get_bottom()
                cross = Cross(Circle(radius=min(0.2, h / 4)).move_to(bottom))
                cross.set_stroke(width=min(10, h * 10))
                gap_crosses.add(cross)
            return gap_crosses

        gap_crosses = get_gap_crosses(mid, left, right, max_k)
        self.play(
            AnimationGroup(*[
                ShowCreation(cross)
                for cross in gap_crosses
            ], lag_ratio=0.005),
            run_time=1,
        )
        self.wait()

        first_gap_cross_text = TexText('直线', '会穿过', '一个曲边三角形')
        first_gap_cross_text.move_to(first_circle_cross_text)
        for mob, color in zip(first_gap_cross_text, [MAROON_B, WHITE, YELLOW]):
            mob.set_color(color)
        self.play(
            FadeOut(gap_crosses),
            FadeOut(curr_gaps),
            Write(first_gap_cross_text),
            line_factor.animate.set_value(bottom_factor),
            run_time=2,
        )
        self.wait()

        #
        second_zoom_factor = 6
        self.play(
            FadeOut(first_gap_cross_text, rate_func=squish_rate_func(smooth, 0, 0.5)),
            ff.zoom_in_on(number, second_zoom_factor, animate=True),
            crossed_gap.animate.scale(second_zoom_factor, about_point=ff.get_axis().number_to_point((number))).set_color(GREY),
            get_line_zoom_animation(number, second_zoom_factor),
            run_time=3,
        )
        self.wait()

        # Closed loop demonstration
        curr_gaps = crossed_gap
        left, right = FareyRational(1, 2), FareyRational(2, 3)
        mid = left.farey_add(right)
        add_circles_in_advance(mid, left, right, max_k)
        new_gaps = get_gaps(mid, left, right, max_k)
        inner_chain_circles = get_chain_circles(mid, left, right, max_k)[1:-1]
        for circle in inner_chain_circles:
            circle.save_state()
            circle.generate_target()
            circle.target.set_color(YELLOW)
            circle.target.get_circle().set_fill(opacity=0.2)
        process_texts = VGroup(*[
            TexText(text)
            for text in (
                '构建屏障，拦截直线', '直线穿过一个新的圆',
                '直线不能从圆的底部穿出', '直线穿入下一个曲边三角形',
                '构建屏障，拦截直线',
            )
        ])
        for text in process_texts:
            text.set_color(YELLOW)
            text.to_edge(DOWN)
        last_remark = RemarkText(
            text[-1], '（似曾相识...?）',
            direction=RIGHT, aligned_edge=DOWN,
            scale_factor=0.5, color=GREY,
        )
        text[-1].add(last_remark)
        self.play(
            Write(process_texts[0]),
            *[MoveToTarget(circle) for circle in inner_chain_circles],
            FadeOut(curr_gaps),
            FadeIn(new_gaps),
        )
        curr_gaps = new_gaps
        self.wait()

        top_factor, bottom_factor = get_line_factors(4, 7)
        bottom_factor -= 0.0005
        mid_factor = (top_factor + bottom_factor) / 2
        self.play(
            FadeTransform(process_texts[0], process_texts[1]),
            line_factor.animate.set_value(mid_factor),
        )
        self.wait()

        gap_crosses = get_gap_crosses(mid, left, right, max_k)
        self.play(
            FadeTransform(process_texts[1], process_texts[2]),
            AnimationGroup(*[
                ShowCreation(cross)
                for cross in gap_crosses
            ], lag_ratio=0.005),
            run_time=1,
        )
        self.wait()

        crossed_gap = find_crossed_gap(curr_gaps)
        curr_gaps.remove(crossed_gap)
        self.add(curr_gaps, crossed_gap)
        self.play(
            FadeOut(gap_crosses),
            FadeTransform(process_texts[2], process_texts[3]),
            *[circle.animate.restore() for circle in inner_chain_circles],
            line_factor.animate.set_value(bottom_factor),
            FadeOut(curr_gaps),
        )
        self.wait()

        third_zoom_factor = 5
        self.play(
            FadeTransform(process_texts[3], process_texts[4]),
            ff.zoom_in_on(number, third_zoom_factor, animate=True),
            get_line_zoom_animation(number, third_zoom_factor, additional_mobs=(crossed_gap, )),
            run_time=3,
        )
        self.wait()

        # Final clean up and conclusion
        self.play(FadeOut(process_texts[4]), FadeOut(crossed_gap))
        self.wait()

        conclusion = TexText(
            '直线', '穿过', '无穷多个圆', '$\\Rightarrow$',
            '无理数$\\alpha$', '有', '无限长的良好近似序列',
        )
        for ind, color in enumerate([MAROON_B, WHITE, YELLOW]):
            conclusion[ind::4].set_color(color)
        conclusion.to_edge(DOWN)
        alpha_line.suspend_updating()
        last_factor = 256
        self.play(
            Write(conclusion[:3], rate_func=squish_rate_func(smooth, 0, 0.2)),
            ff.zoom_in_on(number, last_factor, animate=True),
            run_time=10,
        )
        self.wait()
        self.play(
            GrowFromCenter(conclusion[3]),
            Write(conclusion[4:]),
        )
        self.wait()

        # For irrationals that are not in (0,1), simply split it into two parts
        fade_rect = FullScreenFadeRectangle(fill_opacity=1)
        fade_rect.next_to(conclusion, UP)
        split_text = TexText(
            '$\\alpha$ ', '$ = \\alpha$的整数部分 $+$ ', '$\\alpha$的小数部分',
        )
        split_done_text = TexText(
            '$\\alpha$的良好近似', '$ = \\alpha$的整数部分 $+$ ', '$\\alpha$小数部分的良好近似',
        )
        text_group = VGroup(split_text, split_done_text)
        for text, color in zip(text_group, [MAROON_B, YELLOW]):
            VGroup(text[0], text[-1]).set_color(color)
            text.scale(0.9)
        text_group.arrange(DOWN, buff=0, index_of_submobject_to_align=1)
        split_done_text.move_to(split_text, coor_mask=Y_COOR_MASK)
        text_group.center()

        self.play(FadeIn(fade_rect), Write(split_text[0]))
        self.remove(ff, alpha_line, fade_rect)
        self.wait()
        self.play(FadeIn(split_text[1:]))
        self.wait()
        self.play(
            ShowCreationThenDestructionAround(
                split_text[-1],
                surrounding_rectangle_config={'color': MAROON_B},
            )
        )
        self.wait()
        self.play(
            FadeOut(split_text[-1], shift=UP),
            FadeIn(split_done_text[-1], shift=UP),
        )
        self.wait()
        self.play(
            FadeOut(split_text[0], shift=UP),
            FadeIn(split_done_text[0], shift=UP),
        )
        self.remove(split_text)
        self.add(split_done_text)
        self.wait()

        # The proved statement
        colors = [TEAL, YELLOW, MAROON_B]
        theorem = Tex(
            '\\text{对于任一}', '\\text{无理数} \\alpha ',
            '\\text{, 总是存在无穷多个满足下式的}', '\\text{最简分数} {p \\over q}',
        )
        formula = Tex('\\bigg| ', '{p \\over q}', ' - ', '\\alpha', ' \\bigg|', ' < ', '{1 \\over 2q^2}')
        theorem.to_edge(UP, buff=0.8)
        formula.next_to(theorem, DOWN)
        VGroup(theorem[3], formula[1]).set_color(colors[0])
        VGroup(formula[-1]).set_color(colors[1])
        VGroup(theorem[1], formula[3]).set_color(colors[2])
        group = VGroup(theorem, formula)
        self.play(
            FadeOut(conclusion, rate_func=squish_rate_func(smooth, 0, 0.5)),
            FadeOut(split_done_text, rate_func=squish_rate_func(smooth, 0, 0.5)),
            Write(group),
            run_time=5,
        )
        self.wait()

        # Remark on constructing good approximations
        tick = Tex('\\text{\\ding{51}}', color=GREEN)
        cross = Tex('\\text{\\ding{55}}', color=RED)
        qmark = Tex('?', color=BLUE)
        infinite_remark = TexText('良好近似序列是无限长的', color=GREEN)
        construct_remark = TexText('良好近似序列的构造方法', color=RED)
        symbol_group = VGroup(tick, cross)
        remark_group = VGroup(infinite_remark, construct_remark)
        remark_group.arrange(DOWN, buff=0.8)
        remark_group.move_to(DOWN)
        for symbol, target in zip(symbol_group, remark_group):
            symbol.next_to(target, LEFT, buff=0.4)
        qmark.move_to(cross)
        self.play(Write(infinite_remark), GrowFromCenter(tick))
        self.wait()
        self.play(Write(construct_remark), GrowFromCenter(cross))
        self.wait()
        exercise_alpha = TexText(f'（例如$\\alpha={number}...$）', color=BLUE)
        exercise_alpha.scale(0.8)
        exercise_alpha.next_to(construct_remark, DOWN)
        self.play(
            construct_remark.animate.set_color(BLUE),
            FadeTransform(cross, qmark),
            FadeIn(exercise_alpha),
        )
        self.wait()
        self.play(FadeOut(VGroup(tick, qmark, infinite_remark, construct_remark, exercise_alpha)))
        self.wait()


class Sqrt5IsTheLimit(Scene):
    def construct(self):
        # Match the last scene
        colors = [TEAL, YELLOW, MAROON_B]
        statement = Tex(
            '\\text{对于任一}', '\\text{无理数} \\alpha ',
            '\\text{, 总是存在无穷多个满足下式的}', '\\text{最简分数} {p \\over q}',
        )
        curr_thres = Tex('\\bigg| ', '{p \\over q}', ' - ', '\\alpha', ' \\bigg|', ' < ', '{1 \\over 2q^2}')
        limit_thres = Tex('\\bigg| ', '{p \\over q}', ' - ', '\\alpha', ' \\bigg|', ' < ', '{1 \\over \\sqrt{5} q^2}')
        statement.to_edge(UP, buff=0.8)
        curr_thres.next_to(statement, DOWN)
        limit_thres.move_to(curr_thres)
        VGroup(statement[3], curr_thres[1], limit_thres[1]).set_color(colors[0])
        VGroup(curr_thres[-1], limit_thres[-1]).set_color(colors[1])
        VGroup(statement[1], curr_thres[3], limit_thres[3]).set_color(colors[2])
        self.add(statement, curr_thres)

        # Can we push the limit further? And the answer is yes
        improv_text = TexText('（提升空间?）', color=YELLOW)
        best_text = TexText('（最优结果）', color=YELLOW)
        for text in (improv_text, best_text):
            text.scale(0.8)
            text.next_to(curr_thres[-1], RIGHT, buff=0.8)
        self.play(
            Write(improv_text),
            ShowCreationThenDestructionAround(curr_thres[-1]),
        )
        self.wait()
        self.play(
            *[ReplacementTransform(curr_thres[ind], limit_thres[ind]) for ind in range(6)],
            *[ReplacementTransform(curr_thres[-1][ind], limit_thres[-1][ind]) for ind in (0, 1, -1, -2)],
            FadeTransform(curr_thres[-1][2:-2], limit_thres[-1][2:-2]),
            FadeTransform(improv_text, best_text),
        )
        self.remove(curr_thres)
        self.add(limit_thres)
        self.wait()

        # This is known as Hurwitz theorem (in number theory)
        portrait_hurwitz = ImageMobject('Adolf_Hurwitz.jpg', height=4)
        name_hurwitz = RemarkText(
            portrait_hurwitz, 'Adolf Hurwitz',
            direction=DOWN, aligned_edge=ORIGIN, color=WHITE,
        )
        hurwitz_group = Group(portrait_hurwitz, name_hurwitz)
        hurwitz_group.to_corner(DL)
        self.play(FadeIn(hurwitz_group))
        self.wait()

        title_chn = TexText('赫尔维茨定理', color=YELLOW)
        title_chn.to_edge(UP)
        title_eng = RemarkText(
            title_chn, '(Hurwitz Theorem)',
            direction=DOWN, aligned_edge=ORIGIN, color=YELLOW,
        )
        title_group = VGroup(title_chn, title_eng)
        theorem_group = VGroup(statement, limit_thres, best_text)
        theorem_group.generate_target()
        theorem_group.target[0].next_to(theorem_group.target[1], UP, buff=0)
        theorem_group.target.scale(0.8).next_to(title_group, DOWN)
        title_group.generate_target()
        title_group.next_to(TOP, UP).fade(1)
        self.play(
            MoveToTarget(title_group),
            MoveToTarget(theorem_group),
        )
        self.wait()

        # It can be proven by Ford circles
        ff = FordFractal(
            axis_config={
                'x_range': [0, 1, 1],
                'include_tick_numbers': False
            },
        )
        ff.set_height(3.5)
        ff.next_to(theorem_group, DOWN, buff=0.5)
        ff.shift(RIGHT + 4 * DOWN)
        self.add(ff)
        self.wait()
        ff.suspend_updating()
        self.play(ff.animate.shift(4 * UP))
        ff.resume_updating()
        self.wait()

        # Flash all gaps
        gap_config = {'gap_style': {'fill_opacity': 1, 'stroke_width': 0}}
        gaps_list = []
        all_keys = ff.circle_dict.keys()
        for key1, key2 in it.combinations(all_keys, 2):
            a, b = key1
            c, d = key2
            if abs(a * d - b * c) == 1:
                key3 = (a + c, b + d)
                if key3 in all_keys:
                    gap = FordFractalGap(ff, key1, key2, key3, **gap_config)
                    gaps_list.append(gap)
        random.shuffle(gaps_list)
        all_gaps = VGroup(*gaps_list)
        self.play(
            AnimationGroup(*[
                FadeIn(gap, scale_factor=1, rate_func=there_and_back_with_pause)
                for gap in all_gaps
            ], lag_ratio=0.02),
            run_time=5,
        )
        self.wait()

        # Show the reference 'Fractions'
        ref_image = ImageMobject('Fractions_refpart.png')
        ref_image.match_height(portrait_hurwitz)
        ref_image.move_to(portrait_hurwitz)
        ref_remark = RemarkText(
            ref_image, 'doi: 10.2307/2302799',
            direction=DOWN, aligned_edge=ORIGIN, color=WHITE,
        )
        ref_group = Group(ref_image, ref_remark)
        self.play(
            FadeOut(hurwitz_group, shift=LEFT),
            FadeIn(ref_group, shift=LEFT),
        )
        self.wait()
        ff.suspend_updating()
        self.play(FadeOut(Group(ref_group, ff)))
        self.wait()

        # Coefficient sqrt(5) and exponent 2 cannot be further improved
        limit_thres_copy = limit_thres.copy()
        limit_thres_copy.generate_target()
        limit_thres_copy.target.shift(1.75 * DOWN)
        limit_thres_copy.target.set_color(GREY)
        for ind in [slice(2, -2), -1]:
            mob = limit_thres_copy.target[-1][ind]
            mob.set_fill(opacity=1)
            mob.set_color(GREEN)
        coeff_exp_text = TexText('不能继续增大', color=GREEN)
        coeff_exp_text.scale(0.64)
        coeff_exp_text.next_to(limit_thres_copy.target, RIGHT, buff=0.5)
        self.play(
            MoveToTarget(limit_thres_copy),
            FadeTransform(best_text.copy(), coeff_exp_text),
        )
        self.wait()
        ambitious_thres = Tex(
            '\\bigg| ', '{p \\over q}', ' - ', '\\alpha', ' \\bigg|', ' < ',
            '{1 \\over (\\sqrt{5} + \\varepsilon) q^2}',
        )
        for ind, color in zip([1, 3, -1], [TEAL, MAROON_B, RED]):
            ambitious_thres[ind].set_color(color)
        ambitious_thres.scale(0.8).next_to(limit_thres_copy.get_right(), LEFT, buff=0)
        break_text = TexText('（定理失效）', color=RED)
        break_text.scale(0.64)
        break_text.next_to(coeff_exp_text.get_left(), RIGHT, buff=0)
        tiny_amount = TexText('增加一丁点...', color=RED)
        tiny_amount.scale(0.5).next_to(ambitious_thres[-1][2:-2], DOWN)
        self.play(
            *[ReplacementTransform(limit_thres_copy[ind], ambitious_thres[ind]) for ind in range(6)],
            *[ReplacementTransform(limit_thres_copy[-1][ind], ambitious_thres[-1][ind]) for ind in (0, 1, -1, -2)],
            FadeTransform(limit_thres_copy[-1][2:-2], ambitious_thres[-1][2:-2]),
            FadeTransform(coeff_exp_text, break_text),
            FadeIn(tiny_amount),
        )
        self.wait()

        # One counter example is the golden ratio
        break_group = VGroup(ambitious_thres, break_text, tiny_amount)
        counter_example = Tex('\\text{反例：}', '\\text{黄金分割比}\\varphi = {\\sqrt{5} + 1 \\over 2}', color=MAROON_B)
        counter_example.scale(0.8)
        counter_example.next_to(break_group, DOWN, buff=0.4)
        break_group.add(counter_example)
        break_rect = SurroundingRectangle(break_group, color=RED, buff=0.3)
        self.play(
            Write(counter_example),
            ShowCreation(break_rect, rate_func=squish_rate_func(smooth, 0.4, 1)),
            run_time=3,
        )
        self.wait()


class SummaryOfTwoVideos(Scene):
    def construct(self):
        placeholder = TexText('占位符')
        placeholder.to_edge(UP)
        pip_frame = PictureInPictureFrame()
        pip_frame.set_height(6)
        pip_frame.next_to(placeholder, DOWN)
        self.play(ShowCreation(pip_frame))
        self.wait()

        # Show all titles
        titles = VGroup(*[
            TexText(text)
            for text in [
                '福特圆与法里和的介绍', '福特圆的三维变体——福特球',
                '有理数的丢番图逼近', '无理数的丢番图逼近',
                '笛卡尔定理', '连分数表示', '分式线性变换', '线性代数与群论',
            ]
        ])
        for title in titles:
            title.next_to(pip_frame, UP, coor_mask=Y_COOR_MASK)
        curr_title = None
        for title in titles:
            if curr_title is None:
                self.play(FadeIn(title))
            else:
                self.play(FadeTransform(curr_title, title))
            self.wait()
            curr_title = title
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait()


class OtherFieldsWorthExploring(Scene):
    def construct(self):
        # 1. Descartes' Theorem
        self.remove(*self.mobjects)
        descartes_example = ImageMobject('DescartesTheoremExamples.png')
        descartes_example.set_height(FRAME_HEIGHT)
        curvature = Tex('\\left( \\text{曲率}k = {1 \\over \\text{半径}R} \\right)', color=GREY)
        curvature.to_corner(UR, buff=1).shift(0.7 * DOWN)
        construction_text = TexText('理解福特圆的构造方式', color=YELLOW)
        construction_text.move_to(3 * RIGHT + 1 * DOWN)
        construction_remark = RemarkText(
            construction_text, '（为什么半径与分母平方的倒数有关）',
            direction=DOWN, aligned_edge=ORIGIN, color=YELLOW,
        )
        self.add(descartes_example, curvature)
        self.wait()
        for mob in (construction_text, construction_remark):
            self.play(FadeIn(mob))
            self.wait()

        # 2. Continued fraction
        self.remove(*self.mobjects)
        ft = FareyTree(n_levels=4)
        ft.to_edge(LEFT)
        frac_4_7 = ft.get_fraction_from_numbers(4, 7)
        frac_4_7.set_color(YELLOW)
        frac_tuples = list(
            FareyRational(p, q, level)
            for level, (p, q) in enumerate([(1, 1), (1, 2), (2, 3), (3, 5), (4, 7)])
        )
        arrows = VGroup()
        arrow_colors = [BLUE, RED, GREEN, GREEN]
        for k in range(len(frac_tuples) - 1):
            color = arrow_colors[k]
            frac_a, frac_b = frac_tuples[k:k + 2]
            line = ft.get_line_from_frac_tuples(frac_a, frac_b).copy()
            arrow = Arrow(line.get_start(), line.get_end(), stroke_width=10, buff=0)
            arrow.set_color(color)
            arrows.add(arrow)
        directions = TexText('左', '右', '左左')
        braces = VGroup(*[Brace(d, DOWN) for d in directions])
        nums = VGroup(*[Tex(str(num)) for num in (1, 1, 2)])
        for brace, num in zip(braces, nums):
            brace.put_at_tip(num)
        direction_group = VGroup(directions, braces, nums)
        direction_group.to_corner(UR, buff=1.5)
        continued_frac = Tex('{4 \\over 7}', ' = \\cfrac{1}{1+\\cfrac{1}{1+\\cfrac{1}{2+1}}}')
        continued_frac[0].set_color(YELLOW)
        continued_frac.to_edge(RIGHT).shift(1.5 * DOWN)
        frac_colors = arrow_colors[:-1]
        color_indices = [3, 7, 11]
        other_indices = list(set(range(14)) - set(color_indices))
        for ind, d, num, color in zip(color_indices, directions, nums, frac_colors):
            d.set_color(color)
            num.set_color(color)
            continued_frac[-1][ind].set_color(color)
        self.add(ft, arrows, direction_group)
        self.wait()
        self.play(
            AnimationGroup(*[
                ReplacementTransform(num.copy()[0][0], continued_frac[-1][ind])
                for num, ind in zip(nums, color_indices)
            ], lag_ratio=0.05, rate_func=squish_rate_func(smooth, 0, 0.5)),
            ReplacementTransform(
                frac_4_7.copy()[0], continued_frac[0],
                rate_func=squish_rate_func(smooth, 0, 0.5),
            ),
            AnimationGroup(*[
                FadeIn(continued_frac[-1][ind])
                for ind in other_indices
            ], lag_ratio=0.05, rate_func=squish_rate_func(smooth, 0.3, 1)),
            run_time=2,
        )
        self.wait()

        # 3. Linear fractional transform
        self.remove(*self.mobjects)
        lf_transform = Tex('f', '(z) = {az+b \\over cz+d}', '\\quad (z \\in \\mathbb{C})')
        self.add(lf_transform)
        self.wait()
        symmetries_lft = Tex(
            '\\Gamma = \\Big\\{', 'f', '\\, \\Big| \\,', 'a,\\,b,\\,c,\\,d \\in \\mathbb{Z},\\,|ad-bc|=1', '\\Big\\}',
            color=YELLOW,
        )
        symmetries_lft.shift(DOWN)
        sym_title = TexText('福特圆的对称性', color=YELLOW)
        sym_title.next_to(symmetries_lft, DOWN)
        self.play(
            lf_transform.animate.shift(UP),
            Write(symmetries_lft[0]),
            Write(symmetries_lft[2:]),
            ReplacementTransform(lf_transform[0].copy()[0], symmetries_lft[1]),
            Write(sym_title),
            run_time=2,
        )
        self.wait()

        # 4. Group theory and linear algebra
        title_buff = 0.5
        sym_title.generate_target()
        sym_title.target.center()
        matrix_transform = Tex('M', '= \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}')
        symmetries_mat = Tex(
            '\\text{SL}(2,\\,\\mathbb{Z}) = \\Big\\{',
            'M', '\\, \\Big| \\,', 'a,\\,b,\\,c,\\,d \\in \\mathbb{Z},\\,', '|ad-bc|=1', '\\Big\\}',
            color=YELLOW,
        )
        matrix_transform.to_edge(DOWN)
        symmetries_mat.next_to(sym_title.target, DOWN, buff=title_buff),
        self.play(
            lf_transform.animate.to_edge(UP),
            symmetries_lft.animate.next_to(sym_title.target, UP, buff=title_buff)
                                  .next_to(symmetries_mat.get_right(), LEFT, buff=0, coor_mask=X_COOR_MASK),
            MoveToTarget(sym_title),
        )
        self.wait()
        self.play(
            FadeTransform(lf_transform.copy(), matrix_transform, path_arc=PI / 3),
            FadeTransform(symmetries_lft.copy(), symmetries_mat, path_arc=PI / 3),
        )
        self.wait()
        self.play(symmetries_mat[-2].animate.set_color(BLUE))
        self.wait()
        self.play(FadeOut(VGroup(*self.mobjects)))
        self.wait()


class ReferencesP2(ReferenceScene):
    def get_references(self):
        mechanical_watch_ref = Reference(
            name='Mechanical Watch（机械钟表）',
            authors='Bartosz Ciechanowski',
            pub='个人博客',
            doi='https://ciechanow.ski/mechanical-watch/',
            info='想了解机械钟表的内部构造吗？可以看看这篇博客。\\\\虽说跟数学的关系不大，但其中的动画十分精良，不分享就太可惜了。\\\\（感谢manim幼儿园-一张一弛推荐）',
        )
        gears_ref1 = Reference(
            name='On the Teeth of Wheels',
            authors='Brian Hayes',
            pub='American Scientist, 2000, 88(4): 296-300.',
            doi='https://www.jstor.org/stable/27858048',
            info='《美国科学家》（不是《科学美国人》）上的科普文，主要是介绍钟表匠Brocot寻找齿轮传动比近似的方法。里面提到的安提基特拉机械（Antikythera mechanism）很有意思，也正好反映了有理数近似在天文学中的应用。',
        )
        gears_ref2 = Reference(
            name='Trees, Teeth, and Time: The mathematics of clock making',
            authors='David Austin',
            pub='American Mathematical Society, Feature Column, Monthly Essays on Mathematical Topics, 2008.',
            doi='https://www.ams.org/publicoutreach/feature-column/fcarc-stern-brocot',
            info='美国数学会的一篇专栏，同样是讲Brocot的近似方法，但是稍作改动后可以获得分母更大的近似值。',
        )
        hurwitz_proof_ref1 = Reference(
            name='Fractions',
            authors='Lester R. Ford',
            pub='The American Mathematical Monthly, 1938, 45(9): 586-601.',
            doi='https://doi.org/10.2307/2302799',
            info='上期也列过这篇，再推荐一次是因为它还包含了Hurwitz定理的一个初等证明，用的是分类讨论+简单的代数运算。'
        )
        hurwitz_proof_ref2 = Reference(
            name='An Easy Proof of Hurwitz\'s Theorem',
            authors='Manuel Benito, J. Javier Escribano',
            pub='The American mathematical monthly, 2002, 109(10): 916-918.',
            doi='https://doi.org/10.1080/00029890.2002.11919929',
            info='如题。Hurwitz定理的另一个初等证明，用的则是法里序列的性质。',
        )
        hurwitz_variant_ref = Reference(
            name='On the closeness of approach of complex rational fractions \\\\ to a complex irrational number',
            authors='Lester R. Ford',
            pub='Transactions of the American mathematical society, 1925, 27(2): 146-154.',
            doi='https://www.ams.org/journals/tran/1925-027-02/S0002-9947-1925-1501304-X',
            info='用两个高斯整数的商去近似复数时，也有类似Hurwitz定理的结果，只是分母中系数的最优结果变成了$\\sqrt{3}$。',
        )
        mandelbrot_ref = Reference(
            name='The Mandelbrot Set, the Farey Tree, and the Fibonacci Sequence',
            authors='Robert L. Devaney',
            pub='The American Mathematical Monthly, 1999, 106(4): 289-302.',
            doi='https://doi.org/10.1080/00029890.1999.12005046',
            info='和福特圆分形一样，数学的“分形招牌”Mandelbrot集中也随处可见法里和的影子。',
        )
        inversion_ref = Reference(
            name='Inversion in a Circle',
            authors='Tom Davis',
            pub='个人博客',
            doi='http://www.geometer.org/mathcircles/inversion.pdf',
            info='介绍反演变换的文章，结尾提到了笛卡尔定理与福特圆的关系。反演变换本身也是个相当实用的工具，这篇文章的例子也很丰富，不妨同时了解下。',
        )
        apollonian_ref = Reference(
            name='A Tisket, a Tasket, an Apollonian Gasket',
            authors='Dana Mackenzie',
            pub='The Best Writing on Mathematics, 2011: 13.',
            doi='https://www.americanscientist.org/article/a-tisket-a-tasket-an-apollonian-gasket',
            info='Apollonian Gasket是一类由无穷多相切圆构成的分形，而福特圆其实只是个特例，通常用(0,0,1,1)表示。',
        )
        return (
            mechanical_watch_ref, gears_ref1,
            gears_ref2, hurwitz_proof_ref1,
            hurwitz_proof_ref2, hurwitz_variant_ref,
            mandelbrot_ref, inversion_ref,
            apollonian_ref,
        )


class ThumbnailP2(Scene):
    def construct(self):
        ft = FareyTree(n_levels=4, include_top=False)
        ft.to_edge(LEFT, buff=0.5)
        colors = [RED, BLUE, GREEN]
        tex = Tex('{1 \\over 2}', '\\oplus', '{2 \\over 3}', '=', '{3 \\over 5}')
        tex.scale(2).to_edge(RIGHT)
        dejavu = TexText('（似曾相识...?）', color=YELLOW)
        dejavu.next_to(tex, DOWN, buff=0.5)
        frac_tuples = [(1, 2), (2, 3), (3, 5)]
        fracs = VGroup(*[
            ft.get_fraction_from_numbers(*pair)
            for pair in frac_tuples
        ])
        circles = VGroup(*[
            Circle(radius=frac.get_height() * 0.65, fill_opacity=0.2).move_to(frac)
            for frac in fracs
        ])
        back_circles = circles.copy()
        back_circles.set_color(BLACK).set_fill(opacity=1)
        for frac, text_mob, circle, color in zip(fracs, tex[::2], circles, colors):
            VGroup(frac, circle, text_mob).set_color(color)
        self.add(ft, back_circles, circles, tex, dejavu)
        self.add(fracs)
        self.wait()
