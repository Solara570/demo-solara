import itertools as it
from collections import Counter
from scipy.special import comb
from colorsys import hsv_to_rgb
from manimlib import *

# Constants
WALLIS_RECTS_COLORS = ["#FF0000", "#FF8000", "#FFFF00", "#00FF00", "#0080FF", "#9400D3"]
DEFAULT_UP_COLOR = BLUE
DEFAULT_DOWN_COLOR = RED
ALL_PATH_COLOR = YELLOW
BALANCED_COLOR = ORANGE
NONZERO_COLOR = GREEN
LIMIT_COLOR = RED
AXES_COLOR = GREY
WALLIS_COLOR = BLUE
AN_COLOR = MAROON_B
INNER_CORNER_COLOR = interpolate_color(WALLIS_RECTS_COLORS[-1], WHITE, 0.05)
OUTER_CORNER_COLOR = MAROON_B
FACTOR_COLOR = GOLD_B
BOUND_COLOR = RED
SUB_I_COLOR = YELLOW
SUB_J_COLOR = TEAL


# Functions
def a(n, alpha=1 / 2):
    # a(n) = C(-alpha, n)*(-1)^n = C(alpha+n-1, n)
    numer, denom = alpha + n - 1, n
    result = 1
    while denom >= 1:
        result *= numer / denom
        numer -= 1
        denom -= 1
    return result


def s(n, alpha=1 / 2):
    # s(n) = n * a(n) / alpha
    return n * a(n, alpha) / alpha


def get_cycle_wallis_colors(N, colors=WALLIS_RECTS_COLORS):
    repeat_times = N // len(colors) + 1
    extended_colors = list(it.chain(*it.repeat(colors, repeat_times)))
    return extended_colors[:N]


def jump_few_times(t, jumps=3):
    return np.abs(np.sin(jumps * PI * t))


def hsv_to_color(h, s, v):
    return Color(rgb=hsv_to_rgb(h, s, v))


def central_binomial_coeff(n):
    return int(comb(2 * n, n, exact=True))


def random_walk_string(k, char_pool="UD"):
    return "".join(random.choices(char_pool, k=k))


def inverse_walk_string(path_chars, char_pool="UD"):
    up_char, down_char = char_pool
    inverse_path_chars = (
        up_char if char == down_char else down_char
        for char in path_chars[::-1]
    )
    return "".join(inverse_path_chars)


def mapping_special_walk_string(path_chars, char_pool="UD"):
    balanced_flag = is_balanced(path_chars)
    nonzero_flag = is_nonzero(path_chars)
    if not (balanced_flag or nonzero_flag):
        raise Exception("This is not a balanced or nonzero walk.")

    if balanced_flag:
        split_ind = path_first_max_or_min_ind(path_chars, char_pool)
    else:
        split_ind = path_last_half_cross_ind(path_chars, char_pool)
    first_part, second_part = path_chars[:split_ind], path_chars[split_ind:]
    inv_first_part = inverse_walk_string(first_part)
    return inv_first_part + second_part


def is_balanced(path_chars):
    char_pool = Counter(path_chars)
    assert(len(char_pool) <= 2)
    if len(path_chars) == 0:
        return True
    elif len(char_pool) == 1:
        return False
    else:
        up_times, down_times = char_pool.values()
        return up_times == down_times


def is_nonzero(path_chars):
    char_pool = set(path_chars)
    assert(len(char_pool) <= 2)
    if (len(path_chars) == 0) or (len(char_pool) == 1):
        return True
    else:
        up_char, down_char = char_pool
        counter = 0
        path_shifts = map(lambda c: 1 if c == up_char else -1, path_chars)
        for shift in path_shifts:
            counter += shift
            if counter == 0:
                return False
        return True


def path_final_coord(path_chars, char_pool="UD"):
    up_char, down_char = char_pool
    return sum(1 if char == up_char else -1 for char in path_chars)


def path_max_and_min(path_chars, char_pool="UD"):
    up_char, down_char = char_pool
    path_max, path_min = 0, 0
    coord = 0
    for char in path_chars:
        if char == up_char:
            coord += 1
            path_max = max(path_max, coord)
        else:
            coord -= 1
            path_min = min(path_min, coord)
    return path_max, path_min


def path_all_zeros_ind(path_chars, char_pool="UD"):
    up_char, down_char = char_pool
    coord = 0
    all_zeros_ind = [0]
    for curr_ind, char in enumerate(path_chars, start=1):
        coord += (1 if char == up_char else -1)
        if coord == 0:
            all_zeros_ind.append(curr_ind)
    return all_zeros_ind


def path_last_zero_ind(path_chars, char_pool="UD"):
    return path_all_zeros_ind(path_chars, char_pool)[-1]


def path_first_max_or_min_ind(path_chars, char_pool="UD"):
    if len(path_chars) == 0:
        return 0
    up_char, down_char = char_pool
    path_max, path_min = path_max_and_min(path_chars, char_pool)
    coord = 0
    if path_chars[0] == up_char:
        # Finding first maximum
        for curr_ind, char in enumerate(path_chars, start=1):
            coord += (1 if char == up_char else -1)
            if coord == path_max:
                return curr_ind
    else:
        # Finding first minimum
        for curr_ind, char in enumerate(path_chars, start=1):
            coord += (1 if char == up_char else -1)
            if coord == path_min:
                return curr_ind


def path_last_half_cross_ind(path_chars, char_pool="UD"):
    assert (len(path_chars) % 2 == 0)
    up_char, down_char = char_pool
    path_shifts = [1 if char == up_char else -1 for char in path_chars]
    target = sum(path_shifts) // 2
    coord = 0
    last_half_cross_ind = 0

    if target > 0:
        for curr_ind, char in enumerate(path_chars[:-1], start=1):
            coord += (1 if char == up_char else -1)
            if all([coord == target, path_shifts[curr_ind - 1] > 0, path_shifts[curr_ind] > 0]):
                last_half_cross_ind = curr_ind
    else:
        for curr_ind, char in enumerate(path_chars[:-1], start=1):
            coord += (1 if char == up_char else -1)
            if all([coord == target, path_shifts[curr_ind - 1] < 0, path_shifts[curr_ind] < 0]):
                last_half_cross_ind = curr_ind
    return last_half_cross_ind


def all_random_walk_strings(k, char_pool="UD"):
    return list(
        "".join(chars)
        for chars in it.product(char_pool, repeat=k)
    )


def all_balanced_walk_strings(k, char_pool="UD"):
    return list(filter(is_balanced, all_random_walk_strings(k, char_pool)))


def all_nonzero_walk_strings(k, char_pool="UD"):
    return list(filter(is_nonzero, all_random_walk_strings(k, char_pool)))


def all_limited_walk_strings(k, min_thres=None, max_thres=None, char_pool="UD"):
    def max_filter(path_chars):
        if max_thres is not None:
            path_max, _ = path_max_and_min(path_chars, char_pool)
            return path_max <= max_thres
        return True

    def min_filter(path_chars):
        if min_thres is not None:
            _, path_min = path_max_and_min(path_chars, char_pool)
            return path_min >= min_thres
        return True

    return list(filter(
        lambda path_chars: max_filter(path_chars) and min_filter(path_chars),
        all_random_walk_strings(k, char_pool)
    ))


# Animations
class GrowPath(AnimationGroup):
    def __init__(
        self,
        path,
        **kwargs
    ):
        super().__init__(*[GrowArrow(arrow) for arrow in path], **kwargs)


class AssembleLayer(MoveToTarget):
    def __init__(
        self,
        wallis_rects,
        layer,
        distance_factor_range=(1.2, 1.8),
        scatter_range=(-PI / 3, PI / 3),
        angle_range=(-PI / 4, PI / 4),
        scale_factor_range=(0.9, 1.1),
        **kwargs,
    ):
        origin = wallis_rects.get_rects_corner()
        rect_group = wallis_rects.get_layer(layer)
        rect_group.generate_target()
        for rect in rect_group:
            curr_center = rect.get_center()
            vec = curr_center - origin
            distance = np.linalg.norm(vec)
            vec_unit = vec / distance
            distance_factor = (rect.get_width() + rect.get_height()) / 2 * random.uniform(*distance_factor_range)
            vec += rotate_vector(vec_unit, random.uniform(*scatter_range)) * distance_factor
            new_center = origin + vec
            rect.move_to(new_center)
            rect.rotate(random.uniform(*angle_range), about_point=new_center)
            rect.scale(random.uniform(*scale_factor_range), about_point=new_center)
            rect.set_opacity(0)
        super().__init__(rect_group, **kwargs)


# Mobjects
class Drunk(VMobject):
    def __init__(self, color=GREEN, direction=RIGHT, **kwargs):
        super().__init__(**kwargs)
        self.drunk = SVGMobject("drunk", height=1)
        self.add(self.drunk)
        self.set_color(color)
        self.add_foot_info(direction)
        self.rotate(angle_between_vectors(RIGHT, direction))
        self.step_on(ORIGIN)

    def add_foot_info(self, direction):
        self.foot = VectorizedPoint(self.get_bottom())
        self.foot_front = VectorizedPoint(
            self.get_bottom() + direction)
        self.add(self.foot, self.foot_front)

    def get_position(self):
        return self.foot.get_center()

    def step_on(self, position):
        self.shift(position - self.get_position())
        return self

    def get_direction(self):
        return self.foot_front.get_center() - self.foot.get_center()

    def change_direction(self):
        self.foot_front.shift(-2 * self.get_direction())

    def move_forward(self, step_size=1):
        self.shift(self.get_direction() * step_size)
        return self

    def turn_around(self):
        axis = rotate_vector(self.get_direction(), PI / 2)
        self.drunk.rotate(PI, axis=axis, about_point=self.get_position())
        self.change_direction()
        return self


class House(SVGMobject):
    def __init__(self):
        super().__init__("house")
        self.set_height(2.8)
        self.door_step = VectorizedPoint(
            self.get_bottom() + self.get_width() * RIGHT * 0.19)
        self.add(self.door_step)
        self.set_color(GREY_D)
        self.shift(-self.door_step.get_center())

    def get_position(self):
        return self.door_step.get_center()

    def put_on(self, position):
        self.shift(position - self.get_position())
        return self


class DarkCloud(SVGMobject):
    def __init__(self):
        super().__init__("darkcloud")
        self.set_width(2)
        self.set_color(GREY)
        self.set_stroke(width=0)
        self.set_fill(opacity=0.3)


class RandomWalkPath(VGroup):
    def __init__(
        self,
        path_chars,
        up_char="U", down_char="D",
        up_color=DEFAULT_UP_COLOR, down_color=DEFAULT_DOWN_COLOR,
        axes=None,
        unit_size=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path_chars = path_chars
        self.up_char, self.down_char = up_char, down_char
        self.char_pool = up_char + down_char
        self.up_color, self.down_color = up_color, down_color
        self.axes = axes
        if axes is not None:
            self.unit_size = axes.get_unit_size()
        else:
            self.unit_size = unit_size
        self.check_path_chars()
        self.generate_path()
        if axes is not None:
            self.put_start_at(axes.get_origin())

    def check_path_chars(self):
        assert all([
            char in (self.up_char, self.down_char)
            for char in self.path_chars
        ])

    def generate_path(self):
        for char in self.path_chars:
            if char == self.up_char:
                arrow = Vector((UP + RIGHT) * self.unit_size)
                arrow.set_color(self.up_color)
            else:
                arrow = Vector((DOWN + RIGHT) * self.unit_size)
                arrow.set_color(self.down_color)
            self.add(arrow)
        self.arrange_arrows()

    def arrange_arrows(self):
        point = ORIGIN
        for arrow in self:
            arrow.shift(point - arrow.get_start())
            point = arrow.get_end()
        return self

    def is_balanced(self):
        return is_balanced(self.path_chars)

    def is_nonzero(self):
        return is_nonzero(self.path_chars)

    def split_at_ind(self, ind, new_path=False):
        assert (ind >= 0 and ind <= len(self))
        if new_path:
            first_part = RandomWalkPath(self.path_chars[:ind], **self.get_path_config())
            second_part = RandomWalkPath(self.path_chars[ind:], **self.get_path_config())
            for arrow, template in zip(first_part, self[:ind]):
                arrow.match_style(template)
            for arrow, template in zip(second_part, self[ind:]):
                arrow.match_style(template)
            first_part.put_start_at(self.get_start())
            second_part.put_start_at(first_part.get_end())
        else:
            first_part = self[:ind]
            second_part = self[ind:]
        return (first_part, second_part)

    def put_start_at(self, position):
        start = self.get_arrow_by_index(0).get_start()
        self.shift(position - start)
        return self

    def get_path_config(self):
        return {
            "up_char": self.up_char,
            "down_char": self.down_char,
            "up_color": self.up_color,
            "down_color": self.down_color,
            "unit_size": self.unit_size,
            "axes": self.axes,
        }

    def get_arrow_by_index(self, ind):
        if len(self) == 0:
            return self
        else:
            assert (ind < len(self))
            return self[ind]

    def get_start(self):
        if len(self) == 0:
            return self.get_center()
        else:
            return self[0].get_start()

    def get_end(self):
        if len(self) == 0:
            return self.get_center()
        else:
            return self[-1].get_end()

    def get_start_and_end(self):
        return (self.get_start(), self.get_end())

    def get_final_coord(self):
        return path_final_coord(self.path_chars, self.char_pool)

    def get_path_max_and_min(self):
        if not hasattr(self, "path_max"):
            self.path_max, self.path_min = path_max_and_min(self.path_chars)
        return (self.path_max, self.path_min)

    def get_point_by_ind(self, ind):
        assert (ind >= 0 and ind <= len(self))
        if ind == len(self):
            return self.get_end()
        else:
            return self.get_arrow_by_index(ind).get_start()

    def get_all_zeros_indices(self):
        return path_all_zeros_ind(self.path_chars, self.char_pool)

    def get_all_zeros_points(self):
        return [
            self.get_point_by_ind(ind)
            for ind in self.get_all_zeros_indices()
        ]

    def get_last_zero_ind(self):
        return path_last_zero_ind(self.path_chars, self.char_pool)

    def get_last_zero_point(self):
        ind = self.get_last_zero_ind()
        return self.get_point_by_ind(ind)

    def get_first_extreme_ind(self):
        return path_first_max_or_min_ind(self.path_chars, self.char_pool)

    def get_first_extreme_point(self):
        ind = self.get_first_extreme_ind()
        return self.get_point_by_ind(ind)

    def get_last_half_cross_ind(self):
        return path_last_half_cross_ind(self.path_chars, self.char_pool)

    def get_last_half_cross_point(self):
        ind = self.get_last_half_cross_ind()
        return self.get_point_by_ind(ind)

    def get_inverse_path(self, **override_kwargs):
        path_config = self.get_path_config()
        path_config.update(override_kwargs)
        inverse_path_chars = inverse_walk_string(self.path_chars)
        inverse_path = RandomWalkPath(inverse_path_chars, **path_config)
        if path_config["axes"] is None:
            inverse_path.put_start_at(self.get_start())
        return inverse_path

    def get_range_rectangle(self, **rect_kwargs):
        if len(self) == 0:
            buff = 0
        else:
            arrow = self.get_arrow_by_index(0)
            buff = -arrow.thickness * arrow.tickness_multiplier / (4 * np.sqrt(2))
        kwargs = {
            "stroke_width": 0,
            "buff": buff,
            "fill_color": YELLOW,
            "fill_opacity": 0.4,
        }
        kwargs.update(rect_kwargs)
        range_rect = SurroundingRectangle(
            self,
            **kwargs,
        )
        # The rectangle's position is a bit off due to the arrow's solid stem
        if len(self) > 0:
            direction = self[-1].get_unit_vector()
            length = (3 / 4) * arrow.tickness_multiplier
            shift_vec = direction * length
            range_rect.shift(shift_vec)
        return range_rect

    def get_range_top(self, **line_kwargs):
        kwargs = {
            "color": LIMIT_COLOR,
        }
        kwargs.update(line_kwargs)
        range_top = Line(ORIGIN, RIGHT, **kwargs)
        range_rect = self.get_range_rectangle()
        range_top.match_width(range_rect)
        range_top.next_to(range_rect, UP, buff=0)
        return range_top

    def get_displacement_rect(self, right_side=True, shift=False, **rect_kwargs):
        start_point = self.get_start()
        end_point = self.get_end()
        if self.axes is not None:
            start_point = self.axes.c2p(*self.axes.p2c(start_point, round_to_integer=False))
            end_point = self.axes.c2p(*self.axes.p2c(end_point, round_to_integer=False))
        positive_coord = self.get_final_coord() > 0
        kwargs = {
            "width": 0.2,
            "height": abs((end_point - start_point)[1]),
            "stroke_width": 0,
            "fill_color": DEFAULT_UP_COLOR if positive_coord else DEFAULT_DOWN_COLOR,
            "fill_opacity": 0.3,
        }
        kwargs.update(rect_kwargs)
        rect = Rectangle(**kwargs)
        ref_point = end_point if right_side else start_point
        ref_direction = UP if right_side else DOWN
        ref_direction = ref_direction * (-1 if positive_coord else 1)
        if shift:
            ref_direction = ref_direction + (LEFT if right_side else RIGHT)
        rect.next_to(ref_point, ref_direction, buff=0)
        return rect


class PathAxes(VGroup):
    def __init__(
        self,
        N=4,
        pos_range=None,
        time_range=None,
        unit_size=1,
        include_label=True,
        pos_label_side=RIGHT,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.N = N
        if pos_range is None:
            pos_range = (-N, N, 1)
        if time_range is None:
            time_range = (0, N, 1)
        pos_axis = NumberLine(x_range=pos_range, unit_size=unit_size)
        time_axis = NumberLine(x_range=time_range, unit_size=unit_size)
        time_axis.shift(- time_axis.n2p(0))
        pos_axis.rotate(PI / 2)
        pos_axis.shift(time_axis.n2p(0) - pos_axis.n2p(0))
        self.pos_axis = pos_axis
        self.time_axis = time_axis
        self.add(pos_axis, time_axis)
        if include_label:
            pos_label = TexText("位置")
            pos_label.next_to(pos_axis, pos_label_side, aligned_edge=UP)
            self.pos_label = pos_label
            time_label = TexText("时间")
            time_label.next_to(time_axis, RIGHT)
            self.time_label = time_label
            self.add(pos_label, time_label)

    def get_origin(self):
        return self.time_axis.n2p(0)

    def get_unit_size(self):
        return self.time_axis.get_unit_size()

    def coords_to_point(self, time, pos):
        origin = self.get_origin()
        time_shift = self.time_axis.n2p(time) - origin
        pos_shift = self.pos_axis.n2p(pos) - origin
        return origin + time_shift + pos_shift

    def c2p(self, time, pos):
        return self.coords_to_point(time, pos)

    def point_to_coords(self, point, round_to_integer=True):
        time_coord = self.time_axis.p2n(point)
        pos_coord = self.pos_axis.p2n(point)
        if round_to_integer:
            time_coord = round(time_coord)
            pos_coord = round(pos_coord)
        return (time_coord, pos_coord)

    def p2c(self, point, round_to_integer=True):
        return self.point_to_coords(point, round_to_integer)


class UpDownGroup(VGroup):
    def __init__(self, is_up=True, unselected_opacity=0.2):
        super().__init__()
        self.up_symbol = Vector(UR).set_width(0.4)
        self.down_symbol = Vector(DR).set_width(0.4)
        self.is_up = is_up
        self.unselected_opacity = unselected_opacity
        self.add(self.up_symbol, self.down_symbol)
        self.arrange(DOWN, buff=0.3)
        if self.is_up:
            self.set_to_up()
        else:
            self.set_to_down()

    def get_up_symbol(self):
        return self.up_symbol

    def get_down_symbol(self):
        return self.down_symbol

    def get_selected_symbol(self):
        if self.is_up:
            return self.get_up_symbol()
        else:
            return self.get_down_symbol()

    def get_unselected_symbol(self):
        if not self.is_up:
            return self.get_up_symbol()
        else:
            return self.get_down_symbol()

    def get_up_to_down_vec(self):
        return self.down_symbol.get_center() - self.up_symbol.get_center()

    def set_to_up(self):
        if not self.is_up:
            self.shift(-self.get_up_to_down_vec())
        self.up_symbol.set_opacity(1)
        self.down_symbol.set_opacity(self.unselected_opacity)
        self.is_up = True
        return self

    def set_to_down(self):
        if self.is_up:
            self.shift(self.get_up_to_down_vec())
        self.up_symbol.set_opacity(self.unselected_opacity)
        self.down_symbol.set_opacity(1)
        self.is_up = False
        return self

    def move_selection_to(self, pos):
        if self.is_up:
            self.shift(pos - self.get_up_symbol().get_center())
        else:
            self.shift(pos - self.get_down_symbol().get_center())
        return self


class UpDownSelections(VGroup):
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.up_char = path.up_char
        self.down_char = path.down_char
        self.up_color = path.up_color
        self.down_color = path.down_color
        self.selections = path.path_chars
        for char in self.selections:
            is_up = (char == self.up_char)
            ud_group = UpDownGroup(is_up=is_up)
            ud_group.move_selection_to(ORIGIN)
            ud_group.get_up_symbol().set_color(self.up_color)
            ud_group.get_down_symbol().set_color(self.down_color)
            self.add(ud_group)
        self.arrange(RIGHT, buff=0.3, coor_mask=[1, 0, 0])
        self.center()


class SetCircle(VGroup):
    def __init__(self, text="", radius=1, color=WHITE, **kwargs):
        super().__init__(**kwargs)
        bg_circle = Circle(
            radius=radius,
            fill_opacity=1, fill_color=BLACK, stroke_width=0,
        )
        main_circle = Circle(
            radius=radius,
            fill_opacity=0.3, fill_color=color, stroke_color=color,
        )
        text = Tex(text)
        text.scale(0.8).set_color(color).set_backstroke(width=2)
        self.bg_circle = bg_circle
        self.main_circle = main_circle
        self.text = text
        self.add(bg_circle, main_circle, text)

    def get_main_color(self):
        return self.main_circle.get_color()

    def get_arc_to(self, other_set_circle, angle=PI / 6):
        start = self.get_center()
        end = other_set_circle.get_center()
        arc = ArcBetweenPoints(start, end, angle)
        start_color = self.get_main_color()
        end_color = other_set_circle.get_main_color()
        arc.set_color([start_color, end_color])
        return arc


class LengthLabel(VGroup):
    def __init__(self, mob, text="", color=WHITE, **kwargs):
        super().__init__(**kwargs)
        label = Tex(text)
        label.scale(0.75)
        label.move_to(mob)
        sur_rect = SurroundingRectangle(
            label, color=color,
            fill_opacity=1, fill_color=BLACK, buff=0.2,
        )
        left_arrow = Arrow(sur_rect.get_left(), mob.get_left(), buff=0, tip_width_ratio=3)
        right_arrow = Arrow(sur_rect.get_right(), mob.get_right(), buff=0, tip_width_ratio=3)
        VGroup(label, left_arrow, right_arrow).set_color(color)
        self.add(left_arrow, right_arrow, sur_rect, label)
        self.label = label
        self.frame = VGroup(left_arrow, right_arrow, sur_rect)

    def get_label(self):
        return self.label

    def get_frame(self):
        return self.frame


class WallisRectangles(VGroup):
    rect_config = {
        "stroke_width": 0,
        "stroke_color": GREY,
        "fill_opacity": 1,
    }

    def __init__(self, N=10, alpha=1 / 2, colors=None, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.alpha = alpha
        if colors is None:
            self.rect_colors = tuple(
                hsv_to_color(k / (self.N - 1) * 0.85, 1.0, 1.0)
                for k in range(self.N)
            )
        else:
            self.rect_colors = color_gradient(colors, self.N)
        self.init_rect_groups()
        self.arrange_rects()
        self.add_corner()
        self.put_corner_at(ORIGIN)

    def init_rect_groups(self):
        self.rect_groups = VGroup()
        for k in range(self.N):
            rect_group = VGroup(
                Rectangle(
                    width=a(ind, self.alpha),
                    height=a(k - ind, self.alpha),
                    fill_color=self.rect_colors[k],
                    **self.rect_config,
                )
                for ind in range(k + 1)
            )
            self.rect_groups.add(rect_group)
        self.add(self.rect_groups)

    def arrange_rects(self):
        for group_ind in range(1, self.N):
            curr_group = self.rect_groups[group_ind]
            prev_group = self.rect_groups[group_ind - 1]
            curr_group[0].next_to(prev_group[0], UP, buff=0)
            for mob, ref in zip(curr_group[1:], prev_group):
                mob.next_to(ref, RIGHT, buff=0)

    def add_corner(self):
        self.corner = VectorizedPoint()
        self.corner.next_to(self.rect_groups, DL, buff=0)
        self.add(self.corner)

    def get_rects_corner(self):
        return self.corner.get_center()

    def put_corner_at(self, pos):
        shift_vec = pos - self.get_rects_corner()
        self.shift(shift_vec)
        return self

    def get_unit_size(self):
        return self.rect_groups[0].get_width()

    def set_unit_size(self, unit_size):
        curr_unit_size = self.get_unit_size()
        curr_corner = self.get_rects_corner()
        if curr_unit_size > 1e-6:
            scale_factor = unit_size / curr_unit_size
            self.scale(scale_factor, about_point=self.get_rects_corner())
            return self

    def get_number_of_layers(self):
        return len(self.rect_groups)

    def get_all_layers(self):
        return self.rect_groups

    def get_layer(self, layer):
        return self.rect_groups[layer]

    def get_layers(self, layers):
        return VGroup(
            self.get_layer(layer)
            for layer in layers
        )

    def get_layer_corners(self, layer):
        return [
            rect.get_corner(UR)
            for rect in self.get_layer(layer)
        ]

    def get_inner_corners(self):
        return self.get_layer_corners(-2)

    def get_outer_corners(self):
        return self.get_layer_corners(-1)

    def get_gridlines(self, length_factor=1.1):
        length = self.rect_groups.get_width() * length_factor
        horiz_lines = VGroup(
            DashedLine(ORIGIN, length * RIGHT)
            for _ in range(len(self.rect_groups) + 1)
        )
        horiz_lines[0].next_to(self.rect_groups, DOWN, buff=0)
        for line, rect_group in zip(horiz_lines[1:], self.rect_groups):
            line.move_to(self.rect_groups)
            line.next_to(rect_group, UP, buff=0, coor_mask=[0, 1, 0])
        vert_lines = VGroup(
            DashedLine(ORIGIN, length * UP)
            for _ in range(len(self.rect_groups) + 1)
        )
        vert_lines[0].next_to(self.rect_groups, LEFT, buff=0)
        for line, rect_group in zip(vert_lines[1:], self.rect_groups):
            line.move_to(self.rect_groups)
            line.next_to(rect_group, RIGHT, buff=0, coor_mask=[1, 0, 0])
        gridlines = VGroup(horiz_lines, vert_lines)
        gridlines.set_stroke(width=2)
        gridlines.set_color(GREY)
        return gridlines

    def get_gridline_labels(self, length_factor=1.1, scale_factor=0.8):
        horiz_lines, vert_lines = self.get_gridlines(length_factor)
        horiz_labels = VGroup(
            Tex(f"s_{{{subscript}}}")
            for subscript in range(len(vert_lines))
        )
        horiz_labels.scale(scale_factor)
        vert_labels = horiz_labels.deepcopy()
        for line, label in zip(horiz_lines, vert_labels):
            label.next_to(line, LEFT)
        for line, label in zip(vert_lines, horiz_labels):
            label.next_to(line, DOWN)
        return VGroup(horiz_labels, vert_labels)

    def get_length_labels(self, size_factor=0.6):
        horiz_length_labels = VGroup(
            Tex(f"a_{{{subscript}}}")
            for subscript in range(len(self.rect_groups))
        )
        vert_length_labels = horiz_length_labels.deepcopy()
        for label, rect_group in zip(horiz_length_labels, self.rect_groups):
            rect = rect_group[-1]
            label.set_width(min(label.get_width(), rect.get_width() * size_factor))
            label.next_to(rect, DOWN)
        for label, rect_group in zip(vert_length_labels, self.rect_groups):
            rect = rect_group[0]
            label.set_height(min(label.get_height(), rect.get_height() * size_factor))
            label.next_to(rect, LEFT)
        length_labels = VGroup(horiz_length_labels, vert_length_labels)
        return length_labels

    def get_area_labels(self, size_factor=0.6):
        area_labels = VGroup()
        for curr_sum, rect_group in enumerate(self.rect_groups):
            group_labels = VGroup()
            for curr_x, rect in enumerate(rect_group):
                curr_y = curr_sum - curr_x
                label_text = f"a_{{{curr_x}}} a_{{{curr_y}}}"
                label = Tex(label_text)
                label.set_width(min(label.get_width(), rect.get_width() * size_factor))
                label.set_height(min(label.get_height(), rect.get_height() * size_factor))
                label.move_to(rect)
                group_labels.add(label)
            group_labels.set_color(BLACK)
            area_labels.add(group_labels)
        return area_labels

    def get_cover_sector(self, num_segments=50, alpha_override=None, layer=None):
        if alpha_override is not None:
            alpha = alpha_override
        else:
            alpha = self.alpha
        if layer is not None:
            cover_mob = self.get_layer(layer)
        else:
            cover_mob = self.rect_groups
        ref = VMobject().set_points_as_corners([UP, ORIGIN, RIGHT])
        sector = VMobject()
        sector.set_points_as_corners([
            ref.point_from_proportion(k / (2 * num_segments))
            for k in range(2 * num_segments + 1)
        ])
        curve = ParametricCurve(
            t_func=lambda t: [
                pow(np.cos(t), 2 * alpha),
                pow(np.sin(t), 2 * alpha),
                0
            ],
            t_range=(0, PI / 2, PI / (2 * num_segments)),
        )
        sector.append_vectorized_mobject(curve)
        sector.close_path()
        sector.set_style(
            fill_opacity=0.5, fill_color=GREY,
            stroke_width=5, stroke_color=GREY,
        )
        sector.match_width(cover_mob)
        sector.move_to(cover_mob)
        return sector


class Die(VGroup):
    def __init__(self, num_pips, **kwargs):
        super().__init__(**kwargs)
        frame = RoundedRectangle(
            width=1, height=1,
            corner_radius=0.2,
        )
        frame.set_stroke(width=2)
        frame.set_fill(color=BLUE, opacity=0.2)
        pips = self.get_pips(num_pips)
        self.add(frame, pips)
        self.set_height(0.25)

    def get_pips(self, num_pips):
        pattern = VGroup(Dot() for k in range(9))
        pattern.arrange_in_grid(3, 3, buff=0.1)
        match num_pips:
            case 1:
                indices_to_dim = [0, 1, 2, 3, 5, 6, 7, 8]
            case 2:
                indices_to_dim = [0, 2, 3, 4, 5, 6, 8]
            case 3:
                indices_to_dim = [0, 1, 3, 5, 7, 8]
            case 4:
                indices_to_dim = [1, 3, 4, 5, 7]
            case 5:
                indices_to_dim = [1, 3, 5, 7]
            case 6:
                indices_to_dim = [1, 4, 7]
        for ind in indices_to_dim:
            pattern[ind].set_opacity(0)
        if num_pips == 1:
            pattern[4].scale(1.5).set_color(MAROON_B)
        return pattern


# Things about references that borrowed from previous projects
class Reference(VGroup):
    name_color = BLUE
    author_color = BLUE_A
    pub_color = GREY_A
    doi_color = GREY_A
    info_color = GOLD_A

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

        textext_config = {'alignment': ''}
        text_config = {'font': 'CMU Sans Serif'}

        text_types = [TexText, TexText, TexText, Text, TexText]
        text_configs = [
            textext_config if text_type == TexText else text_config
            for text_type in text_types
        ]
        texts = [name, authors, pub, doi, info]
        colors = [self.name_color, self.author_color, self.pub_color, self.doi_color, self.info_color]
        scale_factors = [1, 0.7, 0.6, 0.6, 0.7]
        texs = VGroup(*[
            TextType(text, **config).scale(factor).set_color(color)
            for TextType, config, text, color, factor in zip(
                text_types, text_configs, texts, colors, scale_factors,
            )
        ])
        texs.arrange(DOWN, aligned_edge=LEFT)
        texs[2:-1].shift(0.5 * RIGHT)
        texs.scale(0.8)
        self.add(*texs)


class ReferenceScene(Scene):
    pause_time_between_pages = 5

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


# Scenes
class DrunkWanderScene(Scene):
    random_seed = 227

    def play_drunk_move_animation(
        self,
        drunk, pos,
        *other_anims,
        traceback=False,
        run_time=1,
        turn_ratio=0.1
    ):
        def is_same_direction(drunk, pos, traceback):
            vec = pos - drunk.get_position()
            direction_flag = (np.dot(vec, drunk.get_direction()) > 0)
            return direction_flag ^ traceback

        if is_same_direction(drunk, pos, traceback):
            self.play(
                ApplyMethod(drunk.step_on, pos, run_time=run_time),
                *other_anims,
            )
        else:
            turn_time = run_time * turn_ratio
            move_time = run_time * (1 - turn_ratio)
            for anim in other_anims:
                anim.run_time *= (1 - turn_ratio)
            self.play(
                ApplyMethod(drunk.turn_around, run_time=turn_time),
            )
            self.play(
                ApplyMethod(drunk.step_on, pos, run_time=move_time),
                *other_anims,
            )


class IntroToWallisProduct(Scene):
    def construct(self):
        # Pi shows up in many infinite series
        pi = Tex("\\pi").scale(5)
        circle = Circle()
        VGroup(pi, circle).set_color(YELLOW)
        circle.generate_target()
        circle.target.surround(pi, buff=1)
        circle.surround(pi, buff=0).set_opacity(0)
        formula_config = {
            "isolate": "\\pi",
            "t2c": {"\\pi": YELLOW},
        }
        formula_group = VGroup(
            Tex("1 - \\frac{1}{3^1} + \\frac{1}{5^1} - \\frac{1}{7^1} + \\frac{1}{9^1} - \\frac{1}{11^1} + \\cdots = \\frac{\\pi^1}{4}", **formula_config),
            Tex("1 + \\frac{1}{3^2} + \\frac{1}{5^2} + \\frac{1}{7^2} + \\frac{1}{9^2} + \\frac{1}{11^2} + \\cdots = \\frac{\\pi^2}{8}", **formula_config),
            Tex("1 - \\frac{1}{3^3} + \\frac{1}{5^3} - \\frac{1}{7^3} + \\frac{1}{9^3} - \\frac{1}{11^3} + \\cdots = \\frac{\\pi^3}{32}", **formula_config),
            Tex("1 + \\frac{1}{3^4} + \\frac{1}{5^4} + \\frac{1}{7^4} + \\frac{1}{9^4} + \\frac{1}{11^4} + \\cdots = \\frac{\\pi^4}{96}", **formula_config),
            Tex("\\cdots"),
        )
        formula_group.set_color_by_gradient(BLUE, GREEN)
        for tex in formula_group:
            tex.set_color_by_tex_to_color_map(formula_config["t2c"])
        formula_leibniz_alt = Tex(
            "1 - \\frac{1}{3} + \\frac{1}{5} - \\frac{1}{7} + \\frac{1}{9} - \\frac{1}{11} + \\cdots = \\frac{\\pi}{4}",
            **formula_config,
        )
        formula_basel = Tex(
            "1 + \\frac{1}{2^2} + \\frac{1}{3^2} + \\frac{1}{4^2} + \\frac{1}{5^2} + \\frac{1}{6^2} + \\cdots = \\frac{\\pi^2}{6}",
            **formula_config,
        )
        formula_group.arrange(DOWN, buff=0.5)
        formula_group.scale(0.9).to_edge(UP, buff=0.75)
        formula_leibniz_alt.shift(UP)
        formula_basel.shift(DOWN)
        pi_targets = VGroup(formula_leibniz_alt[-3].copy(), formula_basel[-4].copy())
        self.play(Write(pi))
        self.wait()
        self.play(MoveToTarget(circle))
        self.wait()
        self.play(
            Write(formula_leibniz_alt, run_time=3),
            Write(formula_basel, run_time=3),
            FadeOut(circle, shift=DOWN, run_time=1),
            ReplacementTransform(pi[0], pi_targets[0], run_time=2.5),
            ReplacementTransform(pi.copy()[0], pi_targets[1], run_time=2.5),
        )
        self.wait()
        self.remove(pi_targets)
        self.play(
            AnimationGroup(
                FadeTransform(formula_leibniz_alt, formula_group[0]),
                FadeTransform(formula_basel, formula_group[1]),
                *[
                    FadeIn(formula, shift=DOWN)
                    for formula in formula_group[2:]
                ],
                lag_ratio=0.4,
                run_time=3,
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    FlashAround(formula.get_part_by_tex("\\pi"))
                    for formula in formula_group[:-1]
                ],
                lag_ratio=0.1,
                run_time=3,
            )
        )
        self.wait()

        # Same goes for products
        add_sign = Tex("+").scale(3)
        mult_sign = Tex("\\times").scale(3)

        self.play(
            FadeOut(formula_group),
            FadeIn(add_sign),
        )
        self.wait()
        self.play(
            ApplyMethod(
                add_sign.rotate, PI / 4,
                path_arc=PI / 4,
                rate_func=overshoot,
            )
        )
        self.remove(add_sign)
        fullscreen_rect = FullScreenFadeRectangle(
            fill_opacity=0.8, z_index=-1,
        )
        fullscreen_rect.scale(1.05)
        self.add(fullscreen_rect)
        self.add(mult_sign)
        self.wait()

        # An infinite product: the Wallis product - set things up
        def wallis_numer(n):
            return (n + 1) if n % 2 == 1 else n

        def wallis_denom(n):
            return n if n % 2 == 1 else (n + 1)

        n_terms = 7
        n_walks = 30
        wallis_prod = Tex("".join(
            [
                "\\frac{%d}{%d} \\times" % (wallis_numer(n), wallis_denom(n))
                for n in range(1, n_terms + 1)
            ] + ["\\cdots"]
        ))
        result = Tex("= \\frac{\\pi}{2}")
        result.scale(1.8)
        pi = result[1]
        circle = Circle(color=YELLOW)
        circle.surround(pi, buff=0.15)
        VGroup(pi, circle).set_color(YELLOW)
        result_group = VGroup(result, circle)
        result_group.next_to(wallis_prod, DOWN)
        group = VGroup(wallis_prod, result_group)
        group.center().set_width(FRAME_WIDTH - 2)

        # The structure of Wallis product
        wallis_prod.generate_target()
        wallis_prod.center()
        wallis_prod_copy = wallis_prod.copy()
        wallis_mult_signs = wallis_prod_copy.get_parts_by_tex("\\times")
        other_parts = wallis_prod_copy.remove(*[
            wallis_mult_signs[k][0]
            for k in range(len(wallis_mult_signs))
        ])
        self.play(
            ReplacementTransform(
                mult_sign.replicate(len(wallis_mult_signs)),
                wallis_mult_signs,
                lag_ratio=0.02,
                run_time=3,
            ),
            Write(other_parts, run_time=3),
            FadeOut(mult_sign, run_time=1),
        )
        self.remove(wallis_mult_signs, other_parts)
        self.add(wallis_prod)
        self.wait()

        trace_indices = list(range(0, 27, 2))
        trace_colors = [BLUE for ind in trace_indices]
        for k in range(len(trace_indices)):
            if k % 4 == 0:
                trace_indices[k] += 2
            elif k % 4 == 1:
                trace_indices[k] -= 2
                trace_colors[k] = YELLOW
            elif k % 4 == 2:
                trace_colors[k] = YELLOW
        trace = VMobject(stroke_opacity=0.7)
        trace.set_points_as_corners([
            wallis_prod[ind].get_center()
            for ind in trace_indices
        ])
        trace.set_color(trace_colors)
        self.play(
            VShowPassingFlash(trace, time_width=2, rate_func=linear),
            run_time=4,
        )
        self.wait()
        wallis_portrait = ImageMobject("JohnWallis.jpg")
        wallis_name = TexText("John Wallis")
        wallis_group = Group(wallis_name, wallis_portrait)
        wallis_group.arrange(DOWN)
        wallis_group.scale(0.75)
        wallis_group.to_corner(DL)
        self.play(
            MoveToTarget(wallis_prod),
            FadeIn(wallis_group, shift=RIGHT),
        )
        self.wait()
        self.play(Write(result), run_time=2)
        self.wait()
        self.play(
            FadeOut(wallis_group),
            ShowCreation(circle),
        )
        self.wait()

        # Tease the connection between paths and "rainbow" rectangles
        bg_rect = BackgroundRectangle(group, fill_opacity=0.6, buff=0.2)
        random_paths = VGroup(
            *[
                RandomWalkPath(random_walk_string(30), unit_size=0.5)
                for k in range(n_walks)
            ],
            z_index=-2,
        )
        for k, walk in enumerate(random_paths):
            walk.shift(random.randrange(-5, 5) * 2 * walk.unit_size * UP)
        random_paths.center()

        self.play(
            AnimationGroup(
                *[
                    GrowPath(path, lag_ratio=0.1)
                    for path in random_paths
                ],
                Animation(fullscreen_rect),
                Animation(bg_rect),
                Animation(group),
                run_time=10,
            ),
        )
        self.wait()

        question = Tex("?")
        question.set_color(circle.get_color())
        question.scale(2).next_to(circle, RIGHT, buff=0.4)
        self.play(FadeIn(question, shift=RIGHT))
        self.wait()

        rainbow = VGroup(
            WallisRectangles(N=50, colors=WALLIS_RECTS_COLORS).rotate(k * PI / 2, about_point=ORIGIN)
            for k in range(4)
        )
        rainbow.set_opacity(0.7)
        rainbow.match_width(circle)
        rainbow.move_to(circle)
        self.play(
            AnimationGroup(
                *[
                    FadeIn(
                        VGroup(part.get_layer(k) for part in rainbow),
                        rate_func=lambda t: there_and_back_with_pause(t, 1 / 2.),
                    )
                    for k in range(rainbow[0].get_number_of_layers())
                ],
                lag_ratio=0.05,
            ),
            run_time=1,
        )
        self.wait()


class StartWithDrunkWandering(DrunkWanderScene):
    def construct(self):
        # Start with drunk wandering
        drunk = Drunk(color=ALL_PATH_COLOR, z_index=1)
        house = House()
        pos_axis = NumberLine(color=BLUE, include_tip=True)

        drunk.generate_target()
        drunk.rotate(1.5 * PI).scale(0)
        self.play(
            MoveToTarget(drunk, path_arc=PI),
            FadeIn(house, shift=DOWN),
            GrowFromPoint(pos_axis, ORIGIN),
            run_time=2,
        )
        self.wait()

        time_text = Tex("t = 0")
        time_text.next_to(ORIGIN, DOWN, buff=1)
        curr_time_text = time_text[-1]
        drunk.save_state()
        self.play(Write(time_text))
        self.wait()

        # Show a few steps and shortcoming for 1-d graph
        total_time = 60
        steps = random.choices([-1, 1], k=total_time)
        num_of_steps_to_show = 15
        coord = 0
        for k, step in enumerate(steps[:num_of_steps_to_show], start=1):
            new_time = k
            new_time_text = Tex(str(new_time))
            new_time_text.move_to(curr_time_text, aligned_edge=LEFT)
            coord += step
            new_pos = pos_axis.n2p(coord)
            self.play_drunk_move_animation(
                drunk, new_pos,
                FadeOut(curr_time_text, shift=0.5 * UP),
                FadeIn(new_time_text, shift=0.5 * UP),
            )
            curr_time_text = new_time_text
        self.wait()

        self.play(
            Restore(drunk),
            FadeOut(curr_time_text, shift=0.5 * DOWN),
            FadeIn(time_text[-1], shift=0.5 * DOWN),
        )
        self.wait()

        # Introduce another axis for time, and we have a 2-d graph
        time_axis = NumberLine(
            x_range=(0, 10, 1), include_tip=True,
            color=WHITE,
        )
        time_axis.rotate(-PI / 2)
        time_axis.shift(-time_axis.n2p(0))

        self.play(
            GrowFromPoint(time_axis, ORIGIN, run_time=1),
            FadeOutToPoint(time_text, ORIGIN, run_time=0.5),
            ApplyMethod(pos_axis.set_color, WHITE, run_time=0.5),
        )
        self.wait()

        new_pos_axis = NumberLine(
            x_range=(-3, 3.5, 1), include_tip=True,
            color=pos_axis.get_color(),
        )
        new_pos_axis.shift(-new_pos_axis.n2p(0))
        group = VGroup(house, time_axis, pos_axis, new_pos_axis, drunk)
        group.generate_target()
        group.target.rotate(PI / 2, about_point=ORIGIN)
        group.target.shift(3.75 * LEFT)
        new_pos_axis.set_opacity(0)
        group.target[2].set_opacity(0)
        self.play(MoveToTarget(group, path_arc=PI / 2), run_time=2)
        self.add(house, time_axis, new_pos_axis, drunk)
        self.wait()

        pos_label = TexText("位置")
        pos_label.match_color(new_pos_axis)
        pos_label.next_to(new_pos_axis, RIGHT, aligned_edge=UP)
        time_label = TexText("时间")
        time_label.match_color(time_axis)
        time_label.next_to(time_axis, DOWN, aligned_edge=RIGHT)
        self.play(
            AnimationGroup(
                Write(pos_label),
                Write(time_label),
                lag_ratio=0.5,
                run_time=1,
            )
        )
        self.wait()

        # An example - a path of length 8
        steps_to_show = 8
        path_chars = "".join(map(lambda n: "U" if n > 0 else "D", steps[:steps_to_show]))
        path = RandomWalkPath(path_chars)
        new_origin = time_axis.n2p(0)
        path.put_start_at(new_origin)

        indicator = VectorizedPoint().move_to(new_origin)

        def update_vert_line(line, width_thres=0.1):
            line.set_color(YELLOW)
            line.set_stroke(width=1)
            start = indicator.get_center()
            end = start.copy()
            end[1] = new_origin[1]
            line.put_start_and_end_on(start, end)

        def update_horiz_line(line, width_thres=0.1):
            line.set_color(YELLOW)
            line.set_stroke(width=1)
            start = indicator.get_center()
            end = start.copy()
            end[0] = new_origin[0]
            line.put_start_and_end_on(start, end)

        def update_glow_dot(dot):
            pos = indicator.get_center()
            dot.move_to(pos)
            dot.set_opacity(
                np.clip(np.linalg.norm(pos - new_origin), 0, 1)
            )

        vert_line, horiz_line = Line(), Line()
        glow_dot = GlowDot()
        vert_line.add_updater(update_vert_line)
        horiz_line.add_updater(update_horiz_line)
        glow_dot.add_updater(update_glow_dot)
        self.add(vert_line, horiz_line, glow_dot)

        coord = 0
        for char, arrow in zip(path_chars, path):
            coord += (1 if char == "U" else -1)
            self.play_drunk_move_animation(
                drunk, new_pos_axis.n2p(coord),
                GrowArrow(arrow),
                ApplyMethod(indicator.shift, arrow.get_vector()),
                run_time=1,
            )
        self.wait()

        # Define 'length' and showcasing many paths of the same length
        fixed_length_text = TexText("长度$=8$")
        variable_length_text = TexText("长度$=2n$")
        for text in (fixed_length_text, variable_length_text):
            text.set_color(ALL_PATH_COLOR)
            text.next_to(pos_label, RIGHT, buff=3)
        glow_dot.clear_updaters()
        self.play(
            FadeOut(Group(vert_line, horiz_line, indicator, glow_dot))
        )
        self.wait()
        self.play(
            Write(fixed_length_text),
            AnimationGroup(*[Indicate(arrow) for arrow in path], lag_ratio=0.1),
            run_time=2,
        )
        self.wait()

        # More examples of normal paths (no constraints)
        path.save_state()
        drunk.save_state()
        limited_path_chars = all_limited_walk_strings(8, max_thres=2, min_thres=-3)
        num_normal_samples = 5
        for alt_path_chars in random.sample(limited_path_chars, num_normal_samples):
            alt_path = RandomWalkPath(alt_path_chars)
            alt_path.put_start_at(new_origin)
            alt_coord = path_final_coord(alt_path_chars)
            alt_drunk = drunk.saved_state.copy()
            alt_drunk.step_on(pos_axis.n2p(alt_coord))
            if alt_path_chars[-1] != path_chars[-1]:
                alt_drunk.turn_around()
            self.play(
                Transform(path, alt_path),
                Transform(drunk, alt_drunk),
            )
            self.wait()
        self.play(Restore(path), Restore(drunk))
        self.wait()
        self.play(FadeTransform(fixed_length_text[-1], variable_length_text[-2:]))
        self.remove(fixed_length_text)
        self.add(variable_length_text)
        self.wait()

        # Definition of set 'Wn' and its size 'wn'
        wn_set_text = Tex("W_n: \\left\\{ \\text{长度为} 2n \\text{的路径} \\right\\}")
        wn_set_text.next_to(pos_label, RIGHT, buff=1)
        wn_num_text = Tex("w_n = |W_n|")
        wn_num_text.next_to(wn_set_text, DOWN, aligned_edge=LEFT)
        wn_group = VGroup(wn_set_text, wn_num_text)
        wn_group.set_color(ALL_PATH_COLOR)
        self.play(FadeTransform(variable_length_text, wn_set_text))
        self.wait()
        self.play(
            FadeTransform(wn_set_text[:2].copy(), wn_num_text[-3:-1]),
            Write(
                VGroup(wn_num_text[:-3], wn_num_text[-1]),
                stroke_color=ALL_PATH_COLOR,
            ),
        )
        self.wait()
        self.add(wn_set_text, wn_num_text)
        self.play(
            Indicate(wn_set_text[1], color=BLUE),
            Indicate(wn_set_text[7:9], color=BLUE),
            run_time=2,
        )
        self.wait()

        # There're two types of paths we care about
        # 1. Balanced paths
        pn_set_text = Tex("P_n: \\left\\{ \\text{长度为} 2n \\text{的“正点到家”路径} \\right\\}")
        pn_set_text.move_to(wn_set_text)
        pn_set_text.next_to(wn_set_text, DOWN, aligned_edge=LEFT, coor_mask=[1, 0, 0])
        pn_num_text = Tex("p_n = |P_n|")
        pn_num_text.next_to(pn_set_text, DOWN, aligned_edge=LEFT)
        pn_group = VGroup(pn_set_text, pn_num_text)
        pn_group.set_color(BALANCED_COLOR)

        alt_balanced_path_chars = list(filter(is_balanced, limited_path_chars))
        random.shuffle(alt_balanced_path_chars)
        balanced_path_chars = alt_balanced_path_chars[-2]
        balanced_path = RandomWalkPath(balanced_path_chars)
        balanced_path.put_start_at(new_origin)
        balanced_drunk = drunk.copy()
        balanced_drunk.step_on(new_origin)
        if balanced_path_chars[-1] != path_chars[-1]:
            balanced_drunk.turn_around()
        balanced_path.save_state()
        VGroup(balanced_path, balanced_drunk).set_color(BALANCED_COLOR)
        balanced_drunk.save_state()
        self.play(
            Transform(path, balanced_path),
            Transform(drunk, balanced_drunk),
            FadeOut(VGroup(wn_set_text, wn_num_text)),
            Write(pn_set_text[-9:-3], stroke_color=BALANCED_COLOR),
        )
        self.remove(path, drunk)
        self.add(balanced_path, balanced_drunk)
        self.wait()
        self.play(
            ApplyMethod(balanced_drunk.shift, 0.3 * LEFT, rate_func=jump_few_times),
            run_time=2,
        )
        self.wait()

        # More examples of balanced paths
        num_balanced_samples = 3
        for alt_path_chars in alt_balanced_path_chars[:num_balanced_samples]:
            alt_path = RandomWalkPath(alt_path_chars)
            alt_path.put_start_at(new_origin)
            alt_coord = path_final_coord(alt_path_chars)
            alt_drunk = balanced_drunk.saved_state.copy()
            alt_drunk.step_on(pos_axis.n2p(alt_coord))
            if alt_path_chars[-1] != balanced_path_chars[-1]:
                alt_drunk.turn_around()
            VGroup(alt_path, alt_drunk).set_color(BALANCED_COLOR)
            self.play(
                Transform(balanced_path, alt_path),
                Transform(balanced_drunk, alt_drunk),
            )
            self.wait()
        self.play(
            FocusOn(balanced_path.get_end(), opacity=0.2, color=BALANCED_COLOR),
            Flash(balanced_path.get_end(), color=BALANCED_COLOR),
        )
        self.wait()

        # Definition of set 'Pn' and its size 'pn'
        self.play(
            Write(
                VGroup(pn_set_text[:-9], pn_set_text[-3:]),
                stroke_color=BALANCED_COLOR,
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeTransform(pn_set_text[:2].copy(), pn_num_text[-3:-1]),
            Write(
                VGroup(pn_num_text[:-3], pn_num_text[-1]),
                stroke_color=BALANCED_COLOR,
            ),
        )
        self.wait()
        self.add(pn_set_text, pn_num_text)
        self.play(Restore(balanced_path), Restore(balanced_drunk))
        self.wait()
        self.play(
            Indicate(pn_set_text[1], color=BLUE),
            Indicate(pn_set_text[7:9], color=BLUE),
            run_time=2,
        )
        self.wait()

        # 2. Nonzero paths
        qn_set_text = Tex("Q_n: \\left\\{ \\text{长度为} 2n \\text{的“不再回家”路径} \\right\\}")
        qn_set_text.move_to(wn_set_text)
        qn_set_text.next_to(wn_set_text, DOWN, aligned_edge=LEFT, coor_mask=[1, 0, 0])
        qn_num_text = Tex("q_n = |Q_n|")
        qn_num_text.next_to(qn_set_text, DOWN, aligned_edge=LEFT)
        qn_group = VGroup(qn_set_text, qn_num_text)
        qn_group.set_color(NONZERO_COLOR)

        alt_nonzero_path_chars = list(filter(is_nonzero, limited_path_chars))
        random.shuffle(alt_nonzero_path_chars)
        nonzero_path_chars = alt_nonzero_path_chars[-2]
        nonzero_path = RandomWalkPath(nonzero_path_chars)
        nonzero_path.put_start_at(new_origin)
        nonzero_drunk = drunk.copy()
        nonzero_drunk.step_on(pos_axis.n2p(path_final_coord(nonzero_path_chars)))
        if nonzero_path_chars[-1] != balanced_path_chars[-1]:
            nonzero_drunk.turn_around()
        nonzero_path.save_state()
        VGroup(nonzero_path, nonzero_drunk).set_color(NONZERO_COLOR)
        nonzero_drunk.save_state()
        self.play(
            Transform(balanced_path, nonzero_path),
            Transform(balanced_drunk, nonzero_drunk),
            FadeOut(VGroup(pn_set_text, pn_num_text)),
            Write(qn_set_text[-9:-3], stroke_color=NONZERO_COLOR),
        )
        self.remove(balanced_path, balanced_drunk)
        self.add(nonzero_path, nonzero_drunk)
        self.wait()

        range_rect = nonzero_path.get_range_rectangle()
        range_rect.set_color(NONZERO_COLOR)
        limit_line = Line(ORIGIN, RIGHT, stroke_width=10)
        limit_line.set_color(LIMIT_COLOR)
        limit_line.match_width(range_rect)
        limit_line.next_to(range_rect, UP, buff=0)
        range_rect.generate_target()
        range_rect.stretch(0, dim=1, about_edge=UP)
        self.play(MoveToTarget(range_rect, run_time=2))
        self.wait()
        self.play(ShowCreation(limit_line))
        self.wait()
        self.play(
            FocusOn(nonzero_path.get_start(), opacity=0.2, color=NONZERO_COLOR),
            Flash(nonzero_path.get_start(), color=NONZERO_COLOR),
        )
        self.wait()

        # Definition of set 'Qn' and its size 'qn'
        self.play(
            Write(
                VGroup(qn_set_text[:-9], qn_set_text[-3:]),
                stroke_color=NONZERO_COLOR,
            ),
            FadeOut(VGroup(range_rect, limit_line)),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeTransform(qn_set_text[:2].copy(), qn_num_text[-3:-1]),
            Write(
                VGroup(qn_num_text[:-3], qn_num_text[-1]),
                stroke_color=NONZERO_COLOR,
            ),
        )
        self.wait()
        self.add(qn_set_text, qn_num_text)
        self.play(
            Indicate(qn_set_text[1], color=BLUE),
            Indicate(qn_set_text[7:9], color=BLUE),
            run_time=2,
        )
        self.wait()

        # More examples of nonzero paths
        num_nonzero_samples = 3
        for alt_path_chars in alt_nonzero_path_chars[:num_nonzero_samples]:
            alt_path = RandomWalkPath(alt_path_chars)
            alt_path.put_start_at(new_origin)
            alt_coord = path_final_coord(alt_path_chars)
            alt_drunk = nonzero_drunk.saved_state.copy()
            alt_drunk.step_on(pos_axis.n2p(alt_coord))
            if alt_path_chars[-1] != nonzero_path_chars[-1]:
                alt_drunk.turn_around()
            VGroup(alt_path, alt_drunk).set_color(NONZERO_COLOR)
            self.play(
                Transform(nonzero_path, alt_path),
                Transform(nonzero_drunk, alt_drunk),
            )
            self.wait()

        # All definitions at once
        qn_set_text.generate_target()
        qn_num_text.generate_target()
        diagram_group = VGroup(
            house, new_pos_axis, pos_label, time_axis, time_label,
            nonzero_drunk, nonzero_path,
        )
        all_definitions = VGroup(
            wn_set_text, wn_num_text,
            pn_set_text, pn_num_text,
            qn_set_text.target, qn_num_text.target,
        )
        all_definitions.arrange_in_grid(
            3, 2,
            aligned_edge=LEFT, h_buff=0.8, v_buff=0.5,
        )
        self.play(
            AnimationGroup(
                MoveToTarget(qn_num_text),
                MoveToTarget(qn_set_text),
                *[
                    FadeIn(mob, shift=2 * DOWN)
                    for mob in all_definitions[:4]
                ][::-1],
                lag_ratio=0.1,
                run_time=3,
            ),
            FadeOut(diagram_group, shift=2 * DOWN, run_time=1)
        )
        self.wait()

        # What're their sizes and relations?
        num_group = VGroup(wn_num_text, pn_num_text, qn_num_text)
        set_group = VGroup(wn_set_text, pn_set_text, qn_set_text)
        num_sur_rect = SurroundingRectangle(num_group, buff=0.2)
        set_sur_rect = SurroundingRectangle(set_group, buff=0.2)
        size_text = TexText("大小是多少?")
        relation_text = TexText("有什么联系?")
        size_text.next_to(num_sur_rect, UP)
        relation_text.next_to(set_sur_rect, UP)

        mixed_colors = [ALL_PATH_COLOR, BALANCED_COLOR, NONZERO_COLOR, ALL_PATH_COLOR]
        for rect in (num_sur_rect, set_sur_rect):
            rect.set_stroke(color=mixed_colors, width=3)
        for text in (size_text, relation_text):
            text.scale(0.8)
            text.set_color(GREY_A)
        self.play(ShowCreation(num_sur_rect), run_time=2)
        self.wait()
        self.play(Write(size_text))
        self.wait()
        self.play(ShowCreation(set_sur_rect), run_time=2)
        self.wait()
        self.play(Write(relation_text))
        self.wait()

        # End of this scene, continue to calculating wn and pn
        wn_set_text.generate_target()
        wn_num_text.generate_target()
        fade_group = VGroup(
            pn_set_text, pn_num_text, qn_set_text, qn_num_text,
            num_sur_rect, set_sur_rect, size_text, relation_text,
        )
        wn_group = VGroup(wn_set_text.target, wn_num_text.target)
        wn_group.arrange(DOWN, aligned_edge=LEFT)
        wn_group.to_corner(UL)
        axes = PathAxes(
            pos_range=(-4, 4, 1), time_range=(0, 10, 1),
        )
        axes.scale(0.7)
        axes.to_edge(RIGHT).shift(0.25 * DOWN)
        axes.set_color(AXES_COLOR)
        eg_path_chars = "UUDDDUUDUU"
        eg_path = RandomWalkPath(eg_path_chars, axes=axes)
        extra_group = VGroup(axes, eg_path)

        self.play(
            MoveToTarget(wn_set_text, run_time=2),
            MoveToTarget(wn_num_text, run_time=2),
            FadeOut(fade_group, run_time=1),
            FadeIn(extra_group, shift=4 * LEFT, run_time=2),
        )
        self.wait()


class CalculatingWnPnAndQn(Scene):
    def construct(self):
        # Setup to match the last scene
        wn_set_text = Tex("W_n: \\left\\{ \\text{长度为} 2n \\text{的路径} \\right\\}")
        wn_num_text = Tex("w_n = |W_n|")
        wn_group = VGroup(wn_set_text, wn_num_text)
        wn_group.arrange(DOWN, aligned_edge=LEFT)
        wn_group.to_corner(UL)
        wn_group.set_color(ALL_PATH_COLOR)

        axes = PathAxes(
            pos_range=(-4, 4, 1), time_range=(0, 10, 1),
        )
        axes.scale(0.7)
        axes.to_edge(RIGHT).shift(0.25 * DOWN)
        axes.set_color(AXES_COLOR)
        eg_path_chars = "UUDDDUUDUU"
        eg_path = RandomWalkPath(eg_path_chars, axes=axes)
        self.add(wn_group, axes, eg_path)
        self.wait()

        # The drunk has to choose between 2 things 2n times
        self.play(
            FlashUnder(wn_set_text[4:9], color=ALL_PATH_COLOR),
            run_time=2,
        )
        self.wait()

        def setup_selections(mob):
            mob.scale(0.7)
            mob.to_edge(LEFT)

        selections = UpDownSelections(eg_path)
        setup_selections(selections)
        self.play(
            AnimationGroup(
                *[
                    ReplacementTransform(
                        arrow.copy(), ud_group.get_selected_symbol()
                    )
                    for arrow, ud_group in zip(eg_path, selections)
                ],
                lag_ratio=0.05,
            ),
            AnimationGroup(
                *[
                    FadeIn(ud_group.get_unselected_symbol())
                    for ud_group in selections
                ],
                lag_ratio=0.05,
            ),
            run_time=3,
        )
        self.wait()

        limited_path_chars = all_limited_walk_strings(10, max_thres=4, min_thres=-4)
        num_normal_samples = 4
        for alt_path_chars in random.sample(limited_path_chars, num_normal_samples):
            alt_path = RandomWalkPath(alt_path_chars, axes=axes)
            alt_selections = UpDownSelections(alt_path)
            setup_selections(alt_selections)
            self.play(
                Transform(eg_path, alt_path),
                Transform(selections, alt_selections),
                run_time=1,
            )
            self.wait()

        # So we have an expression for w_n
        wn_formula = Tex("w_n = |W_n| = 2^{2n}")
        wn_formula.set_color(ALL_PATH_COLOR)
        wn_formula.next_to(wn_set_text, DOWN, aligned_edge=LEFT)

        self.play(
            Transform(wn_num_text, wn_formula[:-4]),
            Write(wn_formula[-4], run_time=1),
            FadeTransform(
                selections[0].copy().set_opacity(0),
                wn_formula[-3],
                run_time=1.5
            ),
            AnimationGroup(
                *[
                    FadeTransform(
                        ud_group.copy().set_opacity(0),
                        wn_formula[-2:]
                    )
                    for ud_group in selections[1:]
                ],
                lag_ratio=0.02,
                run_time=2,
            ),
        )
        self.remove(wn_num_text)
        self.add(wn_formula)
        wn_group = VGroup(wn_set_text, wn_formula)
        self.wait()

        # Next up, Pn - the set of balanced path and its size p_n
        pn_set_text = Tex("P_n: \\left\\{ \\text{长度为} 2n \\text{的“正点到家”路径} \\right\\}")
        pn_num_text = Tex("p_n = |P_n|")
        pn_group = VGroup(pn_set_text, pn_num_text)
        pn_group.arrange(DOWN, aligned_edge=LEFT)
        pn_group.to_corner(UL)
        pn_group.set_color(BALANCED_COLOR)
        balanced_path_chars = list(filter(is_balanced, limited_path_chars))
        random.shuffle(balanced_path_chars)
        balanced_path = RandomWalkPath(balanced_path_chars[0], axes=axes)
        balanced_selections = UpDownSelections(balanced_path)
        setup_selections(balanced_selections)
        self.play(
            FadeOut(wn_group, shift=LEFT),
            FadeIn(pn_group, shift=LEFT),
            Transform(eg_path, balanced_path),
            Transform(selections, balanced_selections),
        )
        self.wait()

        # The drunk starts and ends at position 0
        start_dot = GlowDot(balanced_path.get_start(), color=BALANCED_COLOR)
        end_dot = GlowDot(balanced_path.get_end(), color=BALANCED_COLOR)
        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        Flash(dot.get_center(), color=BALANCED_COLOR),
                        FadeIn(dot, scale=0.1),
                    )
                    for dot in (start_dot, end_dot)
                ],
                lag_ratio=0.1,
                run_time=2,
            )
        )
        self.wait()

        up_symbol, down_symbol = balanced_selections[2].copy()
        text_1, text_2 = TexText("和"), TexText("个数相同")
        for mob in (up_symbol, down_symbol):
            mob.generate_target()
            mob.set_opacity(0)
            mob.target.set_opacity(1)
            mob.target.match_height(text_1)
        balanced_cond_text = VGroup(up_symbol.target, text_1, down_symbol.target, text_2)
        balanced_cond_text.arrange(RIGHT)
        balanced_cond_text.next_to(balanced_selections, DOWN, buff=0.3)
        balanced_cond_rect = SurroundingRectangle(
            VGroup(
                ud_group.get_selected_symbol()
                for ud_group in balanced_selections
            ),
            color=BALANCED_COLOR,
        )
        self.play(ShowCreation(balanced_cond_rect))
        self.wait()
        self.play(
            MoveToTarget(up_symbol),
            MoveToTarget(down_symbol),
            Write(VGroup(balanced_cond_text[1::2])),
        )
        balanced_cond_text = VGroup(up_symbol, text_1, down_symbol, text_2)
        self.wait()

        num_balanced_samples = 4
        for alt_path_chars in random.sample(balanced_path_chars[1:], num_balanced_samples):
            alt_path = RandomWalkPath(alt_path_chars, axes=axes)
            alt_selections = UpDownSelections(alt_path)
            setup_selections(alt_selections)
            self.play(
                Transform(eg_path, alt_path),
                Transform(selections, alt_selections),
                run_time=1,
            )
            self.wait()
        last_balanced_selections = alt_selections

        # And we have a formula for p_n as well
        pn_formula = Tex("p_n = |P_n| = {2n \\choose n}")
        pn_fraction = Tex("\\frac{(2n)!}{(n!)^2}")
        pn_formula.set_color(BALANCED_COLOR)
        pn_fraction.set_color(BALANCED_COLOR)
        pn_formula.next_to(pn_set_text, DOWN, aligned_edge=LEFT)
        pn_fraction.move_to(pn_formula[8:])

        self.play(
            Transform(pn_num_text, pn_formula[:7]),
            Write(pn_formula[7], run_time=1),
            AnimationGroup(
                *[
                    FadeTransform(
                        ud_group.get_selected_symbol().copy(),
                        pn_formula[8:],
                    )
                    for ud_group in last_balanced_selections
                ],
                lag_ratio=0.02,
                run_time=2,
            ),
        )
        self.wait()
        self.play(FadeTransform(pn_formula[8:], pn_fraction), run_time=1)
        self.wait()
        pn_alt_formula = VGroup(pn_formula[:8], pn_fraction)
        self.remove(pn_num_text)
        self.add(pn_alt_formula)
        self.wait()

        # Branching point 1 - we need to first simplify the expression of p_n
        pn_alt_formula.save_state()
        final_pn_formula = Tex("p_n = |P_n| = \\frac{(2n)!}{(n!)^2}")
        final_pn_formula.to_edge(LEFT)
        final_pn_formula.to_edge(UP, buff=1)
        temp_fade_group = Group(
            pn_set_text, axes, eg_path, start_dot, end_dot,
            selections, balanced_cond_text, balanced_cond_rect,
        )
        self.play(
            FadeOut(temp_fade_group),
            Transform(pn_alt_formula[0], final_pn_formula[:8]),
            Transform(pn_alt_formula[1], final_pn_formula[8:]),
        )
        self.wait()
        # End of branching point 1

        # Back to the last set Qn
        self.remove(final_pn_formula)
        pn_alt_formula.restore()
        self.add(temp_fade_group, pn_alt_formula)
        pn_group = VGroup(pn_set_text, pn_alt_formula)
        self.wait()

        qn_set_text = Tex("Q_n: \\left\\{ \\text{长度为} 2n \\text{的“不再回家”路径} \\right\\}")
        qn_num_text = Tex("q_n = |Q_n|")
        qn_group = VGroup(qn_set_text, qn_num_text)
        qn_group.arrange(DOWN, aligned_edge=LEFT)
        qn_group.to_corner(UL)
        qn_group.set_color(NONZERO_COLOR)
        nonzero_path_chars = list(filter(is_nonzero, limited_path_chars))
        random.shuffle(nonzero_path_chars)
        nonzero_path = RandomWalkPath(nonzero_path_chars[0], axes=axes)
        nonzero_selections = UpDownSelections(nonzero_path)
        setup_selections(nonzero_selections)
        nonzero_cond_rect = balanced_cond_rect.copy().set_color(NONZERO_COLOR)
        nonzero_cond_text = Tex("?" * len(balanced_cond_text))
        nonzero_cond_text.move_to(balanced_cond_text)

        self.play(
            FadeOut(start_dot),
            FadeOut(end_dot),
            FadeOut(pn_group, shift=LEFT),
            FadeIn(qn_group, shift=LEFT),
            Transform(eg_path, nonzero_path),
            Transform(selections, nonzero_selections),
            ReplacementTransform(balanced_cond_rect, nonzero_cond_rect),
            ReplacementTransform(balanced_cond_text, nonzero_cond_text),
        )
        self.wait()

        # Qn seems to be more restricted
        crosses = VGroup(*[
            Cross(
                Circle(radius=0.15).move_to(axes.c2p(time=time, pos=0)),
                stroke_color=NONZERO_COLOR
            )
            for time in range(1, len(eg_path) + 1)
        ])
        random.shuffle(crosses.submobjects)
        self.play(
            AnimationGroup(
                *[
                    FadeIn(cross, scale=1 / 3.)
                    for cross in crosses
                ],
                lag_ratio=0.05,
            )
        )
        self.add(crosses)
        self.wait()

        # Setup for the counting scene
        fade_group = VGroup(
            eg_path, selections, nonzero_cond_text, nonzero_cond_rect,
            axes, crosses,
        )
        set_group = VGroup(qn_group, pn_group, wn_group)
        for group in set_group:
            group.generate_target()
        set_group[1:].set_opacity(0)
        set_group_target = VGroup(group.target for group in set_group)
        set_group_target.arrange(UP, buff=0.7, aligned_edge=LEFT)
        set_group_target.to_edge(UP)
        self.play(
            FadeOut(fade_group, run_time=1),
            AnimationGroup(
                *[MoveToTarget(group) for group in set_group],
                lag_ratio=0.1,
                run_time=2,
            )
        )
        self.wait()
        self.play(
            FlashAround(wn_group[-1], color=ALL_PATH_COLOR),
            FlashAround(pn_group[-1], color=BALANCED_COLOR),
        )
        self.wait()
        qn_question = Tex("q_n = |Q_n| = \\; ?")
        qn_question.set_color(NONZERO_COLOR)
        qn_question.next_to(qn_group[-1].get_left(), RIGHT, buff=0)
        self.play(
            Transform(qn_group[-1], qn_question[:-2]),
            Write(qn_question[-2:], stroke_color=NONZERO_COLOR),
        )
        self.wait()

        # A summary of what we have now
        qn_hypothesis = Tex("q_n = |Q_n| \\overset{?}{=} p_n")
        qn_hypothesis.set_color(NONZERO_COLOR)
        qn_hypothesis[-2:].set_color(BALANCED_COLOR)
        qn_hypothesis.next_to(qn_question.get_left(), RIGHT, buff=0)
        qn_group = VGroup(qn_set_text, qn_hypothesis)
        set_group = VGroup(qn_group, pn_group, wn_group)
        an_definition = Tex("a_n = \\frac{p_n}{w_n}")
        an_definition[3:5].set_color(BALANCED_COLOR)
        an_definition[-2:].set_color(ALL_PATH_COLOR)
        an_text = TexText("与沃利斯乘积有关")
        an_text.scale(0.75).next_to(an_definition, DOWN)
        an_group = VGroup(an_definition, an_text)
        an_group.next_to(set_group[1:], RIGHT, buff=1)
        VGroup(set_group, an_group).next_to(TOP, DOWN, coor_mask=[1, 0, 0])

        self.remove(qn_num_text, qn_question)
        self.add(an_group, qn_hypothesis)
        self.wait()
        self.play(
            FlashAround(wn_group[-1], color=ALL_PATH_COLOR),
            FlashAround(pn_group[-1], color=BALANCED_COLOR),
        )
        self.wait()
        self.play(
            FlashAround(an_group, color=WALLIS_COLOR),
        )
        self.wait()
        self.play(
            FlashAround(qn_hypothesis, color=NONZERO_COLOR),
        )
        self.wait()

        # Why introducing the random walks? Seems redundant
        alt_an_formula = Tex(
            """
            &a_{n} = \\frac{1}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{6} \\cdots \\frac{2n-1}{2n}  \\\\
            &(a_{0}=1)
            """
        )
        alt_an_formula.to_edge(RIGHT, buff=1)
        axes = PathAxes(
            N=6, pos_range=(-4, 4, 1),
            unit_size=0.7, include_label=False,
        )
        axes.to_edge(LEFT, buff=1)
        path = RandomWalkPath("UUDUUD", axes=axes)
        wn_circle = SetCircle("W_n", radius=0.7, color=ALL_PATH_COLOR)
        pn_circle = SetCircle("P_n", radius=0.4, color=BALANCED_COLOR)
        qn_circle = SetCircle("Q_n", radius=0.4, color=NONZERO_COLOR)
        wn_circle.move_to(axes.c2p(2, -2.5))
        pn_circle.move_to(axes.c2p(5, -1.5))
        qn_circle.move_to(axes.c2p(5, -3.5))
        circles = VGroup(wn_circle, pn_circle, qn_circle)

        path_group = VGroup(axes, path, circles)
        arrow = Arrow(
            path_group.get_right(), alt_an_formula.get_left(),
            buff=0.8,
        )
        arrow_question_mark = Tex("?").next_to(arrow, UP)
        arrow_group = VGroup(arrow, arrow_question_mark)
        all_mobs = VGroup(path_group, arrow_group, alt_an_formula)
        all_mobs.to_edge(UP, buff=0.8)

        self.play(
            FadeOut(VGroup(set_group, an_text)),
            FadeTransform(an_definition, alt_an_formula),
        )
        self.wait()
        self.play(
            FadeIn(path_group, shift=RIGHT),
            GrowFromCenter(arrow_group),
        )
        self.wait()

        arcs = VGroup(*[
            sc1.get_arc_to(sc2).set_z_index(-1)
            for (sc1, sc2) in it.permutations(circles, 2)
        ])
        an_property = TexText("数列$\\{{a_n\\}}$的性质")
        an_property.next_to(alt_an_formula, DOWN, buff=0.5)
        an_property.next_to(circles, RIGHT, coor_mask=[0, 1, 0])
        self.play(
            *[DrawBorderThenFill(arc) for arc in arcs],
            *[Animation(circle) for circle in circles],
            lag_ratio=0.1,
        )
        self.wait()
        self.play(
            FadeTransform(arcs.copy().set_stroke(opacity=0), an_property)
        )
        self.wait()


class IntroducingQuantityAn(Scene):
    def construct(self):
        # Setup to match the branching point 1 - the form of p_n
        pn_formula = Tex("p_n = |P_n| = \\frac{(2n)!}{(n!)^2}")
        pn_formula.to_edge(LEFT)
        pn_formula.to_edge(UP, buff=1)
        self.add(pn_formula)

        # Expand p_n into the product of 2n rationals
        pn_expansion_1 = Tex(
            """
            = \\frac{1 \\cdot 2 \\cdot 3 \\cdot 4 \\cdot 5 \\cdot 6 \\cdots \\ (2n-1) \\cdot 2n}
            {1 \\cdot 1 \\cdot 2 \\cdot 2 \\cdot 3 \\cdot 3 \\cdots \\ n \\cdot n}
            """
        )
        pn_expansion_1.next_to(pn_formula, RIGHT)
        self.play(
            Write(pn_expansion_1[:24], run_time=2),
            GrowFromCenter(pn_expansion_1[24], run_time=1),
            Write(pn_expansion_1[25:], run_time=2),
        )
        self.wait()

        pn_expansion_2 = Tex(
            """
            = \\frac{1}{1} \\cdot \\frac{2}{1} \\cdot \\frac{3}{2} \\cdot \\frac{4}{2} \\cdot \\frac{5}{3}
             \\cdot \\frac{6}{3} \\cdots \\frac{2n-1}{n} \\cdot \\frac{2n}{n}
            """,
        )
        pn_expansion_2.next_to(pn_expansion_1, DOWN, aligned_edge=LEFT, buff=0.4)
        pe1, pe2 = pn_expansion_1, pn_expansion_2
        self.play(
            TransformFromCopy(pe1[:11:2], pe2[:21:4]),
            TransformFromCopy(pe1[1:12:2], pe2[1:22:4]),
            TransformFromCopy(pe1[12:15], pe2[24:27]),
            TransformFromCopy(pe1[16:20], pe2[27:31]),
            TransformFromCopy(pe1[21], pe2[33]),
            TransformFromCopy(pe1[22:24], pe2[34:36]),
            ReplacementTransform(pe1[24].replicate(6), pe2[2:23:4]),
            ReplacementTransform(pe1[24].replicate(2), pe2[31:37:5]),
            TransformFromCopy(pe1[25:36:2], pe2[3:24:4]),
            TransformFromCopy(pe1[26:35:2], pe2[4:21:4]),
            TransformFromCopy(pe1[36:39], pe2[24:27]),
            TransformFromCopy(pe1[39:41], pe2[32:34]),
            TransformFromCopy(pe1[41], pe2[37]),
            run_time=3,
        )
        self.add(pe2)
        self.wait()

        wallis_prod = Tex(
            """
            \\frac{2}{1} \\cdot \\frac{2}{3} \\cdot \\frac{4}{3} \\cdot \\frac{4}{5} \\cdot
            \\frac{6}{5} \\cdot \\frac{6}{7} \\cdots  \\frac{2n}{2n-1} \\cdot \\frac{2n}{2n+1} \\cdots
            """
        )
        wallis_prod.next_to(pe2[1:], DOWN, aligned_edge=LEFT, buff=1)
        wallis_title = TexText("沃利斯乘积：")
        wallis_title.next_to(wallis_prod, LEFT)
        wallis_group = VGroup(wallis_title, wallis_prod)
        wallis_group.set_color(WALLIS_COLOR)
        self.play(FadeIn(wallis_group, shift=DOWN))
        self.wait()
        self.play(
            FadeOut(pe1, shift=UP),
            pe2.animate.next_to(pn_formula, RIGHT),
            FadeOut(wallis_group, shift=UP),
        )
        self.wait()

        # Introducing a_n = p_n / w_n
        an_formula = Tex(
            "a_n = \\frac{p_n}{w_n} = \\frac{\\text{“正点到家”型路径数}}{\\text{路径总数}}"
        )
        an_formula.next_to(pn_formula, DOWN, aligned_edge=LEFT, buff=0.7)
        an_formula[3:5].set_color(BALANCED_COLOR)
        an_formula[6:8].set_color(ALL_PATH_COLOR)
        numer, line, denom = an_formula[9:-5], an_formula[-5], an_formula[-4:]
        numer.set_color(BALANCED_COLOR)
        denom.set_color(ALL_PATH_COLOR)
        self.play(Write(an_formula[:8]))
        self.wait()
        self.play(FadeIn(an_formula[8:], shift=0.5 * RIGHT))
        self.wait()
        self.add(an_formula)

        an_expansion_1 = Tex(
            """
            = \\frac{1}{1} \\cdot \\frac{2}{1} \\cdot \\frac{3}{2} \\cdot \\frac{4}{2} \\cdot \\frac{5}{3}
             \\cdot \\frac{6}{3} \\cdots \\frac{2n-1}{n} \\cdot \\frac{2n}{n} \\cdot \\frac{1}{2^{2n}}
            """,
        )
        an_expansion_1[1:-6].set_color(BALANCED_COLOR)
        an_expansion_1[-3:].set_color(ALL_PATH_COLOR)
        shift_vec_1 = an_formula[8].get_center() - an_expansion_1[0].get_center()
        an_expansion_1.shift(shift_vec_1)
        self.play(
            FadeTransform(numer, an_expansion_1[1:-6]),
            FadeOut(line, run_time=1),
            FadeTransform(denom, an_expansion_1[-6:]),
            run_time=2,
        )
        self.remove(an_formula[8:])
        self.add(an_expansion_1)
        self.wait()
        self.play(
            an_formula[:8].animate.set_color(WHITE),
            an_expansion_1.animate.set_color(WHITE),
        )
        self.wait()
        self.play(Indicate(an_expansion_1[1:-6], color=BALANCED_COLOR, scale_factor=1.05))
        self.wait()
        self.play(Indicate(an_expansion_1[-3:], color=ALL_PATH_COLOR, scale_factor=1.05))
        self.wait()

        an_expansion_2 = Tex(
            """
            = \\frac{1}{2} \\cdot \\frac{2}{2} \\cdot \\frac{3}{4} \\cdot \\frac{4}{4} \\cdot \\frac{5}{6}
             \\cdot \\frac{6}{6} \\cdots \\frac{2n-1}{2n} \\cdot \\frac{2n}{2n}
            """,
        )
        shift_vec_2 = an_expansion_1[0].get_center() - an_expansion_2[0].get_center()
        an_expansion_2.shift(shift_vec_2)

        ae1, ae2 = an_expansion_1, an_expansion_2
        all_denoms = VGroup(
            ae2[3], ae2[7], ae2[11], ae2[15], ae2[19], ae2[23],
            ae2[-8:-6], ae2[-2:],
        )
        self.play(
            FadeTransform(ae1[:-6], ae2, run_time=2),
            FadeOut(ae1[-6:-3], run_time=1),
            AnimationGroup(
                *[
                    FadeTransform(ae1[-3:], denom)
                    for denom in all_denoms[::-1]
                ],
                lag_ratio=0.02,
                run_time=1.5,
            ),
        )
        self.wait()

        # Here comes the Wallis product ...'s odd half
        even_terms = VGroup(ae2[4:8], ae2[12:16], ae2[20:24], ae2[-6:])
        self.play(even_terms.animate.set_opacity(0.2))
        self.wait()

        shift_vec_wallis = ae2[2].get_left() - wallis_group[1][1].get_left()
        wallis_group.shift(shift_vec_wallis)
        wallis_group.next_to(
            an_formula, DOWN, aligned_edge=LEFT, coor_mask=[0, 1, 0], buff=0.4,
        )
        an_odd_terms = VGroup(ae2[1:4], ae2[9:12], ae2[17:20], ae2[-13:-6])
        wp = wallis_prod
        wallis_odd_terms = VGroup(wp[:3], wp[8:11], wp[16:19], wp[-18:-11])

        self.play(FadeIn(wallis_group, shift=DOWN))
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    FlashAround(VGroup(an_term, wallis_term), color=WALLIS_COLOR)
                    for an_term, wallis_term in zip(an_odd_terms, wallis_odd_terms)
                ],
                lag_ratio=0.2,
            ),
            run_time=2,
        )
        self.wait()

        # A subtle recurrence property of a_n
        an_formula_simp = Tex(
            """
            a_{n} = \\frac{1}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{6} \\cdots \\frac{2n-1}{2n}
            """,
        )
        anp1_formula_simp = Tex(
            """
            a_{n+1} = \\frac{1}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{6} \\cdots \\frac{2n-1}{2n} \\cdot \\frac{2n+1}{2n+2}
            """,
        )
        shift_vec_simp = an_formula_simp[2].get_center() - anp1_formula_simp[4].get_center()
        anp1_formula_simp.shift(shift_vec_simp)
        simp_group = VGroup(an_formula_simp, anp1_formula_simp)
        simp_group.arrange(DOWN, coor_mask=[0, 1, 0], buff=0.4)
        simp_group.next_to(an_formula, DOWN, aligned_edge=LEFT, buff=0.4)
        simp_color = GOLD_B
        simp_group.set_color(simp_color)
        self.play(
            FadeOut(wallis_group, run_time=1),
            TransformFromCopy(an_formula[:2], an_formula_simp[:2], run_time=2),
            TransformFromCopy(ae2[:4], an_formula_simp[2:6], run_time=2),
            TransformFromCopy(ae2[8:12], an_formula_simp[6:10], run_time=2),
            TransformFromCopy(ae2[16:20], an_formula_simp[10:14], run_time=2),
            TransformFromCopy(ae2[-16:-6], an_formula_simp[14:], run_time=2),
        )
        self.wait()
        self.play(FadeTransform(an_formula_simp.copy(), anp1_formula_simp))
        self.add(simp_group)
        self.wait()
        self.play(
            FlashAround(anp1_formula_simp[-10:], color=simp_color)
        )
        self.wait()

        recur_brace = Brace(simp_group, direction=RIGHT)
        recur_tex = Tex("a_{n+1} = a_{n} \\cdot \\frac{2n+1}{2n+2}")
        recur_rect = SurroundingRectangle(
            recur_tex, stroke_width=1, color=simp_color, buff=0.3,
        )
        recur_formula = VGroup(recur_tex, recur_rect)
        recur_brace.put_at_tip(recur_formula)
        recur_group = VGroup(recur_brace, recur_formula)
        recur_group.set_color(simp_color)
        self.play(FadeIn(recur_group, shift=0.5 * RIGHT))
        self.wait()


class ShowAllPathsWithLength4(Scene):
    def construct(self):
        n_label = TexText(f"$n=2$ （醉汉行走$4$步）")
        n_label.scale(0.8)
        n_label_rect = SurroundingRectangle(n_label, buff=0.2)
        n_label_group = VGroup(n_label, n_label_rect)
        n_label_group.set_color(WHITE).to_corner(UL)
        self.add(n_label_group)

        all_axes = VGroup(
            PathAxes(N=4, include_label=False).set_color(AXES_COLOR)
            for k in range(16)
        )
        all_axes.arrange_in_grid(2, 8, buff=2)
        all_axes.set_width(FRAME_WIDTH - 1)
        paths = VGroup(
            RandomWalkPath(ws, axes=axes)
            for ws, axes in zip(all_random_walk_strings(4), all_axes)
        )
        bg_rects = VGroup(
            BackgroundRectangle(axes, color=GREY, fill_opacity=0.2, buff=0.15)
            for axes in all_axes
        )

        groups = VGroup(*[
            VGroup(rect, axes, path)
            for rect, axes, path in zip(bg_rects, all_axes, paths)
        ])
        groups.center()
        groups.next_to(n_label_group, DOWN, buff=0.5, coor_mask=[0, 1, 0])

        wn_title = TexText("路径总数").set_color(ALL_PATH_COLOR)
        pn_title = TexText("“正点到家”型路径数").set_color(BALANCED_COLOR)
        qn_title = TexText("“不再回家”型路径数").set_color(NONZERO_COLOR)
        title_group = VGroup(wn_title, pn_title, qn_title)
        title_group.arrange(RIGHT, buff=1)
        title_group.set_width(8)
        title_group.to_corner(UR)

        self.add(groups, title_group)
        self.bring_to_front(paths)
        self.wait()

        groups.generate_target()
        for group in groups.target:
            rect, axes, path = group
            if path.is_balanced():
                rect.set_color(BALANCED_COLOR)
                path.set_color(BALANCED_COLOR)
            elif path.is_nonzero():
                rect.set_color(NONZERO_COLOR)
                path.set_color(NONZERO_COLOR)
            else:
                path.set_color(AXES_COLOR)

        self.play(MoveToTarget(groups))
        self.wait()

        # Verify formula for w_2 and p_2, and show how big is q_2
        wn_formula = Tex("w_2 = 2^4 = 16")
        pn_formula = TexText("$p_2 = {4 \\choose 2} = 6$")
        qn_formula = Tex("q_2 = 6")
        formula_group = VGroup(wn_formula, pn_formula, qn_formula)
        for formula, title in zip(formula_group, title_group):
            formula.scale(0.6)
            formula.next_to(title, DOWN)
            formula.match_color(title)
        self.play(
            Write(wn_formula, stroke_color=ALL_PATH_COLOR),
            FlashAround(groups, color=ALL_PATH_COLOR),
            run_time=2,
        )
        self.wait()
        self.play(
            Write(pn_formula, stroke_color=BALANCED_COLOR),
            *[
                FlashAround(rect, color=BALANCED_COLOR)
                for rect, _, path in groups
                if path.is_balanced()
            ],
            run_time=2,
        )
        self.wait()
        self.play(
            Write(qn_formula, stroke_color=NONZERO_COLOR),
            *[
                FlashAround(rect, color=NONZERO_COLOR)
                for rect, _, path in groups
                if path.is_nonzero()
            ],
            run_time=2,
        )
        self.wait()


class CountingPathsScene(Scene):
    N = 8
    counting_time = 30
    edge_buff = 1
    show_pq_equal_animation = False
    show_wp_calc_animation = False
    show_shift_to_set_animation = False
    fade_out_everything_at_last = False

    def construct(self):
        N = self.N
        subscript = N // 2
        all_ws = list(all_random_walk_strings(N))
        balanced_list = list(map(is_balanced, all_ws))
        nonzero_list = list(map(is_nonzero, all_ws))
        path = RandomWalkPath("U" * N).scale(0)
        vt = ValueTracker(1)

        n_label = TexText(f"$n={subscript}$ （醉汉行走${2*subscript}$步）")
        n_label.scale(0.8)
        n_label_rect = SurroundingRectangle(n_label, buff=0.2)
        n_label_group = VGroup(n_label, n_label_rect)
        n_label_group.set_color(WHITE).to_corner(UL)
        self.add(n_label_group)

        axes = PathAxes(N=N)
        axes.set_height(min(axes.get_height(), FRAME_HEIGHT - 1))
        axes.to_edge(RIGHT, buff=self.edge_buff)
        axes.set_color(AXES_COLOR)

        all_path_text = TexText("路径总数", "$w_{%d}=$" % subscript)
        all_path_text.set_color(ALL_PATH_COLOR)
        balanced_text = TexText("“正点到家”型路径数", "$p_{%d}=$" % subscript)
        balanced_text.set_color(BALANCED_COLOR)
        nonzero_text = TexText("“不再回家”型路径数", "$q_{%d}=$" % subscript)
        nonzero_text.set_color(NONZERO_COLOR)
        texts = VGroup(all_path_text, balanced_text, nonzero_text)
        texts.arrange(DOWN, aligned_edge=RIGHT)
        texts.to_edge(LEFT, buff=self.edge_buff)
        if self.show_wp_calc_animation:
            texts.next_to(ORIGIN, DOWN, buff=0, coor_mask=[0, 1, 0])
        tail_len = 2 + len(str(subscript))
        names = VGroup(*[t[:- tail_len] for t in texts])
        names.arrange(DOWN, aligned_edge=RIGHT, coor_mask=[1, 0, 0], center=False)

        counter_config = {
            "group_with_commas": False,
            "text_config": {"font": "CMU Serif"},
        }
        balanced_counter = Integer(0, **counter_config)
        all_path_counter = Integer(0, **counter_config)
        nonzero_counter = Integer(0, **counter_config)

        def get_current_index():
            return int(vt.get_value())

        def update_balanced_counter(counter):
            counter.set_color(BALANCED_COLOR)
            counter.next_to(balanced_text, RIGHT)
            if vt.get_value() >= 1:
                counter.set_value(sum(balanced_list[:get_current_index()]))
            else:
                counter.set_value(0)

        def update_all_path_counter(counter):
            counter.set_color(ALL_PATH_COLOR)
            counter.next_to(all_path_text, RIGHT)
            if vt.get_value() >= 1:
                counter.set_value(get_current_index())
            else:
                counter.set_value(0)

        def update_nonzero_counter(counter):
            counter.set_color(NONZERO_COLOR)
            counter.next_to(nonzero_text, RIGHT)
            if vt.get_value() >= 1:
                counter.set_value(sum(nonzero_list[:get_current_index()]))
            else:
                counter.set_value(0)

        def update_path(path):
            if vt.get_value() == 0:
                pass
            else:
                path_chars = all_ws[int(vt.get_value() - 1)]
                next_path = RandomWalkPath(path_chars, unit_size=axes.get_unit_size())
                next_path.put_start_at(axes.get_origin())
                if is_balanced(path_chars):
                    next_path.set_color(BALANCED_COLOR)
                elif is_nonzero(path_chars):
                    next_path.set_color(NONZERO_COLOR)
                else:
                    next_path.set_color(AXES_COLOR)
                path.become(next_path)

        balanced_counter.add_updater(update_balanced_counter)
        all_path_counter.add_updater(update_all_path_counter)
        nonzero_counter.add_updater(update_nonzero_counter)
        path.add_updater(update_path)

        self.add(texts)
        self.add(balanced_counter, all_path_counter, nonzero_counter)
        self.add(axes)
        self.add(path)
        self.wait()

        vt_moving_anim = ApplyMethod(vt.set_value, 2**N, run_time=self.counting_time)
        if self.show_wp_calc_animation:
            w_n = pow(2, 2 * subscript)
            p_n = central_binomial_coeff(subscript)
            w_formula = Tex(f"w_{{{subscript}}} = 2^{{{2*subscript}}} = {w_n}")
            w_formula.match_color(all_path_counter)
            p_formula = Tex(f"p_{{{subscript}}} = {{ {2*subscript} \\choose {subscript} }} = {p_n}")
            p_formula.match_color(balanced_counter)
            formula_group = VGroup(w_formula, p_formula)
            formula_group.scale(0.8)
            formula_group.arrange(RIGHT, buff=0.8)
            formula_group.next_to(texts, UP, aligned_edge=LEFT, buff=0.8)

            vt_moving_anim_updater = turn_animation_into_updater(vt_moving_anim)
            self.add(vt_moving_anim_updater)
            self.play(Write(w_formula), run_time=1)
            self.wait()
            self.play(Write(p_formula), run_time=1)
            self.wait()
            self.wait(self.counting_time - 4)
        else:
            self.play(vt_moving_anim)
        self.wait()

        # The counting matches the prediction
        if self.show_wp_calc_animation:
            for counter in (balanced_counter, all_path_counter, nonzero_counter):
                counter.suspend_updating()
            temp_wn = w_formula[-len(str(w_n)):].copy()
            temp_wn.generate_target()
            for mob_src, mob_dst in zip(temp_wn.target, all_path_counter):
                mob_src.match_height(mob_dst).move_to(mob_dst)
            temp_pn = p_formula[-len(str(p_n)):].copy()
            temp_pn.generate_target()
            for mob_src, mob_dst in zip(temp_pn.target, balanced_counter):
                mob_src.match_height(mob_dst).move_to(mob_dst)
            self.play(
                MoveToTarget(temp_wn),
                MoveToTarget(temp_pn),
                run_time=2,
            )
            self.remove(temp_wn, temp_pn)
            self.wait()

        # Follow up animation: Is q_n equal to p_n?
        if self.show_pq_equal_animation:
            pq_colors = [
                balanced_counter.get_color(),
                nonzero_counter.get_color(),
                balanced_counter.get_color(),
            ]
            avg_color = average_color(*pq_colors[:-1])
            counter_texts = VGroup(
                balanced_counter, nonzero_counter,
                VGroup(*[t[-tail_len:] for t in texts[1:]]),
            )
            set_texts = VGroup(*[t[:-tail_len - 1] for t in texts[1:]])
            sur_rect = SurroundingRectangle(counter_texts)
            sur_rect.set_color(pq_colors)
            equal_text = TexText("相等 ?")
            equal_text.next_to(sur_rect, DOWN)
            equal_text.set_color(avg_color)
            self.play(ShowCreation(sur_rect))
            self.wait()
            self.play(Write(equal_text))
            self.wait()
            if self.show_shift_to_set_animation:
                sur_rect.generate_target()
                sur_rect.target.surround(set_texts)
                relation_text = TexText("有什么联系 ?")
                relation_text.next_to(sur_rect.target, DOWN)
                relation_text.set_color(avg_color)
                self.play(
                    MoveToTarget(sur_rect),
                    FadeTransform(equal_text, relation_text),
                )
                self.wait()

        if self.fade_out_everything_at_last:
            path.clear_updaters()
            self.play(FadeOut(Group(*self.mobjects)))
            self.wait()


class CountingPathsWithLength4(CountingPathsScene):
    N = 4
    counting_time = 5


class CountingPathsWithLength10(CountingPathsScene):
    N = 10
    counting_time = 30
    show_pq_equal_animation = True
    show_shift_to_set_animation = True
    fade_out_everything_at_last = True


class CountingPathsWithLength12(CountingPathsScene):
    N = 12
    counting_time = 40
    show_wp_calc_animation = True
    show_pq_equal_animation = True
    fade_out_everything_at_last = True


class CountingPathsWithLength16(CountingPathsScene):
    N = 16
    counting_time = 15
    show_pq_equal_animation = True
    show_shift_to_set_animation = True
    fade_out_everything_at_last = True


class SpecialCaseForWPQA(Scene):
    def construct(self):
        # n=0 is a "special" case for w_n, p_n, q_n and a_n
        case_colors = [ALL_PATH_COLOR, BALANCED_COLOR, WHITE, NONZERO_COLOR]
        general_title = TexText("一般情况")
        general_cases = VGroup(
            Tex("w_n = 2^{2n}"),
            Tex("p_n = {2n \\choose n}"),
            Tex("a_n = \\frac{p_n}{w_n}"),
            Tex("q_n \\overset{?}{=} p_n"),
        )
        general_title.to_edge(UP).shift(3 * LEFT)
        general_cases.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        general_cases.next_to(general_title, DOWN, buff=0.8)
        for mob, color in zip(general_cases, case_colors):
            mob.set_color(color)
        general_cases[2][3:5].set_color(BALANCED_COLOR)
        general_cases[2][-2:].set_color(ALL_PATH_COLOR)
        general_cases[3][-2:].set_color(BALANCED_COLOR)
        self.add(general_title, general_cases)
        self.wait()

        special_title = TexText("边界情况$(n=0)$")
        special_cases = VGroup(
            Tex("w_0 = 2^{0} = 1 "),
            Tex("p_0 = {0 \\choose 0} = 1"),
            Tex("a_0 = \\frac{p_0}{w_0} = 1"),
            Tex("q_0 = p_0 = 1"),
        )
        special_title.to_edge(UP).shift(3 * RIGHT)
        special_cases.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        special_cases.next_to(special_title, DOWN, buff=0.8)
        for mob, color in zip(special_cases, case_colors):
            mob.set_color(color)
        special_cases[2][3:5].set_color(BALANCED_COLOR)
        special_cases[2][6:8].set_color(ALL_PATH_COLOR)
        special_cases[3][3:5].set_color(BALANCED_COLOR)
        self.play(FadeTransform(general_title.copy(), special_title))
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    FadeTransform(source.copy(), VGroup(target[:2], target[-2:]))
                    for source, target in zip(general_cases, special_cases)
                ],
                lag_ratio=0.1,
            )
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    Write(tex[2:-2], stroke_color=color)
                    for tex, color in zip(special_cases, case_colors)
                ],
                lag_ratio=0.1,
            )
        )
        self.wait()

        # Intuitively,
        general_group = VGroup(general_title, general_cases)
        axes = PathAxes(N=2, include_label=False)
        axes.set_z_index(-2)
        house = House().rotate(PI / 2)
        house.set_z_index(-1)
        drunk = Drunk(color=ALL_PATH_COLOR).rotate(PI / 2)
        house.put_on(axes.c2p(0, 0))
        drunk.step_on(axes.c2p(0, 0))
        diagram = VGroup(axes, house, drunk)
        diagram.move_to(general_cases)
        self.play(
            FadeOut(general_group),
            *[FadeIn(mob) for mob in diagram],
        )
        self.wait()
        glow_dot = GlowDot(axes.c2p(0, 0), glow_factor=1.5)
        empty_path_text = TexText("唯一空路径").scale(0.8)
        empty_path_text.next_to(axes.c2p(0, 0), DR)
        Group(glow_dot, empty_path_text).set_color(ALL_PATH_COLOR)
        self.play(
            FadeIn(glow_dot, scale=0.1),
            FadeIn(empty_path_text, scale=0.8)
        )
        self.wait()
        for color in [BALANCED_COLOR, NONZERO_COLOR, ALL_PATH_COLOR]:
            self.play(
                *[
                    mob.animate.set_color(color)
                    for mob in (drunk, glow_dot, empty_path_text)
                ],
                run_time=1,
            )
        self.wait()


class RelationsBetweenSetsWPQ(Scene):
    def construct(self):
        # a_n is derived from sets P_n and W_n
        an_formula = Tex(
            "a_n = \\frac{p_n}{w_n} = \\frac{\\text{“正点到家”型路径数}}{\\text{路径总数}}"
        )
        VGroup(an_formula[3:5], an_formula[9:-5]).set_color(BALANCED_COLOR)
        VGroup(an_formula[6:8], an_formula[-4:]).set_color(ALL_PATH_COLOR)
        self.add(an_formula)
        self.wait()

        # We can split a normal path into two special paths
        N = 20
        axes = PathAxes(
            N=N, pos_range=(-5, 5, 1), pos_label_side=LEFT,
        )
        axes.set_width(FRAME_WIDTH - 1)
        axes.center().to_edge(UP, buff=0.5)
        eg_path_chars = "DDDUUUDUUUUDDDUUUDUU"
        path = RandomWalkPath(eg_path_chars, axes=axes)
        path_group = VGroup(axes, path)
        path_group.generate_target()
        path_group.move_to(an_formula[6:8].get_center())
        path_group.scale(0.01).set_opacity(0)
        self.play(
            FadeOut(an_formula),
            MoveToTarget(path_group),
            run_time=1,
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    ApplyMethod(
                        arrow.shift, (1 if ind % 2 else -1) * 0.25 * UP,
                        rate_func=there_and_back_with_pause,
                    )
                    for ind, arrow in enumerate(path)
                ],
                lag_ratio=0.05,
                run_time=2,
            )
        )
        self.wait()

        zeros_indices = path.get_all_zeros_indices()
        zeros_points = path.get_all_zeros_points()
        zeros_dots = Group(
            GlowDot(point, color=ALL_PATH_COLOR)
            for point in zeros_points
        )
        self.play(
            AnimationGroup(
                *[
                    FadeIn(dot, scale=0.1)
                    for dot in zeros_dots
                ],
                lag_ratio=0.2,
                run_time=2,
            ),
        )
        self.wait()

        eg_zero_ind = 2
        eg_split_ind = zeros_indices[eg_zero_ind]
        zeros_dots.generate_target()
        zeros_dots.target.set_opacity(0)
        zeros_dots.target[eg_zero_ind].set_opacity(1)
        split_line = DashedLine(UP, DOWN, stroke_width=3)
        split_line.set_color(ALL_PATH_COLOR)
        split_line.set_height(1.5 * FRAME_HEIGHT)
        split_line.move_to(zeros_dots[eg_zero_ind])
        self.play(
            GrowFromCenter(split_line),
            MoveToTarget(zeros_dots),
        )
        self.wait()

        left_part, right_part = path.split_at_ind(eg_split_ind)

        def update_left_rect(rect):
            left_width = (split_line.get_center() - axes.c2p(0, 0))[0]
            rect.set_width(np.clip(left_width, 0.01, None))
            rect.match_height(split_line, stretch=True)
            rect.next_to(split_line, LEFT, buff=0)

        def update_right_rect(rect):
            right_width = (axes.c2p(N, 0) - split_line.get_center())[0]
            rect.set_width(np.clip(right_width, 0.01, None))
            rect.match_height(split_line, stretch=True)
            rect.next_to(split_line, RIGHT, buff=0)

        left_rect = Rectangle(
            stroke_width=0,
            fill_color=AXES_COLOR, fill_opacity=0.2,
        )
        right_rect = left_rect.copy()
        left_rect.add_updater(update_left_rect)
        right_rect.add_updater(update_right_rect)

        for rect in (left_rect, right_rect):
            rect.update().suspend_updating()
            rect.generate_target()
            rect.stretch(factor=0, dim=0)
            rect.move_to(split_line)

        self.play(
            MoveToTarget(left_rect),
            MoveToTarget(right_rect),
            *[Animation(mob) for mob in (axes, path, split_line)],
        )
        for rect in (left_rect, right_rect):
            rect.resume_updating()
        self.wait()

        # First part is a balanced path
        start, end = path.get_point_by_ind(0), path.get_point_by_ind(eg_split_ind)
        balanced_dots = Group(
            GlowDot(point, color=BALANCED_COLOR)
            for point in (start, end)
        )

        def update_left_title(title):
            title.move_to(left_rect)
            title.next_to(TOP, DOWN, coor_mask=[0, 1, 0])
            default_width = left_rect.get_width() * 0.6
            max_width = min(1.8, default_width)
            title.set_width(np.clip(default_width, 0.01, max_width))
            title.set_opacity(np.clip(left_rect.get_width() * 2, 0, 1))

        def update_right_title(title):
            title.move_to(right_rect)
            title.next_to(TOP, DOWN, coor_mask=[0, 1, 0])
            default_width = right_rect.get_width() * 0.6
            max_width = min(1.8, right_rect.get_width())
            title.set_width(np.clip(default_width, 0.01, max_width))
            title.set_opacity(np.clip(right_rect.get_width() * 2, 0, 1))

        left_title = TexText("“正点到家” \\\\ （集合$P$）")
        left_title.set_color(BALANCED_COLOR)
        left_title.add_updater(update_left_title)
        right_title = TexText("“不再回家” \\\\ （集合$Q$）")
        right_title.set_color(NONZERO_COLOR)
        right_title.add_updater(update_right_title)

        self.play(
            zeros_dots.animate.set_opacity(0),
            AnimationGroup(
                *[
                    FadeIn(dot, scale=0.1)
                    for dot in balanced_dots
                ],
                lag_ratio=0.2,
                run_time=1,
            ),
        )
        self.wait()
        left_title.update().suspend_updating()
        self.play(
            Write(left_title, stroke_color=BALANCED_COLOR),
            left_rect.animate.set_color(BALANCED_COLOR),
            left_part.animate.set_color(BALANCED_COLOR),
        )
        left_title.resume_updating()
        self.wait()

        # If we select the last zero, the second part is a nonzero path
        path.generate_target()
        target_left_part, target_right_part = path.target.split_at_ind(zeros_indices[-1])
        target_left_part.set_color(BALANCED_COLOR)
        self.play(
            FadeOut(balanced_dots),
            MoveToTarget(path),
            split_line.animate.move_to(zeros_dots[-1].get_center()),
        )
        self.wait()
        self.play(
            Flash(zeros_points[-1], color=NONZERO_COLOR)
        )
        self.wait()
        target_right_part.set_color(NONZERO_COLOR)
        right_title.update().suspend_updating()
        self.play(
            Write(right_title, stroke_color=NONZERO_COLOR),
            MoveToTarget(path),
            right_rect.animate.set_color(NONZERO_COLOR),
        )
        right_title.resume_updating()
        self.wait()

        # More examples of spliting paths
        # 3 normal examples and 2 special examples
        new_paths_chars = [
            "UDDUDUUUDUDUUDUUDDUU",
            "UDUDUUUDDDDDDDDUDUDU",
            "DDUUDUUUDDUDDUUUUDUD",
            "UUDDUUUDDDUDDUUDDUDU",
            "DDDDUDDUUDUUUDDDDUUD",
        ]
        new_paths_chars.append(new_paths_chars[0])
        for path_ind, path_chars in enumerate(new_paths_chars):
            new_path = RandomWalkPath(path_chars, axes=axes)
            new_path.save_state()
            new_split_ind = new_path.get_last_zero_ind()
            new_split_point = new_path.get_last_zero_point()
            new_left_part, new_right_part = new_path.split_at_ind(new_split_ind)
            new_left_part.set_color(BALANCED_COLOR)
            new_right_part.set_color(NONZERO_COLOR)
            self.play(
                Transform(path, new_path),
                split_line.animate.move_to(new_split_point),
                run_time=1 if (path_ind <= 2) else 2,
            )
            self.wait(2)

            # After normal examples, show length and set labels
            if path_ind == 2:
                parts = [new_path, new_left_part, new_right_part]
                length_label_texts = ["2n", "2k", "2(n-k)"]
                set_label_texts = ["W_{n}", "P_{k}", "Q_{n-k}"]
                colors = [ALL_PATH_COLOR, BALANCED_COLOR, NONZERO_COLOR]
                positions = [-2.5, -4, -4]
                length_labels = VGroup()
                set_labels = VGroup()
                for part, label_text, color, pos in zip(parts, length_label_texts, colors, positions):
                    length_label = LengthLabel(part, "\\text{长度}=" + label_text, color)
                    length_label.next_to(axes.c2p(0, pos), RIGHT, coor_mask=[0, 1, 0])
                    length_labels.add(length_label)
                    self.play(GrowFromCenter(length_label))
                    self.wait()
                for part, label_text, color, pos in zip(parts, set_label_texts, colors, positions):
                    set_label = LengthLabel(part, "\\text{来自}" + label_text, color)
                    set_label.next_to(axes.c2p(0, pos), RIGHT, coor_mask=[0, 1, 0])
                    set_labels.add(set_label)
                self.play(
                    AnimationGroup(
                        *[
                            AnimationGroup(
                                ReplacementTransform(length_label.get_frame(), set_label.get_frame()),
                                FadeTransform(length_label.get_label(), set_label.get_label()),
                            )
                            for length_label, set_label in zip(length_labels, set_labels)
                        ],
                        lag_ratio=0.3,
                    ),
                    run_time=2,
                )
                self.wait()
                self.play(
                    *[
                        FlashUnder(set_label.get_label()[-sub:], color=set_label.get_label().get_color())
                        for set_label, sub in zip(set_labels, [1, 1, 3])
                    ],
                    lag_ratio=0.1,
                )
                self.wait()
                self.play(FadeOut(set_labels))
                self.wait()

        # We can also combine two special paths into a normal path
        fadeout_mobs = Group(self.mobjects.copy())
        fadeout_mobs.remove(axes)
        for mob in fadeout_mobs:
            mob.suspend_updating()
        self.play(FadeOut(fadeout_mobs))
        for mob in fadeout_mobs:
            mob.resume_updating()
        self.wait()

        wn_path_chars = "DDUDUUUDUUUDUDDUUDDU"
        wn_path = RandomWalkPath(wn_path_chars, axes=axes)
        wn_split_ind = wn_path.get_last_zero_ind()
        wn_split_point = wn_path.get_last_zero_point()
        pk_path, qnmk_path = wn_path.split_at_ind(wn_split_ind, new_path=True)
        pk_path.set_color(BALANCED_COLOR)
        qnmk_path.set_color(NONZERO_COLOR)
        split_line.move_to(wn_split_point)

        pk_path.generate_target()
        pk_path.scale(0.9).shift(2 * RIGHT + DOWN)
        qnmk_path.generate_target()
        qnmk_path.scale(0.9).shift(LEFT + UP)
        self.play(
            AnimationGroup(
                FadeIn(pk_path), FadeIn(qnmk_path),
                lag_ratio=0.2,
            )
        )
        self.wait()

        update_mobs = VGroup(left_rect, left_title, right_rect, right_title)
        for mob in update_mobs:
            mob.update()
        pk_label = LengthLabel(pk_path.target, "&\\text{来自}P_{k} \\\\ &\\text{长度}=2k", color=BALANCED_COLOR)
        qnmk_label = LengthLabel(qnmk_path.target, "&\\text{来自}Q_{n-k} \\\\ &\\text{长度}=2(n-k)", color=NONZERO_COLOR)
        wn_label = LengthLabel(wn_path, "&\\text{来自}W_{n} \\\\ &\\text{长度}=2n", color=ALL_PATH_COLOR)
        for label in (pk_label, qnmk_label, wn_label):
            label.next_to(axes.c2p(0, -4), RIGHT, coor_mask=[0, 1, 0])
        for mob in update_mobs:
            mob.suspend_updating()
        self.play(
            MoveToTarget(pk_path),
            FadeIn(VGroup(left_rect, left_title, pk_label)),
        )
        self.wait()
        self.play(
            MoveToTarget(qnmk_path),
            FadeIn(VGroup(right_rect, right_title, qnmk_label)),
        )
        self.wait()

        # Combine and split in action
        wn_label_copy = wn_label.copy()
        self.play(
            FadeOut(update_mobs),
            FadeTransform(pk_label, wn_label),
            FadeTransform(qnmk_label, wn_label_copy),
            FadeIn(wn_path),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeIn(update_mobs),
            FadeTransform(wn_label, pk_label),
            FadeTransform(wn_label_copy, qnmk_label),
            FadeOut(wn_path),
            run_time=2,
        )
        self.wait()
        for mob in update_mobs:
            mob.resume_updating()
        split_time_text = pk_label.get_label()[-2:].copy()
        wn_split_dot = GlowDot(wn_split_point, color=BALANCED_COLOR)
        self.play(
            FadeIn(wn_split_dot, scale=0.1),
            split_time_text.animate.next_to(wn_split_point, DOWN),
        )
        self.wait()

        # p_k and q_(n-k) choices for each part
        pk_count_label = LengthLabel(pk_path, "p_{k}\\text{种选择}", color=BALANCED_COLOR)
        pk_count_label.move_to(pk_label)
        qnmk_count_label = LengthLabel(qnmk_path, "q_{n-k}\\text{种选择}", color=NONZERO_COLOR)
        qnmk_count_label.move_to(qnmk_label)
        total_count_label = LengthLabel(wn_path, "p_{k} q_{n-k}\\text{种选择}", color=ALL_PATH_COLOR)
        total_count_label.next_to(pk_count_label, RIGHT, coor_mask=[0, 1, 0])
        total_count_label.get_label()[:2].set_color(BALANCED_COLOR)
        total_count_label.get_label()[2:6].set_color(NONZERO_COLOR)
        self.play(FadeTransform(pk_label, pk_count_label))
        self.wait()
        self.play(FadeTransform(qnmk_label, qnmk_count_label))
        self.wait()
        total_count_label_copy = total_count_label.copy()
        self.play(
            FadeTransform(pk_count_label, total_count_label),
            FadeTransform(qnmk_count_label, total_count_label_copy),
        )
        self.remove(total_count_label_copy)
        self.wait()

        # We can partition Wn into multiple sets
        basis_dot = Dot(wn_split_dot.get_center())
        basis_dot.set_color(ALL_PATH_COLOR)
        basis_text = TexText("划分依据").scale(0.8)
        basis_text.set_color(ALL_PATH_COLOR)
        basis_text.next_to(basis_dot, DOWN)
        self.play(
            FadeOut(wn_split_dot),
            FadeIn(basis_dot, scale=0.25),
        )
        self.wait()
        self.play(FadeTransform(split_time_text, basis_text))
        self.wait()
        self.play(
            FlashAround(total_count_label.get_label()[:6], color=ALL_PATH_COLOR),
        )
        self.wait()

        all_path_chars = all_random_walk_strings(N)
        all_last_zero_indices = list(range(0, N + 1, 2))
        split_line.set_opacity(0.2)
        wn_path[:len(pk_path)].set_color(BALANCED_COLOR)
        wn_path[len(pk_path):].set_color(NONZERO_COLOR)
        self.add(wn_path, basis_dot)
        self.remove(pk_path, qnmk_path)
        possible_dots = Group(
            GlowDot(axes.c2p(last_zero_ind, 0)).set_opacity(0.5)
            for last_zero_ind in all_last_zero_indices
        )
        possible_texts = all_last_zero_indices.copy()
        possible_texts[-7:] = ["\\cdots", "\\cdots", "2k", "\\cdots", "\\cdots", "2(n-1)", "2n"]
        possible_times = Group(
            Tex(str(text)).scale(0.5).next_to(dot, DOWN).set_opacity(0.5)
            .next_to(axes.c2p(0, -0.5), RIGHT, coor_mask=[0, 1, 0])
            for text, dot in zip(possible_texts, possible_dots)
        )
        Group(possible_dots, possible_times).set_color(ALL_PATH_COLOR)
        self.play(
            FadeOut(basis_text), FadeOut(total_count_label),
            FadeIn(split_line),
            FadeIn(possible_dots),
            FadeIn(possible_times),
        )
        self.wait()

        random.seed(570)
        count_texts = [
            "p_{0}q_{n}", "p_{1}q_{n-1}", "p_{2}q_{n-2}", "p_{3}q_{n-3}", "\\cdots", "\\cdots",
            "p_{k}q_{n-k}", "\\cdots", "\\cdots", "p_{n-1}q_{1}", "p_{n}q_{0}"
        ]
        count_texs = VGroup()
        for i, (last_zero_ind, count_text) in enumerate(zip(all_last_zero_indices, count_texts)):
            # Not ideal but it works
            example_path_chars = random.choice(all_path_chars)
            while path_last_zero_ind(example_path_chars) != last_zero_ind:
                example_path_chars = random.choice(all_path_chars)
            split_point = axes.c2p(last_zero_ind, 0)
            example_path = RandomWalkPath(example_path_chars, axes=axes)
            left_part, right_part = example_path.split_at_ind(last_zero_ind)
            left_part.set_color(BALANCED_COLOR)
            right_part.set_color(NONZERO_COLOR)
            count_tex = Tex(count_text)
            if count_text == "\\cdots":
                count_tex.set_color(WHITE)
            elif count_text == "p_{n-1}q_{1}":
                count_tex[:-2].set_color(BALANCED_COLOR)
                count_tex[-2:].set_color(NONZERO_COLOR)
            else:
                count_tex[:2].set_color(BALANCED_COLOR)
                count_tex[2:].set_color(NONZERO_COLOR)
            count_tex.move_to(split_point)
            count_pos = -3 if (i % 2 == 0) else -4
            count_tex.next_to(axes.c2p(0, count_pos), coor_mask=[0, 1, 0])
            count_tex.set_backstroke()
            count_texs.add(count_tex)
            self.play(
                split_line.animate.move_to(split_point),
                basis_dot.animate.move_to(split_point),
                Transform(wn_path, example_path),
                Write(count_tex),
            )
            self.wait()

        for mob in update_mobs:
            mob.suspend_updating()
        total_rect = SurroundingRectangle(
            count_texs, color=ALL_PATH_COLOR,
            fill_opacity=0.2, buff=0.3,
        )
        total_text = Tex("w_{n}").scale(2)
        total_text.set_color(ALL_PATH_COLOR)
        total_text.next_to(total_rect, UP, buff=0.3)
        fade_group = Group(
            axes, wn_path, split_line, update_mobs,
            possible_dots, possible_times, basis_dot,
        )
        self.play(
            fade_group.animate.fade(0.95),
            FadeIn(total_rect),
            Animation(count_texs),
        )
        self.wait()
        self.play(FadeIn(total_text, shift=UP))
        self.wait()

        # Now we have a formula between p, q and w
        formula = Tex(
            """
            p_{0}q_{n} + p_{1}q_{n-1} + p_{2}q_{n-2} + \\cdots +
            p_{k}q_{n-k} + \\cdots + p_{n}q_{0} = w_{n}
            """
        )
        formula[39:].set_color(ALL_PATH_COLOR)
        for ind in (0, 5, 12, 23, 34):
            formula[ind:ind + 2].set_color(BALANCED_COLOR)
        for ind in (2, 36):
            formula[ind:ind + 2].set_color(NONZERO_COLOR)
        for ind in (7, 14, 25):
            formula[ind:ind + 4].set_color(NONZERO_COLOR)
        self.play(
            FadeOut(fade_group),
            FadeOut(total_rect),
            AnimationGroup(
                ReplacementTransform(total_text, formula[39:]),
                ReplacementTransform(count_texs[0], formula[:4]),
                ReplacementTransform(count_texs[1], formula[5:11]),
                ReplacementTransform(count_texs[2], formula[12:18]),
                AnimationGroup(
                    FadeTransform(source, target)
                    for source, target in zip(count_texs[3:6], formula[19:22])
                ),
                ReplacementTransform(count_texs[6], formula[23:29]),
                AnimationGroup(
                    FadeTransform(source, target)
                    for source, target in zip(count_texs[7:10], formula[30:33])
                ),
                ReplacementTransform(count_texs[10], formula[34:38]),
                lag_ratio=0.2,
                run_time=5,
            ),
            AnimationGroup(
                *[
                    GrowFromCenter(formula[ind])
                    for ind in [4, 11, 18, 22, 29, 33, 38]
                ],
                lag_ratio=0.2,
                run_time=5,
            ),
        )
        self.add(formula)
        self.wait()

        # We call this a "convolution" property of p and q
        subscript_text = TexText("角标和为$n$的所有情况")
        conv_text = TexText("卷积性质")
        for text in (subscript_text, conv_text):
            text.set_color(ALL_PATH_COLOR)
            text.next_to(formula, UP, buff=0.5)
        self.play(FadeIn(subscript_text, shift=UP))
        self.wait()
        self.play(FadeTransform(subscript_text, conv_text))
        self.wait()

        # Dealing with the excessive q's may result in a formula for a_n
        an_formula = Tex("a_n = \\frac{p_n}{w_n}")
        an_formula[3:5].set_color(BALANCED_COLOR)
        an_formula[-2:].set_color(ALL_PATH_COLOR)
        an_formula.shift(DOWN)
        self.play(
            VGroup(formula, conv_text).animate.shift(UP),
            Write(an_formula),
        )
        self.wait()

        q_indices = [2, 7, 14, 25, 36]
        lengths = [2, 4, 4, 4, 2]
        underlines = VGroup(
            Underline(formula[q_ind:q_ind + dl], stretch_factor=1.1)
            for q_ind, dl in zip(q_indices, lengths)
        )
        underlines.set_color(NONZERO_COLOR)
        self.play(
            AnimationGroup(
                *[ShowCreation(underline) for underline in underlines],
                lag_ratio=0.1,
            ),
        )
        self.wait()


class WeNeedPathInversion(Scene):
    def construct(self):
        # We don't need to know the size of each set
        p2_title = Tex("P_{2}")
        p2_axes = VGroup(
            PathAxes(N=4, include_label=False).scale(0.3)
            for k in range(6)
        )
        p2_axes.arrange_in_grid(2, 3, buff=0.5)
        p2_paths = VGroup(
            RandomWalkPath(ws, axes=axes)
            for ws, axes in zip(all_balanced_walk_strings(4), p2_axes)
        )
        p2_rects = VGroup(
            BackgroundRectangle(axes, color=BALANCED_COLOR, fill_opacity=0.2, buff=0.15)
            for axes in p2_axes
        )
        p2_axes.set_color(AXES_COLOR)
        VGroup(p2_title, p2_rects, p2_paths).set_color(BALANCED_COLOR)
        p2_path_group = VGroup(
            VGroup(rect, axes, path)
            for rect, axes, path in zip(p2_rects, p2_axes, p2_paths)
        )
        p2_group = VGroup(p2_title, p2_path_group)
        p2_group.arrange(DOWN)
        p2_group.move_to(LEFT_SIDE / 2).to_edge(UP)

        q2_title = Tex("Q_{2}")
        q2_axes = p2_axes.copy()
        q2_paths = VGroup(
            RandomWalkPath(ws, axes=axes)
            for ws, axes in zip(all_nonzero_walk_strings(4), p2_axes)
        )
        q2_rects = p2_rects.copy()
        VGroup(q2_title, q2_rects, q2_paths).set_color(NONZERO_COLOR)
        q2_path_group = VGroup(
            VGroup(rect, axes, path)
            for rect, axes, path in zip(q2_rects, q2_axes, q2_paths)
        )
        q2_group = VGroup(q2_title, q2_path_group)
        q2_group.arrange(DOWN)
        q2_group.move_to(RIGHT_SIDE / 2).to_edge(UP)
        self.play(FadeIn(p2_group), FadeIn(q2_group))
        self.wait()

        # Finding a 1-to-1 mapping is sufficient
        path_pairs = {
            p: q
            for p, q in it.product(p2_path_group, q2_path_group)
            if p[-1].path_chars == mapping_special_walk_string(q[-1].path_chars)
        }
        target_group = VGroup()
        for p, q in path_pairs.items():
            self.add(p, q)
            p.generate_target()
            q.generate_target()
            arrow = Tex("\\leftrightarrow")
            temp_group = VGroup(p. target, arrow, q.target)
            temp_group.arrange(RIGHT, buff=0.1)
            rect = BackgroundRectangle(
                temp_group, color=AXES_COLOR, fill_opacity=0.2, buff=0.05,
            )
            temp_group.add(rect)
            target_group.add(temp_group)
        target_group.arrange_in_grid(2, 3, h_buff=0.6, v_buff=0.3)
        target_group.next_to(VGroup(p2_title, q2_title), DOWN)
        anims_p = [MoveToTarget(p) for p in path_pairs.keys()]
        anims_q = [MoveToTarget(q) for q in path_pairs.values()]
        anims_arrow = [
            GrowFromCenter(temp_group[1])
            for temp_group in target_group
        ]
        anims_rect = [
            FadeIn(temp_group[-1])
            for temp_group in target_group
        ]

        pq_title = Tex("| P_{2} | = | Q_{2} |")
        pq_title.to_edge(UP)
        pq_title[1:3].set_color(BALANCED_COLOR)
        pq_title[-3:-1].set_color(NONZERO_COLOR)
        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(ar, ap, aq, aa)
                    for ar, ap, aq, aa in zip(anims_rect, anims_p, anims_q, anims_arrow)
                ],
                lag_ratio=0.1,
            ),
            p2_title.animate.move_to(pq_title[1:3]),
            q2_title.animate.move_to(pq_title[-3:-1]),
            Write(VGroup(pq_title[0], pq_title[3:-3], pq_title[-1])),
            run_time=3,
        )
        self.remove(p2_title, q2_title)
        self.add(pq_title, p2_group, q2_group)
        self.wait()

        # To create a 1-to-1 mapping between P and Q, we need path inversion
        inv_title = TexText("回溯")
        inv_title.scale(1.5)
        inv_title.set_color(GOLD_B)
        inv_title.move_to(pq_title)
        orig_axes = PathAxes(8, pos_range=(-4, 4, 1), unit_size=1.5)
        orig_axes.set_width(FRAME_WIDTH / 2 - 2)
        orig_path_chars = "UUDDDUDD"
        orig_path = RandomWalkPath(orig_path_chars, axes=orig_axes)
        orig_group = VGroup(orig_axes, orig_path)
        orig_rect = BackgroundRectangle(
            orig_group, color=AXES_COLOR, fill_opacity=0.2, buff=0.2,
        )
        orig_group.add(orig_rect)
        orig_group.to_edge(LEFT)
        self.play(
            FadeOut(Group(*self.mobjects)),
            FadeIn(Group(inv_title, orig_group)),
        )
        self.wait()

        # Simply takes the original path, and inverse both direction and time
        inv_axes = orig_axes.copy()
        inv_rect = orig_rect.copy()
        inv_rect.to_edge(RIGHT)
        inv_axes.move_to(inv_rect)
        orig_path_shadow = orig_path.copy().set_opacity(0.2)
        inv_path = orig_path.get_inverse_path(axes=inv_axes)
        inv_path_shadow = inv_path.copy().set_opacity(0.2)
        inv_note = TexText("同时反转时间顺序和移动方向")
        inv_note.match_color(inv_title).move_to(inv_title)

        self.play(
            FadeTransform(inv_title, inv_note),
            ReplacementTransform(orig_rect.copy().set_opacity(0), inv_rect),
            ReplacementTransform(orig_axes.copy().set_opacity(0), inv_axes),
        )
        self.wait()
        self.remove(orig_path)
        self.add(orig_path_shadow)
        self.play(
            ReplacementTransform(
                orig_path.copy()[::-1], inv_path,
                lag_ratio=0.4,
                run_time=5,
            ),
        )
        self.wait()
        upper_arrow = CurvedArrow(
            orig_rect.get_right(), inv_rect.get_left(),
            angle=-PI / 4,
        )
        upper_arrow.match_color(inv_title).scale(0.9).shift(UP)
        upper_text = inv_title.copy().set_height(0.3)
        upper_text.next_to(upper_arrow, UP)
        self.play(
            GrowFromEdge(upper_arrow, LEFT),
            Write(upper_text, stroke_color=upper_text.get_color()),
            FadeIn(orig_path),
        )
        self.remove(orig_path_shadow)
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(Indicate(orig_arrow), Indicate(inv_arrow))
                    for orig_arrow, inv_arrow in zip(orig_path[::-1], inv_path)
                ],
                lag_ratio=0.6,
                run_time=2,
            )
        )
        self.wait()

        # Path inversion looks like a horizontal flip
        self.add(orig_path_shadow, inv_path_shadow)
        self.play(FadeOut(inv_path))
        self.wait()
        orig_path_copy = orig_path.copy()
        inv_path_copy = inv_path.copy()
        flip_center = (orig_path.get_center() + inv_path.get_center()) / 2
        self.play(
            orig_path_copy.animate.flip(axis=UP, about_point=flip_center),
        )
        self.wait()
        self.play(
            ReplacementTransform(
                orig_path_copy[::-1], inv_path_copy,
                lag_ratio=0.05,
                run_time=2,
            )
        )
        self.remove(inv_path_copy)
        self.add(inv_path)
        self.wait()

        # Path inversion is also invertible
        orig_path_copy = orig_path.copy()
        inv_path_copy = inv_path.copy()
        lower_arrow = CurvedArrow(
            inv_rect.get_left(), orig_rect.get_right(),
            angle=-PI / 4,
        )
        lower_arrow.match_color(upper_arrow).match_width(upper_arrow).shift(DOWN)
        lower_text = upper_text.copy()
        lower_text.next_to(lower_arrow, DOWN)
        self.play(FadeOut(orig_path))
        self.wait()
        self.play(
            inv_path_copy.animate.flip(axis=UP, about_point=flip_center),
        )
        self.wait()
        self.play(
            ReplacementTransform(
                inv_path_copy[::-1], orig_path_copy,
                lag_ratio=0.05,
                run_time=2,
            ),
            GrowFromEdge(lower_arrow, RIGHT),
            Write(lower_text, stroke_color=lower_text.get_color()),
        )
        self.remove(orig_path_copy)
        self.add(orig_path)
        self.wait()
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait()


class ThePQConversionRuleDissected(Scene):
    def construct(self):
        # We have 3 basic path operations, now it's time to see the conversion rule
        # Setup left frames (basic ops, rules f and g)
        left_frame_config = {
            "width": 6,
            "height": 3,
            "fill_opacity": 0.2,
            "stroke_width": 2,
            "corner_radius": 0.2,
        }
        shadow_opacity = 0.05
        left_part_shift = 0.4 * DOWN + 0.2 * LEFT
        right_part_shift = 0.2 * DOWN + 0.2 * RIGHT

        frames = VGroup(
            VGroup(RoundedRectangle(**left_frame_config))
            for _ in range(3)
        )
        frame_ops, frame_f, frame_g = frames
        basic_title = TexText("规则框架：")
        basic_rules = VGroup(
            TexText(text)
            for text in ["切割成两段", "回溯前半段", "再拼接两段"]
        )
        basic_rules.arrange(DOWN)
        group_ops = VGroup(basic_title, basic_rules)
        group_ops.arrange(RIGHT, aligned_edge=UP)
        group_ops.set_height(frame_ops.get_height() - 1)
        frame_ops.add(group_ops)
        f_title = TexText("规则$f$：$P_{n} \\to Q_{n}$")
        f_title[:3].set_color(ALL_PATH_COLOR)
        f_title[4:6].set_color(BALANCED_COLOR)
        f_title[-2:].set_color(NONZERO_COLOR)
        f_rule = TexText(
            """
            切割点由第一步方向决定 \\\\
            第一步向上 $ \\to $ 第一个最大值处 \\\\
            第一步向下 $ \\to $ 第一个最小值处
            """,
            alignment="",
        )
        f_rule[:11].set_color(ALL_PATH_COLOR)
        f_title.next_to(frame_f.get_top(), DOWN)
        f_rule.set_width(frame_f.get_width() * 0.9)
        f_rule.next_to(f_title, DOWN, buff=0.5)
        group_f = VGroup(f_title)
        frame_f.add(group_f, f_rule)
        g_title = TexText("规则$g$：$Q_{n} \\to P_{n}$")
        g_title[:3].set_color(ALL_PATH_COLOR)
        g_title[-2:].set_color(BALANCED_COLOR)
        g_title[4:6].set_color(NONZERO_COLOR)
        g_rule = TexText(
            """
            切割点由路径终点位置$h$决定 \\\\
            路径最后一次穿过位置$=\\frac{h}{2}$处 \\\\
            （仅接触不算）
            """,
            alignment="",
        )
        g_rule[:13].set_color(ALL_PATH_COLOR)
        g_title.next_to(frame_g.get_top(), DOWN)
        g_rule.set_width(frame_f.get_width() * 0.9)
        g_rule.next_to(g_title, DOWN, buff=0.5)
        frame_g.add(g_title, g_rule)
        frames.arrange(DOWN).set_height(FRAME_HEIGHT - 1)

        # Setup right frame (P-Q path conversion)
        right_frame_config = left_frame_config.copy()
        right_frame_config.update({
            "height": frames.get_height(),
            "width": 9,
        })
        frame_path = RoundedRectangle(**right_frame_config)
        frame_path.next_to(frames, RIGHT, buff=0.2)
        VGroup(frames, frame_path).center().to_edge(UP, buff=0.2)
        eg_x_path_chars = "UDDUUUDUDD"
        axes = PathAxes(
            N=len(eg_x_path_chars),
            pos_range=(-5, 3, 1),
        )
        axes.set_color(AXES_COLOR)
        axes.set_width(frame_path.get_width() - 1)
        axes.move_to(frame_path)
        eg_x_path = RandomWalkPath(eg_x_path_chars, axes=axes)
        frame_path.add(axes, eg_x_path)

        # Start of the scene
        ops = VGroup(
            TexText(text)
            for text in ["切割", "拼接", "回溯"]
        )
        ops.arrange(DOWN, buff=0.8)
        self.play(
            AnimationGroup(
                *[FadeIn(op, shift=RIGHT) for op in ops],
                lag_ratio=0.1,
            ),
        )
        self.wait()
        frame_ops.remove(group_ops)
        frame_f.remove(f_rule)
        frame_g.remove(g_rule)
        self.play(
            AnimationGroup(
                *[FadeIn(frame, shift=RIGHT) for frame in frames],
                FadeIn(frame_path, shift=LEFT),
                lag_ratio=0.1,
            ),
            ops.animate.arrange(DOWN).scale(0.8).move_to(frame_ops),
        )
        frame_path.remove(axes, eg_x_path)
        self.add(axes, eg_x_path)
        self.wait()
        for frame in (frame_f, frame_g):
            self.play(Indicate(frame, scale_factor=1.05))
            self.wait()

        self.play(
            AnimationGroup(
                ReplacementTransform(ops[0], basic_rules[0][:2]),
                ReplacementTransform(ops[2], basic_rules[1][:2]),
                ReplacementTransform(ops[1], basic_rules[2][1:3]),
                lag_ratio=0.1,
                run_time=1,
            ),
            AnimationGroup(
                Write(basic_title),
                Write(basic_rules[0][2:]),
                Write(basic_rules[1][2:]),
                Write(VGroup(basic_rules[2][0], basic_rules[2][3:])),
                lag_ratio=0.1,
                run_time=2,
            ),
        )
        self.remove(ops)
        self.wait()
        self.wait()

        # Introducing rule f
        f_cover_rect = BackgroundRectangle(frame_f, buff=0.1)
        g_cover_rect = BackgroundRectangle(frame_g, buff=0.1)
        path_cover_rect = BackgroundRectangle(frame_path, buff=0.1)
        for rect in (f_cover_rect, g_cover_rect, path_cover_rect):
            rect.set_fill(opacity=0.9)
        self.play(
            FadeIn(g_rule, run_time=3),
            FadeIn(g_cover_rect, run_time=1),
            Write(f_rule, run_time=3),
        )
        self.wait()
        self.play(
            FlashUnder(f_rule[17:20], color=ALL_PATH_COLOR, buff=0),
            FlashUnder(f_rule[-7:-4], color=ALL_PATH_COLOR, buff=0),
            run_time=2,
        )
        self.wait()

        path_x_note = TexText("路径$x \\, (\\in P_{n})$")
        path_x_note.set_color(BALANCED_COLOR)
        path_conv_note = TexText("路径转化中...")
        path_conv_note.set_color(AXES_COLOR)
        path_y_note = TexText("路径$y \\, (\\in Q_{n})$")
        path_y_note.set_color(NONZERO_COLOR)
        for note in (path_x_note, path_y_note, path_conv_note):
            note.scale(0.8)
            note.next_to(frame_path.get_corner(UR), DL)
        self.play(
            Write(path_x_note, stroke_color=BALANCED_COLOR),
            ApplyMethod(eg_x_path.set_color, BALANCED_COLOR),
        )
        self.wait()
        eg_x_path_shadow = eg_x_path.copy()
        eg_x_path_shadow.set_opacity(shadow_opacity)

        first_step_circle = Circle().set_color(ALL_PATH_COLOR)
        first_step_circle.surround(eg_x_path.get_arrow_by_index(0))
        split_ind_x = path_first_max_or_min_ind(eg_x_path_chars)
        split_point_x = eg_x_path.get_point_by_ind(split_ind_x)
        split_line_x = DashedLine(UP, DOWN, stroke_width=3)
        split_line_x.set_color(ALL_PATH_COLOR)
        split_line_x.match_height(axes)
        split_line_x.move_to(axes).move_to(split_point_x, coor_mask=[1, 0, 0])
        split_dot_x = GlowDot(split_point_x)
        left_part_x, right_part_x = eg_x_path.split_at_ind(split_ind_x, new_path=True)
        self.play(ShowCreation(first_step_circle))
        self.wait()
        first_step_circle.generate_target()
        first_step_circle.target.move_to(split_dot_x).scale(1 / 20.).set_stroke(width=0).set_opacity(0)
        split_dot_x.generate_target()
        split_dot_x.move_to(first_step_circle).scale(10).set_opacity(0)
        first_extreme_text = TexText("第一个最值")
        first_extreme_text.scale(0.5)
        first_extreme_text.match_color(split_dot_x)
        first_extreme_text.next_to(split_point_x, UL)
        self.play(
            MoveToTarget(first_step_circle),
            MoveToTarget(split_dot_x),
            FadeInFromPoint(first_extreme_text, split_point_x),
        )
        self.wait()
        self.remove(eg_x_path)
        self.add(eg_x_path_shadow)
        self.play(
            FadeOut(split_dot_x),
            FadeOut(first_extreme_text),
            GrowFromPoint(split_line_x, split_point_x),
            left_part_x.animate.shift(left_part_shift),
            right_part_x.animate.shift(right_part_shift),
            FadeTransform(path_x_note, path_conv_note),
        )
        self.wait()
        left_part_x_inv = left_part_x.get_inverse_path()
        left_part_x_inv.put_start_at(eg_x_path.get_start())
        left_part_x_inv.set_color(NONZERO_COLOR)
        right_part_x.generate_target()
        right_part_x.target.set_color(NONZERO_COLOR)
        right_part_x.target.put_start_at(left_part_x_inv.get_end())
        self.play(
            left_part_x.animate.flip(),
            FadeOut(split_line_x),
        )
        self.wait()
        self.play(
            ReplacementTransform(left_part_x[::-1], left_part_x_inv),
            MoveToTarget(right_part_x),
            FadeTransform(path_conv_note, path_y_note),
        )
        self.wait()

        # Introducing rule g
        eg_y_path_chars = mapping_special_walk_string(eg_x_path_chars)
        eg_y_path = RandomWalkPath(eg_y_path_chars, axes=axes)
        eg_y_path.set_color(NONZERO_COLOR)
        eg_y_path_shadow = eg_y_path.copy()
        eg_y_path_shadow.set_opacity(shadow_opacity)
        self.remove(left_part_x_inv, right_part_x)
        self.add(eg_y_path)
        self.play(
            FadeOut(g_cover_rect),
            FadeIn(f_cover_rect),
        )
        self.wait()

        split_ind_y = path_last_half_cross_ind(eg_y_path_chars)
        split_point_y = eg_y_path.get_point_by_ind(split_ind_y)
        split_line_y = split_line_x.copy()
        split_line_y.move_to(axes).move_to(split_point_y, coor_mask=[1, 0, 0])
        split_dot_y = GlowDot(split_point_y)
        left_part_y, right_part_y = eg_y_path.split_at_ind(split_ind_y, new_path=True)
        final_pos_coord = path_final_coord(eg_y_path_chars)
        final_pos = axes.c2p(0, final_pos_coord)
        final_pos_line = DashedLine(ORIGIN, eg_y_path.get_width() / 2 * RIGHT)
        final_pos_line.match_width(eg_y_path)
        final_pos_line.next_to(final_pos, RIGHT, buff=0)
        final_pos_tex = Tex("h")
        half_final_pos_tex = Tex("\\frac{h}{2}")
        for tex in (final_pos_tex, half_final_pos_tex):
            tex.scale(0.6)
        final_pos_tex.next_to(final_pos, LEFT)
        half_final_pos = axes.c2p(0, final_pos_coord / 2)
        half_final_pos_line = final_pos_line.copy()
        half_final_pos_line.next_to(half_final_pos, RIGHT, buff=0)
        half_final_pos_tex.next_to(half_final_pos, LEFT)
        last_cross_text = TexText("最后一次穿过$\\frac{h}{2}$")
        last_cross_text.scale(0.5)
        last_cross_text.match_color(split_dot_y)
        last_cross_text.next_to(split_point_y, UR)
        for mob in (final_pos_line, half_final_pos_line, final_pos_tex, half_final_pos_tex):
            mob.set_color(ALL_PATH_COLOR)
        self.play(
            GrowFromPoint(final_pos_line, final_pos),
            GrowFromPoint(final_pos_tex, final_pos),
        )
        self.wait()
        self.play(
            ReplacementTransform(final_pos_line.copy(), half_final_pos_line),
            ReplacementTransform(final_pos_tex.copy()[0], half_final_pos_tex[0]),
            FadeInFromPoint(half_final_pos_tex[1:], half_final_pos),
            final_pos_line.animate.set_opacity(0.3),
            final_pos_tex.animate.set_opacity(0.3),
        )
        self.wait()
        self.play(
            FadeOut(final_pos_line),
            FadeOut(final_pos_tex),
            half_final_pos_line.animate.set_opacity(0.3),
            half_final_pos_tex.animate.set_opacity(0.3),
            FadeOutToPoint(half_final_pos_line.copy(), split_point_y),
            FadeInFromPoint(last_cross_text, split_point_y),
            FadeIn(split_dot_y, scale=0.1),
        )
        self.wait()
        self.play(
            FadeOutToPoint(half_final_pos_line, split_point_y),
            FadeOut(half_final_pos_tex),
            GrowFromPoint(split_line_y, split_point_y),
        )
        self.wait()

        self.remove(eg_y_path)
        self.add(eg_y_path_shadow, left_part_y, right_part_y)
        self.play(
            FadeOut(split_dot_y),
            FadeOut(last_cross_text),
            left_part_y.animate.shift(left_part_shift),
            right_part_y.animate.shift(right_part_shift),
            FadeTransform(path_y_note, path_conv_note),
        )
        self.wait()
        left_part_y_inv = left_part_y.get_inverse_path()
        left_part_y_inv.put_start_at(eg_x_path.get_start())
        left_part_y_inv.set_color(BALANCED_COLOR)
        right_part_y.generate_target()
        right_part_y.target.set_color(BALANCED_COLOR)
        right_part_y.target.put_start_at(left_part_y_inv.get_end())
        self.play(
            left_part_y.animate.flip(),
            FadeOut(split_line_y),
        )
        self.wait()
        self.play(
            ReplacementTransform(left_part_y[::-1], left_part_y_inv),
            MoveToTarget(right_part_y),
            FadeTransform(path_conv_note, path_x_note),
        )
        self.remove(left_part_y_inv, right_part_y)
        self.add(eg_x_path)
        self.wait()

        # These two rules work, but why?
        claim = Tex("|P_{n}| = |Q_{n}| \\; ?")
        claim[:4].set_color(BALANCED_COLOR)
        claim[5:9].set_color(NONZERO_COLOR)
        claim.next_to(axes.get_corner(DL), RIGHT, buff=0.5)
        path_y_note_alt = path_y_note.copy()
        path_y_note_alt.next_to(path_x_note, DOWN, aligned_edge=LEFT, buff=0.1)
        self.play(
            FadeOut(f_cover_rect),
            FadeIn(eg_y_path),
            FadeIn(path_y_note_alt),
        )
        self.wait()
        self.play(Write(claim[:-1]))
        self.wait()
        self.play(Write(claim[-1]))
        self.wait()

        self.add(eg_x_path_shadow)
        self.play(
            *[
                FadeOut(mob)
                for mob in (claim, path_x_note, path_y_note_alt, eg_x_path, eg_y_path)
            ],
            FadeIn(path_cover_rect),
        )
        self.wait()

        # First, the rules themselves are mapping
        note_ops = TexText("确定的操作")
        note_ops.next_to(frame_ops, RIGHT, buff=2.5)
        note_split = TexText("确定的切割点选取方式")
        note_rule = TexText("确定的规则")
        note_mapping = TexText("映射")
        note_mapping.set_color(ALL_PATH_COLOR)
        for note in (note_split, note_rule, note_mapping):
            note.next_to(VGroup(frame_f, frame_g), RIGHT)
            note.next_to(note_ops, DOWN, aligned_edge=LEFT, coor_mask=[1, 0, 0])
        arrow_ops = Arrow(note_ops.get_left(), frame_ops.get_right())
        arrow_f = Arrow(note_split.get_left(), frame_f.get_right())
        arrow_g = Arrow(note_split.get_left(), frame_g.get_right())

        self.play(
            AnimationGroup(
                *[Write(note) for note in (note_ops, note_split)],
                lag_ratio=0.1,
            ),
            AnimationGroup(
                *[GrowArrow(arrow) for arrow in (arrow_ops, arrow_f, arrow_g)],
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeTransform(note_ops, note_rule),
            FadeTransform(note_split, note_rule),
            FadeOutToPoint(arrow_ops, arrow_ops.get_end()),
        )
        self.wait()
        self.play(
            FadeTransform(note_rule, note_mapping),
            *[
                arrow.animate.match_color(note_mapping)
                for arrow in (arrow_f, arrow_g)
            ],
        )
        self.wait()

        # Second, the rules are mappings between P and Q
        self.play(
            FadeOut(VGroup(eg_y_path_shadow, path_cover_rect)),
            FadeIn(VGroup(path_x_note, eg_x_path)),
            FadeIn(g_cover_rect),
            FadeOut(VGroup(arrow_f, arrow_g, note_mapping)),
        )
        self.wait()
        left_part_x, right_part_x = eg_x_path.split_at_ind(split_ind_x, new_path=True)
        left_part_x_copy = left_part_x.copy()
        right_part_x_copy = right_part_x.copy()
        left_part_y, right_part_y = eg_y_path.split_at_ind(split_ind_y, new_path=True)
        left_part_y_copy = left_part_y.copy()
        right_part_y_copy = right_part_y.copy()
        part_copies = VGroup(
            left_part_x_copy, right_part_x_copy,
            left_part_y_copy, right_part_y_copy,
        )
        self.remove(eg_x_path, eg_x_path_shadow)
        self.add(left_part_x_copy, right_part_x_copy)

        rect_x_kwargs = {
            "fill_opacity": 0.2,
            "color": BALANCED_COLOR,
        }
        left_part_x_rect = left_part_x.get_range_rectangle(**rect_x_kwargs)
        right_part_x_rect = right_part_x.get_range_rectangle(**rect_x_kwargs)
        left_part_x_top = left_part_x.get_range_top()
        right_part_x_top = right_part_x.get_range_top()
        rect_y_kwargs = {
            "fill_opacity": 0.2,
            "color": NONZERO_COLOR,
        }
        left_part_y_rect = left_part_y.get_range_rectangle(**rect_y_kwargs)
        right_part_y_rect = right_part_y.get_range_rectangle(**rect_y_kwargs)
        left_part_y_top = left_part_y.get_range_top()
        right_part_y_top = right_part_y.get_range_top()
        range_rects = VGroup(
            left_part_x_rect, right_part_x_rect,
            left_part_y_rect, right_part_y_rect,
        )
        range_tops = VGroup(
            left_part_x_top, right_part_x_top,
            left_part_y_top, right_part_y_top,
        )
        for mob_sets in (part_copies, range_rects, range_tops):
            for mob in mob_sets:
                mob.save_state()

        # rule f
        self.play(
            GrowFromPoint(split_line_x, split_point_x),
            GrowFromPoint(left_part_x_rect, split_point_x),
            GrowFromPoint(right_part_x_rect, split_point_x),
            FadeIn(split_dot_x),
        )
        self.wait()
        self.play(
            AnimationGroup(
                GrowFromPoint(left_part_x_top, split_point_x),
                GrowFromPoint(right_part_x_top, split_point_x),
                Animation(split_dot_x),
                Animation(split_line_x),
            ),
        )
        self.wait()

        self.play(
            AnimationGroup(
                GrowFromPoint(first_extreme_text, split_point_x),
                Animation(split_dot_x),
                Animation(split_line_x),
            ),
        )
        self.wait()

        left_part_y_copy.set_color(BALANCED_COLOR)
        left_part_x_rect.generate_target()
        left_part_x_rect.target.move_to(left_part_y_copy.get_range_rectangle())
        self.play(
            AnimationGroup(
                left_part_x_copy.animate.flip(),
                Animation(left_part_x_rect),
                Animation(split_dot_x),
                Animation(split_line_x),
                FadeTransform(path_x_note, path_conv_note),
            ),
        )
        self.play(
            AnimationGroup(
                ReplacementTransform(left_part_x_copy[::-1], left_part_y_copy),
                MoveToTarget(left_part_x_rect),
                left_part_x_top.animate.next_to(left_part_x_rect.target, UP, buff=0),
                Animation(split_dot_x),
                Animation(split_line_x),
            ),
            run_time=2,
        )
        self.add(left_part_x_rect, left_part_x_top)
        self.wait()

        things_to_think = TexText("（思考：如果路径$x$的第一步向下，证明过程需要做哪些修改?）")
        things_to_think.set_color(GREY)
        things_to_think.set_width(frame_path.get_width() * 0.9)
        things_to_think.next_to(frame_path.get_bottom(), UP, buff=0.2)
        right_part_y_copy.set_color(BALANCED_COLOR)
        right_part_x_rect.generate_target()
        right_part_x_rect.target.move_to(right_part_y_copy.get_range_rectangle())
        self.play(
            AnimationGroup(
                ReplacementTransform(right_part_x_copy, right_part_y_copy),
                MoveToTarget(right_part_x_rect),
                right_part_x_top.animate.next_to(right_part_x_rect.target, UP, buff=0),
                Animation(split_line_x),
            ),
            FadeOut(Group(split_dot_x, first_extreme_text)),
            FadeIn(things_to_think),
        )
        self.add(right_part_x_rect, right_part_x_top, split_line_x)
        self.wait()

        for line in (left_part_x_top, right_part_x_top):
            self.play(
                AnimationGroup(
                    WiggleOutThenIn(line),
                    Animation(split_line_x),
                ),
            )
            self.wait()
        aux_x_group = VGroup(
            left_part_x_rect, right_part_x_rect,
            left_part_x_top, right_part_x_top,
            split_line_x,
        )
        self.play(
            FadeOut(aux_x_group),
            Restore(left_part_y_copy),
            Restore(right_part_y_copy),
            FadeTransform(path_conv_note, path_y_note),
            FadeOut(things_to_think),
        )
        self.wait()
        self.play(
            FlashAround(f_title, color=ALL_PATH_COLOR),
            run_time=2,
        )
        self.wait()

        # rule g
        for part in part_copies:
            part.restore()
        self.add(left_part_y_copy, right_part_y_copy)
        self.remove(left_part_x_copy, right_part_x_copy)
        for mob in (final_pos_line, final_pos_tex, half_final_pos_tex, half_final_pos_line):
            mob.set_opacity(1)
        self.play(
            FadeOut(g_cover_rect),
            FadeIn(f_cover_rect),
        )
        self.wait()
        self.play(
            GrowFromPoint(final_pos_line, final_pos),
            GrowFromPoint(final_pos_tex, final_pos),
        )
        self.wait()
        self.play(
            ReplacementTransform(final_pos_line.copy(), half_final_pos_line),
            ReplacementTransform(final_pos_tex.copy()[0], half_final_pos_tex[0]),
            FadeInFromPoint(half_final_pos_tex[1:], half_final_pos),
            final_pos_line.animate.set_opacity(0.3),
            final_pos_tex.animate.set_opacity(0.3),
            FadeInFromPoint(last_cross_text, split_point_y),
            FadeIn(split_dot_y, scale=0.1),
        )
        self.wait()
        self.play(
            half_final_pos_line.animate.set_opacity(0.3),
            half_final_pos_tex.animate.set_opacity(0.3),
            GrowFromPoint(split_line_y, split_point_y),
        )
        self.wait()

        left_part_x_disp = left_part_x_copy.get_displacement_rect(right_side=True, shift=True)
        right_part_x_disp = right_part_x_copy.get_displacement_rect(right_side=False, shift=True)
        left_part_y_disp = left_part_y_copy.get_displacement_rect(right_side=True, shift=True)
        right_part_y_disp = right_part_y_copy.get_displacement_rect(right_side=False, shift=True)
        disps = VGroup(
            left_part_x_disp, right_part_x_disp,
            left_part_y_disp, right_part_y_disp,
        )
        for disp in disps:
            disp.save_state()
        self.play(
            *[
                GrowFromEdge(disp, UP)
                for disp in (left_part_y_disp, right_part_y_disp)
            ],
        )
        self.wait()

        VGroup(left_part_x_copy, right_part_x_copy).set_color(NONZERO_COLOR)
        self.play(
            AnimationGroup(
                left_part_y_copy.animate.flip(),
                Animation(left_part_y_disp),
                FadeTransform(path_y_note, path_conv_note),
            )
        )
        self.play(
            AnimationGroup(
                ReplacementTransform(left_part_y_copy[::-1], left_part_x_copy),
                ReplacementTransform(right_part_y_copy, right_part_x_copy),
                ReplacementTransform(left_part_y_disp, left_part_x_disp),
                ReplacementTransform(right_part_y_disp, right_part_x_disp),
                run_time=2,
            ),
            FadeOut(
                Group(
                    split_dot_y, last_cross_text, final_pos_line, final_pos_tex,
                    half_final_pos_line, half_final_pos_tex,
                ),
                run_time=1,
            ),
        )
        self.wait()
        disps_x = VGroup(left_part_x_disp, right_part_x_disp)
        disps_x.generate_target()
        disps_x.target.move_to(eg_x_path.get_end())
        disps_x.target.scale(0.01).set_opacity(0)
        end_dot = GlowDot(disps_x.target.get_center())
        end_dot.set_color(BALANCED_COLOR)
        end_dot.generate_target()
        end_dot.move_to(disps_x).match_height(disps_x).set_opacity(0)
        self.play(
            FadeOut(split_line_y),
            MoveToTarget(disps_x),
            MoveToTarget(end_dot),
        )
        self.wait()
        self.play(
            Restore(left_part_x_copy),
            Restore(right_part_x_copy),
            FadeOut(end_dot),
            FadeTransform(path_conv_note, path_x_note),
        )
        self.wait()
        self.play(
            FlashAround(g_title, color=ALL_PATH_COLOR),
            run_time=2,
        )
        self.wait()

        # Third, the rules are inverse of each other
        conv_colormap = {
            "x": BALANCED_COLOR,
            "路径$x$": BALANCED_COLOR,
            "P_n": BALANCED_COLOR,
            "y": NONZERO_COLOR,
            "Q_n": NONZERO_COLOR,
            "路径$y$": NONZERO_COLOR,
            "f": ALL_PATH_COLOR,
            "g": ALL_PATH_COLOR,
        }
        conv_chains = VGroup(
            TexText(
                "对$P_n$中的任意路径$x$，均有$x \\overset{f}{\\to} y  \\overset{g}{\\to} x$",
                t2c=conv_colormap,
            ),
            TexText(
                "对$Q_n$中的任意路径$y$，均有$y \\overset{g}{\\to} x  \\overset{f}{\\to} y$",
                t2c=conv_colormap,
            ),
        )
        conv_chains.scale(0.8)
        for chain in conv_chains:
            chain.next_to(frame_path.get_bottom(), UP, buff=0.3)
        split_line_common = split_line_x.copy()
        split_line_common.scale(0.9)

        # From f to g, and back to f
        for mob_sets in (part_copies, range_rects, range_tops, disps):
            for mob in mob_sets:
                mob.restore()
        for range_rect in range_rects:
            range_rect.set_opacity(0.2)
        self.remove(disps)
        self.add(eg_x_path_shadow, eg_y_path_shadow)
        self.play(
            FadeOut(f_cover_rect),
            FadeIn(
                Group(
                    left_part_y_copy, right_part_y_copy, path_y_note_alt,
                ),
            ),
        )
        self.wait()
        self.play(
            FlashAround(VGroup(frame_f, frame_g), color=ALL_PATH_COLOR),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeIn(
                Group(
                    split_dot_x, split_dot_y, split_line_common,
                    first_extreme_text, last_cross_text,
                ),
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    Animation(mob)
                    for mob in (left_part_x_copy, right_part_x_copy)
                ],
                FadeOut(
                    Group(
                        left_part_y_copy, right_part_y_copy,
                        split_dot_y, last_cross_text,
                        path_y_note_alt,
                    ),
                ),
                FadeIn(
                    Group(
                        left_part_x_rect, right_part_x_rect,
                        left_part_x_disp, right_part_x_disp,
                        left_part_x_top, right_part_x_top,
                    ),
                ),
            )
        )
        self.wait()

        left_circle = Circle()
        left_circle.match_color(first_step_circle)
        left_circle.surround(left_part_x_copy.get_arrow_by_index(0))
        right_circle = left_circle.copy()
        right_circle.surround(right_part_x_copy.get_arrow_by_index(0))
        self.play(
            ShowCreation(left_circle),
            ShowCreation(right_circle),
        )
        self.wait()
        self.play(
            left_part_x_copy.animate.flip(),
            left_circle.animate.flip(about_point=left_part_x_copy.get_center()),
            FadeTransform(path_x_note, path_conv_note),
        )
        self.play(
            AnimationGroup(
                ReplacementTransform(left_part_x_copy[::-1], left_part_y_copy),
                ReplacementTransform(left_part_x_rect, left_part_y_rect),
                ReplacementTransform(left_part_x_disp, left_part_y_disp),
                ReplacementTransform(left_part_x_top, left_part_y_top),
                left_circle.animate.surround(left_part_y_copy.get_arrow_by_index(-1)),
                right_circle.animate.surround(right_part_y_copy.get_arrow_by_index(0)),
                split_dot_x.animate.set_opacity(0.2),
                first_extreme_text.animate.set_opacity(0.2),
                ReplacementTransform(right_part_x_copy, right_part_y_copy),
                ReplacementTransform(right_part_x_rect, right_part_y_rect),
                ReplacementTransform(right_part_x_disp, right_part_y_disp),
                ReplacementTransform(right_part_x_top, right_part_y_top),
                FadeTransform(path_conv_note, path_y_note),
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeOutToPoint(left_circle, split_point_y),
            FadeOutToPoint(right_circle, split_point_y),
            FadeIn(split_dot_y, scale=0.1),
        )
        self.wait()

        self.play(
            AnimationGroup(
                WiggleOutThenIn(right_part_y_top),
                Animation(split_line_common),
                Animation(split_dot_y),
            )
        )
        self.wait()
        self.play(
            FadeInFromPoint(last_cross_text, split_point_y),
        )
        self.wait()

        for mob in part_copies:
            mob.restore()
        self.play(
            FadeOut(
                VGroup(
                    left_part_y_rect, right_part_y_rect,
                    left_part_y_top, right_part_y_top,
                    left_part_y_disp, right_part_y_disp,
                ),
            ),
            FadeIn(
                VGroup(
                    left_part_x_copy, right_part_x_copy,
                    path_x_note,
                    conv_chains[0],
                ),
            ),
            FadeTransform(path_y_note, path_y_note_alt),
            split_dot_x.animate.set_opacity(1),
            first_extreme_text.animate.set_opacity(1),
            *[
                Animation(mob)
                for mob in (split_line_common, split_dot_y)
            ],
        )
        self.wait()

        # From g to f, and back to g
        for mob_sets in (part_copies, range_rects, range_tops, disps):
            for mob in mob_sets:
                mob.restore()
        for range_rect in range_rects:
            range_rect.set_opacity(0.2)
        self.play(
            FadeOut(
                Group(
                    conv_chains[0], path_x_note,
                    left_part_x_copy, right_part_x_copy,
                    split_dot_x, first_extreme_text,
                ),
            ),
            FadeTransform(path_y_note_alt, path_y_note),
        )
        self.wait()
        self.play(
            AnimationGroup(
                FadeIn(
                    Group(
                        left_part_y_rect, right_part_y_rect,
                        left_part_y_top, right_part_y_top,
                    ),
                ),
                Animation(split_line_common),
                Animation(split_dot_y),
            )
        )
        self.wait()
        origin_dot = GlowDot(axes.c2p(0, 0))
        origin_dot.set_color(NONZERO_COLOR)
        self.play(
            FadeIn(origin_dot, scale=0.1),
        )
        self.wait()
        self.play(
            left_part_y_copy.animate.flip(),
            origin_dot.animate.flip(about_point=left_part_y_copy.get_center()),
            FadeTransform(path_y_note, path_conv_note),
        )
        self.play(
            AnimationGroup(
                ReplacementTransform(left_part_y_copy[::-1], left_part_x_copy),
                ReplacementTransform(left_part_y_rect, left_part_x_rect),
                ReplacementTransform(left_part_y_top, left_part_x_top),
                split_dot_y.animate.set_opacity(0.2),
                last_cross_text.animate.set_opacity(0.2),
                ReplacementTransform(right_part_y_copy, right_part_x_copy),
                ReplacementTransform(right_part_y_rect, right_part_x_rect),
                ReplacementTransform(right_part_y_top, right_part_x_top),
                FadeTransform(path_conv_note, path_x_note),
                origin_dot.animate.set_color(ALL_PATH_COLOR).move_to(split_point_x),
                Animation(split_line_common),
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeInFromPoint(first_extreme_text, split_point_x),
        )
        self.wait()
        for mob in part_copies:
            mob.restore()
        self.play(
            FadeOut(
                VGroup(
                    left_part_x_rect, right_part_x_rect,
                    left_part_x_top, right_part_x_top,
                ),
            ),
            FadeIn(
                VGroup(
                    left_part_y_copy, right_part_y_copy,
                    path_y_note_alt,
                    conv_chains[1],
                ),
            ),
            split_dot_y.animate.set_opacity(1),
            last_cross_text.animate.set_opacity(1),
            *[
                Animation(mob)
                for mob in (split_line_common, split_dot_x)
            ],
        )
        self.wait()

        # Now look closely at those conversion chains
        conv_chains.generate_target()
        conv_chains[0].set_opacity(0)
        conv_chains.target.arrange(DOWN, aligned_edge=LEFT, buff=frame_path.get_height() / 2.2)
        conv_chains.target.next_to(frame_path.get_top(), DOWN)
        path_cover_rect.save_state()
        path_cover_rect.set_opacity(1)
        self.play(
            FadeIn(path_cover_rect),
            MoveToTarget(conv_chains),
        )
        self.wait()

        pnqn_colormap = {
            "|P_{n}|": BALANCED_COLOR,
            "p_{n}": BALANCED_COLOR,
            "|Q_{n}|": NONZERO_COLOR,
            "q_{n}": NONZERO_COLOR,
        }
        claim_xyx, claim_yxy = conv_chains
        chain_xyx = conv_chains[0][-7:]
        ineq_xyx = Tex("|Q_{n}| \\geq |P_{n}|", t2c=pnqn_colormap)
        ineq_xyx.next_to(chain_xyx, DOWN, buff=0.5)
        chain_yxy = conv_chains[1][-7:]
        ineq_yxy = Tex("|P_{n}| \\geq |Q_{n}|", t2c=pnqn_colormap)
        ineq_yxy.next_to(chain_yxy, DOWN, buff=0.5)
        self.play(
            FlashAround(chain_xyx, color=ALL_PATH_COLOR),
            run_time=2,
        )
        self.wait()
        self.play(FadeIn(ineq_xyx, shift=DOWN))
        self.wait()

        path_config = {
            "unit_size": 0.2,
        }
        p1_path, p2_path = VGroup(
            RandomWalkPath(path_chars, **path_config)
            # for path_chars in random.sample(all_balanced_walk_strings(6), 2)
            for path_chars in ["UUUDDD", "DUDUDU"]
        ).set_color(BALANCED_COLOR)
        q1_path, q2_path = VGroup(
            RandomWalkPath(path_chars, **path_config)
            # for path_chars in random.sample(all_nonzero_walk_strings(6), 2)
            for path_chars in ["UUDUDU", "UUUUDD"]
        ).set_color(NONZERO_COLOR)
        for path in (p1_path, p2_path, q1_path, q2_path):
            dots = Tex("\\cdots")
            dots.match_style(path[2])
            dots.set_width(path[2].get_width() * 0.8)
            dots_2, dots_3 = dots.replicate(2)
            dots_2.rotate(PI / 4 if (path[2].get_vector()[1] > 0) else -PI / 4)
            dots_3.rotate(PI / 4 if (path[3].get_vector()[1] > 0) else -PI / 4)
            path[2].become(dots_2.move_to(path[2]))
            path[3].become(dots_3.move_to(path[3]))
        paths_template = VGroup(p1_path, q1_path, p2_path, q2_path)
        paths_template.arrange_in_grid(2, 2, h_buff=1.5)
        back_rects = VGroup(
            SurroundingRectangle(
                path,
                stroke_width=0.5, fill_opacity=0.1,
            ).match_color(path[0])
            for path in paths_template
        )
        for path, rect in zip(paths_template, back_rects):
            path.add_to_back(rect)
        paths_xyx, paths_yxy = paths_template.replicate(2)
        p1_path_xyx, q1_path_xyx, p2_path_xyx, q2_path_xyx = paths_xyx
        paths_xyx.remove(q1_path_xyx)
        q2_path_xyx.next_to(VGroup(p1_path_xyx, p2_path_xyx), RIGHT, coor_mask=[0, 1, 0])
        p1_path_yxy, q1_path_yxy, p2_path_yxy, q2_path_yxy = paths_yxy
        paths_yxy.remove(p2_path_yxy)
        p1_path_yxy.next_to(VGroup(q1_path_yxy, q2_path_yxy), RIGHT, coor_mask=[0, 1, 0])
        paths_xyx.next_to(claim_xyx, DOWN, aligned_edge=LEFT, buff=0.5)
        paths_yxy.next_to(claim_yxy, DOWN, aligned_edge=LEFT, buff=0.5)

        arrows_xyx = VGroup(
            Arrow(p1_path_xyx.get_right(), q2_path_xyx.get_left()),
            Arrow(p2_path_xyx.get_right(), q2_path_xyx.get_left()),
        )
        mapping_f_xyx = Tex("f")
        mapping_g_xyx = Tex("g?")
        for text in (mapping_f_xyx, mapping_g_xyx):
            text.scale(0.6)
            text.set_color(ALL_PATH_COLOR)
            text.move_to(arrows_xyx)
            text.set_backstroke(width=1)
        cross_xyx = Cross(
            VGroup(mapping_g_xyx, arrows_xyx),
        )
        aux_xyx_group = VGroup(
            paths_xyx, arrows_xyx, mapping_g_xyx, cross_xyx,
        )
        self.play(FadeIn(paths_xyx, shift=DOWN))
        self.wait()
        self.play(
            *[GrowArrow(arrow) for arrow in arrows_xyx],
            Write(mapping_f_xyx, stroke_color=ALL_PATH_COLOR),
        )
        self.wait()
        self.play(
            *[arrow.animate.flip(OUT) for arrow in arrows_xyx],
            FadeTransform(mapping_f_xyx, mapping_g_xyx),
        )
        self.wait()
        self.play(
            ShowCreation(cross_xyx),
        )
        self.wait()

        arrows_yxy = VGroup(
            Arrow(p1_path_yxy.get_right(), q1_path_yxy.get_left()),
            Arrow(p1_path_yxy.get_right(), q2_path_yxy.get_left()),
        )
        mapping_f_yxy = Tex("f?")
        for text in (mapping_f_yxy, ):
            text.scale(0.6)
            text.set_color(ALL_PATH_COLOR)
            text.move_to(arrows_yxy)
            text.set_backstroke(width=1)
        cross_yxy = Cross(
            VGroup(mapping_f_yxy, arrows_yxy),
        )
        aux_yxy_group = VGroup(
            paths_yxy, arrows_yxy, mapping_f_yxy, cross_yxy,
        )
        self.play(
            FadeIn(aux_yxy_group, shift=DOWN),
            FadeIn(ineq_yxy, shift=DOWN),
        )
        self.wait()

        # To make both inequalities hold, p_n has to equal q_n
        conv_chains.generate_target()
        conv_chains.target.arrange(DOWN, aligned_edge=LEFT)
        conv_chains.target.move_to(conv_chains).shift(UP)
        ineq_group = VGroup(ineq_xyx, ineq_yxy)
        ineq_group.generate_target()
        ineq_group.target.arrange(DOWN)
        ineq_group.target.next_to(conv_chains.target, DOWN, buff=1.5)
        self.play(
            MoveToTarget(conv_chains),
            MoveToTarget(ineq_group),
            FadeOut(aux_xyx_group),
            FadeOut(aux_yxy_group),
        )
        self.wait()
        eq_pq_text = VGroup(
            Tex("|P_{n}| = |Q_{n}|", t2c=pnqn_colormap),
            TexText("或者"),
            Tex("p_{n} = q_{n}", t2c=pnqn_colormap),
        )
        eq_pq_text[1].scale(0.6)
        eq_pq_text.arrange(DOWN, buff=0.4)
        eq_pq_text.move_to(ineq_group)
        self.play(
            FadeTransform(ineq_group, eq_pq_text),
        )
        self.wait()


class ThePQConversionRuleInAction(Scene):
    def construct(self):
        random.seed(0)
        # Size settings
        pos_limit = 10
        time_limit = 14
        N = time_limit

        # Make the transformation more interesting by limiting the time of the cut
        def balanced_cut_time_filter(path_chars):
            balanced_flag = is_balanced(path_chars)
            cut_time_flag = path_first_max_or_min_ind(path_chars) >= N // 5
            return balanced_flag and cut_time_flag

        balanced_pool = set(
            filter(
                balanced_cut_time_filter,
                all_limited_walk_strings(N, -pos_limit, pos_limit)
            )
        )
        nonzero_pool = set(
            filter(
                is_nonzero,
                all_limited_walk_strings(N, -pos_limit, pos_limit)
            )
        )
        pool = [
            (x, y)
            for x in balanced_pool
            if (y:=mapping_special_walk_string(x)) in nonzero_pool
        ]
        print(len(pool))

        # Setup background
        bg_rect_x = FullScreenRectangle(fill_color=BALANCED_COLOR, fill_opacity=0.2)
        bg_rect_x.stretch(0.5, 0)
        bg_rect_x.move_to(LEFT_SIDE / 2)
        bg_rect_y = bg_rect_x.copy().set_color(NONZERO_COLOR)
        bg_rect_y.move_to(RIGHT_SIDE / 2)
        bg_rect_line = Line(ORIGIN, UP)
        axes_x = PathAxes(
            N=N,
            pos_range=(-pos_limit, pos_limit, 1),
            time_range=(0, time_limit, 1)
        )
        axes_x.set_color(AXES_COLOR)
        axes_x.set_height(FRAME_HEIGHT - 1)
        axes_x.move_to(bg_rect_x)
        axes_x.to_edge(LEFT)
        axes_y = axes_x.copy()
        axes_y.move_to(bg_rect_y)
        axes_y.to_edge(RIGHT, buff=0.2)
        origin_x = axes_x.get_origin()
        origin_y = axes_y.get_origin()

        tex_colormap = {
            "P_{n}": BALANCED_COLOR,
            "p_{n}": BALANCED_COLOR,
            "Q_{n}": NONZERO_COLOR,
            "q_{n}": NONZERO_COLOR,
            "f": ALL_PATH_COLOR,
            "g": ALL_PATH_COLOR,
        }
        title = Tex("P_{n} \\overset{f}{\\underset{g}{\\rightleftharpoons}} Q_{n}", t2c=tex_colormap)
        title_eq = Tex("p_{n} = q_{n}", t2c=tex_colormap)
        title_eq.next_to(title, DOWN, buff=0.5)
        title_group = VGroup(title, title_eq)
        title_group.to_edge(UP, buff=0.1)
        for tex in title_group:
            tex.set_backstroke(width=1)
        title_rect = SurroundingRectangle(
            title_group,
            stroke_color=bg_rect_line.get_stroke_color(),
            fill_color=BLACK, fill_opacity=0,
            buff=0.2,
        )
        title_group.add_to_back(title_rect)
        bg_rect_line.put_start_and_end_on(title_group.get_bottom(), BOTTOM)
        self.add(bg_rect_x, bg_rect_y, bg_rect_line)
        self.add(axes_x, axes_y)
        self.add(title_group)

        num_samples = 10
        sample_pool = random.sample(pool, num_samples)
        for sample_ind, sample in enumerate(sample_pool):
            # 0. Preparations: showing paths and its shadows
            path_chars_x, path_chars_y = sample
            path_x = RandomWalkPath(path_chars_x, axes=axes_x)
            path_x.set_color(BALANCED_COLOR)
            path_y = RandomWalkPath(path_chars_y, axes=axes_y)
            path_y.set_color(NONZERO_COLOR)
            if sample_ind == 0:
                prev_path_x = path_x
                prev_path_y = path_y
                self.add(path_x, path_y)
            else:
                self.play(
                    FadeIn(path_x),
                    FadeOut(prev_path_x),
                    FadeIn(path_y),
                    FadeOut(prev_path_y),
                )
            self.wait()
            path_x_shadow = path_x.copy()
            path_y_shadow = path_y.copy()
            path_x_shadow.set_opacity(0.2)
            path_y_shadow.set_opacity(0.2)
            path_x.save_state()
            path_y.save_state()
            self.add(path_x_shadow, path_y_shadow)

            # 1. Show how to find split points
            # x-part
            split_ind_x = path_first_max_or_min_ind(path_chars_x)
            split_point_x = path_x.get_point_by_ind(split_ind_x)
            circle_x = Circle().set_color(ALL_PATH_COLOR)
            circle_x.surround(path_x.get_arrow_by_index(0), buff=0.1)
            # y-part
            split_ind_y = path_last_half_cross_ind(path_chars_y)
            split_point_y = path_y.get_point_by_ind(split_ind_y)
            final_pos_coord = path_final_coord(path_chars_y)
            half_final_pos = axes_y.c2p(0, final_pos_coord / 2)
            half_final_pos_line = DashedLine(ORIGIN, path_y.get_width() / 2 * RIGHT)
            half_final_pos_line.match_width(path_y)
            half_final_pos_line.next_to(half_final_pos, RIGHT, buff=0)
            half_final_pos_tex = Tex("\\frac{h}{2}")
            half_final_pos_tex.scale(0.6)
            half_final_pos_tex.next_to(half_final_pos, LEFT)
            for mob in (half_final_pos_line, half_final_pos_tex):
                mob.set_color(ALL_PATH_COLOR)
            # animations
            self.play(
                ShowCreation(circle_x),
                FadeInFromPoint(half_final_pos_line, half_final_pos),
                FadeInFromPoint(half_final_pos_tex, half_final_pos),
            )

            # 2. Show the split points themselves
            split_dot_x = GlowDot(split_point_x)
            split_dot_y = GlowDot(split_point_y)
            for dot in (split_dot_x, split_dot_y):
                dot.set_color(ALL_PATH_COLOR)
            # x-part
            circle_x.generate_target()
            circle_x.target.move_to(split_dot_x).scale(1 / 20.)
            circle_x.target.set_stroke(width=0).set_opacity(0)
            split_dot_x.generate_target()
            split_dot_x.move_to(circle_x).match_width(circle_x).set_opacity(0)
            # y-part
            pass
            # animations
            self.play(
                MoveToTarget(circle_x),
                MoveToTarget(split_dot_x),
                FadeIn(split_dot_y, scale=0.1),
                FadeOutToPoint(half_final_pos_line, split_point_y),
                FadeOutToPoint(half_final_pos_tex, half_final_pos),
            )
            self.remove(circle_x)
            self.wait()

            # 3. Split two parts
            left_part_x, right_part_x = path_x.split_at_ind(split_ind_x, new_path=True)
            left_part_y, right_part_y = path_y.split_at_ind(split_ind_y, new_path=True)
            self.remove(path_x, path_y)
            self.add(left_part_x, right_part_x, left_part_y, right_part_y)
            self.add(split_dot_x, split_dot_y)

            # 4. Flip the first part
            # animations
            self.play(
                FadeOut(split_dot_x),
                FadeOut(split_dot_y),
                left_part_x.animate.flip(),
                left_part_y.animate.flip(),
            )

            # 5. Combine two parts
            # x-part
            left_part_x_inv = left_part_x.get_inverse_path()
            left_part_x_inv.set_color(BALANCED_COLOR)
            left_part_x_inv.put_start_at(origin_x)
            # y-part
            left_part_y_inv = left_part_y.get_inverse_path()
            left_part_y_inv.set_color(NONZERO_COLOR)
            left_part_y_inv.put_start_at(origin_y)
            # animations
            self.play(
                Transform(left_part_x[::-1], left_part_x_inv),
                right_part_x.animate.put_start_at(left_part_x_inv.get_end()),
                Transform(left_part_y[::-1], left_part_y_inv),
                right_part_y.animate.put_start_at(left_part_y_inv.get_end()),
            )
            self.wait()

            # 6. Move to the other side
            # x-part
            group_x = VGroup(left_part_x, right_part_x)
            group_x.generate_target()
            group_x.target.match_color(path_y_shadow)
            group_x.target.move_to(path_y_shadow)
            # y-part
            group_y = VGroup(left_part_y, right_part_y)
            group_y.generate_target()
            group_y.target.match_color(path_x_shadow)
            group_y.target.move_to(path_x_shadow)
            # animations
            self.play(
                MoveToTarget(group_x, path_arc=-PI / 2),
                MoveToTarget(group_y, path_arc=-PI / 2),
            )

            # 7. Tidy up and prepare for the next pair
            path_x.restore()
            path_y.restore()
            self.remove(path_x_shadow, path_y_shadow)
            self.remove(left_part_x, right_part_x, left_part_y, right_part_y)
            self.add(path_x, path_y)
            prev_path_x = path_x
            prev_path_y = path_y
            self.wait()


class ConvolutionPropertyOfAn(Scene):
    def construct(self):
        # Show important definitions and formula up to this point
        wn_title = TexText("路径总数")
        wn_formula = Tex("w_{n} = 2^{2n}")
        wn_group = VGroup(wn_title, wn_formula)
        wn_group.set_color(ALL_PATH_COLOR)
        pn_title = TexText("“正点到家”型 \\\\ 路径数")
        pn_formula = Tex("p_{n} = {2n \\choose n}")
        pn_group = VGroup(pn_title, pn_formula)
        pn_group.set_color(BALANCED_COLOR)
        qn_title = TexText("“不再回家”型 \\\\ 路径数")
        qn_formula = Tex("q_{n} = p_{n}")
        qn_group = VGroup(qn_title, qn_formula)
        qn_group.set_color(NONZERO_COLOR)
        qn_formula[-2:].set_color(BALANCED_COLOR)
        an_title = TexText("“正点到家”型 \\\\ 路径占比")
        an_formula = Tex("a_{n} = \\frac{p_{n}}{w_{n}}")
        an_formula[3:5].set_color(BALANCED_COLOR)
        an_formula[-2:].set_color(ALL_PATH_COLOR)
        an_group = VGroup(an_title, an_formula)
        title_group = VGroup(wn_group, pn_group, qn_group, an_group)
        for group in title_group:
            group.arrange(DOWN)
        title_group.arrange(RIGHT, buff=1.5, aligned_edge=UP)
        title_group.scale(0.8)
        title_group.to_edge(UP)
        for mob in (wn_formula, qn_formula, an_formula):
            mob.next_to(pn_formula, LEFT, coor_mask=[0, 1, 0])

        wpq_colormap = dict()
        strings_to_be_colored = [
            0, 1, 2, 3, 4,
            "k", "n-1", "n-k", "n-2", "n-1", "n",
            "2k", "2(n-k)", "2(n-2)", "2(n-1)", "2n",
        ]
        for s in strings_to_be_colored:
            wpq_colormap[f"p_{{{str(s)}}}"] = BALANCED_COLOR
        for s in strings_to_be_colored:
            wpq_colormap[f"q_{{{str(s)}}}"] = NONZERO_COLOR
        for s in strings_to_be_colored:
            wpq_colormap[f"w_{{{str(s)}}}"] = ALL_PATH_COLOR
        for s in strings_to_be_colored:
            wpq_colormap[f"2^{{{str(s)}}}"] = ALL_PATH_COLOR
        pq_conv_formula = Tex(
            """
            p_{0}q_{n} + p_{1}q_{n-1} + p_{2}q_{n-2} + \\cdots +
            p_{k}q_{n-k} + \\cdots + p_{n}q_{0} = w_{n}
            """,
            t2c=wpq_colormap,
        )
        pq_conv_formula.shift(0.5 * DOWN)
        self.add(title_group, pq_conv_formula)
        self.wait()

        # We got two equalities from bijections between sets WPQ
        for formula in (pq_conv_formula, qn_formula):
            self.play(FlashAround(formula, color=ALL_PATH_COLOR))
            self.wait()

        # p-q convolution is actually p-p convolution
        pp_conv_formula = Tex(
            """
            p_{0}p_{n} + p_{1}p_{n-1} + p_{2}p_{n-2} + \\cdots +
            p_{k}p_{n-k} + \\cdots + p_{n}p_{0} = 2^{2n}
            """,
            t2c=wpq_colormap,
        )
        shift_vec = pq_conv_formula[0].get_center() - pp_conv_formula[0].get_center()
        pp_conv_formula.shift(shift_vec)
        anims = []
        for source, target in zip(pq_conv_formula[:-2], pp_conv_formula[:-3]):
            if source.has_same_shape_as(target):
                anim = ReplacementTransform(source, target)
            else:
                anim = FadeTransform(source, target)
            anims.append(anim)
        anims.append(FadeTransform(pq_conv_formula[-2], pp_conv_formula[-3]))
        anims.append(FadeTransform(pq_conv_formula[-1], pp_conv_formula[-2:]))
        self.play(
            AnimationGroup(*anims, lag_ratio=0.05),
            run_time=3,
        )
        self.wait()

        # Split those extra 2's wisely
        ptwo_frac_formula = Tex(
            """
            \\frac{p_{0}}{2^{0}} \\frac{p_{n}}{2^{2n}} + 
            \\frac{p_{1}}{2^{2}} \\frac{p_{n-1}}{2^{2(n-1)}} + 
            \\frac{p_{2}}{2^{4}} \\frac{p_{n-2}}{2^{2(n-2)}} + 
            \\cdots + 
            \\frac{p_{k}}{2^{2k}} \\frac{p_{n-k}}{2^{2(n-k)}} +
            \\cdots + 
            \\frac{p_{n}}{2^{2n}} \\frac{p_{0}}{2^{0}}
            = 1
            """,
            t2c=wpq_colormap,
        )
        ptwo_frac_formula.set_width(FRAME_WIDTH - 1)
        ptwo_frac_formula.move_to(pq_conv_formula)
        pp, ptwo = pp_conv_formula, ptwo_frac_formula
        self.play(
            AnimationGroup(
                *[
                    ReplacementTransform(pp[s_ind], ptwo[t_ind])
                    for s_ind, t_ind in zip(
                        list(range(39)),
                        [
                            0, 1, 5, 6, 11, 12, 13, 17, 18, 19,
                            20, 29, 30, 31, 35, 36, 37, 38, 47, 48,
                            49, 50, 51, 52, 53, 58, 59, 60, 61, 70,
                            71, 72, 73, 74, 75, 76, 81, 82, 86,
                        ]
                    )
                ],
                lag_ratio=0,
            ),
            AnimationGroup(
                *[
                    GrowFromCenter(ptwo[t_ind])
                    for t_ind in [
                        2, 7, 14, 21, 32, 39, 54, 62, 77, 83,
                    ]
                ],
                lag_ratio=0,
            ),
            AnimationGroup(
                *[
                    FadeTransform(
                        src_mob,
                        VGroup(*[ptwo[t_ind] for t_ind in t_ind_list]),
                        path_arc=-PI / 5,
                    )
                    for src_mob, t_ind_list in zip(
                        pp[39:].replicate(10),
                        [
                            [3, 4], [8, 9, 10],
                            [15, 16], [22, 23, 24, 25, 26, 27, 28],
                            [33, 34], [40, 41, 42, 43, 44, 45, 46],
                            [55, 56, 57], [63, 64, 65, 66, 67, 68, 69],
                            [78, 79, 80], [84, 85],
                        ]
                    )
                ],
                lag_ratio=0,
            ),
            FadeTransform(pp[39:], ptwo[87]),
            run_time=3,
        )
        self.wait()

        # Those 2's go back to w, and the subscripts match up
        pw_frac_formula = Tex(
            """
            \\frac{p_{0}}{w_{0}} \\frac{p_{n}}{w_{n}} + 
            \\frac{p_{1}}{w_{1}} \\frac{p_{n-1}}{w_{n-1}} + 
            \\frac{p_{2}}{w_{2}} \\frac{p_{n-2}}{w_{n-2}} + 
            \\cdots + 
            \\frac{p_{k}}{w_{k}} \\frac{p_{n-k}}{w_{n-k}} +
            \\cdots + 
            \\frac{p_{n}}{w_{n}} \\frac{p_{0}}{w_{0}}
            = 1
            """,
            t2c=wpq_colormap,
        )
        pw_frac_formula.set_width(FRAME_WIDTH - 1)
        pw_frac_formula.move_to(ptwo_frac_formula)
        pw = pw_frac_formula
        ptwo_part_inds = [0, 5, 11, 12, 17, 29, 30, 35, 47, 52, 58, 70, 75, 81, 86, 88]
        pw_part_inds = [0, 5, 10, 11, 16, 25, 26, 31, 40, 45, 50, 59, 64, 69, 74, 76]
        anims_list = []
        for k in range(len(ptwo_part_inds) - 1):
            ptwo_start, ptwo_end = ptwo_part_inds[k], ptwo_part_inds[k + 1]
            pw_start, pw_end = pw_part_inds[k], pw_part_inds[k + 1]
            source = ptwo[ptwo_start:ptwo_end]
            target = pw[pw_start:pw_end]
            if source.has_same_shape_as(target):
                anim = ReplacementTransform(source, target)
            else:
                anim = FadeTransform(source, target)
            anims_list.append(anim)
        trans_anims = AnimationGroup(
            *anims_list,
            lag_ratio=0.05,
            run_time=2,
        )
        self.play(trans_anims)
        self.wait()

        # So they are just a bunch of a's
        pw_copy = pw.copy()
        pw_copy.set_opacity(0.3)
        self.add(pw_copy)
        a_conv_formula = Tex(
            """
            a_{0}a_{n} + a_{1}a_{n-1} + a_{2}a_{n-2} + \\cdots +
            a_{k}a_{n-k} + \\cdots + a_{n}a_{0} = 1
            """,
            t2c=wpq_colormap,
        )
        a_conv_formula.next_to(pw_frac_formula, DOWN)
        a_part_inds = [0, 2, 4, 5, 7, 11, 12, 14, 18, 23, 25, 29, 34, 36, 38, 40]
        anims_list = []
        for k in range(len(pw_part_inds) - 1):
            pw_start, pw_end = pw_part_inds[k], pw_part_inds[k + 1]
            a_start, a_end = a_part_inds[k], a_part_inds[k + 1]
            source = pw[pw_start:pw_end]
            target = a_conv_formula[a_start:a_end]
            if source.has_same_shape_as(target):
                anim = ReplacementTransform(source, target)
            else:
                anim = FadeTransform(source, target)
            anims_list.append(anim)
        trans_anims = AnimationGroup(
            *anims_list,
            lag_ratio=0.1,
            run_time=3,
        )
        self.play(trans_anims)
        self.wait()

        # Now we have both the formula and the convolution property of an
        a_conv_title = TexText("$\\{ a_{n} \\}$的卷积性质")
        a_conv_title.next_to(a_conv_formula, UP)
        a_conv_title.set_color(ALL_PATH_COLOR)
        self.play(
            FadeOut(pw_copy, shift=0.5 * UP),
            FadeIn(a_conv_title, shift=0.5 * UP),
        )
        self.wait()

        title_group.remove(an_formula, an_title)
        self.add(an_title, an_formula)
        alt_an_formula = Tex(
            """
            &a_{n} = \\frac{1}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{6} \\cdots \\frac{2n-1}{2n}  \\\\
            &(a_{0}=1)
            """
        )
        alt_an_formula.shift(1.5 * UP)
        alt_an_title = TexText("$\\{ a_{n} \\}$的表达式")
        alt_an_title.next_to(alt_an_formula, UP)
        alt_an_title.set_color(ALL_PATH_COLOR)
        self.play(
            FadeOut(title_group, shift=UP),
            FadeTransform(an_title, alt_an_title),
            FadeTransform(an_formula, alt_an_formula),
        )
        self.wait()

        # Where's Wallis formula and the circle?
        wallis_prod = Tex(
            """
            \\frac{\\pi}{2} = \\frac{2}{1} \\cdot \\frac{2}{3} \\cdot \\frac{4}{3} \\cdots
            """
        )
        circle = Circle()
        circle.surround(wallis_prod[0], buff=0.1)
        wallis_group = VGroup(wallis_prod, circle)
        wallis_group.set_color(WALLIS_COLOR)
        wallis_group.set_opacity(0.2)
        wallis_group.next_to(RIGHT_SIDE).shift(2 * UP)
        circle.set_fill(opacity=0)
        self.add(wallis_group)
        tilt_angle = PI / 5
        tilt_center = RIGHT_SIDE
        self.play(
            wallis_group.animate.rotate(tilt_angle, about_point=tilt_center).shift(0.5 * LEFT),
            run_time=5,
        )
        self.wait()

        # Centralize convolution property to match up with the next scene
        fadeout_mobs = VGroup(
            alt_an_formula, alt_an_title, wallis_group,
        )
        shift_vec = -a_conv_formula.get_center()
        self.play(
            FadeOut(fadeout_mobs, shift=shift_vec),
            a_conv_title.animate.shift(shift_vec),
            a_conv_formula.animate.shift(shift_vec),
        )
        self.wait()


class ItsGeometryTime(Scene):
    def construct(self):
        # Setup to match up the previous scene
        a_conv_formula = Tex(
            """
            a_{0}a_{n} + a_{1}a_{n-1} + a_{2}a_{n-2} + \\cdots +
            a_{k}a_{n-k} + \\cdots + a_{n}a_{0} = 1
            """,
        )
        a_conv_title = TexText("$\\{ a_{n} \\}$的卷积性质")
        a_conv_title.set_color(ALL_PATH_COLOR)
        a_conv_title.next_to(a_conv_formula, UP)
        self.add(a_conv_formula, a_conv_title)

        # This is one formula of the general case
        sub_pair = (
            (1, 1, 3, 1), (6, 1, 8, 3), (13, 1, 15, 3),
            (24, 1, 26, 3), (35, 1, 37, 1),
        )
        sub_sum_note = TexText("角标和为$n$的两项乘积之和$=1$")
        sub_sum_note.set_color(ALL_PATH_COLOR)
        sub_sum_note.next_to(a_conv_formula, DOWN, buff=1)
        acf = a_conv_formula
        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        Indicate(acf[s:s + sl], color=BLUE),
                        Indicate(acf[e:e + el], color=BLUE),
                    )
                    for s, sl, e, el in sub_pair
                ],
                lag_ratio=0.1,
                run_time=3,
            ),
            FadeIn(sub_sum_note, shift=DOWN, run_time=1),
        )
        self.wait()
        self.play(
            VGroup(a_conv_formula, a_conv_title, sub_sum_note).animate.to_edge(UP)
        )
        self.wait()

        # and infinite formula of specific cases
        num_samples = 6
        left_ns = VGroup(
            Tex(f"n = {n}:")
            for n in range(num_samples)
        )
        right_sums = VGroup(
            Tex(
                " + ".join([
                    f"a_{{{k}}} a_{{{n-k}}}"
                    for k in range(n + 1)
                ]) + " = 1"
            )
            for n in range(num_samples)
        )
        for mobs in (left_ns, right_sums):
            mobs.add(Tex("\\cdots"))
            mobs.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        left_ns.set_color(ALL_PATH_COLOR)
        right_sums.next_to(left_ns, RIGHT, buff=0.3)
        for left_mob, right_mob in zip(left_ns, right_sums):
            right_mob.next_to(left_mob, RIGHT, coor_mask=[0, 1, 0], buff=0.5)
        sum_group = VGroup(left_ns, right_sums)
        sum_group.next_to(sub_sum_note, DOWN, buff=0.5)

        self.play(Write(left_ns[0]), Write(right_sums[0]), run_time=1)
        self.wait()
        self.play(Write(left_ns[1]), Write(right_sums[1]), run_time=1)
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    FadeIn(VGroup(left_ns[k], right_sums[k]), shift=DOWN)
                    for k in range(2, len(left_ns))
                ],
                lag_ratio=0.2,
                run_time=3,
            )
        )
        self.wait()
        sum_group.add(sub_sum_note)
        sum_group.generate_target()
        sum_group.target.to_edge(UP)
        sum_group.target[0].set_opacity(0)
        sum_group.target[-1].set_opacity(0)
        shift_vec = sum_group.target.get_center() - sum_group.get_center()
        self.play(
            MoveToTarget(sum_group),
            FadeOut(VGroup(a_conv_formula, a_conv_title), shift=shift_vec),
        )
        self.remove(sum_group)
        self.add(right_sums)
        self.wait()

        # Each term can be treated as the area of a rectangle
        highlight_colors = WALLIS_RECTS_COLORS
        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        AnimationGroup(
                            FlashAround(mobs[5 * k:5 * k + 4], stroke_width=5, color=color),
                            ApplyMethod(mobs[5 * k:5 * k + 4].set_color, color),
                        )
                        for k in range(len(mobs) // 5)
                    )
                    for mobs, color in zip(right_sums[:-1], highlight_colors)
                ],
                lag_ratio=0.1,
            ),
            run_time=3,
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        FlashAround(mobs[-2:], stroke_width=5, color=color),
                        ApplyMethod(mobs[-1].set_color, color),
                    )
                    for mobs, color in zip(right_sums[:-1], highlight_colors)
                ],
                lag_ratio=0.1,
            ),
            run_time=3,
        )
        self.wait()

        right_sums.generate_target()
        base_width = right_sums.target[0].get_width()
        factor = 1 / 2
        for ind, (mob, color) in enumerate(zip(right_sums.target[:-1], highlight_colors)):
            target_width = base_width * np.sqrt((ind + factor) / factor)
            mob.set_width(target_width)
            mob.set_color(color)
        right_sums.target.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        right_sums.target.center().to_edge(RIGHT)
        self.play(MoveToTarget(right_sums))
        self.wait()

        # Building up the rectangle pattern layer-by-layer
        wallis_rects = WallisRectangles(N=6, colors=WALLIS_RECTS_COLORS)
        wallis_rects.set_width(FRAME_WIDTH * 0.4)
        wallis_rects.next_to(right_sums, LEFT, buff=0.5)
        wallis_rects.to_edge(UP, buff=0.8)
        length_labels = wallis_rects.get_length_labels()
        area_labels = wallis_rects.get_area_labels()

        # layer 0
        rect_a0a0 = wallis_rects.get_layer(0)
        horiz_a0 = length_labels[0][0]
        vert_a0 = length_labels[1][0]
        area_a0a0 = area_labels[0][0]
        layer_0_color = WALLIS_RECTS_COLORS[0]
        self.play(
            FlashAround(right_sums[0], color=layer_0_color),
        )
        self.wait()
        self.play(
            GrowFromCenter(rect_a0a0),
            Write(horiz_a0),
            Write(vert_a0),
        )
        self.wait()
        self.play(
            TransformFromCopy(horiz_a0, area_a0a0[:2]),
            TransformFromCopy(vert_a0, area_a0a0[2:]),
        )
        self.wait()

        # layer 1
        rect_a0a1, rect_a1a0 = wallis_rects.get_layer(1)
        horiz_a1 = length_labels[0][1]
        vert_a1 = length_labels[1][1]
        area_a0a1, area_a1a0 = area_labels[1]
        layer_1_color = WALLIS_RECTS_COLORS[1]
        self.play(
            FlashAround(right_sums[1], color=layer_1_color),
        )
        self.wait()
        down_arrow = Arrow(ORIGIN, DOWN)
        down_arrow.next_to(rect_a0a0, UP)
        left_arrow = Arrow(ORIGIN, LEFT)
        left_arrow.next_to(rect_a0a0, RIGHT)
        VGroup(down_arrow, left_arrow).set_color(layer_0_color)
        self.play(
            FadeIn(down_arrow, shift=0.5 * DOWN),
            FadeIn(left_arrow, shift=0.5 * LEFT),
        )
        self.wait()
        rect_a0a1.generate_target()
        rect_a0a1.set_height(1e-2, stretch=True, about_edge=DOWN).set_opacity(0)
        rect_a1a0.generate_target()
        rect_a1a0.set_width(1e-2, stretch=True, about_edge=LEFT).set_opacity(0)
        self.play(
            MoveToTarget(rect_a0a1),
            MoveToTarget(rect_a1a0),
            down_arrow.animate.rotate(PI).next_to(rect_a0a1.target, UP).set_color(layer_1_color),
            left_arrow.animate.rotate(PI).next_to(rect_a1a0.target, RIGHT).set_color(layer_1_color),
            Write(horiz_a1),
            Write(vert_a1),
        )
        self.wait()
        self.play(
            TransformFromCopy(horiz_a0, area_a0a1[:2]),
            TransformFromCopy(vert_a1, area_a0a1[2:]),
            TransformFromCopy(horiz_a1, area_a1a0[:2]),
            TransformFromCopy(vert_a0, area_a1a0[2:]),
            FadeOut(down_arrow),
            FadeOut(left_arrow),
        )
        self.wait()

        # layer 2
        rect_a0a2, rect_a1a1, rect_a2a0 = wallis_rects.get_layer(2)
        horiz_a2 = length_labels[0][2]
        vert_a2 = length_labels[1][2]
        area_a0a2, area_a1a1, area_a2a0 = area_labels[2]
        layer_2_color = WALLIS_RECTS_COLORS[2]
        self.play(
            FlashAround(right_sums[2], color=layer_2_color),
        )
        self.wait()
        self.play(
            GrowFromPoint(rect_a1a1, rect_a1a1.get_corner(DL)),
            TransformFromCopy(horiz_a1, area_a1a1[:2]),
            TransformFromCopy(vert_a1, area_a1a1[2:]),
        )
        self.wait()
        rect_a0a2.generate_target()
        rect_a0a2.set_height(1e-2, stretch=True, about_edge=DOWN).set_opacity(0)
        rect_a2a0.generate_target()
        rect_a2a0.set_width(1e-2, stretch=True, about_edge=LEFT).set_opacity(0)
        self.play(
            MoveToTarget(rect_a0a2),
            MoveToTarget(rect_a2a0),
            Write(horiz_a2),
            Write(vert_a2),
        )
        self.wait()
        self.play(
            TransformFromCopy(horiz_a0, area_a0a2[:2]),
            TransformFromCopy(vert_a2, area_a0a2[2:]),
            TransformFromCopy(horiz_a2, area_a2a0[:2]),
            TransformFromCopy(vert_a0, area_a2a0[2:]),
        )
        self.wait()

        # layer 3, 4, 5
        for layer in (3, 4, 5):
            self.play(
                FlashAround(right_sums[layer], color=WALLIS_RECTS_COLORS[layer]),
                FadeIn(wallis_rects.get_layer(layer)),
                FadeIn(area_labels[layer]),
                Write(length_labels[0][layer]),
                Write(length_labels[1][layer])
            )
            self.add(area_labels[layer])
            self.wait()

        # Show the total area is now 6
        num_layer_note = TexText("矩形层数：")
        num_layer_6 = TexText("6")
        num_layer_6.next_to(num_layer_note, RIGHT)
        area_sum_note = TexText("图案总面积$=$")
        area_sum_note.next_to(num_layer_note, DOWN, aligned_edge=LEFT, buff=0.5)
        area_sum_6 = num_layer_6.copy()
        area_sum_6.next_to(area_sum_note, RIGHT)
        counting_group = VGroup(num_layer_note, num_layer_6, area_sum_note, area_sum_6)
        counting_group.to_corner(UR, buff=1)
        self.play(
            FadeOut(area_labels),
            FadeTransform(right_sums, counting_group),
        )
        self.wait()

        # Then change it to 10 - add 4 more layers
        unit_size = wallis_rects.get_unit_size()
        corner = wallis_rects.get_rects_corner()
        wallis_rects_10_alt = WallisRectangles(
            N=10, colors=WALLIS_RECTS_COLORS,
        )
        wallis_rects_10_alt.set_unit_size(unit_size)
        wallis_rects_10_alt.put_corner_at(corner)
        length_labels_10_alt = wallis_rects_10_alt.get_length_labels()
        num_layer_10 = TexText("10")
        num_layer_10.next_to(num_layer_note, RIGHT)
        area_sum_10 = num_layer_10.copy()
        area_sum_10.next_to(area_sum_note, RIGHT)
        self.remove(length_labels)
        self.add(length_labels_10_alt[:6])
        self.play(
            AnimationGroup(
                *[
                    ReplacementTransform(
                        wallis_rects.get_layer(layer),
                        wallis_rects_10_alt.get_layer(layer),
                    )
                    for layer in range(0, 6)
                ],
                run_time=1,
            ),
            AnimationGroup(
                *[
                    FadeIn(wallis_rects_10_alt.get_layer(layer))
                    for layer in range(6, 10)
                ],
                lag_ratio=0.3,
                run_time=2,
            ),
            AnimationGroup(
                *[
                    AnimationGroup(
                        Write(length_labels_10_alt[0][layer]),
                        Write(length_labels_10_alt[1][layer])
                    )
                    for layer in range(6, 10)
                ],
                lag_ratio=0.3,
                run_time=2,
            ),
            FadeOut(num_layer_6, shift=0.5 * UP, run_time=2),
            FadeOut(area_sum_6, shift=0.5 * UP, run_time=2),
            FadeIn(num_layer_10, shift=0.5 * UP, run_time=2),
            FadeIn(area_sum_10, shift=0.5 * UP, run_time=2),
        )
        self.add(wallis_rects_10_alt)
        self.wait()

        wallis_rects_10 = wallis_rects_10_alt.copy()
        wallis_rects_10.set_unit_size(unit_size * 0.6)
        wallis_rects_10.put_corner_at(corner)
        length_labels_10 = wallis_rects_10.get_length_labels()
        self.play(
            ReplacementTransform(wallis_rects_10_alt, wallis_rects_10),
            ReplacementTransform(length_labels_10_alt, length_labels_10),
            run_time=2,
        )
        self.wait()

        # Grow layer-by-layer, from 10 up to 20
        curr_wallis_rects = wallis_rects_10
        curr_length_labels = length_labels_10
        curr_num_layer = num_layer_10
        curr_area_sum = area_sum_10
        unit_size = wallis_rects_10.get_unit_size()
        corner = wallis_rects_10.get_rects_corner()

        curr_layers = 10
        target_layers = 20
        for N in range(curr_layers, target_layers):
            new_wallis_rects = WallisRectangles(
                N=N + 1, colors=WALLIS_RECTS_COLORS,
            )
            new_wallis_rects.set_unit_size(unit_size)
            new_wallis_rects.put_corner_at(corner)
            new_length_labels = new_wallis_rects.get_length_labels()
            new_num_layer = Tex(str(N + 1))
            new_num_layer.next_to(num_layer_note, RIGHT)
            new_area_sum = new_num_layer.copy()
            new_area_sum.next_to(area_sum_note, RIGHT)
            self.play(
                AnimationGroup(
                    ReplacementTransform(
                        curr_wallis_rects.get_layer(layer),
                        new_wallis_rects.get_layer(layer),
                    )
                    for layer in range(0, N)
                ),
                AssembleLayer(new_wallis_rects, N),
                AnimationGroup(
                    FadeIn(new_length_labels[0][N]),
                    FadeIn(new_length_labels[1][N]),
                ),
                FadeOut(curr_num_layer, shift=0.5 * UP),
                FadeOut(curr_area_sum, shift=0.5 * UP),
                FadeIn(new_num_layer, shift=0.5 * UP),
                FadeIn(new_area_sum, shift=0.5 * UP),
            )
            self.remove(curr_length_labels, curr_wallis_rects)
            self.add(new_length_labels, new_wallis_rects)
            self.wait()
            curr_wallis_rects = new_wallis_rects
            curr_length_labels = new_length_labels
            curr_num_layer = new_num_layer
            curr_area_sum = new_area_sum
        last_wallis_rects = new_wallis_rects
        last_length_labels = new_length_labels
        last_num_layer = new_num_layer
        last_area_sum = new_area_sum

        # Two ways to calculate area - precise and approximate
        area_approx = Tex(
            "\\frac{\\pi}{4} \\left( \\text{``半径''} \\right)^2 \\approx n"
        )
        area_approx.set_color(WALLIS_COLOR)
        area_approx.to_edge(RIGHT, buff=1.5).shift(1.8 * DOWN)
        self.play(
            Write(area_approx[-1]),
            run_time=1,
        )
        self.wait()
        self.play(
            FadeIn(area_approx[:-1], shift=0.5 * DOWN),
            run_time=1,
        )
        self.wait()
        area_limit = Tex(
            "\\lim_{n \\to \\infty} {\\frac{\\pi \\left( \\text{``半径''} \\right)^2}{4 n}} = 1"
        )
        area_limit.match_color(area_approx)
        area_limit.move_to(area_approx)
        self.play(
            GrowFromCenter(area_limit[:6]),
            ReplacementTransform(area_approx[0], area_limit[6]),
            ReplacementTransform(area_approx[3:10], area_limit[7:14]),
            ReplacementTransform(area_approx[1:3], area_limit[14:16]),
            ReplacementTransform(area_approx[11].copy(), area_limit[16], path_arc=-PI / 2),
            FadeTransform(area_approx[10], area_limit[17]),
            FadeTransform(area_approx[11], area_limit[18]),
            run_time=3,
        )
        self.add(area_limit)
        self.wait()

        radius_text = area_limit[7:13]
        radius_text.generate_target()
        radius_text.target.set_color_by_gradient(*WALLIS_RECTS_COLORS)
        radius_text.target.scale(1.05)
        self.play(
            MoveToTarget(radius_text, rate_func=there_and_back_with_pause)
        )
        self.wait()

        # Introduce quantity s_n as the 'radius'
        area_limit_simp = Tex(
            "\\lim_{n \\to \\infty} {\\frac{\\pi {s}_{n}^{2}}{4 n}} = 1"
        )
        area_limit_simp.match_color(area_limit)
        area_limit_simp.move_to(area_limit)
        sn_note = Tex("s_{n} = \\text{``半径''}")
        sn_note.next_to(radius_text[:-1], UP, aligned_edge=RIGHT, buff=0.8)
        sn_note.set_color(WALLIS_COLOR)
        sector = last_wallis_rects.get_cover_sector()
        self.play(
            ReplacementTransform(area_limit[:8], area_limit_simp[:8]),
            ReplacementTransform(area_limit[12], area_limit_simp[9]),
            ReplacementTransform(area_limit[13], area_limit_simp[8]),
            ReplacementTransform(area_limit[14:], area_limit_simp[10:]),
            ReplacementTransform(area_limit[8:12], sn_note[3:]),
            Write(sn_note[:3], stroke_color=WALLIS_COLOR),
            FadeIn(sector),
            run_time=2,
        )
        self.wait()

        radius_rect = Rectangle()
        radius_rect.set_color(WALLIS_COLOR).rotate(PI)
        radius_rect.surround(last_length_labels[1])
        radius_rect.insert_n_curves(100)
        self.play(
            ShowPassingFlash(radius_rect, time_width=1),
            run_time=3,
        )
        self.wait()

        # Give the expression of s_n
        sn_sum_1 = Tex("s_{n} = \\{ a_{n} \\} \\text{的前} n \\text{项和}")
        sn_sum_2 = Tex("s_{n} = a_0 + a_1 + \\cdots + a_{n-1}")
        sn_sum_2.scale(0.8)
        for mob in (sn_sum_1, sn_sum_2):
            shift_vec = sn_note[0].get_center() - mob[0].get_center()
            mob.shift(shift_vec)
            mob.match_color(sn_note)
        self.play(
            ReplacementTransform(sn_note[:3], sn_sum_1[:3]),
            FadeTransform(sn_note[3:], sn_sum_1[3:]),
        )
        self.remove(sn_note)
        self.add(sn_sum_1)
        self.wait()
        self.play(
            ReplacementTransform(sn_sum_1[:3], sn_sum_2[:3]),
            FadeTransform(sn_sum_1[3:], sn_sum_2[3:]),
        )
        self.remove(sn_sum_1)
        self.add(sn_sum_2)
        self.wait()
        self.play(
            Indicate(sn_sum_2[1]),
            Indicate(sn_sum_2[-3:]),
        )
        self.wait()


class RelationBetweenAnAndSn(Scene):
    def construct(self):
        # We already had the formula for a_n, s_n and their relations
        an_def = VGroup(
            TexText("$\\{ a_{n} \\}$计算式"),
            Tex(
                """
                &a_{n} = \\frac{1}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{6} \\cdots \\frac{2n-3}{2n-2} \\cdot \\frac{2n-1}{2n}  \\\\
                &(a_{0}=1)
                """
            ),
        )
        an_recur = VGroup(
            TexText("$\\{ a_{n} \\}$递推式"),
            Tex("a_{n+1} = a_{n} \\cdot \\frac{2n+1}{2n+2}"),
        )
        sn_def = VGroup(
            TexText("$\\{ s_{n} \\}$定义式"),
            Tex(
                """
                s_{n} &= \\sum_{i=0}^{n-1}{a_{i}} \\\\
                &= a_0 + a_1 + \\cdots + a_{n-1}
                """
            ),
        )
        sn_recur = VGroup(
            TexText("$\\{ s_{n} \\}$递推式"),
            Tex("s_{n+1} = s_{n} + a_{n}"),
        )
        prev_result = VGroup(an_def, sn_def, an_recur, sn_recur)
        VGroup(an_def, an_recur).set_color(AN_COLOR)
        VGroup(sn_def, sn_recur).set_color(WALLIS_COLOR)
        for group in prev_result:
            group.arrange(DOWN)
            group.scale(0.6)
        prev_result.arrange_in_grid(
            2, 2, aligned_edge=UP,
            h_buff=0.5, v_buff=0.5,
        )
        prev_result.to_edge(RIGHT)
        self.add(prev_result)

        # Checking small n values
        grid_texts = [
            "2", "\\cdot", "n", "\\cdot", "a_{n}", "=", "s_{n}",
            "2", "\\cdot", "0", "\\cdot", "1", "=", "0",
            "2", "\\cdot", "1", "\\cdot", "\\frac{1}{2}", "=", "1",
            "2", "\\cdot", "2", "\\cdot", "\\frac{3}{8}", "=", "\\frac{3}{2}",
            "2", "\\cdot", "3", "\\cdot", "\\frac{5}{16}", "=", "\\frac{15}{8}",
            "2", "\\cdot", "4", "\\cdot", "\\frac{35}{128}", "=", "\\frac{35}{16}",
            "2", "\\cdot", "5", "\\cdot", "\\frac{315}{1280}", "=", "\\frac{315}{128}",
        ]
        grid = VGroup(Tex(text) for text in grid_texts)
        grid.arrange_in_grid(7, 7, h_buff=-0.1, v_buff=0.5)
        grid.set_height(FRAME_HEIGHT - 2)
        for mob in grid[:7]:
            mob.scale(1.5)
        for ind, mob in enumerate(grid):
            if ind % 7 == 0:
                mob.set_color(YELLOW)
            elif ind % 7 == 2:
                mob.set_color(YELLOW)
            elif ind % 7 == 4:
                mob.set_color(AN_COLOR)
            elif ind % 7 == 6:
                mob.set_color(WALLIS_COLOR)
        grid.to_corner(UL)
        sep_line = Line(LEFT, RIGHT)
        sep_line_ref_mob = VGroup(grid[2], grid[6])
        sep_line.set_width(sep_line_ref_mob.get_width() * 1.2)
        sep_line.next_to(sep_line_ref_mob, DOWN, buff=0.3)

        an_unsimp = VGroup(
            Tex("1"),
            Tex("\\frac{1}{2}"),
            Tex("\\frac{1}{2} \\cdot \\frac{3}{4}"),
            Tex("\\frac{1}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{6}"),
            Tex("\\frac{1}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{6} \\cdot \\frac{7}{8}"),
            Tex("\\frac{1}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{6} \\cdot \\frac{7}{8} \\cdot \\frac{9}{10}"),
        )
        for ind, (unsimp, simp) in enumerate(zip(an_unsimp, grid[11::7])):
            unsimp.match_color(simp)
            unsimp.match_height(simp)
            unsimp.move_to(simp)
            if ind != 0:
                unsimp.next_to(an_unsimp[0], DOWN, aligned_edge=LEFT, coor_mask=[1, 0, 0])

        self.add(grid[4], sep_line, grid[2::7], an_unsimp)
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    FlashAround(group, color=group.get_color())
                    for group in (an_recur, sn_def, sn_recur)
                ],
                lag_ratio=0.3,
                run_time=3,
            )
        )
        self.wait()
        prev_result.generate_target()
        prev_result.target.next_to(grid, RIGHT, aligned_edge=DOWN, coor_mask=[0, 1, 0])
        prev_rect = SurroundingRectangle(prev_result.target, buff=0.3)
        prev_rect.set_fill(opacity=0.2)
        prev_rect.set_color(AXES_COLOR)
        self.play(
            DrawBorderThenFill(prev_rect),
            MoveToTarget(prev_result),
            AnimationGroup(
                *[
                    FadeTransform(unsimp, simp)
                    for unsimp, simp in zip(an_unsimp, grid[11::7])
                ],
                lag_ratio=0.05,
                run_time=2,
            ),
            FadeIn(grid[6:14:7]),
        )
        self.wait()
        for k in range(len(grid) // 7 - 2):
            an_text = grid[11:12 + 7 * k:7]
            sn_text = grid[20 + 7 * k]
            if k == 0:
                an_rect = SurroundingRectangle(an_text)
                arrow = Arrow(
                    an_rect.get_corner(DR), sn_text.get_left(),
                    tip_width_ratio=3,
                )
                VGroup(an_rect, arrow).set_color(WALLIS_COLOR)
                self.play(
                    ShowCreation(an_rect),
                    GrowArrow(arrow),
                    run_time=0.5,
                )
            else:
                new_an_rect = SurroundingRectangle(an_text)
                new_arrow = Arrow(
                    new_an_rect.get_corner(DR), sn_text.get_left(),
                    tip_width_ratio=3,
                )
                VGroup(new_an_rect, new_arrow).set_color(WALLIS_COLOR)
                self.play(
                    Transform(an_rect, new_an_rect),
                    Transform(arrow, new_arrow),
                    run_time=0.5,
                )
            self.play(
                FadeTransform(an_rect.copy().set_opacity(0), sn_text),
            )
            self.wait(0.5)
        self.play(FadeOut(VGroup(an_rect, arrow)))
        self.wait()

        # Time to 'guess' a formula for s_n by looking at the values
        self.play(
            AnimationGroup(
                FlashAround(grid[-10:-7]),
                FlashAround(grid[-3:]),
                lag_ratio=0.2,
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    FadeIn(grid[offset::7])
                    for offset in (0, 1, 3, 5)
                ],
                lag_ratio=0.2,
            ),
            run_time=2,
        )
        self.add(grid)
        self.wait()

        sep_line.generate_target()
        sep_line.target.shift(0.5 * LEFT)
        grid.generate_target()
        for k in range(len(grid) // 7):
            source_set = grid[7 * k:7 * k + 7]
            if k != 0:
                target_set = grid.target[7 * k:7 * k + 7]
            else:
                target_set = VGroup(grid.target[i] for i in (0, 2, 4, 5, 6))
                grid.target[1:5:2].set_opacity(0)
            target_set.arrange(RIGHT, coor_mask=[1, 0, 0])
            target_set.move_to(source_set)
            target_set.next_to(sep_line.target, DOWN, coor_mask=[1, 0, 0])
        relation_unproven = Tex("s_{n} \\overset{?}{=} 2 n a_{n}")
        colors = [
            WALLIS_COLOR, WALLIS_COLOR, WHITE, WHITE,
            YELLOW, YELLOW, AN_COLOR, AN_COLOR,
        ]
        for mob, color in zip(relation_unproven, colors):
            mob.set_color(color)
        relation_unproven.scale(1.5)
        relation_unproven.next_to(prev_result, UP, buff=1.5)

        self.play(
            MoveToTarget(sep_line, run_time=2),
            MoveToTarget(grid, run_time=2),
            AnimationGroup(
                ReplacementTransform(grid[6].copy(), relation_unproven[:2]),
                FadeTransform(grid[5].copy(), relation_unproven[2]),
                ReplacementTransform(grid[5].copy()[0], relation_unproven[3]),
                ReplacementTransform(grid[0].copy()[0], relation_unproven[4]),
                ReplacementTransform(grid[2].copy()[0], relation_unproven[5]),
                ReplacementTransform(grid[4].copy(), relation_unproven[6:]),
                lag_ratio=0.02,
                run_time=2,
            )
        )
        self.wait()
        # Proving s_n = 2n * a_n
        proof = VGroup(
            TexText("证明:"),
            TexText("(1) 基础情况"),
            TexText("$n=0$时, $s_{0} = {2 \\times 0} \\times a_{0} = 0$, 成立."),
            TexText("(2) 一般情况"),
            TexText("设$n=k$时, $s_{k} = {2k} a_{k}$成立, 则有:"),
            Tex(
                """
                s_{k+1} &= s_{k} + a_{k} &&\\left( \\{ s_{n} \\}\\text{递推式} \\right) \\\\
                &= (2k+1) \\cdot a_{k} &&\\left( \\text{归纳假设} s_{k} = {2k} a_{k} \\right) \\\\
                &= (2k+1) \\cdot \\frac{2k+2}{2k+1} a_{k+1} &&\\left( \\{ a_{n} \\}\\text{递推式} \\right) \\\\
                &= 2(k+1) a_{k+1} &&
                """
            ),
            TexText("即$n=k+1$时, $s_{k+1} = {2(k+1)} a_{k+1}$同样成立."),
            TexText("由数学归纳法, $s_{n} = {2n} a_{n}$成立."),
        )
        proof.arrange(DOWN, aligned_edge=LEFT)
        proof.scale(0.65)
        proof.next_to(grid, RIGHT, aligned_edge=UP, buff=1.5)
        proof[1:].shift(0.2 * DOWN)
        proof[3:].shift(0.2 * DOWN)
        proof[-1:].shift(0.2 * DOWN)
        VGroup(proof[0], proof[1], proof[3], proof[-1]).set_color(YELLOW)
        self.play(
            FadeOut(relation_unproven, run_time=1),
            FadeOut(VGroup(prev_rect, prev_result), run_time=1),
            FadeIn(proof, run_time=2),
        )
        self.wait()
        self.play(
            FlashAround(proof[1:3], color=YELLOW),
            run_time=2,
        )
        self.wait()
        self.play(
            FlashAround(proof[3:-1], color=YELLOW),
            run_time=2,
        )
        self.wait()

        # Back to the relation and expand it
        relation_proven = Tex("s_{n} = 2n a_{n}")
        relation_proven[:2].set_color(WALLIS_COLOR)
        relation_proven[3:5].set_color(YELLOW)
        relation_proven[5:].set_color(AN_COLOR)
        relation_proven.match_width(relation_unproven)
        relation_proven.move_to(relation_unproven)
        snan_relation = Tex("s_{n} = 2n a_{n}")
        snan_relation.match_height(sn_recur[-1])
        snan_relation.set_color(WALLIS_COLOR)
        snan_relation.next_to(sn_recur, DOWN, buff=0.15)
        sn_recur.add(snan_relation)
        self.play(
            FadeOut(proof, run_time=1),
            FadeIn(relation_proven, run_time=2),
            FadeIn(VGroup(prev_rect, prev_result), run_time=2),
        )
        self.wait()

        snan_expansion_1 = Tex(
            """
            s_{n} = 2n \\cdot
            \\frac{1}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{6} \\cdots \\frac{2n-3}{2n-2} \\cdot \\frac{2n-1}{2n}
            """
        )
        snan_expansion_1[:2].set_color(WALLIS_COLOR)
        snan_expansion_1[3:5].set_color(YELLOW)
        snan_expansion_1[6:].set_color(AN_COLOR)
        snan_expansion_1.move_to(relation_proven)
        snan_expansion_2 = Tex(
            "s_{n} = \\frac{3}{2} \\cdot \\frac{5}{4} \\cdots \\frac{2n-1}{2n-2}"
        )
        snan_expansion_2.set_color(WALLIS_COLOR)
        snan_expansion_2.move_to(relation_proven)
        sae1, sae2 = snan_expansion_1, snan_expansion_2
        self.play(
            ReplacementTransform(relation_proven[:5], sae1[:5]),
            FadeTransform(relation_proven[5:], sae1[5:]),
        )
        self.wait()
        self.play(
            FlashAround(sae1[3:6]),
            sae1[3:6].animate.set_color(AXES_COLOR).set_opacity(0.5),
            FlashAround(sae1[-2:]),
            sae1[-3:].animate.set_color(AXES_COLOR).set_opacity(0.5),
        )
        self.wait()
        self.play(
            FlashAround(sae1[6]),
            sae1[6].animate.set_color(AXES_COLOR).set_opacity(0.5),
        )
        self.wait()

        # Now we complete the formula
        sn_calc = VGroup(
            TexText("$\\{ s_{n} \\}$计算式"),
            Tex(
                """
                &s_{n} = \\frac{3}{2} \\cdot \\frac{5}{4} \\cdots \\frac{2n-1}{2n-2} \\\\
                &(s_{0}=0, \\, s_{1}=1)
                """
            ),
        )
        sn_calc.arrange(DOWN)
        sn_calc.set_color(WALLIS_COLOR)
        sn_calc.scale(0.6)
        sn_calc.next_to(sn_def.get_top(), DOWN, buff=0)
        self.play(
            FadeOut(sae1[3:7], shift=0.5 * UP),
            FadeOut(sae1[34:], shift=0.5 * DOWN),
            ReplacementTransform(sae1[:3], sae2[:3]),
            ReplacementTransform(sae1[10], sae2[3], path_arc=PI / 4),
            ReplacementTransform(sae1[7:10], sae2[4:7]),
            ReplacementTransform(sae1[14], sae2[7], path_arc=PI / 4),
            ReplacementTransform(sae1[11:13], sae2[8:10]),
            ReplacementTransform(VGroup(sae1[13], sae1[15:20], sae1[20:24]), sae2[10:13]),
            ReplacementTransform(sae1[30:34], sae2[13:17], path_arc=PI / 4),
            ReplacementTransform(sae1[24:29], sae2[17:]),
            FadeOut(sae1[29], scale=0),
            FadeTransform(sn_def, sn_calc),
            run_time=2,
        )
        prev_result = VGroup(an_def, sn_calc, an_recur, sn_recur)
        self.remove(sae1)
        self.add(sae2)
        self.wait()

        fadeout_mobs = VGroup(
            grid, sep_line, prev_result, prev_rect,
        )
        limit = Tex("\\lim_{n \\to \\infty} {\\frac{\\pi {s}_{n}^{2}}{4 n}} = 1")
        limit[7:10:2].set_color(WALLIS_COLOR)
        self.play(
            FadeOut(fadeout_mobs),
            snan_expansion_2.animate.center().to_edge(UP),
            FadeIn(limit),
        )
        self.wait()


class PiBehindTheWallisProduct(Scene):
    def construct(self):
        # Setup to match up the previous scene
        sn_calc = Tex("s_{n} = \\frac{3}{2} \\cdot \\frac{5}{4} \\cdots \\frac{2n-1}{2n-2}")
        sn_calc.to_edge(UP)
        sn_calc.set_color(WALLIS_COLOR)
        limit_orig = Tex("\\lim_{n \\to \\infty} {\\frac{\\pi {s}_{n}^{2}}{4 n}} = 1")
        limit_orig[7:10:2].set_color(WALLIS_COLOR)
        self.add(sn_calc, limit_orig)

        # Expand s_n^2 out
        limit_sepn = Tex("\\lim_{n \\to \\infty} {\\frac{{s}_{n}^{2}}{n}} = \\frac{4}{\\pi}")
        limit_sepn[6:9:2].set_color(WALLIS_COLOR)
        self.play(
            AnimationGroup(
                ReplacementTransform(limit_orig[:6], limit_sepn[:6]),
                ReplacementTransform(limit_orig[7:11], limit_sepn[6:10]),
                ReplacementTransform(limit_orig[12:14], limit_sepn[10:12]),
            ),
            AnimationGroup(
                ReplacementTransform(limit_orig[11], limit_sepn[12], path_arc=PI / 2),
                FadeTransform(limit_orig[14], limit_sepn[13]),
                ReplacementTransform(limit_orig[6], limit_sepn[14], path_arc=-PI / 2),
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.remove(limit_orig)
        self.add(limit_sepn)
        self.wait()

        limit_expand = Tex(
            """
            \\lim_{n \\to \\infty} 
            \\left( \\frac{3}{2} \\cdot \\frac{3}{2} \\cdot \\frac{5}{4} \\cdot \\frac{5}{4}
            \\cdots \\frac{2n-1}{2n-2} \\cdot \\frac{2n-1}{2n-2} \\right)
            \\cdot \\frac{1}{n} = \\frac{4}{\\pi}
            """
        )
        limit_expand[7:44].set_color(WALLIS_COLOR)
        self.play(
            ReplacementTransform(limit_sepn[:6], limit_expand[:6]),
            FadeTransform(limit_sepn[6:9], limit_expand[6:45]),
            FadeTransform(limit_sepn[9].copy().set_opacity(0), limit_expand[45]),
            FadeTransform(limit_sepn[6:9].copy().set_opacity(0), limit_expand[46]),
            ReplacementTransform(limit_sepn[9:11], limit_expand[47:49]),
            ReplacementTransform(limit_sepn[11:], limit_expand[49:]),
            run_time=2,
        )
        self.remove(limit_sepn)
        self.add(limit_expand)
        self.play(FadeOut(sn_calc))
        self.wait()

        # Slide the numerators 1 slot to the right
        limit_slide = Tex(
            """
            \\lim_{n \\to \\infty} 
            \\left( \\frac{1}{2} \\cdot \\frac{3}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{4}
            \\cdots \\frac{2n-3}{2n-2} \\cdot \\frac{2n-1}{2n-2} \\right)
            \\cdot \\frac{2n-1}{n} = \\frac{4}{\\pi}
            """
        )
        limit_slide[7:44].set_color(WALLIS_COLOR)
        self.play(
            ReplacementTransform(limit_expand[:7], limit_slide[:7]),
            ReplacementTransform(limit_expand[46], limit_slide[7], path_arc=PI / 2),
            ReplacementTransform(limit_expand[8:11], limit_slide[8:11]),
            ReplacementTransform(limit_expand[7], limit_slide[11], path_arc=-PI / 4),
            ReplacementTransform(limit_expand[12:15], limit_slide[12:15]),
            ReplacementTransform(limit_expand[11], limit_slide[15], path_arc=-PI / 4),
            ReplacementTransform(limit_expand[16:19], limit_slide[16:19]),
            ReplacementTransform(limit_expand[15], limit_slide[19], path_arc=-PI / 4),
            ReplacementTransform(limit_expand[20:22], limit_slide[20:22]),
            FadeTransform(limit_expand[19], limit_slide[22], path_arc=-PI / 4),
            ReplacementTransform(limit_expand[22:24], limit_slide[23:25]),
            FadeTransform(limit_expand[24], limit_slide[25:29], path_arc=-PI / 4),
            ReplacementTransform(limit_expand[29:35], limit_slide[29:35]),
            ReplacementTransform(limit_expand[25:29], limit_slide[35:39], path_arc=-PI / 4),
            ReplacementTransform(limit_expand[39:46], limit_slide[39:46]),
            ReplacementTransform(limit_expand[35:39], limit_slide[46:50], path_arc=-PI / 4),
            ReplacementTransform(limit_expand[47:], limit_slide[50:]),
            run_time=2,
        )
        self.remove(limit_expand)
        self.add(limit_slide)
        self.wait()

        # The last fraction approaches 2 as n goes to infinity
        last_frac = limit_slide[46:52]
        last_frac_rect = SurroundingRectangle(last_frac)
        arrow = Tex("\\to").rotate(-PI / 2)
        two = Tex("2")
        arrow.next_to(last_frac_rect, DOWN)
        two.next_to(arrow, DOWN)
        two_group = VGroup(arrow, two)
        two_group.set_color(YELLOW)
        self.play(
            last_frac.animate.set_color(YELLOW),
            ShowCreation(last_frac_rect),
            FadeIn(two_group, shift=0.5 * DOWN),
        )
        self.wait()

        # Remove the last 2 from the equation...
        limit_wo2 = Tex(
            """
            \\lim_{n \\to \\infty} 
            \\frac{1}{2} \\cdot \\frac{3}{2} \\cdot \\frac{3}{4} \\cdot \\frac{5}{4}
            \\cdots \\frac{2n-3}{2n-2} \\cdot \\frac{2n-1}{2n-2}
            = \\frac{2}{\\pi}
            """
        )
        limit_wo2[6:43].set_color(WALLIS_COLOR)
        shift_group = VGroup(
            limit_slide[6], limit_slide[44:52],
            last_frac_rect, two_group,
        )
        self.play(
            ReplacementTransform(limit_slide[:6], limit_wo2[:6]),
            ReplacementTransform(limit_slide[7:44], limit_wo2[6:43]),
            ReplacementTransform(limit_slide[52:], limit_wo2[43:]),
            FadeOut(shift_group, shift=2 * DOWN),
        )
        self.remove(limit_slide)
        self.add(limit_wo2)
        self.wait()

        # Remove the last 2 from the equation...
        limit_inv = Tex(
            """
            \\lim_{n \\to \\infty} 
            \\frac{2}{1} \\cdot \\frac{2}{3} \\cdot \\frac{4}{3} \\cdot \\frac{4}{5}
            \\cdots \\frac{2n-2}{2n-3} \\cdot \\frac{2n-2}{2n-1}
            = \\frac{\\pi}{2}
            """
        )
        limit_inv[6:43].set_color(WALLIS_COLOR)
        self.play(
            AnimationGroup(
                ReplacementTransform(limit_wo2[6:21:4], limit_inv[8:21:4], path_arc=PI / 2),
                ReplacementTransform(limit_wo2[8:21:4], limit_inv[6:21:4], path_arc=PI / 2),
                ReplacementTransform(limit_wo2[24:28], limit_inv[29:33], path_arc=PI / 2),
                ReplacementTransform(limit_wo2[29:33], limit_inv[24:28], path_arc=PI / 2),
                ReplacementTransform(limit_wo2[34:38], limit_inv[39:43], path_arc=PI / 2),
                ReplacementTransform(limit_wo2[39:43], limit_inv[34:38], path_arc=PI / 2),
                ReplacementTransform(limit_wo2[44], limit_inv[46], path_arc=PI / 2),
                ReplacementTransform(limit_wo2[46], limit_inv[44], path_arc=PI / 2),
                lag_ratio=0.05,
            ),
            AnimationGroup(
                ReplacementTransform(limit_wo2[:6], limit_inv[:6]),
                ReplacementTransform(limit_wo2[7:21:2], limit_inv[7:21:2]),
                ReplacementTransform(limit_wo2[21:24], limit_inv[21:24]),
                ReplacementTransform(limit_wo2[28:43:5], limit_inv[28:43:5]),
                ReplacementTransform(limit_wo2[43::2], limit_inv[43::2]),
            ),
        )
        self.remove(limit_wo2)
        self.add(limit_inv)
        self.wait()

        # Yay! We got the Wallis product!
        wallis_prod = Tex(
            """
            \\frac{2}{1} \\cdot \\frac{2}{3} \\cdot \\frac{4}{3} \\cdot \\frac{4}{5} \\cdot
            \\frac{6}{5} \\cdot \\frac{6}{7} \\cdot \\frac{8}{7} \\cdot \\frac{8}{9} \\cdots
            = \\frac{\\pi}{2}
            """
        )
        wallis_prod.next_to(limit_inv, UP, buff=1.2)
        wallis_prod[:-4].set_color(WALLIS_COLOR)
        self.play(
            FadeTransform(limit_inv.copy()[:-4], wallis_prod[:-4]),
            ReplacementTransform(limit_inv.copy()[-4:], wallis_prod[-4:]),
        )
        self.wait()
        self.play(
            wallis_prod.animate.center().scale(1.5),
            FadeOut(limit_inv, shift=1.5 * DOWN),
        )
        self.wait()

        # And the pi behind this product
        pi = wallis_prod[-3]
        circle = Circle()
        circle.set_color(YELLOW).surround(pi, buff=0.15)
        wallis_rects = WallisRectangles(
            N=20, colors=WALLIS_RECTS_COLORS,
        )
        wallis_rects.set_height(FRAME_HEIGHT - 2).center()
        wallis_rects.set_opacity(0.2)
        self.play(
            AnimationGroup(
                FadeIn(wallis_rects),
                ShowCreation(circle),
                Animation(wallis_prod),
                pi.animate.set_color(YELLOW),
            ),
            run_time=2,
        )
        self.wait()
        self.add_sound("yay")
        self.wait(8)
        sector = wallis_rects.get_cover_sector()
        self.play(
            AnimationGroup(
                wallis_rects.animate.set_opacity(1),
                FadeIn(sector),
                FadeOut(wallis_prod),
                FadeOut(circle),
            ),
        )
        self.wait()


class FatalFlawsInTheProof(Scene):
    def construct(self):
        # Setup to match the last scene
        wallis_rects = WallisRectangles(
            N=20, colors=WALLIS_RECTS_COLORS
        )
        wallis_rects.set_height(FRAME_HEIGHT - 2).center()
        sector = wallis_rects.get_cover_sector()
        self.add(wallis_rects, sector)

        # However, there're 2 fatal flaws in this proof
        flaw_left = Tex("\\text{为什么是} \\frac{1}{4} \\text{圆} \\,?")
        flaw_right = Tex("\\frac{\\text{近似面积}}{\\text{精确面积}} \\to 1 \\,?")
        flaws = VGroup(flaw_left, flaw_right)
        flaws.set_color(GREY_B)
        flaws.scale(0.8)
        flaw_left.next_to(wallis_rects, LEFT, buff=1)
        flaw_right.next_to(wallis_rects, RIGHT, buff=1)

        dc_left = DarkCloud()
        dc_right = dc_left.copy()
        dcs = VGroup(dc_left, dc_right)
        dc_left.next_to(flaw_left, UP, buff=0.5)
        dc_right.next_to(flaw_right, UP, buff=0.5)
        self.play(
            FadeIn(flaws, shift=0.5 * DOWN),
            FadeIn(dcs, shift=0.5 * DOWN),
        )
        self.wait()

        # Why does this diagram approximate 1/4 circle?
        dims = [3, 1.5, 2]
        corner = wallis_rects.get_rects_corner()
        unit_size = wallis_rects.get_unit_size()
        for dim in dims:
            alpha = 1 / dim
            new_wallis_rects = WallisRectangles(
                N=20, colors=WALLIS_RECTS_COLORS,
                alpha=alpha,
            )
            new_wallis_rects.match_height(wallis_rects)
            new_wallis_rects.put_corner_at(corner)
            self.play(
                AnimationGroup(
                    Transform(wallis_rects, new_wallis_rects),
                    Animation(sector),
                ),
            )
            self.wait()

        # And why does the limit approach 1?
        arrows = VGroup(
            Vector(DL / 5, tip_width_ratio=3).next_to(point, UR, buff=0.1)
            for point in wallis_rects.get_outer_corners()
        )
        arrows.set_color(GREY)
        layer_text = TexText("20层")
        layer_text.add_updater(lambda mob: mob.next_to(wallis_rects.get_corner(UR), DR))
        self.play(
            AnimationGroup(
                *[
                    GrowArrow(arrow)
                    for arrow in arrows
                ],
                lag_ratio=0.02,
            )
        )
        self.wait()
        self.play(
            FadeOut(VGroup(arrows, dcs, flaws)),
            FadeIn(layer_text),
            VGroup(wallis_rects, sector).animate.shift(3 * LEFT)
        )
        layer_text.clear_updaters()
        self.wait()

        # We need to observe diagrams with more rectangles... and less rectangles
        # N_big = 30
        N_big = 100
        corner = wallis_rects.get_rects_corner()
        big_wallis_rects = WallisRectangles(
            N=N_big, colors=WALLIS_RECTS_COLORS,
        )
        big_layer_text = TexText(str(N_big) + "层")
        small_ref_wallis_rects = WallisRectangles(
            N=5, colors=WALLIS_RECTS_COLORS[:-1],
        )
        small_disp_wallis_rects = WallisRectangles(
            N=6, colors=WALLIS_RECTS_COLORS,
        )
        small_layer_text = TexText("5层")
        for lt in (big_layer_text, small_layer_text):
            lt.next_to(wallis_rects.get_corner(UR), DR)
        for wr in (big_wallis_rects, small_ref_wallis_rects):
            wr.match_height(wallis_rects)
            wr.put_corner_at(corner)
        small_disp_wallis_rects.set_unit_size(small_ref_wallis_rects.get_unit_size())
        small_disp_wallis_rects.put_corner_at(corner)

        self.play(
            AnimationGroup(
                FadeOut(wallis_rects),
                FadeIn(big_wallis_rects),
                FadeTransform(layer_text, big_layer_text),
                Animation(sector),
            )
        )
        self.wait()
        self.play(
            AnimationGroup(
                FadeOut(big_wallis_rects),
                FadeIn(small_disp_wallis_rects.get_layers(range(5))),
                FadeTransform(big_layer_text, small_layer_text),
                Animation(sector),
            )
        )
        self.wait()

        # Bad approximation isn't useless, as it shows a connection between the sector and rectangles
        sector.save_state()
        inner_layers = small_disp_wallis_rects.get_layers(range(4))
        outer_layer = small_disp_wallis_rects.get_layer(4)
        extra_layer = small_disp_wallis_rects.get_layer(5)
        self.play(
            AnimationGroup(
                inner_layers.animate.set_opacity(0.2),
                Animation(sector),
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                inner_layers.animate.set_opacity(1),
                outer_layer.animate.set_opacity(0.2),
                sector.animate.set_fill(opacity=0.8),
            ),
        )
        self.wait()

        # A way to constrain the area of the sector
        oneless_layer_text = Tex("&(n-1)\\text{层矩形} \\\\ &\\text{图案面积}=n-1")
        exact_layer_text = Tex("&n\\text{层矩形} \\\\ &\\text{图案面积}=n")
        onemore_layer_text = Tex("&(n+1)\\text{层矩形} \\\\ &\\text{图案面积}=n+1")
        n_layer_texts = VGroup(oneless_layer_text, exact_layer_text, onemore_layer_text)
        for lt in n_layer_texts:
            lt.next_to(wallis_rects.get_corner(UR), DR)
        upper_area_text = Tex("\\text{扇形面积} < n+1")
        lower_area_text = Tex("\\text{扇形面积} > n-1")
        area_texts = VGroup(upper_area_text, lower_area_text)
        area_texts.arrange(DOWN)
        area_texts.next_to(small_disp_wallis_rects, RIGHT, aligned_edge=DOWN, buff=1)
        area_texts.shift(1.5 * UP)
        area_texts.set_color(YELLOW)

        self.play(
            AnimationGroup(
                outer_layer.animate.set_opacity(1),
                FadeTransform(small_layer_text, exact_layer_text),
                Restore(sector),
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                AnimationGroup(
                    *[
                        Indicate(rect, color=rect.get_color())
                        for rect in outer_layer
                    ],
                    lag_ratio=0.05,
                ),
                Animation(sector),
            ),
        )
        self.wait()

        # An extra layer will give the upper bound
        self.play(
            AnimationGroup(
                AssembleLayer(small_disp_wallis_rects, 5),
                FadeTransform(exact_layer_text[0], onemore_layer_text[:5]),
                FadeTransform(exact_layer_text[1:4], onemore_layer_text[5:8]),
                FadeTransform(exact_layer_text[4:-1], onemore_layer_text[8:-3]),
                FadeTransform(exact_layer_text[-1], onemore_layer_text[-3:]),
                Animation(sector),
            ),
        )
        self.wait()
        self.play(Write(upper_area_text, stroke_color=YELLOW))
        self.wait()

        # Remove one layer will give the lower bound
        self.play(
            AnimationGroup(
                FadeOut(extra_layer),
                FadeTransform(onemore_layer_text[:5], exact_layer_text[0]),
                FadeTransform(onemore_layer_text[5:8], exact_layer_text[1:4]),
                FadeTransform(onemore_layer_text[8:-3], exact_layer_text[4:-1]),
                FadeTransform(onemore_layer_text[-3:], exact_layer_text[-1]),
                Animation(sector),
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                AnimationGroup(
                    *[
                        Indicate(rect, color=rect.get_color())
                        for rect in inner_layers[-1]
                    ],
                    lag_ratio=0.05,
                ),
                Animation(sector),
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                FadeOut(outer_layer),
                FadeTransform(exact_layer_text[0], oneless_layer_text[:5]),
                FadeTransform(exact_layer_text[1:4], oneless_layer_text[5:8]),
                FadeTransform(exact_layer_text[4:-1], oneless_layer_text[8:-3]),
                FadeTransform(exact_layer_text[-1], oneless_layer_text[-3:]),
                sector.animate.set_fill(opacity=0.8),
            ),
        )
        self.wait()
        self.play(Write(lower_area_text, stroke_color=YELLOW))
        self.wait()

        # We have a bound of the area of the sector
        area_bound_text = Tex("n-1 < \\text{扇形面积} < n+1")
        area_bound_text.move_to(area_texts)
        area_bound_text.set_color(YELLOW)
        self.play(
            AnimationGroup(
                ReplacementTransform(area_texts[0], area_bound_text[4:]),
                ReplacementTransform(area_texts[1][:4], area_bound_text[4:8]),
                ReplacementTransform(area_texts[1][4], area_bound_text[3], path_arc=-PI / 2),
                ReplacementTransform(area_texts[1][5:], area_bound_text[:3], path_arc=-PI / 2),
            ),
            AnimationGroup(
                FadeIn(outer_layer),
                Restore(sector),
                FadeTransform(oneless_layer_text[:5], exact_layer_text[0]),
                FadeTransform(oneless_layer_text[5:8], exact_layer_text[1:4]),
                FadeTransform(oneless_layer_text[8:-3], exact_layer_text[4:-1]),
                FadeTransform(oneless_layer_text[-3:], exact_layer_text[-1]),
            ),
        )
        self.wait()

        # And then the limit comes out naturally
        area_frac_text = Tex("\\frac{n-1}{n} < \\frac{\\text{扇形面积}}{\\text{图案面积}} < \\frac{n+1}{n}")
        area_frac_text.set_color(YELLOW)
        area_frac_text.move_to(area_bound_text)
        abt, aft = area_bound_text, area_frac_text
        self.play(
            ReplacementTransform(abt[:3], aft[:3]),
            GrowFromCenter(aft[3]),
            ReplacementTransform(exact_layer_text[-1].copy(), aft[4]),
            ReplacementTransform(abt[3], aft[5]),
            ReplacementTransform(abt[4:8], aft[6:10]),
            GrowFromCenter(aft[10]),
            ReplacementTransform(exact_layer_text[-6:-2].copy(), aft[11:15]),
            ReplacementTransform(abt[8:], aft[15:19]),
            GrowFromCenter(aft[19]),
            ReplacementTransform(exact_layer_text[-1].copy(), aft[20]),
            run_time=2,
        )
        self.remove(area_bound_text)
        self.add(area_frac_text)
        self.wait()

        _, mid_frac, _ = fracs = aft[:5], aft[6:15], aft[16:]
        rects = VGroup(
            SurroundingRectangle(frac)
            for frac in fracs
        )
        arrows = VGroup(
            Tex("\\to").rotate(-PI / 2).next_to(rect, DOWN)
            for rect in rects
        )
        ones = VGroup(
            Tex("1").next_to(arrow, DOWN)
            for arrow in arrows
        )
        for mobs in (rects, arrows, ones):
            mobs.set_color(YELLOW)
        self.play(
            *[ShowCreation(rect) for rect in rects[::2]],
            *[GrowFromCenter(arrow) for arrow in arrows[::2]],
            *[Write(one) for one in ones[::2]],
        )
        self.wait()
        self.play(
            *[TransformFromCopy(rect, rects[1]) for rect in rects[::2]],
            *[TransformFromCopy(arrow, arrows[1]) for arrow in arrows[::2]],
            *[TransformFromCopy(one, ones[1]) for one in ones[::2]],
        )
        self.wait()

        # This (kinda) resolves the second fatal flaw - why the limit is 1
        arrows[1].generate_target()
        arrow_target = arrows[1].target
        arrow_target.rotate(PI / 2).next_to(mid_frac, RIGHT)
        ones[1].generate_target()
        one_target = ones[1].target
        one_target.next_to(arrow_target, RIGHT)
        self.play(
            FadeOut(
                VGroup(aft[:6], aft[15:], rects, arrows[::2], ones[::2], exact_layer_text),
            ),
            MoveToTarget(arrows[1], path_arc=PI / 2),
            MoveToTarget(ones[1], path_arc=PI / 2),
        )
        self.wait()

        # But it relies on the condition that the sector crosses the outer layer...
        assumption_t2c = {
            "最外层矩形的右上角": OUTER_CORNER_COLOR,
            "次外层矩形的右上角": INNER_CORNER_COLOR,
            "扇形": sector.get_color(),
        }
        assumption = VGroup(
            TexText("对任意大小的矩形图案：", t2c=assumption_t2c),
            TexText("最外层矩形的右上角在扇形外", t2c=assumption_t2c),
            TexText("次外层矩形的右上角在扇形内", t2c=assumption_t2c),
        )
        assumption.arrange(DOWN, aligned_edge=LEFT)
        implies = Tex("\\Rightarrow").rotate(-PI / 2)
        implies.next_to(aft, UP)
        assumption.next_to(implies, UP)

        dot_radius = 0.06
        outer_corners = small_ref_wallis_rects.get_outer_corners()
        outer_dots = VGroup(
            Dot(corner, radius=dot_radius).set_color(OUTER_CORNER_COLOR)
            for corner in outer_corners
        )
        inner_corners = small_ref_wallis_rects.get_inner_corners()
        inner_dots = VGroup(
            Dot(corner, radius=dot_radius).set_color(INNER_CORNER_COLOR)
            for corner in inner_corners
        )

        self.play(FadeIn(implies, shift=DOWN))
        self.wait()
        self.play(
            FadeIn(assumption[1]),
            *[FadeIn(dot, scale=0.1) for dot in outer_dots],
        )
        self.wait()
        self.play(
            FadeIn(assumption[2]),
            *[FadeIn(dot, scale=0.1) for dot in inner_dots],
        )
        self.wait()
        self.play(FadeIn(assumption[0]))
        self.wait()

        # ... and always crosses the outer layer, no matter what the size is
        backup_wallis_group = VGroup(
            small_disp_wallis_rects.get_layers(range(5)),
            outer_dots, inner_dots,
        )
        prev_wallis_group = backup_wallis_group
        max_size = 15
        for N in range(6, max_size + 1):
            new_wallis_rects = WallisRectangles(
                N=N, colors=WALLIS_RECTS_COLORS[:-1],
            )
            new_wallis_rects.match_height(sector)
            new_wallis_rects.put_corner_at(corner)
            new_outer_corners = new_wallis_rects.get_outer_corners()
            new_outer_dots = VGroup(
                Dot(corner, radius=dot_radius).set_color(OUTER_CORNER_COLOR)
                for corner in new_outer_corners
            )
            new_inner_corners = new_wallis_rects.get_inner_corners()
            new_inner_dots = VGroup(
                Dot(corner, radius=dot_radius).set_color(INNER_CORNER_COLOR)
                for corner in new_inner_corners
            )
            new_wallis_group = VGroup(
                new_wallis_rects,
                new_outer_dots, new_inner_dots,
            )
            self.remove(prev_wallis_group)
            self.add(new_wallis_group, sector)
            self.wait()
            prev_wallis_group = new_wallis_group
        self.play(
            AnimationGroup(
                FadeOut(new_wallis_group),
                FadeIn(backup_wallis_group),
                Animation(sector),
            ),
        )
        self.wait()

        # It is also the answer to "Why it approaches a 1/4 circle"
        approach_text = TexText("矩形图案逼近$\\dfrac{1}{4}$圆")
        approach_text.set_color(YELLOW)
        approach_text.next_to(implies, DOWN)
        self.play(
            FadeTransform(
                VGroup(aft[6:15], arrows[1], ones[1]),
                approach_text,
            )
        )
        self.wait()

        # It's geometry time... again!
        fade_group = VGroup(assumption, implies, approach_text)
        shift_group = VGroup(backup_wallis_group, sector)
        shift_group.generate_target()
        shift_group.target.set_height(FRAME_HEIGHT - 3)
        shift_group.target.center().to_edge(UP)
        self.play(
            MoveToTarget(shift_group),
            FadeOut(fade_group, shift=2 * RIGHT),
        )
        self.wait()


class AnalyticalGeometryToTheRescue(Scene):
    def construct(self):
        # Setup to match up the previous scene
        wallis_rects = WallisRectangles(
            N=5, colors=WALLIS_RECTS_COLORS[:-1],
        )
        wallis_rects.set_height(FRAME_HEIGHT - 2)
        dot_radius = 0.06
        outer_corners = wallis_rects.get_outer_corners()
        outer_dots = VGroup(
            Dot(corner, radius=dot_radius).set_color(OUTER_CORNER_COLOR)
            for corner in outer_corners
        )
        inner_corners = wallis_rects.get_inner_corners()
        inner_dots = VGroup(
            Dot(corner, radius=dot_radius).set_color(INNER_CORNER_COLOR)
            for corner in inner_corners
        )
        sector = wallis_rects.get_cover_sector()
        wallis_group = VGroup(wallis_rects, outer_dots, inner_dots, sector)
        wallis_group.set_height(FRAME_HEIGHT - 3)
        wallis_group.center().to_edge(UP)
        self.add(wallis_group)
        self.wait()

        # Construct a coordinate system
        axes = VGroup(Line(ORIGIN, 10 * RIGHT), Line(ORIGIN, 10 * UP))
        axes.set_height(wallis_group.get_height() * 1.05)
        axes.next_to(wallis_group.get_corner(DL), UR, buff=0)
        for arrow in axes:
            arrow.add_tip(width=0.2, length=0.2)
        axes.set_color(AXES_COLOR)
        gridlines = wallis_rects.get_gridlines(length_factor=1.3)
        horiz_lines, vert_lines = gridlines
        self.play(
            FadeIn(axes[0], shift=0.5 * UP),
            FadeIn(axes[1], shift=0.5 * RIGHT),
            AnimationGroup(*[ShowCreation(line) for line in horiz_lines]),
            AnimationGroup(*[ShowCreation(line) for line in vert_lines]),
            Animation(outer_dots),
            Animation(inner_dots),
            Animation(sector),
            run_time=2,
        )
        self.wait()

        # The gridlines are sitting on s_n
        length_labels = wallis_rects.get_length_labels()
        gridline_labels = wallis_rects.get_gridline_labels(length_factor=1.3)
        self.play(
            AnimationGroup(
                *[
                    GrowFromEdge(label, UP)
                    for label in length_labels[0]
                ],
                lag_ratio=0.1,
            ),
            AnimationGroup(
                *[
                    GrowFromEdge(label, RIGHT)
                    for label in length_labels[1]
                ],
                lag_ratio=0.1,
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    GrowFromEdge(label, UP)
                    for label in gridline_labels[0]
                ],
                lag_ratio=0.1,
            ),
            AnimationGroup(
                *[
                    GrowFromEdge(label, RIGHT)
                    for label in gridline_labels[1]
                ],
                lag_ratio=0.1,
            ),
            length_labels.animate.set_opacity(0.5).set_color(AXES_COLOR)
        )
        self.wait()

        # So the two coordinates are both associated with s_n
        coords_scale_factor = 0.6
        inner_coords = VGroup(
            Tex(f"(s_{{{k}}},s_{{{5-k}}})")
            .scale(coords_scale_factor)
            .next_to(dot, DL, buff=dot_radius * 1.2)
            for k, dot in zip(range(1, 5), inner_dots)
        )
        inner_coords.set_color(INNER_CORNER_COLOR)
        outer_coords = VGroup(
            Tex(f"(s_{{{k}}},s_{{{6-k}}})")
            .scale(coords_scale_factor)
            .next_to(dot, UR, buff=dot_radius * 1.2)
            for k, dot in zip(range(1, 6), outer_dots)
        )
        outer_coords.set_color(OUTER_CORNER_COLOR)
        horiz_sn, vert_sn = gridline_labels

        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        GrowFromCenter(coord[k])
                        for coord in inner_coords
                    )
                    for k in [0, 3, 6]
                ],
                *[
                    AnimationGroup(
                        GrowFromCenter(coord[k])
                        for coord in outer_coords
                    )
                    for k in [0, 3, 6]
                ],
            ),
            AnimationGroup(
                *[
                    TransformFromCopy(horiz_sn[k], inner_coords[k - 1][1:3])
                    for k in range(1, 5)
                ],
                *[
                    TransformFromCopy(horiz_sn[k], outer_coords[k - 1][1:3])
                    for k in range(1, 6)
                ],
                lag_ratio=0.05,
            ),
            AnimationGroup(
                *[
                    TransformFromCopy(vert_sn[5 - k], inner_coords[k - 1][4:6])
                    for k in range(1, 5)
                ],
                *[
                    TransformFromCopy(vert_sn[6 - k], outer_coords[k - 1][4:6])
                    for k in range(1, 6)
                ],
                lag_ratio=0.05,
            ),
            run_time=2,
        )
        self.wait()

        # Show the relevant quantities in the diagram
        quantities_colors = [
            WHITE, AXES_COLOR,
            OUTER_CORNER_COLOR, OUTER_CORNER_COLOR,
            INNER_CORNER_COLOR, INNER_CORNER_COLOR,
        ]

        def setup_quantities(quantities):
            for mob, color in zip(quantities, quantities_colors):
                mob.set_color(color)
            quantities.arrange(DOWN, aligned_edge=LEFT)
            for ind in (1, 2, 4):
                quantities[ind:].shift(1 / 3 * DOWN)
            quantities.move_to(RIGHT_SIDE / 2)
            quantities.next_to(wallis_rects, RIGHT, coor_mask=[0, 1, 0])

        quantities_5 = VGroup(
            TexText(text)
            for text in [
                "矩形层数：5",
                "扇形半径：$s_{5}$",
                "最外层坐标：$(s_{k},s_{6-k})$",
                "最外层角标和：6",
                "次外层坐标：$(s_{k},s_{5-k})$",
                "次外层角标和：5",
            ]
        )
        setup_quantities(quantities_5)
        shift_group = VGroup(
            axes, wallis_rects, sector, gridlines,
            inner_dots, outer_dots, length_labels, gridline_labels,
            inner_coords, outer_coords,
        )
        self.play(
            shift_group.animate.shift(3 * LEFT),
            FadeIn(quantities_5, shift=LEFT),
        )
        self.wait()
        self.play(
            Indicate(gridline_labels[0][-1], scale_factor=2),
            Indicate(gridline_labels[1][-1], scale_factor=2),
            FlashAround(quantities_5[1], color=AXES_COLOR),
            run_time=2,
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    Indicate(coord, color=OUTER_CORNER_COLOR)
                    for coord in outer_coords
                ],
                lag_ratio=0.05,
            ),
            FlashAround(quantities_5[2:4], color=OUTER_CORNER_COLOR),
            run_time=2,
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    Indicate(coord, color=INNER_CORNER_COLOR)
                    for coord in inner_coords
                ],
                lag_ratio=0.05,
            ),
            FlashAround(quantities_5[4:], color=INNER_CORNER_COLOR),
            run_time=2,
        )
        self.wait()

        # Same goes for other number of layers
        quantities_n = VGroup(
            TexText(text)
            for text in [
                "矩形层数：$n$",
                "扇形半径：$s_{n}$",
                "最外层坐标：$(s_{k},s_{n+1-k})$",
                "最外层角标和：$n+1$",
                "次外层坐标：$(s_{k},s_{n-k})$",
                "次外层角标和：$n$",
            ]
        )
        setup_quantities(quantities_n)
        quantities_n.next_to(quantities_5.get_left(), RIGHT, buff=0)

        new_coords_scale_factor = 0.5
        new_inner_coords_texts = [
            "(s_{1},s_{n-1})", "(s_{2},s_{n-2})", "\\dots", "\\dots",
            "(s_{k},s_{n-k})", "\\dots", "\\dots",
            "(s_{n-2},s_{2})", "(s_{n-1},s_{1})",
        ]
        new_outer_coords_texts = [
            "(s_{1},s_{n})", "(s_{2},s_{n-1})", "\\dots", "\\dots",
            "(s_{k},s_{n+1-k})", "\\dots", "\\dots", "\\dots",
            "(s_{n-1},s_{2})", "(s_{n},s_{1})",
        ]

        curr_N = len(new_outer_coords_texts)
        new_wallis_rects = WallisRectangles(
            N=curr_N,
            colors=WALLIS_RECTS_COLORS[:-1],
        )
        new_wallis_rects.match_height(sector)
        new_wallis_rects.put_corner_at(wallis_rects.get_rects_corner())
        new_outer_corners = new_wallis_rects.get_outer_corners()
        new_outer_dots = VGroup(
            Dot(corner, radius=dot_radius).set_color(OUTER_CORNER_COLOR)
            for corner in new_outer_corners
        )
        new_inner_corners = new_wallis_rects.get_inner_corners()
        new_inner_dots = VGroup(
            Dot(corner, radius=dot_radius).set_color(INNER_CORNER_COLOR)
            for corner in new_inner_corners
        )
        new_inner_coords = VGroup(
            Tex(text)
            .scale(new_coords_scale_factor)
            .next_to(dot, DL, buff=dot_radius * 1.2)
            for text, dot in zip(new_inner_coords_texts, new_inner_dots)
        )
        new_inner_coords.set_color(INNER_CORNER_COLOR)
        new_outer_coords = VGroup(
            Tex(text)
            .scale(new_coords_scale_factor)
            .next_to(dot, UR, buff=dot_radius * 1.2)
            for text, dot in zip(new_outer_coords_texts, new_outer_dots)
        )
        new_outer_coords.set_color(OUTER_CORNER_COLOR)

        self.play(
            AnimationGroup(
                FadeTransform(wallis_rects, new_wallis_rects),
                FadeOut(VGroup(gridlines, gridline_labels, length_labels)),
                FadeOut(VGroup(inner_coords, outer_coords, inner_dots, outer_dots)),
                FadeIn(VGroup(new_inner_dots, new_outer_dots)),
                FadeIn(VGroup(new_inner_coords, new_outer_coords)),
                Animation(sector),
            ),
            AnimationGroup(
                FadeTransform(mob_5, mob_n)
                for mob_5, mob_n in zip(quantities_5, quantities_n)
            )
        )
        self.bring_to_front(new_inner_dots, new_outer_dots, sector, new_inner_coords, new_outer_coords)
        self.wait()
        self.play(
            FlashAround(quantities_n[1], color=AXES_COLOR),
            run_time=2,
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    Indicate(coord, color=OUTER_CORNER_COLOR)
                    for coord in new_outer_coords
                ],
                lag_ratio=0.05,
            ),
            FlashAround(quantities_n[2:4], color=OUTER_CORNER_COLOR),
            run_time=2,
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    Indicate(coord, color=INNER_CORNER_COLOR)
                    for coord in new_inner_coords
                ],
                lag_ratio=0.05,
            ),
            FlashAround(quantities_n[4:], color=INNER_CORNER_COLOR),
            run_time=2,
        )
        self.wait()
        general_inner_coord = new_inner_coords[4]
        new_inner_coords.remove(general_inner_coord)
        general_outer_coord = new_outer_coords[4]
        new_outer_coords.remove(general_outer_coord)
        self.play(
            FadeOut(new_inner_coords),
            FadeOut(new_outer_coords),
        )
        self.wait()

        # From the diagram, we can conclude two inequalities
        inner_ineq = Tex("s_{k}^{2} + s_{n-k}^{2} < s_{n}^{2}")
        inner_cond = Tex("(0 < k < n)")
        outer_ineq = Tex("s_{k}^{2} + s_{n+1-k}^{2} > s_{n}^{2}")
        outer_cond = Tex("(0 < k < n+1)")
        for ineq in (inner_ineq, outer_ineq):
            ineq.scale(0.8)
        for ineq in (inner_cond, outer_cond):
            ineq.scale(0.6)
        inner_ineq.move_to(new_wallis_rects)
        inner_ineq.set_color(INNER_CORNER_COLOR)
        inner_cond.next_to(inner_ineq, DOWN, aligned_edge=LEFT)
        inner_cond.set_color(INNER_CORNER_COLOR)
        outer_ineq.next_to(new_wallis_rects.get_corner(UR), DL, buff=-0.3)
        outer_ineq[-3::2].set_color(AXES_COLOR)
        outer_ineq.set_color(OUTER_CORNER_COLOR)
        outer_cond.next_to(outer_ineq, DOWN, aligned_edge=RIGHT)
        outer_cond.set_color(OUTER_CORNER_COLOR)

        self.play(
            FadeOut(VGroup(general_outer_coord[ind] for ind in (0, 3, -1))),
            *[GrowFromCenter(outer_ineq[ind]) for ind in (1, 3, 5, 11, 13)],
            ReplacementTransform(general_outer_coord[1], outer_ineq[0]),
            ReplacementTransform(general_outer_coord[2], outer_ineq[2]),
            ReplacementTransform(general_outer_coord[4], outer_ineq[4]),
            ReplacementTransform(general_outer_coord[5:-1], outer_ineq[6:11]),
            TransformFromCopy(quantities_n[1][-2], outer_ineq[12]),
            TransformFromCopy(quantities_n[1][-1], outer_ineq[14]),
            run_time=2,
        )
        self.wait()
        self.play(
            # FadeIn(inner_bg_rect),
            FadeOut(VGroup(general_inner_coord[ind] for ind in (0, 3, -1))),
            *[GrowFromCenter(inner_ineq[ind]) for ind in (1, 3, 5, 9, 11)],
            ReplacementTransform(general_inner_coord[1], inner_ineq[0]),
            ReplacementTransform(general_inner_coord[2], inner_ineq[2]),
            ReplacementTransform(general_inner_coord[4], inner_ineq[4]),
            ReplacementTransform(general_inner_coord[5:-1], inner_ineq[6:9]),
            TransformFromCopy(quantities_n[1][-2], inner_ineq[10]),
            TransformFromCopy(quantities_n[1][-1], inner_ineq[12]),
            run_time=2,
        )
        self.wait()
        self.play(
            Write(inner_cond, stroke_color=INNER_CORNER_COLOR),
            Write(outer_cond, stroke_color=OUTER_CORNER_COLOR),
        )
        self.wait()

        # Then we let it grow
        curr_wallis_rects = new_wallis_rects
        curr_outer_dots = new_outer_dots
        curr_inner_dots = new_inner_dots
        front_group = VGroup(
            sector, inner_ineq, inner_cond, outer_ineq, outer_cond,
        )
        final_dot_radius = dot_radius / 2

        final_N = 50
        start, end = curr_N + 1, final_N + 1
        for N in range(start, end):
            self.remove(curr_wallis_rects, curr_outer_dots, curr_inner_dots)
            new_wallis_rects = WallisRectangles(
                N=N, colors=WALLIS_RECTS_COLORS[:-1],
            )
            new_wallis_rects.match_height(sector)
            new_wallis_rects.put_corner_at(curr_wallis_rects.get_rects_corner())
            alpha = (N - start) / (end - 1 - start)
            r = final_dot_radius * alpha + dot_radius * (1 - alpha)
            new_outer_corners = new_wallis_rects.get_outer_corners()
            new_outer_dots = VGroup(
                Dot(corner, radius=r).set_color(OUTER_CORNER_COLOR)
                for corner in new_outer_corners
            )
            new_inner_corners = new_wallis_rects.get_inner_corners()
            new_inner_dots = VGroup(
                Dot(corner, radius=r).set_color(INNER_CORNER_COLOR)
                for corner in new_inner_corners
            )
            self.add(new_wallis_rects, new_outer_dots, new_inner_dots)
            self.bring_to_front(front_group)
            self.add(inner_ineq, inner_cond)
            curr_wallis_rects = new_wallis_rects
            curr_outer_dots = new_outer_dots
            curr_inner_dots = new_inner_dots
            self.wait()


class TheCrucialLogicChain(Scene):
    def construct(self):
        # Setup the logic chain
        rect_template = RoundedRectangle(
            width=FRAME_WIDTH / 2 - 1, height=FRAME_HEIGHT / 2 - 1.5,
            fill_opacity=0.1, fill_color=GREY,
        )
        inner_ineq = Tex("&s_{k}^{2} + s_{n-k}^{2} < s_{n}^{2} \\\\ &(0 < k < n)")
        inner_ineq[-7:].scale(0.6, about_point=inner_ineq[-7:].get_corner(UL))
        inner_ineq.set_color(INNER_CORNER_COLOR)
        outer_ineq = Tex("&s_{k}^{2} + s_{n+1-k}^{2} > s_{n}^{2} \\\\ &(0 < k < n+1)")
        outer_ineq[-9:].scale(0.6, about_point=outer_ineq[-9:].get_corner(UL))
        outer_ineq.set_color(OUTER_CORNER_COLOR)
        top_left_group = VGroup(inner_ineq, outer_ineq)
        top_left_group.arrange(DOWN, aligned_edge=LEFT)

        top_right_group = VGroup(
            TexText("对任意大小的矩形图案："),
            TexText("次外层矩形的右上角在扇形内").set_color(INNER_CORNER_COLOR),
            TexText("最外层矩形的右上角在扇形外").set_color(OUTER_CORNER_COLOR),
        )
        top_right_group.arrange(DOWN, aligned_edge=LEFT)

        bottom_left_group = VGroup(
            TexText("关键的极限"),
            Tex("\\lim_{n \\to \\infty} {\\frac{\\pi {s}_{n}^{2}}{4 n}} = 1"),
        )
        bottom_left_group.set_color(WALLIS_COLOR)
        bottom_left_group.arrange(DOWN)

        alt_bottom_left_group = VGroup(
            TexText("沃利斯乘积"),
            Tex(
                """
            \\frac{2}{1} \\cdot \\frac{2}{3} \\cdot \\frac{4}{3} \\cdot \\frac{4}{5} \\cdot
            \\frac{6}{5} \\cdot \\frac{6}{7} \\cdot \\frac{8}{7} \\cdot \\frac{8}{9} \\cdots
            = \\frac{\\pi}{2}
            """
            ),
        )
        alt_bottom_left_group.set_color(WALLIS_COLOR)
        alt_bottom_left_group.arrange(DOWN, buff=0.5)

        text = TexText(
            """
            扇形被矩形图案限制 \\\\[2ex]
            $n-1 < \\begin{aligned} \\text{扇形} \\\\[-0.5ex] \\text{面积} \\end{aligned} < n+1$
            """
        )
        text[-3:].set_color(OUTER_CORNER_COLOR)
        text[-8:-4].set_color(GREY)
        text[-12:-9].set_color(INNER_CORNER_COLOR)
        wallis_rects = WallisRectangles(N=20, colors=WALLIS_RECTS_COLORS[:-1])
        wallis_rects.match_height(text)
        bottom_right_group = VGroup(text, wallis_rects)
        bottom_right_group.arrange(RIGHT, buff=0.5)
        sector = wallis_rects.get_cover_sector()
        bottom_right_group.add(sector)

        top_mid_arrow = Tex("\\Leftrightarrow")
        bottom_mid_arrow = Tex("\\Leftarrow")
        mid_left_blank = VectorizedPoint()
        mid_mid_blank = VectorizedPoint()
        mid_right_arrow = Tex("\\Downarrow")

        groups = VGroup(
            top_left_group, top_right_group, bottom_left_group, bottom_right_group,
            alt_bottom_left_group,
        )
        for group in groups:
            rect = rect_template.copy()
            width_thres = rect.get_width() * 0.9
            height_thres = rect.get_height() * 0.9
            if group.get_width() > width_thres:
                group.set_width(width_thres)
            if group.get_height() > height_thres:
                group.set_height(height_thres)
            group.move_to(rect)
            group.add_to_back(rect)

        grid = VGroup(
            top_left_group, top_mid_arrow, top_right_group,
            mid_left_blank, mid_mid_blank, mid_right_arrow,
            bottom_left_group, bottom_mid_arrow, bottom_right_group,
        )
        grid.arrange_in_grid(3, 3, h_buff=-2.5, v_buff=-0.8)
        grid.to_edge(UP)
        self.add(grid)
        self.wait()

        # Wave through the chain
        chain = VGroup(
            top_left_group, top_mid_arrow, top_right_group, mid_right_arrow,
            bottom_right_group, bottom_mid_arrow, bottom_left_group,
        )
        factors = it.cycle([0.2, 0.5])
        self.play(
            AnimationGroup(
                *[
                    Indicate(
                        mob, color=None,
                        scale_factor=1 + f / np.pow(ind + 1, 1.5),
                    )
                    for ind, (mob, f) in enumerate(zip(chain, factors))
                ],
                lag_ratio=0.3,
            ),
            run_time=3,
        )
        self.wait()

        # From limit to Wallis product
        alt_bottom_left_group[1].scale(1.2, about_edge=DOWN)
        alt_bottom_left_group.move_to(bottom_left_group)
        self.play(
            AnimationGroup(
                FadeTransform(source, target)
                for source, target in zip(bottom_left_group, alt_bottom_left_group)
            )
        )
        self.wait()

        # Deja vu?
        center_wallis_prod = Tex(
            """
            \\frac{2}{1} \\cdot \\frac{2}{3} \\cdot \\frac{4}{3} \\cdot \\frac{4}{5} \\cdot
            \\frac{6}{5} \\cdot \\frac{6}{7} \\cdot \\frac{8}{7} \\cdot \\frac{8}{9} \\cdots
            = \\frac{\\pi}{2}
            """
        )
        center_wallis_prod.scale(1.5)
        center_wallis_prod[:-4].set_color(WALLIS_COLOR)
        pi = center_wallis_prod[-3]
        pi.set_color(YELLOW)
        circle = Circle()
        circle.set_color(YELLOW).surround(pi, buff=0.15)
        circle.generate_target()
        circle.surround(alt_bottom_left_group[-1][-3], buff=0.1).set_opacity(0)
        center_wallis_rects = WallisRectangles(
            N=20, colors=WALLIS_RECTS_COLORS,
        )
        center_wallis_rects.set_height(FRAME_HEIGHT - 2).center()
        center_wallis_rects.set_opacity(0.2)
        grid = VGroup(
            top_left_group, top_mid_arrow, top_right_group,
            mid_left_blank, mid_mid_blank, mid_right_arrow,
            alt_bottom_left_group, bottom_mid_arrow, bottom_right_group,
        )
        wallis_rects = bottom_right_group[-2]
        bottom_right_group.remove(wallis_rects)
        wallis_prod = alt_bottom_left_group[-1]
        alt_bottom_left_group.remove(wallis_prod)
        self.play(
            AnimationGroup(
                MoveToTarget(circle),
                ReplacementTransform(wallis_rects, center_wallis_rects),
                ReplacementTransform(wallis_prod, center_wallis_prod),
                run_time=2,
            ),
            AnimationGroup(
                FadeOut(grid),
                run_time=1,
            ),
        )
        self.wait()
        self.add_sound("yay")
        self.wait(8)


class FinalScalingTrick(Scene):
    def construct(self):
        # Recall the form of s_n...
        sn_form = Tex("s_{n} = \\frac{3}{2} \\cdot \\frac{5}{4} \\cdots \\frac{2n-1}{2n-2}")
        sn_form.set_color(WALLIS_COLOR)
        self.add(sn_form)
        self.wait()

        si_form = Tex("s_{i} = \\frac{3}{2} \\cdot \\frac{5}{4} \\cdots \\frac{2i-1}{2i-2}")
        si_form.set_color(WALLIS_COLOR)
        sipj_form = Tex(
            """
            s_{i+j} = \\frac{3}{2} \\cdot \\frac{5}{4} \\cdots \\frac{2i-1}{2i-2}
            \\cdot \\frac{2i+1}{2i} \\cdot \\frac{2i+3}{2i+2} \\cdots \\frac{2(i+j)-1}{2(i+j)-2}
            """
        )
        sipj_form.set_color(FACTOR_COLOR)
        shift_vec = si_form[2].get_center() - sipj_form[4].get_center()
        sipj_form.shift(shift_vec)
        single_forms = VGroup(si_form, sipj_form)
        single_forms.arrange(DOWN, coor_mask=[0, 1, 0], buff=0.5)
        single_forms.to_edge(UP, buff=1)

        self.play(FadeTransform(sn_form, si_form))
        self.wait()
        self.play(
            AnimationGroup(
                TransformFromCopy(si_form[:2], sipj_form[:2]),
                TransformFromCopy(VGroup(si_form[1]), sipj_form[2:4]),
                TransformFromCopy(si_form[2:], sipj_form[4:24]),
                run_time=1,
            ),
            AnimationGroup(
                *[
                    GrowFromCenter(sipj_form[start:end])
                    for start, end in zip(
                        [24, 25, 32, 33, 42, 45],
                        [25, 32, 33, 42, 45, 62],
                    )
                ],
                lag_ratio=0.02,
                run_time=2,
            ),
        )
        self.wait()
        brace_orig = Brace(sipj_form[25:])
        extra_term_tex = TexText("额外的$j$项")
        brace_orig.put_at_tip(extra_term_tex)
        VGroup(brace_orig, extra_term_tex).set_color(FACTOR_COLOR)
        self.play(
            sipj_form[5:24].animate.set_color(WALLIS_COLOR),
            GrowFromEdge(brace_orig, ORIGIN),
            Write(extra_term_tex, stroke_color=extra_term_tex.get_color()),
        )
        self.wait()

        # ...and the recurrence property
        sipj_simp = Tex(
            """
            s_{i+j} = s_{i}
            \\left( 1+\\frac{1}{2i} \\right)
            \\left( 1+\\frac{1}{2(i+1)}  \\right)
            \\cdots \\left( 1+\\frac{1}{2(i+j-1)} \\right)
            """
        )
        VGroup(sipj_simp[:4], sipj_simp[7:]).set_color(FACTOR_COLOR)
        sipj_simp[5:7].set_color(WALLIS_COLOR)
        sipj_simp.move_to(sipj_form)
        si_form.generate_target()
        si_form.target.match_width(sipj_simp[5:7]).move_to(sipj_simp[5:7])
        si_form.target.set_opacity(0)

        extra_term_tex.generate_target()
        brace_simp = Brace(sipj_simp[7:])
        brace_simp.put_at_tip(extra_term_tex.target)
        brace_simp.set_color(FACTOR_COLOR)
        self.play(
            AnimationGroup(
                ReplacementTransform(sipj_form[:5], sipj_simp[:5]),
                FadeTransform(sipj_form[5:24], sipj_simp[5:7]),
                ReplacementTransform(sipj_form[24], sipj_simp[7]),
                ReplacementTransform(sipj_form[25:27], VGroup(sipj_simp[8])),
                ReplacementTransform(sipj_form[27:32], sipj_simp[9:14]),
                ReplacementTransform(VGroup(sipj_form[32]), sipj_simp[14:16]),
                ReplacementTransform(sipj_form[33:35], VGroup(sipj_simp[16])),
                ReplacementTransform(sipj_form[35:39], sipj_simp[17:21]),
                ReplacementTransform(sipj_form[39:42], sipj_simp[21:26]),
                ReplacementTransform(sipj_form[42].copy(), sipj_simp[26]),
                ReplacementTransform(sipj_form[42:45], sipj_simp[27:30]),
                ReplacementTransform(sipj_form[44].copy(), sipj_simp[30]),
                ReplacementTransform(sipj_form[45:51], VGroup(sipj_simp[31])),
                ReplacementTransform(sipj_form[51:55], sipj_simp[32:36]),
                ReplacementTransform(sipj_form[55:], sipj_simp[36:43]),
                ReplacementTransform(VectorizedPoint(sipj_form.get_right()), sipj_simp[43]),
                MoveToTarget(si_form),
                ReplacementTransform(brace_orig, brace_simp),
                MoveToTarget(extra_term_tex),
                run_time=3,
            )
        )
        self.remove(sipj_form, si_form)
        self.add(sipj_simp)
        self.wait()

        # Add square
        sipj_square = Tex(
            """
            s_{i+j}^{2} = s_{i}^{2}
            \\left( 1+\\frac{1}{2i} \\right)^{2}
            \\left( 1+\\frac{1}{2(i+1)} \\right)^{2}
            \\cdots \\left( 1+\\frac{1}{2(i+j-1)} \\right)^{2}
            """
        )
        VGroup(sipj_square[:5], sipj_square[9:]).set_color(FACTOR_COLOR)
        sipj_square[6:9].set_color(WALLIS_COLOR)
        sipj_square.move_to(sipj_simp)

        extra_term_tex.generate_target()
        brace_square = Brace(sipj_square[9:])
        brace_square.put_at_tip(extra_term_tex.target)
        brace_square.set_color(FACTOR_COLOR)
        self.play(
            AnimationGroup(
                *[
                    FadeIn(sipj_square[ind], shift=DOWN)
                    for ind in (1, 7, 17, 30, 48)
                ],
                lag_ratio=0.05,
                run_time=2,
            ),
            ReplacementTransform(sipj_simp[0], sipj_square[0]),
            ReplacementTransform(sipj_simp[1:6], sipj_square[2:7]),
            ReplacementTransform(sipj_simp[6:15], sipj_square[8:17]),
            ReplacementTransform(sipj_simp[15:27], sipj_square[18:30]),
            ReplacementTransform(sipj_simp[27:], sipj_square[31:48]),
            ReplacementTransform(brace_simp, brace_square),
            MoveToTarget(extra_term_tex),
        )
        self.remove(sipj_simp)
        self.add(sipj_square)
        self.wait()

        # The factors all have the same form
        factor_form_tex = Tex(
            """
            \\text{因子形式：} \\left( 1+\\frac{1}{2m} \\right)^2 \\quad
            (m = i ,\\, i+1 ,\\, \\dots ,\\, i+j-1)
            """
        )
        factor_form_tex.set_width(brace_square.get_width() * 0.9)
        brace_square.put_at_tip(factor_form_tex)
        self.play(
            brace_square.animate.set_color(WHITE),
            FadeTransform(extra_term_tex, factor_form_tex),
        )
        self.wait()

        # Focus on this factor and expand it
        factor = Tex("\\left( 1+\\frac{1}{2m} \\right)^2")
        self.play(
            FadeOut(
                VGroup(factor_form_tex, brace_square, sipj_square),
                shift=UP + 0.5 * RIGHT,
            ),
            ReplacementTransform(factor_form_tex[5:14], factor),
        )
        self.remove(factor_form_tex)
        self.add(factor)
        self.wait()

        factor_expand = Tex("\\left( 1+\\frac{1}{2m} \\right)^2 = 1 + \\frac{1}{m} + \\frac{1}{4m^2}")
        L = len(factor)
        self.play(
            ReplacementTransform(factor, factor_expand[:L]),
            FadeIn(factor_expand[L:], shift=1.5 * LEFT),
        )
        self.remove(factor)
        self.add(factor_expand)
        self.wait()

        # There's a lower bound...
        factor_part = factor_expand[-11:]
        lower_ineq = Tex("<")
        lower_ineq.rotate(PI / 2)
        lower_ineq.next_to(factor_part, DOWN)
        lower_bound = Tex("\\text{下限：} 1 + \\frac{1}{m}")
        lower_bound_simp = Tex("\\text{下限：} \\frac{m+1}{m}")
        for tex in (lower_bound, lower_bound_simp):
            tex.generate_target()
            tex.target[3:].next_to(lower_ineq, DOWN)
            shift_vec = tex.target[-1].get_center() - tex[-1].get_center()
            tex.shift(shift_vec)
        VGroup(lower_bound, lower_ineq, lower_bound_simp).set_color(BOUND_COLOR)
        self.play(
            factor_expand[:L + 1].animate.set_opacity(0.5),
            FadeIn(lower_ineq, shift=0.5 * DOWN),
            Write(lower_bound[:3], stroke_color=BOUND_COLOR),
            ReplacementTransform(factor_part[:5].copy(), lower_bound[3:]),
        )
        self.add(lower_bound)
        self.wait()
        self.play(ReplacementTransform(lower_bound, lower_bound_simp))
        self.wait()

        # ...and an upper bound
        upper_ineq = lower_ineq.copy()
        upper_ineq.next_to(factor_part, UP)
        upper_bound = Tex("\\text{上限：} 1 + \\frac{1}{m} + \\frac{1}{m^2}")
        upper_bound_ex = Tex("\\text{上限：} 1 + \\frac{1}{m} + \\frac{1}{m(m-1)}")
        upper_bound_simp = Tex("\\text{上限：} \\frac{m}{m-1}")
        for tex in (upper_bound, upper_bound_ex, upper_bound_simp):
            tex.generate_target()
            tex.target[3:].next_to(upper_ineq, UP)
            shift_vec = tex.target[-1].get_center() - tex[-1].get_center()
            tex.shift(shift_vec)
        VGroup(upper_bound, upper_bound_ex, upper_bound_simp).set_color(BOUND_COLOR)
        self.play(
            FadeIn(upper_ineq, shift=0.5 * UP),
            Write(upper_bound[:3], stroke_color=BOUND_COLOR),
            ReplacementTransform(factor_part[:8].copy(), upper_bound[3:11]),
            ReplacementTransform(factor_part[8].copy(), VectorizedPoint(upper_bound[11].get_left())),
            ReplacementTransform(factor_part[-2:].copy(), upper_bound[-2:]),
        )
        self.add(upper_bound)
        self.wait()
        self.play(
            ReplacementTransform(upper_bound[:12], upper_bound_ex[:12]),
            ReplacementTransform(upper_bound[12:], upper_bound_ex[12:]),
        )
        self.add(upper_bound_ex)
        self.wait()
        self.play(
            ReplacementTransform(upper_bound_ex[:3], upper_bound_simp[:3]),
            ReplacementTransform(upper_bound_ex[3:10], VGroup(upper_bound_simp[3])),
            ReplacementTransform(upper_bound_ex[11], upper_bound_simp[3]),
            ReplacementTransform(upper_bound_ex[10], upper_bound_simp[4]),
            ReplacementTransform(upper_bound_ex[13:16], upper_bound_simp[5:]),
            FadeOut(upper_bound_ex[12::4], shift=0.15 * DOWN + 1.25 * LEFT),
        )
        self.remove(upper_bound_ex)
        self.add(upper_bound_simp)
        self.wait()

        # So we have the bound of the factor
        bound_colormap = {
            "\\frac{m+1}{m}": BOUND_COLOR,
            "\\left( 1+\\frac{1}{2m} \\right)^2": FACTOR_COLOR,
            "\\frac{m}{m-1}": BOUND_COLOR,
        }
        factor_bound = Tex(
            "\\frac{m+1}{m} < \\left( 1+\\frac{1}{2m} \\right)^2 < \\frac{m}{m-1}",
            t2c=bound_colormap,
        )
        lower_bound_simp.generate_target()
        lower_bound_simp.target[:3].scale(1e-2, about_edge=RIGHT).set_opacity(0)
        lower_bound_simp.target.next_to(factor_bound[:5].get_right(), LEFT, buff=0)
        upper_bound_simp.generate_target()
        upper_bound_simp.target[:3].scale(1e-2, about_edge=RIGHT).set_opacity(0)
        upper_bound_simp.target.next_to(factor_bound[-5:].get_right(), LEFT, buff=0)
        lower_ineq.generate_target()
        lower_ineq.target.set_color(WHITE).rotate(-PI / 2).move_to(factor_bound[5])
        upper_ineq.generate_target()
        upper_ineq.target.set_color(WHITE).rotate(-PI / 2).move_to(factor_bound[-6])
        factor_expand.generate_target()
        factor_expand.target.set_opacity(1).set_color(FACTOR_COLOR)
        factor_expand.target[9:].scale(1e-2, about_edge=LEFT).set_opacity(0)
        factor_expand.target[:9].next_to(factor_bound[6:].get_left(), RIGHT, buff=0)
        temp = VGroup(
            factor_expand, lower_bound_simp, upper_bound_simp,
            lower_ineq, upper_ineq,
        )
        self.play(
            *[
                MoveToTarget(mob, path_arc=-PI / 2)
                for mob in temp
            ],
            run_time=2,
        )
        self.remove(temp)
        self.add(factor_bound)
        self.wait()

        # Back to s(i+j)^2 and si^2, with a few more terms
        sipj_square.center()
        self.play(
            factor_bound.animate.scale(0.8).to_edge(UP),
            FadeIn(sipj_square, shift=UP),
        )
        self.wait()

        square_extra = Tex(
            """
            s_{i+j}^{2} = s_{i}^{2}
            \\left( 1+\\frac{1}{2i} \\right)^{2}
            \\left( 1+\\frac{1}{2(i+1)} \\right)^{2}
            \\left( 1+\\frac{1}{2(i+2)} \\right)^{2}
            \\cdots \\left( 1+\\frac{1}{2(i+j-2)} \\right)^{2}
            \\left( 1+\\frac{1}{2(i+j-1)} \\right)^{2}
            """
        )
        parts_extra = VGroup(
            square_extra[start:end]
            for start, end in zip(
                [0, 5, 6, 9, 18, 31, 44, 47, 62],
                [5, 6, 9, 18, 31, 44, 47, 62, 77],
            )
        )
        square_extra.set_width(FRAME_WIDTH - 1)
        square_extra[6:9].set_color(WALLIS_COLOR)
        VGroup(square_extra[:5], square_extra[9:]).set_color(FACTOR_COLOR)
        self.play(
            ReplacementTransform(sipj_square[:31], square_extra[:31]),
            ReplacementTransform(VGroup(sipj_square[31]), square_extra[31:44]),
            ReplacementTransform(VGroup(sipj_square[32]), square_extra[44:46]),
            ReplacementTransform(VGroup(sipj_square[33]), square_extra[46:62]),
            ReplacementTransform(sipj_square[34:], square_extra[62:]),
        )
        self.remove(sipj_square)
        self.add(square_extra)
        self.wait()

        square_lower = Tex(
            """
            s_{i+j}^{2} > s_{i}^{2}
            \\frac{i+1}{i} \\frac{i+2}{i+1} \\frac{i+3}{i+2}
            \\cdots \\frac{i+j-1}{i+j-2} \\frac{i+j}{i+j-1}
            """
        )
        parts_lower = VGroup(
            square_lower[start:end]
            for start, end in zip(
                [0, 5, 6, 9, 14, 21, 28, 31, 42],
                [5, 6, 9, 14, 21, 28, 31, 42, 51],
            )
        )
        ineq_lower = VGroup(
            Tex(">").rotate(-PI / 2) if ind >= 3 else VectorizedPoint()
            for ind in range(len(parts_lower))
        )
        ineq_lower.scale(0.75)
        ineq_lower.next_to(parts_extra, DOWN)
        for source, ref in zip(ineq_lower, parts_extra):
            source.next_to(ref, DOWN, coor_mask=[1, 0, 0])
        lower_scale_factor = square_extra[0].get_height() / square_lower[0].get_height()
        square_lower.scale(lower_scale_factor)
        square_lower.next_to(ineq_lower, DOWN)
        for source, ref in zip(parts_lower, parts_extra):
            source.next_to(ref, DOWN, coor_mask=[1, 0, 0])
            source.match_color(ref[0])
        parts_lower[3:].set_color(BOUND_COLOR)

        square_upper = Tex(
            """
            s_{i+j}^{2} < s_{i}^{2}
            \\frac{i}{i-1} \\frac{i+1}{i} \\frac{i+2}{i+1}
            \\cdots \\frac{i+j-2}{i+j-3} \\frac{i+j-1}{i+j-2}
            """
        )
        parts_upper = VGroup(
            square_upper[start:end]
            for start, end in zip(
                [0, 5, 6, 9, 14, 19, 26, 29, 40],
                [5, 6, 9, 14, 19, 26, 29, 40, 51],
            )
        )
        ineq_upper = ineq_lower.copy()
        ineq_upper.next_to(parts_extra, UP, coor_mask=[0, 1, 0])
        upper_scale_factor = square_extra[0].get_height() / square_upper[0].get_height()
        square_upper.scale(upper_scale_factor)
        square_upper.next_to(ineq_upper, UP)
        for source, ref in zip(parts_upper, parts_extra):
            source.next_to(ref, UP, coor_mask=[1, 0, 0])
            source.match_color(ref[0])
        parts_upper[3:].set_color(BOUND_COLOR)
        bg_rects = VGroup(
            SurroundingRectangle(
                VGroup(a, b, c), color=color,
                stroke_width=0, fill_opacity=0.15,
                buff=0.05,
            )
            for a, b, c, color in zip(
                parts_upper, parts_extra, parts_lower,
                it.cycle([GREEN_A, GREEN_E])
            )
        )
        for rect in bg_rects:
            rect.set_height(
                VGroup(parts_upper, parts_lower).get_height() * 1.1,
                stretch=True,
            )
        bg_rects.arrange(RIGHT, coor_mask=[0, 1, 0], center=False)
        for rect in bg_rects:
            rect.generate_target()
            rect.stretch(1e-1, dim=1).set_opacity(0)

        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        FadeIn(part_lower, shift=DOWN),
                        FadeIn(symbol_lower, shift=0.5 * DOWN),
                        FadeIn(symbol_upper, shift=0.5 * UP),
                        FadeIn(part_upper, shift=UP),
                        MoveToTarget(rect),
                    )
                    for part_lower, part_upper, symbol_lower, symbol_upper, rect in zip(
                        parts_lower[3:], parts_upper[3:], ineq_lower[3:], ineq_upper[3:],
                        bg_rects[3:]
                    )
                ],
                lag_ratio=0.05,
                run_time=2,
            )
        )
        self.wait()

        # Reveal the bound inequlities
        for parts in (parts_lower, parts_upper):
            parts.generate_target()
            parts.target.scale(1.5)
            parts.target.arrange(RIGHT, coor_mask=[1, 0, 0], center=False, buff=0.15)
            parts.target[2:].arrange(RIGHT, coor_mask=[1, 0, 0], center=False, buff=0.3)
            parts.target.next_to(TOP, DOWN, coor_mask=[1, 0, 0])
            parts[:3].set_opacity(0).move_to(parts_extra[:3])
        parts_lower.target.next_to(parts_extra, DOWN, coor_mask=[0, 1, 0])
        parts_upper.target.next_to(parts_extra, UP, coor_mask=[0, 1, 0])

        self.play(
            AnimationGroup(
                FadeOut(bg_rects[3:]),
                FadeOut(ineq_lower, shift=0.5 * DOWN),
                FadeOut(ineq_upper, shift=0.5 * UP),
                run_time=1,
            ),
            AnimationGroup(
                MoveToTarget(parts_lower),
                MoveToTarget(parts_upper),
                parts_extra.animate.set_opacity(0.2),
                factor_bound.animate.set_opacity(0.2),
                run_time=2,
            ),
        )
        self.wait()

        # Most factors cancel, only one survives
        cross_config = {
            "stroke_width": [2, 6, 2],
            "stroke_color": GREEN,
        }
        crosses_lower = VGroup(
            Cross(
                square_lower[start:start + length],
                **cross_config,
            )
            for start, length in zip(
                [9, 18, 14, 25, 21, 28, 37, 31, 46],
                [3, 3, 3, 3, 3, 3, 5, 5, 5],
            )
        )
        crosses_upper = VGroup(
            Cross(
                square_upper[start:start + length],
                **cross_config,
            )
            for start, length in zip(
                [9, 18, 14, 23, 19, 26, 35, 29, 46],
                [1, 1, 3, 3, 3, 3, 5, 5, 5],
            )
        )
        for crosses in (crosses_lower, crosses_upper):
            for cross in crosses:
                cross.remove(cross[0])
                cross.scale(1.2)
                cross.rotate(PI)

        self.play(
            AnimationGroup(
                AnimationGroup(
                    *[
                        ShowCreation(cross)
                        for cross in crosses
                    ],
                    lag_ratio=0.05,
                    run_time=2,
                )
                for crosses in (crosses_lower, crosses_upper)
            ),
        )
        self.wait()

        # Cleanup the coefficients...
        bound_lower = Tex("s_{i+j}^{2} > s_{i}^{2} \\cdot \\frac{i+j}{i}")
        bound_lower.move_to(square_lower)
        bound_upper = Tex("s_{i+j}^{2} < s_{i}^{2} \\cdot \\frac{i+j-1}{i-1}")
        bound_upper.move_to(square_upper)
        bounds = VGroup(bound_upper, bound_lower)
        bounds.arrange(DOWN, buff=1.5)
        for bound in bounds:
            bound[:5].set_color(FACTOR_COLOR)
            bound[6:9].set_color(WALLIS_COLOR)
            bound[10:].set_color(BOUND_COLOR)
        self.play(
            AnimationGroup(
                ReplacementTransform(square_lower[:9], bound_lower[:9]),
                ReplacementTransform(VectorizedPoint(square_lower[12].get_left()), bound_lower[9]),
                ReplacementTransform(square_lower[42:45], bound_lower[10:13]),
                ReplacementTransform(square_lower[12:14], bound_lower[13:]),
                ReplacementTransform(square_upper[:9], bound_upper[:9]),
                ReplacementTransform(VectorizedPoint(square_upper[10].get_left()), bound_upper[9]),
                ReplacementTransform(square_upper[40:45], bound_upper[10:15]),
                ReplacementTransform(square_upper[10:14], bound_upper[15:]),
                run_time=2,
            ),
            FadeOut(
                VGroup(
                    square_lower[9:12], square_lower[14:42], square_lower[45:],
                    square_upper[9:10], square_upper[14:40], square_upper[45:],
                    crosses_lower, crosses_upper,
                    parts_extra, factor_bound,
                ),
                run_time=1,
            ),
        )
        self.add(bound_lower, bound_upper)
        self.wait()

        # ... and flip it for convienience
        bound_lower_flip = Tex("s_{i}^{2} < s_{i+j}^{2} \\cdot \\frac{i}{i+j}")
        bound_lower_flip.move_to(bound_lower)
        bound_upper_flip = Tex("s_{i}^{2} > s_{i+j}^{2} \\cdot \\frac{i-1}{i+j-1}")
        bound_upper_flip.move_to(bound_upper)
        bounds_flip = VGroup(bound_upper_flip, bound_lower_flip)
        for bound in bounds_flip:
            bound[:3].set_color(WALLIS_COLOR)
            bound[4:9].set_color(FACTOR_COLOR)
            bound[10:].set_color(BOUND_COLOR)
        symbol_upper = bound_upper[5]
        symbol_upper.generate_target()
        symbol_upper.target.rotate(PI).move_to(bound_upper_flip[3])
        symbol_lower = bound_lower[5]
        symbol_lower.generate_target()
        symbol_lower.target.rotate(PI).move_to(bound_lower_flip[3])
        self.play(
            AnimationGroup(
                ReplacementTransform(bound_upper[6:9], bound_upper_flip[:3], path_arc=PI),
                MoveToTarget(symbol_upper, path_arc=PI),
                ReplacementTransform(bound_upper[:5], bound_upper_flip[4:9], path_arc=PI),
                ReplacementTransform(bound_upper[9], bound_upper_flip[9]),
                ReplacementTransform(bound_upper[16:], bound_upper_flip[10:13], path_arc=PI),
                ReplacementTransform(bound_upper[15], bound_upper_flip[13]),
                ReplacementTransform(bound_upper[10:15], bound_upper_flip[14:], path_arc=PI),
                ReplacementTransform(bound_lower[6:9], bound_lower_flip[:3], path_arc=PI),
                MoveToTarget(symbol_lower, path_arc=PI),
                ReplacementTransform(bound_lower[:5], bound_lower_flip[4:9], path_arc=PI),
                ReplacementTransform(bound_lower[9], bound_lower_flip[9]),
                ReplacementTransform(bound_lower[14:], bound_lower_flip[10:11], path_arc=PI),
                ReplacementTransform(bound_lower[13], bound_lower_flip[11]),
                ReplacementTransform(bound_lower[10:13], bound_lower_flip[12:], path_arc=PI),
                run_time=2,
            ),
        )
        self.remove(bound_upper, bound_lower)
        self.add(bound_upper_flip, bound_lower_flip)
        self.wait()

        # Start proving the inequalities we need
        sep_line = DashedLine(BOTTOM, TOP)
        sep_line.set_color(GREY)
        bottom_line = Line(LEFT_SIDE, RIGHT_SIDE)
        bottom_line.shift(3 * DOWN)
        inner_bg_rect = FullScreenRectangle(
            fill_color=INNER_CORNER_COLOR,
            fill_opacity=0.05,
        )
        inner_bg_rect.stretch(0.5, 0)
        inner_bg_rect.move_to(LEFT_SIDE / 2)
        outer_bg_rect = inner_bg_rect.copy()
        outer_bg_rect.set_color(OUTER_CORNER_COLOR)
        outer_bg_rect.move_to(RIGHT_SIDE / 2)

        subscript_colormap = {
            "{i}": SUB_I_COLOR,
            "{j}": SUB_J_COLOR,
        }
        bound_lower_sub = Tex(
            "s_{{i}}^{2} < s_{{i}+{j}}^{2} \\cdot \\frac{{i}}{{i}+{j}}",
            t2c=subscript_colormap,
        )
        bound_lower_sub.move_to(inner_bg_rect).to_edge(UP)
        bound_upper_sub = Tex(
            "s_{{i}}^{2} > s_{{i}+{j}}^{2} \\cdot \\frac{{i}-1}{{i}+{j}-1}",
            t2c=subscript_colormap,
        )
        bound_upper_sub.move_to(outer_bg_rect).to_edge(UP)

        inner_ineq = Tex("s_{k}^{2} + s_{n-k}^{2} \\overset{?}{<} s_{n}^{2}")
        inner_ineq.set_color(INNER_CORNER_COLOR)
        inner_ineq.move_to(inner_bg_rect).next_to(bottom_line, UP, coor_mask=[0, 1, 0])
        outer_ineq = Tex("s_{k}^{2} + s_{n+1-k}^{2} \\overset{?}{>} s_{n}^{2}")
        outer_ineq.set_color(OUTER_CORNER_COLOR)
        outer_ineq.move_to(outer_bg_rect).next_to(bottom_line, UP, coor_mask=[0, 1, 0])
        self.play(
            FadeIn(VGroup(inner_bg_rect, outer_bg_rect)),
            FadeIn(VGroup(inner_ineq, outer_ineq)),
            ShowCreation(sep_line),
            ReplacementTransform(bound_lower_flip, bound_lower_sub),
            ReplacementTransform(bound_upper_flip, bound_upper_sub),
            run_time=2,
        )
        self.wait()

        # Left side - inner inequality
        part_scale_factor = 0.7
        value_buff = 0.3
        inner_k_colormap = {
            "{k}": SUB_I_COLOR,
            "{(n-k)}": SUB_J_COLOR,
        }
        inner_k_tex = Tex(
            "s_{{k}}^{2} < s_{{k}+{(n-k)}}^{2} \\cdot \\frac{{k}}{{k}+{(n-k)}}",
            t2c=inner_k_colormap,
        )
        inner_k_value = VGroup(
            Tex("i=k").set_color(SUB_I_COLOR),
            Tex("j=n-k").set_color(SUB_J_COLOR),
        )
        inner_k_value.arrange(DOWN, aligned_edge=LEFT, buff=value_buff)
        inner_k_brace = Brace(inner_k_value, RIGHT)
        inner_k_group = VGroup(inner_k_value, inner_k_brace, inner_k_tex)
        inner_k_group.arrange(RIGHT)

        inner_nmk_colormap = {
            "{(n-k)}": SUB_I_COLOR,
            "{n-k}": SUB_I_COLOR,
            "{k}": SUB_J_COLOR,
        }
        inner_nmk_tex = Tex(
            "s_{n-k}^{2} < s_{{(n-k)}+{k}}^{2} \\cdot \\frac{n-k}{{(n-k)}+{k}}",
            t2c=inner_nmk_colormap,
        )
        inner_nmk_value = VGroup(
            Tex("i=n-k").set_color(SUB_I_COLOR),
            Tex("j=k").set_color(SUB_J_COLOR),
        )
        inner_nmk_value.arrange(DOWN, aligned_edge=LEFT, buff=value_buff)
        inner_nmk_brace = Brace(inner_nmk_value, RIGHT)
        inner_nmk_group = VGroup(inner_nmk_value, inner_nmk_brace, inner_nmk_tex)
        inner_nmk_group.arrange(RIGHT)
        inner_group = VGroup(inner_k_group, inner_nmk_group)
        inner_group.arrange(DOWN, aligned_edge=LEFT, buff=1.5)
        inner_group.scale(part_scale_factor)
        inner_group.move_to(LEFT_SIDE / 2)

        inner_k_part = Tex("s_{k}^{2} < s_{n}^{2} \\cdot \\frac{k}{n}")
        inner_k_part.set_color(INNER_CORNER_COLOR)
        inner_k_part.scale(part_scale_factor)
        inner_k_part.next_to(inner_k_tex.get_left(), RIGHT, buff=0)
        inner_nmk_part = Tex("s_{n-k}^{2} < s_{n}^{2} \\cdot \\frac{n-k}{n}")
        inner_nmk_part.set_color(INNER_CORNER_COLOR)
        inner_nmk_part.scale(part_scale_factor)
        inner_nmk_part.next_to(inner_nmk_tex.get_left(), RIGHT, buff=0)

        self.play(FadeIn(inner_k_group, shift=DOWN))
        self.wait()
        self.play(FadeIn(inner_nmk_group, shift=2 * DOWN))
        self.wait()
        self.play(
            ReplacementTransform(inner_k_tex[:6], inner_k_part[:6]),
            FadeTransform(inner_k_tex[6:13], inner_k_part[6:7]),
            ReplacementTransform(inner_k_tex[13], inner_k_part[7]),
            ReplacementTransform(inner_k_tex[14:16], inner_k_part[8:10]),
            FadeTransform(inner_k_tex[16:], inner_k_part[10:]),
            run_time=2,
        )
        self.remove(inner_k_tex)
        self.add(inner_k_part)
        inner_k_group = VGroup(inner_k_value, inner_k_brace, inner_k_part)
        self.wait()
        self.play(
            ReplacementTransform(inner_nmk_tex[:8], inner_nmk_part[:8]),
            FadeTransform(inner_nmk_tex[8:15], inner_nmk_part[8:9]),
            ReplacementTransform(inner_nmk_tex[15], inner_nmk_part[9]),
            ReplacementTransform(inner_nmk_tex[16:20], inner_nmk_part[10:14]),
            FadeTransform(inner_nmk_tex[20:], inner_nmk_part[14:]),
            run_time=2,
        )
        self.remove(inner_nmk_tex)
        self.add(inner_nmk_part)
        inner_nmk_group = VGroup(inner_nmk_value, inner_nmk_brace, inner_nmk_part)
        self.wait()

        inner_checkmark = Tex("\\checkmark")
        inner_checkmark.set_color(INNER_CORNER_COLOR)
        inner_checkmark.next_to(inner_ineq, RIGHT, buff=0.5)
        self.play(
            ReplacementTransform(
                inner_ineq[-5], inner_checkmark,
                path_arc=-PI / 2,
            ),
            VGroup(inner_k_group, inner_nmk_group).animate.move_to(LEFT_SIDE / 2),
        )
        self.wait()

        # Right side - outer inequality
        outer_k_part = Tex("s_{k}^{2} > s_{n}^{2} \\cdot \\frac{k-1}{n-1}")
        outer_k_part.set_color(OUTER_CORNER_COLOR)
        outer_k_value = VGroup(
            Tex("i=k").set_color(SUB_I_COLOR),
            Tex("j=n-k").set_color(SUB_J_COLOR),
        )
        outer_k_value.arrange(DOWN, aligned_edge=LEFT, buff=value_buff)
        outer_k_brace = Brace(outer_k_value, RIGHT)
        outer_k_group = VGroup(outer_k_value, outer_k_brace, outer_k_part)
        outer_k_group.arrange(RIGHT)

        outer_nmkp1_part = Tex("s_{n-k+1}^{2} > s_{n}^{2} \\cdot \\frac{n-k}{n-1}")
        outer_nmkp1_part.set_color(OUTER_CORNER_COLOR)
        outer_nmkp1_value = VGroup(
            Tex("i=n-k+1").set_color(SUB_I_COLOR),
            Tex("j=k-1").set_color(SUB_J_COLOR),
        )
        outer_nmkp1_value.arrange(DOWN, aligned_edge=LEFT, buff=value_buff)
        outer_nmkp1_brace = Brace(outer_nmkp1_value, RIGHT)
        outer_nmkp1_group = VGroup(outer_nmkp1_value, outer_nmkp1_brace, outer_nmkp1_part)
        outer_nmkp1_group.arrange(RIGHT)
        outer_group = VGroup(outer_k_group, outer_nmkp1_group)
        outer_group.arrange(DOWN, aligned_edge=LEFT, buff=1.5)
        outer_group.scale(part_scale_factor)
        outer_group.move_to(RIGHT_SIDE / 2)

        self.play(
            FadeIn(outer_k_group, shift=DOWN),
            FadeIn(outer_nmkp1_group, shift=2 * DOWN),
        )
        self.wait()

        outer_checkmark = Tex("\\checkmark")
        outer_checkmark.set_color(OUTER_CORNER_COLOR)
        outer_checkmark.next_to(outer_ineq, RIGHT, buff=0.5)
        self.play(
            ReplacementTransform(
                outer_ineq[-5], outer_checkmark,
                path_arc=-PI / 2,
            ),
        )
        self.wait()

        # The restrictions on subscript i limit the choices of k
        self.play(
            AnimationGroup(
                *[
                    FlashAround(value[0], color=SUB_I_COLOR)
                    for value in (
                        inner_k_value, inner_nmk_value,
                        outer_k_value, outer_nmkp1_value
                    )
                ],
                lag_ratio=0.1,
            ),
            run_time=3,
        )
        self.wait()
        problem = TexText("（思考：$k=1$时，上面的不等式并不成立，但不影响最终结果。为什么?）")
        problem.set_color(GREY)
        problem.set_width(FRAME_WIDTH / 2 - 0.5)
        problem.next_to(outer_nmkp1_group, DOWN, buff=0.2)
        self.play(FadeIn(problem))
        self.wait()


class OutroSceneWallis(Scene):
    def construct(self):
        frame_rect = ScreenRectangle()
        frame_rect.set_height(6)
        title = TexText("总结")
        title.next_to(frame_rect, UP)
        self.add(title, frame_rect)
        self.wait()


class MoreOnWallisProduct(Scene):
    def construct(self):
        # Former contributions
        portraits = Group(
            ImageMobject(image_name)
            for image_name in ("JohanWastlund", "DonaldKnuth", "AbaMbirika")
        )
        names = VGroup(
            TexText(name)
            for name in ("Johan Wastlund", "Donald Knuth", "Aba Mbirika")
        )
        for image in portraits:
            image.set_height(5)
        portraits.arrange(RIGHT, buff=0.5)
        # For some reason, 'ä' doesn't fill
        # Needs a little patch... literally
        dots = Dot().replicate(2)
        dots.set_height(0.045)
        dots.arrange(RIGHT, buff=0.05)
        dots.next_to(names[0][6], UP, buff=0.05)
        names[0].add(dots)
        for name, image in zip(names, portraits):
            name.next_to(image, UP)
            name.scale(0.9)
            name.save_state()
            image.save_state()
        former_group = Group(portraits, names)
        self.add(former_group)
        self.wait()

        article = ImageMobject("wallis_articles")
        article.set_height(4).to_edge(RIGHT)
        self.play(
            former_group.animate.scale(0.6).to_edge(LEFT),
            GrowFromEdge(article, RIGHT),
        )
        self.wait()

        # Added something in the opening for clarification
        work_part = TexText("内容来源于前人的工作$\\dots$", font_size=42)
        work_part.next_to(names, UP, aligned_edge=RIGHT)
        summary_part = TexText("$\\dots$及其总结", font_size=42)
        summary_part.next_to(article, UP, aligned_edge=LEFT)
        for part in (work_part, summary_part):
            part.set_color(YELLOW)
            part.to_edge(UP, buff=0.8)
        self.add(work_part, summary_part)
        self.wait()
        self.remove(work_part, summary_part)
        self.wait()

        # Johan's original idea
        unit_size = 1.2
        wallis_rects = WallisRectangles(
            N=6, colors=color_gradient(WALLIS_RECTS_COLORS, 11)[:6],
        )
        wallis_rects.set_unit_size(unit_size)
        wallis_rects.center()
        sector = wallis_rects.get_cover_sector()
        sector.set_fill(opacity=0)
        length_labels = wallis_rects.get_length_labels()
        wallis_group = VGroup(wallis_rects, length_labels, sector)
        wallis_group.to_edge(RIGHT)

        self.play(
            AnimationGroup(
                Restore(portraits[0]),
                Restore(names[0]),
                FadeIn(wallis_group),
                run_time=2,
            ),
            AnimationGroup(
                FadeOut(article),
                FadeOut(portraits[1:]),
                FadeOut(names[1:]),
                run_time=1,
            )
        )
        self.wait()

        prob_temp = WallisRectangles(
            N=11, colors=WALLIS_RECTS_COLORS,
        )
        prob_temp.set_unit_size(unit_size)
        layer_6 = prob_temp.get_layers(range(6))
        max_x = layer_6.get_right()[0]
        max_y = layer_6.get_top()[1]
        prob_diagram = VGroup()
        for layer in prob_temp.get_all_layers():
            included_rects = VGroup(
                rect
                for rect in layer
                if (rect.get_x() < max_x) and (rect.get_y() < max_y)
            )
            prob_diagram.add(included_rects)
        prob_diagram.center().next_to(wallis_rects, LEFT, buff=2)
        horiz_dices = VGroup(Die(k) for k in range(1, 7))
        vert_dices = horiz_dices.copy()
        for hd, vd, layer in zip(horiz_dices, vert_dices, prob_temp.get_layers(range(6))):
            hd.next_to(layer[-1], DOWN, buff=0.2)
            vd.next_to(layer[0], LEFT, buff=0.2)
        prob_group = VGroup(prob_diagram, horiz_dices, vert_dices)
        arrow = Tex("\\Rightarrow")
        arrow.move_to((prob_group.get_right() + wallis_group.get_left()) / 2)

        self.play(
            FadeIn(VGroup(prob_group, arrow), shift=DOWN),
        )
        self.wait()

        # a_n can be obtained using generating function fairly easily
        an_formula = Tex("a_{n} = {{\\frac{1}{2}+n-1} \\choose {n}}")
        an_formula.next_to(wallis_group, RIGHT, aligned_edge=UP, buff=0)
        all_things = Group(
            portraits[0], names[0],
            prob_group, arrow, wallis_group, an_formula,
        )
        all_things.generate_target()
        all_things.target[:2].set_opacity(0)
        shift_vec = -all_things.target[2:].get_center()
        all_things.target.shift(shift_vec)
        all_things[-1].set_opacity(0)
        self.play(
            MoveToTarget(all_things),
            run_time=2,
        )
        self.remove(portraits, names)
        self.wait()

        # We can extend this idea to other 'circles'
        an_2 = an_formula
        dim_2 = Tex("x^2 + y^2 = R^2")
        dim_2.set_color(GREY)
        dim_2.next_to(wallis_group, UP, aligned_edge=RIGHT)
        wallis_rects_2 = WallisRectangles(
            N=20, colors=WALLIS_RECTS_COLORS,
        )
        wallis_rects_2.match_width(wallis_rects)
        wallis_rects_2.put_corner_at(wallis_rects.get_rects_corner())
        move_group = VGroup(
            prob_group, arrow,
            wallis_rects, length_labels, sector,
            wallis_rects_2, dim_2,
        )
        move_group[-2:].set_opacity(0)
        move_group.generate_target()
        move_group.target[:-1].scale(1.8)
        shift_vec = (LEFT_SIDE + 0.5 * RIGHT) - move_group.target[2].get_left()
        move_group.target.shift(shift_vec)
        move_group.target[:4].set_opacity(0)
        move_group.target[-2:].set_opacity(1)
        move_group.target[-1].next_to(move_group.target[2], UP, aligned_edge=RIGHT, buff=0)
        arrow = Tex("\\Downarrow")
        arrow.move_to(RIGHT_SIDE / 2)
        wallis_prod_2 = Tex(
            """
            \\frac{2}{1} \\cdot \\frac{2}{3} \\cdot \\frac{4}{3} \\cdot \\frac{4}{5} 
            \\cdot \\frac{6}{5} \\cdot \\frac{6}{7} \\cdots = \\frac{\\pi}{2}
            """,
        )
        wallis_prod_2.next_to(arrow, DOWN, buff=0.5)
        wallis_prod_2.set_color(YELLOW)
        an_2.generate_target()
        an_2.target.next_to(arrow, UP, buff=0.5)
        self.play(
            MoveToTarget(move_group),
            MoveToTarget(an_2),
            FadeIn(arrow, shift=LEFT),
            FadeIn(wallis_prod_2, shift=LEFT),
            run_time=1,
        )
        self.remove(*self.mobjects)
        self.add(
            wallis_rects_2, sector, dim_2,
            an_2, arrow, wallis_prod_2,
        )
        self.wait()

        an_colormap = {
            "\\frac{1}{3}": RED,
            "\\frac{1}{4}": RED,
        }
        an_3 = Tex("a_{n} = {{\\frac{1}{3}+n-1} \\choose {n}}", t2c=an_colormap)
        an_4 = Tex("a_{n} = {{\\frac{1}{4}+n-1} \\choose {n}}", t2c=an_colormap)
        for an in (an_3, an_4):
            an.next_to(arrow, UP, buff=0.5)
        dim_colormap = {
            "{3}": RED_E,
            "{4}": RED_E,
        }
        dim_3 = Tex("x^{3} + y^{3} = R^{3}", t2c=dim_colormap)
        dim_4 = Tex("x^{4} + y^{4} = R^{4}", t2c=dim_colormap)
        for dim in (dim_3, dim_4):
            dim.set_color(GREY)
            dim.move_to(dim_2)
            for key, color in dim_colormap.items():
                dim.get_parts_by_tex(key).set_color(color)
        wallis_rects_3 = WallisRectangles(
            N=20, colors=WALLIS_RECTS_COLORS, alpha=1 / 3,
        )
        wallis_rects_4 = WallisRectangles(
            N=20, colors=WALLIS_RECTS_COLORS, alpha=1 / 4,
        )
        for wr in (wallis_rects_3, wallis_rects_4):
            wr.match_width(wallis_rects_2)
            wr.put_corner_at(wallis_rects_2.get_rects_corner())
        sector_3 = wallis_rects_3.get_cover_sector()
        sector_4 = wallis_rects_4.get_cover_sector()
        for s in (sector_3, sector_4):
            s.set_fill(opacity=0)
        wallis_prod_3 = Tex(
            """
            \\frac{2}{1} \\cdot \\frac{3}{4} \\cdot \\frac{5}{4} \\cdot \\frac{6}{7} 
            \\cdot \\frac{8}{7} \\cdot \\frac{9}{10} \\cdots
            = \\frac{\\Gamma^3 \\big( \\frac{1}{3} \\big)}{2 \\sqrt{3} \\pi}
            """,
        )
        wallis_prod_4 = Tex(
            """
            \\frac{2}{1} \\cdot \\frac{4}{5} \\cdot \\frac{6}{5} \\cdot \\frac{8}{9} 
            \\cdot \\frac{10}{9} \\cdot \\frac{12}{13} \\cdots
            = \\frac{\\Gamma^2 \\big( \\frac{1}{4} \\big)}{4 \\sqrt{\\pi}}
            """,
        )
        for wp in (wallis_prod_3, wallis_prod_4):
            wp.scale(0.9)
            wp.next_to(arrow, DOWN, buff=0.5)
            wp.set_color(YELLOW)
        gamma_note = TexText("（$\\Gamma(\\cdot)$为伽马函数）")
        gamma_note.set_color(GREY).scale(0.6)
        gamma_note.next_to(wallis_prod_3, DOWN, aligned_edge=RIGHT)

        self.play(
            ReplacementTransform(wallis_rects_2, wallis_rects_3),
            ReplacementTransform(sector, sector_3),
            ReplacementTransform(dim_2, dim_3),
            ReplacementTransform(an_2, an_3),
            FadeTransform(wallis_prod_2, wallis_prod_3),
            FadeIn(gamma_note),
        )
        self.wait(2)
        self.play(
            ReplacementTransform(wallis_rects_3, wallis_rects_4),
            ReplacementTransform(sector_3, sector_4),
            ReplacementTransform(dim_3, dim_4),
            ReplacementTransform(an_3, an_4),
            FadeTransform(wallis_prod_3, wallis_prod_4),
            gamma_note.animate.next_to(wallis_prod_4, DOWN, aligned_edge=RIGHT)
        )
        self.wait(2)

        # Higher dimesional cases - Blender scene
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait()

        # Simple trick for general cases
        colors = [BLUE, GREEN, YELLOW]
        example = Tex("\\text{例：} \\prod_{k=1}^{+\\infty}{\\frac{(2k)(2k)}{(2k-1)(2k+1)}} = \\frac{\\pi}{2}")
        example[2:-4].set_color(colors[0])
        example[-3:].set_color(colors[-1])
        steps = VGroup(
            Tex(text)
            for text in (
                "\\text{(1) 去系数：} \\frac{(k)(k)}{\\big(k-\\frac{1}{2}\\big)\\big(k+\\frac{1}{2}\\big)}",
                "\\text{(2) $k$变$1$：} \\frac{(1)(1)}{\\big(1-\\frac{1}{2}\\big)\\big(1+\\frac{1}{2}\\big)}",
                "\\text{(3) 补伽马：} \\frac{\\Gamma(1)\\Gamma(1)}{\\Gamma\\big(1-\\frac{1}{2}\\big)\\Gamma\\big(1+\\frac{1}{2}\\big)}",
                "\\text{(4) 取倒数：} \\frac{\\Gamma\\big(1-\\frac{1}{2}\\big)\\Gamma\\big(1+\\frac{1}{2}\\big)}{\\Gamma(1)\\Gamma(1)}",
                "\\text{(5) 算结果：} \\frac{\\pi}{2}",
            )
        )
        steps.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        steps.set_color_by_gradient(*colors)
        trick = VGroup(example, steps)
        trick.arrange(DOWN, buff=0.8)
        conditions = VGroup(
            TexText("（要求：分子项数$=$分母项数）"),
            TexText("（要求：分子各项和$=$分母各项和）"),
        )
        conditions.set_color(GREY)
        conditions.scale(0.5)
        for cond, step in zip(conditions, steps):
            cond.next_to(step, RIGHT)
        trick.add(conditions)
        trick.set_height(FRAME_HEIGHT - 1.5)
        # trick.next_to(ORIGIN, RIGHT)
        trick.move_to(RIGHT_SIDE / 2)
        trick.to_edge(UP)
        underline = Underline(example)
        underline.set_color(GREY)
        trick.add(underline)
        self.play(FadeIn(trick))
        self.wait()

        # Show some interesting products to end the video -
        # the golden ratio, a cubic root, and a trig function
        example_expand = Tex(
            """
            \\frac{2}{1} \\cdot \\frac{2}{3} \\cdot \\frac{4}{3} \\cdot \\frac{4}{5} \\cdot
            \\frac{6}{5} \\cdot \\frac{6}{7} \\cdot \\frac{8}{7} \\cdot \\frac{8}{9} \\cdots
            = \\frac{\\pi}{2}
            """,
        )
        scale_factor = example[-1].get_height() / example_expand[-1].get_height()
        example_expand.scale(scale_factor)
        example_expand[:-4].set_color(colors[0])
        example_expand[-3:].set_color(colors[-1])
        trick.generate_target()
        for mob in trick.target[-2:]:
            mob.set_opacity(0).scale(1e-2, about_edge=RIGHT)
        for step in trick.target[1]:
            step[6:].set_opacity(0).scale(1e-2, about_edge=RIGHT)
        instructions = VGroup(
            step[:6]
            for step in trick.target[1]
        )
        instructions.arrange(DOWN, aligned_edge=LEFT)
        instructions.next_to(RIGHT_SIDE / 2, DOWN, buff=0)
        example_expand.next_to(instructions, UP, buff=1)
        trick.target[0].next_to(example_expand, UP, buff=0.5)
        self.play(
            MoveToTarget(trick),
            FadeIn(example_expand),
        )
        instructions = VGroup(
            step[:6]
            for step in trick[1]
        )
        self.remove(trick)
        self.add(example, instructions)
        self.wait()

        products = VGroup(
            Tex(
                """
                \\prod_{k=1}^{+\\infty}{\\frac{(5k-3)(5k-2)}{(5k-4)(5k-1)}}
                = \\frac{\\sqrt{5}+1}{2}
                """,
            ),
            Tex(
                """
                \\prod_{k=1}^{+\\infty}{\\frac{(6k-4)(6k-3)}{(6k-5)(6k-2)}}
                = \\sqrt[3]{4}
                """,
            ),
            Tex(
                """
                \\prod_{k=1}^{+\\infty}{\\frac{(7k-4)(7k-3)}{(7k-6)(7k-1)}}
                = \\frac{1}{2} \\csc \\frac{\\pi}{14}
                """,
            ),
        )
        products_expand = VGroup(
            Tex(
                """
                \\frac{2}{1} \\cdot \\frac{3}{4} \\cdot \\frac{7}{6} \\cdot \\frac{8}{9} \\cdot
                \\frac{12}{11} \\cdot \\frac{13}{14} \\cdot \\frac{17}{16} \\cdot \\frac{18}{19} \\cdots
                = \\frac{\\sqrt{5}+1}{2}
                """,
            ),
            Tex(
                """
                \\frac{2}{1} \\cdot \\frac{3}{4} \\cdot \\frac{8}{7} \\cdot \\frac{9}{10} \\cdot
                \\frac{14}{13} \\cdot \\frac{15}{16} \\cdot \\frac{20}{19} \\cdot \\frac{21}{22} \\cdots
                = \\sqrt[3]{4}
                """,
            ),
            Tex(
                """
                \\frac{3}{1} \\cdot \\frac{4}{6} \\cdot \\frac{10}{8} \\cdot \\frac{11}{13} \\cdot
                \\frac{17}{15} \\cdot \\frac{18}{20} \\cdot \\frac{24}{22} \\cdot \\frac{25}{27} \\cdots
                = \\frac{1}{2} \\csc \\frac{\\pi}{14}
                """,
            ),
        )
        end_lengths = [
            7,
            4,
            10,
        ]
        prev_prod = example
        prev_prod_expand = example_expand
        for prod, prod_expand, L in zip(products, products_expand, end_lengths):
            for mob in (prod, prod_expand):
                mob.scale(scale_factor)
                mob[:-(L + 1)].set_color(colors[0])
                mob[-L:].set_color(colors[-1])
            prod.move_to(example)
            prod_expand.move_to(example_expand)
            self.play(
                FadeTransform(prev_prod, prod),
                FadeTransform(prev_prod_expand, prod_expand),
            )
            prev_prod = prod
            prev_prod_expand = prod_expand
            self.wait(1.5)


class ReferencesWallis(ReferenceScene):
    def get_references(self):
        wastlund_ref = Reference(
            name='An Elementary Proof of the Wallis Product Formula for pi',
            authors='Johan Wastlund',
            pub='The American Mathematical Monthly, 2007, 114(10): 914-917.',
            doi='https://www.math.chalmers.se/~wastlund/monthly.pdf',
            info='''
            沃利斯乘积的初等证明。矩形图案逼近圆的几何构造方法就来自这篇文章。其中也包括了
            卷积性质和矩形-扇形大小关系的初等证明。\\\\
            它的中文版可以在《数学译林》2015年第1期上找到。
            ''',
        )
        # Patch for Johan's name again XD
        authors = wastlund_ref[1]
        dots = Dot().replicate(2)
        dots.set_height(0.02)
        dots.arrange(RIGHT, buff=0.015)
        dots.next_to(authors[6], UP, buff=0.015)
        dots.match_color(authors)
        wastlund_ref[1].add(dots)

        knuth_ref = Reference(
            name='Stanford Lecture: Donald Knuth - ``Why Pi?"(2010)',
            authors='Stanford Online',
            pub='视频',
            doi='https://www.youtube.com/watch?v=cI6tt9QfRdo',
            info='''
            Donald Knuth在斯坦福的讲座，内容很详细，非常值得一看。原本讲的是卡塔兰数
            为什么和$\\pi$有关，但是它的${2n \\choose n}$因子同样牵扯到沃利斯乘积。
            ''',
        )

        aba_ref = Reference(
            name='''
            The Wallis Product, a Connection to Pi and Probability, \\\\
            and Maybe the Gamma Function Perhaps?
            ''',
            authors='Aba Mbirika',
            pub='.',
            doi='https://people.uwec.edu/mbirika/M395_Riemann/Riemann_class_Pi_talk-HANDOUT.pdf',
            info='.',
        )
        aba_ref[-3::2].set_opacity(0)

        coin_flip_ref = Reference(
            name='On a coin-flip problem and its connection to pi',
            authors='Aba Mbirika',
            pub='.',
            doi='https://people.uwec.edu/mbirika/CoinFlipProblem.pdf',
            info='连同上面的参考资料，是沃利斯乘积证明的文字版。',
        )
        coin_flip_ref[-3].set_opacity(0)

        zhihu_p1_ref = Reference(
            name='Wallis乘积中的$\\pi$从何而来？（上）',
            authors='Solara570',
            pub='.',
            doi='https://zhuanlan.zhihu.com/p/34313702',
            info='.',
        )
        zhihu_p1_ref[-3::2].set_opacity(0)

        zhihu_p2_ref = Reference(
            name='Wallis乘积中的$\\pi$从何而来？（下）',
            authors='Solara570',
            pub='.',
            doi='https://zhuanlan.zhihu.com/p/34600273',
            info='连同上面的参考资料一起，是对前面四篇参考资料的文字版总结，也是这期视频的主要内容。',
        )
        zhihu_p2_ref[-3].set_opacity(0)

        walk_proof_ref = Reference(
            name='Random Walks and Catalan Factorization',
            authors='\\"Omer E\\u gecio\\u glu',
            pub='.',
            doi='https://people.bath.ac.uk/masadk/papers/catalan.pdf',
            info='给出了“正点到家”路径数和“不再回家”路径数相等的证明思路。',
        )
        # Why is this name free of patches? Strange.
        walk_proof_ref[-3].set_opacity(0)

        loaded_dice_ref = Reference(
            name='Problems for mathematicians, young and old',
            authors='Paul R. Halmos',
            pub='书籍',
            doi='.',
            info='''
            问题7A就是结尾提到的概率题：是否存在两个作弊骰子，
            使得每种点数和（2到12）出现的概率都相同？
            ''',
        )
        loaded_dice_ref[-2].set_opacity(0)

        return (
            wastlund_ref, knuth_ref,
            aba_ref, coin_flip_ref,
            zhihu_p1_ref, zhihu_p2_ref,
            walk_proof_ref, loaded_dice_ref,
        )


class ThumbnailWallis(IntroToWallisProduct):
    # Same as the end of the intro, so no need to make thumbnail this time
    pass


# Obsolete?
class FormalProofOfPQBijection(Scene):
    def construct(self):
        claim = VGroup(
            TexText("集合$P$, $Q$之间存在映射$f: P \\to Q$, $g: Q \\to P$,"),
            TexText("若$\\forall x \\in P$, $g(f(x))=x$, 且$\\forall y \\in Q$, $f(g(y))=y$,"),
            TexText("则(1)$|P|=|Q|$, (2)$f,g$均为双射, (3)$f,g$互为逆映射."),
        )
        claim.arrange(DOWN, aligned_edge=LEFT)
        claim.set_color(YELLOW_A)
        claim.to_edge(UP)
        proof_mark = TexText("证明:")
        proof_main = VGroup(
            TexText("$\\forall x_1, x_2 \\in P$, 若$f(x_1)=f(x_2)$, 则$x_1=g(f(x_1))=g(f(x_2))=x_2$,"),
            TexText("因此$f$是单射."),
            TexText("$\\forall y \\in Q$, $\\exists x=g(y) \\in P$, 使得$f(x)=f(g(y))=y$,"),
            TexText("因此$f$是满射."),
            TexText("综上, $f$是双射, 所以$|P|=|Q|$."),
            TexText("同理$g$也是双射, 并且由题设的关系可知$f,g$互逆."),
        )
        proof_main.arrange(DOWN, aligned_edge=LEFT)
        proof = VGroup(proof_mark, proof_main)
        proof.scale(0.8)
        proof.arrange(RIGHT, aligned_edge=UP)
        proof.set_color(BLUE_A)
        proof.next_to(claim, DOWN, buff=0.75)
        self.add(claim, proof)
        self.wait()
