#coding=utf-8

import numpy as np
import itertools as it
from constants import *
from utils.bezier import interpolate
from utils.config_ops import digest_config
from utils.rate_functions import *

from animation.animation import Animation
from animation.composition import AnimationGroup
from animation.creation import ShowCreation, Write, FadeIn, FadeOut, DrawBorderThenFill, GrowFromCenter
from animation.transform import Transform, MoveToTarget, ApplyMethod
from animation.indication import Indicate
from mobject.types.vectorized_mobject import VMobject
from mobject.svg.tex_mobject import TexMobject, TextMobject
from mobject.svg.brace import Brace
from mobject.geometry import Square, Line, DashedLine
from mobject.shape_matchers import SurroundingRectangle
from scene.scene import Scene
from scene.three_d_scene import ThreeDScene

from custom.custom_helpers import *
from custom.custom_mobjects import FakeQEDSymbol

# self.skip_animations
# self.force_skipping()
# self.revert_to_original_skipping_status()

#####
## Functions

def generate_pattern(dim):
    """Generate a random pattern of Calisson tilling with a given dimension ``dim``."""
    pattern = np.random.randint(dim + 1, size = (dim, dim))
    # Try to make the tilling more balanced...
    for k in random.sample(range(dim), max(1, dim // 3)):
        pattern.T[k].sort()
    pattern.sort()
    pattern.T.sort()
    return pattern[-1::-1, -1::-1]

#####
## Constants
# POS = (phi, theta, distance)
# To make it more convincing, the distance is set to be HUGE.

DIAG_POS  = (np.arctan(np.sqrt(2)), np.pi/4., 10000000)
UP_POS    = (0, 0, 10000000)
FRONT_POS = (np.pi/2., 0, 10000000)
RIGHT_POS = (np.pi/2., np.pi/2., 10000000)

AMM_PATTERN = np.array([
    [5, 5, 5, 4, 3],
    [5, 5, 5, 1, 0],
    [5, 5, 1, 0, 0],
    [5, 5, 1, 0, 0],
    [3, 3, 0, 0, 0],
])

MPACC_PATTERN = np.array([
    [7, 7, 7, 7, 6, 4, 3],
    [6, 6, 5, 5, 5, 2, 1],
    [6, 6, 4, 3, 2, 1, 0],
    [6, 5, 3, 2, 1, 1, 0],
    [5, 3, 2, 1, 1, 0, 0],
    [3, 2, 2, 1, 0, 0, 0],
    [3, 1, 0, 0, 0, 0, 0],
])

DARK_GRAY = "#404040"
DARK_GREY = DARK_GRAY
LIGHT_GRAY = "#B0B0B0"
LIGHT_GREY = LIGHT_GRAY

TILE_RED, TILE_GREEN, TILE_BLUE = map(darken, [RED, GREEN, BLUE])
TILE_COLOR_SET = [TILE_GREEN, TILE_RED, TILE_BLUE]
RHOMBI_COLOR_SET = [TILE_RED, TILE_GREEN, TILE_BLUE]


#####
## Mobjects

class CalissonTilling(VMobject):
    CONFIG = {
        "dimension" : 3,
        "pattern" : None,
        "enable_fill"  : False,
        "enable_shuffle" : False,        # FOR TRANSFORMATION ONLY!
        "height" : 4,
        "tile_colors" : TILE_COLOR_SET,  # UP-DOWN, OUT-IN, RIGHT-LEFT
        "fill_opacity" : 1,
        "tile_config": {
            "side_length"  : 1,
            "stroke_width" : 5,
            "stroke_color" : WHITE,
        },
        "border_config" : {
            "stroke_width" : 5,
            "stroke_color" : WHITE,
            "mark_paths_closed" : True,
        },
    }
    def __init__(self, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.init_tiles()
        self.init_pattern()
        self.init_border()
        if self.enable_shuffle:
            self.shuffle_tiles()
        self.add(self.border, self.tiles)
        self.adjust_size()
        
    def init_tiles(self):
        out_std = Square(**self.tile_config)
        out_std.shift((RIGHT+UP) / 2.)
        dim = self.dimension
        out_tiles = VGroup()
        for i in range(dim ** 2):
            tile = out_std.copy()
            tile.shift((i % dim) * UP + (i // dim) * RIGHT)
            out_tiles.add(tile)
        up_tiles = out_tiles.copy()
        up_tiles.rotate(-2.*np.pi/3., axis = [1, 1, 1], about_point = ORIGIN)
        right_tiles = out_tiles.copy()
        right_tiles.rotate(2.*np.pi/3., axis = [1, 1, 1], about_point = ORIGIN)

        all_tiles = [out_tiles, right_tiles, up_tiles]
        for tile_set, color in zip(all_tiles, self.tile_colors):
            for tile in tile_set:
                tile.set_fill(color = color, opacity = self.enable_fill)
        self.backup_tiles = VGroup(*all_tiles).copy()

    def init_pattern(self):
        if self.pattern is None:
            self.pattern = np.zeros((self.dimension, self.dimension), dtype = int)
        self.set_pattern(self.pattern)

    def set_pattern(self, pattern):
        self.pattern = pattern
        self.tiles = self.pattern_to_tiles(self.pattern)
        return self

    def init_border(self):
        anchors = np.array([
            [0, 0, 1], [1, 0, 1], [1, 0, 0],
            [1, 1, 0], [0, 1, 0], [0, 1, 1],
            [0, 0, 1],
        ]) * self.dimension
        border = VMobject(**self.border_config)
        border.set_anchor_points(anchors, mode = "corners")
        self.border = border

    def pattern_to_tiles(self, pattern):
        boxpos = [
            [i, j, k] for i, j, k in it.product(range(self.dimension), repeat = 3)
            if (k < pattern[i][j])
        ]
        tiles = self.backup_tiles.copy()
        directions = [OUT, RIGHT, UP]
        for pos in boxpos:
            i, j, k = pos
            tile_indices  = map(self.get_tile_index, [[i, j], [j, k], [k, i]])
            for tile_set, tile_index, direction in zip(tiles, tile_indices, directions):
                tile_set[tile_index].shift(direction)
        return tiles

    def change_height(self, height):
        self.height = height
        self.adjust_size()
        return self

    def adjust_size(self):
        self.set_height(self.height)
        self.move_to(ORIGIN)
        return self

    def shuffle_tiles(self):
        for tile_set in self.tiles:
            tile_set.submobjects = list_shuffle(tile_set.submobjects)
        return self

    def get_dimension(self):
        return self.dimension

    def get_border(self):
        return self.border

    def get_all_tiles(self):
        return self.tiles
    
    def get_directions(self):
        return ["out", "up", "right"]

    def get_positions(self):
        return list(it.product(range(self.dimension), repeat = 2))

    def get_tile_set(self, direction):
        directions = self.get_directions()
        direction_index = directions.index(direction)
        return self.get_all_tiles()[direction_index]

    def get_tile(self, direction, position):
        tile_set = self.get_tile_set(direction)
        pos_index = self.get_tile_index(position)
        return tile_set[pos_index]

    def get_tile_index(self, position):
        i, j = position
        return i * self.dimension + j

    # Color filling
    def set_tile_fill(self, direction, position, opacity):
        tile = self.get_tile(direction, position)
        tile.set_fill(opacity = opacity)
        return self

    def set_tile_set_fill(self, direction, opacity):
        tile_set = self.get_tile_set(direction)
        positions = self.get_positions()
        for position in positions:
            self.set_tile_fill(direction, position, opacity)
        return self

    def set_all_tiles_fill(self, opacity):
        directions = self.get_directions()
        positions = self.get_positions()
        for direction, position in it.product(directions, positions):
            self.set_tile_fill(direction, position, opacity)
        return self

    def fill_tile(self, direction, position):
        self.set_tile_fill(direction, position, 1)
        return self

    def unfill_tile(self, direction, position):
        self.set_tile_fill(direction, position, 0)
        return self

    def fill_tile_set(self, direction):
        self.set_tile_set_fill(direction, 1)
        return self

    def unfill_tile_set(self, direction):
        self.set_tile_set_fill(direction, 0)
        return self

    def fill_all_tiles(self):
        self.set_all_tiles_fill(1)
        return self

    def unfill_all_tiles(self):
        self.set_all_tiles_fill(0)
        return self       


class Reflines(VMobject):
    CONFIG = {
        "line_config" : {
            "stroke_color" : DARK_GRAY,
            "stroke_width" : 2,
        },
        "dimension" : 3,
    }
    def __init__(self, ct_or_hexagon, **kwargs):
        digest_config(self, kwargs, locals())
        self.mob_type = "ct" if isinstance(ct_or_hexagon, CalissonTilling) else "hexagon"
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        dim, border = self.get_dimension_and_border()
        axis = self.get_rotation_axis()
        anchors = border.get_anchors()
        num_pts = dim + 1
        lpts, rpts = pts = [
            [interpolate(sv, ev, alpha) for alpha in np.linspace(0, 1, num_pts)][1:-1]
            for sv, ev in (anchors[0:2:1], anchors[-2:-4:-1])
        ]
        lines = VGroup(*[
            DashedLine(lp, rp, **self.line_config)
            for lp, rp in zip(lpts, rpts)
        ])
        for angle in np.linspace(0, 2*np.pi, 6, endpoint = False):
            self.add(lines.copy().rotate_about_origin(angle, axis = axis))
        for k in range(3):
            self.add(DashedLine(anchors[k], anchors[k+3], **self.line_config))
        # Hacky solution to the perspective problem in 3D
        if self.mob_type == "ct":
            height = self.ct_or_hexagon.get_height()
            self.shift(height * (IN+LEFT+DOWN))

    def get_dimension_and_border(self):
        if self.mob_type == "ct":
            dim = self.ct_or_hexagon.get_dimension()
            border = self.ct_or_hexagon.get_border()
        else:
            dim = self.dimension
            border = self.ct_or_hexagon
        return dim, border

    def get_rotation_axis(self):
        return [1, 1, 1] if self.mob_type == "ct" else OUT


class Rhombus120(VMobject):
    CONFIG = {
        "side_length" : 0.8,
        "angle" : 0,
        "rhombus_config": {
            "stroke_width" : 5,
            "mark_paths_closed" : True,
        },
        "refline_config" : {
            "stroke_color" : DARK_GRAY,
            "stroke_width" : 2,
        },
    }
    def generate_points(self):
        anchors = np.array([
            UP, np.sqrt(3)*LEFT, DOWN, np.sqrt(3)*RIGHT, UP
        ]) / 2. * self.side_length
        rhombus = VMobject(**self.rhombus_config)
        rhombus.set_anchor_points(anchors, mode = "corners")
        rhombus.rotate(self.angle)
        self.add(rhombus)        
        self.rhombus = rhombus

    def get_refline(self):
        if not hasattr(self, "refline"):
            anchors = self.rhombus.get_anchors()
            self.refline = DashedLine(anchors[0], anchors[2], **self.refline_config)
        return self.refline

    def get_fill_color(self):
        return self.rhombus_config.get("fill_color", WHITE)

    def generate_counter_tex(self, counter):
        color = self.get_fill_color()
        counter_str = str(counter) if isinstance(counter, int) else counter
        counter_tex = TexMobject(counter_str)
        counter_tex.set_color(color).next_to(self.rhombus, DOWN)
        return counter_tex

    def get_counter_tex(self):
        if not hasattr(self, "counter_tex"):
            self.counter_tex = self.generate_counter_tex(0)
        return self.counter_tex

    def change_counter_tex_animation(self, counter, run_time = 1):
        old_tex = self.get_counter_tex()
        new_tex = self.generate_counter_tex(counter)
        return Transform(old_tex, new_tex, run_time = run_time)

class RRhombus(Rhombus120):
    CONFIG = {
        "angle" : -np.pi/3,
        "rhombus_config" : {"fill_color" : TILE_RED},
    }

class HRhombus(Rhombus120):
    CONFIG = {
        "angle" : 0,
        "rhombus_config" : {"fill_color" : TILE_GREEN},
    }

class LRhombus(Rhombus120):
    CONFIG = {
        "angle" : np.pi/3,
        "rhombus_config" : {"fill_color" : TILE_BLUE},
    }


#####
## Animations

class TilesGrow(Transform):
    CONFIG = {
        "submobject_mode" : "lagged_start",
        "lag_factor" : 3,
        "run_time" : 3,
    }
    def __init__(self, ct_mob, **kwargs):
        digest_config(self, kwargs, locals())
        source_tiles = mobs_shuffle(ct_mob.get_all_tiles())
        target_tiles = source_tiles.copy()
        for tile in source_tiles:
            tile.scale_in_place(0)
            tile.set_stroke(width = 0)
        Transform.__init__(self, source_tiles, target_tiles, **kwargs)


#####
## Main scenes

class CalissonTillingScene(ThreeDScene):
    def setup(self):
        self.dimensions = [5, 5, 6]
        self.patterns = [generate_pattern(dim) for dim in self.dimensions]
        self.cts = [
            CalissonTilling(dimension = dim, pattern = pattern, enable_shuffle = True)
            for dim, pattern in zip(self.dimensions, self.patterns)
        ]

    def count_through_set(self, ct_mob, direction, time_intval = 0.2):
        positions = list_shuffle(ct_mob.get_positions())
        for position in positions:
            ct_mob.fill_tile(direction, position)
            self.wait(time_intval)
        self.wait()
        self.play_unfill_set(ct_mob, direction)
        self.wait()

    def count_through_multiple_sets(self, ct_mob, directions, time_intval = 0.2):
        for direction in directions:
            self.count_through_set(ct_mob, direction, time_intval)

    def fill_and_unfill_multiple_sets(self, ct_mob, directions, **kwargs):
        for direction in directions:
            self.play_fill_set(ct_mob, direction, **kwargs)
            self.wait()
            self.play_unfill_set(ct_mob, direction, **kwargs)
            self.wait()

    def play_fill_set(self, ct_mob, direction, **kwargs):
        ct_mob.generate_target(use_deepcopy = True)
        ct_mob.target.fill_tile_set(direction)
        self.play(MoveToTarget(ct_mob, **kwargs))

    def play_unfill_set(self, ct_mob, direction, **kwargs):
        ct_mob.generate_target(use_deepcopy = True)
        ct_mob.target.unfill_tile_set(direction)
        self.play(MoveToTarget(ct_mob, **kwargs))

    def play_fill_all_tiles(self, ct_mob, **kwargs):
        ct_mob.generate_target(use_deepcopy = True)
        ct_mob.target.fill_all_tiles()
        self.play(MoveToTarget(ct_mob, **kwargs))

    def adjust_text_to_camera(self, tex):
        tex.rotate(np.pi/4, axis = RIGHT)
        tex.rotate(3*np.pi/4, axis = OUT)
        

class TillingProblem3DPart(CalissonTillingScene):
    CONFIG = {
        "random_seed" : hash("Solara570") % 2**32,
    }
    def construct(self):
        self.show_border_and_reflines()
        self.show_initial_tilling()
        self.count_tiles()
        self.change_tillings()
        self.sync_with_2d_part()

    def show_border_and_reflines(self):
        self.set_camera_orientation(*DIAG_POS)
        self.wait()

        init_ct = self.cts[0]
        border = init_ct.get_border()
        reflines = Reflines(init_ct)
        self.play(ShowCreation(border))
        self.wait(3)
        self.play(
            Write(reflines, rate_func = smooth, submobject_mode = "lagged_start"),
            Animation(border),
            run_time = 3,
        )
        self.play(
            Indicate(VGroup(border, reflines), scale_factor = 1.05),
            run_time = 2,
        )
        self.wait()

        self.border = border
        self.reflines = reflines

    def show_initial_tilling(self):
        init_ct = self.cts[0]
        self.wait(7)
        self.play(TilesGrow(init_ct), run_time = 5)
        self.wait(7) # Show claim sync

    def count_tiles(self):
        init_ct = self.cts[0]
        directions = init_ct.get_directions()
        self.count_through_multiple_sets(init_ct, directions)

    def change_tillings(self):
        # Pattern doesn't matter
        old_ct = self.cts[0]
        pattern_ct = self.cts[1]
        self.play(
            Transform(old_ct, pattern_ct, path_arc = np.pi, path_arc_axis = np.array([1., 1., 1.])),
            run_time = 3,
        )
        self.wait()
        self.fill_and_unfill_multiple_sets(old_ct, old_ct.get_directions())

        # Size doesn't matter either
        old_reflines = self.reflines
        size_ct = self.cts[2]
        old_tiles = old_ct.get_all_tiles()
        new_reflines = Reflines(size_ct)
        self.play(FadeOut(old_tiles), run_time = 1)
        self.play(FadeOut(old_reflines), Animation(self.border), run_time = 1)
        self.play(FadeIn(new_reflines), Animation(self.border), run_time = 1)
        self.wait()
        self.play(TilesGrow(size_ct))
        self.wait()
        self.fill_and_unfill_multiple_sets(size_ct, size_ct.get_directions())

    def sync_with_2d_part(self):
        self.wait(15)


class TillingProblem2DPart(CalissonTillingScene):
    def construct(self):
        self.initialize_texts()
        self.show_rhombi()
        self.show_reflines()
        self.show_counter()
        self.show_claim()
        self.count_tiles()
        self.ask_about_how_to_prove()
        self.countdown()

    def initialize_texts(self):
        geom_text = TextMobject("一个边长为$n$的正六边形", "和一些边长为1的菱形")
        eqtri_text = TextMobject("它们都是由若干正三角形组成的")
        three_types_text = TextMobject("因为朝向不同，菱形被分成三种")
        try_tilling_text = TextMobject("现在用这些菱形", "镶嵌", "正六边形...")
        remark = TextMobject("（无间隙且不重叠地覆盖）")
        claim_text = TextMobject("最终的图案中", "每种菱形的数量一定都是$n^2$")
        twist_text = TextMobject("改变菱形的摆放方式", "或者改变正六边形的大小", "这个结论依然成立")
        how_to_prove_text = TextMobject("如何证明？", "")
        
        for text in (geom_text, claim_text, twist_text):
            text.arrange_submobjects(DOWN, aligned_edge = LEFT)
        try_tilling_text[1].set_color(GREEN)
        remark.scale(0.5)
        remark.set_color(GREEN)
        remark.next_to(try_tilling_text[1], DOWN, buff = 0.1)
        how_to_prove_text.set_color(YELLOW)

        bg_texts = VGroup(
            geom_text,
            eqtri_text,
            three_types_text,
            VGroup(try_tilling_text, remark),
        )
        q_texts = VGroup(
            claim_text,
            twist_text,
        )
        for texts in (bg_texts, q_texts, how_to_prove_text):
            texts.arrange_submobjects(DOWN, aligned_edge = LEFT, buff = 1)
            texts.to_corner(LEFT+UP)

        self.bg_texts = bg_texts
        self.q_texts = q_texts
        self.how_to_prove_text = how_to_prove_text

    def show_rhombi(self):
        self.wait()
        rhombi = VGroup(*[RhombusType() for RhombusType in (RRhombus, HRhombus, LRhombus)])
        rhombi.arrange_submobjects(RIGHT, aligned_edge = DOWN, buff = 1)
        rhombi.to_edge(RIGHT, buff = 1)
        rhombi.to_edge(UP)
        hexagon_text, rhombi_text = self.bg_texts[0]
        self.play(
            Write(hexagon_text),
            run_time = 1
        )
        self.wait()
        self.play(
            ShowCreation(rhombi, submobject_mode = "all_at_once"),
            Write(rhombi_text),
            run_time = 1
        )
        self.wait()
        self.rhombi = rhombi

    def show_reflines(self):
        reflines = VGroup(*[rhombus.get_refline() for rhombus in self.rhombi])
        eqtri_text = self.bg_texts[1]
        self.play(
            Write(eqtri_text),
            ShowCreation(reflines),
            Animation(self.rhombi),
            run_time = 3
        )
        self.play(
            Indicate(VGroup(self.rhombi, reflines), scale_factor = 1.05),
            run_time = 2,
        )
        self.wait()
        self.reflines = reflines

    def show_counter(self):
        counters = VGroup(*[rhombus.get_counter_tex() for rhombus in self.rhombi])
        three_types_text = self.bg_texts[2]
        try_tilling_text, remark = self.bg_texts[3]
        self.play(
            FadeOut(self.reflines),
            ApplyMethod(self.rhombi.set_fill, None, 1),
            Write(three_types_text),
            run_time = 2
        )
        self.wait()
        self.play(Write(try_tilling_text), run_time = 2)
        self.play(Write(remark), run_time = 1)
        self.wait()
        self.play(FadeIn(counters), run_time = 1)
        self.wait(4)
        self.counters = counters

    def show_claim(self):
        self.play(FadeOut(self.bg_texts), run_time = 2)
        self.wait()
        claim_text = self.q_texts[0]
        self.play(Write(claim_text), run_time = 2)
        self.wait(2)

    def count_tiles(self):
        pattern_text, size_text, still_hold_text = self.q_texts[1]
        init_ct = self.cts[0]
        num = init_ct.get_dimension() ** 2
        time_intval = 0.2
        for k in (1, 0, 2):
            rhombus = self.rhombi[k]
            for m in range(num):
                self.play(rhombus.change_counter_tex_animation(m+1), run_time = time_intval)
            self.wait(3)
        self.play(Write(pattern_text), run_time = 1)
        self.reset_counters()
        self.wait()

        pattern_ct = self.cts[1]
        num = pattern_ct.get_dimension() ** 2
        for k in (1, 0, 2):
            rhombus = self.rhombi[k]
            self.play(rhombus.change_counter_tex_animation(num), run_time = 1)
            self.wait(3)
        self.reset_counters()
        self.play(Write(size_text), run_time = 1)
        self.wait()

        self.wait(4)
        size_ct = self.cts[2]
        num = size_ct.get_dimension() ** 2
        for k in (1, 0, 2):
            rhombus = self.rhombi[k]
            self.play(rhombus.change_counter_tex_animation(num), run_time = 1)
            self.wait(3)

        self.wait()
        self.play(Write(still_hold_text), run_time = 1)
        self.wait()

    def ask_about_how_to_prove(self):
        claim_text = self.q_texts[0]
        self.wait()
        self.play(self.q_texts.shift, DOWN, run_time = 1)
        claim_rect = SurroundingRectangle(
            VGroup(claim_text, self.how_to_prove_text),
            stroke_color = YELLOW, buff = 0.3
        )
        self.play(
            Write(self.how_to_prove_text),
            ShowCreation(claim_rect),
            run_time = 1,
        )
        self.wait()
        self.claim_rect = claim_rect

    def countdown(self):
        countdown_texts = VGroup(*[TextMobject(str(k)) for k in range(5, -1, -1)])
        countdown_texts.set_color(YELLOW)
        for text in countdown_texts:
            text.scale(1.5)
            text.next_to(self.how_to_prove_text[0], RIGHT, buff = 3)
            self.add(text)
            self.wait()
            self.remove(text)

    def reset_counters(self):
        reset_anims = [
            rhombus.change_counter_tex_animation(0)
            for rhombus in self.rhombi
        ]
        self.play(AnimationGroup(*reset_anims), run_time = 1)
        self.wait()


class DifferentViews(CalissonTillingScene):
    def construct(self):
        self.add_tilling()
        self.set_constants()
        self.show_brace_and_text()
        self.change_perspective()
        self.sync_with_2d_part()

    def add_tilling(self):
        self.set_camera_orientation(*DIAG_POS)
        self.tilling = CalissonTilling(
            dimension = 5, pattern = AMM_PATTERN, enable_fill = True
        )
        self.add(self.tilling)
        self.wait(3)

    def set_constants(self):
        self.braces, self.texts = None
        self.camera_pos = None
        raise Exception("Fill in the blanks!")

    def show_brace_and_text(self):
        brace1, brace2 = self.braces
        self.play(GrowFromCenter(brace1), GrowFromCenter(brace2), run_time = 2)
        self.wait()
        self.play(FadeIn(self.texts), run_time = 1)
        self.wait()
        
    def change_perspective(self):
        self.move_camera(*self.camera_pos, run_time = 6)
        self.wait()

    def sync_with_2d_part(self):
        self.wait(20)

    # Not elegant at all, but it just works!
    def get_basic_brace(self, direction):
        anchors = self.tilling.get_border().get_anchors()
        lines = [Line(anchors[k], anchors[k+1]) for k in range(len(anchors)-1)]
        if direction == "upleft":
            brace = Brace(lines[0], direction = DOWN)
        elif direction == "left":
            brace = self.get_basic_brace("upleft")
            brace.rotate(2*np.pi/3, axis = [1, -1, 1], about_point = anchors[1])
        elif direction == "downleft":
            brace = Brace(lines[2], direction = RIGHT)
        elif direction == "downright":
            brace = Brace(lines[3], direction = UP)
            brace.rotate(-np.pi/2., axis = RIGHT, about_point = anchors[3])
        else:
            raise Exception("There's no such direction")
        return brace

    def get_brace_and_text(self, perspective, direction):
        anchors = self.tilling.get_border().get_anchors()
        text = TexMobject("n")
        self.add_fixed_orientation_mobjects(text)
        if perspective == "Up":
            if direction == "x":
                brace = self.get_basic_brace("downleft")
            else:
                brace = self.get_basic_brace("upleft")
        elif perspective == "Front":
            if direction == "x":
                brace = self.get_basic_brace("downleft")
                brace.rotate(np.pi/2., axis = UP, about_point = anchors[3])
            else:
                brace = self.get_basic_brace("left")
                brace.rotate(-np.pi/2., axis = OUT, about_point = anchors[1])
        elif perspective == "Right":
            if direction == "x":
                brace = self.get_basic_brace("downright")
            else:
                brace = self.get_basic_brace("left")
        else:
            raise Exception("There's no such perspective")
        brace.put_at_tip(text)
        return brace, text

    def get_braces_and_texts(self, perspective):
        x_part = self.get_brace_and_text(perspective, "x")
        y_part = self.get_brace_and_text(perspective, "y")
        braces, texts = zip(x_part, y_part)
        return VGroup(braces), VGroup(texts)


class ViewFromUpSide(DifferentViews):
    def set_constants(self):
        self.braces, self.texts = self.get_braces_and_texts("Up")
        self.camera_pos = UP_POS


class ViewFromFrontSide(DifferentViews):
    def set_constants(self):
        self.braces, self.texts = self.get_braces_and_texts("Front")
        self.camera_pos = FRONT_POS


class ViewFromRightSide(DifferentViews):
    def set_constants(self):
        self.braces, self.texts = self.get_braces_and_texts("Right")
        self.camera_pos = RIGHT_POS


class TillingSolution3DPart(CalissonTillingScene):
    def construct(self):
        self.setup_tilling()
        self.fill_tilling()
        self.shake_camera_around()

    def setup_tilling(self):
        self.wait(2)
        self.set_camera_orientation(*DIAG_POS)
        self.tilling = CalissonTilling(dimension = 5, pattern = AMM_PATTERN)
        self.play(Write(self.tilling, rate_func = smooth), run_time = 3)
        self.wait(2)
        
    def fill_tilling(self):
        self.play_fill_all_tiles(self.tilling, run_time = 2)
        self.wait(2)

    def shake_camera_around(self):
        nudge = TAU / 30
        nangles = 60
        phi, theta, distance = DIAG_POS
        self.move_camera(phi - nudge, theta, run_time = 1)
        self.wait()
        for angle in np.linspace(TAU/4, 5*TAU/4, nangles-1):
            camera_angle = [phi - np.sin(angle)*nudge, theta + np.cos(angle)*nudge]
            self.move_camera(*camera_angle, run_time = 3./nangles)
        self.move_camera(*DIAG_POS, run_time = 1)
        self.wait()


class TillingSolution2DPart(CalissonTillingScene):
    def construct(self):
        self.initialize_texts()
        self.proof_part_1()
        self.proof_part_2()

    def initialize_texts(self):
        proof_texts = TextMobject("“证明”：", "涂颜色", "+", "换视角")
        proof_texts.arrange_submobjects(RIGHT)
        proof_texts.to_corner(LEFT+UP)

        imagine_3d_text = TextMobject("（想象这是一个三维图案...）")
        imagine_3d_text.to_corner(RIGHT+UP)
        imagine_3d_text.set_color(YELLOW)

        rhombi = VGroup(*[
            RhombusType(rhombus_config = {"fill_opacity" : 1})
            for RhombusType in (RRhombus, HRhombus, LRhombus)
        ])
        time_texts = VGroup(*[
            TexMobject("\\times", "n^2").scale(1.2).set_color(rhombus.get_fill_color())
            for rhombus in rhombi
        ])
        rhombi_and_texts = VGroup(*[
            VGroup(rhombus, time_text).arrange_submobjects(RIGHT)
            for rhombus, time_text in zip(rhombi, time_texts)
        ])
        rhombi_and_texts.arrange_submobjects(RIGHT, buff = 2)
        rhombi_and_texts.to_edge(UP, buff = 1.4)

        equation = TexMobject(*["n^2" if k % 2 == 0 else "=" for k in range(5)])
        for text, color in zip(equation[::2], RHOMBI_COLOR_SET):
            text.set_color(color)
        qed = FakeQEDSymbol(jagged_percentage = 0.1)
        qed.set_height(equation.get_height())
        conclusions = VGroup(equation, qed)
        conclusions.arrange_submobjects(RIGHT, buff = 1)
        conclusions.to_corner(RIGHT+UP)

        self.proof_texts = proof_texts
        self.imagine_3d_text = imagine_3d_text
        self.rhombi = rhombi
        self.time_texts = time_texts
        self.rhombi_and_texts = rhombi_and_texts
        self.conclusions = conclusions

    def proof_part_1(self):
        proof_text, paint_text, and_text, perspective_text = self.proof_texts
        self.play(Write(proof_text), run_time = 1)
        self.wait(4)
        self.play(Write(paint_text), run_time = 1)
        self.wait(3)
        self.play(FadeIn(self.imagine_3d_text), run_time = 1)
        self.wait(6)
        self.play(FadeOut(self.imagine_3d_text), run_time = 1)
        self.wait()
        self.play(Write(VGroup(and_text, perspective_text), run_time = 1))
        self.wait()

    def proof_part_2(self):
        self.play(
            DrawBorderThenFill(self.rhombi, submobject_mode = "all_at_once"),
            run_time = 2
        )
        self.wait()
        self.wait(12)
        self.play(Write(self.time_texts), run_time = 3)
        self.wait()
        source_texts = VGroup(*[texts[1] for texts in self.time_texts])
        target_texts = VGroup(*self.conclusions[0][::2])
        equal_signs = VGroup(self.conclusions[0][1::2])
        qed = self.conclusions[1]
        self.play(
            Transform(source_texts.copy(), target_texts),
            Write(equal_signs),
            run_time = 2
        )
        self.wait()
        self.play(FadeIn(qed))
        self.wait(2)


class EndScene(Scene):
    def construct(self):
        theorem_sc = TextMobject("可利颂镶嵌定理")
        theorem_sc.scale(1.8)
        theorem_eng = TextMobject("(Calisson Tilling Theorem)")
        theorem_eng.set_width(theorem_sc.get_width())
        theorem = VGroup(theorem_sc, theorem_eng)
        theorem.arrange_submobjects(DOWN)
        theorem.to_edge(RIGHT).shift(UP)
        self.play(FadeIn(theorem), run_time = 1)
        author = TextMobject("@Solara570")
        author.scale(1.5)
        support = TextMobject("(Powered by @3Blue1Brown)")
        support.set_width(author.get_width())
        names = VGroup(author, support)
        names.arrange_submobjects(DOWN)
        names.to_corner(RIGHT+DOWN)
        self.play(FadeIn(names), run_time = 1)
        self.wait(3)


#####
## Thumbnail

class Thumbnail3DPart(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(*DIAG_POS)
        # The cover of "Mathematical Puzzles: A Connoisseur's Collection"
        tilling = CalissonTilling(
            dimension = 7,
            pattern = MPACC_PATTERN,
            enable_fill = True
        )
        self.add(tilling)


class Thumbnail2DPart(Scene):
    def construct(self):
        pairs = VGroup(*[
            VGroup(RhombusType(), TexMobject("\\times 49"))
            for RhombusType in (RRhombus, HRhombus, LRhombus)
        ])
        for pair in pairs:
            rhombus, text = pair
            rhombus.set_stroke(width = 0)
            rhombus.set_fill(opacity = 1)
            text.set_color(rhombus.get_fill_color())
            text.set_stroke(width = 3)
            text.scale(1.5)
            pair.arrange_submobjects(RIGHT)
        pairs.arrange_submobjects(DOWN, aligned_edge = RIGHT, buff = 0.5)
        pairs.set_height(6)
        self.add(pairs)








