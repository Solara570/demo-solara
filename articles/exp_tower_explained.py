##########################################################
#  A brief introduction on power tower, i.e. tetration.  #
#         https://zhuanlan.zhihu.com/p/179279978         #
##########################################################

# It's a mixture of ancient (relatively) and modern (also relatively) code.
# So be prepared to see some quirks and boo-boos.

from big_ol_pile_of_manim_imports import *
from sympy import solve, symbols
from custom.custom_mobjects import *

#####
## Helpers

BASE_LOWER_BOUND = pow(np.exp(-1), np.e)
BASE_UPPER_BOUND = pow(np.e, np.exp(-1))

def is_close(a, b, thres = 1E-8):
    return abs(a - b) < thres

def get_exp_iter_type(base):
    assert (base > 0)
    if base < BASE_LOWER_BOUND:
        return "osc"
    elif base <= BASE_UPPER_BOUND:
        return "conv"
    else:
        return "boom"

def get_exp_iter_info(base, init_val = None, n_times = 10000, conv_thres = 1E-6, osc_thres = 1E-5):
    exp_func = lambda x: pow(base, x)
    iter_type = get_exp_iter_type(base)
    if init_val is None:
        init_val = base
    iter_seq = [init_val, exp_func(init_val)]
    critical_vals = []
    for k in range(n_times):
        new_val = exp_func(iter_seq[-1])
        iter_seq.append(new_val)
        if new_val > 10.:
        # Divergence: explode to infinity
            break
        elif is_close(new_val, iter_seq[-2], conv_thres):
        # Convergence: close to the last value in the sequence
            critical_vals = [iter_seq[-1]]
            break
        elif is_close(new_val, iter_seq[-3], conv_thres):
        # Undetermined: either it's hard to reach convergence criteria, or a real oscillating behavior
            if iter_type == "osc":
                critical_vals = iter_seq[-2:]
                break
            elif iter_type == "conv" and is_close(iter_seq[-2], iter_seq[-1], osc_thres):
                critical_vals = [np.average(iter_seq[-2:])]
                break
    if iter_type != "boom" and critical_vals == []:
    # The worst case...
        critical_vals = iter_seq[-2:] if iter_type == "osc" else [np.average(iter_seq[-2:])]
    return iter_seq, iter_type, critical_vals

def get_point_seq_from_iter_seq(iter_seq):
    point_seq = [(iter_seq[0], 0)]
    for k in range(len(iter_seq) - 1):
        point_seq.append((iter_seq[k], iter_seq[k+1]))
        point_seq.append((iter_seq[k+1], iter_seq[k+1]))
    return point_seq

def get_diag_angle(mob):
    corner_ur = mob.get_critical_point(UR)
    corner_dl = mob.get_critical_point(DL)
    return angle_of_vector(corner_ur - corner_dl)

def exp_solver(base):
    x = symbols('x')
    return solve(base**x - x, x)


#####
## Mobjects

class IterCriticalPoints(VMobject):
    CONFIG = {
        "dot_radius" : 0.08,
    }
    def __init__(self, axes, iter_type, critical_vals, **kwargs):
        self.axes = axes
        self.iter_type = iter_type
        self.critical_vals = critical_vals
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        color = GREEN if self.iter_type == "conv" else RED
        coords = []
        if len(self.critical_vals) == 2:
            a, b = self.critical_vals
            coords = [(a, b), (b, a)]
        elif len(self.critical_vals) == 1:
            a = self.critical_vals[0]
            coords = [(a, a)]
        for coord in coords:
            pos = self.axes.coords_to_point(*coord)
            self.add(Dot(pos, radius = self.dot_radius, color = color))


class IterSpiderWeb(VMobject):
    CONFIG = {
        "line_width" : 1,
        "line_color" : YELLOW,
    }
    def __init__(self, axes, iter_seq, **kwargs):
        self.iter_seq = iter_seq
        self.axes = axes
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        point_seq = get_point_seq_from_iter_seq(self.iter_seq)
        for k in range(len(point_seq) - 1):
            start_pt = self.axes.coords_to_point(*point_seq[k])
            end_pt = self.axes.coords_to_point(*point_seq[k+1])
            self.add(
                Line(
                    start_pt, end_pt,
                    stroke_width = self.line_width, stroke_color = self.line_color
                )
            )

class ModIterSpiderWeb(VMobject):
    CONFIG = {
        "line_width" : 1,
        "line_color" : YELLOW,
    }
    def __init__(self, axes, iter_seq, **kwargs):
        self.iter_seq = iter_seq
        self.axes = axes
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        iter_seq = self.iter_seq
        point_seq = [(iter_seq[0], 0)]
        for k in range(1, len(iter_seq)-1, 2): 
            point_seq.append((iter_seq[k-1], iter_seq[k]))
            point_seq.append((iter_seq[k+1], iter_seq[k]))
        for k in range(len(point_seq) - 1):
            start_pt = self.axes.coords_to_point(*point_seq[k])
            end_pt = self.axes.coords_to_point(*point_seq[k+1])
            self.add(
                Line(
                    start_pt, end_pt,
                    stroke_width = self.line_width, stroke_color = self.line_color
                )
            )


class ExpDots(VMobject):
    CONFIG = {
        "angle" : TAU / 8,
        "gap_buff" : 0.1,
    }
    def generate_points(self):
        dots = TexMobject("...")
        dots.arrange_submobjects(RIGHT, buff = self.gap_buff)
        dots.rotate(self.angle)
        self.add(dots)


class ExpTower(VMobject):
    CONFIG = {
        "order"   : 5,
        "element" : "x",
        "tower_texts" : None,
        "scale_factor" : 0.8,
        "height_stretch_factor" : 0.8,
        "is_infinite"  : True,
        "is_fraction_form" : False,
        "expdots_buff" : 0.05,
    }
    def __init__(self, **kwargs):
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        tower_texts = self.get_tower_texts()
        tower_tex = self.generate_tower_tex_from_texts(tower_texts)
        self.tower_tex = tower_tex
        self.add(tower_tex.center())

    def get_tower_texts(self):
        if self.tower_texts is None:
            self.tower_texts = [self.element] * self.order
        assert len(self.tower_texts) == self.order
        return self.tower_texts

    def generate_tower_tex_from_texts(self, texts):
        tower_tex = VGroup(*[
            TexMobject(text)
            for text in self.tower_texts
        ])
        for k, part in enumerate(tower_tex):
            part.scale(self.scale_factor**k)
            if k > 0:
                buff = 0.05 / np.sqrt(k)
                part.stretch(self.height_stretch_factor, 1)
                if self.is_fraction_form:
                    last_term = tower_tex[k-1]
                    ref_point = (last_term.get_critical_point(UR) + last_term.get_critical_point(RIGHT))/2.
                    part.next_to(ref_point, RIGHT+UP, buff = buff)
                else:
                    part.next_to(tower_tex[k-1], RIGHT+UP, buff = buff)
        if self.is_infinite:
            exp_dots = ExpDots(angle = get_diag_angle(tower_tex))
            if self.is_fraction_form:
                exp_dots.next_to(tower_tex[-1][1], RIGHT+UP, buff = self.expdots_buff)
            else:
                exp_dots.next_to(tower_tex[-1], RIGHT+UP, buff = self.expdots_buff)
            exp_dots.next_to(tower_tex[-1], RIGHT+UP, buff = self.expdots_buff)
            tower_tex.add(exp_dots)
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
        text.set_color(self.text_color)
        text.scale_to_fit_height(rect.get_height() * self.text_height_factor)
        text.move_to(rect)
        self.group = VGroup(rect, text)
        self.add(self.group)

    def get_text_mob(self):
        return self.group[1]


#####
## Scenes

class HowToCalculateInfiniteSum(Scene):
    def construct(self):
        infinite_sum = TexMobject("1", "+\\dfrac{1}{2}", "+\\dfrac{1}{4}", "+\\dfrac{1}{8}", "+\\dfrac{1}{16}", "+\\cdots", "=2")
        partial_sum = TexMobject("1", ",", "\\,\\dfrac{3}{2}", ",", "\\,\\dfrac{7}{4}",  ",", "\\,\\dfrac{15}{8}", ",", "\\,\\dfrac{31}{16}", ",", "\\,\\cdots", "\\to 2")
        infinite_sum.scale(1.5).shift(1.5*UP)
        infinite_sum[-1].set_color(YELLOW)
        partial_sum.scale(1.5).shift(1.5*DOWN)
        partial_sum[-1].set_color(YELLOW)
        self.add(infinite_sum[:-1])
        for k in range(len(infinite_sum[:-1])):
            if k == 0:
                rect = SurroundingRectangle(infinite_sum[0])
                self.play(ShowCreation(rect))
                self.wait(0.5)
                self.play(ReplacementTransform(infinite_sum[0].deepcopy(), partial_sum[0]))
                self.wait()
            else:
                new_rect = SurroundingRectangle(infinite_sum[:k+1])
                self.play(Transform(rect, new_rect))
                self.wait(0.5)
                self.play(
                    ReplacementTransform(infinite_sum[:k+1].deepcopy(), partial_sum[2*k]),
                    Write(partial_sum[2*k-1])
                )
                self.wait()
        self.play(FadeOut(rect))
        self.wait(0.5)
        self.play(Write(partial_sum[-1]))
        self.wait()
        self.play(ReplacementTransform(partial_sum[-1].deepcopy(), infinite_sum[-1]))
        self.wait(3)
        self.play(FadeOut(VGroup(partial_sum, infinite_sum[-1])))
        self.wait()


class HowToCalculateNestedFraction(Scene):
    CONFIG = {
        "cover_rect_config" : {
            "stroke_width" : 0,
            "fill_color" : GREY,
            "fill_opacity" : 1,
            "buff" : 0.02,
        }
    }
    def construct(self):
        infinite_frac = TexMobject("1+\\cfrac{1}{1 + \\cfrac{1}{1 + \\cfrac{1}{1 + \\cfrac{1}{1 + \\cdots}}}}", "=\\dfrac{\\sqrt{5}+1}{2}")
        partial_results = TexMobject("1", ",", "\\,2", ",", "\\,\\dfrac{3}{2}",  ",", "\\,\\dfrac{5}{3}", ",", "\\,\\dfrac{8}{5}", ",", "\\,\\cdots", "\\to \\dfrac{\\sqrt{5}+1}{2}")
        infinite_frac.scale(1.2).shift(UP)
        infinite_frac[-1].set_color(YELLOW)
        partial_results.scale(1.2).shift(2.5*DOWN)
        partial_results[-1].set_color(YELLOW)
        self.add(infinite_frac[0])
        for k in range(6):
            if k == 0:
                sur_rect = SurroundingRectangle(infinite_frac[0])
                cover_rect = SurroundingRectangle(infinite_frac[0][1:], **self.cover_rect_config)
                self.play(ShowCreation(sur_rect), DrawBorderThenFill(cover_rect))
                self.wait(0.5)
                self.play(ReplacementTransform(infinite_frac[0][0].deepcopy(), partial_results[0]))
                self.wait()
            elif k < 5:
                new_cover_rect = SurroundingRectangle(infinite_frac[0][4*k+1:], **self.cover_rect_config)
                self.play(Transform(cover_rect, new_cover_rect))
                self.wait(0.5)
                self.play(
                    ReplacementTransform(infinite_frac[0][:4*k+1].deepcopy(), partial_results[2*k]),
                    Write(partial_results[2*k-1])
                )
                self.wait()
            else:
                self.play(FadeOut(cover_rect))
                self.wait(0.5)
                self.play(
                    ReplacementTransform(infinite_frac[0].deepcopy(), partial_results[2*k]),
                    Write(partial_results[2*k-1])
                )
                self.wait()
        self.play(FadeOut(sur_rect))
        self.wait(0.5)
        self.play(Write(partial_results[-1]))
        self.wait()
        self.play(ReplacementTransform(partial_results[-1].deepcopy(), infinite_frac[-1]))
        self.wait(3)
        self.play(FadeOut(VGroup(partial_results, infinite_frac[-1])))
        self.wait()


class HowToCalculatePowerTower(HowToCalculateNestedFraction):
    def construct(self):
        power_tower = ExpTower(
            order = 4, tower_texts = ["\\left(\\dfrac{1}{" + str(k) + "}\\right)" for k in range(2, 6)],
            is_fraction_form = True,
        )
        div_text = TextMobject("不存在", color = YELLOW)
        div_text.move_to(power_tower[0]).next_to(power_tower, RIGHT, buff = 1, coor_mask = [1,0,0])
        partial_results = TexMobject("0.5", ",", "\\,0.7937\\dots", ",", "\\,0.5905\\dots",  ",", "\\,0.7397\\dots", ",", "\\,\\cdots", "\\text{（极限不存在）}")
        power_tower.scale(1.5).shift(1.2*UP)
        partial_results.shift(2.2*DOWN)
        partial_results[-1].set_color(YELLOW)
        self.add(power_tower)
        for k in range(5):
            if k == 0:
                sur_rect = SurroundingRectangle(power_tower[0][0])
                cover_rect = SurroundingRectangle(power_tower[0][1:], **self.cover_rect_config)
                self.play(DrawBorderThenFill(cover_rect), ShowCreation(sur_rect))
                self.wait(0.5)
                self.play(ReplacementTransform(power_tower[0][0].deepcopy(), partial_results[0]))
                self.wait()
            elif k < 4:
                new_sur_rect = SurroundingRectangle(power_tower[0][:k+1])
                new_cover_rect = SurroundingRectangle(power_tower[0][k+1:], **self.cover_rect_config)
                self.play(Transform(cover_rect, new_cover_rect), Transform(sur_rect, new_sur_rect))
                self.wait(0.5)
                self.play(
                    ReplacementTransform(power_tower[0][:k+1].deepcopy(), partial_results[2*k]),
                    Write(partial_results[2*k-1])
                )
                self.wait()
            else:
                new_sur_rect = SurroundingRectangle(power_tower[0])
                self.play(Transform(sur_rect, new_sur_rect), FadeOut(cover_rect))
                self.wait(0.5)
                self.play(
                    ReplacementTransform(power_tower[0].deepcopy(), partial_results[2*k]),
                    Write(partial_results[2*k-1])
                )
                self.wait()
        self.play(FadeOut(sur_rect))
        self.wait(0.5)
        self.play(Write(partial_results[-1]))
        self.wait()
        div_text_source = partial_results[-1][3:-1].deepcopy()
        self.play(ApplyMethod(div_text_source.move_to, div_text))
        self.wait(3)
        self.play(FadeOut(VGroup(partial_results, div_text_source)))
        self.wait()


class IntroToCobwebPlot(Scene):
    CONFIG = {
        "iter_line_config" : {
            "color" : YELLOW,
            "stroke_width" : 1,
        }
    }
    def construct(self):
        # Basic setup
        def f(x):
            return 1.6**x
        axes = Axes(x_min = -3, x_max = 3.5, y_min = -1.5, y_max = 5.5)
        axes.move_to(1.5*RIGHT)
        diag_graph = axes.get_graph(lambda x: x, color = MAROON_B, stroke_width = 3)
        diag_remark = TexMobject("y=x", color = MAROON_B)
        diag_remark.next_to(diag_graph.get_critical_point(UR), RIGHT)
        func_graph = axes.get_graph(f, color = BLUE)
        func_remark = TexMobject("y=f(x)", color = BLUE)
        func_remark.next_to(func_graph.get_critical_point(UR), RIGHT)
        self.add(axes, diag_graph, func_graph, diag_remark, func_remark)
        # Cobweb setup
        x_0 = -2
        dot = Dot(axes.coords_to_point(x_0, 0), color = YELLOW)
        dot.save_state()
        lines = VGroup()
        self.add(dot)
        # Text remark setup
        title = TextMobject("蛛网图的绘制", color = YELLOW).to_corner(UL)
        step_1 = TextMobject("(1) 在$x$轴上找到迭代起点，开始绘制")
        step_2 = TextMobject("(2) 用", "$y=f(x)$", "更新纵坐标，完成一次函数迭代")
        step_2[1].set_color(BLUE)
        step_3 = TextMobject("(3) 用", "$y=x$", "更新横坐标，为下一次迭代做准备")
        step_3[1].set_color(MAROON_B)
        step_4 = TextMobject("(4) 重复(2)(3)两步，直到结果满足需求")
        steps = VGroup(step_1, step_2, step_3, step_4)
        steps.scale(0.7).arrange_submobjects(DOWN, aligned_edge = LEFT)
        steps.next_to(title, DOWN, aligned_edge = LEFT)
        coord_ph = TexMobject("\\Big(\\quad\\quad\\quad,\\quad\\quad\\quad\\Big)")
        coord_ph.scale(1.2).to_corner(DL).shift(UP)
        ph_x_pos = (coord_ph[0].get_center() + coord_ph[1].get_top())/2.
        ph_y_pos = (coord_ph[2].get_center() + coord_ph[1].get_top())/2.
        max_height = coord_ph[0].get_height()*0.5
        max_width = coord_ph.get_width()*0.4
        coord_remark = TextMobject("黄点坐标：", color = YELLOW)
        coord_remark.next_to(coord_ph, UP, aligned_edge = LEFT)
        self.add(title, steps, coord_remark, coord_ph)
        # Start iterating
        def get_scaling_factor(mob):
            return min(max_width/mob.get_width(), max_height/mob.get_height())
        def set_x_coord(mob):
            mob.scale(get_scaling_factor(mob))
            mob.move_to(ph_x_pos)
        def set_y_coord(mob):
            mob.scale(get_scaling_factor(mob))
            mob.move_to(ph_y_pos)
        x_coord_text = "x_0"
        y_coord_text = "0"
        x_coord_mob = TexMobject(x_coord_text, color = YELLOW)
        y_coord_mob = TexMobject(y_coord_text, color = YELLOW)
        set_x_coord(x_coord_mob)
        set_y_coord(y_coord_mob)
        x_coord_mob_copy = x_coord_mob.deepcopy()
        y_coord_mob_copy = y_coord_mob.deepcopy()
        self.add(x_coord_mob, y_coord_mob)
        n_iters = 5
        for k in range(n_iters):
            # Update vertical line
            curr_x, curr_y = axes.point_to_coords(dot.get_center())
            new_y = f(curr_x)
            vert_line = Line(axes.coords_to_point(curr_x, curr_y), axes.coords_to_point(curr_x, new_y), **self.iter_line_config)
            y_coord_text = "f(" + x_coord_text + ")"
            new_y_coord_mob = TexMobject(y_coord_text, color = YELLOW)
            set_y_coord(new_y_coord_mob)
            if k == 0:
                self.play(
                    ApplyMethod(dot.move_to, axes.coords_to_point(curr_x, new_y)),
                    ShowCreation(vert_line),
                    ReplacementTransform(y_coord_mob, new_y_coord_mob),
                )
            else:
                y_coord_mob.generate_target()
                y_coord_mob.target.set_height(new_y_coord_mob[2:-1].get_height())
                y_coord_mob.target.move_to(new_y_coord_mob[2:-1])
                self.play(
                    ApplyMethod(dot.move_to, axes.coords_to_point(curr_x, new_y)),
                    ShowCreation(vert_line),
                    MoveToTarget(y_coord_mob),
                    FadeInFromLarge(VGroup(new_y_coord_mob[:2], new_y_coord_mob[-1])),
                )
            self.remove(y_coord_mob)
            y_coord_mob = new_y_coord_mob
            self.add(y_coord_mob)
            self.wait()
            # Update horizontal line, except for the last time
            if k < n_iters-1:
                new_x = new_y
                x_coord_text = y_coord_text
                horiz_line = Line(axes.coords_to_point(curr_x, new_y), axes.coords_to_point(new_x, new_y), **self.iter_line_config)
                new_x_coord_mob = y_coord_mob.deepcopy()
                set_x_coord(new_x_coord_mob)
                self.play(
                    ApplyMethod(dot.move_to, axes.coords_to_point(new_x, new_y)),
                    ShowCreation(horiz_line),
                    ShrinkToCenter(x_coord_mob),
                    ReplacementTransform(y_coord_mob.deepcopy(), new_x_coord_mob)
                )
                x_coord_mob = new_x_coord_mob
                self.wait()
            lines.add(vert_line, horiz_line)
        self.wait(3)
        self.play(
            FadeOut(lines), Restore(dot), ShrinkToCenter(x_coord_mob), ShrinkToCenter(y_coord_mob),
            Write(x_coord_mob_copy), Write(y_coord_mob_copy),
        )
        self.wait(2)

    
class CobwebPlotForPowerTower(Scene):
    CONFIG = {
        "axes_config" : {
            "x_max" : 4.5, "y_max" : 4.5,
            "x_min" : -0.5, "y_min" : -0.5,
            "number_line_config" : {"unit_size" : 1.4},
        },
        "diag_line_config" : {
            "color" : MAROON_B,
            "stroke_width" : 3,
        },
        "starting_base" : 1.8,
    }
    def construct(self):
        self.axes_setup()
        self.legends_setup()
        self.exp_graph_and_tracker_setup()
        self.curr_base_value_setup()
        self.curr_power_tower_value_setup()
        self.changing_base()

    def axes_setup(self):
        # Axes setup
        axes = Axes(**self.axes_config)
        x_max = self.axes_config["x_max"]
        y_max = self.axes_config["y_max"]
        axes.center()
        x_title = TexMobject("x").next_to(axes.coords_to_point(x_max, 0), DOWN)
        y_title = TexMobject("y").next_to(axes.coords_to_point(0, y_max), LEFT)
        x_labels = VGroup(*[TexMobject(str(k)).next_to(axes.coords_to_point(k, 0), DOWN) for k in range(1, int(x_max)+1)])
        y_labels = VGroup(*[TexMobject(str(k)).next_to(axes.coords_to_point(0, k), LEFT) for k in range(1, int(y_max)+1)])
        origin_label = TexMobject("0").next_to(axes.coords_to_point(0, 0), DR)
        axes_group = VGroup(axes, x_labels, y_labels, x_title, y_title, origin_label)
        axes_group.set_color(LIGHT_GREY)
        axes_group.to_edge(LEFT, buff = 0.5)
        self.add(axes_group)
        diag_graph = axes.get_graph(lambda x: x, **self.diag_line_config)
        self.add(diag_graph)
        self.axes = axes
        self.axes_group = axes_group
        self.diag_graph = diag_graph

    def legends_setup(self):
        # Legends setup
        exp_line = Line(ORIGIN, RIGHT)
        exp_text = TexMobject("y=a^x").scale(1.2)
        exp_group = VGroup(exp_line, exp_text).arrange_submobjects(RIGHT, buff = 0.5).set_color(BLUE)
        diag_line = Line(ORIGIN, RIGHT, **self.diag_line_config)
        diag_text = TexMobject("y=x").scale(1.2)
        diag_group = VGroup(diag_line, diag_text).arrange_submobjects(RIGHT, buff = 0.5).set_color(MAROON_B)
        legends_group = VGroup(exp_group, diag_group)
        legends_group.arrange_submobjects(DOWN, buff = 0.3, aligned_edge = LEFT)
        legends_rect = SurroundingRectangle(legends_group, stroke_width = 3, color = LIGHT_GREY, buff = 0.3)
        legends_group.add(legends_rect)
        legends_group.shift(3*RIGHT).to_edge(UP)
        self.add(legends_group)
        self.legends_group = legends_group

    def exp_graph_and_tracker_setup(self):
        # Graph setup
        base = self.starting_base
        base_tracker = ValueTracker(base)
        iter_seq, iter_type, critical_vals = get_exp_iter_info(base)
        func = lambda x: pow(base, x)
        func_graph = self.get_func_graph(self.axes, base_tracker)
        spider_web, critical_points = self.get_spider_web_and_critical_points(self.axes, base_tracker)
        self.add(func_graph, spider_web, critical_points)
        self.func_graph = func_graph
        self.spider_web = spider_web
        self.critical_points = critical_points
        self.base_tracker = base_tracker

    def curr_base_value_setup(self):
        # Current base value setup
        a_text = TexMobject("a=").scale(1.5).set_color(BLUE)
        a_text.next_to(self.legends_group, DOWN, aligned_edge = LEFT, buff = 0.8)
        a_val_text = DecimalNumber(2, num_decimal_places = 6, show_ellipsis = True).scale(1.2).set_color(BLUE)
        a_val_text.next_to(a_text, RIGHT)
        a_val_text.add_updater(lambda m: m.set_value(self.base_tracker.get_value()))
        self.add(a_text, a_val_text)
        self.a_text = a_text

    def curr_power_tower_value_setup(self):
        # Current power tower value setup
        pt_text = TexMobject("a^{a^{a^{a^{a^{\\cdots}}}}}").scale(1.5)
        pt_text.next_to(self.a_text, DOWN, aligned_edge = LEFT, buff = 0.6)
        pt_text.add_updater(self.set_color_based_on_base_val)
        equal_sign = TexMobject("=").scale(1.2).set_color(GREEN)
        equal_sign.move_to(pt_text[0]).next_to(pt_text, RIGHT, coor_mask = [1,0,0])
        equal_sign.add_updater(self.set_conv_opacity)
        conv_val_text = DecimalNumber(2, num_decimal_places = 6, show_ellipsis = True).scale(1.2)
        conv_val_text.set_color(GREEN)
        conv_val_text.next_to(equal_sign, RIGHT)
        conv_val_text.add_updater(self.set_conv_opacity)
        conv_val_text.add_updater(self.set_power_tower_value)
        div_text = TextMobject("不存在!", color = RED, background_stroke_width = 0).scale(1.2)
        div_text.move_to(pt_text[0]).next_to(pt_text, RIGHT, coor_mask = [1,0,0])
        div_text.add_updater(self.set_div_opacity)
        self.add(pt_text, equal_sign, conv_val_text, div_text)
        self.pt_text = pt_text
        self.equal_sign = equal_sign
        self.conv_val_text = conv_val_text

    def set_color_based_on_base_val(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_color(GREEN if BASE_LOWER_BOUND <= curr_base <= BASE_UPPER_BOUND else RED)

    def set_conv_opacity(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_fill(opacity = 1 if BASE_LOWER_BOUND <= curr_base <= BASE_UPPER_BOUND else 0)

    def set_div_opacity(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_fill(opacity = 0 if BASE_LOWER_BOUND <= curr_base <= BASE_UPPER_BOUND else 1)

    def set_power_tower_value(self, mob):
        curr_base = self.base_tracker.get_value()
        if BASE_LOWER_BOUND < curr_base < BASE_UPPER_BOUND:
            solutions = exp_solver(curr_base)
            power_tower_value = exp_solver(curr_base)[0]
        elif is_close(curr_base, BASE_UPPER_BOUND):
            power_tower_value = np.exp(1)
        elif is_close(curr_base, BASE_LOWER_BOUND):
            power_tower_value = np.exp(-1)
        else:
            power_tower_value = 0
        mob.set_value(power_tower_value)

    def changing_base(self):
        # Start changing the base
        self.func_graph.add_updater(self.func_graph_update)
        self.spider_web.add_updater(self.spider_web_update)
        self.critical_points.add_updater(self.critical_points_update)
        self.play(
            ApplyMethod(self.base_tracker.set_value, 0.02),
            rate_func = None, run_time = 10,
        )
        self.wait(2)
        self.play(
            ApplyMethod(self.base_tracker.set_value, 1.8),
            rate_func = None, run_time = 10,
        )
        self.wait(2)

    def func_graph_update(self, func_graph):
        new_func_graph = self.get_func_graph(self.axes, self.base_tracker)
        func_graph.become(new_func_graph)

    def spider_web_update(self, spider_web):
        new_spider_web, new_critical_points = self.get_spider_web_and_critical_points(self.axes, self.base_tracker)
        spider_web.become(new_spider_web)

    def critical_points_update(self, critical_points):
        new_spider_web, new_critical_points = self.get_spider_web_and_critical_points(self.axes, self.base_tracker)
        critical_points.become(new_critical_points)

    def get_func_graph(self, axes, tracker):
        base = tracker.get_value()
        func = lambda x: pow(base, x)
        return axes.get_graph(func, color = BLUE)

    def get_spider_web_and_critical_points(self, axes, tracker):
        base = tracker.get_value()
        func = lambda x: pow(base, x)
        iter_seq, iter_type, critical_vals = get_exp_iter_info(base, 1)
        return IterSpiderWeb(axes, iter_seq), IterCriticalPoints(axes, iter_type, critical_vals)


class CobwebPlotForIncreasingCase(CobwebPlotForPowerTower):
    CONFIG = {
        "div_remark" : "蛛网图发散",
        "crit_title" : "临界情况(1):",
        "crit_remark" : "$y=a^x$与$y=x$恰好相切",
        "axes_config" : {
            "x_max" : 3.5, "y_max" : 3.5,
            "x_min" : -0.5, "y_min" : -0.5,
            "number_line_config" : {"unit_size" : 1.6},
        },
        "starting_base" : 1,
    }
    def construct(self):
        self.axes_setup()
        self.legends_setup()
        self.exp_graph_and_tracker_setup()
        self.curr_base_value_setup()
        self.curr_power_tower_value_setup()
        self.cobweb_status_setup()
        self.changing_base()

    def cobweb_status_setup(self):
        conv_text = TextMobject("蛛网图收敛", color = GREEN, background_stroke_width = 0)
        div_text = TextMobject(self.div_remark, color = RED, background_stroke_width = 0)
        crit_text_1 = TextMobject(self.crit_title, color = YELLOW, background_stroke_width = 0)
        crit_text_2 = TextMobject(self.crit_remark, color = YELLOW, background_stroke_width = 0)
        crit_text_2.scale(0.85)
        crit_text = VGroup(crit_text_1, crit_text_2).arrange_submobjects(DOWN, aligned_edge = LEFT)
        texts = VGroup(conv_text, div_text, crit_text)
        texts.scale(1.2)
        conv_text.next_to(self.pt_text, DOWN, aligned_edge = LEFT, buff = 0.6)
        for text in (div_text, crit_text):
            text.move_to(conv_text)
            text.next_to(conv_text.get_critical_point(UL), DR, buff = 0)
        conv_text.add_updater(self.set_strict_conv_opacity)
        div_text.add_updater(self.set_div_opacity)
        crit_text.add_updater(self.set_crit_opacity)
        # Small tweak to previously added mobjects
        self.pt_text.add_updater(self.set_color_based_on_base_val)
        self.conv_val_text.add_updater(self.set_color_based_on_base_val)
        self.equal_sign.add_updater(self.set_color_based_on_base_val)
        self.add(conv_text, div_text, crit_text)
    
    def changing_base(self):
        # Start changing the base
        self.func_graph.add_updater(self.func_graph_update)
        self.spider_web.add_updater(self.spider_web_update)
        self.critical_points.add_updater(self.critical_points_update)
        self.critical_points.add_updater(self.set_color_based_on_base_val)
        for val in (BASE_UPPER_BOUND, PI**2/6, BASE_UPPER_BOUND, 1):
            self.play(
                ApplyMethod(self.base_tracker.set_value, val),
                rate_func = None, run_time = 3,
            )
            self.wait(2)

    def set_color_based_on_base_val(self, mob):
        curr_base = self.base_tracker.get_value()
        if is_close(curr_base, BASE_UPPER_BOUND):
            color = YELLOW
        elif curr_base < BASE_UPPER_BOUND:
            color = GREEN
        else:
            color = RED
        mob.set_color(color)

    def set_conv_opacity(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_fill(opacity = 1 if curr_base <= BASE_UPPER_BOUND else 0)

    def set_strict_conv_opacity(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_fill(opacity = 1 if curr_base < BASE_UPPER_BOUND else 0)

    def set_crit_opacity(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_fill(opacity = 1 if is_close(curr_base, BASE_UPPER_BOUND) else 0)

    def set_div_opacity(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_fill(opacity = 1 if curr_base > BASE_UPPER_BOUND else 0)


class CriticalCaseUpperBound(CobwebPlotForIncreasingCase):
    def changing_base(self):
        self.func_graph.add_updater(self.func_graph_update)
        self.spider_web.add_updater(self.spider_web_update)
        self.critical_points.add_updater(self.critical_points_update)
        self.critical_points.add_updater(self.set_color_based_on_base_val)
        self.play(ApplyMethod(self.base_tracker.set_value, BASE_UPPER_BOUND), run_time = 1)
        self.wait()


class CobwebPlotForDecreasingCase(CobwebPlotForIncreasingCase):
    CONFIG = {
        "div_remark": "蛛网图在两点间振荡",
        "crit_title" : "临界情况(2):",
        "crit_remark" : "?????",
        "axes_config" : {
            "x_max" : 1.2, "y_max" : 1.2,
            "x_min" : -0.2, "y_min" : -0.2,
            "number_line_config" : {"unit_size" : 4.5},
        }
    }

    def changing_base(self):
        # Start changing the base
        self.func_graph.add_updater(self.func_graph_update)
        self.spider_web.add_updater(self.spider_web_update)
        self.critical_points.add_updater(self.critical_points_update)
        self.critical_points.add_updater(self.set_color_based_on_base_val)
        for val, rt in zip([BASE_LOWER_BOUND, 0.02, BASE_LOWER_BOUND, 1], [5, 3, 3, 5]):
            self.play(
                ApplyMethod(self.base_tracker.set_value, val),
                rate_func = None, run_time = rt,
            )
            self.wait(2)

    def set_color_based_on_base_val(self, mob):
        curr_base = self.base_tracker.get_value()
        if is_close(curr_base, BASE_LOWER_BOUND):
            color = YELLOW
        elif curr_base > BASE_LOWER_BOUND:
            color = GREEN
        else:
            color = RED
        mob.set_color(color)

    def set_conv_opacity(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_fill(opacity = 1 if curr_base >= BASE_LOWER_BOUND else 0)

    def set_strict_conv_opacity(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_fill(opacity = 1 if curr_base > BASE_LOWER_BOUND else 0)

    def set_crit_opacity(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_fill(opacity = 1 if is_close(curr_base, BASE_LOWER_BOUND) else 0)

    def set_div_opacity(self, mob):
        curr_base = self.base_tracker.get_value()
        mob.set_fill(opacity = 1 if curr_base < BASE_LOWER_BOUND else 0)


class CriticalCaseLowerBound(CobwebPlotForDecreasingCase):
    def changing_base(self):
        self.func_graph.add_updater(self.func_graph_update)
        self.spider_web.add_updater(self.spider_web_update)
        self.critical_points.add_updater(self.critical_points_update)
        self.critical_points.add_updater(self.set_color_based_on_base_val)
        self.play(ApplyMethod(self.base_tracker.set_value, BASE_LOWER_BOUND), run_time = 1)
        self.wait()


class FlipUpperLeftPart(CobwebPlotForPowerTower):
    CONFIG = {
        "axes_config" : {
            "x_max" : 1.5, "y_max" : 1.5,
            "x_min" : -0.2, "y_min" : -0.2,
            "number_line_config" : {"unit_size" : 2.8},
        },
        "flip_remark" : "沿$y=x$翻折\\\\绿色部分",
        "starting_base" : 0.4,
    }
    def construct(self):
        self.axes_setup()
        self.copy_axes_and_graphs()
        self.add_exp_and_log_graphs()

    def copy_axes_and_graphs(self):
        axes_copy, axes_group_copy, diag_graph_copy = mobs_copy = VGroup(self.axes, self.axes_group, self.diag_graph).deepcopy()
        mobs_copy.to_edge(RIGHT, buff = 0.5)
        arrow = Arrow(self.axes.get_right(), axes_copy.get_left(), buff = 0.5)
        text = TextMobject(self.flip_remark)
        text.set_width(arrow.get_width()*0.8).next_to(arrow, UP)
        self.add(mobs_copy, arrow, text)
        self.axes_copy = axes_copy

    def add_exp_and_log_graphs(self):
        base = self.starting_base
        exp_func = lambda x: pow(base, x)
        x_0 = float(exp_solver(base)[0])
        # On the left
        exp_l_1 = self.axes.get_graph(exp_func, x_max = x_0, color = GREEN)
        exp_l_2 = self.axes.get_graph(exp_func, x_min = x_0, color = BLUE)
        fx_l = TexMobject("f(x)=a^x").scale(0.8)
        fx_l.set_color([GREEN, BLUE])
        fx_l.next_to(exp_l_2.get_critical_point(DR), UP, buff = 0.6, aligned_edge = RIGHT)
        # On the right
        orig_exp = self.axes_copy.get_graph(exp_func, x_max = x_0, color = GREEN)
        exp_r_1 = DashedMobject(orig_exp, color = GREEN)
        exp_r_2 = self.axes_copy.get_graph(exp_func, x_min = x_0, color = BLUE)
        log_r = orig_exp.deepcopy().flip(about_point = self.axes_copy.coords_to_point(0, 0), axis = UR)
        fx_r = TexMobject("f(x)=a^x", color = BLUE).scale(0.8)
        fx_r.next_to(exp_r_2.get_critical_point(DR), UP, buff = 0.6, aligned_edge = RIGHT)
        gx_r = TexMobject("g(x)=\\log_{a}{x}", color = GREEN).scale(0.8)
        gx_r.next_to(self.axes_copy.get_right(), LEFT, buff = 0)
        gx_r.next_to(log_r, DOWN, coor_mask = [0,1,0])
        func_graphs = VGroup(exp_l_1, exp_l_2, exp_r_1, exp_r_2, log_r)
        texts = VGroup(fx_l, fx_r, gx_r)
        self.add(func_graphs, texts)
        self.wait()
        self.func_graphs = func_graphs
        self.texts = texts


class ExampleCobwebPlotAfterFlipping(FlipUpperLeftPart):
    def construct(self):
        super().construct()
        self.add_example_iters()

    def add_example_iters(self):
        a = self.starting_base
        xs = [1, a, pow(a, a), pow(a, pow(a, a))]
        coords = []
        for i in range(len(xs)-1):
            x = xs[i]
            coords.append([1, 0] if x == 1 else [x, x])
            coords.append([x, xs[i+1]])
        # On the left
        arrows_l = VGroup()
        for k in range(len(coords)-1):
            x_0, y_0 = coords[k]
            x_1, y_1 = coords[k+1]
            arrow = Arrow(
                self.axes.coords_to_point(x_0, y_0), self.axes.coords_to_point(x_1, y_1),
                rectangular_stem_width = 0.03, tip_length = 0.15, color = YELLOW, buff = 0,
            )
            arrows_l.add(arrow)
        # On the right
        arrows_r = arrows_l.deepcopy()
        arrows_r.shift(self.axes_copy.get_center() - self.axes.get_center())
        prev_arrows = arrows_r[2:4].deepcopy().fade(0.7)
        arrows_r[2:4].flip(about_point = self.axes_copy.coords_to_point(0, 0), axis = UR)
        arrows_r[2].shift(DOWN*0.1)
        arrows_r[-1].shift(RIGHT*0.1)
        self.add(arrows_l, arrows_r, prev_arrows)
        self.wait()


class FullCobwebPlotAfterFlipping(FlipUpperLeftPart):
    CONFIG = {
        "flip_remark" : "沿$y=x$翻折绿色部分\\\\并去掉蛛网图重叠部分",
    }
    def construct(self):
        super().construct()
        self.add_full_cobweb_plot()

    def add_full_cobweb_plot(self):
        a = self.starting_base
        iter_seq, iter_type, critical_vals = get_exp_iter_info(a, init_val = 1)
        cobweb_l = IterSpiderWeb(self.axes, iter_seq, line_width = 1.5)
        cobweb_r = ModIterSpiderWeb(self.axes_copy, iter_seq, line_width = 1.5)
        self.add(cobweb_l, cobweb_r)
        self.wait()


class DivCobwebPlotExample(FullCobwebPlotAfterFlipping, ZoomedScene):
    CONFIG = {
        "flip_remark" : "沿$y=x$翻折绿色部分\\\\并去掉蛛网图重叠部分",
        "starting_base" : 0.055,
        "zoom_factor" : 0.05,
    }

    def construct(self):
        super().construct()
        self.add_zoom_image()

    def add_zoom_image(self):
        iter_seq, iter_type, critical_vals = get_exp_iter_info(self.starting_base, init_val = 1)
        x, y = max(critical_vals), min(critical_vals)
        self.zoomed_camera.frame.move_to(self.axes_copy.coords_to_point(x, y))
        self.activate_zooming()
        # Add lines for guidance
        line_1 = Line(
            self.zoomed_camera.frame.get_critical_point(UL), self.zoomed_display.get_critical_point(DL),
            stroke_width = 1,
        )
        line_2 = Line(
            self.zoomed_camera.frame.get_critical_point(UR), self.zoomed_display.get_critical_point(DR),
            stroke_width = 1
        )
        arrow = Arrow(ORIGIN, UR*0.8, color = RED)
        arrow.next_to(self.zoomed_display.get_center(), DL, buff = 0.1)
        remark = TextMobject("这里有一个交点!", color = RED).scale(0.5)
        remark.next_to(arrow, DOWN, buff = 0.1)
        self.add_foreground_mobjects(line_1, line_2, arrow, remark)
        self.wait()
        

class BriefSummaryTexts(Scene):
    def construct(self):
        texts = [
            "$a$", "$a>e^{1/e}$", "$a=e^{1/e}$", "$1<a<e^{1/e}$", "$a=1$", "$(1/e)^e<a<1$", "$a=(1/e)^e$", "$0<a<(1/e)^e$",
            "$a^{a^{a^{a^{a^{\\cdots}}}}}$的值域/值", "—", "$e$", "$(1,e)$", "$1$", "$(1/e, 1)$", "$1/e$", "$(0, 1/e)$",
            "蛛网图的状态", "发散", "收敛", "收敛", "两点间振荡",
            "$f(x)=a^x$\\\\不动点的个数", "0", "1", "1", "2",
            "$f(x)=a^x$与$g(x)=\\log_a{x}$\\\\的交点个数", "3",
            "$e=2.718282\\dots$", "$e^{1/e}=1.444668\\dots$",
            "$1/e=0.367879\\dots$", "$(1/e)^e=0.065988\\dots$"
        ]
        indices_to_highlight = [2, 6, 10, 14, 18, 23, 28, 32, 33, 34, 35]
        texts_group = VGroup()
        for k, text in enumerate(texts):
            text_mob = TextMobject(text)
            if k in indices_to_highlight:
                text_mob.set_color(YELLOW)
            texts_group.add(text_mob)
        texts_group.arrange_submobjects_in_grid(5, buff = 0.9)
        texts_group.set_width(FRAME_WIDTH*0.95)
        self.add(texts_group)
        self.wait()


class TwoWaysOfExpressingComplexNumbers(Scene):
    def construct(self):
        x = 2.2
        y = 1.3
        r = np.sqrt(x**2+y**2)
        t = np.arctan(y/x)
        z = x + y * 1j
        plane = ComplexPlane(unit_size = 2).fade(0.4)
        plane_labels = plane.get_coordinate_labels().fade(0.4)
        dot = Dot(plane.number_to_point(z), color = YELLOW)
        # x
        line_x = Line(plane.number_to_point(0), plane.number_to_point(x), color = GREEN)
        brace_x = Brace(line_x, color = GREEN)
        text_x = TexMobject("x", color = GREEN).scale(1.5)
        brace_x.put_at_tip(text_x)
        # y
        line_y = Line(plane.number_to_point(x), plane.number_to_point(z), color = RED)
        brace_y = Brace(line_y, direction = RIGHT, color = RED)
        text_y = TexMobject("y", color = RED).scale(1.5)
        brace_y.put_at_tip(text_y)
        # r
        line_r = Line(plane.number_to_point(0), plane.number_to_point(z), color = PINK)
        brace_r = Brace(line_r, direction = rotate_vector(RIGHT, t+PI/2), color = PINK)
        text_r = TexMobject("r", color = PINK).scale(1.5)
        brace_r.put_at_tip(text_r, buff = 0.1)
        # theta
        angle_t = Sector(
            angle = t, fill_color = MAROON_B, stroke_color = MAROON_B,
            radius = 0.02, fill_opacity = 0.3, stroke_width = 3,
        )
        text_t = TexMobject("\\theta", color = MAROON_B)
        text_t.next_to(angle_t, RIGHT, buff = 0.1)
        self.add(plane, plane_labels)
        self.add(angle_t, text_t, line_r, brace_r, text_r)
        self.add(line_x, brace_x, text_x, line_y, brace_y, text_y)
        self.add(dot)
        # Two ways to express a complex number
        two_forms = TexMobject("z=x+iy=re^{i\\theta}")
        for i, color in zip([2, 5, 7, 10], [GREEN, RED, PINK, MAROON_B]):
            two_forms[i].set_color(color)
        two_forms.scale(1.5).to_corner(UL)
        expression_x = TexMobject("x=r\\cos\\theta")
        expression_x[0].set_color(GREEN), expression_x[2].set_color(PINK), expression_x[-1].set_color(MAROON_B)
        expression_y = TexMobject("y=r\\sin\\theta")
        expression_y[0].set_color(RED), expression_y[2].set_color(PINK), expression_y[-1].set_color(MAROON_B)
        expression_r = TexMobject("r=\\sqrt{x^2+y^2}")
        expression_r[0].set_color(PINK), expression_r[4].set_color(GREEN), expression_r[-2].set_color(RED)
        expression_t = TexMobject("\\theta = \\arctan{\\dfrac{y}{x}}")
        expression_t[0].set_color(MAROON_B), expression_t[-3].set_color(RED), expression_t[-1].set_color(GREEN)
        expressions = VGroup(expression_x, expression_y, expression_r, expression_t)
        expressions.scale(1.2).arrange_submobjects(DOWN, aligned_edge = LEFT, buff = 0.5)
        expressions.next_to(two_forms, DOWN, aligned_edge = LEFT, buff = 0.8)
        exp_group = VGroup(two_forms, expressions)
        rect = BackgroundRectangle(exp_group)
        self.add(rect, exp_group)
        self.wait()


class ComplexIterationForI(ZoomedScene):
    CONFIG = {
        "base" : 1j,
        "num_iters" : 80,
        "complex_plane_config" : {
            "center_point" : 6*LEFT+3*DOWN,
            "unit_size" : 5,
        },
        "line_config" : {
            "color" : YELLOW,
            "stroke_width" : 2,
            "background_stroke_width" : 1,
        },
        "zoomed_display_height": 6,
        "zoomed_display_width": 6,
        "zoom_factor" : 0.1,
    }
    def construct(self):
        plane = ComplexPlane(**self.complex_plane_config).fade(0.4)
        plane_labels = plane.get_coordinate_labels().fade(0.4)
        # Start Iteration
        power = self.base
        dots = VGroup(Dot(plane.number_to_point(power), color = YELLOW, radius = 0.05))
        lines = VGroup()
        for k in range(self.num_iters):
            power = pow(self.base, power)
            dot = Dot(plane.number_to_point(power), color = YELLOW, radius = 0.05/(k+1))
            line = Line(dots[-1].get_center(), dot.get_center(), **self.line_config)
            dots.add(dot)
            lines.add(line)
        self.add(plane, plane_labels)
        self.add(dots, lines)
        # Add zoomed image
        self.zoomed_camera.frame.move_to(plane.number_to_point(0.438283 + 0.360592 * 1j))
        self.activate_zooming()
        line_1 = Line(
            self.zoomed_camera.frame.get_critical_point(UR), self.zoomed_display.get_critical_point(UL),
            stroke_width = 1,
        )
        line_2 = Line(
            self.zoomed_camera.frame.get_critical_point(DR), self.zoomed_display.get_critical_point(DL),
            stroke_width = 1
        )
        self.add(line_1, line_2)
        self.wait()


#####
## Banner

class Banner(Scene):
    def construct(self):
        SIDE_SCALING_FACTOR = 2
        exp_tower = TexMobject("{\\sqrt{2}}^{{\\sqrt{2}}^{{\\sqrt{2}}^{{\\sqrt{2}}^{...}}}}")
        # exp_tower = ExpTower(element = "\\sqrt{2}", order = 4, is_infinite = True)
        two, left_eq = left_two = TexMobject("2", "=")
        left_two.scale(SIDE_SCALING_FACTOR)
        two.set_color(GREEN)
        right_eq, four = right_four = TexMobject("=", "4")
        right_four.scale(SIDE_SCALING_FACTOR)
        four.set_color(RED)
        equation = VGroup(left_two, exp_tower, right_four)
        equation.arrange_submobjects(RIGHT, aligned_edge = DOWN, buff = 0.25)
        equation.set_width(12)
        neq = TexMobject("\\neq")
        neq.move_to(right_eq).match_width(right_eq)
        equation.remove(right_eq).add(neq)
        equation.set_background_stroke(width = 0)
        equation.center()
        self.add(equation)
        self.wait()



