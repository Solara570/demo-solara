#########################################################
#  How to create 'perfect' patterns by power functions  #
#    Part 1 - https://zhuanlan.zhihu.com/p/359951067    #
#    Part 2 - https://zhuanlan.zhihu.com/p/360837831    #
#########################################################

from big_ol_pile_of_manim_imports import *

OMEGA = rotate_vector(UP, PI/6)

def get_argument(z):
    if z.real == 0 and z.imag != 0:
        argument = PI/2 if z.imag > 0 else -PI/2.
    elif z.imag == 0 and z.real < 0:
        argument = PI
    else:
        argument = np.arctan(z.imag/z.real)
        if z.real < 0 and z.imag > 0:
            argument += PI
        elif z.real < 0 and z.imag < 0:
            argument -= PI
    return argument



class EisensteinPlane(ComplexPlane):
    CONFIG = {
        "x_radius" : 10,
        "y_radius" : 5,
        "color" : GREEN_D,
        "secondary_color" : GREEN_E,
    }
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_matrix(np.array([
            [1,     -1/2,     0],
            [0, np.sqrt(3)/2, 0],
            [0,       0,      1],
        ]))

    def coords_to_point(self, x, y):
        x, y = np.array([x, y])
        result = self.axes.get_center()
        result += x * self.get_x_unit_size() * RIGHT
        result += y * self.get_y_unit_size() * OMEGA
        return result

    def point_to_coords(self, point):
        new_point = point - self.axes.get_center()
        y = new_point[1] / (np.sqrt(3)/2. * self.get_y_unit_size())
        x = (new_point[0] + y/2) / self.get_x_unit_size()
        return x, y

    def get_y_unit_size(self):
        return self.axes.get_height() / (2.0 * np.sqrt(3)/2 * self.y_radius)


class PolygonWithMark(VMobject):
    CONFIG = {
        "update_rotation" : False,
        "update_scale" : False,
        "polygon_config" : {
            "start_angle" : 0,
            "fill_opacity" : 0.5,
            "fill_color" : BLUE,
            "stroke_color" : WHITE,
            "stroke_width" : 1,
        },
    }
    def __init__(self, n = 6, init_center = ORIGIN, **kwargs):
        super().__init__(**kwargs)
        self.polygon = RegularPolygon(n = n, **self.polygon_config)
        self.polygon.scale(0.5)
        self.polygon.move_to(init_center)
        self.init_center = init_center
        self.center_mark = VectorizedPoint(init_center)
        self.add(self.polygon, self.center_mark)
        self.add_polygon_position_updater()
        if self.update_rotation:
            self.add_polygon_orientation_updater()
        if self.update_scale:
            self.add_polygon_scaling_updater()

    def get_center_mark(self):
        return self.center_mark

    def get_vertices(self):
        return self.polygon.get_vertices()

    def get_side_length(self):
        return get_norm(self.get_vertices()[0] - self.get_vertices()[1])

    def add_polygon_position_updater(self):
        self.polygon.add_updater(lambda m: m.move_to(self.center_mark))

    def add_polygon_orientation_updater(self):
        init_vec1 = self.polygon.get_vertices()[0] - self.init_center
        init_vec2 = self.init_center - ORIGIN
        self.init_angle = angle_between_vectors(init_vec1, init_vec2)
        def get_rotation_angle():
            vec1 = self.polygon.get_vertices()[0] - self.polygon.get_center()
            vec2 = self.polygon.get_center() - ORIGIN
            new_angle = angle_between_vectors(vec1, vec2)
            return new_angle - self.init_angle
        self.polygon.add_updater(lambda m: m.rotate(get_rotation_angle(), about_point = m.get_center()))

    def add_polygon_scaling_updater(self):
        self.init_size = 0.5
        def get_scaling_factor():
            init_norm = get_norm(self.init_center)
            curr_norm = get_norm(self.polygon.get_center())
            old_size = get_norm(self.polygon.get_vertices()[0] - self.polygon.get_vertices()[1])
            if init_norm == 0:
                exponent = 0
                new_size = 0
            elif init_norm > 1.1:
                exponent = math.log(curr_norm, init_norm)
                new_size = self.init_size * math.pow(init_norm, exponent-1)
            else:
                exponent = 1
                new_size = self.init_size
            return exponent * fdiv(new_size, old_size)
        self.polygon.add_updater(lambda m: m.scale(get_scaling_factor(), about_point = m.get_center()))

    def clear_polygon_updaters(self):
        self.polygon.clear_updaters()


class RegularTriangle(PolygonWithMark):
    def __init__(self, **kwargs):
        PolygonWithMark.__init__(self, n = 3, **kwargs)

    def get_center_of_mass(self):
        return center_of_mass(self.polygon.get_anchors()[:-1])

    def add_polygon_position_updater(self):
        self.polygon.add_updater(
            lambda m: m.shift(self.get_center_mark().get_center() - self.get_center_of_mass())
        )


class RegularHexagon(PolygonWithMark):
    CONFIG = {
        "update_stroke" : True,
        "side_length" : 1,
        "polygon_config" : {
            "fill_opacity" : 0.5,
            "fill_color" : BLUE,
            "stroke_color" : WHITE,
            "stroke_width" : 3,
        },
    }
    def __init__(self, **kwargs):
        PolygonWithMark.__init__(self, n = 6, **kwargs)
        self.scale(self.side_length)
        if self.update_stroke:
            self.add_updater(lambda m: m.set_stroke(width = min(self.get_height(), 3)))


class AreaText(TexMobject):
    CONFIG = {
        "color" : ORANGE,
        "stroke_width" : 0,
        "background_stroke_width" : 0,
        "height_multiple" : 0.4,
        "width_multiple" : 0.6,
    }
    def __init__(self, target_mob = None, area = 1, **kwargs):
        self.area = area
        TexMobject.__init__(self, str(self.area), **kwargs)
        if target_mob is not None:
            curr_height, curr_width = self.get_height(), self.get_width()
            new_height, new_width = target_mob.get_height(), target_mob.get_width()
            height_factor = new_height/curr_height * self.height_multiple,
            width_factor = new_width/curr_width * self.width_multiple
            self.scale(min(height_factor, width_factor))
            self.move_to(target_mob)


class HexagonSetupScene(Scene):
    CONFIG = {
        "a_size" : 10,
        "b_size" : 5,
    }
    def construct(self):
        self.setup_hexagons()
        self.wait()
        self.highlight_one_hexagon()

    def setup_hexagons(self):
        # Add complex plane and a bunch of hexagons
        plane = self.get_complex_plane()
        all_hexagons = self.get_all_hexagons()
        self.plane = plane
        self.all_hexagons = all_hexagons
        self.add(plane, all_hexagons)

    def highlight_one_hexagon(self):
        one_hexagon = self.all_hexagons[6]
        one_hexagon.set_color(YELLOW)

    def remove_one_hexagon(self):
        one_hexagon = self.all_hexagons[6]
        self.all_hexagons.remove(one_hexagon)

    def get_complex_plane(self):
        plane = ComplexPlane(color = GREEN, secondary_color = GREEN_E)
        plane.fade(0.3)
        return plane

    def get_all_eisenstein_integers(self, sort_by_norm = True):
        w = 1./2 - np.sqrt(3)/2 * 1j
        all_eis = [
            a + b*w
            for a in range(-self.a_size, self.a_size+1)
            for b in range(-self.b_size, self.b_size+1)
        ]
        if sort_by_norm:
            all_eis.sort(key = lambda z: z.real**2+z.imag**2)
        return all_eis

    def get_all_hexagons(self):
        all_eis = self.get_all_eisenstein_integers()
        return VGroup(*[
            RegularHexagon(init_center = complex_to_R3(ei))
            for ei in all_eis
        ])


class TransformationExampleScene(HexagonSetupScene):
    def construct(self):
        self.setup_hexagons()
        self.wait()
        self.show_transformation_examples()
        self.reset_hexagons()

    def show_transformation_examples(self):
        # Save state for gif looping
        eh_index = 12
        self.all_hexagons.save_state()
        # Highlight a specific hexagon
        example_hexagon = self.all_hexagons[eh_index]
        self.all_hexagons.generate_target()
        for k in range(len(self.all_hexagons)):
            if k == eh_index:
                self.all_hexagons.target[k].set_fill(opacity = 1)
                self.all_hexagons.target[k].set_color(PINK)
            else:    
                self.all_hexagons.target[k].fade(0.85)
        self.play(MoveToTarget(self.all_hexagons), run_time = 1)
        self.wait()
        # 1. Mark
        arrow = Arrow(ORIGIN, 1.5*LEFT, color = PINK)
        arrow.next_to(example_hexagon, RIGHT)
        z_text = TexMobject("\\dfrac{3}{2}+\\dfrac{\\sqrt{3}}{2}i", color = PINK)
        z_text.next_to(arrow, RIGHT)
        z_text_rect = BackgroundRectangle(z_text)
        self.play(ShowCreation(arrow), FadeIn(z_text_rect), Write(z_text))
        arrow.add_updater(lambda m: m.next_to(example_hexagon, RIGHT))
        z_text.add_updater(lambda m: m.next_to(arrow, RIGHT))
        z_text_rect.add_updater(lambda m: m.move_to(z_text))
        self.wait()
        # 2. Move
        text_position = 3*LEFT+2.5*UP
        move_text = TextMobject(
            "移动到", "$\\left(\\dfrac{3}{2}+\\dfrac{\\sqrt{3}}{2}i\\right)^2$",
            color = YELLOW,
        )
        move_text[1][1:-2].set_color(PINK)
        move_text.move_to(text_position)
        move_text_rect = BackgroundRectangle(move_text)
        self.play(FadeIn(move_text_rect), Write(move_text), run_time = 1)
        self.wait()
        self.play(
            AnimationGroup(*[
                ComplexHomotopy(lambda z, t: z**(1+t), h.get_center_mark())
                for h in self.all_hexagons
            ]),
            run_time = 3,
        )
        self.play(FadeOut(VGroup(move_text_rect, move_text)))
        self.wait()
        # 3. Rotate
        rotate_text = TextMobject(
            "旋转", "$\\arg\\left(\\dfrac{3}{2}+\\dfrac{\\sqrt{3}}{2}i\\right)$",
            color = YELLOW,
        )
        rotate_text[1][4:-1].set_color(PINK)
        rotate_text.move_to(text_position)
        rotate_text_rect = BackgroundRectangle(rotate_text)
        self.play(FadeIn(rotate_text_rect), Write(rotate_text), run_time = 1)
        self.wait()
        self.play(
            AnimationGroup(*[
                Rotate(h, np.arctan(h.init_center[1]/h.init_center[0]), rate_func = smooth)
                for h in self.all_hexagons[1:]
            ]),
            Animation(VGroup(rotate_text_rect, rotate_text)),
            run_time = 2,
        )
        self.play(FadeOut(VGroup(rotate_text_rect, rotate_text)))
        self.wait()
        # 4. Scale
        scale_text = TextMobject(
            "边长缩放为", "$2\\left|\\dfrac{3}{2}+\\dfrac{\\sqrt{3}}{2}i\\right|$", "倍",
            color = YELLOW,
        )
        scale_text[1][6:-5].set_color(PINK)
        scale_text.move_to(text_position)
        scale_text_rect = BackgroundRectangle(scale_text)
        self.play(FadeIn(scale_text_rect), Write(scale_text), run_time = 1)
        self.wait()
        self.play(
            AnimationGroup(*[
                ApplyMethod(h.scale, 2*abs(R3_to_complex(h.init_center)))
                for h in self.all_hexagons
            ]),
            Animation(VGroup(scale_text_rect, scale_text)),
            FadeOut(VGroup(arrow, z_text_rect, z_text)),
            run_time = 2,
        )
        self.play(FadeOut(VGroup(scale_text_rect, scale_text)))
        self.wait(3)

    def reset_hexagons(self):
        # 5. Reset
        self.play(FadeOut(self.all_hexagons))
        self.all_hexagons.restore()
        self.all_hexagons.save_state()
        self.all_hexagons.fade(1)
        self.play(self.all_hexagons.restore, run_time = 1)
        self.wait(2)


class AllInOneDemonstration(HexagonSetupScene):
    CONFIG = {
        "a_size" : 10,
        "b_size" : 5,
        "transformation_time" : 10,
    }
    def construct(self):
        self.setup_hexagons()
        self.wait()
        self.move_things_around()
        self.reset_hexagons()

    def move_things_around(self):
        self.all_hexagons.save_state()
        # Special hexagons with norm 0 or 1 that need extra care on scaling
        one_hexagon, downleft_hexagon, upleft_hexagon, right_hexagon = self.all_hexagons[:4]
        downleft_2_hexagon, upleft_2_hexagon, right_2_hexagon = self.all_hexagons[7:10]
        right_hexagon.polygon.add_updater(
            lambda m: m.set_width(2*(right_2_hexagon.get_left()[0] - m.get_center()[0]))
        )
        one_hexagon.polygon.add_updater(
            lambda m: m.set_width(2*(right_hexagon.get_left()[0]))
        )
        def get_scaling_factor(m):
            curr_size = get_norm(m.get_vertices()[0] - m.get_vertices()[1])
            new_size = right_hexagon.get_side_length()
            return new_size / curr_size
        upleft_hexagon.polygon.add_updater(
            lambda m: m.scale(get_scaling_factor(m), about_point = m.get_center())
        )
        downleft_hexagon.polygon.add_updater(
            lambda m: m.scale(get_scaling_factor(m), about_point = m.get_center())
        )
        self.play(
            AnimationGroup(*[
                ComplexHomotopy(lambda z, t: z**(1+t), h.get_center_mark())
                for h in self.all_hexagons
            ]),
            run_time = self.transformation_time,
        )
        self.wait(2)

    def reset_hexagons(self):
        self.play(self.all_hexagons.fade, 1, run_time = 1)
        self.all_hexagons.restore()
        self.play(FadeIn(self.all_hexagons), run_time = 1)

    def get_all_eisenstein_integers(self, sort_by_norm = True):
        w = -1./2 + np.sqrt(3)/2 * 1j
        all_eis = [
            a+b*w
            for a in range(-self.a_size, self.a_size+1)
            for b in range(-self.b_size, self.b_size+1)
            if (a+b*w).imag>0 or (a>=0 and (a+b*w).imag>=0)
        ]
        if sort_by_norm:
            all_eis.sort(key = lambda z: z.real**2+z.imag**2)
        return all_eis

    def get_all_hexagons(self):
        all_eis = self.get_all_eisenstein_integers()
        return VGroup(*[
            RegularHexagon(
                init_center = complex_to_R3(ei),
                update_rotation = True,
                update_scale = (k>=4),
            )
            for k, ei in enumerate(all_eis)
        ])


class HexagonTransformationScene(AllInOneDemonstration):
    CONFIG = {
        "a_size" : 10,
        "b_size" : 10,
        "transformation_time" : 5,
        "zoom_out_factor" : 0.15,
    }
    def construct(self):
        self.setup_hexagons()
        self.fade_in_mobs()
        self.move_things_around()
        self.clear_hexagon_updaters()
        self.show_area_and_zoom_out()
        self.fade_out_mobs()

    def fade_in_mobs(self):
        all_mobs = Group(*self.mobjects)
        all_mobs.save_state()
        all_mobs.fade(1)
        self.play(all_mobs.restore, run_time = 1)
        self.wait()

    def clear_hexagon_updaters(self):
        for h in self.all_hexagons:
            h.clear_polygon_updaters()

    def show_area_and_zoom_out(self):
        # Add area
        area_texts = VGroup(*[
            AreaText(target_mob = h, area = int(np.round((h.get_side_length())**2, 0)))
            for h in self.all_hexagons
        ])
        self.play(FadeOut(self.plane), run_time = 1)
        self.play(LaggedStart(FadeIn, area_texts), run_time = 2)
        self.wait()
        # Zoom out and reset
        self.play(Group(*self.mobjects).scale, self.zoom_out_factor, {"about_point" : ORIGIN}, run_time = 5)
        self.wait(2)

    def fade_out_mobs(self):
        self.play(FadeOut(Group(*self.mobjects)), run_time = 1)
        self.wait(1)
        


class SameInitialPatternDifferentPlane(HexagonSetupScene):
    def construct(self):
        self.setup_hexagons()
        self.wait()

    def get_complex_plane(self):
        plane = EisensteinPlane(
            x_radius = 20, y_radius = 20,
            secondary_line_ratio = 0
        )
        aux_main_lines = VGroup()
        for line_set in (plane.main_lines, plane.axes):
            for line in line_set:
                aux_main_line = line.deepcopy()
                aux_main_line.rotate(PI/3, about_point = aux_main_line.get_center())
                aux_main_lines.add(aux_main_line)
        plane.add(aux_main_lines)
        plane.aux_main_lines = aux_main_lines
        plane.set_color(GREEN_D)
        plane.set_stroke(width = 2)
        return plane


class OneAndOmegaPlane(SameInitialPatternDifferentPlane):
    CONFIG = {
        "a_size" : 6,
        "b_size" : 6,
        "one_color" : RED,
        "omega_color" : PINK,
    }
    def construct(self):
        self.setup_hexagons()
        self.add_center_zs()
        self.add_unit_vectors()
        self.zoom_in_a_bit()

    def add_unit_vectors(self):
        one_vec = Vector(RIGHT, color = self.one_color)
        one_text = TexMobject("1", color = self.one_color, background_stroke_width = 3)
        one_text.next_to(one_vec, DOWN, buff = 0)
        omega_vec = Vector(OMEGA, color = self.omega_color)
        omega_text = TexMobject("\\omega", color = self.omega_color, background_stroke_width = 3)
        omega_text.next_to(omega_vec, LEFT, buff = 0)
        self.add(one_vec, omega_vec, one_text, omega_text)

    def add_center_zs(self):
        for h in self.all_hexagons:
            x, y = self.plane.point_to_coords(h.get_center())
            x, y = int(np.round(x, 0)), int(np.round(y, 0))
            if y == 0:
                z_string = str(x)
            elif x != 0 and abs(y) == 1:
                z_string = str(x) + ("+" if y>0 else "-") + "\\omega"
            elif x != 0 and abs(y) != 1:
                z_string = str(x) + ("+" if y>0 else "-") + str(abs(y)) + "\\omega"
            elif x == 0 and abs(y) == 1:
                z_string = ("" if y>0 else "-") + "\\omega"
            else:
                z_string = ("" if y>0 else "-") + str(abs(y)) + "\\omega"
            tex = TexMobject(z_string, color = YELLOW, background_stroke_width = 2)
            max_height, max_width = 0.2, 0.8
            tex.scale(10)
            scaling_factor = min(max_height/tex.get_height(), max_width/tex.get_width())
            tex.scale(scaling_factor)
            tex.move_to(h.get_center())
            self.add(tex)

    def zoom_in_a_bit(self):
        Group(*self.mobjects).scale(1.7, about_point = ORIGIN)
        self.wait()

    def get_complex_plane(self):
        plane = EisensteinPlane(
            x_radius = 20, y_radius = 20,
            secondary_line_ratio = 0
        )
        aux_main_lines = VGroup()
        for line_set in (plane.main_lines, plane.axes):
            for line in line_set:
                aux_main_line = line.deepcopy()
                aux_main_line.rotate(PI/3, about_point = aux_main_line.get_center())
                aux_main_lines.add(aux_main_line)
        plane.add(aux_main_lines)
        plane.aux_main_lines = aux_main_lines
        plane.set_color(GREEN_D)
        plane.set_stroke(width = 2)
        return plane


class SameFinalPatternDifferentPlane(SameInitialPatternDifferentPlane):
    def construct(self):
        self.setup_hexagons()
        self.zoom_out_a_bit()

    def zoom_out_a_bit(self):
        Group(*self.mobjects).scale(0.5, about_point = ORIGIN)
        self.wait()

    def get_all_hexagons(self):
        all_eis = self.get_all_eisenstein_integers()
        all_hexagons = VGroup()
        for ei in all_eis:
            init_center = complex_to_R3(ei**2)
            angle = get_argument(ei)
            scaling_factor = 2*abs(ei)
            hexagon = RegularHexagon(
                init_center = init_center,
                polygon_config = {"fill_opacity" : 0.35,},
                update_rotation = False,
                update_scale = False,
            )
            hexagon.rotate(angle, about_point = hexagon.get_center())
            hexagon.scale(scaling_factor, about_point = hexagon.get_center())
            hexagon.set_stroke(width = 3)
            all_hexagons.add(hexagon)
        return all_hexagons


class AllInOneDemonstrationWithColor(AllInOneDemonstration):
    CONFIG = {
        "transformation_time" : 8,
    }
    def construct(self):
        self.setup_hexagons()
        self.color_designated_hexagons()
        self.move_things_around()
        self.reset_hexagons()

    def color_designated_hexagons(self):
        center_hexagon = self.all_hexagons[4]
        center = center_hexagon.get_center()
        surrounding_hexagons = VGroup()
        for h in self.all_hexagons:
            if 0.1 < get_norm(h.get_center() - center) < 1.1:
                surrounding_hexagons.add(h)
        center_hexagon.set_color(PINK)
        surrounding_hexagons.set_color(YELLOW)


class SquareSetupScene(Scene):
    CONFIG = {
        "a_size" : 7,
        "b_size" : 4,
    }
    def construct(self):
        self.setup_squares()
        self.highlight_one_square()
        Group(*self.mobjects).scale(1.5)

    def setup_squares(self):
        # Add complex plane and a bunch of squares
        plane = self.get_complex_plane()
        all_squares = self.get_all_squares()
        self.plane = plane
        self.all_squares = all_squares
        self.add(plane, all_squares)

    def highlight_one_square(self):
        one_square = self.all_squares[4]
        one_square.set_color(YELLOW)

    def get_complex_plane(self):
        plane = ComplexPlane(color = GREEN, secondary_color = GREEN_E)
        plane.fade(0.3)
        return plane

    def get_all_gaussian_integers(self, sort_by_norm = True):
        all_gis = [
            a + b*1j
            for a in range(-self.a_size, self.a_size+1)
            for b in range(-self.b_size, self.b_size+1)
        ]
        if sort_by_norm:
            all_gis.sort(key = lambda z: z.real**2+z.imag**2)
        return all_gis

    def get_all_squares(self):
        all_gis = self.get_all_gaussian_integers()
        return VGroup(*[
            PolygonWithMark(n = 4, init_center = complex_to_R3(gi))
            for gi in all_gis
        ])


class SquareTransformationScene(Scene):
    CONFIG = {
        "n_rows" : 12,
        "n_cols" : 12,
    }
    def construct(self):
        # Setup
        plane = ComplexPlane(color = GREEN, secondary_color = GREEN_E)
        plane.fade(0.3)
        square_positions = self.get_square_positions()
        squares = VGroup(*[
            PolygonWithMark(n = 4, init_center = p, update_rotation = True)
            for p in square_positions
        ])
        setup_mobs = VGroup(plane, squares)
        self.play(FadeIn(setup_mobs), run_time = 1)
        self.wait(1)
        # Showing the same procedure
        self.play(
            AnimationGroup(*[
                ComplexHomotopy(lambda z, t: z**(1+t), s.get_center_mark())
                for s in squares
            ]),
            run_time = 5,
        )
        self.wait(2)
        for s in squares:
            s.clear_polygon_updaters()
        self.play(
            AnimationGroup(*[
                ApplyMethod(s.scale, 2*np.sqrt(get_norm(s.get_center())))
                for s in squares
            ]),
            run_time = 3,
        )
        self.wait()
        area_texts = VGroup(*[
            AreaText(target_mob = s, area = int(np.round((s.get_side_length())**2/2, 0)))
            for s in squares
        ])
        self.play(FadeOut(plane), run_time = 1)
        self.play(LaggedStart(FadeIn, area_texts), run_time = 2)
        self.wait()
        all_mobs = VGroup(squares, area_texts)
        self.play(all_mobs.scale, 0.1, {"about_point" : ORIGIN}, run_time = 5)
        self.wait(2)
        self.play(all_mobs.fade, 1, run_time = 1)
        self.wait()

    def get_square_positions(self, sort_by_norm = True):
        side_length = PolygonWithMark(n = 4).get_side_length()
        all_positions = [
            np.array([col*np.sqrt(2)*side_length, row*np.sqrt(2)*side_length, 0])
            for col in range(-self.n_cols, self.n_cols+1)
            for row in range(0, self.n_rows)
            if (row > 0 or col > 0)
        ]
        if sort_by_norm:
            all_positions.sort(key = lambda p: get_norm(p))
        return all_positions


class TriangleSetupScene(Scene):
    CONFIG = {
        "a_size" : 10,
        "b_size" : 10,
    }
    def construct(self):
        self.setup_triangles()
        self.highlight_one_triangle()
        Group(*self.mobjects).scale(1.5, about_point = ORIGIN)

    def setup_triangles(self):
        # Add complex plane and a bunch of triangles
        plane = self.get_complex_plane()
        all_triangles = self.get_all_triangles()
        self.plane = plane
        self.all_triangles = all_triangles
        self.add(plane, all_triangles)

    def highlight_one_triangle(self):
        one_triangle = self.all_triangles[5]
        one_triangle.set_color(YELLOW)

    def get_complex_plane(self):
        plane = ComplexPlane(color = GREEN, secondary_color = GREEN_E)
        plane.fade(0.3)
        return plane

    def get_all_centers_and_flip_flags(self, sort_by_norm = True):
        w = -1./2 + np.sqrt(3)/2 * 1j
        all_centers_and_flip_flags = [
            (a+b*w+1, (a+b)%3==1)
            for a in range(-self.a_size, self.a_size+1)
            for b in range(-self.b_size, self.b_size+1)
            if (a+b)%3 != 2
        ]
        if sort_by_norm:
            all_centers_and_flip_flags.sort(key = lambda tpl: tpl[0].real**2+tpl[0].imag**2)
        return all_centers_and_flip_flags

    def get_all_triangles(self):
        all_centers_and_flip_flags = self.get_all_centers_and_flip_flags()
        all_triangles = VGroup()
        for center, flip_flag in all_centers_and_flip_flags:
            triangle = RegularTriangle(init_center = complex_to_R3(center))
            if flip_flag:
                triangle.flip(about_point = triangle.get_center_of_mass())
            all_triangles.add(triangle)
        return all_triangles


class TriangleTransformationScene(TriangleSetupScene):
    CONFIG = {
        "a_size" : 12,
        "b_size" : 12,
    }
    def construct(self):
        self.setup_triangles()
        self.move_things_around()

    def move_things_around(self):
        # Fade in
        all_mobs = Group(*self.mobjects)
        all_mobs.save_state()
        all_mobs.fade(1)
        self.play(all_mobs.restore, run_time = 1)
        self.wait()
        # Move
        self.play(
            AnimationGroup(*[
                ComplexHomotopy(lambda z, t: z**(1+t), triangle.get_center_mark())
                for triangle in self.all_triangles
            ]),
            run_time = 3,
        )
        self.wait()
        # Rotate
        rotation_anims = AnimationGroup(*[
            Rotate(
                triangle, get_argument(center),
                about_point = triangle.get_center_of_mass(), rate_func = smooth,
            )
            for triangle, (center, flip_flag) in
            zip(self.all_triangles, self.get_all_centers_and_flip_flags())
        ])
        self.play(rotation_anims, run_time = 3)
        self.wait()
        # Scale
        scaling_anims = AnimationGroup(*[
            ApplyMethod(triangle.scale, 2*abs(center), {"about_point" : triangle.get_center_of_mass()})
            for triangle, (center, flip_flag) in
            zip(self.all_triangles, self.get_all_centers_and_flip_flags())
        ])
        self.play(scaling_anims, run_time = 3)
        self.wait()
        # Add area
        area_texts = VGroup(*[
            AreaText(
                target_mob = triangle,
                area = int(np.round((triangle.get_side_length())**2/3, 0)),
                width_multiple = 0.4, height_multiple = 0.4,
                background_stroke_width = 0.5,
            ).move_to(triangle.get_center_of_mass())
            for triangle in self.all_triangles
        ])
        self.play(FadeOut(self.plane), run_time = 1)
        self.play(LaggedStart(FadeIn, area_texts), run_time = 2)
        self.wait()
        # Zoom out and reset
        self.play(Group(*self.mobjects).scale, 0.2, {"about_point" : ORIGIN}, run_time = 5)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time = 1)
        self.wait(1)

    def get_all_centers_and_flip_flags(self, sort_by_norm = True):
        w = -1./2 + np.sqrt(3)/2 * 1j
        all_centers_and_flip_flags = [
            (a+b*w+1, (a+b)%3==1)
            for a in range(-self.a_size, self.a_size+1)
            for b in range(-self.b_size, self.b_size+1)
            if (a+b)%3 != 2
        ]
        if sort_by_norm:
            all_centers_and_flip_flags.sort(key = lambda tpl: tpl[0].real**2+tpl[0].imag**2)
        return all_centers_and_flip_flags


class TransformingSceneTemplate(Scene):
    CONFIG = {
        "n_sides" : 4,
        "n_rows" : 25,
        "n_cols" : 25,
        "exponent" : 6,
        "scaling_factor" : 1e7,
        "dr_text" : None,
        "dr_text_factor" : 3,
    }
    def construct(self):
        polygon_positions = self.get_polygon_positions()
        polygons = VGroup(*[
            PolygonWithMark(n = self.n_sides, init_center = p)
            for p in polygon_positions
        ])
        plane = self.get_background_plane()
        self.add(plane, polygons)
        self.play(
            AnimationGroup(*[
                ComplexHomotopy(lambda z, t: z**(1+(self.exponent-1)*t), p.get_center_mark())
                for p in polygons
            ]),
            run_time = 3,
        )
        self.wait(2)
        self.play(
            AnimationGroup(*[
                ApplyMethod(p.scale, sf)
                for p, sf in zip(polygons, self.get_scaling_factors(polygon_positions))
            ]),
            run_time = 3
        )
        self.play(
            AnimationGroup(*[
                ApplyMethod(p.rotate, ra, {"about_point" : p.get_center()})
                for p, ra in zip(polygons, self.get_rotation_angle(polygon_positions))
            ]),
            run_time = 3
        )
        texts = VGroup(*[
            AreaText(target_mob = p, area = "")
            for p in polygons
        ])
        self.wait(2)
        self.play(FadeOut(plane), run_time = 1)
        self.play(FadeIn(texts))
        self.play(
            ApplyMethod(VGroup(polygons, texts).scale, 1/self.scaling_factor, {"about_point" : ORIGIN}),
            run_time = 10,
        )
        if self.dr_text is None:
            self.dr_text = "$z^{%d}$" % self.exponent
        tag = TextMobject(self.dr_text)
        tag.scale(self.dr_text_factor).set_color(YELLOW).to_corner(DR)
        self.add(tag)
        self.wait(2)

    def get_background_plane(self):
        return ComplexPlane()

    def get_polygon_positions(self):
        side_length = PolygonWithMark(n = self.n_sides).get_side_length()
        return [
            np.array([col*np.sqrt(2)*side_length, row*np.sqrt(2)*side_length, 0])
            for col in range(-self.n_cols, self.n_cols+1)
            for row in range(0, self.n_rows)
            if (col==0 and row!=0 and self.exponent < 4)
            or (col>0 and np.arctan(row/col) < 2*PI/self.exponent)
            or (col<0 and np.arctan(row/col)+PI < 2*PI/self.exponent)
        ]

    def get_scaling_factors(self, polygon_positions):
        scaling_factors = []
        for pos in polygon_positions:
            z = complex(pos[0], pos[1])
            sf = (self.exponent) * abs(z)**(self.exponent-1)
            scaling_factors.append(sf)
        return scaling_factors

    def get_rotation_angle(self, polygon_positions):
        rotation_angles = []
        for pos in polygon_positions:
            ra = np.arctan(pos[1]/pos[0])
            rotation_angles.append((self.exponent-1)*ra)
        return rotation_angles


class Z6TransformingScene(TransformingSceneTemplate):
    CONFIG = {
        "n_sides" : 6,
        "n_rows" : 25,
        "n_cols" : 25,
        "exponent" : 6,
        "scaling_factor" : 1e7,
        "dr_text" : "正六边形，平移到$z^6$ + 旋转$5\\arg(z)$ + 缩放$6|z|^5$倍",
        "dr_text_factor" : 1,
    }

    def get_polygon_positions(self):
        side_length = PolygonWithMark(n = self.n_sides).get_side_length()
        w = 1./2 - np.sqrt(3)/2 * 1j
        return [
            complex_to_R3(col + row*w)
            for col in range(-self.n_cols, self.n_cols+1)
            for row in range(-self.n_rows, self.n_rows+1)
            if 0 <= get_argument(col + row*w) < PI/3
        ]


class Z4TransformingSceneZoomOut(TransformingSceneTemplate):
    CONFIG = {
        "n_sides" : 4,
        "n_rows" : 25,
        "n_cols" : 25,
        "exponent" : 4,
        "scaling_factor" : 3*1e4,
        "dr_text" : "正方形，平移到$z^4$ + 旋转$3\\arg(z)$ + 缩放$4|z|^3$倍",
        "dr_text_factor" : 1,
    }


class Z4TransformingSceneZoomIn(Z4TransformingSceneZoomOut):
    CONFIG = {
        "n_rows" : 10,
        "n_cols" : 10,
        "scaling_factor" : 30,
    }


class Z3TransformingScene(TriangleTransformationScene):
    CONFIG = {
        "a_size" : 25,
        "b_size" : 25,
        "dr_text" : "正三角形，平移到$z^3$ + 旋转$2\\arg(z)$ + 缩放$3|z|^2$倍",
        "dr_text_factor" : 1,
    }

    def move_things_around(self):
        # Move
        self.play(
            AnimationGroup(*[
                ComplexHomotopy(lambda z, t: z**(1+2*t), triangle.get_center_mark())
                for triangle in self.all_triangles
            ]),
            run_time = 3,
        )
        self.wait()
        # Rotate
        rotation_anims = AnimationGroup(*[
            Rotate(
                triangle, 2*get_argument(center),
                about_point = triangle.get_center_of_mass(), rate_func = smooth,
            )
            for triangle, (center, flip_flag) in
            zip(self.all_triangles, self.get_all_centers_and_flip_flags())
        ])
        self.play(rotation_anims, run_time = 3)
        self.wait()
        # Scale
        scaling_anims = AnimationGroup(*[
            ApplyMethod(triangle.scale, 3*(abs(center))**2, {"about_point" : triangle.get_center_of_mass()})
            for triangle, (center, flip_flag) in
            zip(self.all_triangles, self.get_all_centers_and_flip_flags())
        ])
        self.play(scaling_anims, run_time = 3)
        self.wait()
        self.play(FadeOut(self.plane), run_time = 1)
        # Zoom out and reset
        self.play(Group(*self.mobjects).scale, 0.002, {"about_point" : ORIGIN}, run_time = 5)
        self.wait(2)
        tag = TextMobject(self.dr_text)
        tag.scale(self.dr_text_factor).set_color(YELLOW).to_corner(DR)
        self.add(tag)
        self.wait()


class BannerP1(HexagonTransformationScene):
    CONFIG = {
        "part_text" : "上篇",
        "zoom_out_factor" : 0.1,
    }
    def construct(self):
        self.setup_hexagons()
        self.fade_in_mobs()
        self.move_things_around()
        self.clear_hexagon_updaters()
        self.show_area_and_zoom_out()
        self.add_part_text()

    def add_part_text(self):
        part = TextMobject(self.part_text, color = YELLOW)
        part.scale(2)
        part_rect = BackgroundRectangle(part)
        part_rect.scale(2)
        part_rect.to_corner(DR, buff = 0)
        part.move_to(part_rect)
        self.add(part_rect, part)
        self.wait()


class BannerP2(BannerP1):
    CONFIG = {
        "part_text" : "下篇",
    }



