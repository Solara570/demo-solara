from big_ol_pile_of_manim_imports import *

class PolygonWithMark(VMobject):
    def __init__(self, n = 6, init_center = ORIGIN, **kwargs):
        super().__init__(**kwargs)
        self.polygon = RegularPolygon(
            n = n, start_angle = 0,
            fill_opacity = 0.5, fill_color = BLUE,
            stroke_color = WHITE, stroke_width = 2,
        )
        self.polygon.scale(0.5)
        self.polygon.move_to(init_center)
        self.init_center = init_center
        self.center_mark = VectorizedPoint(init_center)
        self.add(self.polygon, self.center_mark)
        self.add_polygon_position_updater()
        self.add_polygon_orientation_updater()

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

    def clear_polygon_updaters(self):
        self.polygon.clear_updaters()


class RegularHexagon(RegularPolygon):
    CONFIG = {
        "fill_opacity" : 0.5,
        "fill_color" : BLUE,
        "stroke_color" : WHITE,
        "stroke_width" : 3,
        "side_length" : 1,
    }
    def __init__(self, **kwargs):
        RegularPolygon.__init__(self, n = 6, start_angle = 0, **kwargs)
        self.scale(self.side_length)
        self.add_updater(lambda m: m.set_stroke(width = min(self.get_height(), 3)))


class AreaText(TexMobject):
    CONFIG = {
        "color" : ORANGE,
        "stroke_width" : 0,
        "background_stroke_width" : 0,
    }
    def __init__(self, target_mob = None, area = 1, **kwargs):
        self.area = area
        TexMobject.__init__(self, str(self.area), **kwargs)
        if target_mob is not None:
            curr_height, curr_width = self.get_height(), self.get_width()
            new_height, new_width = target_mob.get_height(), target_mob.get_width()
            height_factor, width_factor = new_height*0.4/curr_height, new_width*0.6/curr_width
            self.scale(min(height_factor, width_factor))
            self.move_to(target_mob)
            # def get_scale_factor(mob):
            #     curr_height, curr_width = mob.get_height(), mob.get_width()
            #     new_height, new_width = target_mob.get_height(), target_mob.get_width()
            #     height_factor, width_factor = new_height*0.4/curr_height, new_width*0.6/curr_width
            #     return min(height_factor, width_factor)
            # self.add_updater(lambda m: m.move_to(target_mob))
            # self.add_updater(lambda m: m.scale(get_scale_factor(m)))


class GeneratingHexagonalPattern(Scene):
    CONFIG = {
        "a_size" : 12,
        "b_size" : 12,
    }
    def construct(self):
        all_hexagons = self.get_all_hexagons()
        all_texts = self.get_all_texts(all_hexagons)
        # Intro
        init_hexagon = RegularHexagon(side_length = 1)
        self.play(GrowFromCenter(init_hexagon), run_time = 2)
        self.wait(1.5)
        init_area = TextMobject("面积为", "1")
        init_area.scale(1.5)
        init_area.next_to(init_hexagon, UP)
        init_area[1].set_color(ORANGE)
        self.play(Write(init_area))
        self.wait(2)
        init_text = AreaText(init_hexagon, area = 1)
        self.play(
            FadeOut(init_area[0]),
            ReplacementTransform(init_area[1], init_text),
            run_time = 1,
        )
        self.wait(2)
        # Constructing the first batch of hexagons (area = 1)
        area_1_hexagons = all_hexagons[:3]
        area_1_texts = all_texts[:3]
        self.play(
            init_hexagon.move_to, area_1_hexagons[-1],
            init_text.move_to, area_1_texts[-1],
        )
        self.add(area_1_hexagons[-1], area_1_texts[-1])
        self.remove(init_hexagon, init_text)
        self.wait(1)
        for h, t in zip(area_1_hexagons[:2], area_1_texts[:2]):
            h.generate_target()
            t.generate_target()
            h.move_to(area_1_hexagons[-1])
            t.move_to(area_1_texts[-1])
            h.fade(1)
            t.fade(1)
        self.play(
            AnimationGroup(*[
                MoveToTarget(mob)
                for mob in [area_1_hexagons[0], area_1_hexagons[1], area_1_texts[0], area_1_texts[1]]
            ]),
            run_time = 2,
        )
        self.wait(1.5)
        # Constructing the second batch of hexagons (area = 3)
        area_3_hexagons = all_hexagons[3:6]
        for h, angle in zip(area_3_hexagons, [-PI*2/3, PI*2/3, PI]):
            # Rotate each hexagon to match the gap between area 1 hexagons
            h.rotate(angle, about_point = h.get_center())
        area_3_texts = all_texts[3:6]
        self.play(
            AnimationGroup(*[DrawBorderThenFill(h) for h in area_3_hexagons]),
            run_time = 3,
        )
        on_display_mobs = VGroup(all_hexagons[:6], all_texts[:3])
        other_mobs = VGroup(all_hexagons[6:], all_texts[3:])
        self.play(
            on_display_mobs.scale, 0.8, {"about_point" : ORIGIN},
            run_time = 1,
        )
        other_mobs.scale(0.8, about_point = ORIGIN)
        self.wait()
        text = TextMobject("面积为", "3")
        text.scale(1.5).next_to(area_3_hexagons[0], UP, buff = 0.5)
        text[1].set_color(ORANGE)
        self.play(Write(text))
        self.wait(1.5)
        self.play(
            FadeOut(text[0]),
            ReplacementTransform(text[1], area_3_texts),
            run_time = 2,
        )
        self.wait(2)
        # Constructing the third batch of hexagons (area = 4)
        on_display_mobs = VGroup(all_hexagons[:6], all_texts[:6])
        other_mobs = VGroup(all_hexagons[6:], all_texts[6:])
        self.play(
            on_display_mobs.scale, 0.8, {"about_point" : ORIGIN},
            run_time = 1,
        )
        other_mobs.scale(0.8, about_point = ORIGIN)
        self.wait()
        area_4_hexagons = all_hexagons[6:9]
        for h, angle in zip(area_4_hexagons, [PI/3, PI, PI*2/3]):
            # Rotate each hexagon to match the gap between area 1 and area 3 hexagons
            h.rotate(angle, about_point = h.get_center())
        area_4_texts = all_texts[6:9]
        self.play(
            AnimationGroup(*[
                DrawBorderThenFill(h) for h in area_4_hexagons
            ]),
            run_time = 3,
        )
        self.wait()
        self.play(FadeIn(area_4_texts), run_time = 2)
        self.wait()
        # Showing the fourth and fifth batch of hexagons (area = 7 or 9)
        on_display_mobs = VGroup(all_hexagons[:9], all_texts[:9])
        other_mobs = VGroup(all_hexagons[9:], all_texts[9:])
        self.play(
            on_display_mobs.scale, 0.55, {"about_point" : ORIGIN},
            run_time = 1,
        )
        other_mobs.scale(0.55, about_point = ORIGIN)
        self.wait(1.5)
        area_7_hexagons = all_hexagons[9:15]
        area_7_texts = all_texts[9:15]
        self.play(FadeIn(area_7_hexagons), run_time = 2)
        self.wait()
        self.play(FadeIn(area_7_texts), run_time = 1)
        self.wait()
        area_9_hexagons = all_hexagons[15:18]
        area_9_texts = all_texts[15:18]
        self.play(FadeIn(area_9_hexagons), run_time = 2)
        self.wait()
        self.play(FadeIn(area_9_texts), run_time = 1)
        self.wait()
        # Continue constructing until the screen is filled
        break_indices = [18, 21, 27, 30, 36, 39, 42]
        for k in range(len(break_indices[:-1])):
            start, end = break_indices[k], break_indices[k+1]
            mobs_to_show = VGroup(all_hexagons[start:end], all_texts[start:end])
            self.play(FadeIn(mobs_to_show), run_time = 1)
            self.wait(0.5)
        other_mobs = VGroup(all_hexagons[42:], all_texts[42:])
        self.play(FadeIn(other_mobs), run_time = 1)
        self.wait()
        # Zoom out and showing the properties
        all_mobs = VGroup(all_hexagons, all_texts) 
        self.play(all_mobs.scale, 0.25, {"about_point" : ORIGIN}, run_time = 5)
        self.wait(2)
        you_may_have_noticed = TextMobject("你可能已经注意到了", "，所有正六边形的面积都是整数", color = YELLOW)
        integer_area_text = TexMobject(
            ",\\,".join(map(str, [1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25, "\\dots"])),
            color = ORANGE,
        )
        integer_area_text.scale(1.2)
        integer_area_text.to_edge(DOWN, buff = 0.5)
        you_may_have_noticed.next_to(integer_area_text, UP)
        bg_rect = BackgroundRectangle(
            VGroup(you_may_have_noticed, integer_area_text),
            fill_opacity = 0.9,
        )
        bg_rect.scale(1.5)
        self.play(FadeIn(VGroup(bg_rect, you_may_have_noticed[0])), run_time = 1)
        self.wait()
        self.play(FadeIn(you_may_have_noticed[1]), run_time = 1)
        self.wait()
        self.play(Write(integer_area_text), run_time = 2)
        self.wait(2)
        self.play(FadeOut(VGroup(you_may_have_noticed, integer_area_text)), run_time = 1)
        # Generate the same pattern using a different method
        spdm_text = TextMobject("还是上面这个图形，不过这次我们换个方式来看", color = YELLOW)
        spdm_text.move_to(bg_rect)
        self.play(FadeIn(spdm_text))
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait()

    def get_all_eisenstein_integers(self, sort_by_norm = True):
        w = 1./2 - np.sqrt(3)/2 * 1j
        all_eis = [
            a + b*w
            for a in range(0, self.a_size+1)
            for b in range(-self.b_size, self.b_size+1)
            if a>0 or b>0
        ]
        if sort_by_norm:
            all_eis.sort(key = lambda z: z.real**2+z.imag**2)
        return all_eis

    def get_all_hexagons(self):
        all_eisenstein_integers = self.get_all_eisenstein_integers()
        all_hexagons = VGroup()
        for ei in all_eisenstein_integers:
            position = complex_to_R3(ei**2)
            side_length = np.sqrt(get_norm(complex_to_R3(ei**2)))
            angle = angle_of_vector(complex_to_R3(ei))
            hexagon = RegularHexagon(side_length = side_length)
            hexagon.move_to(position)
            hexagon.rotate(angle)
            all_hexagons.add(hexagon)
        return all_hexagons

    def get_all_texts(self, all_hexagons):
        all_texts = VGroup()
        for h in all_hexagons:
            text = AreaText(target_mob = h, area = int(np.round((h.side_length)**2, 0)))
            all_texts.add(text)
        return all_texts


class SamePatternDifferentMethod(Scene):
    CONFIG = {
        "n_rows" : 12,
        "n_cols" : 12,
    }
    def construct(self):
        # 1. Construct a complex plane
        plane = ComplexPlane(color = GREEN, secondary_color = GREEN_E)
        plane.fade(0.3)
        step1 = TextMobject("1. 构建复平面")
        step1.move_to(1.5*DOWN)
        self.play(Write(step1))
        self.remove(step1)
        self.add_foreground_mobjects(step1)
        self.play(Write(plane), run_time = 2)
        self.remove_foreground_mobjects(step1)
        self.add(step1)
        self.wait(2)
        self.play(FadeOut(step1), run_time = 1)
        # 2. Add a hexagon with side length 1/2 on 1+0i
        hexagon_positions = self.get_polygon_positions()
        hexagons = VGroup(*[
            PolygonWithMark(n = 6, init_center = p)
            for p in hexagon_positions
        ])
        step2 = TextMobject("2. 在", "$1+0i$", "处放一个正六边形，边长为", "$1/2$")
        step2[1::2].set_color(YELLOW)
        step2.move_to(1.5*DOWN)
        self.play(Write(step2), run_time = 2)
        std_hexagon = hexagons[2]
        self.play(GrowFromCenter(std_hexagon))
        self.wait(2)
        self.play(FadeOut(step2), run_time = 1)
        # 3. Add remaining hexagons that all touches at corners
        remaining_hexagons = VGroup(*hexagons[:2], *hexagons[3:])
        step3 = TextMobject("3. 用同样的正六边形平铺上半平面，保持顶点接触")
        step3.move_to(1.5*DOWN)
        self.play(Write(step3), run_time = 2)
        self.play(
            LaggedStart(GrowFromCenter, remaining_hexagons, lag_ratio = 0.3),
            run_time = 5,
        )
        self.wait()
        self.play(FadeOut(step3), run_time = 1)
        self.wait()
        # (!) 4. Apply z -> z^2 transformation
        step4_1 = TextMobject("(!) 4. 对正六边形的中心做变换：", "$z \\mapsto z^2$")
        step4_1[-1].set_color(YELLOW)
        step4_2 = TextMobject("如果正六边形的中心是", "$a+bi$", "，就将它移动到", "$(a+bi)^2$")
        step4_2[1::2].set_color(YELLOW)
        step4_3 = TextMobject("并且在移动过程中", "保持正六边形与原点的角度关系")
        step4_3[-1].set_color(PINK)
        step4 = VGroup(step4_1, step4_2, step4_3)
        step4.arrange_submobjects(DOWN, aligned_edge = LEFT)
        step4.move_to(2*DOWN)
        bg_rect = BackgroundRectangle(step4)
        bg_rect.scale(1.5)
        self.play(FadeIn(bg_rect), Write(step4_1), run_time = 2)
        self.wait()
        self.play(Write(step4_2), run_time = 2)
        self.wait()
        self.play(Write(step4_3), run_time = 2)
        self.wait(2)
        ref_hex = hexagons[4]
        origin_dot = Dot(color = PINK)
        origin_dot.add_updater(lambda m: m.move_to(ORIGIN))
        center_dot = Dot(color = PINK)
        center_dot.add_updater(lambda m: m.move_to(ref_hex.get_center()))
        vertex_dot = Dot(color = PINK)
        vertex_dot.add_updater(lambda m: m.move_to(ref_hex.get_vertices()[1]))
        center_line = Line(origin_dot.get_center(), center_dot.get_center(), color = PINK)
        vertex_line = Line(center_dot.get_center(), vertex_dot.get_center(), color = PINK)
        angle = vertex_line.get_angle() - center_line.get_angle() - PI
        angle_indicator = Sector(
            outer_radius = 0.2, fill_opacity = 0.9,
            start_angle = center_line.get_angle()+PI, angle = angle,
            color = PINK,
        ).move_arc_center_to(center_dot.get_center())
        angle_arrow = Arrow(ORIGIN, RIGHT, color = PINK)
        angle_arrow.next_to(angle_indicator, LEFT)
        angle_text = TextMobject("保持不变", color = PINK)
        angle_text.next_to(angle_arrow, LEFT)
        self.play(
            FocusOn(center_dot),
            GrowFromCenter(center_dot), GrowFromCenter(vertex_dot), GrowFromCenter(origin_dot),
            GrowFromCenter(center_line), GrowFromCenter(vertex_line),
            ShowCreation(angle_indicator), ShowCreation(angle_arrow), Write(angle_text),
            run_time = 2,
        )
        self.wait()
        center_line.add_updater(lambda m: m.put_start_and_end_on(origin_dot.get_center(), center_dot.get_center()))
        vertex_line.add_updater(lambda m: m.put_start_and_end_on(center_dot.get_center(), vertex_dot.get_center()))
        angle_indicator.add_updater(
            lambda m: m.become(
                Sector(
                    outer_radius = 0.2, fill_opacity = 0.9,
                    start_angle = center_line.get_angle()+PI, angle = angle,
                    color = PINK,
                ).move_arc_center_to(center_dot.get_center())
            )
        )
        angle_arrow.add_updater(lambda m: m.next_to(angle_indicator, LEFT))
        angle_text.add_updater(lambda m: m.next_to(angle_arrow, LEFT))
        self.play(
            AnimationGroup(*[
                ComplexHomotopy(lambda z, t: z**(1+t), h.get_center_mark())
                for h in hexagons
            ]),
            run_time = 8,
        )
        self.wait(2)
        for h in hexagons:
            h.clear_polygon_updaters()
        step4_aux_mobs = VGroup(*[
            angle_text, angle_arrow, angle_indicator, vertex_line, center_line,
            center_dot, vertex_dot, origin_dot, bg_rect, step4,
        ])
        for mob in step4_aux_mobs:
            mob.clear_updaters()
        self.play(FadeOut(step4_aux_mobs))
        self.wait()
        # 5. Scale hexagons
        step5_1 = TextMobject("5. 缩放每个正六边形")
        step5_2 = TextMobject("如果正六边形中心到原点的距离为", "$N$", "，就将它放大为", "$2\\sqrt{N}$", "倍")
        step5_2.scale(0.95)
        step5_2[1::2].set_color(YELLOW)
        step5 = VGroup(step5_1, step5_2)
        step5.arrange_submobjects(DOWN, aligned_edge = LEFT)
        step5.move_to(2.5*DOWN)
        bg_rect = BackgroundRectangle(step5)
        bg_rect.scale(1.5)
        self.play(FadeIn(bg_rect), Write(step5_1), run_time = 2)
        self.wait()
        self.play(Write(step5_2), run_time = 2)
        self.wait()
        self.play(
            AnimationGroup(*[
                ApplyMethod(h.scale, 2*np.sqrt(get_norm(h.get_center())))
                for h in hexagons
            ]),
            Animation(VGroup(bg_rect, step5)),
            run_time = 3,
        )
        self.wait()
        self.play(FadeOut(plane), FadeOut(bg_rect), FadeOut(step5))
        self.wait()
        # 6. Define the unit area
        area_texts = VGroup(*[
            AreaText(target_mob = h, area = int(np.round((h.get_side_length())**2, 0)))
            for h in hexagons
        ])
        area_1_texts = area_texts[:3]
        step6 = TextMobject("6. 确定单位面积")
        step6.move_to(2.2*DOWN)
        bg_rect = BackgroundRectangle(step6)
        bg_rect.scale(1.5)
        self.play(FadeIn(bg_rect), Write(step6), run_time = 2)
        self.wait()
        self.play(FadeIn(area_1_texts))
        self.wait()
        self.play(FadeOut(VGroup(bg_rect, step6)))
        self.play(LaggedStart(FadeIn, area_texts[3:]), run_time = 3)
        all_mobs = VGroup(hexagons, area_texts)
        self.play(all_mobs.scale, 0.1, {"about_point" : ORIGIN}, run_time = 5)
        self.wait(2)
        self.play(all_mobs.fade, 1, run_time = 1)
        self.wait()

    def get_polygon_positions(self, sort_by_norm = True):
        side_length = PolygonWithMark(n = 6).get_side_length()
        all_positions = [
            np.array([(col*2-row%2)*side_length, row*np.sqrt(3)*side_length, 0])
            for row in range(0, self.n_rows)
            for col in range(-self.n_cols, self.n_cols+1)
            if (row > 0 or col > 0)
        ]
        if sort_by_norm:
            all_positions.sort(key = lambda p: get_norm(p))
        return all_positions


class SameProcedureAppliesToSquares(Scene):
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
            PolygonWithMark(n = 4, init_center = p)
            for p in square_positions
        ])
        setup_mobs = VGroup(plane, squares)
        title = TextMobject("类似的方法也可以应用在正方形上！", color = YELLOW)
        title.move_to(1.5*DOWN)
        bg_rect = BackgroundRectangle(title)
        bg_rect.scale(1.5)
        self.play(FadeIn(setup_mobs), FadeIn(bg_rect), Write(title), run_time = 2)
        self.wait(2)
        self.play(FadeOut(bg_rect), FadeOut(title), run_time = 1)
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
        self.play(all_mobs.scale, 0.1, {"about_point" : ORIGIN}, run_time = 8)
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


class EndingScene(Scene):
    def construct(self):
        title = TextMobject("P3. 共用顶点的六边形铺陈")
        title.scale(1.2).set_color(YELLOW).to_corner(UL, buff = 1)
        intro_1 = TextMobject("有没有觉得这种构造很神奇？不妨试着思考下面两个问题吧：", alignment = "")
        question_1 = TextMobject("(1) 为什么所有正六边形的面积都是整数？（可以在正三角形网格上找找原因）")
        question_2 = TextMobject("""
            (2) 为什么面积数列$\\{1,\\,3,\\,4,\\,7,\\,\\dots\\}$中的数都具有$x^2+xy+y^2 (x,y\\in\\mathbb{Z})$形式？
            """,
        )
        intro_2 = TextMobject("利用复平面可以解决下面两个问题：")
        question_3 = TextMobject("""
            (3) 为什么所有具备$x^2+xy+y^2 (x,y\\in\\mathbb{Z})$形式的正整数都在面积数列中？
            """,
        )
        question_4 = TextMobject("(4) 都是整数面积，而且相邻正六边形恰好共用顶点...这种构造的合理性在哪？")
        intro_3 = TextMobject("搞清楚了正六边形的情况，还可以继续拓展：")
        question_5 = TextMobject("(5) 刚才还展示了利用正方形的构造，它的面积数列有什么特点？")
        question_6 = TextMobject("(6) 正三角形有没有类似的构造？")
        comment = VGroup(
            intro_1, question_1, question_2, intro_2, question_3 ,question_4,
            intro_3, question_5, question_6
        )
        comment.arrange_submobjects(DOWN, aligned_edge = LEFT)
        comment.scale(0.7)
        comment.next_to(title, DOWN, buff = 0.5, aligned_edge = LEFT)
        questions = VGroup(question_1, question_2, question_3, question_4, question_5, question_6)
        for q in questions:
            q.set_color(BLUE_B)
            q.shift(0.5*RIGHT)
            q.scale(0.9, about_point = q.get_critical_point(UL))
        group_1 = VGroup(intro_1, question_1, question_2)
        group_2 = VGroup(intro_2, question_3, question_4)
        group_3 = VGroup(intro_3, question_5, question_6)
        for mob in [title, group_1, group_2, group_3]:
            self.play(FadeIn(mob), run_time = 1.5)
            self.wait(2.5)
        self.wait()
        square = Square(side_length = 1.6).to_corner(DR)
        author = TextMobject("@Solara570")
        author.match_width(square).next_to(square.get_top(), DOWN)
        self.play(FadeIn(author))
        self.wait(5)


class TransformingPolygonScene(Scene):
    CONFIG = {
        "n_sides" : 6,
        "n_rows" : 12,
        "n_cols" : 8,
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
                ComplexHomotopy(lambda z, t: z**(1+t), p.get_center_mark())
                for p in polygons
            ]),
            run_time = 3,
        )
        self.wait(2)
        self.play(
            AnimationGroup(*[
                ApplyMethod(p.scale, 2*np.sqrt(get_norm(p.get_center())))
                for p in polygons
            ]),
            run_time = 3
        )
        texts = VGroup(*[
            AreaText(target_mob = p, area = int(np.round((p.get_side_length())**2, 0)))
            for p in polygons
        ])
        self.wait(2)
        self.play(FadeOut(plane), run_time = 1)
        self.play(FadeIn(texts))
        self.play(
            ApplyMethod(VGroup(polygons, texts).scale, 1/5, {"about_point" : ORIGIN}),
            run_time = 10,
        )
        self.wait(2)

    def get_background_plane(self):
        return ComplexPlane()

    def get_polygon_positions(self):
        raise Exception("Need to be implemented in subclasses.")


class TransformingHexagonsScene(TransformingPolygonScene):
    CONFIG = {
        "n_sides" : 6,
        "n_rows" : 18,
        "n_cols" : 18,
    }
    def get_polygon_positions(self):
        side_length = PolygonWithMark(n = self.n_sides).get_side_length()
        return [
            np.array([(col*2-row%2)*side_length, row*np.sqrt(3)*side_length, 0])
            for col in range(-self.n_cols, self.n_cols+1)
            for row in range(0, self.n_rows)
            if (row > 0 or col > 0)
        ]

class TransformingSquaresScene(TransformingPolygonScene):
    CONFIG = {
        "n_sides" : 4,
        "n_rows" : 15,
        "n_cols" : 15,
    }
    def get_polygon_positions(self):
        side_length = PolygonWithMark(n = self.n_sides).get_side_length()
        return [
            np.array([col*np.sqrt(2)*side_length, row*np.sqrt(2)*side_length, 0])
            for col in range(-self.n_cols, self.n_cols+1)
            for row in range(0, self.n_rows)
            if (row > 0 or col > 0)
        ]

class Thumbnail(TransformingHexagonsScene):
    pass



