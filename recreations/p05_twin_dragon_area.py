from manimlib import *


class FSquare(Square):
    CONFIG = {
        "side_length": 4,
        "angle": PI / 4,
        "color": BLUE,
        "stroke_width": 0,
        "stroke_color": GREY,
        "fill_opacity": 1,
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        Square.__init__(self, **kwargs)
        self.set_height(self.side_length)
        self.rotate(self.get_angle())

    def get_side_length(self):
        return self.side_length

    def get_angle(self):
        return self.angle

    def get_side_anchors(self):
        return self.get_anchors()[:2]

    def get_side_direction(self):
        sp, ep = self.get_side_anchors()
        return ep - sp

    def get_diagonal_anchors(self):
        return self.get_anchors()[1::4]

    def get_diagonal_direction(self):
        sp, ep = self.get_diagonal_anchors()
        return ep - sp

    def get_new_centers(self):
        anchors = self.get_anchors()
        return (anchors[2:4].mean(axis=0), anchors[-2:].mean(axis=0))

    def get_square_color(self):
        return self.color

    def get_sub_squares(self):
        new_centers = self.get_new_centers()
        new_angle = self.get_angle() - PI / 4
        new_side_length = self.get_side_length() / np.sqrt(2)
        color = self.get_square_color()
        sub_squares = VGroup(*[
            FSquare(
                side_length=new_side_length,
                angle=new_angle, color=color,
            ).move_to(center)
            for center in new_centers
        ])
        return sub_squares

    def get_sub_triangles(self):
        center = self.get_center()
        color = self.get_square_color()
        height = self.get_side_length() / np.sqrt(2)
        angles = [self.angle + PI / 4 + k * PI / 2 for k in range(4)]
        sub_triangles = VGroup(*[
            FRightTriangle(height=height, angle=angle, color=color)
            .move_apex_to(center)
            for angle in angles
        ])
        return sub_triangles

    def get_split_animation(self):
        sq1, sq2 = self.get_sub_squares()
        anims = [
            ReplacementTransform(self, sq1),
            ReplacementTransform(self.deepcopy(), sq2),
        ]
        return AnimationGroup(*anims)


class FRightTriangle(Polygon):
    CONFIG = {
        "height": 2,
        "angle": 0,
        "color": BLUE,
        "stroke_width": 0,
        "fill_opacity": 1,
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        points = [ORIGIN, UP, RIGHT]
        Polygon.__init__(self, *points, **kwargs)
        self.set_height(self.height)
        self.rotate(self.angle, about_point=ORIGIN)

    def get_hypotenuse(self):
        vertices = self.get_vertices()
        return vertices[2] - vertices[1]

    def get_apex(self):
        return self.get_vertices()[0]

    def move_apex_to(self, point):
        self.shift(point - self.get_apex())
        return self

    def get_shift_vector(self):
        return rotate_vector(self.get_hypotenuse(), PI / 2) * 0.05

    def get_rotation_center(self):
        return self.get_vertices()[2]


class LengthMarker(VMobject):
    # Only for the four cardinal directions, and it's fixed
    CONFIG = {
        "length_thres": 0.2,
        "max_number_height": 0.3,
        "max_stroke_width": 2,
        "direction": UP,
        "buff": 0.2,
        "scaling": 1,
    }

    def __init__(self, mob, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.mob = mob
        self.add_lines()
        self.add_numbers()

    def add_lines(self):
        lines = VGroup(Line(ORIGIN, self.direction), Line(ORIGIN, self.direction), Line())

        def lines_updater(lines):
            lines[0].next_to(self.get_mob_corners()[0], direction=self.direction, buff=self.buff)
            lines[1].next_to(self.get_mob_corners()[1], direction=self.direction, buff=self.buff)
            lines[2].put_start_and_end_on(lines[0].get_center(), lines[1].get_center())
            factor = self.max_stroke_width / self.length_thres
            if self.is_up_or_down():
                lines.set_height(self.get_max_height())
                lines.set_stroke(width=factor * self.get_max_height())
            else:
                lines.set_width(self.get_max_width())
                lines.set_stroke(width=factor * self.get_max_width())
        lines.add_updater(lines_updater)
        self.add(lines)
        self.lines = lines

    def add_numbers(self):
        number = DecimalNumber()

        def number_updater(number):
            number.set_value(self.lines[2].get_length() * self.scaling)
            number.next_to(self.lines[2], self.direction, buff=0.1)
            number.set_height(self.max_number_height)
            if self.is_up_or_down():
                thres = self.lines[2].get_length() * 0.7
                if number.get_width() > thres:
                    number.set_width(thres)
            else:
                thres = self.lines[2].get_length() * 0.3
                if number.get_height() > thres:
                    number.set_height(thres)
        number.add_updater(number_updater)
        self.add(number)
        self.number = number

    def get_lines(self):
        return self.lines

    def get_number(self):
        return self.number

    def is_up_or_down(self):
        return abs(self.direction[0]) < 1e-6

    def get_mob_corners(self):
        corners = [
            self.direction + rotate_vector(self.direction, angle)
            for angle in (PI / 2, -PI / 2)
        ]
        return [self.mob.get_corner(corner) for corner in corners]

    def get_max_height(self):
        return min([self.length_thres, self.mob.get_width()])

    def get_max_width(self):
        return min([self.length_thres, self.mob.get_height()])


class IntroOnSquareSplitting(Scene):
    CONFIG = {
        "square_side_length": 4 * np.sqrt(2),
    }

    def construct(self):
        sl = self.square_side_length
        square = FSquare(side_length=sl, angle=0, color=BLUE_D)
        size_text = Tex("\\text{面积}=", "1")
        size_text[-1].set_color(BLUE_D)
        lm1 = LengthMarker(square, direction=UP, scaling=1 / sl)
        lm2 = LengthMarker(square, direction=LEFT, scaling=1 / sl)
        size_text.to_corner(DR, buff=1)
        square.scale(1e-7)
        self.add(square, lm1, lm2)
        self.wait()
        # Show square and its dimension
        self.play(square.set_width, sl, {"stretch": True}, run_time=2)
        self.wait()
        self.play(square.set_height, sl, {"stretch": True}, run_time=2)
        self.wait()
        self.play(Write(size_text))
        self.wait()
        # Fold in half
        small_square = FSquare(side_length=sl / np.sqrt(2), angle=PI / 4, color=BLUE_D)
        small_size_text = Tex("\\text{面积}=", "\\frac{1}{2}")
        small_size_text.next_to(size_text.get_left(), RIGHT, buff=0)
        small_size_text[-1].set_color(GOLD_D)
        triangles = small_square.get_sub_triangles()
        triangles.set_color(BLUE_D)
        for triangle in triangles:
            triangle.rotate(-PI, axis=triangle.get_hypotenuse())
        self.remove(square)
        self.add(small_square, triangles)
        self.play(triangles.set_color, GOLD_D)
        self.wait()
        fold_anim = AnimationGroup(*[
            Rotate(triangle, PI, axis=triangle.get_hypotenuse())
            for triangle in triangles
        ])
        self.play(
            fold_anim,
            TransformMatchingTex(size_text, small_size_text),
            run_time=3,
        )
        self.play(FadeOut(VGroup(lm1, lm2)))
        self.wait()
        # Transform 4 triangles into 2 squares
        self.remove(small_square)
        scatter_anim = AnimationGroup(*[
            ApplyMethod(triangle.shift, triangle.get_shift_vector(), rate_func=there_and_back)
            for triangle in triangles
        ])
        rotation_anim = AnimationGroup(*[
            Rotate(triangle, -1.5 * PI, about_point=triangle.get_rotation_center(), rate_func=smooth)
            for triangle in triangles[::2]
        ])
        self.play(scatter_anim, run_time=2)
        self.play(rotation_anim, run_time=3)
        self.wait()
        # Split two small squares and transform once again
        tiny_squares = small_square.get_sub_squares()
        scatter_anim_list = []
        rotation_anim_list = []
        all_tiny_triangles = VGroup()
        for tiny_square in tiny_squares:
            tiny_triangles = tiny_square.get_sub_triangles()
            tiny_triangles.set_color(GOLD_D)
            for k, triangle in enumerate(tiny_triangles):
                all_tiny_triangles.add(triangle)
                scatter_anim = ApplyMethod(
                    triangle.shift, triangle.get_shift_vector(), rate_func=there_and_back
                )
                scatter_anim_list.append(scatter_anim)
                if k % 2 == 0:
                    rotation_anim = Rotate(
                        triangle, -1.5 * PI, about_point=triangle.get_rotation_center(), rate_func=smooth
                    )
                    rotation_anim_list.append(rotation_anim)
        self.remove(triangles)
        self.play(AnimationGroup(*scatter_anim_list), run_time=2)
        self.play(AnimationGroup(*rotation_anim_list), run_time=3)
        self.wait(3)
        # Show the total area doesn't change
        sur_rect = SurroundingRectangle(small_size_text)
        constant_text = TexText("保持不变", color=YELLOW)
        constant_text.next_to(sur_rect, UP)
        self.play(ShowCreation(sur_rect))
        self.play(Write(constant_text))
        self.wait(2)
        # Fade out everything for the next scene
        tiny_tiny_squares = VGroup(*[s.get_sub_squares() for s in tiny_squares])
        tiny_tiny_squares.set_color(GOLD_D)
        self.add(tiny_tiny_squares)
        self.remove(all_tiny_triangles)
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait()


class TwinDragonTilingScene(Scene):
    CONFIG = {
        "num_iters": 12,
        # "num_iters": 3,
    }

    def construct(self):
        sl = 4
        colors = [RED_D, BLUE_D, RED_D, GREEN_D, GOLD_D, GREEN_D, RED_D, BLUE_D, RED_D]
        squares = VGroup(*[
            FSquare(side_length=sl, color=colors[k])
            for k in range(9)
        ])
        squares.arrange_in_grid(3, 3, buff=0)

        # Relevant texts
        length_text = Tex("\\text{初始对角线长度}=", "1")
        length_text[-1].set_color(GOLD_D)
        area_text = Tex("\\text{橙色区域总面积}=", "\\frac{1}{2}")
        area_text[-1].set_color(GOLD_D)
        texts = VGroup(length_text, area_text)
        texts.arrange(DOWN, aligned_edge=LEFT).scale(0.7)
        bg_rect = BackgroundRectangle(texts, buff=0.25)
        text_group = VGroup(bg_rect, texts)
        text_group.to_corner(DR, buff=0)
        # Show the squares
        center_ind = 4
        center_square = squares[center_ind]
        other_squares = [squares[k] for k in (3, 5, 1, 7, 0, 8, 2, 6)]
        self.play(
            FadeIn(center_square),
            FadeIn(text_group),
        )
        self.wait()
        idle_animations = self.get_idle_animation(bg_rect, length_text, area_text)
        self.play(
            AnimationGroup(*[
                ReplacementTransform(center_square.deepcopy().fade(1), square)
                for square in other_squares
            ], lag_ratio=0.5),
            idle_animations,
            run_time=3,
        )
        self.bring_to_front(text_group)
        self.wait()

        for k in range(self.num_iters):
            split_anims = AnimationGroup(*[
                square.get_split_animation()
                for square in self.get_square_filter()
            ])
            self.play(
                split_anims,
                idle_animations,
                run_time=1.5 if k < 5 else 1,
            )
            self.bring_to_front(text_group)
            self.wait(0.5 if k < 5 else 0.2)
        self.wait(1)

        # Duplicate the half, make it full
        # Hint
        hint = TexText("复制", "，移动", "，填空", color=YELLOW)
        # hint.scale(1.5)
        hint_bg_rect = BackgroundRectangle(hint, buff=0.2)
        hint_group = VGroup(hint_bg_rect, hint)
        for k, word in enumerate(hint):
            anim_list = []
            if k == 0:
                anim_list.append(FadeIn(hint_bg_rect))
            anim_list.append(FadeIn(hint[k]))
            self.play(AnimationGroup(*anim_list))
            self.wait(0.5)
        self.wait(0.5)
        self.play(FadeOut(hint_group))
        self.wait()

        # Action
        half = VGroup(*self.get_square_filter())
        direction = half[0].get_side_direction()
        other_half = half.deepcopy()
        other_half.generate_target()
        other_half.target.shift(direction).fade(0.1)
        other_half.fade(0)
        new_area_text = Tex("\\text{橙色区域总面积}=", "1")
        new_area_text[-1].set_color(GOLD_D)
        new_area_text.scale(0.7)
        new_area_text.next_to(area_text.get_left(), RIGHT, buff=0)
        idle_animations = self.get_idle_animation(bg_rect, length_text)
        self.play(
            MoveToTarget(other_half),
            idle_animations,
            TransformMatchingTex(area_text, new_area_text),
            run_time=3,
        )
        self.wait()

        # Scale properly to show its tiling property
        all_squares = VGroup(half, other_half)
        all_squares.generate_target()
        all_squares.target.set_height(7.5)
        all_squares.target.center().to_edge(LEFT, buff=1.5)
        new_text_group = VGroup(bg_rect, length_text, new_area_text)
        new_text_group.generate_target()
        new_text_group.target.center().to_edge(RIGHT, buff=0)
        # idle_animations = self.get_idle_animation(bg_rect, length_text, new_area_text)
        self.play(
            MoveToTarget(all_squares),
            MoveToTarget(new_text_group),
            run_time=3,
        )
        self.wait(3)

        # Fade out everything for the ending
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait()

    def get_square_filter(self):
        return filter(lambda mob: type(mob) == FSquare, self.mobjects)

    def get_idle_animation(self, *mobs):
        return AnimationGroup(*[Animation(mob) for mob in mobs])


class EndingScene(Scene):
    def construct(self):
        # Title
        title = TexText("P5. 双龙曲线的面积与平铺性质")
        title.scale(1.2).set_color(YELLOW).to_corner(UL, buff=1)
        # Some comments
        intro_1 = TexText("双龙曲线是平面填充曲线.")
        intro_2 = TexText("如果初始线段的长度是1, 那么最终图案的面积也是1.")
        intro_3 = TexText("刚才的演示顺带说明了它可以平铺整个二维平面.")
        group = VGroup(intro_1, intro_2, intro_3)
        group.arrange(DOWN, aligned_edge=LEFT)
        group.scale(0.7)
        group.next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)

        # Some useful sources
        source_title = TexText("关于龙曲线的硬核介绍:")
        source_link = TexText("http://user42.tuxfamily.org/dragon/index.html")
        source_link.scale(0.8)
        source_link.next_to(source_title, DOWN, aligned_edge=LEFT)
        source_link.shift(0.5 * RIGHT)
        source_group = VGroup(source_title, source_link)
        source_group.scale(0.7)
        source_group.next_to(group, DOWN, aligned_edge=LEFT, buff=1)
        source_group.set_color(BLUE_A)

        # WM
        square = Square(side_length=1.6).to_corner(DR)
        author = TexText("@Solara570")
        author.match_width(square).next_to(square.get_top(), DOWN)
        self.play(FadeIn(title), run_time=1)
        self.wait(1.5)
        group = VGroup(intro_1, intro_2, intro_3, source_group)
        for mob in group:
            self.play(FadeIn(mob), run_time=1)
            self.wait(1.5)
        self.play(FadeIn(author), run_time=1)
        self.wait(1.5)
        self.wait(3)


class Thumbnail(Scene):
    CONFIG = {
        "num_iters": 4,
    }

    def construct(self):
        square = FSquare(side_length=3)
        self.add(square)

        for k in range(self.num_iters):
            split_anims = AnimationGroup(*[
                s.get_split_animation()
                for s in self.mobjects
            ])
            self.play(split_anims, run_time=1)
        for mob in self.mobjects:
            mob.set_color(random_bright_color())
        self.wait(1)
