from big_ol_pile_of_manim_imports import *


class ImageAsPixels(VMobject):
    CONFIG = {
        "height": 3.5,
        "num_of_round_digits": 8,
    }

    def __init__(self, image_filename, **kwargs):
        super().__init__(**kwargs)
        self.image_filename = image_filename
        self.generate_pixels()

    def generate_pixels(self):
        self.image = ImageMobject(self.image_filename)
        n = self.get_dimension()
        pixels = VGroup(*[
            Square(
                stroke_width=0, fill_opacity=1,
                fill_color=rgba_to_color(int_rgba / 255.0),
            )
            for row in self.image.pixel_array
            for int_rgba in row
        ])
        pixels.arrange_submobjects_in_grid(n, n, buff=0)
        pixels.set_height(self.height)
        self.add(pixels)
        self.pixels = pixels
        self.reorder_pixels_by_row()    # Reorder pixels right now to make life easier

    def get_dimension(self):
        return len(self.image.pixel_array)

    def get_pixel(self, x, y):
        n = self.get_dimension()
        ind = x + y * n
        return self.pixels[ind]

    def get_position_from_coords(self, x, y):
        origin_pixel = self.get_pixel(0, 0)
        unit_length = self.get_unit_length()
        return origin_pixel.get_center() + unit_length * (x * RIGHT + y * UP)

    def get_coords_from_position(self, position):
        origin_pixel = self.get_pixel(0, 0)
        unit_length = self.get_unit_length()
        x = int(np.round((position[0] - origin_pixel.get_center()[0]) / unit_length, 0))
        y = int(np.round((position[1] - origin_pixel.get_center()[1]) / unit_length, 0))
        return x, y

    def get_unit_length(self):
        origin_pixel = self.get_pixel(0, 0)
        return origin_pixel.get_width()

    def get_discrete_cat_map_animation(self, **anim_kwargs):
        self.pixels.generate_target()
        n = self.get_dimension()
        for x in range(n):
            for y in range(n):
                ind = x + y * n
                target_pixel = self.pixels.target[ind]
                target_pos = self.get_position_from_coords(2 * x + y, x + y)
                target_pixel.move_to(target_pos)
        return MoveToTarget(self.pixels, **anim_kwargs)

    def get_cut_and_glue_animation(self, **anim_kwargs):
        self.pixels.generate_target()
        n = self.get_dimension()
        for x in range(n):
            for y in range(n):
                ind = x + y * n
                pixel = self.pixels.target[ind]
                curr_pos = pixel.get_center()
                curr_x, curr_y = self.get_coords_from_position(curr_pos)
                target_x, target_y = curr_x % n, curr_y % n
                target_pos = self.get_position_from_coords(target_x, target_y)
                pixel.move_to(target_pos)
        return MoveToTarget(self.pixels, **anim_kwargs)

    def apply_discrete_cat_map(self):
        n = self.get_dimension()
        for x in range(n):
            for y in range(n):
                pixel = self.get_pixel(x, y)
                target_x, target_y = (2 * x + y) % n, (x + y) % n
                target_pos = self.get_position_from_coords(target_x, target_y)
                pixel.move_to(target_pos)
        self.reorder_pixels_by_row()
        return self

    def reorder_pixels_by_row(self):
        # Use rounding to minimize the effect of errors
        self.pixels.submobjects.sort(
            key=lambda m: (
                np.round(m.get_center()[1], self.num_of_round_digits),
                np.round(m.get_center()[0], self.num_of_round_digits)
            )
        )

    def reorder_pixels_by_distance(self):
        center_x, center_y = self.get_center()[:2]
        self.pixels.submobjects.sort(
            key=lambda m: (
                np.round((m.get_center()[0] - center_x)**2 + (m.get_center()[1] - center_y)**2, 0),
            )
        )


class DemonstrateDiscreteCatMap(Scene):
    def construct(self):
        self.show_image()
        self.transform_image()
        self.fade_out_everything()

    def show_image(self):
        # Display image and its dimensions
        image = ImageAsPixels("ichihime_59.png", height=5)
        image.reorder_pixels_by_distance()
        image.shift(0.5 * UP)
        bottom_brace_text = BraceText(image, "59像素", brace_direction=BOTTOM)
        left_brace_text = BraceText(image, "59像素", brace_direction=LEFT)
        bottom_brace_anim = bottom_brace_text.creation_anim(label_anim=Write)
        left_brace_anim = left_brace_text.creation_anim(label_anim=Write)
        self.wait(0.5)
        self.play(
            ShowCreation(image),
            submobject_mode="lagged_start", run_time=3,
        )
        self.play(bottom_brace_anim, left_brace_anim, run_time=1.5)
        self.wait()
        self.play(FadeOut(bottom_brace_text), FadeOut(left_brace_text))
        self.wait()
        # Zoom in and show its pixels
        image.reorder_pixels_by_row()
        image.save_state()
        bottom_left_corner = image.get_critical_point(DL)
        image.generate_target()
        image.target.scale(15, about_point=bottom_left_corner)
        self.play(FocusOn(bottom_left_corner))
        self.play(MoveToTarget(image), run_time=3)
        self.wait(0.5)
        # Show how coordinates are determined
        label_pixels_text = TextMobject("以左下角为基准，\\\\给像素点", "定义坐标")
        label_pixels_text[-1].set_color(YELLOW)
        label_pixels_text.next_to(bottom_left_corner, DL)
        self.play(Write(label_pixels_text), run_time=1.5)
        self.wait()
        arrow_x = Arrow(ORIGIN, 3 * RIGHT, color=YELLOW)
        arrow_x_text = TexMobject("x", color=YELLOW).scale(1.5)
        arrow_x.next_to(bottom_left_corner, DR, buff=0.5)
        arrow_x_text.next_to(arrow_x, DOWN)
        arrow_y = Arrow(ORIGIN, 3 * UP, color=YELLOW)
        arrow_y_text = TexMobject("y", color=YELLOW).scale(1.5)
        arrow_y.next_to(bottom_left_corner, UL, buff=0.5)
        arrow_y_text.next_to(arrow_y, LEFT)
        arrow_group = VGroup(arrow_x, arrow_x_text, arrow_y, arrow_y_text)
        self.play(
            GrowArrow(arrow_x), GrowArrow(arrow_y),
            Write(arrow_x_text), Write(arrow_y_text),
            run_time=1,
        )
        self.wait()
        pixel_width = image.get_unit_length()
        coordinate_texts = VGroup(*[
            TexMobject(f"({x},\\,{y})", color=YELLOW)
            .set_width(pixel_width * 0.7)
            .move_to(image.get_pixel(x, y))
            for x in range(8)
            for y in range(5)
        ])
        coord_00_x, coord_00_y = coordinate_texts[0].get_center()[:2]
        coordinate_texts.submobjects.sort(
            key=lambda m: np.round(
                (m.get_center()[0] - coord_00_x)**2 + (m.get_center()[1] - coord_00_y)**2, 0
            ),
        )
        image.generate_target()
        image.target.set_stroke(color=YELLOW, width=1)
        self.play(
            Write(coordinate_texts),
            MoveToTarget(image),
            submobject_mode="lagged_start", run_time=2,
        )
        self.wait()
        self.play(
            FadeOut(coordinate_texts), FadeOut(arrow_group), FadeOut(label_pixels_text),
        )
        start_transforming_text = TextMobject("接下来对图像做变换", color=YELLOW)
        start_transforming_text.center().shift(3 * DOWN)
        self.play(FadeIn(start_transforming_text), Restore(image), run_time=2)
        self.wait()
        self.play(FadeOut(start_transforming_text))
        self.image = image

    def transform_image(self):
        image = self.image
        # Now start transforming the image!
        image.generate_target()
        image.target.set_height(3.5)
        image.target.move_to(4 * LEFT + 1.5 * DOWN)
        self.play(MoveToTarget(image))
        self.wait()
        # Initialize step texts
        step_1 = TextMobject("1. 移动像素点")
        step_1_remark = TexMobject("(x,\\,y) \\mapsto (2x+y,\\,x+y)")
        step_2 = TextMobject("2. 裁切与拼接")
        step_2_remark = TextMobject("把超出原图像的部分平移回来")
        VGroup(step_1, step_2).set_color(BLUE).scale(0.9)
        VGroup(step_1_remark, step_2_remark).scale(0.7)
        steps_group = VGroup(step_1, step_1_remark, step_2, step_2_remark)
        steps_group.arrange_submobjects(DOWN, aligned_edge=LEFT)
        step_1_remark.shift(0.5 * RIGHT)
        step_2_remark.shift(0.5 * RIGHT)
        steps_group.center().shift(4 * RIGHT + 2 * DOWN)
        # Step 1. Apply linear transformation
        self.play(FadeIn(step_1), FadeIn(step_1_remark))
        self.wait(2)
        cat_map_anim = image.get_discrete_cat_map_animation(run_time=3)
        self.play(cat_map_anim)
        self.wait()
        # Step 2. Cut and glue
        self.play(FadeIn(step_2), FadeIn(step_2_remark))
        self.wait(2)
        cut_and_glue_anim = image.get_cut_and_glue_animation(run_time=3)
        self.play(cut_and_glue_anim)
        image.reorder_pixels_by_row()
        self.wait()
        # Discrete cat map is the overall effect
        sur_rect = SurroundingRectangle(steps_group)
        dcm_text = TextMobject("离散猫变换", color=YELLOW)
        dcm_text.next_to(sur_rect, UP)
        self.play(ShowCreation(sur_rect), run_time=1)
        self.play(Write(dcm_text), run_time=1)
        self.wait()
        for k in range(3):
            run_time = max([2 - k, 1])
            cat_map_anim = image.get_discrete_cat_map_animation(run_time=run_time)
            self.play(cat_map_anim)
            self.wait()
            cut_and_glue_anim = image.get_cut_and_glue_animation(run_time=run_time)
            self.play(cut_and_glue_anim)
            image.reorder_pixels_by_row()
            self.wait()
        # See discrete cat map for another image with size 142x142
        another_example_text = TextMobject("再换一个", "142像素$\\times$142像素", "的图像试试看!")
        another_example_text[1].set_color(YELLOW)
        another_example_text.shift(2 * UP)
        self.play(FadeIn(another_example_text))
        self.wait(1.5)

    def fade_out_everything(self):
        self.play(FadeOut(VGroup(*self.mobjects)))
        self.wait()


class Example142x142(Scene):
    def construct(self):
        # Setup
        image = ImageAsPixels("ichihime_142.png")
        image.shift(3.5 * LEFT + 1.5 * DOWN)
        counter_text = TextMobject("离散猫变换的次数:")
        counter_num = TextMobject("0", color=YELLOW)
        counter_text.next_to(image, RIGHT, aligned_edge=DOWN, buff=1)
        counter_num.next_to(counter_text, RIGHT, buff=0.25)
        self.play(FadeIn(image), FadeIn(counter_text), FadeIn(counter_num))
        self.wait(2)
        # Demonstrate first few moves
        for k in range(3):
            cat_map_anim = image.get_discrete_cat_map_animation(run_time=3 - k)
            self.play(cat_map_anim)
            self.wait()
            cut_and_glue_anim = image.get_cut_and_glue_animation(run_time=3 - k)
            new_counter_num = TextMobject(str(k + 1), color=YELLOW)
            new_counter_num.next_to(counter_text, RIGHT, buff=0.25)
            self.play(
                cut_and_glue_anim,
                Transform(counter_num, new_counter_num)
            )
            image.reorder_pixels_by_row()
            self.wait()
        # Move the image to give space for explanations
        image.generate_target()
        image.target.set_height(5)
        image.target.center().shift(0.5 * UP + 3 * LEFT)
        counter_group = VGroup(counter_text, counter_num)
        counter_group.generate_target()
        counter_group.target.next_to(image.target, DOWN, buff=0.4)
        self.play(MoveToTarget(image), MoveToTarget(counter_group), run_time=2)
        self.wait()
        # Show results for the remaining steps until the image restores
        for k in range(105 - 3):
            image.apply_discrete_cat_map()
            new_counter_num = TextMobject(str(k + 4), color=YELLOW)
            new_counter_num.next_to(counter_text, RIGHT, buff=0.25)
            counter_num.become(new_counter_num)
            self.wait(1. / 3)
        self.wait(3)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=1)
        self.wait()


class Remark142x142(Scene):
    def construct(self):
        # Image gets scrambled after a few cat maps (No. 15)
        image_scrambled_text = TextMobject("经过几次变换，\\\\图像已经被打乱")
        image_scrambled_text.shift(3.5 * RIGHT)
        self.play(Write(image_scrambled_text), run_time=2)
        self.wait()
        self.play(FadeOut(image_scrambled_text))
        self.wait()
        # Sometimes the image partially reassembles itself (No. 35)
        image_reassembled_text = TextMobject("但是偶尔也会出现\\\\“规则”的图案")
        image_reassembled_text.shift(3.5 * RIGHT)
        self.play(Write(image_reassembled_text), run_time=2)
        self.wait()
        # Continue the mapping to see what happens
        lets_continue_text = TextMobject("继续对它做离散猫变换，\\\\看看最终会发生什么...")
        lets_continue_text.scale(0.8).set_color(YELLOW)
        lets_continue_text.next_to(image_reassembled_text, DOWN, buff=0.5)
        self.play(Write(lets_continue_text), run_time=1)
        self.wait()
        self.play(
            FadeOut(image_reassembled_text),
            FadeOut(lets_continue_text),
        )
        self.wait()
        # Express this map using matrix and modular arithmetic
        linalg_lang_text = TextMobject("用线性代数的语言，\\\\离散猫变换其实是：")
        map_tex = TexMobject(
            "(x,\\,y) \\mapsto (x,\\,y)",
            "\\begin{pmatrix} 2 & 1 \\\\ 1 & 1 \\end{pmatrix}",
            "\\mod N",
        )
        map_tex[-1].next_to(map_tex[1], RIGHT, buff=0.3)
        linalg_lang_group = VGroup(linalg_lang_text, map_tex)
        linalg_lang_group.arrange_submobjects(DOWN, aligned_edge=LEFT, buff=0.5)
        linalg_lang_group.shift(3.5 * RIGHT)
        self.play(FadeIn(linalg_lang_group), run_time=1)
        self.wait()
        # Show details part 1
        map_tex.save_state()
        map_tex.generate_target()
        map_tex.target[1].set_color(YELLOW)
        map_tex.target[-1].set_color(BLUE)
        move_pixel_text = TextMobject("移动\\\\像素", color=YELLOW).scale(0.75)
        move_pixel_text.next_to(map_tex[1], DOWN, buff=1)
        move_pixel_arrow = Arrow(
            move_pixel_text.get_top(), map_tex[1].get_bottom(),
            color=YELLOW, buff=0.1,
        )
        cut_and_glue_text = TextMobject("裁切\\\\拼接", color=BLUE).scale(0.75)
        cut_and_glue_text.next_to(map_tex[-1], DOWN)
        cut_and_glue_text.next_to(move_pixel_text, RIGHT, coor_mask=[0, 1, 0])
        cut_and_glue_arrow = Arrow(
            cut_and_glue_text.get_top(), map_tex[-1].get_bottom(),
            color=BLUE, buff=0.1,
        )
        details_1_group = VGroup(
            move_pixel_text, move_pixel_arrow,
            cut_and_glue_text, cut_and_glue_arrow,
        )
        self.play(MoveToTarget(map_tex), FadeIn(details_1_group), run_time=1)
        self.wait(2)
        self.play(Restore(map_tex), FadeOut(details_1_group), run_time=1)
        self.wait()
        # Show details part 2
        map_tex.save_state()
        map_tex.generate_target()
        map_tex.target[-1][-1].set_color(GREEN)
        size_of_image_text = TextMobject("图像大小", color=GREEN).scale(0.75)
        size_of_image_text.next_to(map_tex, DOWN, aligned_edge=RIGHT, buff=0.75)
        size_of_image_arrow = Arrow(
            size_of_image_text.get_top(), map_tex[-1][-1],
            color=GREEN, buff=0.2,
        )
        details_2_group = VGroup(size_of_image_text, size_of_image_arrow)
        self.play(MoveToTarget(map_tex), FadeIn(details_2_group), run_time=1)
        self.wait(2)
        self.play(Restore(map_tex), FadeOut(details_2_group), run_time=1)
        self.wait()
        self.play(FadeOut(linalg_lang_group), run_time=1)
        self.wait()
        # The image will restore eventually (No. ~100)
        image_restore_text = TextMobject("经过数次变换，图像最终复原")
        image_size_text = TextMobject("这个图像边长", "142", "像素")
        image_size_text[1].set_color(GREEN)
        num_of_maps_text = TextMobject("复原时经过", "105", "次变换")
        num_of_maps_text[1].set_color(YELLOW)
        matrix_form_text = TexMobject(
            "\\begin{pmatrix} 2 & 1 \\\\ 1 & 1 \\end{pmatrix}",
            "^{105}",
            "\\equiv \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}",
            "\\mod 142",
        )
        matrix_form_text[1].set_color(YELLOW)
        matrix_form_text[-1][-3:].set_color(GREEN)
        matrix_form_text[-1].next_to(matrix_form_text[-2], RIGHT, buff=0.3)
        restore_formula_group = VGroup(
            image_size_text, num_of_maps_text, matrix_form_text
        )
        restore_formula_group.arrange_submobjects(DOWN, aligned_edge=LEFT)
        restore_formula_group.scale(0.85)
        restore_group = VGroup(image_restore_text, restore_formula_group)
        restore_group.arrange_submobjects(DOWN, aligned_edge=LEFT, buff=0.7)
        restore_group.center().shift(3.5 * RIGHT)
        self.play(FadeIn(image_restore_text))
        self.wait()
        for mob in restore_formula_group:
            self.play(FadeIn(mob))
            self.wait()
        # Fade out everything
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=1)
        self.wait()


class EndingScene(Scene):
    def construct(self):
        title = TextMobject("P4. 离散猫变换")
        title.scale(1.2).set_color(YELLOW).to_corner(UL, buff=1)
        intro_1 = TextMobject("由数学家Vladimir Arnold提出，因演示效果时常用猫的照片而得名")
        intro_2 = TextMobject("""
            核心在于变换矩阵
            $A = \\begin{pmatrix} 2 & 1 \\\\ 1 & 1 \\end{pmatrix}$
            和图像大小$N$
        """)
        intro_3 = TextMobject("""
            如果图像经过$k$次变换时复原，那么
            $A^{k} \\equiv \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix} \\mod N$
        """)
        question_1 = TextMobject("(1) 为什么图像最终会复原？")
        question_2 = TextMobject("(2) 之前演示了$N=59$的情况，对它而言，复原时的变换次数$k$是多少？")
        question_3 = TextMobject("(3) 挑选一个你喜欢的素数作为$N$，用矩阵对角化的方法求复原时的变换次数。")
        group = VGroup(intro_1, intro_2, intro_3, question_1, question_2, question_3)
        group.arrange_submobjects(DOWN, aligned_edge=LEFT)
        group.scale(0.7)
        group.next_to(title, DOWN, buff = 0.5, aligned_edge = LEFT)
        questions = VGroup(question_1, question_2, question_3)
        for q in questions:
            q.set_color(BLUE_B)
            q.shift(0.2 * RIGHT)
            q.scale(0.9, about_point=q.get_critical_point(UL))
        questions.shift(0.1 * DOWN)
        square = Square(side_length=1.6).to_corner(DR)
        author = TextMobject("@Solara570")
        author.match_width(square).next_to(square.get_top(), DOWN)
        self.play(FadeIn(title), run_time=1)
        self.wait(1.5)
        for mob in group:
            self.play(FadeIn(mob), run_time=1)
            self.wait(1.5)
        self.play(FadeIn(author), run_time=1)
        self.wait(1.5)
        self.wait(3)


class Thumbnail(Scene):
    def construct(self):
        image = ImageAsPixels("ichihime_59.png")
        image.set_width(14)
        for k in range(8):
            image.apply_discrete_cat_map()
        image.set_fill(opacity=0.15)
        question = TexMobject(
            "\\begin{pmatrix} 2 & 1 \\\\ 1 & 1 \\end{pmatrix}",
            "^{???}",
            "\\equiv \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}",
            "\\mod 2855",
        )
        question[1].set_color(YELLOW)
        question[-1][-4:].set_color(GREEN)
        question[-1].next_to(question[-2], RIGHT, buff=0.3)
        question.set_width(11)
        self.add(image, question)
        self.wait()
