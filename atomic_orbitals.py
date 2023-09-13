from manimlib import *


# Constants
X_COOR_MASK = [1, 0, 0]
Y_COOR_MASK = [0, 1, 0]
Z_COOR_MASK = [0, 0, 1]


# Mobjects
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


# Part 1 Scenes
class IntroSceneP1(Scene):
    def construct(self):
        # What is an "atom"?
        atom_text = TexText("原子", color=YELLOW)
        atom_text.scale(2)
        self.play(FadeIn(atom_text))
        self.wait()

        # "Uncuttable" in Greek time
        text_colors = (RED, MAROON_B)
        uncut_eng = TexText("a", "tom")
        uncut_chn = TexText("不可", "分割")
        text_group = VGroup(uncut_eng, uncut_chn)
        text_group.arrange(DOWN).center()
        for text in text_group:
            for part, color in zip(text, text_colors):
                part.set_color(color)
        atom_text.generate_target()
        atom_text.target.next_to(text_group, UP, buff=0.5)
        self.play(
            MoveToTarget(atom_text),
            Write(uncut_eng),
            Write(uncut_chn),
        )
        self.wait()

        # It's correct at the time, but it's inaccurate now
        daltons_symbols = ImageMobject("Daltons_symbols.png")
        daltons_symbols.set_height(6)
        daltons_remark = RemarkText(
            daltons_symbols,
            "道尔顿圆圈符号 \\\\ 图片源：Atom - Wikipedia",
            aligned_edge=ORIGIN, scale_factor=0.5, color=GREY,
        )
        daltons_group = Group(daltons_symbols, daltons_remark)
        daltons_group.to_edge(UP).to_edge(RIGHT, buff=1)
        text_group.add(atom_text)
        self.play(
            FadeIn(daltons_group),
            text_group.animate.shift(4 * LEFT),
        )
        self.wait()

        self.play(FadeOut(text_group), FadeOut(daltons_group))
        self.wait()

        # There're many types of demonstration
        images = Group(*[
            ImageMobject(f"atom_{suffix}.png")
            for suffix in ("cloud", "contour", "density")
        ])
        for image in images:
            image.set_height(4)
        images.arrange(RIGHT, buff=0.5)
        images.set_width(13).center()

        remarks = VGroup(*[
            RemarkText(
                image, text,
                direction=UP, aligned_edge=ORIGIN, color=YELLOW
            )
            for image, text in zip(
                images,
                ["电子云", "电子云轮廓", "概率密度剖面"],
            )
        ])
        source_remark = RemarkText(
            images, "图片源：Atom - Wikipedia",
            aligned_edge=RIGHT, color=GREY, scale_factor=0.5,
        )

        mob_group = Group(images, remarks, source_remark)
        self.play(FadeIn(mob_group))
        self.wait()
        self.play(FadeOut(mob_group))
        self.wait()

        # Yet they're all manifestations of the wavefunction
        psi = Tex("\\Psi", color=YELLOW).scale(3)
        psi_name = TexText("原子轨道（波函数）", color=YELLOW)
        psi_name.next_to(psi, UP, buff=0.5)
        self.play(
            GrowFromCenter(psi),
            Write(psi_name),
            run_time=2,
        )
        self.wait()

        # Video main topics - basics & notation, figures, hybrids
        psi_group = VGroup(psi_name, psi)
        psi_group.generate_target()
        psi_group.target[1].scale(1 / 2)
        psi_group.target.arrange(RIGHT, buff=0.5)
        psi_group.target.set_color(WHITE)
        psi_group.target.scale(1.2).center().to_edge(UP)
        topics_squares = VGroup(*[Square() for k in range(3)])
        topics_squares.set_height(4).arrange(RIGHT, buff=0.5)
        topics_content = Group(*[
            ImageMobject(f"topic_{suffix}.png")
            for suffix in ("notation", "atomic_3dxy", "hybrid-sp3d2")
        ])
        for content, square in zip(topics_content, topics_squares):
            content.match_height(square)
            content.move_to(square)
        topics_remarks = VGroup(*[
            RemarkText(
                mob, text,
                direction=UP, aligned_edge=ORIGIN, scale_factor=0.8, color=YELLOW,
            )
            for mob, text in zip(
                topics_squares,
                ("记号", "图像", "杂化轨道"),
            )
        ])
        topics_group = Group(topics_squares, topics_remarks, topics_content)
        topics_group.next_to(psi_group.target, DOWN, buff=0.6)
        self.play(
            MoveToTarget(psi_group),
            FadeIn(topics_group, shift=UP),
            run_time=2,
        )
        self.wait()


class WavefunctionAsDescription(Scene):
    def construct(self):
        # We need QM to describe atoms more accurately
        qm_text = TexText("量子力学", color=YELLOW)
        atom_text = TexText("原子", color=YELLOW)
        for text, direction in zip([qm_text, atom_text], [LEFT_SIDE, RIGHT_SIDE]):
            text.move_to(direction / 2)
            text.to_edge(UP, buff=0.5)
        griffiths_full = ImageMobject("IntroToQM_Griffiths.png")
        griffiths_part = ImageMobject("IntroToQM_Griffiths_part.png")
        for image in (griffiths_full, griffiths_part):
            image.set_height(5.5)
            image.next_to(qm_text, DOWN, buff=0.5)

        self.play(GrowFromCenter(atom_text))
        self.wait()
        self.play(
            FadeIn(qm_text, shift=DOWN),
            FadeIn(griffiths_full, shift=UP),
        )
        self.wait()
        self.play(FadeIn(griffiths_part))
        self.remove(griffiths_full)
        self.wait()

        # Wavefunction is used to describe microscopic-level object
        psi = Tex("\\Psi")
        wave_func_text = TexText("波函数")
        psi.scale(2)
        psi.move_to(griffiths_part)
        wave_func_text.next_to(psi, UP, buff=0.5)
        self.add(psi, wave_func_text, griffiths_part)
        self.play(FadeOut(griffiths_part))
        self.wait()
        self.play(Indicate(psi))
        self.wait()

        # As its name suggests, it's a function of position and time
        psi_func = Tex("\\Psi", "(x,y,z", ",t", ")")
        psi_func.move_to(psi)
        variables = psi_func[1:]
        variables.generate_target()
        variables.scale(1.5).shift(1.5 * RIGHT).fade(1)
        self.play(
            ReplacementTransform(psi[0], psi_func[0], run_time=1),
            MoveToTarget(variables, lag_ratio=0.05, run_time=1.5),
        )
        self.add(psi_func)
        self.wait()

        # The wavefunction encodes the properties of the state
        property_text = TexText("物理性质")
        property_text.move_to(atom_text).move_to(wave_func_text, coor_mask=Y_COOR_MASK)
        property_eg = VGroup(*[
            TexText(text).scale(0.75).set_color(GREY_C)
            for text in ("能量", "角动量", "电子距核远近", "轨道``形状\'\'?", "$\\cdots$")
        ])
        property_eg.arrange(DOWN, aligned_edge=LEFT)
        property_eg.next_to(psi_func.get_top(), DOWN, buff=0)
        property_eg.next_to(property_text, DOWN, coor_mask=X_COOR_MASK)
        arrow = Arrow(wave_func_text, property_text, color=BLUE, buff=2)
        self.play(
            FadeTransform(wave_func_text.deepcopy(), property_text),
            GrowArrow(arrow),
        )
        self.add(property_text)
        self.wait()
        self.play(
            AnimationGroup(*[
                FadeIn(mob, shift=DOWN)
                for mob in property_eg
            ], lag_ratio=0.08)
        )
        self.wait()

        # And we know it satisfies the Schroedinger equation
        req_full_text = TexText("满足", "薛定谔方程", "的")
        req_simp_text = TexText("满足", "定态", "薛定谔方程", "的")
        simp_psi_func = Tex("\\Psi", "(x,y,z", ")")
        for text in (req_full_text, req_simp_text):
            text.next_to(wave_func_text, UP)
            text[1:-1].set_color(BLUE)
        simp_psi_func.move_to(psi_func)
        self.play(Write(req_full_text))
        self.wait()
        self.play(
            ReplacementTransform(req_full_text[0], req_simp_text[0]),
            GrowFromCenter(req_simp_text[1]),
            ReplacementTransform(req_full_text[-2:], req_simp_text[-2:]),
            ReplacementTransform(psi_func[:-2], simp_psi_func[:-1]),
            ShrinkToCenter(psi_func[-2]),
            ReplacementTransform(psi_func[-1], simp_psi_func[-1]),
        )
        psi_func = simp_psi_func
        self.add(req_simp_text, psi_func)
        self.wait()

        s_eqn = Tex(
            """
            -{\\hbar^2 \\over 2\\mu}
            \\left({\\partial^2 \\Psi \\over \\partial x^2}
            +{\\partial^2 \\Psi \\over \\partial y^2}
            +{\\partial^2 \\Psi \\over \\partial z^2}
            \\right)
            +V\\Psi=E\\Psi
            """,
        )
        s_eqn.scale(0.6).move_to(req_simp_text).next_to(psi_func, DOWN, buff=1)
        s_eqn.set_color(BLUE)
        self.play(FadeTransform(req_simp_text[1:-1].deepcopy(), s_eqn))
        self.add(s_eqn)
        self.wait()

        # Now we take a look at the SE - end of the scene
        other_mobs = Group(
            req_simp_text[0], req_simp_text[-1], wave_func_text, psi_func,
            arrow, qm_text, atom_text, property_text, property_eg,
        )
        s_eqn_text = req_simp_text[1:-1]
        s_eqn_text.generate_target()
        s_eqn_text.target.set_color(YELLOW).center().to_edge(UP)
        s_eqn.generate_target()
        s_eqn.target.scale(1 / 0.6).set_color(WHITE)
        s_eqn.target.center()

        self.play(
            FadeOut(other_mobs, run_time=1),
            MoveToTarget(s_eqn_text, run_time=2),
            MoveToTarget(s_eqn, run_time=2),
        )
        self.wait()


class AnalyzeSchroedingerEquation(Scene):
    def construct(self):
        # Setup the scene
        s_eqn_text = TexText("定态薛定谔方程", color=YELLOW)
        s_eqn_text.to_edge(UP)
        s_eqn = Tex(
            "-{\\hbar^2 \\over 2\\mu}",
            "\\left(",
            "{\\partial^2 \\Psi \\over \\partial x^2}",
            "+",
            "{\\partial^2 \\Psi \\over \\partial y^2}",
            "+",
            "{\\partial^2 \\Psi \\over \\partial z^2}",
            "\\right)",
            "+",
            "V",
            "\\Psi",
            "=",
            "E",
            "\\Psi",
        )
        self.add(s_eqn_text, s_eqn)

        # It's a PDE of some variables with a potential term
        s_eqn_text.generate_target()
        pde_text = TexText("（微分方程）", color=YELLOW)
        title_group = VGroup(s_eqn_text.target, pde_text)
        title_group.arrange(RIGHT)
        title_group.to_edge(UP)
        self.play(
            MoveToTarget(s_eqn_text),
            FadeIn(pde_text, shift=LEFT),
        )
        self.wait()

        find_psi_text = TexText("找到满足方程的$\\Psi$", color=MAROON_B)
        find_psi_text.scale(0.75).next_to(s_eqn, UP, buff=0.5)
        s_eqn.save_state()
        self.play(Write(find_psi_text), run_time=2)
        self.wait()
        self.play(
            FadeOut(find_psi_text),
            VGroup(s_eqn[0], s_eqn[-3:]).animate.fade(0.8),
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                Indicate(mob, color=BLUE)
                for mob in s_eqn[2:7:2]
            ], lag_ratio=0.05),
            run_time=1,
        )
        self.wait()
        self.play(Indicate(s_eqn[-5], color=BLUE))
        self.wait()
        self.play(s_eqn.animate.restore().shift(1.5 * UP))
        s_eqn.save_state()
        self.wait()

        # For atoms, the potential term is relatively simple: just coulomb potential
        potential_term = s_eqn[-5]
        other_terms = VGroup(s_eqn[:-5], s_eqn[-4:])
        self.play(
            potential_term.animate.set_color(BLUE),
            FocusOn(potential_term),
            other_terms.animate.fade(0.8),
        )
        self.wait()

        cp = TexText("原子：库仑势")
        cp_types = TexText("原子核-电子", "\&", "电子-电子")
        cp_group = VGroup(cp, cp_types)
        carbon_cp = TexText("碳", "原子：库仑势")
        carbon_all_cp = VGroup(*[
            TexText(f"原子核-电子{a}")
            for a in range(1, 7)
        ] + [
            TexText(f"电子{a}-电子{b}")
            for a, b in it.combinations(range(1, 7), 2)
        ])
        carbon_all_cp.arrange_in_grid(
            n_rows=6, v_buff_ratio=0.8,
            fill_rows_first=False,
        )
        hydrogen_cp = TexText("氢", "原子：库仑势")
        hydrogen_all_cp = TexText("原子核-电子1")
        for mob in (cp, carbon_cp, hydrogen_cp):
            mob.set_color(BLUE)
            mob.next_to(potential_term, DOWN, buff=0.6)
        for mob1, mob2 in zip(
            [cp, carbon_cp, hydrogen_cp], [cp_types, carbon_all_cp, hydrogen_all_cp]
        ):
            mob2.scale(0.6)
            mob2.set_color(BLUE_A)
            mob2.next_to(mob1, DOWN)
        carbon_all_cp.next_to(BOTTOM, UP, coor_mask=X_COOR_MASK)
        self.play(FadeIn(cp_group, shift=DOWN))
        self.wait()
        self.play(
            GrowFromCenter(carbon_cp[0]),
            ReplacementTransform(cp[0], carbon_cp[1]),
        )
        self.add(carbon_cp)
        self.wait()
        self.play(
            AnimationGroup(*[
                FadeTransform(cp_types[0].copy(), target)
                for target in carbon_all_cp[:6]
            ], lag_ratio=0.03),
            AnimationGroup(*[
                FadeTransform(cp_types[-1].copy(), target)
                for target in carbon_all_cp[6:]
            ], lag_ratio=0.03),
            FadeOut(cp_types),
            run_time=3,
        )
        self.wait()
        self.play(
            FadeTransform(carbon_cp, hydrogen_cp),
            FadeTransform(carbon_all_cp[0], hydrogen_all_cp[0]),
            FadeOut(carbon_all_cp[1:], shift=DOWN)
        )
        self.wait()

        # Substitute and we get Schroedinger equation for the hydrogen atom
        hydrogen_s_eqn = Tex(
            "-{\\hbar^2 \\over 2\\mu}",
            "\\left(",
            "{\\partial^2 \\Psi \\over \\partial x^2}",
            "+",
            "{\\partial^2 \\Psi \\over \\partial y^2}",
            "+",
            "{\\partial^2 \\Psi \\over \\partial z^2}",
            "\\right)",
            "-",
            "{e^2 \\over 4 \\pi \\varepsilon_0 \\sqrt{x^2+y^2+z^2}}",
            "\\Psi",
            "=",
            "E",
            "\\Psi",
        )
        hydrogen_s_eqn.move_to(s_eqn)
        hydrogen_group = VGroup(hydrogen_cp, hydrogen_all_cp)
        hydrogen_pde = TexText("氢原子", "定态薛定谔方程")
        hydrogen_pde.match_color(pde_text).to_edge(UP)
        self.play(
            ReplacementTransform(s_eqn_text[0], hydrogen_pde[1]),
            FadeTransform(pde_text[0], hydrogen_pde[0], path_arc=PI / 2),
            FadeOut(hydrogen_group, scale=0, shift=UP),
            AnimationGroup(*[
                FadeTransform(s_eqn[ind], hydrogen_s_eqn[ind])
                for ind in range(len(s_eqn))
            ]),
            run_time=2,
        )
        self.add(hydrogen_pde)
        self.wait()

        # Solve it to find atomic orbitals of the hydrogen atom
        hydrogen_ao_text = TexText("氢原子的“", "原子", "轨道", "”：", "满足方程的$\\Psi$", color=YELLOW)
        hydrogen_ao_text[1:3].set_color(BLUE)
        hydrogen_ao_text.scale(0.9)
        hydrogen_ao_text.next_to(hydrogen_s_eqn, DOWN, buff=1)
        hydrogen_ao_text.shift(2 * LEFT)

        self.play(Write(hydrogen_ao_text[-1]))
        self.wait()
        self.play(Write(hydrogen_ao_text[:-1]))
        self.wait()

        # Caveat on naming
        orbit_ul = Underline(hydrogen_ao_text[2], color=RED)
        orbit_remark = TexText("?")
        orbital_full_name = TexText("轨", "道波", "函", "数")
        orbital_abbre_name = TexText("轨", "函")
        for mob in (orbit_remark, orbital_full_name, orbital_abbre_name):
            mob.scale(0.9)
            mob.set_color(color=RED)
            mob.next_to(orbit_ul, DOWN, buff=MED_SMALL_BUFF)
        nucleus = Circle(radius=0.25, stroke_width=1, fill_opacity=0.9, color=RED)
        nucleus_sign = Tex("+", color=BLACK)
        nucleus_sign.set_height(nucleus.get_height() * 0.6)
        orbit = Circle(radius=1.5, color=BLUE, n_components=16)
        orbital = Annulus(inner_radius=1.25, outer_radius=1.75, fill_opacity=0.7, color=BLUE)
        nucleus.move_to(4 * RIGHT + 1.5 * DOWN)
        for mob in (nucleus_sign, orbit, orbital):
            mob.move_to(nucleus)

        self.play(
            ShowCreation(orbit_ul), Write(orbit_remark),
            GrowFromCenter(VGroup(nucleus, nucleus_sign)),
            DrawBorderThenFill(orbit),
            run_time=2,
        )
        self.add(orbit)
        self.wait()
        self.play(
            DrawBorderThenFill(orbital, run_time=1),
            FadeOut(orbit, run_time=2),
        )
        self.add(orbital)
        self.wait()
        self.play(FadeTransform(orbit_remark, orbital_full_name))
        self.wait()
        self.play(
            FadeOut(orbital_full_name[1::2]),
            ReplacementTransform(orbital_full_name[0], orbital_abbre_name[0]),
            ReplacementTransform(orbital_full_name[2], orbital_abbre_name[1]),
        )
        self.add(orbital_abbre_name)
        self.wait()
        self.play(FadeOut(orbit_ul), FadeOut(orbital_abbre_name))
        self.wait()

        # Now we tackle the SE - end of the scene
        atom_diagram = VGroup(nucleus, nucleus_sign, orbital)
        self.play(FadeOut(atom_diagram), FadeOut(hydrogen_ao_text))
        self.wait()


class SolutionsOfHydrogenSE(Scene):
    def construct(self):
        # Setup the scene
        title_pde = TexText("氢原子定态薛定谔方程", color=YELLOW)
        hydrogen_s_eqn = Tex(
            "-{\\hbar^2 \\over 2\\mu}",
            "\\left(",
            "{\\partial^2 \\Psi \\over \\partial x^2}",
            "+",
            "{\\partial^2 \\Psi \\over \\partial y^2}",
            "+",
            "{\\partial^2 \\Psi \\over \\partial z^2}",
            "\\right)",
            "-",
            "{e^2 \\over 4 \\pi \\varepsilon_0 \\sqrt{x^2+y^2+z^2}}",
            "\\Psi",
            "=",
            "E",
            "\\Psi",
        )
        title_pde.to_edge(UP)
        hydrogen_s_eqn.shift(1.5 * UP)
        self.add(title_pde, hydrogen_s_eqn)

        # Show solution doc
        doc_imgs = Group(*[
            ImageMobject(f"doc-{i:02}.png")
            for i in range(1, 22)
        ])
        doc_imgs.set_height(7).next_to(title_pde.get_top(), DOWN, buff=-0.05)
        doc_imgs.arrange(RIGHT)
        doc_imgs.next_to(RIGHT_SIDE, RIGHT)
        shift_vec = -doc_imgs[-1].get_center()
        self.play(
            doc_imgs.animate.shift(shift_vec),
            run_time=20,
        )
        self.wait()

        # Show solution
        title_wavefunc = TexText("氢原子轨道波函数", color=YELLOW)
        title_wavefunc.move_to(title_pde)
        target_img = doc_imgs[17]
        solution_img = ImageMobject("doc-18-zoom.png")
        solution_img.match_width(target_img).move_to(target_img)
        solution_img.shift(1.1 * UP)
        solution_img.generate_target()
        solution_img.target.set_width(9).fade(0.1)
        solution_img.target.next_to(title_pde, DOWN, buff=0.3)
        shift_vec = -target_img.get_center()
        doc_imgs.generate_target()
        doc_imgs.target.shift(shift_vec).fade(1)
        self.remove(title_pde)
        self.add(title_wavefunc, doc_imgs)
        self.play(
            FadeOut(hydrogen_s_eqn, run_time=2),
            MoveToTarget(doc_imgs, run_time=5),
            MoveToTarget(solution_img, run_time=5),
        )
        self.remove(doc_imgs)
        self.wait()

        # It looks daunting, but we're only trying to understand the structure
        rect = Rectangle(
            height=0.9,
            width=solution_img.get_width() * 0.9,
            color=RED
        )
        rect.shift(1.3 * UP)
        self.play(ShowCreation(rect))
        self.wait()
        orbital_img = ImageMobject("orbital-3d-collection.png")
        height = (rect.get_bottom() - solution_img.get_bottom())[1] * 0.99
        orbital_img.set_height(height)
        orbital_img.next_to(solution_img.get_bottom(), UP, buff=0)
        self.play(FadeIn(orbital_img, shift=UP))
        self.wait()
        cover_rect = Rectangle(
            width=orbital_img.get_width(), height=orbital_img.get_height(),
            color=BLACK, fill_opacity=1, stroke_width=0,
        )
        cover_rect.move_to(orbital_img)
        psi_func = Tex("\\Psi", "_{n ", "\\ell", " m}", "(", "r", ", ", "\\theta", ", ", "\\varphi", ")")
        psi_func.scale(1.5)
        psi_func.move_to(rect)
        self.add(cover_rect, psi_func, orbital_img)

        self.play(
            FadeOut(solution_img),
            FadeOut(rect),
        )
        self.wait()

        # There's a coordinate transform happening
        coord_img = ImageMobject("coordinate_transform.png")
        coord_img.to_edge(RIGHT, buff=0.8)
        coord_remark = RemarkText(
            coord_img,
            "（$\\theta$和$\\varphi$写反了吗？并没有，这只不过是传统的延续）",
            aligned_edge=ORIGIN, scale_factor=0.4, color=GREY,
        )
        coord_group = Group(coord_img, coord_remark)
        self.play(*[
            AnimationGroup(*[
                Indicate(mob)
                for mob in psi_func[-6::2]
            ], lag_ratio=0.05)
        ])
        self.wait()
        self.play(
            psi_func.animate.shift(LEFT),
            FadeIn(coord_group, shift=LEFT),
            FadeOut(orbital_img, shift=LEFT),
        )
        self.wait()

        # The most interesting part are the subscripts nlm
        sub_n, sub_l, sub_m = sub_group = psi_func[1:4]
        sub_colors = (RED, GREEN, BLUE)
        self.play(*[
            AnimationGroup(*[
                Indicate(
                    mob, color=color,
                    rate_func=there_and_back_with_pause,
                )
                for mob, color in zip(psi_func[1:4], sub_colors)
            ], lag_ratio=0.05, run_time=2),
        ])
        self.wait()

        quant_n, quant_l, quant_m = quant_group = VGroup(*[
            Tex(text, color=color)
            for text, color in zip(["n", "\\ell", "m"], sub_colors)
        ])
        quant_group.arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        quant_group.next_to(psi_func[:5], DOWN, buff=0.9)
        sub_group.generate_target()
        for mob, color in zip(sub_group.target, sub_colors):
            mob.set_color(color)
        self.play(
            AnimationGroup(*[
                FadeTransform(source_mob.copy(), target_mob[0], path_arc=-PI / 3)
                for source_mob, target_mob in zip(sub_group, quant_group)
            ], run_time=2),
            MoveToTarget(sub_group, run_time=1),
            FadeOut(coord_group, run_time=1),
        )
        self.add(quant_group)
        self.wait()

        # nlm are quantized and they have some constraints
        descrp_type_n, descrp_type_l, descrp_type_m = descrp_type_group = VGroup(*[
            TexText(text, color=color)
            for text, color in zip(["正整数", "自然数", "整数"], sub_colors)
        ])
        for mob1, mob2 in zip(descrp_type_group, quant_group):
            mob1.next_to(mob2, LEFT, buff=0.1)

        self.play(
            AnimationGroup(*[
                FadeTransform(mob2.copy(), mob1)
                for mob1, mob2 in zip(descrp_type_group, quant_group)
            ], lag_ratio=0.05),
            run_time=2,
        )
        self.add(descrp_type_group)
        self.wait()

        constraints = VGroup(
            Tex("n", " > ", "\\ell"),
            Tex("\\ell", " \\geq ", "|", "m", "|"),
        )
        constraints.arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        constraints.next_to(quant_group, RIGHT, buff=2)
        for mob in constraints:
            mob.set_color_by_tex_to_color_map(
                dict(zip(["n", "\\ell", "m"], sub_colors))
            )
        self.play(FadeIn(constraints, shift=2 * RIGHT))
        self.wait()

        # So nlm are called quantum numbers
        descrp_name_n, descrp_name_l, descrp_name_m = descrp_name_group = VGroup(
            TexText("主", "量子数", color=sub_colors[0]),
            TexText("角", "量子数", color=sub_colors[1]),
            TexText("磁", "量子数", color=sub_colors[2]),
        )
        for mob1, mob2 in zip(descrp_name_group, descrp_type_group):
            mob1.next_to(mob2.get_right(), LEFT, buff=0)
        self.play(
            AnimationGroup(*[
                FadeTransform(mob2, mob1[1])
                for mob1, mob2 in zip(descrp_name_group, descrp_type_group)
            ], lag_ratio=0.05),
            run_time=2,
        )
        self.wait()
        self.play(ShowCreationThenDestructionAround(psi_func))
        self.wait()
        self.play(FadeOut(constraints))
        self.wait()

        # Moreover, each of them determines a unique property
        arrow_group = VGroup(*[
            Arrow(ORIGIN, 1.5 * RIGHT, color=GREY, buff=0.2).move_to(quant)
            for quant in quant_group
        ])
        arrow_group.arrange(DOWN, coor_mask=X_COOR_MASK).next_to(quant_group, RIGHT, buff=0.3)
        prop_group = VGroup(*[
            TexText(prop_name, prop_value)
            for prop_name, prop_value in zip(
                ["能量", "角动量大小", "角动量$z$分量"],
                ["($\\approx -13.6$eV$/n^2$)", "(=$\\sqrt{\\ell (\\ell + 1)} \\hbar$)", "($= m \\hbar$)"],
            )
        ])
        for prop, color, arrow in zip(prop_group, sub_colors, arrow_group):
            prop[0].set_color(color)
            prop[1].set_color(GREY)
            prop.next_to(arrow, RIGHT, buff=0.2)
        for arrow, prop, descrp_name in zip(arrow_group, prop_group, descrp_name_group):
            self.play(
                GrowArrow(arrow),
                Write(prop),
                run_time=2,
            )
            self.wait()
            self.play(Write(descrp_name[0]))
            self.wait()

        # One set of nlm defines an atomic orbital
        group = VGroup(descrp_name_group, quant_group, arrow_group, prop_group)
        self.play(ShowCreationThenDestructionAround(group))
        self.wait()
        self.play(
            FadeOutToPoint(group.copy(), psi_func.get_center()),
            ShowCreationThenDestructionAround(psi_func),
        )
        self.wait()

        # Examples of atomic orbitals
        psi_100 = Tex(
            "\\Psi", "_{1", " , ", "0", " , ", "0}", "(r, \\theta, \\varphi)", "=",
            "\\sqrt{1 \\over (a_0^{*})^3} \\cdot {1 \\over \\sqrt{\\pi}} \\mathrm{e}^{-\\sigma}",
        )
        psi_31m1 = Tex(
            "\\Psi", "_{3", " , ", "1", " , ", "-1}", "(r, \\theta, \\varphi)", "=",
            "\\sqrt{1 \\over (a_0^{*})^3} \\cdot {1 \\over 18 \\sqrt{\\pi}} \\left( 4\\sigma-2\\sigma^2 \\right) \\mathrm{e}^{-{\\sigma \\over 3}} \\sin \\theta \\cdot \\mathrm{e}^{-\\mathrm{i}\\varphi}",
        )
        psi_422 = Tex(
            "\\Psi", "_{4", " , ", "2", " , ", "2}", "(r, \\theta, \\varphi)", "=",
            "\\sqrt{1 \\over (a_0^{*})^3} \\cdot {1 \\over 32 \\sqrt{6 \\pi}} \\left( 6\\sigma^2-2\\sigma^3 \\right) \\mathrm{e}^{-{\\sigma \\over 4}} \\sin^2\\theta \\cdot \\mathrm{e}^{2\\mathrm{i}\\varphi}",
        )
        psi_group = VGroup(psi_100, psi_31m1, psi_422)
        psi_group.scale(0.8)
        psi_group.arrange(
            DOWN, aligned_edge=LEFT,
            index_of_submobject_to_align=-2,
            buff=1.5,
        )
        psi_group.center()
        for psi in psi_group:
            for mob, color in zip(psi[1:6:2], sub_colors):
                mob.set_color(color)
        psi_remark = Tex(
            """
            & \\text{约化玻尔半径} a_0^{*}= {4 \\pi \\varepsilon_0 \\hbar^2 \\over \\mu e^2}
            \\\\
            & \\text{参数} \\sigma = {r \\over a_0^{*}}
            """,
            color=GREY,
        )
        psi_remark_rect = SurroundingRectangle(
            psi_remark, color=psi_remark.get_color(),
            stroke_width=1, buff=0.3,
        )
        psi_remark_group = VGroup(psi_remark, psi_remark_rect)
        psi_remark_group.scale(0.5)
        psi_remark_group.next_to(psi_100, RIGHT)
        psi_remark_group.next_to(psi_422.get_right(), LEFT, buff=0, coor_mask=X_COOR_MASK)
        self.play(
            FadeOut(group),
            FadeTransform(psi_func, psi_100[:-2]),
        )
        self.wait()
        self.play(
            Write(psi_100[-2:]),
            FadeIn(psi_remark_group),
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                FadeTransform(source_mob, target_mob)
                for source_mob, target_mob in zip(psi_100.copy(), psi_31m1)
            ])
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                FadeTransform(source_mob, target_mob)
                for source_mob, target_mob in zip(psi_31m1.copy(), psi_422)
            ])
        )
        self.wait()

        # A simpler (and more familiar) notation
        covered_part = VGroup(*[mob[-2:] for mob in psi_group], psi_remark_group)
        cover_rect = Rectangle(
            width=covered_part.get_width(), height=covered_part.get_height(),
            color=BLACK, fill_opacity=0.98, stroke_width=0,
        )
        cover_rect.scale(1.01).move_to(covered_part)
        self.play(FadeIn(cover_rect))
        self.wait()

        orbital_names = VGroup(
            TexText("1", "s", "", "轨道"),
            TexText("3", "p", "$_{-1}$", "轨道"),
            TexText("4", "d", "$_{2}$", "轨道"),
        )
        for name, psi in zip(orbital_names, psi_group):
            name.next_to(psi[-1].get_left(), RIGHT, buff=0.4)
            for part, color in zip(name[:-1], sub_colors):
                part.set_color(color)
        subshell_names = VGroup(*[
            TexText(f"{number}", " $\\rightarrow$ ", f"{alphabet}", f"{extra}")
            for number, alphabet, extra in zip(
                range(6),
                "spdfgh",
                ["harp (锐线系)", "rincipal (主线系)", "iffuse (漫线系)", "undamental (基线系)", "", ""]
            )] + [Tex("\\cdots")]
        )
        for name in subshell_names[:-1]:
            for mob, color in zip(name, [GREEN, WHITE, GREEN, GREY]):
                mob.set_color(color)
        for i, name in enumerate(subshell_names):
            if i > 0:
                name.next_to(subshell_names[0].get_left(), RIGHT, buff=0)
            name.shift(i * 0.75 * DOWN)
        subshell_names.scale(0.8)
        subshell_names.center().to_edge(RIGHT, buff=0.5)

        # n - shell, l - subshell, m - remains the same
        def get_writing_animation(mobject, ind_or_slice):
            return AnimationGroup(*[
                Write(component[ind_or_slice])
                for component in mobject
            ],
                lag_ratio=0.05,
                run_time=2,
            )

        self.play(get_writing_animation(orbital_names, 0))
        self.wait()
        self.play(
            get_writing_animation(orbital_names, 1),
            get_writing_animation(subshell_names[:-1], slice(3)),
            Write(subshell_names[-1], run_time=3),
        )
        self.wait()
        self.play(get_writing_animation(subshell_names[:4], -1))
        self.wait()
        self.play(get_writing_animation(orbital_names, slice(2, -1)))
        self.wait()
        for orbital_name in orbital_names:
            self.play(Write(orbital_name[-1]))
            self.wait()
        self.play(
            AnimationGroup(*[
                Indicate(mob[:-2])
                for mob in psi_group

            ],
                lag_ratio=0.05,
                run_time=2,
            )
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                Indicate(mob)
                for mob in orbital_names

            ],
                lag_ratio=0.05,
                run_time=2,
            )
        )
        self.wait()

        # But what does it mean - end of the scene
        title_arrow = Arrow(ORIGIN, RIGHT, color=YELLOW)
        phys_meaning = TexText("物理含义?", color=YELLOW)
        title_arrow.next_to(title_wavefunc, RIGHT)
        phys_meaning.next_to(title_arrow, RIGHT)
        self.play(
            FadeOut(VGroup(cover_rect, orbital_names, subshell_names))
        )
        self.wait()
        self.play(
            GrowArrow(title_arrow),
            Write(phys_meaning),
        )
        self.wait()


class InterpretationAndVisualization(Scene):
    def construct(self):
        # Setup the scene
        title_wavefunc = TexText("氢原子轨道波函数", color=YELLOW)
        title_wavefunc.to_edge(UP)
        title_arrow = Arrow(ORIGIN, RIGHT, color=YELLOW)
        phys_meaning = TexText("物理含义?", color=YELLOW)
        title_arrow.next_to(title_wavefunc, RIGHT)
        phys_meaning.next_to(title_arrow, RIGHT)
        old_title_group = VGroup(title_wavefunc, title_arrow, phys_meaning)
        psi_100 = Tex(
            "\\Psi", "_{1", " , ", "0", " , ", "0}", "(r, \\theta, \\varphi)", "=",
            "\\sqrt{1 \\over (a_0^{*})^3} \\cdot {1 \\over \\sqrt{\\pi}} \\mathrm{e}^{-\\sigma}",
        )
        psi_31m1 = Tex(
            "\\Psi", "_{3", " , ", "1", " , ", "-1}", "(r, \\theta, \\varphi)", "=",
            "\\sqrt{1 \\over (a_0^{*})^3} \\cdot {1 \\over 18 \\sqrt{\\pi}} \\left( 4\\sigma-2\\sigma^2 \\right) \\mathrm{e}^{-{\\sigma \\over 3}} \\sin \\theta \\cdot \\mathrm{e}^{-\\mathrm{i}\\varphi}",
        )
        psi_422 = Tex(
            "\\Psi", "_{4", " , ", "2", " , ", "2}", "(r, \\theta, \\varphi)", "=",
            "\\sqrt{1 \\over (a_0^{*})^3} \\cdot {1 \\over 32 \\sqrt{6 \\pi}} \\left( 6\\sigma^2-2\\sigma^3 \\right) \\mathrm{e}^{-{\\sigma \\over 4}} \\sin^2\\theta \\cdot \\mathrm{e}^{2\\mathrm{i}\\varphi}",
        )
        psi_group = VGroup(psi_100, psi_31m1, psi_422)
        psi_group.scale(0.8)
        psi_group.arrange(
            DOWN, aligned_edge=LEFT,
            index_of_submobject_to_align=-2,
            buff=1.5,
        )
        psi_group.center()
        sub_colors = (RED, GREEN, BLUE)
        for psi in psi_group:
            psi.set_color(WHITE)
            for mob, color in zip(psi[1:6:2], sub_colors):
                mob.set_color(color)
        psi_remark = Tex(
            """
            & \\text{约化玻尔半径} a_0^{*}= {4 \\pi \\varepsilon_0 \\hbar^2 \\over \\mu e^2}
            \\\\
            & \\text{参数} \\sigma = {r \\over a_0^{*}}
            """,
            color=GREY,
        )
        psi_remark_rect = SurroundingRectangle(
            psi_remark, color=psi_remark.get_color(),
            stroke_width=1, buff=0.3,
        )
        psi_remark_group = VGroup(psi_remark, psi_remark_rect)
        psi_remark_group.scale(0.5)
        psi_remark_group.next_to(psi_100, RIGHT)
        psi_remark_group.next_to(psi_422.get_right(), LEFT, buff=0, coor_mask=X_COOR_MASK)

        self.add(old_title_group, psi_group, psi_remark_group)
        self.wait()

        # A common way to interpret the wavefunction - Probability
        title_form1 = TexText("波函数", "—", "概率")
        title_form2 = TexText("波函数的模方", "—", "概率")
        title_form3 = TexText("波函数的模方", "—", "概率密度")
        title_form4 = TexText("波函数的模方", "=", "概率密度")
        for title in (title_form1, title_form2, title_form3, title_form4):
            title.set_color(YELLOW)
            title.arrange(RIGHT)
        for title in (title_form1, title_form4):
            title.to_edge(UP)
        for title in (title_form2, title_form3):
            title.next_to(title_form1, DOWN, index_of_submobject_to_align=1, buff=0)
        self.play(
            AnimationGroup(*[
                FadeTransform(source_mob, target_mob)
                for source_mob, target_mob in zip(old_title_group, title_form1)
            ])
        )
        self.wait()

        # The quantity we care about is "abs(phi)^2", a real-valued function
        psi_100_sqr = Tex(
            "\\big| \\Psi", "_{1", " , ", "0", " , ", "0}", "(r, \\theta, \\varphi) \\big|^2", "=",
            "{1 \\over \\pi (a_0^{*})^3} \\cdot \\mathrm{e}^{-2\\sigma}",
        )
        psi_31m1_sqr = Tex(
            "\\big| \\Psi", "_{3", " , ", "1", " , ", "-1}", "(r, \\theta, \\varphi) \\big|^2", "=",
            "{1 \\over {324\\pi} (a_0^{*})^3} \\cdot \\left( 4\\sigma-2\\sigma^2 \\right)^2 \\mathrm{e}^{-{2\\sigma \\over 3}} \\sin^2 \\theta",
        )
        psi_422_sqr = Tex(
            "\\big| \\Psi", "_{4", " , ", "2", " , ", "2}", "(r, \\theta, \\varphi) \\big|^2", "=",
            "{1 \\over {6144 \\pi} (a_0^{*})^3} \\cdot \\left( 6\\sigma^2-2\\sigma^3 \\right)^2 \\mathrm{e}^{-{\\sigma \\over 2}} \\sin^4\\theta",
        )
        psi_sqr_group = VGroup(psi_100_sqr, psi_31m1_sqr, psi_422_sqr)
        psi_sqr_group.scale(0.8)
        for psi, psi_sqr in zip(psi_group, psi_sqr_group):
            psi_sqr.set_color(WHITE)
            shift_vec = psi[-2].get_center() - psi_sqr[-2].get_center()
            psi_sqr.shift(shift_vec)
            for old_mob, new_mob in zip(psi, psi_sqr):
                new_mob.match_color(old_mob)
        alt_psi_100_sqr = Tex(
            "\\big| \\Psi", "_{1", " , ", "0", " , ", "0}", "(r, \\theta, \\varphi) \\big|^2", "=",
            "{1 \\over \\pi (a_0^{*})^3} \\cdot \\mathrm{e}^{-2r / a_0^{*}}",
        )
        shift_vec = psi_100_sqr[-2].get_center() - alt_psi_100_sqr[-2].get_center()
        alt_psi_100_sqr.shift(shift_vec)

        complex_parts = VGroup(*[
            mob[-1][-4:]
            for mob in (psi_31m1, psi_422)
        ])
        self.play(complex_parts.animate.set_color(PINK))
        self.wait()
        self.play(
            AnimationGroup(*[
                AnimationGroup(*[
                    FadeTransform(old_mob, new_mob)
                    for old_mob, new_mob in zip(psi, psi_sqr)
                ], lag_ratio=0.05)
                for psi, psi_sqr in zip(psi_group, psi_sqr_group)
            ], lag_ratio=0.05, run_time=3),
            AnimationGroup(*[
                FadeTransform(source_mob, target_mob)
                for source_mob, target_mob in zip(title_form1, title_form2)
            ], run_time=2)
        )
        self.wait()

        # It represents how likely we can find an electron around a position
        # i.e. the probability density
        covered_part = VGroup(*[mob[-2:] for mob in psi_sqr_group], psi_remark_group)
        cover_rect = Rectangle(
            width=covered_part.get_width(), height=covered_part.get_height(),
            color=BLACK, fill_opacity=0.98, stroke_width=0,
        )
        cover_rect.scale(1.01).move_to(covered_part)
        prob_meaning = TexText("电子", "有多大的可能性出现", "\\\\ 在$(r, \\theta, \\varphi)$的位置附近")
        prob_meaning.scale(0.8)
        prob_meaning.move_to(cover_rect)
        prob_meaning.next_to(psi_31m1_sqr[:-2], RIGHT, coor_mask=Y_COOR_MASK)
        prob_meaning_arrows = VGroup(*[
            Arrow(prob_meaning.get_left(), psi_sqr[-3].get_right())
            for psi_sqr in psi_sqr_group
        ])
        for mob in (prob_meaning[1], prob_meaning_arrows):
            mob.set_color(MAROON_B)
        self.play(
            FadeIn(cover_rect, run_time=1),
            Write(prob_meaning, run_time=2),
            AnimationGroup(*[
                GrowArrow(arrow)
                for arrow in prob_meaning_arrows
            ], lag_ratio=0.1, run_time=3)
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                FadeTransform(source_mob, target_mob)
                for source_mob, target_mob in zip(title_form2, title_form3)
            ])
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                FadeTransform(source_mob, target_mob)
                for source_mob, target_mob in zip(title_form3, title_form4)
            ]),
            FadeOut(VGroup(cover_rect, prob_meaning, prob_meaning_arrows)),
            run_time=2,
        )
        self.wait()

        # Since it's a random variable, we can use sampling to learn about it
        _, samp_and_stats, _ = sas_group = TexText("(", "采样+统计", ")", color=YELLOW)
        sas_group.next_to(title_form4, RIGHT, buff=0.4)
        self.play(Write(sas_group))
        self.wait()
        self.play(FadeOut(psi_sqr_group[-2:]))
        self.wait()

        orbital_1s_text = TexText("1s轨道", color=YELLOW)
        samp_and_stats.generate_target()
        samp_and_stats.target.next_to(orbital_1s_text, RIGHT, buff=0.05)
        prob_density_text = title_form4[-1]
        prob_density_text.generate_target()
        prob_density_text.target.set_color(WHITE)
        prob_density_text.target.next_to(orbital_1s_text, DOWN, aligned_edge=LEFT)
        prob_density_formula = psi_100_sqr[:-2]
        prob_density_formula.save_state()
        prob_density_formula.generate_target()
        prob_density_formula.target.next_to(prob_density_text.target, RIGHT)
        sample_text = TexText("采样点数:", color=ORANGE)
        sample_text.next_to(prob_density_text.target, DOWN, buff=1, aligned_edge=LEFT)
        sample_number = Integer(0, color=ORANGE, group_with_commas=False)
        sample_number.next_to(sample_text, RIGHT)
        shrink_group = VGroup(
            orbital_1s_text,
            samp_and_stats.target,
            prob_density_text.target,
            prob_density_formula.target,
            sample_text,
            sample_number,
        )
        shrink_group.scale(0.75).to_corner(UL)
        self.play(
            AnimationGroup(
                Write(orbital_1s_text),
                MoveToTarget(samp_and_stats),
                FadeOut(sas_group[::2]),
                MoveToTarget(prob_density_text),
                FadeOut(title_form4[:-1]),
                MoveToTarget(prob_density_formula),
                FadeOut(psi_100_sqr[-2:]),
                FadeOut(psi_remark_group),
                FadeIn(sample_text),
                FadeIn(sample_number),
                lag_ratio=0.05,
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            ChangeDecimalToValue(sample_number, 50000, rate_func=linear),
            run_time=40,
        )
        self.wait()

        # The result is "Electron Cloud"
        electron_cloud = TexText("电子云", color=YELLOW)
        electron_cloud.match_height(samp_and_stats)
        electron_cloud.next_to(samp_and_stats.get_left(), RIGHT, buff=0)
        self.play(FadeTransform(samp_and_stats, electron_cloud))
        self.wait()

        # We can also draw the contour surface of probability density

        for source_mob, target_mob in zip(psi_100_sqr, alt_psi_100_sqr):
            target_mob.match_color(source_mob)

        fade_group = VGroup(electron_cloud, prob_density_text, sample_text, sample_number)
        self.play(
            FadeOut(fade_group),
            prob_density_formula.animate.restore(),
            FadeIn(alt_psi_100_sqr[-2:]),
        )
        self.wait()

        contour_color = "#E1BD6C"
        contour_text = TexText("概率密度等值面", color=YELLOW)
        contour_text.match_height(electron_cloud)
        contour_text.next_to(electron_cloud.get_left(), RIGHT, buff=0)
        set_value_text = TexText("=定值", color=contour_color)
        set_value_text.next_to(alt_psi_100_sqr, RIGHT)
        sur_rect = SurroundingRectangle(
            VGroup(alt_psi_100_sqr[-1], set_value_text),
            buff=0.2,
        )

        all_points_text = TexText("找到所有满足条件的 \\\\ ", "坐标$(r, \\theta, \\varphi)$")
        all_points_text[-1].set_color(contour_color)
        all_points_text.scale(0.75)
        all_points_text.next_to(sur_rect, RIGHT)
        self.play(Write(contour_text))
        self.wait()
        self.play(Write(set_value_text))
        self.wait()
        self.play(ShowCreation(sur_rect))
        self.wait()
        self.play(FadeIn(all_points_text, shift=RIGHT))
        self.wait()

        # It doesn't matter how you choose the isovalue, 1s is always spherical
        def get_radius(prob_dens):
            return -0.5 * np.log(prob_dens)

        def get_prob_dens(radius):
            return np.exp(-2 * radius)

        isovalue_text = TexText("选取的定值:")
        isovalue_number = DecimalNumber(
            0.04,
            num_decimal_places=4, group_with_commas=False,
        )
        radius_number = DecimalNumber(get_radius(isovalue_number.get_value()))
        radius_number.shift(10 * RIGHT + 5 * DOWN)
        isovalue_number.next_to(isovalue_text, RIGHT, aligned_edge=DOWN)
        isovalue_number.add_updater(
            lambda mob: mob.set_value(get_prob_dens(radius_number.get_value()))
        )
        unit_aligner = Tex("{(a_{0}^{*})}")
        unit_aligner.next_to(isovalue_number, RIGHT)
        isovalue_unit = Tex("{(a_{0}^{*})}^{-3}")
        isovalue_unit.shift(unit_aligner[0][0].get_left() - isovalue_unit[0][0].get_left())
        isovalue_group = VGroup(isovalue_text, isovalue_number, isovalue_unit)
        isovalue_group.scale(0.75)
        isovalue_group.next_to(all_points_text, DOWN, buff=1.5, aligned_edge=RIGHT)
        isovalue_group.set_color(contour_color)
        isovalue_unit.add_updater(
            lambda mob: mob.next_to(
                isovalue_number, RIGHT,
                buff=0.1, coor_mask=X_COOR_MASK,
            )
        )
        self.play(Write(isovalue_group))
        self.wait()

        for value in (0.1, 0.04):
            self.play(
                ChangeDecimalToValue(radius_number, get_radius(value)),
                rate_func=linear,
                run_time=5,
            )
            self.wait()

        # That's the second way of visualizing an atomic orbital - end of the scene
        contour_real_name = TexText("电子云轮廓", color=YELLOW)
        contour_real_name.match_height(contour_text)
        contour_real_name.next_to(contour_text.get_left(), RIGHT, buff=0)
        self.play(FadeTransform(contour_text, contour_real_name))
        self.wait()


class ExamplesOfPhysicsTypeOrbitals(Scene):
    def construct(self):
        # Setup the scene
        orbital_names = [
            "1s", "2s", "3s",
            "2p$_{0}$", "2p$_{\\pm 1}$",
            "3d$_{0}$", "3d$_{\\pm 1}$", "3d$_{\\pm 2}$",
        ]
        titles = VGroup(*[
            TexText(orbital, "轨道", color=YELLOW)
            for orbital in orbital_names
        ])
        for title in titles:
            title.scale(1.2)
            title.center().to_edge(UP)
        dot_method, contour_method = visual_methods = VGroup(*[
            TexText(method, color=MAROON_B)
            for method in ("电子云(散点)", "电子云轮廓(等值面)")
        ])
        for k, method in enumerate(visual_methods):
            method.scale(0.8)
            method.center()
            method.shift(3 * (LEFT if k % 2 == 0 else RIGHT))
        visual_methods.next_to(titles[0], DOWN, buff=0.5)
        self.play(GrowFromCenter(titles[0][-1]))
        self.wait()
        self.play(
            AnimationGroup(*[
                Write(method)
                for method in visual_methods
            ], lag_ratio=0.1),
        )
        self.add(visual_methods)
        self.wait()

        # Showing a bunch of examples
        orbitals = ("1s", "2s", "3s", "2p0", "2p1", "3d0", "3d1", "3d2")
        cloud_images = Group(*[
            ImageMobject(f"dots-{orbital}.png")
            for orbital in orbitals
        ])
        contour_images = Group(*[
            ImageMobject(f"contour-{orbital}.png")
            for orbital in orbitals
        ])
        for image in cloud_images:
            image.set_height(5)
            image.next_to(dot_method, DOWN)
        for image in contour_images:
            image.set_height(5)
            image.next_to(contour_method, DOWN)

        # Flashing through all listed orbitals
        self.play(
            Write(titles[0][0]),
            FadeIn(cloud_images[0]),
            FadeIn(contour_images[0]),
        )
        self.wait()
        for ind in range(1, len(titles)):
            self.play(
                AnimationGroup(*[
                    FadeTransform(group[ind - 1], group[ind])
                    for group in (titles, cloud_images, contour_images)
                ]),
                run_time=1,
            )
            self.wait()

        # 3d orbitals with phase
        title_3d_orbital = TexText("3d轨道", color=YELLOW)
        title_3d_orbital.scale(1.2)
        title_3d_orbital.center().to_edge(UP)
        psi_3d = Tex(
            "\\Psi_{3\\mathrm{d}}(r, \\theta, \\varphi) =",
            "\\left[ \\text{关于$r$和$\\theta$的实函数} \\right] \\cdot ",
            "\\mathrm{e}^{\\mathrm{i} m \\varphi}",
        )
        psi_3d[1].set_color(GREY)
        psi_3d[-1].set_color(PINK)
        psi_3d.center().next_to(title_3d_orbital, DOWN, buff=0.5)

        self.play(
            FadeTransform(titles[-1], title_3d_orbital),
            FadeIn(psi_3d),
            FadeOut(
                Group(*[
                    group[-1]
                    for group in (cloud_images, contour_images)
                ])
            ),
            FadeOut(visual_methods),
        )
        self.wait()

        sur_rect = SurroundingRectangle(psi_3d[-1], color=PINK)
        phase = TexText("颜色", " $\\rightarrow$ ", "相位")
        phase.scale(0.8)
        phase[::2].set_color(PINK)
        shift_vec = sur_rect.get_bottom() - phase[-1].get_top() + 0.2 * DOWN
        phase.shift(shift_vec)
        self.play(
            ShowCreationThenDestruction(sur_rect),
            Write(phase),
        )
        self.wait()

        # Images comparison
        image_3d_physics = ImageMobject("orbital-3d-physics.png")
        image_3d_chemistry = ImageMobject("orbital-3d-collection.png")
        for image in (image_3d_physics, image_3d_chemistry):
            image.set_width(FRAME_WIDTH)
        image_3d_physics.next_to(phase, DOWN, buff=0, coor_mask=Y_COOR_MASK)
        image_3d_chemistry.next_to(BOTTOM, UP, buff=0.5, coor_mask=Y_COOR_MASK)
        self.play(FadeIn(image_3d_physics, shift=UP))
        self.wait()

        self.play(
            FadeOut(VGroup(psi_3d, phase)),
            image_3d_physics.animate.next_to(title_3d_orbital, DOWN, buff=-0.2),
            FadeIn(image_3d_chemistry, shift=UP),
            run_time=2,
        )
        self.wait()

        select_rect = Rectangle(width=11, height=2.5, color=YELLOW)
        select_rect.flip()
        select_rect.move_to(image_3d_physics)
        self.play(ShowCreation(select_rect))
        self.wait()
        phys_perspective = TexText("``物理'' \\\\ 视角", color=YELLOW)
        phys_perspective.next_to(select_rect, LEFT, buff=-0.4)
        self.play(
            AnimationGroup(*[
                mob.animate.shift(RIGHT)
                for mob in (image_3d_physics, image_3d_chemistry, select_rect)
            ]),
            Write(phys_perspective),
        )
        self.wait()

        # Upsides and downsides of physics-type orbitals
        cover_rect = FullScreenFadeRectangle(fill_opacity=0.95)
        cover_rect.next_to(image_3d_physics, DOWN, buff=0, coor_mask=Y_COOR_MASK)
        up_and_down_phys = VGroup(
            TexText("\\ding{51} 量子数物理意义明确", color=GREEN),
            TexText("\\ding{55} 复函数理解起来不直观", color=RED),
        )
        up_and_down_phys.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        up_and_down_phys.center().move_to(image_3d_chemistry)
        up_and_down_phys.next_to(image_3d_physics, DOWN, coor_mask=X_COOR_MASK)
        self.play(
            FadeIn(cover_rect),
            Write(up_and_down_phys[0]),
        )
        self.wait()
        self.play(Write(up_and_down_phys[1]))
        self.wait()

        # Take a look at the wavefunction again
        psi_pm_form = Tex(
            "\\Psi_{n, \\ell, ", "m}", " (r, \\theta, \\varphi) = ",
            "\\left[ \\text{由$n$, $\\ell$和$|m|$确定的实函数} \\right] ", "\\cdot ",
            "\\mathrm{e}^{\\mathrm{i} m \\varphi}",
        )
        psi_mm_form = Tex(
            "\\Psi_{n, \\ell, ", "-m}", " (r, \\theta, \\varphi) = ",
            "\\left[ \\text{由$n$, $\\ell$和$|m|$确定的实函数} \\right] ", "\\cdot ",
            "\\mathrm{e}^{-\\mathrm{i} m \\varphi}",
        )
        for form in (psi_pm_form, psi_mm_form):
            for part, color in zip(form, [WHITE, PINK, WHITE, GREY, GREY, PINK]):
                part.set_color(color)
        psi_pm_form.move_to(up_and_down_phys, coor_mask=Y_COOR_MASK)
        self.play(
            FadeOut(up_and_down_phys),
            Write(psi_pm_form),
        )
        self.wait()
        self.play(Indicate(psi_pm_form[-1]))
        self.wait()

        # We can use psi_(n,l,m) and psi_(n,l,-m) to form real-valued orbitals
        psi_pm_form.generate_target()
        for mob in (psi_pm_form.target, psi_mm_form):
            mob.scale(0.7)
        psi_form_group = VGroup(psi_pm_form.target, psi_mm_form)
        psi_form_group.arrange(
            DOWN, buff=0.8,
            aligned_edge=RIGHT, index_of_submobject_to_align=-2)
        psi_form_group.move_to(psi_pm_form).to_edge(LEFT, buff=0.5)
        self.play(
            MoveToTarget(psi_pm_form),
            FadeTransform(psi_pm_form.copy(), psi_mm_form),
        )
        self.wait()
        self.play(*[
            Indicate(form[-3])
            for form in (psi_pm_form, psi_mm_form)
        ])
        self.wait()

        psi_form_group = VGroup(psi_pm_form, psi_mm_form)
        brace = Brace(psi_form_group, RIGHT)
        euler_formula = Tex("\\mathrm{e}^{\\mathrm{i} t} = \\cos t + \\mathrm{i} \\sin t", color=PINK)
        result_cos = Tex("\\text{相加后乘以} {1 \\over 2}", " \\rightarrow", "\\cos {m\\varphi}", color=GREEN)
        result_sin = Tex("\\text{相减后乘以} {1 \\over 2\\mathrm{i}}", " \\rightarrow", "\\sin {m\\varphi}", color=RED)
        for formula in (euler_formula, result_cos, result_sin):
            formula.scale(0.8)
            formula.next_to(brace, RIGHT)
        self.play(FadeIn(euler_formula))
        self.wait()
        self.play(
            GrowFromCenter(brace),
            FadeTransform(euler_formula, result_cos)
        )
        self.wait()
        self.play(FadeTransform(result_cos, result_sin))
        self.wait()
        result_cos.generate_target()
        result_sin.generate_target()
        result_cos.fade(1)
        result_group = VGroup(result_cos.target, result_sin.target)
        result_group.arrange(DOWN)
        result_group.next_to(euler_formula.get_left(), RIGHT, buff=0)
        self.play(*[
            MoveToTarget(mob)
            for mob in (result_cos, result_sin)
        ])
        self.wait()

        # And that's the chemistry-type orbitals
        chem_perspective = TexText("``化学'' \\\\ 视角", color=YELLOW)
        select_rect.generate_target()
        select_rect.target.move_to(image_3d_chemistry)
        chem_perspective.next_to(select_rect.target, LEFT, buff=1 - 0.4)
        self.play(
            FadeOut(
                VGroup(
                    cover_rect,
                    psi_pm_form, psi_mm_form,
                    brace, result_cos, result_sin,
                ),
                run_time=1,
            ),
            MoveToTarget(select_rect, run_time=2),
            Write(chem_perspective, run_time=2),
        )
        self.wait()


class ShowingChemistryTypeOrbitals(Scene):
    def get_color_tags(self, color_plus, color_minus):
        text_plus = Tex("\\text{: } ", "\\Psi>0")
        text_minus = Tex("\\text{: } ", "\\Psi<0")
        square_plus = Square(
            side_length=text_plus.get_height(),
            color=color_plus, fill_opacity=1
        )
        square_minus = square_plus.deepcopy()
        group_plus = VGroup(square_plus, text_plus)
        group_minus = VGroup(square_minus, text_minus)
        groups = VGroup(group_plus, group_minus)
        for group, color in zip(groups, [color_plus, color_minus]):
            group.arrange(RIGHT)
            group[0].set_color(color)
            group[1][-1].set_color(color)
        groups.arrange(DOWN, aligned_edge=LEFT)
        groups.scale(0.6)
        groups.to_corner(DL).shift(0.5 * UP)
        return groups

    def construct(self):
        # Linear combination of two physics-type orbitals
        new_orbital_vector = Matrix(
            [["\\Psi_{n,\\ell,?}", ],
             ["\\Psi_{n,\\ell,?}", ]],
        )
        equal_sign = Tex("=")
        outer_coeff = Tex("{1 \\over \\sqrt{2}}")
        coeff_matrix = Matrix(
            [["1", "1"],
             ["-\\mathrm{i}", "\\mathrm{i}"]],
            h_buff=1, v_buff=0.8
        )
        old_orbital_vector = Matrix(
            [["\\Psi_{n,\\ell,m}", ],
             ["\\Psi_{n,\\ell,-m}", ]],
            element_alignment_corner=LEFT,
        )
        for vector in (new_orbital_vector, old_orbital_vector):
            for row in vector.get_rows():
                element = row[0][0]
                element[1].set_color(RED)
                element[3].set_color(GREEN)
                element[5:].set_color(BLUE)
        coeff_group = VGroup(outer_coeff, coeff_matrix)
        coeff_group.set_color(MAROON_B)
        lin_comb = VGroup(
            new_orbital_vector, equal_sign, outer_coeff,
            coeff_matrix, old_orbital_vector
        )
        lin_comb.scale(1.25)
        lin_comb.arrange(RIGHT)
        coeff_group_remark = TexText("(经过归一化的系数矩阵)", color=MAROON_B)
        coeff_group_remark.match_width(coeff_group).next_to(coeff_group, DOWN)
        new_orbital_remark = TexText("组合后的轨道", color=YELLOW)
        new_orbital_remark.next_to(new_orbital_vector, UP)
        setup_group = VGroup(lin_comb, coeff_group_remark, new_orbital_remark)
        self.play(FadeIn(setup_group))
        self.wait()
        self.play(Indicate(coeff_group))
        self.wait()

        # We care about its orientation
        arrow = Arrow(ORIGIN, DOWN, color=BLUE)
        orientation_text = TexText("空间朝向", color=BLUE)
        arrow.next_to(new_orbital_vector.get_rows()[0][0][0][5], UP)
        orientation_text.next_to(arrow, UP)
        new_orbital_remark.generate_target()
        new_orbital_remark.target.next_to(orientation_text, UP)
        self.play(
            MoveToTarget(new_orbital_remark),
            GrowArrow(arrow),
            Write(orientation_text),
        )
        self.wait()

        # p-orbitals
        title_colors = [RED, GREEN, BLUE, YELLOW]
        tag_colors_2p = ["#6FADE7", "#E7C36F"]
        images_2p = Group(*[
            ImageMobject(f"chem-{orbital}.png")
            for orbital in ("2px", "2py", "2pz")
        ])
        images_2p.arrange(RIGHT, buff=0.5)
        images_2p.set_width(13)
        titles_2p = VGroup(*[
            TexText("2", "p", f"$_{orientation}$", "轨道")
            for orientation in list("xyz")
        ])
        for title, image in zip(titles_2p, images_2p):
            title.next_to(image, UP)
            for mob, color in zip(title, title_colors):
                mob.set_color(color)
        tags_2p = self.get_color_tags(*tag_colors_2p)
        self.play(
            FadeIn(
                Group(
                    images_2p,
                    VGroup(*[
                        VGroup(title[0], title[1], title[3])
                        for title in titles_2p
                    ]),
                ),
                shift=UP,
            ),
            FadeIn(tags_2p, shift=UP),
            FadeOut(
                VGroup(setup_group, arrow, orientation_text),
                shift=UP,
            ),
        )
        self.wait()

        self.play(
            AnimationGroup(*[
                Write(mob)
                for mob in [title[2] for title in titles_2p]
            ], lag_ratio=0.1)
        )
        self.wait()

        # d-orbitals
        images_3d_upper = Group(*[
            ImageMobject(f"chem-{orbital}.png")
            for orbital in ("3dx2-y2", "3dz2", "3dxy")
        ])
        images_3d_lower = Group(*[
            ImageMobject(f"chem-{orbital}.png")
            for orbital in ("3dxz", "3dyz")
        ])
        images_3d = Group(*(
            images_3d_upper.submobjects + images_3d_lower.submobjects
        ))
        for parts, direction in zip([images_3d_upper, images_3d_lower], [UP, DOWN]):
            parts.scale(0.8)
            parts.arrange(RIGHT, buff=0.5)
            parts.center().shift(1.8 * direction)
        titles_3d = VGroup(*[
            TexText("3", "d", f"$_{orientation}$", "轨道")
            for orientation in ["{x^2-y^2}", "{z^2}", "{xy}", "{xz}", "{yz}"]
        ])
        for title, image in zip(titles_3d, images_3d):
            title.scale(0.8)
            title.next_to(image, UP, buff=0)
            for mob, color in zip(title, title_colors):
                mob.set_color(color)
        group_3d = Group(titles_3d, images_3d)
        group_3d.center()
        self.play(
            FadeOut(Group(images_2p, titles_2p), shift=UP),
            FadeIn(images_3d, shift=UP),
            FadeIn(
                VGroup(*[
                    VGroup(title[0], title[1], title[3])
                    for title in titles_3d
                ]),
                shift=UP,
            )
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                Write(titles_3d[ind][2])
                for ind in (0, 3, 4, 2)
            ], lag_ratio=0.2),
            run_time=2,
        )
        self.wait()

        sur_rect = SurroundingRectangle(images_3d[1], buff=-0.1)
        sur_rect.shift(0.2 * RIGHT)
        self.play(ShowCreationThenDestruction(sur_rect))
        self.wait()
        self.play(Write(titles_3d[1][2]))
        self.wait()

        # f-orbitals and g-orbitals
        tag_colors_4f = ["#AD99E7", "#DAE799"]
        tag_colors_5g = ["#50E7C9", "#E786AB"]
        orbital_names = (
            ["4fyz2", "4fz3", "4fxz2"],
            ["4fxyz", "4fz(x2-y2)"],
            ["4fy(3x2-y2)", "4fx(x2-3y2)"],
            ["5gyz3", "5gz4", "5gxz3"],
            ["5gxyz2", "5gz2(x2-y2)"],
            ["5gy3z", "5gx3z"],
            ["5gxy(x2-y2)", "5g(x4+y4)"],
        )
        orientations = (
            ["{yz^2}", "{z^3}", "{xz^2}"],
            ["{xyz}", "{z\\left( x^2-y^2 \\right)}"],
            ["{y\\left( 3x^2-y^2 \\right)}", "{x\\left( x^2-3y^2 \\right)}"],
            ["{z^3 y}", "{z^4}", "{z^3 x}"],
            ["{z^2 xy}", "{z^2 \\left( x^2-y^2 \\right)}"],
            ["{z y^3}", "{z x^3}"],
            ["{xy \\left( x^2-y^2 \\right)}", "{x^4+y^4}"],
        )
        images_4f5g = Group()
        titles_4f5g = VGroup()
        for orbital_set, orientation_set in zip(orbital_names, orientations):
            image_set = Group()
            title_set = VGroup()
            for orbital, orientation in zip(orbital_set, orientation_set):
                image = ImageMobject(f"chem-{orbital}.png")
                image_set.add(image)
                title = TexText(orbital[0], orbital[1], f"$_{orientation}$", "轨道")
                title_set.add(title)
            image_set.arrange(RIGHT, buff=0.5).center()
            for title, image in zip(title_set, image_set):
                title.next_to(image, UP)
                for mob, color in zip(title, title_colors):
                    mob.set_color(color)
            images_4f5g.add(image_set)
            titles_4f5g.add(title_set)

        old_group = group_3d
        old_tags = tags_2p
        for ind, (images, titles) in enumerate(zip(images_4f5g, titles_4f5g)):
            new_group = Group(images, titles)
            switch_anims_list = [
                FadeOut(old_group, shift=UP),
                FadeIn(new_group, shift=UP),
            ]
            old_group = new_group
            if ind == 0 or ind == 3:
                if ind == 0:
                    new_tags = self.get_color_tags(*tag_colors_4f)
                else:
                    new_tags = self.get_color_tags(*tag_colors_5g)
                switch_anims_list.append(FadeTransform(old_tags, new_tags))
                old_tags = new_tags
            self.play(AnimationGroup(*switch_anims_list))
            self.wait(4)

        # We can make orbitals more concentrated by hybridization - end of the scene
        fade_rect = FullScreenFadeRectangle(fill_opacity=0.9)
        more_concentrated_text = TexText("继续聚焦轨道的方向?", color=YELLOW)
        hybridization_text = TexText("轨道", "杂化", color=YELLOW)
        hybridization_text.scale(1.5)
        self.play(
            FadeIn(fade_rect, run_time=1),
            Write(more_concentrated_text, run_time=2),
        )
        self.wait()
        self.play(
            fade_rect.animate.set_fill(opacity=1),
            FadeTransform(more_concentrated_text, hybridization_text),
        )
        self.wait()


class ShowingHybridizationSP(Scene):
    def construct(self):
        # Setup the scene
        hybridization_text = TexText("轨道", "杂化", color=YELLOW)
        hybridization_text.scale(1.5)
        self.add(hybridization_text)

        # Hybridization is actually linear combination
        wavefunc_text = TexText("波函数", color=BLUE)
        combo_text = TexText("线性组合", color=BLUE)
        title = TexText("轨道杂化", "—", "波函数线性组合", color=YELLOW)
        title.arrange(RIGHT)
        title.to_edge(UP)
        wavefunc_text.next_to(hybridization_text[0], DOWN, aligned_edge=RIGHT)
        combo_text.next_to(hybridization_text[1], DOWN, aligned_edge=LEFT)

        self.play(
            hybridization_text[0].animate.set_color(BLUE),
            hybridization_text[1].animate.set_color(GREY),
            FadeTransform(hybridization_text[0].copy(), wavefunc_text),
        )
        self.wait()
        self.play(
            hybridization_text[0].animate.set_color(GREY),
            wavefunc_text.animate.set_color(GREY),
            hybridization_text[1].animate.set_color(BLUE),
            FadeTransform(hybridization_text[1].copy(), combo_text),
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                FadeTransform(middle_part, title_part)
                for middle_part, title_part in zip(
                    (hybridization_text, VGroup(wavefunc_text, combo_text)),
                    title[::2],
                )
            ], lag_ratio=0.1),
            Write(title[1]),
        )
        self.wait()

        # We just saw an example of linear combination
        new_orbital_vector = Matrix(
            [["``化学'' \\\\ 视角 \\\\ 轨道", ], ],
            element_to_mobject=TexText,
        )
        equal_sign = Tex("=")
        outer_coeff = Tex("{1 \\over \\sqrt{2}}")
        coeff_matrix = Matrix(
            [["1", "1"],
             ["-\\mathrm{i}", "\\mathrm{i}"]],
            h_buff=1, v_buff=0.8,
        )
        old_orbital_vector = Matrix(
            [["``物理'' \\\\ 视角 \\\\ 轨道", ], ],
            element_to_mobject=TexText,
        )
        for vector in (new_orbital_vector, old_orbital_vector):
            vector.scale(0.7)
        lin_comb = VGroup(
            new_orbital_vector, equal_sign, outer_coeff,
            coeff_matrix, old_orbital_vector
        )
        lin_comb.scale(1.25)
        lin_comb.arrange(RIGHT)
        lin_comb[2:-1].set_color(MAROON_B)
        complex_text = TexText("复函数", color=BLUE)
        complex_text.next_to(old_orbital_vector, UP, buff=0.5)
        real_text = TexText("实函数", color=BLUE)
        real_text.next_to(new_orbital_vector, UP, buff=0.5)
        self.play(FadeIn(lin_comb))
        self.wait()
        self.play(
            old_orbital_vector.animate.set_color(BLUE),
            Write(complex_text),
        )
        self.wait()
        self.play(
            FadeTransform(complex_text.copy(), real_text),
            new_orbital_vector.animate.set_color(BLUE),
            old_orbital_vector.animate.set_color(GREY),
            complex_text.animate.set_color(GREY),
        )
        self.wait()
        fade_group = VGroup(lin_comb, complex_text, real_text, title)

        # An example of 2s+2p
        image_height = 3
        images = Group(*[
            ImageMobject(f"hybrid-demo-{name}.png", height=image_height)
            for name in ("2s", "2pz", "sp2", "2s", "2pz", "sp1")
        ])
        image_2s_p, image_2p_p, image_sp_p, image_2s_m, image_2p_m, image_sp_m = images
        plus_p, plus_m = VGroup(Tex("+"), Tex("-")).scale(2).set_color(YELLOW)
        plus_equation = Group(image_2s_p, plus_p, image_2p_p, image_sp_p)
        minus_equation = Group(image_2s_m, plus_m, image_2p_m, image_sp_m)
        equations = [plus_equation, minus_equation]
        for ind, equation in enumerate(equations):
            equation.arrange(RIGHT)
            equation[-1].shift(1. * RIGHT)
            equation.center().shift(2 * (UP if ind % 2 == 0 else DOWN))
        texts = VGroup(*[
            TexText(text)
            for text in ("2s轨道", "2p轨道", "sp$_{z-}$杂化轨道", "2s轨道", "2p轨道", "sp$_{z+}$杂化轨道")
        ])
        texts[2::3].set_color(YELLOW)
        text_2s_p, text_2p_p, text_sp_p, text_2s_m, text_2p_m, text_sp_m = texts
        for text, image in zip(texts, images):
            text.scale(0.6)
            text.next_to(image, UP, buff=0)

        center_equation = plus_equation.deepcopy()
        center_texts = texts[:2].deepcopy()
        Group(center_equation, center_texts).shift(2 * DOWN)
        self.play(
            AnimationGroup(*[
                FadeIn(text, shift=UP)
                for text in center_texts
            ], lag_ratio=0.2),
            AnimationGroup(*[
                FadeIn(image, shift=UP)
                for image in center_equation[::2]
            ], lag_ratio=0.2),
            FadeOut(fade_group, shift=UP),
        )
        self.add(center_texts)
        self.wait()

        center_diff_arrow, center_same_arrow = center_addon_arrows = VGroup(*[
            Arrow(ORIGIN, 2.5 * LEFT, color=color)
            for color in [RED, GREEN]
        ])
        center_addon_arrows.arrange(DOWN, buff=1)
        center_addon_arrows.move_to(
            (center_equation[-2].get_center() + center_equation[-1].get_center()) / 2
        )
        center_addon_arrows.shift(0.3 * DL)
        center_diff_text, center_same_text = center_addon_texts = VGroup(*[
            TexText(text, color=color)
            for text, color in zip(["抵消", "叠加"], [RED, GREEN])
        ])
        center_addon_group = VGroup(center_addon_arrows, center_addon_texts)
        for text, arrow, direction in zip(center_addon_texts, center_addon_arrows, [UP, DOWN]):
            text.scale(0.6)
            text.next_to(arrow, direction)
        self.play(
            AnimationGroup(*[
                GrowArrow(arrow)
                for arrow in center_addon_arrows
            ], lag_ratio=0.2),
        )
        self.wait()

        self.play(
            Rotate(center_same_arrow, PI),
            SpinInFromNothing(center_addon_texts[1]),
            Write(center_equation[1]),
        )
        self.wait()
        self.play(
            Rotate(center_diff_arrow, PI),
            SpinInFromNothing(center_addon_texts[0]),
        )
        self.wait()
        self.play(
            FadeIn(center_equation[-1], shift=RIGHT)
        )
        self.wait()

        # 2s+2p works, so does 2s-2p
        minus_addon_group = center_addon_group.copy()
        minus_addon_group.move_to(
            (minus_equation[-2].get_center() + minus_equation[-1].get_center()) / 2
        )
        minus_addon_group.shift(0.3 * DL)
        for mob_set in minus_addon_group:
            mob1, mob2 = mob_set
            mob1_center = np.array(mob1.get_center())
            mob1.move_to(mob2)
            mob2.move_to(mob1_center)
        minus_addon_group.generate_target()
        minus_addon_group.move_to(center_addon_group)
        minus_addon_group.fade(1)

        minus_part = Group(text_2s_m, text_2p_m, minus_equation)
        minus_part.generate_target()
        for source_mob, target_mob in zip(
            minus_part, (center_texts[0], center_texts[1], center_equation),
        ):
            source_mob.move_to(target_mob)
            source_mob.fade(1)

        center_shift_vec = plus_equation.get_center() - center_equation.get_center()
        self.play(
            ReplacementTransform(center_texts, texts[:2]),
            ReplacementTransform(center_equation, plus_equation),
            center_addon_group.animate.shift(center_shift_vec),
            MoveToTarget(minus_part),
            MoveToTarget(minus_addon_group),
            run_time=2,
        )
        self.wait()

        # We call them sp-hybridization orbitals - end of the scene
        text_slices = [
            slice(-4, None, None),
            slice(None, 2, None),
            slice(2, -4, None),
        ]
        for text_slice in text_slices:
            self.play(
                AnimationGroup(*[
                    Write(text[0][text_slice], stroke_color=YELLOW)
                    for text in texts[2::3]
                ])
            )
            self.wait()


class JourneyOfHybridization(Scene):
    def construct(self):
        ref_line = Line(10 * UP, 10 * DOWN, stroke_width=1)
        ref_line.move_to(2 / 3 * LEFT_SIDE + 1 / 3 * RIGHT_SIDE)
        max_width = (ref_line.get_center()[0] - LEFT_SIDE[0]) * 0.8
        # self.add(ref_line)

        # Setup the scene
        orbital_names = (
            "sp",
            "sp$^{3}$",
            "d$^{3}$s",
            "sp$^{3}$d",
            "sp$^{3}$d$^{2}$",
            "sd$^{2}$f$^{3}$",
        )
        orbital_titles = VGroup(*[
            TexText(f"{name}杂化轨道", color=YELLOW)
            for name in orbital_names
        ])
        for title in orbital_titles:
            title.to_corner(UL, buff=0.3)

        sections = VGroup(*[
            TexText(text, color=BLUE)
            for text in (
                "参与杂化的轨道 $\\vec{v}$",
                "组合系数矩阵 $M$",
                "生成的杂化轨道 $( = M \\vec{v})$"
            )
        ])
        y_positions = [2.5, 1, -2.8]
        for section, y in zip(sections, y_positions):
            section.scale(0.75)
            section.to_edge(LEFT, buff=0.5)
            section.shift(y * UP)
        section_dots = VGroup(*[
            Dot(radius=0.05, color=BLUE).move_to(section.get_left() + 0.2 * LEFT)
            for section in sections
        ])

        ao_list = (
            ["s", "p$_{z}$"],
            ["s", "p$_{x}$", "p$_{y}$", "p$_{z}$"],
            ["s", "d$_{xy}$", "d$_{yz}$", "d$_{xz}$"],
            ["s", "p$_{x}$", "p$_{y}$", "p$_{z}$", "d$_{z^2}$"],
            ["s", "p$_{x}$", "p$_{y}$", "p$_{z}$", "d$_{x^2-y^2}$", "d$_{z^2}$"],
            ["s", "f$_{x^3}$", "f$_{y^3}$", "f$_{z^3}$", "d$_{x^2-y^2}$", "d$_{z^2}$"],
        )
        vectors = VGroup(*[
            Tex("\\left[ \\text{" + (", ".join(aos)) + "} \\right] ^{\\mathrm{T}}")
            for aos in ao_list
        ])
        for vector in vectors:
            vector.scale(0.6)
            if vector.get_width() > max_width:
                vector.set_width(max_width)
            vector.next_to(sections[0], DOWN, buff=0.2, aligned_edge=LEFT)
            vector.shift(0.1 * RIGHT)

        matrices = VGroup(
            Tex(
                """
                {1 \\over \\sqrt{2}}
                \\begin{bmatrix}
                1 & 1 \\\\
                1 & -1
                \\end{bmatrix}
                """
            ),
            Tex(
                """
                {1 \\over 2}
                \\begin{bmatrix}
                1 &  1 &  1 &  1 \\\\
                1 &  1 & -1 & -1 \\\\
                1 & -1 &  1 & -1 \\\\
                1 & -1 & -1 &  1 \\\\
                \\end{bmatrix}
                """
            ),
            Tex(
                """
                {1 \\over 2}
                \\begin{bmatrix}
                1 &  1 &  1 &  1 \\\\
                1 &  1 & -1 & -1 \\\\
                1 & -1 &  1 & -1 \\\\
                1 & -1 & -1 &  1 \\\\
                \\end{bmatrix}
                """
            ),
            Tex(
                """
                \\begin{bmatrix}
                \\vspace{0.5em}
                {1 \\over \\sqrt{3}} &  {1 \\over \\sqrt{2}} & -{1 \\over \\sqrt{6}} & 0 & 0 \\\\
                \\vspace{0.5em}
                {1 \\over \\sqrt{3}} & -{1 \\over \\sqrt{2}} & -{1 \\over \\sqrt{6}} & 0 & 0 \\\\
                \\vspace{0.5em}
                {1 \\over \\sqrt{3}} & 0 & {2 \\over \\sqrt{6}} & 0 & 0 \\\\
                \\vspace{0.5em}
                0 & 0 & 0 & {1 \\over \\sqrt{2}} & {1 \\over \\sqrt{2}} \\\\
                0 & 0 & 0 & -{1 \\over \\sqrt{2}} & {1 \\over \\sqrt{2}} \\\\
                \\end{bmatrix}
                """
            ),
            Tex(
                """
                \\begin{bmatrix}
                \\vspace{0.5em}
                {1 \\over \\sqrt{6}} & -{1 \\over \\sqrt{2}} & 0 & 0 & {1 \\over 2} & -{1 \\over \\sqrt{12}} \\\\
                \\vspace{0.5em}
                {1 \\over \\sqrt{6}} & {1 \\over \\sqrt{2}} & 0 & 0 & {1 \\over 2} & -{1 \\over \\sqrt{12}} \\\\
                \\vspace{0.5em}
                {1 \\over \\sqrt{6}} & 0 & {1 \\over \\sqrt{2}} & 0 & -{1 \\over 2} & -{1 \\over \\sqrt{12}} \\\\
                \\vspace{0.5em}
                {1 \\over \\sqrt{6}} & 0 & -{1 \\over \\sqrt{2}} & 0 & -{1 \\over 2} & -{1 \\over \\sqrt{12}} \\\\
                \\vspace{0.5em}
                {1 \\over \\sqrt{6}} & 0 & 0 & {1 \\over 2} & 0 & {1 \\over \\sqrt{3}} \\\\
                {1 \\over \\sqrt{6}} & 0 & 0 & -{1 \\over 2} & 0 & {1 \\over \\sqrt{3}} \\\\
                \\end{bmatrix}
                """
            ),
            Tex(
                """
                \\begin{bmatrix}
                \\vspace{0.5em}
                {1 \\over \\sqrt{6}} & -{1 \\over \\sqrt{2}} & 0 & 0 & {1 \\over 2} & -{1 \\over \\sqrt{12}} \\\\
                \\vspace{0.5em}
                {1 \\over \\sqrt{6}} & {1 \\over \\sqrt{2}} & 0 & 0 & {1 \\over 2} & -{1 \\over \\sqrt{12}} \\\\
                \\vspace{0.5em}
                {1 \\over \\sqrt{6}} & 0 & {1 \\over \\sqrt{2}} & 0 & -{1 \\over 2} & -{1 \\over \\sqrt{12}} \\\\
                \\vspace{0.5em}
                {1 \\over \\sqrt{6}} & 0 & -{1 \\over \\sqrt{2}} & 0 & -{1 \\over 2} & -{1 \\over \\sqrt{12}} \\\\
                \\vspace{0.5em}
                {1 \\over \\sqrt{6}} & 0 & 0 & {1 \\over 2} & 0 & {1 \\over \\sqrt{3}} \\\\
                {1 \\over \\sqrt{6}} & 0 & 0 & -{1 \\over 2} & 0 & {1 \\over \\sqrt{3}} \\\\
                \\end{bmatrix}
                """
            ),
        )
        for matrix in matrices:
            matrix.scale(0.6)
            if matrix.get_width() > max_width:
                matrix.set_width(max_width)
            matrix.next_to(sections[1], DOWN, buff=0.2, aligned_edge=LEFT)
            matrix.shift(0.1 * RIGHT)

        hybrid_groups = [
            VGroup(title, vector, matrix)
            for title, vector, matrix in zip(orbital_titles, vectors, matrices)
        ]
        group_sp, group_sp3, group_d3s, group_sp3d, group_sp3d2, group_sd2f3 = hybrid_groups

        # sp hybridization
        self.add(sections, section_dots)
        self.add(group_sp)
        self.wait()

        # sp3 hybridization, with a glimpse on how to get the coefficients
        self.play(self.get_fade_animation(group_sp, group_sp3))
        self.wait()
        sur_rect = SurroundingRectangle(
            VGroup(sections[:2], group_sp3[-2:]),
            color=RED, fill_opacity=0.2,
        )
        question = TexText("?", color=RED)
        question.scale(3).move_to(sur_rect)
        calc_method = TexText("点群+特征标表", color=GREEN)
        calc_method.scale(0.6)
        calc_method.next_to(sur_rect, DOWN)
        self.play(
            DrawBorderThenFill(sur_rect),
            Write(question),
        )
        self.wait()
        self.play(
            FadeTransform(question, calc_method),
            sur_rect.animate.set_color(GREEN),
        )
        self.wait()
        self.play(
            FadeOut(sur_rect), FadeOut(calc_method),
        )

        # d3s hybridization
        self.play(self.get_fade_animation(group_sp3, group_d3s))
        self.wait()

        # sp3d hybridization, treat it as if it's a combo of sp2 and pd
        self.play(self.get_fade_animation(group_d3s, group_sp3d))
        self.wait()
        color_sp2, color_pd = TEAL, MAROON_B
        vector_part_sp2 = SurroundingRectangle(
            vectors[3][0][1:8],
            stroke_width=0, fill_opacity=0.4, color=color_sp2,
        )
        vector_part_pd = SurroundingRectangle(
            vectors[3][0][9:15],
            stroke_width=0, fill_opacity=0.4, color=color_pd,
        )
        height = max(vector_part_sp2.get_height(), vector_part_pd.get_height())
        for vector_part in (vector_part_sp2, vector_part_pd):
            vector_part.set_height(height, stretch=True)
            vector_part.move_to(vectors[3][0][:-1], coor_mask=Y_COOR_MASK)
        matrix_part_sp2 = vector_part_sp2.copy()
        matrix_part_sp2.set_height(1.6).set_width(2.2, stretch=True)
        matrix_part_sp2.move_to([-5.4, -0.15, 0])
        matrix_part_pd = vector_part_pd.copy()
        matrix_part_pd.set_height(1.1).set_width(1.3, stretch=True)
        matrix_part_pd.move_to([-3.4, -1.45, 0])
        rect_parts = VGroup(vector_part_sp2, vector_part_pd, matrix_part_sp2, matrix_part_pd)
        self.play(FadeIn(rect_parts))
        self.wait()

        combo_sp2_pd = Tex(" = ", "\\text{sp}^2", " + ", "\\text{pd}", color=YELLOW)
        combo_sp2_pd[1].set_color(color_sp2)
        combo_sp2_pd[3].set_color(color_pd)
        combo_sp2_pd.next_to(group_sp3d[0][0][:-4], RIGHT)
        cover_rect = SurroundingRectangle(
            group_sp3d[0][0][-4:],
            color=BLACK, fill_opacity=1, stroke_width=0, buff=0.01,
        )
        self.play(
            FadeIn(cover_rect),
            Write(combo_sp2_pd),
        )
        self.wait()
        fade_group = VGroup(cover_rect, combo_sp2_pd, rect_parts)
        self.play(FadeOut(fade_group))
        self.wait()

        # sp3d2 and sd2f3 hybridization
        self.play(self.get_fade_animation(group_sp3d, group_sp3d2))
        self.wait()
        self.play(self.get_fade_animation(group_sp3d2, group_sd2f3))
        self.wait()

        # Fade out everything and add reference line
        curr_mobs = Group(*self.mobjects)
        h_line = Line(LEFT_SIDE, RIGHT_SIDE, color=GOLD)
        v_line = Line(TOP, BOTTOM, color=GOLD)
        self.play(
            FadeOut(curr_mobs),
            FadeIn(VGroup(h_line, v_line)),
        )
        self.wait()

        other_hybrids = VGroup(*[
            TexText(hybrid_name)
            for hybrid_name in ("sp$^{2}$d", "spd$^{4}$", "sp$^{3}$d$^{3}$", "p$^{3}$d$^{5}$")
        ])
        directions = [UL, UR, DL, DR]
        for hybrid_text, direction in zip(other_hybrids, directions):
            hybrid_text.next_to(ORIGIN, direction)
            hybrid_text.set_color(YELLOW)
            self.play(Write(hybrid_text), run_time=1)
            self.wait()

    def get_fade_animation(self, source_group, target_group, **kwargs):
        return AnimationGroup(*[
            FadeTransform(source_mob, target_mob)
            for source_mob, target_mob in zip(source_group, target_group)
        ], **kwargs)


class OutroSceneP1(Scene):
    def construct(self):
        # Summary
        frame_rect = PictureInPictureFrame()
        frame_rect.set_height(6).shift(0.25 * DOWN)
        self.add(frame_rect)
        titles = VGroup(*[
            Tex("\\text{微分方程}", "\\rightarrow", "\\text{原子轨道}"),
            Tex("\\text{采样/等值面绘制}", "\\rightarrow", "\\text{原子轨道形状}"),
            Tex("\\text{(特殊的)线性组合}", "\\rightarrow", "\\text{轨道的多种表示方法}"),
            Tex("\\text{点群和特征标表}", "\\rightarrow", "\\text{杂化轨道}"),
            Tex("\\text{对称性匹配的组合}", "\\rightarrow", "\\text{分子轨道}"),
        ])
        for ind, title in enumerate(titles):
            title.to_edge(UP)
            for mob, color in zip(title, [YELLOW, WHITE, BLUE]):
                mob.set_color(color)
            if ind == 0:
                self.add(title)
                self.wait()
            else:
                self.play(
                    AnimationGroup(*[
                        FadeTransform(source_mob, target_mob)
                        for source_mob, target_mob in zip(titles[ind - 1], titles[ind])
                    ])
                )
                self.wait()

        # Coming up in the next video
        curr_mobs = VGroup(frame_rect, titles[-1])
        trailer_title = TexText("下期预告", color=YELLOW)
        trailer_title.to_edge(UP)
        drawing_image = ImageMobject("py-mpl-ao.png")
        drawing_image.set_height(5)
        drawing_image.center()
        rect = Rectangle(
            height=drawing_image.get_height(), width=drawing_image.get_width(),
            color=YELLOW, stroke_width=10,
        ).move_to(drawing_image)
        self.play(
            FadeOut(curr_mobs),
            FadeIn(Group(trailer_title, rect, drawing_image)),
        )
        self.wait()

        # Frame in frame scene
        self.remove(*self.mobjects)
        self.wait()

        unitary_text = TexText("酉变换：")
        new_orbital_vector = Matrix(
            [["``化学'' \\\\ 视角 \\\\ 轨道", ], ],
            element_to_mobject=TexText,
        )
        equal_sign = Tex("=")
        outer_coeff = Tex("{1 \\over \\sqrt{2}}")
        coeff_matrix = Matrix(
            [["1", "1"],
             ["-\\mathrm{i}", "\\mathrm{i}"]],
            h_buff=1, v_buff=0.8,
        )
        old_orbital_vector = Matrix(
            [["``物理'' \\\\ 视角 \\\\ 轨道", ], ],
            element_to_mobject=TexText,
        )
        perspective_group = VGroup(
            unitary_text, new_orbital_vector, equal_sign, outer_coeff,
            coeff_matrix, old_orbital_vector
        )

        orthogonal_text = TexText("正交变换：")
        hybrid_vector = Matrix(
            [["(sp$^{3}$)$_{a}$", ],
             ["(sp$^{3}$)$_{b}$", ],
             ["(sp$^{3}$)$_{c}$", ],
             ["(sp$^{3}$)$_{d}$", ]],
            element_to_mobject=TexText,
        )
        hybrid_outer_coeff = Tex("{1 \\over 2}")
        hybrid_matrix = Tex(
            """
            \\begin{bmatrix}
                1 &  1 &  1 &  1 \\\\
                1 &  1 & -1 & -1 \\\\
                1 & -1 &  1 & -1 \\\\
                1 & -1 & -1 &  1 \\\\
            \\end{bmatrix}
            """
        )
        ao_vector = Matrix(
            [["s", ], ["p$_{x}$", ], ["p$_{y}$", ], ["p$_{z}$", ]],
            element_to_mobject=TexText,
        )
        hybrid_group = VGroup(
            orthogonal_text, hybrid_vector, equal_sign.copy(), hybrid_outer_coeff,
            hybrid_matrix, ao_vector,
        )

        all_groups = VGroup(perspective_group, hybrid_group)
        for vector in (new_orbital_vector, old_orbital_vector):
            vector.scale(0.7)
        for vector in (hybrid_vector, ao_vector):
            vector.scale(0.8)
        for group in all_groups:
            group[0].set_color(YELLOW)
            group.arrange(RIGHT)
            VGroup(group[1], group[-1]).set_color(BLUE)
            group[3:-1].set_color(YELLOW)
        all_groups.arrange(
            DOWN, buff=2.5,
            index_of_submobject_to_align=0, aligned_edge=RIGHT,
        )
        all_groups.set_width(FRAME_WIDTH - 1)
        all_groups.center()
        unitary_extra = Tex("\\text{（逆} = \\text{共轭转置）}")
        orthogonal_extra = Tex("\\text{（逆} = \\text{转置）}")
        extra_group = VGroup(unitary_extra, orthogonal_extra)
        for extra, orig_group in zip(extra_group, all_groups):
            target_text = orig_group[0]
            extra.set_color(YELLOW_A)
            extra.scale(0.5).next_to(target_text, DOWN)

        self.add(all_groups, extra_group)
        self.wait()

        # Point group and character table play an important role in hybridization
        character_table_Td = Tex(
            """
            \\begin{tabular}{l|ccccc|cc}
            $T_d$ &
                $E$ &
                8$C_{3}$ &
                3$C_{2}$ &
                6$S_{4}$ &
                6$\\sigma_{\\text{d}}$ &
                一次函数 &
                二次函数 \\\\ \\hline
            $A_1$ & $+1$ & $+1$ & $+1$ & $+1$ & $+1$ & s & - \\\\
            $A_2$ & $+1$ & $+1$ & $+1$ & $-1$ & $-1$ & - & - \\\\
            $E$     & $+2$ & $-1$ & $+2$ & $0 $ & $0 $ & - & (d$_{z^2}$, d$_{x^2-y^2}$) \\\\
            $T_1$ & $+3$ & $0 $ & $-1$ & $+1$ & $-1$ & - & - \\\\
            $T_2$ & $+3$ & $0 $ & $-1$ & $-1$ & $+1$ & (p$_{x}$, p$_{y}$, p$_{z}$) & (d$_{xy}$, d$_{xz}$, d$_{yz}$)
            \\end{tabular}
            """
        )
        character_table_Td.set_width(11)
        character_table_Td.move_to(perspective_group)
        self.play(
            FadeOut(perspective_group),
            FadeOut(unitary_extra),
            FadeIn(character_table_Td),
        )
        self.wait()


class ReferencesAll(ReferenceScene):
    def get_references(self):
        orbitron_ref = Reference(
            name='The Orbitron',
            authors='Mark Winter',
            pub='网站',
            doi='https://winter.group.shef.ac.uk/orbitron/',
            info='比较全的原子轨道演示，同时包括电子云和轮廓两种可视化手段，还有原子轨道的表达式、节面、径向分布函数等等很多内容。'
        )
        cloud_rust_ref = Reference(
            name='Hydrogenic Orbitals',
            authors='Alvin Q. Meng',
            pub='网站',
            doi='https://al2me6.github.io/evanescence/',
            info='使用电子云来可视化轨道。除了原子轨道和杂化轨道之外，还包括$\\text{H}_{2}^{+}$的分子轨道。'
        )
        ao_view_ref = Reference(
            name='Hydrogen Atoms under Magnification: Direct Observation \\\\ of the Nodal Structure of Stark States',
            authors='A. S. Stodolna, et al.',
            pub='Physical Review Letters, 2013, 110(21): 213001.',
            doi='https://doi.org/10.1103/PhysRevLett.110.213001',
            info='原子轨道是计算出来的，除了通过光谱间接验证结果，这篇介绍了一种用实验手段直接观测激发态氢原子轨道的方法。',
        )
        hybrid_basics_ref = Reference(
            name='Symmetry, Hybridization and Bonding in Molecules',
            authors='Zvonimir B. Maksić',
            pub='Symmetry. Pergamon, 1986: 697-723.',
            doi='https://doi.org/10.1016/B978-0-08-033986-3.50050-1',
            info='详细介绍了如何由$\\text{T}_\\text{d}$点群计算出$\\text{sp}^3$杂化轨道。',
        )
        point_group_ref = Reference(
            name='Character tables for chemically important point groups',
            authors='Achim Gelessus',
            pub='网站',
            doi='http://symmetry.jacobs-university.de/',
            info='比较全的点群和特征标表，可以用于计算其他的杂化轨道。',
        )
        return (
            orbitron_ref, cloud_rust_ref,
            ao_view_ref, hybrid_basics_ref,
            point_group_ref,
        )


# Part 2 Scene(s?)

class LastPartRecap(Scene):
    def construct(self):
        monitor = SVGMobject(
            "monitor.svg", height=13,
            color=GREY_A, opacity=1,
        )
        monitor.shift(DOWN)
        self.add(monitor)

        images = Group(*[
            ImageMobject(f"{prefix}-3d0.png")
            for prefix in ("dots", "contour")
        ])
        images.arrange(RIGHT, buff=1)
        images.set_height(5.5)
        images.center().shift(0.5 * DOWN)
        self.play(
            AnimationGroup(*[
                GrowFromCenter(image)
                for image in images
            ], lag_ratio=0.2),
            run_time=2,
        )
        self.wait()

        remarks = VGroup(*[
            TexText(text, color=YELLOW)
            for text in ("电子云", "电子云轮廓")
        ])
        for remark, image in zip(remarks, images):
            remark.next_to(image, UP, buff=0.5)
            self.play(FadeIn(remark, shift=UP))
            self.wait()

        psi = Tex("\\Psi_{n \\ell m}")
        arrow = Arrow(ORIGIN, 3 * RIGHT, color=BLUE)
        visual_method = TexText("先这样 \\\\ 再这样 \\\\ 再那样 \\\\ 就行了", color=BLUE)
        psi.scale(2.5)
        psi.move_to(images[0])
        arrow.center().shift(0.5 * DOWN)
        visual_method.scale(0.8)
        visual_method.next_to(arrow, UP, buff=0.5)
        self.play(
            FadeOut(Group(images[0], remarks[0])),
            Write(psi),
        )
        self.wait()
        self.play(
            GrowFromCenter(arrow, run_time=1),
            Write(visual_method, run_time=2),
        )
        self.wait()

        inner_mobs = Group(images[1], remarks[1], psi, arrow, visual_method)
        all_mobs = Group(monitor, inner_mobs)
        self.play(all_mobs.animate.scale(1 / 2).to_edge(UP))
        self.wait()

        show_me_code_text = VGroup(
            Text("Talk is cheap.", font="Consolas"),
            Text("Show me the code.", font="Consolas"),
        )
        show_me_code_text.set_color(MAROON_B)
        show_me_code_text.arrange(DOWN)
        show_me_code_text.set_width(inner_mobs.get_width())
        show_me_code_text.center().shift(UP)
        self.play(
            inner_mobs.animate.fade(0.95),
            Write(show_me_code_text),
        )
        self.wait()

        addon_mark = VGroup(
            Line(ORIGIN, rotate_vector(RIGHT, 2 * PI / 3)),
            Line(ORIGIN, RIGHT)
        )
        addon_mark.set_color(MAROON_B)
        addon_mark.scale(0.5).rotate(-5 / 6 * PI)
        addon_mark.move_to(1.05 * RIGHT + 0.2 * UP)
        python_icon = ImageMobject("python-icon.png")
        python_icon.set_height(1)
        python_icon.next_to(addon_mark, DOWN, buff=0)
        self.play(
            AnimationGroup(*[
                FadeIn(mob, shift=0.5 * UP)
                for mob in (addon_mark, python_icon)
            ], lag_ratio=0.2),
        )
        self.wait()


# Overlays

class P1OverlayProbabilityFlow(Scene):
    def construct(self):
        text = Tex("\\left| 3\\text{d}_2 \\right> \\text{概率流}")
        text.scale(2)
        self.add(text)
        self.wait()


class P1OverlayRabiOscillation(Scene):
    def construct(self):
        text = VGroup(
            Tex("\\left| 3\\text{d}_0 \\right>"),
            Tex(" {\\leftrightarrow} "),
            Tex("\\left| 2\\text{p}_0 \\right> \\text{拉比振荡}"),
        )
        text.arrange(RIGHT)
        text.scale(2)
        self.add(text)
        self.wait()


class P2OverlayArrow(Scene):
    def construct(self):
        arrow = Vector(
            RIGHT,
            stroke_width=10,
            color=YELLOW,
        )
        self.add(arrow)
        self.wait()


class P2OverlayLine(Scene):
    def construct(self):
        arrow = Line(
            ORIGIN, RIGHT,
            stroke_width=10,
            color=YELLOW,
        )
        self.add(arrow)
        self.wait()


class P2OverlayRectangle(Scene):
    def construct(self):
        rect = Rectangle(
            height=FRAME_HEIGHT - 1,
            width=FRAME_WIDTH - 1,
            stroke_width=10,
            color=YELLOW,
        )
        self.add(rect)
        self.wait()


class P2OverlayCircle(Scene):
    def construct(self):
        circle = Circle(
            radius=(FRAME_HEIGHT - 1) / 2,
            stroke_width=10,
            color=YELLOW,
        )
        self.add(circle)
        self.wait()


class P2OverlayVariableVec(Scene):
    def construct(self):
        limit = 6
        N = 50
        colors = [RED, YELLOW, GREEN]

        line = Line(
            limit * LEFT, limit * RIGHT,
            color=GREY,
            stroke_width=3,
        )
        dots = VGroup(*[
            Dot(radius=0.06).move_to(limit * LEFT + k * (2 * limit * RIGHT) / (N - 1))
            for k in range(N)
        ])
        dots.set_color_by_gradient(*colors)
        text_vec = VGroup(
            TexText("数组"),
            Text("vec", font="Consolas"),
        )
        text_vec.arrange(RIGHT)
        text_colors = color_gradient(colors, len(text_vec) + 1)
        for k, character in enumerate(text_vec):
            color1, color2 = text_colors[k], text_colors[k + 1]
            text_vec[k].set_color([color1, color2])

        text_vec.scale(1.5)
        text_vec.next_to(line, UP, buff=0.8)
        arrow_left = Vector(0.8 * UP, color=WHITE)
        arrow_left.next_to(dots[0], DOWN)
        arrow_right = arrow_left.deepcopy()
        arrow_right.next_to(dots[-1], DOWN)
        text_left = Text("-limit", font="Consolas")
        text_left.next_to(arrow_left, DOWN, aligned_edge=LEFT)
        text_right = Text("limit", font="Consolas")
        text_right.next_to(arrow_right, DOWN, aligned_edge=RIGHT)
        text_npoints = Text("n_points", font="Consolas")
        text_npoints.move_to(text_left, coor_mask=Y_COOR_MASK)
        remark_npoints = TexText("点的数目(含2个端点)")
        remark_npoints.scale(0.75)
        remark_npoints.next_to(text_npoints, DOWN)
        self.add(
            line, dots, text_vec,
            arrow_left, arrow_right, text_left, text_right,
            text_npoints, remark_npoints,
        )
        self.wait()


class P2OverlayVariableStep(Scene):
    def construct(self):
        limit = 4.5
        N = 25
        colors = [RED, YELLOW, GREEN]

        line = Line(
            limit * LEFT, limit * RIGHT,
            color=GREY,
            stroke_width=3,
        )
        dots = VGroup(*[
            Dot(radius=0.06).move_to(limit * LEFT + k * (2 * limit * RIGHT) / (N - 1))
            for k in range(N)
        ])
        dots.set_color_by_gradient(*colors)

        text_step = Text("step", font="Consolas")
        ind = 3
        brace_step = Brace(
            VGroup(
                VectorizedPoint(dots[ind].get_center()),
                VectorizedPoint(dots[ind + 1].get_center()),
            )
        )
        brace_step.put_at_tip(text_step)

        formula_step = Text("= 2 * limit / (n_point - 1)", font="Consolas")
        formula_step.next_to(text_step, RIGHT)

        part_length = formula_step[1:8]
        part_num = formula_step[-11:]
        part_length.set_color(BLUE)
        part_num.set_color(MAROON_B)

        text_length = TexText("区间长度", color=part_length.get_color())
        text_length.next_to(part_length, DOWN, buff=0.5)
        text_num = TexText("分段数目", color=part_num.get_color())
        text_num.next_to(part_num, DOWN).move_to(text_length, coor_mask=Y_COOR_MASK)

        self.add(
            line, dots,
            text_step, brace_step, formula_step,
            text_length, text_num,
        )
        self.wait()


class P2OverlayControlInMPL(Scene):
    def construct(self):
        texts = VGroup(*[
            TexText(func, "：", f"鼠标{key}键")
            for func, key in zip(["旋转", "缩放", "平移"], ["左", "右", "中"])
        ])
        texts.set_color(BLACK)
        texts.arrange(DOWN)
        self.add(texts)
        self.wait()


# Thumbnail

class ThumbnailP1(Scene):
    def construct(self):
        # Just the overlay part
        psi = Tex("\\Psi", color=YELLOW)
        psi.scale(5)
        self.add(psi)
        self.wait()
