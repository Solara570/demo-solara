from big_ol_pile_of_manim_imports import *
from custom.custom_mobjects import *


class MainScene(Scene):
    CONFIG = {
        "default_buff" : 0.5,
    }
    def construct(self):
        self.show_problem()
        self.countdown()
        self.show_step_by_step_solution()
        self.show_full_solution()
        self.fadeout_everything()

    def show_problem(self):
        title = TextMobject(
            "“$*$”是集合$\\mathcal{S}$上的二元运算，满足结合律。\\\\",
            "（即对于$\\mathcal{S}$中的任意三个元素$X$、$Y$和$Z$，$(X*Y)*Z=X*(Y*Z)$均成立）\\\\",
            "$\\mathcal{S}$中的三个元素$A$、$B$和$C$满足如下关系：",
            alignment = "",
        )
        rules = VGroup(*[
            TextMobject(text) for text in [
                "(1) $A*B=B*A$", "(2) $A*C=C*A$", "(3) $A*B*A=A$", "(4) $B*A*B=B$"
            ]
        ])
        problem = TextMobject("证明：$B*C=C*B$")
        note = TextMobject("")
        # Arrange stuff
        title[1].scale(0.75).set_color(LIGHT_GREY)
        title.arrange_submobjects(DOWN, aligned_edge = LEFT)
        rules.arrange_submobjects(DOWN)
        title.to_corner(UL, buff = self.default_buff)
        rules.next_to(title, DOWN, aligned_edge = LEFT, buff = self.default_buff)
        problem.next_to(rules, DOWN, aligned_edge = LEFT, buff = self.default_buff)
        self.play(Write(title))
        self.wait()
        self.play(FadeInFromDown(rules), submobjects_mode = "lagged_start")
        self.wait()
        self.play(Write(problem))
        self.wait(2)
        self.title = title
        self.rules = rules
        self.problem = problem

    def countdown(self):
        # 5-second countdown
        t = 5
        circle = Circle(radius = 0.8, stroke_color = WHITE)
        circle.move_to(DOWN*2)
        disk = VMobject()
        t_tracker = ValueTracker(t)
        for k in range(t, 0, -1):
            num_text = TextMobject(str(k), color = BLACK, background_stroke_width = 0)
            num_text.set_height(circle.get_height() * 0.5).move_to(circle)
            disk.add_updater(
                lambda m, dt: m.become(
                    Sector(
                        outer_radius = circle.get_height()/2., color = WHITE,
                        start_angle = PI/2, angle = (t_tracker.get_value()*2*PI) % (2*PI),
                    ).move_arc_center_to(circle.get_center())
                )
            )
            self.add(circle, disk, num_text)
            self.play(t_tracker.increment_value, -1, rate_func = None, run_time = 1)
            self.remove(circle, disk, num_text)
        self.remove(t_tracker)
        self.wait()

    def show_step_by_step_solution(self):
        # Associativity of *
        title, rules, problem = self.title, self.rules, self.problem
        assoc_rule = TextMobject("(0) $*$满足结合律")
        assoc_rule.next_to(rules, UP, buff = 2*self.default_buff, aligned_edge = LEFT)
        # assoc_rule.to_corner(UL, buff = self.default_buff)
        # Rearrange stuff
        title.add_updater(lambda m: m.next_to(rules, UP, aligned_edge = LEFT, buff = self.default_buff))
        problem.add_updater(lambda m: m.next_to(rules, DOWN, aligned_edge = LEFT, buff = self.default_buff))
        rules.generate_target()
        rules.target.next_to(assoc_rule, DOWN, aligned_edge = LEFT)
        problem.generate_target()
        problem.target[-3:].fade(1)
        problem.target[:3].set_color(GREEN)
        self.play(
            MoveToTarget(rules), FadeOut(title), MoveToTarget(problem), Write(assoc_rule),
            run_time = 1,
        )
        title.clear_updaters()
        problem.clear_updaters()
        self.wait()
        # Step 1. Copy LHS and record its position
        step1_text = problem[3:6].deepcopy()
        self.play(ApplyMethod(step1_text.move_to, problem[-3:]), path_arc = PI/2.)
        self.RHS_position = step1_text.get_left()
        self.wait()
        # Step 2. Expand B
        step2_text = TexMobject(*[s for s in "B*A*B*C"])
        self.move_text_to_RHS(step2_text)
        step2_text.generate_target()
        step2_text[-2:].move_to(step1_text[-2:])
        for mob in step2_text[1:-2]:
            mob.move_to(step1_text[0]).fade(1).scale(0)
        self.remove_and_add(step1_text, step2_text)
        sur_rect = SurroundingRectangle(step2_text[:-2])
        self.play(ShowCreation(sur_rect), self.get_rules_highlight_animation(4))
        self.wait(0.5)
        sur_rect.add_updater(lambda m: m.become(SurroundingRectangle(step2_text[:-2])))
        self.play(MoveToTarget(step2_text))
        self.wait(0.5)
        sur_rect.clear_updaters()
        self.play(FadeOut(sur_rect), self.get_rules_reset_animation())
        self.wait()
        # Step 3. Move A to the right
        step3_text = step2_text.deepcopy()
        self.remove_and_add(step2_text, step3_text)
        sur_rect = SurroundingRectangle(step3_text[2])
        self.play(ShowCreation(sur_rect), self.get_rules_highlight_animation(1, 2))
        sur_rect.add_updater(lambda m: m.move_to(step3_text[2].get_center()))
        self.wait(0.5)
        for mob in (step3_text[4], step3_text[-1]):
            self.play(Swap(step3_text[2], mob))
            self.wait(0.5)
        sur_rect.clear_updaters()
        # Step 4. Expand A
        step4_text = TexMobject(*[s for s in "B*B*C*A*B*A"])
        self.move_text_to_RHS(step4_text)
        self.arrange_mobs_based_on_x_position(step3_text)
        self.play(
            FadeOut(sur_rect), self.get_rules_reset_animation(),
            ReplacementTransform(step3_text, step4_text[:-4]),
        )
        self.wait()
        sur_rect = SurroundingRectangle(step4_text[-5])
        self.play(ShowCreation(sur_rect), self.get_rules_highlight_animation(3))
        self.wait(0.5)
        step4_text.generate_target()
        for mob in step4_text[-4:]:
            mob.move_to(step4_text[-5]).fade(1).scale(0)
        sur_rect.add_updater(lambda m: m.become(SurroundingRectangle(step4_text[-5:])))
        self.play(MoveToTarget(step4_text))
        sur_rect.clear_updaters()
        self.wait(0.5)
        # Step 5. Expand A another time
        step5_text = TexMobject(*[s for s in "B*B*C*A*B*A*B*A"])
        self.move_text_to_RHS(step5_text)
        step5_text.generate_target()
        for mob in step5_text[-4:]:
            mob.move_to(step5_text[-5]).fade(1).scale(0)
        self.play(FadeOut(sur_rect))
        sur_rect = SurroundingRectangle(step5_text[-5])
        self.play(ShowCreation(sur_rect))
        self.wait(0.5)
        self.remove_and_add(step4_text, step5_text)
        sur_rect.add_updater(lambda m: m.become(SurroundingRectangle(step5_text[-5:])))
        self.play(MoveToTarget(step5_text))
        sur_rect.clear_updaters()
        self.wait(0.5)
        self.play(FadeOut(sur_rect), self.get_rules_reset_animation())
        self.wait()
        # Step 6. Move all three A's to the left
        step6_text = step5_text.deepcopy()
        self.remove_and_add(step5_text, step6_text)
            # First A
        sur_rect = SurroundingRectangle(step6_text[6])
        self.play(ShowCreation(sur_rect), self.get_rules_highlight_animation(1, 2))
        sur_rect.add_updater(lambda m: m.move_to(step6_text[6].get_center()))
        self.wait(0.5)
        for i in (4, 2, 0):
            self.play(Swap(step6_text[6], step6_text[i]))
            self.wait(0.5)
        sur_rect.clear_updaters()
        self.play(FadeOut(sur_rect))
        self.wait(0.5)
            # Second A
        sur_rect = SurroundingRectangle(step6_text[10])
        self.play(ShowCreation(sur_rect))
        self.wait(0.5)
        sur_rect.add_updater(lambda m: m.move_to(step6_text[10].get_center()))
        self.play(Swap(*[step6_text[i] for i in (2, 4, 8, 10)]))
        sur_rect.clear_updaters()
        self.play(FadeOut(sur_rect))
        self.wait(0.5)
            # Third A
        sur_rect = SurroundingRectangle(step6_text[14])
        self.play(ShowCreation(sur_rect))
        self.wait(0.5)
        sur_rect.add_updater(lambda m: m.move_to(step6_text[14].get_center()))
        self.play(Swap(*[step6_text[i] for i in (4, 8, 12, 14)]))
        sur_rect.clear_updaters()
        # Step 7. Contract
        step7_text = TexMobject(*[s for s in "A*B*A*B*A*C*B*B"])
        self.move_text_to_RHS(step7_text)
        self.arrange_mobs_based_on_x_position(step6_text)
        self.play(
            FadeOut(sur_rect), self.get_rules_reset_animation(),
            ReplacementTransform(step6_text, step7_text),
        )
        self.wait()
        sur_rect = SurroundingRectangle(step7_text[:5])
        self.play(ShowCreation(sur_rect), self.get_rules_highlight_animation(3))
        self.wait(0.5)
        sur_rect.add_updater(lambda m: m.become(SurroundingRectangle(step7_text[:5])))
        step7_text.generate_target()
        for mob in step7_text.target[1:5]:
            mob.move_to(step7_text[0]).fade(1).scale(0)
        step7_text.target[5:].next_to(step7_text[0], RIGHT, buff = 0.15)
        self.play(MoveToTarget(step7_text))
        sur_rect.clear_updaters()
        self.play(FadeOut(sur_rect))
        self.wait(0.5)
        # Step 8. Contract another time
        step8_text = step7_text.deepcopy()
        self.remove_and_add(step7_text, step8_text)
        sur_rect = SurroundingRectangle(step8_text[:9])
        self.play(ShowCreation(sur_rect))
        self.wait(0.5)
        sur_rect.add_updater(lambda m: m.become(SurroundingRectangle(step8_text[:9])))
        step8_text.generate_target()
        for mob in step8_text.target[5:9]:
            mob.move_to(step8_text[0]).fade(1).scale(0)
        step8_text.target[9:].next_to(step8_text[0], RIGHT, buff = 0.15)
        self.play(MoveToTarget(step8_text))
        sur_rect.clear_updaters()
        step8_text.remove(*[mob for mob in step8_text[1:9]])
        self.wait(0.5)
        # Step 9. Move A to the right
        step9_text = TexMobject(*[s for s in "A*C*B*B"])
        self.move_text_to_RHS(step9_text)
        self.arrange_mobs_based_on_x_position(step8_text)
        self.play(
            FadeOut(sur_rect), self.get_rules_reset_animation(),
            ReplacementTransform(step8_text, step9_text),
        )
        self.wait()
        sur_rect = SurroundingRectangle(step9_text[0])
        self.play(ShowCreation(sur_rect), self.get_rules_highlight_animation(1, 2))
        sur_rect.add_updater(lambda m: m.move_to(step9_text[0].get_center()))
        self.wait(0.5)
        self.play(Swap(*[step9_text[i] for i in (4, 2, 0)]))
        sur_rect.clear_updaters()
        # Step 10. Contract for the last time
        step10_text = TexMobject(*[s for s in "C*B*A*B"])
        self.move_text_to_RHS(step10_text)
        self.arrange_mobs_based_on_x_position(step9_text)
        self.play(
            FadeOut(sur_rect), self.get_rules_reset_animation(),
            ReplacementTransform(step9_text, step10_text),
        )
        self.wait()
        sur_rect = SurroundingRectangle(step10_text[2:])
        self.play(ShowCreation(sur_rect), self.get_rules_highlight_animation(4))
        self.wait(0.5)
        sur_rect.add_updater(lambda m: m.become(SurroundingRectangle(step10_text[2:])))
        step10_text.generate_target()
        for mob in step10_text.target[3:]:
            mob.move_to(step10_text[2].get_center()).fade(1).scale(0)
        self.play(MoveToTarget(step10_text))
        self.wait(0.5)
        sur_rect.clear_updaters()
        self.play(
            FadeOut(sur_rect), self.get_rules_reset_animation(),
            ApplyMethod(problem[:3].set_color, WHITE)
        )
        self.wait()

    def show_full_solution(self):
        remark = TextMobject("证明过程：")
        remark.set_color(YELLOW)
        solution = TexMobject("""
            \\begin{aligned}
            B*C &= B*A*B*C \\\\
                &= B*B*C*A \\\\
                &= B*B*C*A*B*A*B*A \\\\
                &= A*B*A*B*A*C*B*B \\\\
                &= A*C*B*B \\\\
                &= C*B*A*B \\\\
                &= C*B
            \\end{aligned}
        """)
        solution.scale(0.8)
        note = TextMobject(
            "（$C$右边的$B$由$A$产生，左边的$B$被$A$消除。\\\\简单来说，$B$通过$A$实现了与$C$的交换）",
            alignment = ""
        )
        note.scale(0.7).set_color(LIGHT_GREY)
        group = VGroup(remark, solution, note)
        group.arrange_submobjects(DOWN, aligned_edge = LEFT).to_edge(RIGHT)
        qed = QEDSymbol().scale(0.6)
        qed.next_to(solution[-1], RIGHT, buff = 0.4)
        self.play(Write(remark))
        self.wait(0.5)
        self.play(Write(solution))
        self.wait(0.5)
        self.play(DrawBorderThenFill(qed))
        self.wait()
        self.play(Write(note))
        self.wait(5)

    def fadeout_everything(self):
        mobs = VGroup(*self.mobjects)
        self.play(FadeOut(mobs), run_time = 2)

    def get_rules_highlight_animation(self, *indices):
        light_indices = [k for k in (1, 2, 3, 4) if k in indices]
        dim_indices = [k for k in (1, 2, 3, 4) if k not in indices]
        self.rules.generate_target()
        for k in (1, 2, 3, 4):
            self.rules.target[k-1].set_color(YELLOW if k in indices else GREY)
        return MoveToTarget(self.rules)

    def get_rules_reset_animation(self):
        self.rules.generate_target()
        self.rules.target.set_color(WHITE)
        return MoveToTarget(self.rules)

    def move_text_to_RHS(self, mob):
        mob.next_to(self.RHS_position, RIGHT, buff = 0)

    def arrange_mobs_based_on_x_position(self, mob_group):
        mob_group.submobjects.sort(key = lambda m: m.get_center()[0])

    def remove_and_add(self, mob1, mob2):
        self.remove(mob1)
        self.add(mob2)


class EndingScene(Scene):
    def construct(self):
        title = TextMobject("P2. 结合律与交换律")
        title.scale(1.2).set_color(YELLOW).to_corner(UL, buff = 1)
        comment_1 = TextMobject("""
            这是$\\emph{Which Way Did the Bycicle Go? ...and Other Intriguing}$ \\\\
            $\\emph{Mathematical Mysteries}$中的第124题。其实它更像解谜游戏中 \\\\
            会出现的问题......
            """, alignment = ""
        )
        comment_2 = TextMobject("""
            不过类似的“展开消除”技巧也可以在数学证明中用到，比如 \\\\
            Moore-Penrose伪逆的唯一性证明，感兴趣的朋友可以试试看：
            """,
            alignment = ""
        )
        mpinv_problem_1 = TextMobject("设矩阵$A \\in \\mathbb{C}_{m \\times n}$，如果矩阵$X \\in \\mathbb{C}_{n \\times m}$满足：")
        mpinv_problem_2 = TexMobject("""
            &\\text{(1) } (AX)^{\\mathrm{H}} = AX \\quad \\text{（$M^{\\mathrm{H}}$表示$M$的共轭转置）} \\\\
            &\\text{(2) } (XA)^{\\mathrm{H}} = XA \\\\
            &\\text{(3) } AXA = A \\\\
            &\\text{(4) } XAX = X \\\\
            &\\text{则$X$为$A$的伪逆矩阵。证明：$X$唯一（假设$X$存在）} \\\\
        """)
        mpinv_problem_2.next_to(mpinv_problem_1, DOWN, aligned_edge = LEFT)
        mpinv_problem_rect = SurroundingRectangle(
            VGroup(mpinv_problem_1, mpinv_problem_2), stroke_color = WHITE, buff = 0.4
        )
        mpinv_problem = VGroup(mpinv_problem_1, mpinv_problem_2, mpinv_problem_rect)
        comment = VGroup(comment_1, comment_2)
        comment.arrange_submobjects(DOWN)
        comment.scale(0.7)
        comment.next_to(title, DOWN, buff = 0.5, aligned_edge = LEFT)
        mpinv_problem.scale(0.6)
        mpinv_problem.next_to(comment, DOWN, buff = 0.3, aligned_edge = LEFT)
        square = Square(side_length = 1.6).to_corner(DR)
        author = TextMobject("@Solara570")
        author.match_width(square).next_to(square.get_top(), DOWN)
        bgm = TextMobject("BGM: Puddles - Stanley Gurvich")
        bgm.scale(0.5).next_to(author, UP, buff = 0.6, aligned_edge = RIGHT)
        for mob in [title, comment_1, comment_2, mpinv_problem, bgm, author]:
            self.play(FadeIn(mob))
            self.wait()
        self.wait(3)


class Thumbnail(Scene):
    def construct(self):
        rules = VGroup(*[
            TextMobject(text) for text in [
                "(0) $*$满足结合律", "(1) $A*B=B*A$", "(2) $A*C=C*A$", "(3) $A*B*A=A$", "(4) $B*A*B=B$",
            ]
        ])
        rules.arrange_submobjects(DOWN, aligned_edge = LEFT)
        problem = TextMobject("证明：$B*C=C*B$")
        group = VGroup(rules, problem)
        group.scale(1.2)
        group.arrange_submobjects(DOWN, aligned_edge = LEFT, buff = 0.6)
        self.add(group)
        self.wait()

