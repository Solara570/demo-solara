from manimlib.constants import *
from manimlib.animation.composition import AnimationGroup
from manimlib.animation.creation import ShowCreation, Write
from manimlib.animation.fading import FadeInFromDown, FadeOut
from manimlib.animation.indication import ShowCreationThenDestruction
from manimlib.animation.transform import Transform, ReplacementTransform
from manimlib.mobject.shape_matchers import SurroundingRectangle
from manimlib.mobject.svg.tex_mobject import TexMobject
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.scene.scene import Scene

class EulerProductIntuitiveProof(Scene):
    def construct(self):
        self.show_infinite_product()
        self.construct_infinite_sum()
        self.show_claim()
        self.reset_everything()

    def setup(self):
        prod_tex_strings = [
            "\\left[" +
            "+".join([f"f({p}^{k})" for k in range(4)] + ["\\cdots"]) +
            "\\right] \\times"
            for p in [2, 3, 5, 7]
        ] + ["\\cdots"]
        prod_tex = TexMobject(*prod_tex_strings)
        prod_tex.arrange_submobjects(DOWN, aligned_edge = LEFT, buff = 0.5)
        prod_tex.scale(0.8)
        
        sum_tex_strings = [
            f"f({3*k+1}) + f({3*k+2}) + f({3*k+3}) +"
            for k in range(3)
        ] + ["f(10) + \\cdots"]
        sum_tex = TexMobject(*sum_tex_strings)
        sum_tex.arrange_submobjects(DOWN, aligned_edge = LEFT, buff = 0.25)
        sum_tex.scale(0.8)

        VGroup(prod_tex, sum_tex).arrange_submobjects(RIGHT, aligned_edge = UP)
        VGroup(prod_tex, sum_tex).shift(DOWN)
        prod_tex.to_edge(LEFT, buff = 1)
        sum_tex.to_edge(RIGHT, buff = 1)

        # It shouldn't be hard-coded, but... hey, it's only a demo.
        k_list = (
            [0, 0, 0, 0], [1, 1, 1, 1], [2, 1, 1, 1], [1, 2, 1, 1],
            [3, 1, 1, 1], [1, 1, 2, 1], [2, 2, 1, 1], [1, 1, 1, 2],
            [4, 1, 1, 1], [1, 3, 1, 1], [2, 1, 2, 1],
        )

        self.prod_tex = prod_tex
        self.sum_tex = sum_tex
        self.k_list = k_list

    def get_sum_term(self, n):
        if n == 10:
            return self.sum_tex[-1][:-4]
        else:
            q = (n-1) // 3
            r = (n-1) % 3
            return self.sum_tex[q][5*r:5*r+4]

    def get_plus_symbol(self, n):
        if n == 10:
            return self.sum_tex[-1][-4]
        else:
            q = (n-1) // 3
            r = (n-1) % 3
            return self.sum_tex[q][5*r+4]

    def get_highlighted_term(self, term_tex, k):
        return term_tex[6*k-5:6*k]

    def get_highlighted_terms(self, n):
        return VGroup(*[
            self.get_highlighted_term(term_tex, k)
            for term_tex, k in zip(self.prod_tex[:-1], self.k_list[n])
        ])

    def get_highlight_rectangle(self, term_tex, k):
        return SurroundingRectangle(self.get_highlighted_term(term_tex, k))

    def get_highlight_rectangles(self, n):
        return VGroup(*[
            self.get_highlight_rectangle(term_tex, k)
            for term_tex, k in zip(self.prod_tex[:-1], self.k_list[n])
        ])

    def get_times_symbols(self):
        return VGroup(*[term_tex[-1] for term_tex in self.prod_tex[:-1]])
    
    def get_cdots_symbol(self):
        return self.prod_tex[-1]

    def show_infinite_product(self):
        self.play(
            FadeInFromDown(self.prod_tex), lag_ratio = 0.01, run_time = 2
        )
        self.wait()

    def construct_infinite_sum(self):
        for n in range(1, 11):
            # Get highlighted terms
            highlighted_terms = self.get_highlighted_terms(n)
            # Update highlight rectangles
            if n == 1:
                rects = self.get_highlight_rectangles(1)
                self.play(ShowCreation(rects), lag_ratio = 0.2)
                self.wait()
            else:
                new_rects = self.get_highlight_rectangles(n)
                self.play(Transform(rects, new_rects))
                if n <= 4:
                    self.wait()
            # Show the detailed construction of the first four terms
            if n <= 4:
                # Make copies of the elements that are going to be moved
                highlighted_terms_copy = self.get_highlighted_terms(n).deepcopy()
                times_symbols_copy = self.get_times_symbols().deepcopy()
                cdots_copy = self.get_cdots_symbol().deepcopy()
                # Move highlighted terms into position
                arranged_terms_list = []
                for i in range(4):
                    arranged_terms_list.append(highlighted_terms_copy[i])
                    arranged_terms_list.append(times_symbols_copy[i])
                arranged_terms_list.append(cdots_copy)
                arranged_terms = VGroup(*arranged_terms_list)
                arranged_terms.arrange_submobjects(RIGHT, buff = 0.2)
                arranged_terms.next_to(self.sum_tex, UP, aligned_edge = RIGHT, buff = 0.5)
                # Move highlighted terms into position
                anims_list = []
                for i in range(4):
                    anims_list.append(
                        ReplacementTransform(
                            self.get_highlighted_terms(n)[i].deepcopy(),
                            arranged_terms[2*i],
                            lag_ratio = 0, run_time = 2
                        )
                    )
                    anims_list.append(
                        ReplacementTransform(
                            self.get_times_symbols()[i].deepcopy(),
                            arranged_terms[2*i+1],
                            lag_ratio = 0, run_time = 2
                        )
                    )
                anims_list.append(
                    ReplacementTransform(
                        self.get_cdots_symbol().deepcopy(),
                        arranged_terms[-1],
                        lag_ratio = 0, run_time = 2
                    )
                )
                self.play(AnimationGroup(*anims_list))
                self.wait()
                if n == 1:
                    self.play(
                        Transform(arranged_terms, self.get_sum_term(n))
                    )
                else:
                    self.play(
                        Transform(arranged_terms, self.get_sum_term(n)),
                        Write(self.get_plus_symbol(n-1)),
                    )
                self.wait()
            # And show the result for the remaining terms
            else:
                self.play(
                    Transform(
                        VGroup(
                            self.get_highlighted_terms(n).deepcopy(),
                            self.get_times_symbols().deepcopy(),
                            self.get_cdots_symbol().deepcopy(),
                        ),
                        self.get_sum_term(n),
                        lag_ratio = 0,
                    ),
                    Write(self.get_plus_symbol(n-1)),
                )
        # Add \cdots to the end.
        self.wait()
        self.play(
            FadeOut(rects),
            Write(self.sum_tex[-1][-4:])
        )
        self.wait()

    def show_claim(self):
        lhs, equal, rhs = claim = TexMobject(
            "\\prod_{r=1}^{+\\infty}{\\sum_{k=0}^{+\\infty}{f(p_r^k)}",
            "=",
            "\\sum_{n=1}^{+\\infty}{f(n)}"
        )
        claim.scale(1.2)
        claim.to_edge(UP)
        lhs.set_color(GREEN)
        rhs.set_color(BLUE)

        prod_tex_rect = SurroundingRectangle(self.prod_tex, color = GREEN)
        sum_tex_rect = SurroundingRectangle(self.sum_tex, color = BLUE)
        self.play(
            ShowCreationThenDestruction(prod_tex_rect),
            Write(lhs),
            run_time = 2,
        )
        self.play(
            ShowCreationThenDestruction(sum_tex_rect),
            Write(rhs),
            run_time = 2,
        )
        self.play(Write(equal))
        self.wait(3)

    def reset_everything(self):
        self.play(FadeOut(VGroup(*self.mobjects), lag_ratio = 0))
        self.wait()

class EulerProductIntuitiveProofCover(Scene):
    def construct(self):
        lhs, equal, rhs = claim = TexMobject(
            "\\prod_{r=1}^{+\\infty}{\\sum_{k=0}^{+\\infty}{f(p_r^k)}",
            "=",
            "\\sum_{n=1}^{+\\infty}{f(n)}"
        )
        claim.scale(1.8)
        lhs.set_color(GREEN)
        rhs.set_color(BLUE)
        self.add(claim)
        self.wait()

