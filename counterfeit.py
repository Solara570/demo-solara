from manimlib import *
from custom.custom_mobjects import PauseButton
from custom.custom_helpers import brighten


#####
# Constants
DARK_GREY = "#333333"
GREYS = [GREY, DARK_GREY]
GREENS = [GREEN_A, GREEN_E]
REDS = [RED_A, RED_E]
GOLDS = [GOLD_A, GOLD_E]
BLUES = [BLUE_A, BLUE_E]
COIN_BUFF = 0.1
PAN_BALANCE_BUFF = 0.02


#####
# Mobjects
class Coin(VGroup):
    CONFIG = {
        "radius": 0.35,
        "inner_radius_factor": 0.85,
        "num_of_sectors": 30,
        "coin_colors": GOLDS,
        "coin_stroke_width": 2,
        "coin_text_color": None,
    }

    def __init__(self, coin_text=None, **kwargs):
        super().__init__(**kwargs)
        self.coin_stroke_color, self.coin_fill_color = self.coin_colors
        self.coin_text = "" if coin_text is None else coin_text
        if self.coin_text_color is None:
            self.coin_text_color = brighten(self.coin_stroke_color, 0.5)
        self.add_coin()
        self.add_coin_text()

    def add_coin(self):
        outer_circle = Circle(
            radius=self.radius,
            fill_opacity=1,
            fill_color=self.coin_fill_color,
            stroke_color=self.coin_stroke_color,
            stroke_width=self.coin_stroke_width,
        )
        sectors = VGroup(*[
            AnnularSector(
                start_angle=k * TAU / self.num_of_sectors,
                angle=TAU / self.num_of_sectors,
                outer_radius=self.radius,
                inner_radius=self.radius * self.inner_radius_factor,
                fill_opacity=1,
                fill_color=self.coin_fill_color,
                stroke_color=self.coin_stroke_color,
                stroke_width=self.coin_stroke_width,
            )
            for k in range(self.num_of_sectors)
        ])
        self.coin = VGroup(outer_circle, sectors)
        self.add(self.coin)

    def add_coin_text(self):
        self.coin_text = Tex(
            str(self.coin_text), color=self.coin_text_color,
        )
        self.add(self.coin_text)

        def update_coin_text(mob):
            max_height = self.coin.get_height() * 0.4
            max_width = self.coin.get_width() * 0.6
            mob.set_width(max_width)
            if mob.get_height() > max_height:
                mob.set_height(max_height)
            mob.move_to(self.coin.get_center())
        self.coin_text.add_updater(update_coin_text)


class UnknownCoin(Coin):
    pass


class HeavierCoin(Coin):
    CONFIG = {
        "coin_colors": BLUES,
    }

    def __init__(self, coin_text="+", **kwargs):
        super().__init__(coin_text=coin_text, **kwargs)


class LighterCoin(Coin):
    CONFIG = {
        "coin_colors": REDS,
    }

    def __init__(self, coin_text="-", **kwargs):
        super().__init__(coin_text=coin_text, **kwargs)


class RealCoin(Coin):
    CONFIG = {
        "coin_colors": GREENS,
    }

    def __init__(self, coin_text="\\checkmark", **kwargs):
        super().__init__(coin_text=coin_text, **kwargs)


class PanBalance(VGroup):
    CONFIG = {
        "max_tilt_angle": PI / 24,
        "part_stroke_width": 3,
        "height": 4,
        "max_num_of_coins_in_a_row": 4,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_pan_balance()
        self.add_part_updaters()
        self.add_screen_text()

    def create_pan_balance(self):
        pan_balance = SVGMobject(
            "pan_balance.svg",
            stroke_color=WHITE, stroke_width=0
        )
        self.add(pan_balance)
        # Set colors and strokes
        for part in pan_balance[:-4]:
            part.set_fill(color=GREY)
            part.set_stroke(width=self.part_stroke_width, color=DARK_GREY)
        for part in pan_balance[-4:]:
            part.set_fill(color=DARK_GREY)
        self.pan_balance = pan_balance
        # Small tweaks
        beam = self.get_beam()
        beam.set_stroke(width=0)
        center_axle = self.get_center_axle()
        center_axle.move_to(beam.get_center())
        self.set_height(self.height)

    def add_part_updaters(self):
        self.add_axles_updaters()
        self.add_pans_updaters()

    def add_axles_updaters(self):
        # Add left/right axles' updaters
        beam = self.get_beam()
        left_axle = self.get_left_axle()
        right_axle = self.get_right_axle()
        left_axle.add_updater(lambda m: m.move_to(beam.get_anchors()[1]))
        right_axle.add_updater(lambda m: m.move_to(beam.get_anchors()[13]))

    def add_pans_updaters(self):
        # Add left/right pans' updaters
        left_axle = self.get_left_axle()
        right_axle = self.get_right_axle()
        left_pan = self.get_left_pan()
        right_pan = self.get_right_pan()
        left_pan.add_updater(
            lambda m: m.shift(
                left_axle.get_center() - self.get_left_pan_center()
            )
        )
        right_pan.add_updater(
            lambda m: m.shift(
                right_axle.get_center() - self.get_right_pan_center()
            )
        )

    def add_screen_text(self):
        self.screen_text = ScreenText(self)
        self.add(self.screen_text)

    # Put-stuff-on-the-balance methods
    def arrange_coins(self, coins, arrangement=None):
        num_of_coins = len(coins)
        max_length = self.max_num_of_coins_in_a_row
        if arrangement is not None:
            assert(sum(arrangement) == num_of_coins)
        elif num_of_coins % max_length == 0:
            arrangement = [max_length for k in range(num_of_coins // max_length)]
        else:
            arrangement = [num_of_coins % max_length] + num_of_coins // max_length * [max_length]
        full_coin_set = VGroup()
        for k in range(len(arrangement)):
            start_index = sum(arrangement[:k])
            end_index = start_index + arrangement[k]
            curr_coin_set = coins[start_index:end_index]
            curr_coin_set.arrange(RIGHT, buff=COIN_BUFF)
            full_coin_set.add(curr_coin_set)
        full_coin_set.arrange(DOWN, buff=COIN_BUFF / 2)

    def put_coins_on_left_pan(self, coins, arrangement=None, add_updater=False):
        self.arrange_coins(coins, arrangement=arrangement)
        coins.next_to(self.get_left_pan_top(), UP, buff=PAN_BALANCE_BUFF)
        if add_updater:
            self.add_left_pan_updater(coins)

    def put_coins_on_right_pan(self, coins, arrangement=None, add_updater=False):
        self.arrange_coins(coins, arrangement=arrangement)
        coins.next_to(self.get_right_pan_top(), UP, buff=PAN_BALANCE_BUFF)
        if add_updater:
            self.add_right_pan_updater(coins)

    def add_left_pan_updater(self, coins):
        coins.add_updater(
            lambda m: m.next_to(self.get_left_pan_top(), UP, buff=PAN_BALANCE_BUFF)
        )

    def add_right_pan_updater(self, coins):
        coins.add_updater(
            lambda m: m.next_to(self.get_right_pan_top(), UP, buff=PAN_BALANCE_BUFF)
        )

    # Tilt animations
    def get_beam_rotation_animation(self, angle, should_wiggle=False):
        beam = self.get_beam()
        curr_beam_angle = self.get_beam_angle()
        if should_wiggle:
            max_wiggle_angle = self.max_tilt_angle / 3
            rotate_angle = random.choice([max_wiggle_angle, - max_wiggle_angle])
            rotate_rate_func = wiggle
        else:
            rotate_angle = angle - curr_beam_angle
            rotate_rate_func = smooth
        return Rotate(
            beam, rotate_angle,
            about_edge=None, about_point=self.get_center_axle().get_center(),
            rate_func=rotate_rate_func,
        )

    def get_tilt_animation(self, angle, screen_text_anim_type=None, *added_anims, **kwargs):
        should_wiggle = kwargs.get("should_wiggle", False)
        anims = [
            self.get_beam_rotation_animation(angle, should_wiggle=should_wiggle),
            Animation(self.get_pans(), suspend_mobject_updating=False),
            Animation(self.get_stand(), suspend_mobject_updating=False),
            Animation(self.get_axles(), suspend_mobject_updating=False),
            Animation(self.get_screen(), suspend_mobject_updating=False),
        ]
        if screen_text_anim_type is not None:
            if screen_text_anim_type == "hide":
                anims.append(self.get_screen_text().animate.hide())
            elif screen_text_anim_type == "show":
                anims.append(self.get_screen_text().animate.show())
        for added_anim in added_anims:
            anims.append(added_anim)
        return AnimationGroup(*anims, **kwargs)

    def get_tilt_left_animation(self, screen_text_anim_type=None, *added_anims, **kwargs):
        return self.get_tilt_animation(
            self.max_tilt_angle, screen_text_anim_type, *added_anims, **kwargs
        )

    def get_tilt_right_animation(self, screen_text_anim_type=None, *added_anims, **kwargs):
        return self.get_tilt_animation(
            -self.max_tilt_angle, screen_text_anim_type, *added_anims, **kwargs
        )

    def get_balance_animation(self, screen_text_anim_type=None, *added_anims, **kwargs):
        return self.get_tilt_animation(
            0, screen_text_anim_type, *added_anims, **kwargs
        )

    def get_wiggle_animation(self, screen_text_anim_type=None, *added_anims, **kwargs):
        return self.get_tilt_animation(
            0, screen_text_anim_type, should_wiggle=True, *added_anims, **kwargs
        )

    # Getters
    def get_stand(self):
        return self.pan_balance[0]

    def get_beam(self):
        return self.pan_balance[1]

    def get_beam_angle(self):
        beam_right = self.get_beam().get_anchors()[-4]
        beam_left = self.get_beam().get_anchors()[0]
        return angle_of_vector(beam_right - beam_left)

    def get_left_pan(self):
        return self.pan_balance[3]

    def get_left_pan_center(self):
        left_pan = self.get_left_pan()
        return np.mean(left_pan.get_anchors()[12:25:4], axis=0)

    def get_left_pan_top(self):
        return self.get_left_pan().get_anchors()[-2]

    def get_right_pan(self):
        return self.pan_balance[2]

    def get_right_pan_center(self):
        right_pan = self.get_right_pan()
        return np.mean(right_pan.get_anchors()[12:25:4], axis=0)

    def get_right_pan_top(self):
        return self.get_right_pan().get_anchors()[-2]

    def get_pans(self):
        return self.pan_balance[2:4]

    def get_center_axle(self):
        return self.pan_balance[4]

    def get_left_axle(self):
        return self.pan_balance[5]

    def get_right_axle(self):
        return self.pan_balance[6]

    def get_axles(self):
        return self.pan_balance[4:7]

    def get_screen(self):
        return self.pan_balance[-1]

    def get_screen_text(self):
        return self.screen_text


class ScreenText(VGroup):
    CONFIG = {
        "width_factor": 0.4,
        "override_display_color": None,
    }

    def __init__(self, pan_balance=None, symbol="?", **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.pan_balance = PanBalance() if pan_balance is None else pan_balance
        if self.override_display_color is None:
            self.display_color = self.pan_balance.get_stand().get_fill_color()
        else:
            self.display_color = self.override_display_color
        self.hidden_color = self.pan_balance.get_screen().get_fill_color()
        self.add_text()
        self.add_text_updater()

    def add_text(self):
        self.text = Tex(str(self.symbol), color=self.display_color)
        screen = self.pan_balance.get_screen()
        self.text.set_width(screen.get_width() * self.width_factor)
        self.add(self.text)

    def add_text_updater(self):
        screen = self.pan_balance.get_screen()
        # Set appropriate width
        self.text.add_updater(
            lambda m: m.set_width(screen.get_width() * self.width_factor)
        )
        # Move to screen center
        self.text.add_updater(
            lambda m: m.move_to(screen.get_center())
        )

    def show(self):
        return self.set_color(self.display_color)

    def hide(self):
        return self.set_color(self.hidden_color)


class PossibilityBranch(VGroup):
    CONFIG = {
        "width": 4,
        "height": 2,
        "arrow_thickness": 0.05,
        "circle_stroke_width": 3,
        "root_radius": 0.54,
        "branch_radius": None,
        "text_scaling_factor": 1,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.branch_radius is None:
            self.branch_radius = self.root_radius * 2 / 3
        xs_and_ys = (
            [0, 0], [-self.width, -self.height],
            [0, -self.height], [self.width, -self.height]
        )
        arrows = VGroup(*[
            Arrow(
                ORIGIN, x * RIGHT + y * UP, color=GREY_A,
                thickness=self.arrow_thickness,
                buff=(self.branch_radius + self.root_radius) / 2
            )
            for (x, y) in xs_and_ys[1:]
        ])
        for arrow in arrows:
            nudge = (self.root_radius - self.branch_radius) / 2
            arrow.shift(nudge * arrow.get_unit_vector())
        circles = VGroup(*[
            Circle(
                radius=radius,
                stroke_width=self.circle_stroke_width, stroke_color=WHITE,
                fill_color=BLACK, fill_opacity=1,
            )
            .move_to(x * RIGHT + y * UP)
            for radius, (x, y) in zip([self.root_radius] + [self.branch_radius] * 3, xs_and_ys)
        ])
        labels = VGroup(*[
            self.get_text_label(text, scaling_factor=self.text_scaling_factor).move_to(arrow)
            for text, arrow in zip(["向左倾斜", "平衡", "向右倾斜"], arrows)
        ])
        self.add(arrows, circles, labels)
        self.arrows = arrows
        self.circles = circles
        self.labels = labels

    def get_text_label(self, text, scaling_factor=1):
        text = TexText(text)
        label = SurroundingRectangle(
            text, buff=0.2,
            stroke_width=1, stroke_color=WHITE,
            fill_opacity=1, fill_color=BLACK,
        )
        text_label = VGroup(label, text)
        text_label.set_height(self.height / 4)
        text_label.scale(scaling_factor)
        return text_label

    def get_root_circle(self):
        return self.circles[0]

    def get_branch_circles(self):
        return self.circles[1:]

    def get_root_radius(self):
        return self.root_radius

    def get_branch_radius(self):
        return self.branch_radius

    def graft_in(self, mob):
        curr_pos = self.get_root_circle().get_center()
        target_pos = mob.get_center()
        self.shift(target_pos - curr_pos)
        return self


class TexMatrix(Matrix):
    CONFIG = {
        "v_buff": 0.7,
        "h_buff": 1.05,
        "height": None,
        "scale_factor": None,
    }

    def __init__(self, matrix, **kwargs):
        map_to_string_matrix = [list(map(str, row)) for row in matrix]
        super().__init__(map_to_string_matrix, **kwargs)
        if self.height is not None:
            self.set_height(self.height)
        elif self.scale_factor is not None:
            self.scale(self.scale_factor)

    def get_entry(self, row_ind, col_ind):
        return self.mob_matrix[row_ind][col_ind][0]


class WeighingMatrix(TexMatrix):
    CONFIG = {
        "height": 1.5,
    }


class HVector(TexMatrix):
    CONFIG = {
        "h_buff": 1,
        "scale_factor": 0.8,
    }

    def __init__(self, *entries, **kwargs):
        self.entries = entries
        super().__init__([list(entries)], **kwargs)

    def get_column(self, col_ind):
        return self.get_entry(0, col_ind)

    def get_columns(self):
        return VGroup(*[
            self.get_column(k)
            for k in range(len(self.entries))
        ])

    def get_entries(self):
        return self.get_columns()


class VVector(TexMatrix):
    CONFIG = {
        "v_buff": 0.65,
        "scale_factor": 0.8,
    }

    def __init__(self, *entries, **kwargs):
        self.entries = entries
        super().__init__([[entry] for entry in entries], **kwargs)

    def get_row(self, row_ind):
        return self.get_entry(row_ind, 0)

    def get_rows(self):
        return VGroup(*[
            self.get_row(k)
            for k in range(len(self.entries))
        ])

    def get_entries(self):
        return self.get_rows()


class LargeBrackets(VGroup):
    def __init__(self, mob):
        super().__init__()
        brackets = TexText("\\Huge(", "\\Huge)")
        brackets.match_height(mob, stretch=True)
        brackets[0].next_to(mob, LEFT)
        brackets[1].next_to(mob, RIGHT)
        self.add(brackets)
        self.brackets = brackets


#####
# Part 1 Scenes


class IntroTo12CoinsPuzzle(Scene):
    def construct(self):
        self.show_coins()
        self.show_pan_balance()
        self.show_questions()

    def show_coins(self):
        # Show 12 coins
        coins = VGroup(*[Coin(k) for k in range(1, 13)])
        coins.arrange(RIGHT, buff=COIN_BUFF)
        self.play(
            AnimationGroup(
                *[GrowFromCenter(coin) for coin in coins],
                lag_ratio=0.2
            ),
            run_time=2,
        )
        self.wait()
        # 1 counterfeit at most: maybe one is lighter...
        light_5 = LighterCoin("5").move_to(coins[5 - 1])
        light_arrow = Arrow(ORIGIN, UP, fill_color=light_5.coin_fill_color)
        light_text = TexText("轻一些", color=light_5.coin_fill_color)
        light_arrow.next_to(light_5, DOWN)
        light_text.next_to(light_arrow, DOWN)
        light_group = VGroup(light_5, light_arrow, light_text)
        self.play(FadeIn(light_group), run_time=0.5)
        self.wait()
        # maybe one is heavier...
        heavy_3 = HeavierCoin("3").move_to(coins[3 - 1])
        heavy_arrow = Arrow(ORIGIN, UP, fill_color=heavy_3.coin_fill_color)
        heavy_text = TexText("重一些", color=heavy_3.coin_fill_color)
        heavy_arrow.next_to(heavy_3, DOWN)
        heavy_text.next_to(heavy_arrow, DOWN)
        heavy_group = VGroup(heavy_3, heavy_arrow, heavy_text)
        self.play(FadeOut(light_group), FadeIn(heavy_group), run_time=0.5)
        self.wait()
        # or maybe all coins are real.
        real_coins = VGroup(*[RealCoin().move_to(coin) for coin in coins])
        self.play(FadeOut(heavy_group), FadeIn(real_coins), run_time=0.5)
        self.wait()
        self.play(FadeOut(real_coins))
        self.wait()
        self.coins = coins

    def show_pan_balance(self):
        # The difference in weight can be captured by a pan balance
        pb = PanBalance()
        pb.next_to(TOP, UP)
        self.play(
            pb.animate.next_to(TOP, DOWN, buff=0.5),
            self.coins.animate.shift(1.5 * DOWN),
            run_time=1
        )
        self.wait()
        self.pb = pb

    def show_questions(self):
        # Move pan balance and coins to show questions
        figure_group = VGroup(self.coins, self.pb)
        self.play(figure_group.animate.to_edge(LEFT))
        self.wait()
        # Using at most 3 weighings to determine the answer
        rule = TexText("3次称量机会", color=YELLOW)
        questions = BulletedList("是否有假币？", "假币的编号？", "是轻还是重？")
        questions.arrange(DOWN, buff=0.2)
        text_group = VGroup(rule, questions)
        text_group.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        text_group.scale(0.8)
        text_group.next_to(self.pb, RIGHT, aligned_edge=DOWN, buff=0.3)
        self.play(Write(rule))
        self.wait()
        for q in questions:
            self.play(FadeIn(q))
            self.wait()
        # Fade questions for the next scene
        self.play(FadeOut(text_group), figure_group.animate.center())
        self.wait()


class PauseAndThinkAboutIt(Scene):
    def construct(self):
        full_screen_rect = FullScreenFadeRectangle()
        pause_button = PauseButton(color=RED)
        pause_button.set_height(2).fade(0.5).to_corner(DR)
        self.add(full_screen_rect, pause_button)
        self.wait()


class ShowPossibleApproaches(Scene):
    def setup(self):
        coins = VGroup(*[Coin(k) for k in range(1, 13)])
        coins.arrange(RIGHT, buff=COIN_BUFF)
        coins.shift(1.5 * DOWN)
        pb = PanBalance()
        pb.next_to(TOP, DOWN, buff=0.5)
        VGroup(coins, pb).center()
        self.add(pb, coins)
        self.coins = coins
        self.pb = pb

    def construct(self):
        self.show_one_by_one_approach()
        self.show_grouping_approach()

    def show_one_by_one_approach(self):
        pb = self.pb
        coin_1, coin_2, coin_3, coin_4 = self.coins[:4]
        for coin in self.coins[:4]:
            coin.save_state()
        # Weighing one-by-one...
        self.play(
            coin_1.animate.next_to(pb.get_left_pan_top(), UP, buff=PAN_BALANCE_BUFF),
            coin_2.animate.next_to(pb.get_right_pan_top(), UP, buff=PAN_BALANCE_BUFF),
        )
        pb.put_coins_on_left_pan(coin_1, add_updater=True)
        pb.put_coins_on_right_pan(coin_2, add_updater=True)
        self.play(pb.get_wiggle_animation("hide"), run_time=1)
        self.wait()
        coin_1.clear_updaters()
        coin_2.clear_updaters()
        self.play(
            pb.get_balance_animation(
                "show", Restore(coin_1), Restore(coin_2),
            ),
        )
        self.wait()
        self.bring_to_front(self.coins)
        self.play(
            coin_3.animate.next_to(pb.get_left_pan_top(), UP, buff=PAN_BALANCE_BUFF),
            coin_4.animate.next_to(pb.get_right_pan_top(), UP, buff=PAN_BALANCE_BUFF),
        )
        pb.put_coins_on_left_pan(coin_3, add_updater=True)
        pb.put_coins_on_right_pan(coin_4, add_updater=True)
        self.play(pb.get_wiggle_animation("hide"), run_time=1)
        self.wait()
        coin_3.clear_updaters()
        coin_4.clear_updaters()
        # ...is too slow
        cross = Cross(self.pb)
        cross.set_stroke(width=10)
        self.play(ShowCreation(cross))
        self.wait()
        self.play(pb.get_balance_animation("show", FadeOut(cross)))
        self.wait()

    def show_grouping_approach(self):
        pb = self.pb
        coins = self.coins
        # Two 6-coin groups
        left_6_coins, right_6_coins = coins[:6], coins[6:]
        left_6_coins.generate_target()
        pb.put_coins_on_left_pan(left_6_coins.target, [3, 3])
        right_6_coins.generate_target()
        pb.put_coins_on_right_pan(right_6_coins.target, [3, 3])
        self.play(MoveToTarget(left_6_coins), MoveToTarget(right_6_coins))
        self.wait()
        # If it's balanced, then all coins are real
        real_coins = VGroup(*[
            RealCoin(k + 1).move_to(coin)
            for k, coin in enumerate(coins)
        ])
        pb.put_coins_on_left_pan(left_6_coins, [3, 3], add_updater=True)
        pb.put_coins_on_right_pan(right_6_coins, [3, 3], add_updater=True)
        self.play(pb.get_wiggle_animation("hide"))
        self.play(FadeIn(real_coins))
        self.wait()
        self.play(pb.get_balance_animation("show", FadeOut(real_coins)))
        self.wait()
        left_6_coins.clear_updaters()
        right_6_coins.clear_updaters()
        # Add a temporary animation method to simplify things

        def play_3_coin_groups_animation(
            left_coins, right_coins, idle_coins,
            left_arrangement=None, right_arrangement=None,
        ):
            left_coins.generate_target()
            pb.put_coins_on_left_pan(left_coins.target, arrangement=left_arrangement)
            right_coins.generate_target()
            pb.put_coins_on_right_pan(right_coins.target, arrangement=right_arrangement)
            idle_coins.generate_target()
            idle_coins.target.arrange(RIGHT, buff=COIN_BUFF)
            idle_coins.target.next_to(pb, DOWN, buff=0.5)
            self.play(
                AnimationGroup(*[
                    MoveToTarget(coin_set)
                    for coin_set in [left_coins, right_coins, idle_coins]
                ])
            )
            self.wait()
        # Play a few examples
        coin_set_examples = [
            (coins[:4], coins[4:-4], coins[-4:]),  # Three 4-coin groups (4-4-4)
            (coins[:5], coins[5:-2], coins[-2:]),  # (5-5-2)
            (coins[:2], coins[2:-8], coins[-8:]),  # (2-2-8)
            (coins[:3], coins[3:-6], coins[-6:]),  # (3-3-6)
        ]
        for k, (left_coins, right_coins, idle_coins) in enumerate(coin_set_examples):
            left_arrangement = [2, 3] if k == 1 else None
            right_arrangement = [2, 3] if k == 1 else None
            play_3_coin_groups_animation(
                left_coins, right_coins, idle_coins,
                left_arrangement, right_arrangement,
            )
        # How to tell whether a weighing is good or not
        sur_rect = SurroundingRectangle(VGroup(pb, coins))
        good_or_bad_text = TexText("这个称量方法好吗？", color=YELLOW)
        good_or_bad_text.next_to(sur_rect, UP)
        self.play(ShowCreation(sur_rect))
        self.play(Write(good_or_bad_text))
        self.wait()
        # Focus on the coins, and move on to the next scene
        coins.generate_target()
        coins.target.arrange(RIGHT, buff=COIN_BUFF)
        coins.target.center()
        self.play(
            AnimationGroup(*[
                FadeOut(mob, shift=UP) for mob in (sur_rect, good_or_bad_text, pb)
            ], run_time=0.5),
            MoveToTarget(coins, run_time=1),
        )
        self.wait()


class NarrowDownThePossibilities(Scene):
    CONFIG = {
        "random_seed": 570,
    }

    def setup(self):
        coins = VGroup(*[Coin(k) for k in range(1, 13)])
        coins.arrange(RIGHT, buff=COIN_BUFF)
        self.add(coins)
        self.coins = coins

    def construct(self):
        self.show_all_possibilities_and_codes()
        self.narrow_down_to_one()

    def show_all_possibilities_and_codes(self):
        # Arrange stuff
        light_coins = VGroup(*[
            LighterCoin(k + 1).move_to(coin)
            for k, coin in enumerate(self.coins)
        ])
        heavy_coins = VGroup(*[
            HeavierCoin(k + 1).move_to(coin)
            for k, coin in enumerate(self.coins)
        ])
        real_coin = RealCoin("")
        biased_coins = VGroup(light_coins, heavy_coins)
        biased_coins.arrange(DOWN)
        all_coins = VGroup(biased_coins, real_coin)
        all_coins.arrange(RIGHT)
        # Show all possibilities animation
        light_text = TexText("轻一些", color=light_coins[0].coin_fill_color)
        light_text.next_to(light_coins, UP, aligned_edge=LEFT)
        heavy_text = TexText("重一些", color=heavy_coins[0].coin_fill_color)
        heavy_text.next_to(heavy_coins, DOWN, aligned_edge=LEFT)
        real_text = TexText("都是\\\\真币", color=real_coin.coin_fill_color)
        real_text.next_to(light_text, RIGHT, aligned_edge=DOWN)
        real_text.next_to(real_coin, UP, coor_mask=[1, 0, 0])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    TransformFromCopy(self.coins[k], light_coins[k]),
                    ReplacementTransform(self.coins[k], heavy_coins[k])
                )
                for k in range(12)
            ], lag_ratio=0.2, run_time=3),
            Write(light_text, run_time=1),
            Write(heavy_text, run_time=1),
        )
        self.wait()
        self.play(FadeIn(VGroup(real_text, real_coin), shift=UP))
        self.wait()
        # Generate new coins for transformation
        new_light_coins = VGroup(*[
            LighterCoin(str(k + 1) + "-").move_to(coin)
            for k, coin in enumerate(light_coins)
        ])
        new_heavy_coins = VGroup(*[
            HeavierCoin(str(k + 1) + "+").move_to(coin)
            for k, coin in enumerate(heavy_coins)
        ])
        new_real_coin = RealCoin("\\text{A}").move_to(real_coin)
        # For lighter coins, use minus sign as a code
        sur_rect_3 = SurroundingRectangle(VGroup(light_coins[2], heavy_coins[2]))
        sur_rect_3_lighter = SurroundingRectangle(light_coins[2])
        self.play(ShowCreation(sur_rect_3))
        self.wait()
        self.play(ReplacementTransform(sur_rect_3, sur_rect_3_lighter))
        self.wait()
        self.play(
            Transform(light_coins[2], new_light_coins[2]),
            FadeOut(sur_rect_3_lighter),
        )
        self.wait()
        # For heavier coins, use plus sign as a code
        sur_rect_4 = SurroundingRectangle(VGroup(light_coins[3], heavy_coins[3]))
        sur_rect_4_heavier = SurroundingRectangle(heavy_coins[3])
        self.play(ShowCreation(sur_rect_4))
        self.wait()
        self.play(ReplacementTransform(sur_rect_4, sur_rect_4_heavier))
        self.wait()
        self.play(
            ReplacementTransform(heavy_coins[3], new_heavy_coins[3]),
            FadeOut(sur_rect_4_heavier),
        )
        self.wait()
        # Each biased possibility can be encoded in this way
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    ReplacementTransform(light_coins[k], new_light_coins[k]),
                    ReplacementTransform(heavy_coins[k], new_heavy_coins[k]),
                )
                for k in range(12)
            ], lag_ratio=0.2, run_time=3),
            FadeOut(light_text, run_time=1),
            FadeOut(heavy_text, run_time=1),
        )
        self.wait()
        # As for all-real possibility, use "A" to represent
        self.play(ReplacementTransform(real_coin, new_real_coin), FadeOut(real_text))
        self.wait()
        self.all_poss = VGroup(*(
            [new_real_coin] + new_light_coins.submobjects + new_heavy_coins.submobjects
        ))

    def narrow_down_to_one(self):
        # Remove possibilities until only one left
        self.all_poss.shuffle()
        rem_poss = self.all_poss[0]
        self.play(
            AnimationGroup(*[
                FadeOut(
                    mob,
                    shift=ORIGIN if "A" in mob.coin_text.tex_string else (
                        UP if "-" in mob.coin_text.tex_string else DOWN
                    )
                )
                for mob in self.all_poss[1:]
            ], lag_ratio=0.2),
            run_time=5,
        )
        self.wait()
        rem_poss.generate_target()
        rem_poss.target.center().scale(3)
        self.play(MoveToTarget(rem_poss))
        self.wait()
        # Show what that code represents
        color = rem_poss.coin_fill_color
        coin_text = rem_poss.coin_text
        if "A" in rem_poss.coin_text.tex_string:
            # For all-coins-are-real possibility
            code_meaning = TexText("所有硬币都是真的", color=color)
            code_meaning.scale(1.2).next_to(rem_poss, UP, buff=0.4)
            self.play(FadeTransform(coin_text.deepcopy(), code_meaning))
        else:
            # For one-coin-is-counterfeit possibility
            coin_num = coin_text.tex_string[:-1]
            coin_num_tex = coin_text[0][:-1]
            coin_sign = coin_text.tex_string[-1]
            coin_sign_tex = coin_text[0][-1]
            loh = "轻" if coin_sign == "-" else "重"
            code_meaning = TexText(str(coin_num), "号是偏", loh, "的假币")
            code_meaning.scale(1.2).next_to(rem_poss, UP, buff=0.4)
            code_meaning[::2].set_color(color)
            self.play(
                FadeTransform(coin_num_tex.deepcopy(), code_meaning[0]),
                Write(code_meaning[1]),
                FadeTransform(coin_sign_tex.deepcopy(), code_meaning[2]),
                Write(code_meaning[-1]),
            )
        self.wait()


class RevisitHalfAndHalfGrouping(Scene):
    CONFIG = {
        "pan_balance_height": 3.5,
    }

    def setup(self):
        coins = VGroup(*[Coin(k) for k in range(1, 13)])
        coins.arrange(RIGHT, buff=COIN_BUFF)
        pb = PanBalance(height=self.pan_balance_height)
        pb.next_to(TOP, DOWN, buff=1.5)
        left_coins, right_coins = VGroup(*coins[:6]), VGroup(*coins[6:])
        pb.put_coins_on_left_pan(left_coins, arrangement=[3, 3], add_updater=True)
        pb.put_coins_on_right_pan(right_coins, arrangement=[3, 3], add_updater=True)
        self.add(pb, left_coins, right_coins)
        self.left_coins = left_coins
        self.right_coins = right_coins
        self.coins = coins
        self.pb = pb

    def construct(self):
        coins = self.coins
        pb = self.pb
        # For 6-6 grouping, it's possible to determine with only 1 weighing...
        self.wait()
        self.play(pb.get_wiggle_animation("hide"), run_time=3)
        self.wait()
        balance_poss = RealCoin("\\text{A}")
        balance_poss.next_to(self.pb, DOWN, buff=1.5)
        real_coins = VGroup(*[
            RealCoin(k + 1).move_to(coin)
            for k, coin in enumerate(coins)
        ])
        self.play(FadeIn(real_coins), FadeIn(balance_poss))
        self.wait()
        one_weighing_remark = TexText("1次称量就能完成任务", "...?")
        one_weighing_remark.next_to(balance_poss, RIGHT, buff=0.5)
        self.play(Write(one_weighing_remark[0]))
        self.wait()
        self.play(Write(one_weighing_remark[1]))
        self.wait()
        # ...but only possible. There're other possibilities.
        self.play(FadeOut(one_weighing_remark))
        self.wait()
        tilt_left_text = TexText("向左倾斜", color=YELLOW)
        balance_text = TexText("平衡", color=YELLOW)
        tilt_right_text = TexText("向右倾斜", color=YELLOW)
        pb_states_texts = VGroup(tilt_left_text, balance_text, tilt_right_text)
        pb_states_texts.next_to(pb, DOWN)
        tilt_left_text.to_edge(LEFT)
        tilt_right_text.to_edge(RIGHT)
        self.play(Write(balance_text))
        self.wait()
        self.play(
            pb.get_balance_animation("show"), FadeOut(real_coins),
            Write(tilt_left_text), Write(tilt_right_text)
        )
        self.wait()
        # If the balance tilts left, then 12 possibilities remains, rather than 1.
        self.play(pb.get_tilt_left_animation("hide"))
        self.wait()
        left_coins_copy = self.left_coins.deepcopy()
        left_coins_copy.clear_updaters()
        right_coins_copy = self.right_coins.deepcopy()
        right_coins_copy.clear_updaters()
        tilt_left_poss = VGroup(*[
            VGroup(*[HeavierCoin(str(k) + "+") for k in range(1, 7)]),
            VGroup(*[LighterCoin(str(k) + "-") for k in range(7, 13)]),
        ])
        for coin_set in tilt_left_poss:
            coin_set.arrange_in_grid(2, 3, buff=COIN_BUFF)
        tilt_left_poss.arrange(RIGHT, buff=COIN_BUFF)
        tilt_left_poss.next_to(tilt_left_text, DOWN, aligned_edge=LEFT)
        self.play(ReplacementTransform(left_coins_copy, tilt_left_poss[0]))
        self.wait()
        self.play(ReplacementTransform(right_coins_copy, tilt_left_poss[1]))
        self.wait()
        # Same goes for tilting right.
        self.play(pb.get_tilt_right_animation())
        self.wait()
        left_coins_copy = self.left_coins.deepcopy()
        left_coins_copy.clear_updaters()
        right_coins_copy = self.right_coins.deepcopy()
        right_coins_copy.clear_updaters()
        tilt_right_poss = VGroup(*[
            VGroup(*[LighterCoin(str(k) + "-") for k in range(1, 7)]),
            VGroup(*[HeavierCoin(str(k) + "+") for k in range(7, 13)]),
        ])
        for coin_set in tilt_right_poss:
            coin_set.arrange_in_grid(2, 3, buff=COIN_BUFF)
        tilt_right_poss.arrange(RIGHT, buff=COIN_BUFF)
        tilt_right_poss.next_to(tilt_right_text, DOWN, aligned_edge=RIGHT)
        self.play(
            ReplacementTransform(left_coins_copy, tilt_right_poss[0]),
            ReplacementTransform(right_coins_copy, tilt_right_poss[1]),
        )
        self.wait()
        # It's easier when balanced, but harder when tilted.
        tilt_left_group = VGroup(tilt_left_text, tilt_left_poss)
        balance_group = VGroup(balance_text, balance_poss)
        tilt_right_group = VGroup(tilt_right_text, tilt_right_poss)
        coin_poss_groups = VGroup(tilt_left_group, balance_group, tilt_right_group)
        sur_rects_group = VGroup()
        for group in coin_poss_groups:
            sur_rect = SurroundingRectangle(group)
            group.add(sur_rect)
            sur_rects_group.add(sur_rect)
        self.play(ShowCreation(sur_rects_group), lag_ratio=0.3, run_time=3)
        self.wait()
        for group in coin_poss_groups:
            group.save_state()
            group.generate_target()
            group.target.set_color(GREY).fade(0.9)
        self.play(
            pb.get_balance_animation(),
            MoveToTarget(tilt_left_group), MoveToTarget(tilt_right_group),
        )
        self.wait()
        self.play(
            pb.get_tilt_left_animation(),
            MoveToTarget(balance_group), Restore(tilt_left_group),
        )
        self.wait()
        self.play(
            pb.get_tilt_right_animation(),
            MoveToTarget(tilt_left_group), Restore(tilt_right_group),
        )
        self.wait()


class AssessingAWeighingMethod(Scene):
    CONFIG = {
        "pan_balance_height": 3.5,
    }

    def setup(self):
        # Setup coins and pan balance
        coins = VGroup(*[Coin(k) for k in range(1, 3)])
        coins.arrange(RIGHT, buff=COIN_BUFF)
        pb = PanBalance(height=self.pan_balance_height)
        pb.next_to(TOP, DOWN)
        left_coin, right_coin = coins
        pb.put_coins_on_left_pan(left_coin, add_updater=True)
        pb.put_coins_on_right_pan(right_coin, add_updater=True)
        self.add(pb, left_coin, right_coin)
        self.coins = coins
        self.pb = pb
        # Setup possible pan balance states after weighing
        tilt_left_text = TexText("向左倾斜", color=YELLOW)
        balance_text = TexText("平衡", color=YELLOW)
        tilt_right_text = TexText("向右倾斜", color=YELLOW)
        pb_states_texts = VGroup(tilt_left_text, balance_text, tilt_right_text)
        pb_states_texts.next_to(pb, DOWN, buff=0.7)
        tilt_left_text.to_edge(LEFT)
        tilt_right_text.to_edge(RIGHT)
        self.add(pb_states_texts)
        # Setup possibilities of coins
        tilt_left_poss = VGroup(
            HeavierCoin("1+"),
            LighterCoin("2-")
        )
        tilt_left_poss.arrange(DOWN, buff=COIN_BUFF)
        tilt_right_poss = VGroup(
            LighterCoin("1-"),
            HeavierCoin("2+"),
        )
        tilt_right_poss.arrange(DOWN, buff=COIN_BUFF)
        balance_poss = VGroup(*(
            [HeavierCoin(str(k) + "+") for k in range(3, 13)] +
            [RealCoin("\\text{A}")] +
            [LighterCoin(str(k) + "-") for k in range(12, 2, -1)]
        ))
        balance_poss.arrange_in_grid(3, 7, buff=COIN_BUFF)
        balance_poss.next_to(balance_text, DOWN)
        tilt_left_poss.next_to(tilt_left_text, DOWN, buff=1)
        tilt_left_poss.next_to(balance_poss, LEFT, coor_mask=[0, 1, 0])
        tilt_right_poss.next_to(tilt_right_text, DOWN, buff=1)
        tilt_right_poss.next_to(balance_poss, RIGHT, coor_mask=[0, 1, 0])
        self.add(tilt_left_poss, tilt_right_poss, balance_poss)
        # Grouping stuff
        tilt_left_group = VGroup(tilt_left_text, tilt_left_poss)
        balance_group = VGroup(balance_text, balance_poss)
        tilt_right_group = VGroup(tilt_right_text, tilt_right_poss)
        coin_poss_groups = VGroup(tilt_left_group, balance_group, tilt_right_group)
        self.coin_poss_groups = coin_poss_groups

    def construct(self):
        pb = self.pb
        coin_poss_groups = self.coin_poss_groups
        tilt_left_group, balance_group, tilt_right_group = coin_poss_groups
        # Surround each group for clearance
        sur_rects_group = VGroup()
        for group in coin_poss_groups:
            sur_rect = SurroundingRectangle(group)
            group.add(sur_rect)
            sur_rects_group.add(sur_rect)
        self.play(ShowCreation(sur_rects_group), lag_ratio=0.3, run_time=3)
        self.wait()
        # Almost no progress when it's balanced
        for group in coin_poss_groups[::2]:
            group.save_state()
            group.generate_target()
            group.target.set_color(GREY).fade(0.9)
        self.play(
            pb.get_wiggle_animation("hide"),
            MoveToTarget(tilt_left_group), MoveToTarget(tilt_right_group),
        )
        self.wait()
        # Balance is the worst outcome for this particular weighing
        self.play(Indicate(balance_group, color=None))
        self.wait()
        # Remove boundaries for possibility rearrangement
        self.play(
            pb.get_balance_animation("show"),
            Restore(tilt_left_group), Restore(tilt_right_group)
        )
        self.play(Uncreate(sur_rects_group), lag_ratio=0.3, run_time=3)
        for group in coin_poss_groups:
            sur_rect = group[-1]
            group.remove(sur_rect)
        self.wait()
        # Free existing coins, add extra coins, and set up animations
        for coin in self.coins:
            coin.clear_updaters()
        extra_coins = VGroup(*[Coin(k) for k in range(3, 9)])
        extra_coins.next_to(pb, UP, buff=0.5)
        self.add(extra_coins)
        left_coins = VGroup(*self.coins, *extra_coins[:2])
        right_coins = VGroup(*extra_coins[2:])
        left_coins.generate_target()
        right_coins.generate_target()
        pb.put_coins_on_left_pan(left_coins.target)
        pb.put_coins_on_right_pan(right_coins.target)
        # Rearrange current possibilities as even as possible
        pb_states_texts = VGroup(*[group[0] for group in coin_poss_groups])
        tilt_left_text, balance_text, tilt_right_text = pb_states_texts
        old_poss = VGroup()
        for group in coin_poss_groups:
            for mob in group[-1]:
                old_poss.add(mob)
        new_poss = VGroup(*(
            [HeavierCoin("1+")] +
            [LighterCoin("2-")] +
            [HeavierCoin(str(k) + "+") for k in range(3, 13)] +
            [RealCoin("\\text{A}")] +
            [LighterCoin(str(k) + "-") for k in range(12, 2, -1)] +
            [LighterCoin("1-")] +
            [HeavierCoin("2+")]
        ))  # Merely to match mobjects in old_poss so that the transformation looks pretty
        new_tilt_left_poss, new_balance_poss, new_tilt_right_poss = new_coin_poss = [
            VGroup(*[new_poss[ind] for ind in indices])
            for indices in (
                [0, 24, 2, 3, 20, 19, 18, 17],
                [8, 9, 10, 11, 12, 13, 14, 15, 16],
                [23, 1, 22, 21, 4, 5, 6, 7],
            )
        ]
        new_tilt_left_poss.arrange_in_grid(2, 4, buff=COIN_BUFF)
        new_tilt_left_poss.next_to(tilt_left_text, DOWN, aligned_edge=LEFT, buff=0.5)
        new_tilt_right_poss.arrange_in_grid(2, 4, buff=COIN_BUFF)
        new_tilt_right_poss.next_to(tilt_right_text, DOWN, aligned_edge=RIGHT, buff=0.5)
        new_balance_poss.arrange_in_grid(3, 3, buff=COIN_BUFF)
        new_balance_poss.next_to(balance_text, DOWN)
        self.play(
            MoveToTarget(left_coins), MoveToTarget(right_coins),
            AnimationGroup(*[
                ReplacementTransform(old_mob, new_mob, path_arc=PI / 4)
                for old_mob, new_mob in zip(old_poss, new_poss)
            ], lag_ratio=0.02, run_time=4)
        )
        pb.put_coins_on_left_pan(left_coins, add_updater=True)
        pb.put_coins_on_right_pan(right_coins, add_updater=True)
        self.wait()
        # Regrouping and adding back the surrounding rectangles
        new_tilt_left_group = VGroup(tilt_left_text, new_tilt_left_poss)
        new_balance_group = VGroup(balance_text, new_balance_poss)
        new_tilt_right_group = VGroup(tilt_right_text, new_tilt_right_poss)
        new_coin_poss_group = VGroup(new_tilt_left_group, new_balance_group, new_tilt_right_group)
        sur_rects_group = VGroup()
        for group in new_coin_poss_group:
            sur_rect = SurroundingRectangle(group)
            group.add(sur_rect)
            sur_rects_group.add(sur_rect)
        self.play(ShowCreation(sur_rects_group), lag_ratio=0.3, run_time=3)
        self.wait()
        # Even we get unlucky, there're only 9 possibilities to narrow down
        for group in new_coin_poss_group[::2]:
            group.save_state()
            group.generate_target()
            group.target.set_color(GREY).fade(0.9)
        self.play(
            pb.get_wiggle_animation("hide"),
            MoveToTarget(new_tilt_left_group), MoveToTarget(new_tilt_right_group),
        )
        self.wait()
        # Evenly distributing possibilities is the key
        pan_balance_group = VGroup(pb, left_coins, right_coins)
        self.play(Restore(new_tilt_left_group), Restore(new_tilt_right_group))
        self.play(
            pan_balance_group.animate.shift(4 * UP),
            new_coin_poss_group.animate.shift(UP),
        )
        self.wait()
        even_distrib_text = TexText("均分可能性")
        even_distrib_text.scale(1.5).shift(2.5 * UP)
        arrows = VGroup(*[
            Arrow(
                even_distrib_text.get_bounding_box_point(direction),
                group.get_bounding_box_point(-direction),
            )
            for direction, group in zip([DL, DOWN, DR], new_coin_poss_group)
        ])
        self.play(Write(even_distrib_text))
        self.play(
            AnimationGroup(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.3),
            run_time=2,
        )
        self.wait()
        for k in range(3):
            self.play(ApplyWave(even_distrib_text))
            self.wait()


class RevealTheFirstWeighing(Scene):
    CONFIG = {
        "random_seed": 5 * 7 + 0,
        "pan_balance_height": 3.5,
    }

    def setup(self):
        # Setup coin possibilities
        all_poss = VGroup(*(
            [LighterCoin(str(k) + "-") for k in range(1, 13)] +
            [HeavierCoin(str(k) + "+") for k in range(1, 13)]
        ))
        all_poss.arrange_in_grid(2, 12, buff=COIN_BUFF)
        real_poss = RealCoin("\\text{A}")
        real_poss.next_to(all_poss, RIGHT, buff=COIN_BUFF)
        all_poss.add(real_poss)
        all_poss.center().shift(1.5 * UP)
        # Setup pan balance states texts
        tilt_left_text = TexText("向左倾斜", color=YELLOW).to_edge(LEFT)
        balance_text = TexText("平衡", color=YELLOW)
        tilt_right_text = TexText("向右倾斜", color=YELLOW).to_edge(RIGHT)
        pb_states_texts = VGroup(tilt_left_text, balance_text, tilt_right_text)
        self.add(all_poss)
        self.all_poss = all_poss
        self.pb_states_texts = pb_states_texts

    def construct(self):
        self.show_problem_when_randomly_divides()
        self.show_what_properties_must_be_satisfied()
        self.reveal_first_weighing()

    def show_problem_when_randomly_divides(self):
        all_poss = self.all_poss
        # Randomly divide possibilities
        self.pb_states_texts.set_stroke(color=YELLOW)
        self.play(Write(self.pb_states_texts), run_time=3)
        self.wait()
        all_poss_copy = all_poss.deepcopy()
        tilt_left_poss, balance_poss, tilt_right_poss = \
            VGroup(*all_poss_copy[:8]), VGroup(*all_poss_copy[8:-8]), VGroup(*all_poss_copy[-8:])
        for poss, text, aligned_edge, (row, col) in zip(
            [tilt_left_poss, balance_poss, tilt_right_poss],
            self.pb_states_texts, [LEFT, ORIGIN, RIGHT], [(2, 4), (3, 3), (2, 4)]
        ):
            poss.arrange_in_grid(row, col, buff=COIN_BUFF)
            poss.next_to(text, DOWN, aligned_edge=aligned_edge)
        new_indices = list(range(25))
        random.shuffle(new_indices)
        for old_ind, new_ind in zip(range(25), new_indices):
            all_poss[old_ind].generate_target()
            all_poss[old_ind].target.move_to(all_poss_copy[new_ind])
        all_poss.save_state()
        self.play(
            AnimationGroup(*[
                MoveToTarget(all_poss[k], path_arc=PI / 3)
                for k in range(len(all_poss))
            ], lag_ratio=0.1, run_time=3)
        )
        self.wait()
        # There will be contradictions if not careful
        real_poss = all_poss[-1]
        contra_arrow = Arrow(ORIGIN, 2 * DOWN + RIGHT)
        contra_text = TexText("都是真币，但是 \\\\ 天平依然倾斜?")
        contra_arrow.next_to(real_poss, UL, buff=0.1)
        contra_text.next_to(contra_arrow, UL, buff=0.1)
        self.play(Write(contra_text), GrowArrow(contra_arrow))
        self.wait()
        self.play(FadeOut(contra_text), FadeOut(contra_arrow))
        self.wait()
        # Revert to the start
        self.play(Restore(all_poss))
        self.wait()
        self.poss_positions = list(map(lambda m: m.get_center(), all_poss_copy))

    def show_what_properties_must_be_satisfied(self):
        all_poss = self.all_poss
        poss_positions = self.poss_positions
        # "A" must correspond to the balance state
        real_poss = all_poss[-1]
        real_poss_position = poss_positions[12]
        self.play(real_poss.animate.move_to(real_poss_position))
        self.wait()
        # For coin that is not on the balance, its coin possibilities must
        # also correspond to the balanced state
        nine_to_twelve_poss = VGroup(*all_poss[8:12], *all_poss[20:24])
        position_indices = [16, 15, 14, 13, 8, 9, 10, 11]
        self.play(
            AnimationGroup(*[
                poss.animate.move_to(poss_positions[ind])
                for poss, ind in zip(nine_to_twelve_poss, position_indices)
            ], lag_ratio=0.1, run_time=2)
        )
        sur_rect = SurroundingRectangle(nine_to_twelve_poss, color=WHITE)
        remark_text = TexText("（假设9-12号硬币不在天平上）")
        remark_text.scale(0.8).next_to(sur_rect, DOWN)
        self.play(ShowCreation(sur_rect))
        self.play(Write(remark_text))
        self.wait()
        self.play(FadeOut(sur_rect), FadeOut(remark_text))
        self.wait()
        # For coin that is on the balance, its coin possibilities must
        # correspond to two tilted states, respectively.
        one_poss = VGroup(all_poss[12], all_poss[0])
        position_indices = [0, 17]
        sur_rect = SurroundingRectangle(one_poss, color=WHITE)
        remark_text = TexText("假设1号是假币， \\\\ 并且放在天平左边")
        remark_text.scale(0.8).next_to(sur_rect, UP)
        self.play(ShowCreation(sur_rect))
        self.play(Write(remark_text))
        self.wait()
        for poss, ind in zip(one_poss, position_indices):
            self.play(poss.animate.move_to(poss_positions[ind]))
            self.wait()
        self.play(FadeOut(sur_rect), FadeOut(remark_text))
        self.wait()
        # Same goes for other coins that are on the balance
        two_to_eight_poss = VGroup(*[
            VGroup(all_poss[i], all_poss[i - 12])
            for i in range(13, 20)
        ])
        position_indices = (
            [(i, i + 17) for i in range(1, 4)] +
            [(i + 17, i) for i in range(4, 8)]
        )
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    poss[0].animate.move_to(poss_positions[indices[0]]),
                    poss[1].animate.move_to(poss_positions[indices[1]])
                )
                for poss, indices in zip(two_to_eight_poss, position_indices)
            ], lag_ratio=0.1, run_time=3)
        )
        self.wait()

    def reveal_first_weighing(self):
        # Setup pan balance and coins
        pb = PanBalance(height=self.pan_balance_height)
        pb.next_to(TOP, UP).to_edge(LEFT, buff=0.4)
        left_coins = VGroup(*[Coin(k) for k in range(1, 5)])
        right_coins = VGroup(*[Coin(k) for k in range(5, 9)])
        idle_coins = VGroup(*[Coin(k) for k in range(9, 13)])
        idle_coins.arrange(RIGHT, buff=COIN_BUFF)
        pb.put_coins_on_left_pan(left_coins, add_updater=True)
        pb.put_coins_on_right_pan(right_coins, add_updater=True)
        # Setup title
        first_weighing_title = TexText("第一次称量").scale(1.25)
        ph_coins = VGroup(*[Coin() for k in range(12)])
        ph_coins.arrange_in_grid(3, 4, buff=COIN_BUFF)
        ph_coins.next_to(first_weighing_title, DOWN)
        title_group = VGroup(first_weighing_title, ph_coins)
        title_group.to_corner(UR, buff=0.4)
        idle_coins.move_to(ph_coins[-4:]).to_edge(UP, buff=-1)
        self.play(
            VGroup(pb, left_coins, right_coins).animate.to_edge(UP, buff=0.4),
            idle_coins.animate.move_to(ph_coins[-4:]),
            VGroup(self.all_poss, self.pb_states_texts).animate.shift(0.8 * DOWN),
        )
        self.wait()
        self.play(Write(first_weighing_title))
        self.wait()


class WhatIfItStaysBalanced(Scene):
    CONFIG = {
        "pan_balance_height": 3.5,
    }

    def setup(self):
        # Setup pan balance states texts
        tilt_left_text = TexText("向左倾斜", color=YELLOW).to_edge(LEFT)
        balance_text = TexText("平衡", color=YELLOW)
        tilt_right_text = TexText("向右倾斜", color=YELLOW).to_edge(RIGHT)
        pb_states_texts = VGroup(tilt_left_text, balance_text, tilt_right_text)
        pb_states_texts.shift(0.8 * DOWN)
        self.pb_states_texts = pb_states_texts
        # Setup coin possibilities
        all_poss = VGroup(*(
            [LighterCoin(str(k) + "-") for k in range(1, 13)] +
            [HeavierCoin(str(k) + "+") for k in range(1, 13)] +
            [RealCoin("\\text{A}")]
        ))
        tilt_left_poss = VGroup(*[all_poss[k] for k in (12, 13, 14, 15, 4, 5, 6, 7)])
        balance_poss = VGroup(*[all_poss[k] for k in (20, 21, 22, 23, 24, 11, 10, 9, 8)])
        tilt_right_poss = VGroup(*[all_poss[k] for k in (0, 1, 2, 3, 16, 17, 18, 19)])
        for poss, text, aligned_edge, (row, col) in zip(
            [tilt_left_poss, balance_poss, tilt_right_poss],
            pb_states_texts, [LEFT, ORIGIN, RIGHT], [(2, 4), (3, 3), (2, 4)]
        ):
            poss.arrange_in_grid(row, col, buff=COIN_BUFF)
            poss.next_to(text, DOWN, aligned_edge=aligned_edge)
        self.tilt_left_poss = tilt_left_poss
        self.balance_poss = balance_poss
        self.tilt_right_poss = tilt_right_poss
        # Setup pan balance and coins
        pb = PanBalance(height=self.pan_balance_height)
        pb.next_to(TOP, UP).to_edge(LEFT, buff=0.4)
        weighing_coins = VGroup(*[Coin(k) for k in range(1, 13)])
        left_coins = VGroup(*weighing_coins.submobjects[:4])
        right_coins = VGroup(*weighing_coins.submobjects[4:-4])
        idle_coins = VGroup(*weighing_coins.submobjects[-4:])
        idle_coins.arrange(RIGHT, buff=COIN_BUFF)
        pb.put_coins_on_left_pan(left_coins, add_updater=True)
        pb.put_coins_on_right_pan(right_coins, add_updater=True)
        self.pb = pb
        self.weighing_coins = weighing_coins
        # Setup title
        first_weighing_title = TexText("第一次称量").scale(1.25)
        ph_coins = VGroup(*[Coin() for k in range(12)])
        ph_coins.arrange_in_grid(3, 4, buff=COIN_BUFF)
        ph_coins.next_to(first_weighing_title, DOWN)
        title_group = VGroup(first_weighing_title, ph_coins)
        title_group.to_corner(UR, buff=0.4)
        self.first_weighing_title = first_weighing_title
        self.ph_coins = ph_coins
        # Setup a rectangle that covers the upper half of the screen
        cover_rect = FullScreenRectangle(fill_color=BLACK)
        cover_rect.set_height(FRAME_HEIGHT / 2, stretch=True)
        cover_rect.next_to(TOP, DOWN, buff=0)
        cover_rect.generate_target()
        cover_rect.target.fade(1)
        cover_rect.save_state()
        cover_rect.fade(1)
        self.cover_rect = cover_rect
        # Add relevant elements onto the screen
        VGroup(pb, left_coins, right_coins).to_edge(UP, buff=0.4)
        idle_coins.move_to(ph_coins[-4:])
        self.add(
            pb_states_texts, tilt_left_poss, balance_poss, tilt_right_poss,
            pb, left_coins, right_coins, idle_coins, first_weighing_title,
        )

    def construct(self):
        self.show_balanced_situation()
        self.evenly_divide_possibilities_again()
        self.show_second_weighing()
        self.one_more_weighing_is_enough()

    def show_balanced_situation(self):
        # If it's balanced, then 1-8 are real coins
        self.play(self.pb.get_wiggle_animation("hide"))
        self.wait()
        real_one_to_eight_coins = VGroup(*[
            RealCoin(k + 1).move_to(coin)
            for k, coin in enumerate(self.weighing_coins[:8])
        ])
        self.play(
            AnimationGroup(*[
                Transform(self.weighing_coins[k], real_one_to_eight_coins[k])
                for k in range(8)
            ], lag_ratio=0.1),
            FadeOut(self.tilt_left_poss), FadeOut(self.tilt_right_poss),
            run_time=2,
        )
        self.wait()
        second_weighing_title = TexText("第二次称量")
        second_weighing_title.match_height(self.first_weighing_title)
        second_weighing_title.move_to(self.first_weighing_title)
        self.play(
            self.pb.get_balance_animation("show"),
            self.weighing_coins[:4].animate.move_to(self.ph_coins[:4]),
            self.weighing_coins[4:-4].animate.move_to(self.ph_coins[4:-4]),
            ReplacementTransform(self.first_weighing_title, second_weighing_title),
            run_time=2,
        )
        self.wait()
        self.second_weighing_title = second_weighing_title

    def evenly_divide_possibilities_again(self):
        # Move balanced state possibilities up
        self.add(self.cover_rect)
        self.bring_to_front(self.balance_poss)
        self.play(
            self.balance_poss.animate.move_to(TOP / 2),
            Restore(self.cover_rect),
        )
        self.wait()
        # "A" must correspond to the balanced state
        tilt_left_text, balance_text, tilt_right_text = self.pb_states_texts
        real_poss = self.balance_poss[4]
        self.play(real_poss.animate.next_to(balance_text, DOWN))
        self.wait()
        # Add 2 more possibilities (No. 12 not on the pan balance)
        twelve_poss = self.balance_poss[3:6:2]
        self.play(twelve_poss.animate.next_to(balance_text, DOWN))
        self.wait()
        # Divide the remaining 6 into two groups
        heavier_poss = self.balance_poss[:3]
        lighter_poss = self.balance_poss[-3:]
        lighter_poss.generate_target()
        lighter_poss.target.arrange(LEFT, buff=COIN_BUFF)
        lighter_poss.target.next_to(tilt_right_text, DOWN)
        self.play(heavier_poss.animate.next_to(tilt_left_text, DOWN))
        self.wait()
        self.play(MoveToTarget(lighter_poss))
        self.wait()
        self.lighter_poss = lighter_poss
        self.heavier_poss = heavier_poss
        self.still_balance_poss = self.balance_poss[3:6]

    def show_second_weighing(self):
        # Setup the second weighing
        pb = self.pb
        self.play(MoveToTarget(self.cover_rect))
        self.wait()
        # Put 9-11 on the left pan
        nine_to_eleven_coins = VGroup(*self.weighing_coins[-4:-1])
        nine_to_eleven_coins.generate_target()
        pb.put_coins_on_left_pan(nine_to_eleven_coins.target)
        self.play(
            AnimationGroup(*[
                nine_to_eleven_coins[k].animate.move_to(nine_to_eleven_coins.target[k])
                for k in range(3)
            ], lag_ratio=0.1),
            run_time=2,
        )
        pb.put_coins_on_left_pan(nine_to_eleven_coins, add_updater=True)
        self.wait()
        # But no coins on the right pan at the time
        ghost_coins = VGroup(*[
            Coin(coin_colors=GREYS)
            for k in range(3)
        ])
        pb.put_coins_on_right_pan(ghost_coins)
        for k in range(3):
            self.play(ShowPassingFlash(ghost_coins), lag_ratio=0.01, run_time=3)
        # Use 3 of the real coins as counter weight
        curr_real_coins = self.weighing_coins[:-4]
        sur_rect = SurroundingRectangle(curr_real_coins)
        self.play(ShowCreationThenDestruction(sur_rect))
        self.wait()
        one_to_three_coins = VGroup(*self.weighing_coins[:3])
        self.play(
            AnimationGroup(*[
                one_to_three_coins[k].animate.move_to(ghost_coins[k])
                for k in range(3)
            ], lag_ratio=0.1),
            run_time=2,
        )
        pb.put_coins_on_right_pan(one_to_three_coins, add_updater=True)
        self.wait()
        self.add(nine_to_eleven_coins)
        self.add(one_to_three_coins)
        self.nine_to_eleven_coins = nine_to_eleven_coins
        self.one_to_three_coins = one_to_three_coins

    def one_more_weighing_is_enough(self):
        # Show all the possible states for the second weighing
        tilt_left_group, balance_group, tilt_right_group = [
            VGroup(a, b)
            for a, b in zip(
                [self.heavier_poss, self.still_balance_poss, self.lighter_poss],
                self.pb_states_texts,
            )
        ]
        for group in tilt_left_group, balance_group, tilt_right_group:
            group.generate_target()
            group.target.set_color(GREY).fade(0.9)
            group.save_state()
        # Tilt-left state
        self.play(
            self.pb.get_tilt_left_animation("hide"),
            MoveToTarget(balance_group), MoveToTarget(tilt_right_group),
            run_time=2,
        )
        self.wait()
        # Tilt-right state
        self.play(
            self.pb.get_tilt_right_animation("hide"),
            MoveToTarget(tilt_left_group), Restore(tilt_right_group),
            run_time=2,
        )
        self.wait()
        # Balanced state
        self.play(
            self.pb.get_balance_animation("hide"),
            MoveToTarget(tilt_right_group), Restore(balance_group),
            run_time=2,
        )
        self.wait()
        # No matter what comes, a third weighing is enough
        self.bring_to_front(self.cover_rect)
        one_more_weighing_text = TexText("第三次称量可以确定结果！")
        one_more_weighing_text.scale(1.2).move_to(TOP / 2)
        self.play(
            Write(one_more_weighing_text, run_time=2),
            Restore(self.cover_rect, run_time=1),
        )
        self.wait()


class WhatIfItTiltsLeft(WhatIfItStaysBalanced):
    def construct(self):
        self.show_tilts_left_situation()
        self.evenly_divide_possibilities_again()
        self.show_second_weighing()
        self.one_more_weighing_is_enough()
        self.show_a_possible_third_weighing()

    def show_tilts_left_situation(self):
        # If it tilts left, then one in 1-8 is a counterfeit
        self.play(self.pb.get_tilt_left_animation("hide"))
        self.wait()
        real_nine_to_twelve_coins = VGroup(*[
            RealCoin(k + 9).move_to(coin)
            for k, coin in enumerate(self.weighing_coins[-4:])
        ])
        self.play(
            AnimationGroup(*[
                Transform(self.weighing_coins[k + 8], real_nine_to_twelve_coins[k])
                for k in range(4)
            ], lag_ratio=0.1),
            FadeOut(self.balance_poss), FadeOut(self.tilt_right_poss),
            run_time=2,
        )
        self.wait()
        second_weighing_title = TexText("第二次称量")
        second_weighing_title.match_height(self.first_weighing_title)
        second_weighing_title.move_to(self.first_weighing_title)
        self.play(
            self.pb.get_balance_animation("show"),
            self.weighing_coins[:4].animate.move_to(self.ph_coins[:4]),
            self.weighing_coins[4:-4].animate.move_to(self.ph_coins[4:-4]),
            ReplacementTransform(self.first_weighing_title, second_weighing_title),
            run_time=2,
        )
        self.wait()
        self.second_weighing_title = second_weighing_title

    def evenly_divide_possibilities_again(self):
        # Move balanced state possibilities up
        self.add(self.cover_rect)
        self.bring_to_front(self.tilt_left_poss)
        self.play(
            self.tilt_left_poss.animate.move_to(TOP / 2),
            Restore(self.cover_rect),
        )
        self.wait()
        # Assume "4+" and "8-" corresponds to the balanced state
        tilt_left_text, balance_text, tilt_right_text = self.pb_states_texts
        four_and_eight_poss = VGroup(*self.tilt_left_poss[3::4])
        four_and_eight_poss.generate_target()
        four_and_eight_poss.target.arrange(RIGHT, buff=COIN_BUFF)
        four_and_eight_poss.target.next_to(balance_text, DOWN)
        self.play(MoveToTarget(four_and_eight_poss))
        self.wait()
        # Divide the remaining 6 into two groups
        heavier_poss = VGroup(*[self.tilt_left_poss[k] for k in (0, 1, 4)])
        lighter_poss = VGroup(*[self.tilt_left_poss[k] for k in (2, 5, 6)])
        heavier_poss.generate_target()
        heavier_poss.target.arrange(RIGHT, buff=COIN_BUFF)
        heavier_poss.target.next_to(tilt_left_text, DOWN)
        lighter_poss.generate_target()
        lighter_poss.target.arrange(RIGHT, buff=COIN_BUFF)
        lighter_poss.target.next_to(tilt_right_text, DOWN)
        self.play(MoveToTarget(heavier_poss))
        self.wait()
        self.play(MoveToTarget(lighter_poss))
        self.wait()
        self.lighter_poss = lighter_poss
        self.heavier_poss = heavier_poss
        self.four_and_eight_poss = four_and_eight_poss

    def show_second_weighing(self):
        # Setup the second weighing
        pb = self.pb
        self.play(MoveToTarget(self.cover_rect))
        self.wait()
        # Setup weighing coins' target
        coin_1, coin_2, coin_3, coin_5, coin_6, coin_7, coin_9, coin_10 = coins = [
            self.weighing_coins[k]
            for k in (0, 1, 2, 4, 5, 6, 8, 9)
        ]
        for coin in coins:
            coin.generate_target()
        left_coins_target = VGroup(*[coin.target for coin in (coin_1, coin_2, coin_6, coin_7)])
        right_coins_target = VGroup(*[coin.target for coin in (coin_3, coin_5, coin_9, coin_10)])
        pb.put_coins_on_left_pan(left_coins_target)
        pb.put_coins_on_right_pan(right_coins_target)
        # Put 1-2 on the left pan
        self.play(
            AnimationGroup(*[
                MoveToTarget(coin) for coin in (coin_1, coin_2)
            ], lag_ratio=0.3, run_time=1)
        )
        self.wait()
        # Put 5 on the right pan
        self.play(MoveToTarget(coin_5))
        self.wait()
        # Put 3 on the right pan
        self.play(MoveToTarget(coin_3))
        self.wait()
        # Put 6-7 on the left pan
        self.play(
            AnimationGroup(*[
                MoveToTarget(coin) for coin in (coin_6, coin_7)
            ], lag_ratio=0.3, run_time=1)
        )
        self.wait()
        # But not enough coins on the right pan at the time
        ghost_coins = VGroup(*[
            Coin(coin_colors=GREYS).move_to(coin.target)
            for coin in (coin_9, coin_10)
        ])
        self.play(ShowPassingFlash(ghost_coins), lag_ratio=0.01, run_time=3)
        self.wait()
        # Use 2 of the real coins as counter weight
        curr_real_coins = self.weighing_coins[-4:]
        sur_rect = SurroundingRectangle(curr_real_coins)
        self.play(ShowCreationThenDestruction(sur_rect))
        self.wait()
        self.play(
            AnimationGroup(*[
                MoveToTarget(coin) for coin in (coin_9, coin_10)
            ], lag_ratio=0.3, run_time=1)
        )
        self.wait()
        # This is the second weighing
        left_coins = VGroup(coin_1, coin_2, coin_6, coin_7)
        right_coins = VGroup(coin_3, coin_5, coin_9, coin_10)
        pb.put_coins_on_left_pan(left_coins, add_updater=True)
        pb.put_coins_on_right_pan(right_coins, add_updater=True)
        self.wait()
        self.add(left_coins)
        self.add(right_coins)
        self.left_coins = left_coins
        self.right_coins = right_coins

    def one_more_weighing_is_enough(self):
        # Show all the possible states for the second weighing
        tilt_left_group, balance_group, tilt_right_group = [
            VGroup(a, b)
            for a, b in zip(
                [self.heavier_poss, self.four_and_eight_poss, self.lighter_poss],
                self.pb_states_texts,
            )
        ]
        for group in tilt_left_group, balance_group, tilt_right_group:
            group.generate_target()
            group.target.set_color(GREY).fade(0.9)
            group.save_state()
        # Tilt-left state
        self.play(
            self.pb.get_tilt_left_animation("hide"),
            MoveToTarget(balance_group), MoveToTarget(tilt_right_group),
            run_time=2,
        )
        self.wait()
        # Balanced state
        self.play(
            self.pb.get_balance_animation("hide"),
            MoveToTarget(tilt_left_group), Restore(balance_group),
            run_time=2,
        )
        self.wait()
        # Tilt-right state
        self.play(
            self.pb.get_tilt_right_animation("hide"),
            MoveToTarget(balance_group), Restore(tilt_right_group),
            run_time=2,
        )
        self.wait()
        self.left_coins.clear_updaters()
        self.right_coins.clear_updaters()
        # Same as before, a third weighing is enough
        self.bring_to_front(self.cover_rect)
        one_more_weighing_text = TexText("第三次称量可以确定结果！")
        one_more_weighing_text.scale(1.2).move_to(TOP / 2)
        self.play(
            FadeIn(one_more_weighing_text, run_time=1),
            Restore(self.cover_rect, run_time=1),
        )
        self.wait()
        self.play(
            FadeOut(one_more_weighing_text),
            MoveToTarget(self.cover_rect),
            Restore(tilt_left_group), Restore(balance_group),
        )
        self.wait()

    def show_a_possible_third_weighing(self):
        # Some possibilities are eliminated if it tilts right.
        additional_real_coins = VGroup(*[
            RealCoin(ind + 1).move_to(self.weighing_coins[ind])
            for ind in (0, 1, 3, 4, 7)
        ])
        self.play(
            AnimationGroup(*[
                Transform(self.weighing_coins[ind], real_coin)
                for ind, real_coin in zip([0, 1, 3, 4, 7], additional_real_coins)
            ], lag_ratio=0.1),
            FadeOut(self.heavier_poss), FadeOut(self.four_and_eight_poss),
            run_time=2,
        )
        self.wait()
        third_weighing_title = TexText("第三次称量")
        third_weighing_title.match_height(self.second_weighing_title)
        third_weighing_title.move_to(self.second_weighing_title)
        self.play(
            self.pb.get_balance_animation("show"),
            AnimationGroup(*[
                self.weighing_coins[k].animate.move_to(self.ph_coins[k])
                for k in (0, 1, 2, 4, 5, 6, 8, 9)
            ]),
            ReplacementTransform(self.second_weighing_title, third_weighing_title),
            run_time=2,
        )
        self.wait()
        self.third_weighing_title = third_weighing_title
        # Again, another division
        tilt_left_text, balance_text, tilt_right_text = self.pb_states_texts
        three_poss, six_poss, seven_poss = self.lighter_poss
        for poss, text in zip([six_poss, three_poss, seven_poss], self.pb_states_texts):
            poss.generate_target()
            poss.target.next_to(text, DOWN)
        self.play(
            AnimationGroup(*[
                MoveToTarget(poss, path_arc=-PI / 3)
                for poss in self.lighter_poss
            ], lag_ratio=0.1, run_time=2)
        )
        self.wait()
        # Setup the third weighing
        pb = self.pb
        coin_6 = VGroup(self.weighing_coins[5])
        coin_7 = VGroup(self.weighing_coins[6])
        coin_6.generate_target()
        pb.put_coins_on_right_pan(coin_6.target)
        coin_7.generate_target()
        pb.put_coins_on_left_pan(coin_7.target)
        self.play(
            AnimationGroup(*[
                MoveToTarget(coin)
                for coin in (coin_6, coin_7)
            ], lag_ratio=0.1, run_time=1)
        )
        self.add(coin_6, coin_7)
        pb.put_coins_on_right_pan(coin_6, add_updater=True)
        pb.put_coins_on_left_pan(coin_7, add_updater=True)
        self.wait()
        # Assume it tilts left, then 6 is the lighter counterfeit coin
        self.play(
            pb.get_tilt_left_animation("hide"),
            FadeOut(three_poss), FadeOut(seven_poss),
        )
        self.wait()
        # Only one possibility remains
        final_real_coins = VGroup(*[
            RealCoin(ind + 1).move_to(self.weighing_coins[ind])
            for ind in (2, 6)
        ])
        lighter_coin = VGroup(LighterCoin("6"))
        lighter_coin.move_to(coin_6)
        self.play(
            AnimationGroup(*[
                Transform(self.weighing_coins[ind], real_coin)
                for ind, real_coin in zip([2, 6], final_real_coins)
            ], lag_ratio=0.1),
            Transform(coin_6, lighter_coin),
            run_time=2,
        )
        self.wait()
        # Now reveal the identity of the counterfeit
        coin_6.clear_updaters()
        other_mobs = Group(*self.mobjects)
        other_mobs.remove(coin_6)
        self.add(coin_6)
        coin_6.generate_target()
        coin_6.target.scale(2).center()
        self.play(
            other_mobs.animate.shift(FRAME_HEIGHT * UP),
            MoveToTarget(coin_6),
        )
        self.wait()
        identity = TexText("6", "号是偏", "轻", "的假币")
        identity[::2].set_color(RED_A)
        identity.scale(1.5)
        identity.next_to(coin_6, UP, buff=0.5)
        self.play(Write(identity))
        self.wait()
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait()


class ABriefSummary(Scene):
    def construct(self):
        # Even though this graph will quickly get cluttered...
        first_branch = PossibilityBranch(width=4.5, height=2.5)
        second_root_radius = first_branch.get_branch_radius()
        second_branches = VGroup(*[
            PossibilityBranch(
                width=1.5, height=1.4, root_radius=second_root_radius,
                circle_stroke_width=2,
                arrow_thickness=0.03, text_scaling_factor=0.7,
            )
            .graft_in(circle)
            for circle in first_branch.get_branch_circles()
        ])
        third_root_radius = second_branches[0].get_branch_radius()
        third_branches = VGroup(*[
            PossibilityBranch(
                width=0.5, height=0.8, root_radius=third_root_radius,
                circle_stroke_width=1,
                arrow_thickness=0.015, text_scaling_factor=0.4,
            )
            .graft_in(circle)
            for second_branch in second_branches
            for circle in second_branch.get_branch_circles()
        ])
        branches_group = VGroup(first_branch, second_branches, third_branches)
        branches_group.to_edge(UP)
        for branch_set in branches_group:
            self.play(FadeIn(branch_set), lag_ratio=0.02, run_time=1)
            self.wait(0.5)
        # ... there's only one key point: divide the possibilities evenly
        keypoint_text = TexText("均分可能性", color=YELLOW)
        keypoint_text.scale(1.2)
        keypoint_text.next_to(branches_group, DOWN, buff=0.5)
        self.play(Write(keypoint_text))
        self.wait()
        # Design weighings according to this principle
        arrow = Arrow(ORIGIN, 2 * RIGHT, fill_color=YELLOW)
        design_text = TexText("设计称量方式", color=YELLOW)
        design_text.scale(1.2)
        arrow.move_to(keypoint_text)
        design_text.next_to(arrow, RIGHT)
        keypoint_text.generate_target()
        keypoint_text.target.next_to(arrow, LEFT)
        self.play(MoveToTarget(keypoint_text))
        self.play(GrowArrow(arrow), Write(design_text))
        self.wait()
        # For more coins, it's necessary to add another layer
        self.play(FadeOut(VGroup(keypoint_text, arrow, design_text)))
        self.wait()
        fourth_root_radius = third_branches[0].get_branch_radius()
        fourth_branches = VGroup(*[
            PossibilityBranch(
                width=0.16, height=0.4, root_radius=fourth_root_radius,
                branch_radius=0.07,
                circle_stroke_width=0.5,
                arrow_thickness=0.0075, text_scaling_factor=0.1,
            )
            .graft_in(circle)
            for third_branch in third_branches
            for circle in third_branch.get_branch_circles()
        ])
        self.play(FadeIn(fourth_branches), lag_ratio=0.02, run_time=3)
        self.wait()
        branches_group.add(fourth_branches)
        self.play(FadeOut(branches_group))
        self.wait()


class SameAppliesToMostSimilarPuzzles(Scene):
    def construct(self):
        # Pan balance
        pb = PanBalance()
        pb.to_corner(UL, buff=0.3)
        # Instructions
        counterfeit_text = TexText("至多一枚假币", color=YELLOW)
        weighing_text = TexText("3次称量机会", color=YELLOW)
        questions = BulletedList("是否有假币？", "假币的编号？", "是轻还是重？")
        questions.arrange(DOWN, buff=0.2)
        text_group = VGroup(counterfeit_text, weighing_text, questions)
        text_group.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        text_group.scale(0.8)
        text_group.next_to(pb, RIGHT, aligned_edge=DOWN, buff=0.6)
        # Coins
        new_coins = VGroup(*[Coin(k) for k in range(1, 40)])
        new_coins.arrange_in_grid(3, 13, buff=COIN_BUFF)
        new_coins.to_edge(DOWN)
        old_coins = VGroup(*[Coin(k) for k in range(1, 13)])
        old_coins.arrange(RIGHT, buff=COIN_BUFF)
        old_coins.move_to(new_coins)
        # Setup the old problem
        self.add(pb, text_group, old_coins)
        self.wait()
        # Transform to the new problem
        new_weighing_text = TexText("4次称量机会", color=YELLOW)
        new_weighing_text.match_height(weighing_text)
        new_weighing_text.move_to(weighing_text)
        self.play(
            AnimationGroup(*[
                Transform(old_coin, new_coin)
                for old_coin, new_coin in zip(old_coins[:12], new_coins[:12])
            ], lag_ratio=0.02, run_time=1),
            AnimationGroup(*[
                FadeIn(new_coin, direction=DOWN)
                for new_coin in new_coins[12:]
            ], lag_ratio=0.02, run_time=3),
            FocusOn(new_weighing_text[0][0], run_time=1),
            ReplacementTransform(weighing_text, new_weighing_text, run_time=1),
        )
        self.wait()


class ABetterApproachTeaser(Scene):
    CONFIG = {
        "coin_config": {
            "radius": 0.25,
            "coin_stroke_width": 1,
        },
    }

    def construct(self):
        # Setup test matrix
        test_array = [
            list(map(str, [0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1])),
            list(map(str, [0, -1, 1, 1, 0, 0, 0, -1, 1, 1, -1, -1])),
            list(map(str, [-1, 0, 1, -1, 0, -1, 1, 0, 1, -1, 0, 1])),
        ]
        test_matrix = Matrix(test_array)
        text_H = Tex("H=")
        matrix_group = VGroup(text_H, test_matrix)
        matrix_group.arrange(RIGHT)
        matrix_group.set_width(FRAME_WIDTH - 2)
        self.play(FadeIn(matrix_group, lag_ratio=0.01, run_time=1.5))
        self.wait()
        self.play(matrix_group.animate.to_edge(UP))
        self.wait()
        # Setup pan balances and coins
        pb_1, pb_2, pb_3 = pbs = VGroup(*[
            PanBalance(height=1.6)
            for k in range(3)
        ])
        pbs.arrange(RIGHT, buff=0.5)
        coin_sets = VGroup(*[
            VGroup(*[
                Coin(k, **self.coin_config)
                for k in coin_numbers
            ])
            for coin_numbers in [
                (5, 6, 7, 8), (9, 10, 11, 12),  # first weighing
                (2, 8, 11, 12), (3, 4, 9, 10),  # second weighing
                (1, 4, 6, 10), (3, 7, 9, 12),  # third weighing
            ]
        ])
        for pb, coin_set in zip(pbs, coin_sets[::2]):
            pb.add(coin_set)
            pb.put_coins_on_left_pan(coin_set, arrangement=[2, 2], add_updater=True)
        for pb, coin_set in zip(pbs, coin_sets[1::2]):
            pb.add(coin_set)
            pb.put_coins_on_right_pan(coin_set, arrangement=[2, 2], add_updater=True)
        weighing_texts = VGroup(*[
            TexText(text, color=YELLOW).scale(0.75).next_to(pb, UP, buff=0.5)
            for text, pb in zip(["第一次称量", "第二次称量", "第三次称量"], pbs)
        ])
        pb_group = VGroup(weighing_texts, pbs)
        pb_group.next_to(matrix_group, DOWN, buff=0.8)
        self.play(FadeIn(pb_group), run_time=2)
        self.wait()
        # Show a possible configuration and its answer
        self.play(
            pb_1.get_tilt_right_animation("hide"),
            pb_2.get_wiggle_animation("hide"),
            pb_3.get_tilt_right_animation("hide"),
            run_time=1,
        )
        self.wait()
        lighter_coin_6 = LighterCoin("6")
        lighter_coin_6.generate_target()
        lighter_coin_6.move_to(pb_group).fade(1).scale(0)
        lighter_coin_6.target.scale(1.25).next_to(pb_group, DOWN, buff=0.4)
        self.play(MoveToTarget(lighter_coin_6))
        self.wait()


class ThumbnailPart1(IntroTo12CoinsPuzzle):
    pass

#####
# Part 2 Scenes


class RecapOnLastVideo(Scene):
    def construct(self):
        self.recap_problem()
        self.can_we_avoid_these_branches()

    def recap_problem(self):
        coins = VGroup(*[Coin(k) for k in range(1, 13)])
        coins.arrange(RIGHT, buff=COIN_BUFF)
        pb = PanBalance()
        pb.to_corner(UL, buff=0.4)
        pb.shift(0.5 * DOWN)
        weighing_text = TexText("3次称量机会", color=YELLOW)
        counterfeit_text = TexText("至多1枚假币", color=YELLOW)
        questions = BulletedList("是否有假币？", "假币的编号？", "是轻还是重？")
        questions.arrange(DOWN, buff=0.2)
        text_group = VGroup(weighing_text, counterfeit_text, questions)
        text_group.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        text_group.scale(0.8)
        text_group.next_to(pb, RIGHT, aligned_edge=DOWN, buff=0.6)
        # Show coins, pan balance and objectives
        self.play(
            AnimationGroup(*[
                GrowFromCenter(coin)
                for coin in coins
            ], lag_ratio=0.1, run_time=2)
        )
        self.wait()
        pb.shift(5 * UP)
        self.play(
            pb.animate.shift(5 * DOWN),
            coins.animate.shift(2 * DOWN),
        )
        self.wait()
        self.play(Write(weighing_text))
        self.wait()
        self.play(Write(counterfeit_text))
        self.wait()
        self.play(Write(questions))
        self.wait()

    def can_we_avoid_these_branches(self):
        # Clear everything for the next scene
        self.remove(*self.mobjects)
        avoid_text = TexText("能否避免这种“分支结构”？", color=YELLOW)
        avoid_text.scale(1.2)
        avoid_text.to_edge(DOWN, buff=0.5)
        self.play(Write(avoid_text))
        self.wait()


class HowToDesignAWorkaround(Scene):
    def construct(self):
        proc_texts = VGroup(*[
            TexText(text)
            for text in [
                "第一次称量", "第一次称量结果",
                "第二次称量", "第二次称量结果",
                "第三次称量", "第三次称量结果",
            ]
        ])
        proc_texts.arrange(DOWN, buff=1.2)
        proc_texts.to_edge(UP)
        arrows = VGroup(*[
            Arrow(proc_texts[k].get_bottom(), proc_texts[k + 1].get_top(), color=WHITE)
            for k in range(len(proc_texts) - 1)
        ])
        arrows[1::2].set_color(YELLOW)
        infl_texts = VGroup(*[
            TexText("影响", color=YELLOW).scale(0.8).next_to(arrow)
            for arrow in arrows[1::2]
        ])
        infl_groups = VGroup(*[
            VGroup(arrow, infl_text)
            for arrow, infl_text in zip(arrows[1::2], infl_texts)
        ])
        proc_groups = VGroup(*[
            VGroup(weighing_text, arrow, result_text)
            for weighing_text, arrow, result_text in zip(
                proc_texts[::2], arrows[::2], proc_texts[1::2]
            )
        ])
        # Setup
        self.add(proc_texts, arrows, infl_texts)
        self.wait()
        # It's better if previous results don't influence next weighings
        crosses = VGroup(*[Cross(infl_group) for infl_group in infl_groups])
        crosses.set_stroke(width=10)
        self.play(ShowCreation(crosses, lag_ratio=0.2), run_time=2)
        self.wait()
        self.play(FadeOut(VGroup(infl_groups, crosses)), run_time=2)
        self.wait()
        # The idea is to integrate all weighing outcomes and analyse result
        proc_groups.generate_target()
        proc_groups.target.arrange(RIGHT, buff=0.8)
        proc_groups.target.shift(2 * UP)
        self.play(MoveToTarget(proc_groups))
        self.wait()
        final_lines = VGroup(*[
            Line(group.get_bottom() + 0.2 * DOWN, 0.5 * DOWN, stroke_width=5)
            for group in proc_groups
        ])
        final_arrow = Arrow(0.5 * DOWN, 1.5 * DOWN, fill_color=WHITE, buff=0)
        result_text = TexText("与假币有关的信息", color=YELLOW)
        result_text.next_to(final_arrow, DOWN)
        self.play(ShowCreation(final_lines))
        self.play(GrowArrow(final_arrow))
        self.wait()
        self.play(Write(result_text))
        self.wait()
        self.play(
            Flash(result_text.get_left() + 0.4 * LEFT, color=GREEN, line_length=0.15),
            Flash(result_text.get_right() + 0.4 * RIGHT, color=GREEN, line_length=0.15),
        )
        self.wait()
        # Indicate important stuff
        weighing_rects = VGroup(*[
            SurroundingRectangle(text) for text in proc_texts[::2]
        ])
        result_rect = SurroundingRectangle(VGroup(proc_texts[1::2], result_text))
        self.play(
            ShowCreationThenDestruction(weighing_rects),
            lag_ratio=0.2, run_time=3,
        )
        self.wait()
        self.play(ShowCreation(result_rect), run_time=2)
        self.wait()
        # Now focus on a single weighing, and transform it into an expression
        curr_mobs = Group(*self.mobjects)
        pb = PanBalance()
        pb.to_corner(UL, buff=0.5)
        coins = VGroup(*[Coin(k) for k in range(1, 13)])
        coins.arrange_in_grid(3, 4, buff=COIN_BUFF)
        coins.next_to(pb, RIGHT, buff=0.5)
        pb.put_coins_on_left_pan(coins[:4])
        pb.put_coins_on_right_pan(coins[4:-4])
        pb_mobs = VGroup(pb, coins)
        pb_mobs.shift(FRAME_HEIGHT * DOWN)
        self.play(
            curr_mobs.animate.shift(FRAME_HEIGHT * UP),
            pb_mobs.animate.shift(FRAME_HEIGHT * UP),
            run_time=2,
        )
        self.add(pb, coins)
        self.wait()


class PanBalanceToMatrixAbstraction(Scene):
    def setup(self):
        # Setup pan balance to match with the last scene
        pb = PanBalance()
        pb.to_corner(UL, buff=0.5)
        coins = VGroup(*[Coin(k) for k in range(1, 13)])
        coins.arrange_in_grid(3, 4, buff=COIN_BUFF)
        coins.next_to(pb, RIGHT, buff=0.5)
        coins.save_state()
        left_coins, right_coins, idle_coins = coins[:4], coins[4:-4], coins[-4:]
        pb.put_coins_on_left_pan(left_coins, add_updater=True)
        pb.put_coins_on_right_pan(right_coins, add_updater=True)
        self.add(pb, left_coins, right_coins, idle_coins)
        self.pb = pb
        self.left_coins = left_coins
        self.right_coins = right_coins
        self.idle_coins = idle_coins
        self.coins = coins

    def construct(self):
        self.show_what_pan_balance_does()
        self.replace_pan_balance_with_expression()
        self.define_weighing_matrix_and_result_vector()
        self.define_standard_vector_and_bias_vector()
        self.problem_recap()

    def show_what_pan_balance_does(self):
        # Pan balance can measure torque or mass
        left_torque, right_torque = torque_texts = VGroup(
            TexText("左边的", "力矩"), TexText("右边的", "力矩")
        )
        left_mass, right_mass = mass_texts = VGroup(
            TexText("左边的", "质量"), TexText("右边的", "质量")
        )
        torque_mass_texts = VGroup(left_torque, right_torque, left_mass, right_mass)
        for k, text in enumerate(torque_mass_texts):
            text.set_color(YELLOW)
            pan = self.pb.get_left_pan() if k % 2 == 0 else self.pb.get_right_pan()
            text.next_to(pan, DOWN, buff=3.5)
        self.play(Write(left_torque), Write(right_torque))
        self.wait()
        self.play(
            FadeTransform(left_torque[1], left_mass[1]),
            FadeTransform(right_torque[1], right_mass[1])
        )
        self.remove(torque_texts)
        self.add(mass_texts)
        self.wait()
        # It tilts according to the masses on both sides
        greater_sign = Tex(">", color=YELLOW)
        greater_sign.scale(1.5)
        greater_sign.move_to(mass_texts.get_center())
        self.play(
            self.pb.get_tilt_left_animation("hide"),
            GrowFromCenter(greater_sign),
            run_time=2,
        )
        self.wait()
        self.play(
            self.pb.get_balance_animation("show"),
            FadeOut(greater_sign),
            run_time=2,
        )
        self.wait()
        self.mass_texts = mass_texts

    def replace_pan_balance_with_expression(self):
        left_side = Tex("m_1", "+", "m_2", "+", "m_3", "+", "m_4")
        right_side = Tex("m_5", "+", "m_6", "+", "m_7", "+", "m_8")
        left_side.move_to(self.mass_texts[0])
        right_side.move_to(self.mass_texts[1])
        left_side.generate_target()
        right_side.generate_target()
        left_side.fade(1).match_width(self.left_coins).move_to(self.left_coins)
        right_side.fade(1).match_width(self.right_coins).move_to(self.right_coins)
        self.play(
            MoveToTarget(left_side, lag_ratio=0.1),
            MoveToTarget(right_side, lag_ratio=0.1),
            FadeOut(self.mass_texts),
            run_time=2,
        )
        self.wait()
        # Show possible outcomes and the corresponding expressions
        signs = VGroup(*[Tex(symbol) for symbol in (">", "=", "<", "?")])
        for sign in signs:
            sign.scale(1.2)
            sign.move_to(self.mass_texts.get_center())
        self.play(
            self.pb.get_tilt_left_animation("hide"),
            GrowFromCenter(signs[0]),
            run_time=2,
        )
        self.wait()
        self.play(
            self.pb.get_balance_animation("hide"),
            Transform(signs[0], signs[1]),
            run_time=2,
        )
        self.wait()
        self.play(
            self.pb.get_tilt_right_animation("hide"),
            Transform(signs[0], signs[2]),
            run_time=2,
        )
        self.wait()
        # Reset
        self.play(
            self.pb.get_balance_animation("show"),
            ReplacementTransform(signs[0], signs[-1]),
            run_time=2,
        )
        # Setup the whole expression
        question_mark = signs[-1]
        final_expr = Tex(
            "&(-1)", "\\cdot ", "m_{1}", "+", "(-1)", "\\cdot ", "m_{2}", "+",
            "(-1)", "\\cdot ", "m_{3}", "+", "(-1)", "\\cdot ", "m_{4}", "+\\\\",
            "&1", "\\cdot ", "m_{5}", "+", "1", "\\cdot ", "m_{6}", "+",
            "1", "\\cdot ", "m_{7}", "+", "1", "\\cdot ", "m_{8}", "+\\\\",
            "&0", "\\cdot ", "m_{9}", "+", "0", "\\cdot ", "m_{10}", "+",
            "0", "\\cdot ", "m_{11}", "+", "0", "\\cdot ", "m_{12}",
            alignment="",
        )
        final_expr.scale(0.9)
        final_expr.arrange_in_grid(3, 16, h_buff=-0.2)
        zero = Tex("0")
        zero.generate_target()
        zero.scale(0).fade(0).next_to(left_side.get_right(), LEFT, buff=0)
        question_mark.generate_target()
        expr_group = VGroup(zero.target, question_mark.target, final_expr)
        expr_group.arrange(RIGHT, buff=0.7)
        expr_group.center().to_edge(DOWN, buff=0.8)
        for k in range(32):
            if (k < 16) and (k % 4 < 2):  # if it's a coefficient of left side
                coeff_tex = final_expr[k]
                coeff_tex.generate_target()
                target_ind = k // 4 * 2
                target_mob = left_side[target_ind]
                coeff_tex.scale(0).move_to(target_mob.get_left())
            elif (k >= 16) and (k % 4 < 2):  # if it's a coefficient of right side
                coeff_tex = final_expr[k]
                coeff_tex.generate_target()
                target_ind = (k - 16) // 4 * 2
                target_mob = right_side[target_ind]
                coeff_tex.scale(0).move_to(target_mob.get_left())
        # Move all terms on one side
        self.play(
            # left side coefficients
            AnimationGroup(*[
                MoveToTarget(final_expr[k], path_arc=-PI / 3)
                for k in range(16) if (k % 4 < 2)
            ]),
            # right side coefficients
            AnimationGroup(*[
                MoveToTarget(final_expr[k])
                for k in range(16, 32) if (k % 4 < 2)
            ]),
            # left side variables
            AnimationGroup(*[
                Transform(
                    left_side[k], final_expr[(k // 2) * 4 + 2 + (k % 2)],
                    path_arc=-PI / 3,
                )
                for k in range(len(left_side))
            ]),
            # right side variables
            AnimationGroup(*[
                Transform(
                    right_side[k], final_expr[(k // 2) * 4 + 18 + (k % 2)],
                )
                for k in range(len(right_side))
            ]),
            # question mark and zero
            MoveToTarget(question_mark),
            MoveToTarget(zero),
            run_time=3,
        )
        self.play(Write(final_expr[15]))  # Add a plus sign to the end of the first line
        self.wait()
        # For the rest coins, add a term with coefficient 0
        idle_texts = Tex("m_9", "m_{10}", "m_{11}", "m_{12}")
        for text, target_mob, coin in zip(idle_texts, final_expr[-13::4], self.coins[-4:]):
            text.become(target_mob)
            text.generate_target()
            text.scale(0).fade(0).move_to(coin)
        self.play(
            AnimationGroup(*[
                MoveToTarget(text)
                for text in idle_texts
            ]),
            run_time=2,
        )
        self.wait()
        zero_coefficients = VGroup(final_expr[-15::4], final_expr[-14::4])
        self.play(
            FadeIn(zero_coefficients),
            Write(final_expr[31::4]),  # Add plus signs to finish off the expression
        )
        self.remove(left_side, right_side, idle_texts)
        self.add(final_expr)
        self.wait()
        # Rearrange again to make the 0 lands on the right side
        expr_group = VGroup(final_expr, question_mark, zero)
        expr_group.generate_target()
        expr_group.target.arrange(RIGHT, buff=0.7)
        expr_group.target.center().to_edge(DOWN, buff=0.8)
        self.play(
            MoveToTarget(expr_group, path_arc=-PI / 3),
            run_time=2,
        )
        self.wait()
        # Highlight relevant terms
        mass_texts = final_expr[2::4]
        coeff_texts = final_expr[::4]
        highlight_colors = [RED, GREEN, BLUE]
        self.play(
            AnimationGroup(*[
                Indicate(text, color=YELLOW)
                for text in mass_texts
            ], lag_ratio=0.05),
            run_time=2,
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                Indicate(text, color=highlight_colors[k // 4])
                for k, text in enumerate(coeff_texts)
            ], lag_ratio=0.05),
            run_time=2,
        )
        self.wait()
        # Show what the coefficients represent
        left_side_expr = final_expr[:15]
        right_side_expr = final_expr[16:31]
        idle_expr = final_expr[32:]
        exprs = [left_side_expr, right_side_expr, idle_expr]
        coin_sets = [self.left_coins, self.right_coins, self.idle_coins]
        for expr, coins, color in zip(exprs, coin_sets, highlight_colors):
            expr_sur_rect = SurroundingRectangle(expr, color=color, buff=0.15)
            coins_sur_rect = SurroundingRectangle(coins, color=color, buff=0.15)
            self.play(
                Indicate(expr, color=color, scale_factor=1),
                ShowCreationThenDestruction(expr_sur_rect),
                ShowCreationThenDestruction(coins_sur_rect),
                run_time=2,
            )
            self.wait()
        # An example of changing coefficients
        for k in range(3):
            self.play(
                CyclicReplace(*[expr[0] for expr in exprs]),
                CyclicReplace(*[coin_set[0] for coin_set in coin_sets[::-1]]),
                run_time=2,
            )
            self.wait()
        # Refresh coins and show possible outcomes
        self.add(self.left_coins, self.right_coins, self.idle_coins)
        signs = VGroup(*[Tex(symbol) for symbol in ("=", "<", ">")])
        for sign in signs:
            sign.set_color(YELLOW)
            sign.scale(2)
            sign.move_to(question_mark)
        question_mark.save_state()
        pb_anims = [
            anim_getter("hide", run_time=2)
            for anim_getter in (
                self.pb.get_wiggle_animation,
                self.pb.get_tilt_left_animation,
                self.pb.get_tilt_right_animation,
            )
        ]
        for sign, pb_anim in zip(signs, pb_anims):
            self.play(
                pb_anim,
                Transform(question_mark, sign, run_time=1),
            )
            self.wait()
            self.play(
                self.pb.get_balance_animation("show"),
                Restore(question_mark),
            )
            self.wait()
        # It is transformed into an expression
        coeff_texts = final_expr[::4]
        highlight_colors = [RED, GREEN, BLUE]
        self.play(
            AnimationGroup(*[
                Indicate(text, color=highlight_colors[k // 4])
                for k, text in enumerate(coeff_texts)
            ], lag_ratio=0.05),
            run_time=2,
        )
        self.wait()
        self.play(Indicate(question_mark, scale_factor=2), run_time=2)
        self.wait()
        # Now we can put pan balance aside and focus on this expression
        pb_group = VGroup(self.pb, self.coins)
        expr_group = VGroup(final_expr, question_mark, zero)
        self.play(
            pb_group.animate.shift(FRAME_HEIGHT * UP),
            expr_group.animate.to_edge(UP, buff=0.6),
            run_time=2,
        )
        self.wait()
        self.final_expr = final_expr
        self.question_mark = question_mark
        self.zero = zero

    def define_weighing_matrix_and_result_vector(self):
        # Organize LHS into a vector product
        others = VGroup(self.question_mark, self.zero)
        others.save_state()
        final_expr = self.final_expr
        final_expr.save_state()
        expr_sur_rect = SurroundingRectangle(final_expr)
        coeff_texts = final_expr[::4]
        mass_texts = final_expr[2::4]
        self.play(
            ShowCreationThenDestruction(expr_sur_rect),
            others.animate.shift(4 * RIGHT),
            run_time=2,
        )
        self.play(
            ApplyMethod(mass_texts.set_color, BLUE),
            ApplyMethod(coeff_texts.set_color, YELLOW),
            lag_ratio=0.1, run_time=2,
        )
        self.wait()
        weighing_vector_1 = HVector(*(["-1"] * 4 + ["1"] * 4 + ["0"] * 4))
        mass_vector = VVector(*["m_{" + str(k) + "}" for k in range(1, 13)])
        vectors = VGroup(weighing_vector_1, mass_vector)
        vectors.arrange(RIGHT).center().to_edge(RIGHT)
        self.play(
            AnimationGroup(*[
                ReplacementTransform(
                    text if k >= 4 else text[1:-1],
                    weighing_vector_1.get_column(k)
                )
                for k, text in enumerate(coeff_texts)
            ], lag_ratio=0.1, run_time=2),
            AnimationGroup(*[
                FadeOut(VGroup(text[0], text[-1]))
                for text in coeff_texts[:4]
            ], lag_ratio=0.2, run_time=2),
            Write(weighing_vector_1.get_brackets(), run_time=1),
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                ReplacementTransform(text, mass_vector.get_row(k))
                for k, text in enumerate(mass_texts)
            ], lag_ratio=0.1, run_time=2),
            AnimationGroup(*[
                FadeOut(final_expr[1::2])
            ], lag_ratio=0.2, run_time=2),
            Write(mass_vector.get_brackets(), run_time=1),
        )
        self.wait()
        # For multiple weighings, the row vectors can be integrated
        weighing_vector_2 = HVector(*(["1"] * 6 + ["-1"] * 6))
        weighing_vector_3 = HVector(*(["0"] * 8 + ["1"] * 2 + ["-1"] * 2))
        weighing_vectors = [weighing_vector_1, weighing_vector_2, weighing_vector_3]
        extra_vectors = VGroup(weighing_vector_2, weighing_vector_3)
        extra_vectors.arrange(DOWN)
        extra_vectors.next_to(weighing_vector_1, DOWN, buff=1)
        self.play(FadeIn(extra_vectors, shift=UP))
        self.wait()
        weighing_matrix = WeighingMatrix([
            vector.entries
            for vector in weighing_vectors
        ])
        weighing_matrix.next_to(mass_vector, LEFT)
        entry_anims_list = []
        for i in range(3):
            for j in range(12):
                vector = weighing_vectors[i]
                vector_entry = vector.get_column(j)
                matrix_entry = weighing_matrix.get_entry(i, j)
                entry_anims_list.append(
                    ReplacementTransform(vector_entry, matrix_entry)
                )
        bracket_anims_list = [
            ReplacementTransform(vector.get_brackets(), weighing_matrix.get_brackets())
            for vector in weighing_vectors
        ]
        self.play(
            AnimationGroup(*entry_anims_list),
            AnimationGroup(*bracket_anims_list),
            run_time=2,
        )
        self.wait()
        # Highlight rows and columns
        highlight_colors = [RED, GREEN, BLUE]
        rows_texts = VGroup(*[
            TexText(text, color=color)
            .scale(0.6)
            .next_to(vector.get_column(0), LEFT)
            .next_to(weighing_matrix, LEFT, coor_mask=[1, 0, 0])
            for text, color, vector in zip(
                ["第一次称量", "第二次称量", "第三次称量"], highlight_colors, weighing_vectors
            )
        ])
        columns_coins = VGroup(*[
            Coin(k + 1)
            .scale(0.7)
            .next_to(column, UP)
            for k, column in enumerate(weighing_matrix.get_columns())
        ])
        for text in rows_texts:
            text.generate_target()
        for coin in columns_coins:
            coin.generate_target()
        rows_texts.fade(1)
        columns_coins.fade(1)
        self.play(
            AnimationGroup(*[
                MoveToTarget(text)
                for text in rows_texts
            ], lag_ratio=0.2),
            AnimationGroup(*[
                Indicate(row, scale_factor=1, color=highlight_colors[k % 3])
                for k, row in enumerate(weighing_matrix.get_rows())
            ], lag_ratio=0.2),
            run_time=2,
        )
        self.wait()
        self.play(
            AnimationGroup(*[
                MoveToTarget(coin)
                for coin in columns_coins
            ], lag_ratio=0.2),
            AnimationGroup(*[
                Indicate(column, scale_factor=1, color=highlight_colors[k % 3])
                for k, column in enumerate(weighing_matrix.get_columns())
            ], lag_ratio=0.2),
            run_time=3,
        )
        self.wait()
        calc_group = VGroup()
        # Show result vector
        equal_sign = Tex("=")
        result_vector = VVector(*["?"] * 3)
        result_group = VGroup(equal_sign, result_vector)
        result_group.arrange(RIGHT, buff=0.3)
        result_group.next_to(mass_vector, RIGHT)
        self.add(result_group)
        all_mobs = VGroup(
            weighing_matrix, mass_vector, result_group,
            rows_texts, columns_coins,
        )
        self.play(all_mobs.animate.shift(4 * LEFT))
        self.wait()
        result_texts = VGroup(*[
            TexText(text, color=color)
            .scale(0.6)
            .next_to(result_vector.get_row(k), RIGHT)
            .next_to(result_vector, RIGHT, coor_mask=[1, 0, 0])
            for (k, text), color in zip(
                enumerate(["第一次称量结果", "第二次称量结果", "第三次称量结果"]),
                highlight_colors,
            )
        ])
        for text in result_texts:
            text.generate_target()
        result_texts.fade(1)
        self.play(
            AnimationGroup(*[
                MoveToTarget(text)
                for text in result_texts
            ], lag_ratio=0.2),
            AnimationGroup(*[
                Indicate(row, scale_factor=1, color=highlight_colors[k % 3])
                for k, row in enumerate(result_vector.get_rows())
            ], lag_ratio=0.2),
            run_time=2,
        )
        all_mobs.add(result_texts)
        self.wait()
        # Center the equation and define two terms
        all_mobs.generate_target()
        all_mobs.target.shift(2.2 * RIGHT)
        all_mobs.target[-3:].fade(1)
        self.play(MoveToTarget(all_mobs))
        self.wait()
        weighing_matrix_text = TexText("称量矩阵", color=YELLOW)
        weighing_matrix_text.next_to(weighing_matrix, UP)
        result_vector_text = TexText("结果向量", color=PINK)
        result_vector_text.next_to(result_vector, UP)
        self.play(
            Indicate(weighing_matrix, scale_factor=1, color=YELLOW),
            Write(weighing_matrix_text),
        )
        self.wait()
        self.play(
            Indicate(result_vector, scale_factor=1, color=PINK),
            Write(result_vector_text),
        )
        self.wait()
        # Cover all columns of the matrix as we don't know its entries yet
        cover_texts = VGroup(*[
            TexText("第", str(k + 1), "列", color=GREY)
            .scale(0.8)
            .arrange(DOWN, buff=0.15)
            .move_to(column)
            for k, column in enumerate(weighing_matrix.get_columns())
        ])
        cover_rects = VGroup(*[
            SurroundingRectangle(
                cover_text, stroke_color=GREY, stroke_width=2,
                fill_color=BLACK, fill_opacity=1, buff=0.1,
            )
            for cover_text in cover_texts
        ])
        self.play(FadeIn(cover_rects))
        self.play(FadeIn(cover_texts))
        self.wait()
        # Now move onto the mass vector part
        self.play(FadeOut(weighing_matrix_text), FadeOut(result_vector_text))
        self.wait()
        self.remove(weighing_matrix.elements)
        weighing_matrix = VGroup(
            weighing_matrix.get_brackets(), cover_rects, cover_texts,
        )
        weighing_matrix.save_state()
        self.play(
            mass_vector.animate.center(),
            FadeOut(weighing_matrix),
            FadeOut(result_group),
            run_time=2,
        )
        self.wait()
        self.weighing_matrix = weighing_matrix
        self.mass_vector = mass_vector
        self.result_group = result_group

    def define_standard_vector_and_bias_vector(self):
        # Every entry can be further divided into two terms
        mass_vector = self.mass_vector
        standard_vector = VVector(*["m"] * 12)
        bias_vector = VVector(*["b_{" + str(k) + "}" for k in range(1, 13)])
        add_sign = TexText("+")
        sum_group = VGroup(standard_vector, add_sign, bias_vector)
        sum_group.arrange(RIGHT).center()
        sum_group.generate_target()
        sum_group[0].center()
        sum_group[1].scale(0)
        sum_group[2].center()
        sum_group.fade(1)
        self.play(
            FadeOut(mass_vector, run_time=1.5),
            MoveToTarget(sum_group, run_time=2),
        )
        self.wait()
        self.play(Indicate(standard_vector), run_time=2)
        self.wait()
        self.play(Indicate(bias_vector), run_time=2)
        self.wait()
        self.play(
            AnimationGroup(*[
                Indicate(entry[1:])
                for entry in bias_vector.get_entries()
            ], lag_ratio=0.1),
            run_time=2,
        )
        self.wait()
        # Focus on one entry of the bias vector
        sixth_entry = bias_vector.get_row(6 - 1)
        sixth_entry.save_state()
        sixth_entry.generate_target()
        sixth_entry.target.set_color(YELLOW).scale(2).shift(1.5 * RIGHT)
        self.play(MoveToTarget(sixth_entry))
        self.wait()
        coin_fill_colors = [
            CoinType().coin_fill_color
            for CoinType in [RealCoin, LighterCoin, HeavierCoin]
        ]
        sixth_remarks = VGroup(*[
            VGroup(
                CoinType(6),
                TexText(remark_text, color=color)
            )
            .scale(1.2)
            .arrange(RIGHT, buff=0.3)
            .next_to(sixth_entry, DOWN, aligned_edge=LEFT, buff=0.5)
            for CoinType, remark_text, color in zip(
                [RealCoin, LighterCoin, HeavierCoin],
                ["真币", "轻的假币", "重的假币"],
                coin_fill_colors
            )
        ])
        sixth_possibilities = VGroup(*[
            Tex(possibility, color=YELLOW).scale(1.5)
            .next_to(sixth_entry, RIGHT, buff=0.4)
            for possibility in ("=0", "<0", ">0")
        ])
        for k in range(len(sixth_remarks)):
            if k == 0:
                self.play(
                    FadeIn(sixth_remarks[0]),
                    FadeIn(sixth_possibilities[0]),
                )
            else:
                self.play(
                    FadeOut(sixth_possibilities[k - 1]),
                    FadeOut(sixth_remarks[k - 1]),
                    FadeIn(sixth_possibilities[k]),
                    FadeIn(sixth_remarks[k]),
                )
            self.wait()
        self.play(
            FadeOut(sixth_possibilities[-1]),
            FadeOut(sixth_remarks[-1]),
        )
        self.wait()
        # Show possible values of bias 'b'
        text_colors = [
            CoinType().coin_fill_color
            for CoinType in [LighterCoin, RealCoin, HeavierCoin]
        ]
        b_values = VGroup(*[
            TexText(text, color=text_colors[k // 2])
            for k, text in enumerate(["-1", "偏轻", "0", "真币", "+1", "偏重"])
        ])
        b_values.arrange_in_grid(3, 2, aligned_edge=LEFT)
        b_brace = Brace(b_values, direction=LEFT)
        b_equal = Tex("=")
        b_equal.next_to(b_brace, LEFT)
        b_group = VGroup(b_equal, b_brace, b_values)
        b_group.next_to(sixth_entry, RIGHT)
        self.play(
            sixth_entry.animate.set_color(WHITE),
            FadeIn(b_group, shift=LEFT),
        )
        self.wait()
        # Signs, not the number itself, are the most important part
        signs = VGroup(b_values[0][0][0], b_values[-2][0][0])
        self.play(Indicate(signs, scale_factor=1, lag_ratio=0.2))
        self.wait()
        self.play(Restore(sixth_entry), FadeOut(b_group))
        self.wait()
        # Define the rest two terms
        standard_vector_text = TexText("标准向量", color=YELLOW)
        standard_vector_text.next_to(standard_vector, LEFT, buff=0.6)
        bias_vector_text = TexText("偏差向量", color=PINK)
        bias_vector_text.next_to(bias_vector, RIGHT, buff=0.6)
        self.play(
            Indicate(standard_vector, scale_factor=1),
            Write(standard_vector_text),
            run_time=2,
        )
        self.wait()
        self.play(
            Indicate(bias_vector, scale_factor=1, color=PINK),
            Write(bias_vector_text),
            run_time=2,
        )
        self.wait()
        # Back to the ol' matrix form
        brackets = LargeBrackets(VGroup(standard_vector, bias_vector))
        self.play(
            FadeOut(standard_vector_text, run_time=1),
            FadeOut(bias_vector_text, run_time=1),
            Write(brackets, run_time=2),
        )
        self.wait()
        sum_group.add(brackets)
        self.sum_group = sum_group

    def problem_recap(self):
        weighing_matrix = self.weighing_matrix
        sum_group = self.sum_group
        result_group = self.result_group
        sum_group.generate_target()
        equation_group = VGroup(weighing_matrix, sum_group.target, result_group)
        equation_group.arrange(RIGHT)
        equation_group.set_width(FRAME_WIDTH - 1).center()
        self.play(
            FadeIn(weighing_matrix),
            FadeIn(result_group),
            MoveToTarget(sum_group),
            run_time=2,
        )
        self.wait()
        equation_group = VGroup(weighing_matrix, sum_group, result_group)
        # All entries are same in the standard vector
        # and at most 1 entry is non-zero in the bias vector
        standard_vector, add_sign, bias_vector, brackets = sum_group
        self.play(
            standard_vector.animate.set_color(YELLOW),
            bias_vector.animate.set_color(PINK),
            run_time=2,
        )
        self.wait()
        # standard_constraint = TexText("相等", color=YELLOW)
        standard_constraint = Text("相等", color=YELLOW, font="STZhongSong")  # hack
        standard_constraint.scale(0.8).next_to(standard_vector, UP)
        bias_constraint = TexText("至多一个\\\\非零元素", color=PINK)
        bias_constraint.scale(0.8).next_to(bias_vector, UP)
        self.play(
            Indicate(standard_vector.elements, color=None, lag_ratio=0.1),
            run_time=2,
        )
        self.wait()
        self.play(Write(standard_constraint))
        self.wait()
        self.play(
            FadeOut(standard_constraint),
            standard_vector.animate.set_color(WHITE),
            Indicate(bias_vector.elements, color=None, lag_ratio=0.1),
            run_time=2,
        )
        self.play(Write(bias_constraint))
        self.wait()
        # Weighing is the same thing as multiplying a matrix on the left
        self.play(
            FadeOut(bias_constraint),
            bias_vector.animate.set_color(WHITE),
            Indicate(weighing_matrix, scale_factor=1.1, color=None)
        )
        self.wait()
        self.play(Indicate(result_group[-1], color=BLUE))
        self.wait()
        # We want to know what's inside the bias vector
        question_rect = SurroundingRectangle(
            bias_vector, stroke_color=PINK,
            fill_color=BLACK, fill_opacity=0.7,
        )
        question_mark = TexText("?", color=PINK)
        question_mark.set_width(question_rect.get_width() * 0.6)
        question_mark.move_to(question_rect)
        question_rect.add(question_mark)
        self.play(FadeIn(question_rect))
        self.wait()
        # But to do so, we need to use the result vector
        self.play(Indicate(result_group[-1], color=BLUE))
        self.wait()
        arrow = ArcBetweenPoints(
            result_group[-1].get_bottom() + 0.3 * DOWN,
            bias_vector.get_row(9).get_right() + 0.5 * RIGHT,
            angle=-PI / 3,
        )
        arrow.add_tip()
        arrow.set_color(BLUE)
        self.play(ShowCreation(arrow))
        self.wait()
        self.play(FadeOut(arrow), FadeOut(question_rect))
        self.wait()
        # Center weighing matrix
        all_mobs = VGroup(*self.mobjects)
        all_mobs.remove(weighing_matrix)
        weighing_matrix.generate_target()
        weighing_matrix.target.set_height(1.5).center()
        self.play(FadeOut(all_mobs), MoveToTarget(weighing_matrix))
        self.wait()
        design_text = TexText("如何设计称量矩阵?", color=YELLOW)
        design_text.next_to(weighing_matrix, UP, buff=0.8)
        self.play(Write(design_text))
        self.wait()


class WeighingMatrixMustSatisfyFiveConditions(Scene):
    def setup(self):
        # Setup to match with the last scene
        weighing_matrix = WeighingMatrix([[-1] * 12, [0] * 12, [1] * 12])
        cover_texts = VGroup(*[
            TexText("第", str(k + 1), "列", color=GREY)
            .scale(0.8)
            .arrange(DOWN, buff=0.15)
            .move_to(column)
            for k, column in enumerate(weighing_matrix.get_columns())
        ])
        cover_rects = VGroup(*[
            SurroundingRectangle(
                cover_text, color=None, fill_color=None, fill_opacity=0,
                stroke_color=GREY, stroke_width=2, buff=0.1,
            )
            for cover_text in cover_texts
        ])
        cover_group = VGroup(*[
            VGroup(rect, text)
            for rect, text in zip(cover_rects, cover_texts)
        ])
        weighing_matrix = VGroup(cover_group, weighing_matrix.get_brackets())
        weighing_matrix.set_height(1.5)
        design_text = TexText("如何设计称量矩阵?", color=YELLOW)
        design_text.next_to(weighing_matrix, UP, buff=0.8)
        self.add(weighing_matrix, design_text)
        self.weighing_matrix = weighing_matrix
        self.design_text = design_text

    def construct(self):
        self.eliminate_standard_vector()
        self.show_all_possible_result_vectors()
        self.last_conditions_for_weighing_matrix()

    def eliminate_standard_vector(self):
        weighing_matrix = self.weighing_matrix
        standard_vector = VVector(*["m"] * 12)
        bias_vector = VVector(*["b_{" + str(k) + "}" for k in range(1, 13)])
        add_sign = TexText("+")
        sum_group = VGroup(standard_vector, add_sign, bias_vector)
        sum_group.arrange(RIGHT)
        brackets = LargeBrackets(sum_group)
        sum_group.add(brackets)
        equal_sign = Tex("=")
        result_vector = VVector(*["?"] * 3)
        # Move stuff to the top
        weighing_matrix.generate_target()
        equation_group = VGroup(
            weighing_matrix.target, sum_group, equal_sign, result_vector
        )
        equation_group.arrange(RIGHT)
        equation_group.center().set_width(FRAME_WIDTH - 1).to_edge(UP)
        self.play(
            MoveToTarget(weighing_matrix),
            FadeIn(equation_group[1:]),
            FadeOut(self.design_text),
            run_time=2,
        )
        equation_group = VGroup(
            weighing_matrix, sum_group, equal_sign, result_vector
        )
        self.wait()
        # 'm' is an unknown variable which needs to be eliminated
        unknown_text = TexText("$m$未知", color=YELLOW)
        unknown_text.next_to(standard_vector, DOWN)
        self.play(
            Indicate(standard_vector.elements, scale_factor=1, lag_ratio=0.05),
            Write(unknown_text),
        )
        self.wait()
        eliminate_text = TexText("“归零”", color=YELLOW)
        eliminate_text.move_to(unknown_text)
        self.play(ReplacementTransform(unknown_text, eliminate_text))
        self.wait()
        weighing_matrix.save_state()
        self.play(
            FadeOut(eliminate_text),
            weighing_matrix.animate.set_color(color=BLUE),
            standard_vector.animate.set_color(BLUE),
        )
        self.wait()
        # Weighing matrix times standard vector equals zero vector
        columns_copy = weighing_matrix[0].deepcopy()
        add_signs = VGroup(*[
            Tex("+").scale(0.8)
            for k in range(len(columns_copy) - 1)
        ])
        for k in range(len(columns_copy) - 1):
            add_signs[k].next_to(columns_copy[k], RIGHT, buff=0.1)
            columns_copy[k + 1].next_to(add_signs[k], RIGHT, buff=0.1)
        column_sum = VGroup(columns_copy, add_signs)
        brackets = LargeBrackets(column_sum)
        column_sum.add(brackets)
        m_text = Tex("m")
        zero_vector = VGroup(Tex("="), VVector(*[0] * 3))
        zero_vector.arrange(RIGHT)
        column_sum_group = VGroup(m_text, column_sum, zero_vector)
        column_sum_group.arrange(RIGHT).center().set_width(FRAME_WIDTH - 1).to_edge(DOWN, buff=0.8)
        self.play(
            equation_group.animate.shift(2 * UP),
            ReplacementTransform(weighing_matrix[0].deepcopy(), column_sum[0]),
            ReplacementTransform(standard_vector.elements.deepcopy(), m_text),
            Write(column_sum[1:]),
            run_time=3,
        )
        self.play(FadeIn(zero_vector))
        self.wait()
        rule_1 = TexText("-", "每一行元素的和为0", color=YELLOW)
        rule_1.next_to(column_sum[:-1], UP)
        self.play(
            Write(rule_1),
            FadeOut(m_text), FadeOut(column_sum[-1]),
            zero_vector.animate.next_to(column_sum[:-1], RIGHT),
            run_time=2,
        )
        self.wait()
        # Now we can remove standard vector
        self.play(FadeOut(VGroup(rule_1, column_sum[:-1], zero_vector)))
        self.wait()
        equation_group = VGroup(
            weighing_matrix, bias_vector, equal_sign, result_vector
        )
        equation_group.generate_target()
        equation_group.target[0].restore()
        equation_group.target.arrange(RIGHT).center().to_edge(UP)
        self.play(
            VGroup(sum_group[0], sum_group[1], sum_group[-1]).animate.shift(2 * UP).fade(1),
            MoveToTarget(equation_group),
            run_time=2,
        )
        self.wait()
        self.equation_group = equation_group

    def show_all_possible_result_vectors(self):
        weighing_matrix, bias_vector, equal_sign, result_vector = self.equation_group
        self.play(Indicate(bias_vector, scale_factor=1, color=YELLOW))
        self.wait()
        # all zero bias -> all zero result
        for mob in self.equation_group:
            mob.save_state()
        all_zero_bias = VVector(*["0"] * 12)
        all_zero_bias.match_height(bias_vector).move_to(bias_vector).set_color(YELLOW)
        all_zero_result = VVector(*["0"] * 3)
        all_zero_result.match_height(result_vector).move_to(result_vector).set_color(GREEN)
        self.play(Transform(bias_vector, all_zero_bias))
        self.wait()
        self.play(Transform(result_vector, all_zero_result))
        self.wait()
        self.play(Restore(bias_vector), Restore(result_vector))
        self.wait()
        # one non-zero bias -> one column of the matrix
        sixth_entry = bias_vector.get_row(5)
        other_entries = VGroup(*[
            bias_vector.get_row(k)
            for k in (list(range(5)) + list(range(6, 12)))
        ])
        other_zero_entries = VGroup(*[
            Tex("0").scale(0.8).move_to(entry)
            for entry in other_entries
        ])
        sixth_entry.save_state()
        sixth_rect = SurroundingRectangle(sixth_entry)
        self.play(
            ShowCreationThenDestruction(sixth_rect),
            Indicate(sixth_entry, scale_factor=1),
        )
        self.wait()
        self.play(Transform(other_entries, other_zero_entries))
        self.wait()
        sixth_column = weighing_matrix[0][5]
        other_columns = VGroup(weighing_matrix[0][:5], weighing_matrix[0][6:])
        self.play(
            sixth_entry.animate.set_color(YELLOW),
            other_entries.animate.fade(0.9),
            sixth_column.animate.set_color(YELLOW),
            other_columns.animate.fade(0.8),
        )
        self.wait()
        # Show two possibilities of the sixth entry
        sixth_entry.save_state()
        sixth_plus_one, sixth_minus_one = VGroup(*[
            Tex(text, color=color).scale(0.8).move_to(sixth_entry)
            for text, color in zip(["+1", "-1"], [BLUE_E, RED_E])
        ])
        plus_sign, minus_sign = VGroup(*[
            Tex(text, color=color).scale(1.5).next_to(equal_sign, RIGHT)
            for text, color in zip(["+\\,", "-\\,"], [BLUE_E, RED_E])
        ])
        sixth_column_copy = sixth_column.deepcopy()
        sixth_column_copy.generate_target()
        sixth_column_copy.target.scale(1.5).next_to(plus_sign, RIGHT).set_color(BLUE_E)
        self.play(Transform(sixth_entry, sixth_plus_one))
        self.wait()
        self.play(
            MoveToTarget(sixth_column_copy, path_arc=-PI / 3),
            FadeOut(result_vector),
        )
        result_vector.fade(1)
        self.play(Write(plus_sign))
        self.wait()
        self.play(
            Transform(sixth_entry, sixth_minus_one),
        )
        self.wait()
        self.play(
            Transform(plus_sign, minus_sign),
            sixth_column_copy.animate.set_color(RED_E).next_to(minus_sign, RIGHT),
        )
        self.wait()
        self.play(
            FadeOut(plus_sign), FadeOut(sixth_column_copy),
            AnimationGroup(*[Restore(mob) for mob in self.equation_group])
        )
        self.wait()
        # All 25 possible result vectors
        plus_columns = weighing_matrix[0]
        other_mobs = VGroup(weighing_matrix[1], bias_vector, equal_sign, result_vector)
        self.play(
            FadeOut(other_mobs, run_time=1),
            plus_columns.animate.arrange(RIGHT, buff=0.55).move_to(DOWN + 0.5 * RIGHT).set_color(BLUE_E)
        )
        plus_signs = VGroup(*[
            Tex("+\\,", color=BLUE_E).scale(0.6).next_to(column, LEFT, buff=0.05)
            for column in plus_columns
        ])
        self.play(FadeIn(plus_signs))
        self.wait()
        minus_columns = plus_columns.deepcopy()
        minus_columns.shift(2 * DOWN).set_color(RED_E)
        minus_signs = VGroup(*[
            Tex("-\\,", color=RED_E).scale(0.6).next_to(column, LEFT, buff=0.05)
            for column in minus_columns
        ])
        self.play(
            ReplacementTransform(plus_signs.deepcopy(), minus_signs),
            ReplacementTransform(plus_columns.deepcopy(), minus_columns),
        )
        self.wait()
        zero_column = VVector(*["0"] * 3)
        zero_column.set_color(GREEN)
        zero_column.scale(0.8).next_to(VGroup(plus_columns, minus_columns), LEFT, buff=0.6)
        self.play(Write(zero_column))
        self.wait()
        self.zero_column = zero_column
        self.plus_group = VGroup(*[
            VGroup(column, sign)
            for column, sign in zip(plus_columns, plus_signs)
        ])
        self.minus_group = VGroup(*[
            VGroup(column, sign)
            for column, sign in zip(minus_columns, minus_signs)
        ])
        self.all_columns = VGroup(self.zero_column, self.plus_group, self.minus_group)

    def last_conditions_for_weighing_matrix(self):
        all_columns = self.all_columns
        zero_column, plus_group, minus_group = all_columns
        # all_distinct_text = TexText("25个结果向量各不相同", color=YELLOW)
        all_distinct_text = Text("结果向量各不相同", font="STZhongSong", color=YELLOW)
        all_distinct_text.shift(2 * UP)
        all_distinct_rect = SurroundingRectangle(all_columns)
        self.play(
            Write(all_distinct_text),
            ShowCreationThenDestruction(all_distinct_rect),
        )
        self.wait()
        # An example that won't work
        replaced_vectors = plus_group[3:5]
        alternate_vectors = VGroup(*[
            VVector(0, 1, 1).match_width(vector).move_to(vector)
            for vector in replaced_vectors
        ])
        self.play(FadeOut(replaced_vectors), FadeIn(alternate_vectors))
        self.wait()
        alternate_rect = SurroundingRectangle(alternate_vectors)
        alternate_rect.set_stroke(color=YELLOW)
        alternate_cross = Cross(alternate_vectors)
        alternate_cross.set_color(YELLOW)
        alternate_cross.set_stroke(width=5)
        self.play(ShowCreation(alternate_rect))
        self.play(DrawBorderThenFill(alternate_cross))
        self.wait()
        self.play(
            FadeOut(VGroup(alternate_rect, alternate_cross, alternate_vectors, all_distinct_text)),
            FadeIn(replaced_vectors),
        )
        self.wait()
        # Zero vector already showed up, so there're no all-zero columns
        all_rules = VGroup(*[
            TexText("-", rule_text, color=YELLOW)
            for rule_text in [
                "不能出现全零的列", "任意两列不同", "任意两列的和不为零向量",
                "元素只能是-1, 0或1", "每一行元素的和为0",
            ]
        ])
        all_rules.arrange(DOWN, aligned_edge=LEFT).scale(0.8).to_edge(UP, buff=0.6)
        self.play(Indicate(zero_column))
        self.wait()
        for rule in all_rules:
            self.play(Write(rule))
            self.wait()
        self.play(
            FadeOut(self.all_columns),
            all_rules.animate.set_color(WHITE).scale(5 / 8).to_corner(UL)
        )
        self.wait()


class DesignAProperWeighingMatrix(Scene):
    def setup(self):
        all_rules = VGroup(*[
            TexText("-", rule_text)
            for rule_text in [
                "不能出现全零的列", "任意两列不同", "任意两列的和不为零向量",
                "元素只能是-1, 0或1", "每一行元素的和为0",
            ]
        ])
        all_rules.arrange(DOWN, aligned_edge=LEFT).scale(0.5).to_corner(UL)
        self.add(all_rules)
        self.all_rules = all_rules

    def construct(self):
        # List all possible combinations
        possible_combs_list = list(it.product([0, 1, -1], repeat=3))
        all_possible_columns = VGroup(*[
            VVector(*comb)
            for comb in possible_combs_list
        ])
        for column in all_possible_columns:
            column.get_brackets().set_color(GREY).fade(0.2)
        all_possible_columns.arrange_in_grid(3, 9, h_buff_ratio=1)
        all_possible_columns.scale(0.8).to_edge(DOWN)
        # All entries come from {-1, 0, 1}
        self.play(
            AnimationGroup(*[
                FadeIn(column)
                for column in all_possible_columns
            ], lag_ratio=0.1),
            self.all_rules[-2].animate.set_color(GREEN),
            run_time=3,
        )
        self.wait()
        # No all-zero column
        zero_column = all_possible_columns[0]
        self.play(
            FadeOut(zero_column),
            self.all_rules[0].animate.set_color(GREEN),
        )
        possible_combs_list.pop(0)
        all_possible_columns.remove(zero_column)
        self.wait()
        # Select 12 from the remaining 26 vectors
        select_12_from_26_text = TexText("不重复选择12个", color=YELLOW)
        select_12_from_26_text.scale(1.2).move_to(2.5 * UP + RIGHT)
        self.play(Write(select_12_from_26_text))
        self.play(self.all_rules[1].animate.set_color(GREEN))
        self.wait()
        self.play(FadeOut(select_12_from_26_text))
        self.wait()
        # Construct index mapping for the grouping
        index_map = {}
        for k in range(len(possible_combs_list)):
            if not (k in index_map.keys() or k in index_map.values()):
                curr_vec = possible_combs_list[k]
                comp_vec = tuple(-entry for entry in curr_vec)
                comp_index = possible_combs_list.index(comp_vec)
                index_map[k] = comp_index
        # Rearrange into 13 groups
        all_possible_columns.generate_target()
        temp_group = VGroup(*[
            VGroup(all_possible_columns.target[k], all_possible_columns.target[index_map[k]])
            .arrange(DOWN, buff=0.8)
            for k in index_map.keys()
        ])
        temp_group.arrange(RIGHT, buff=0.2).set_width(FRAME_WIDTH - 1)
        temp_group.to_edge(DOWN, buff=0.8)
        self.play(MoveToTarget(all_possible_columns), run_time=3)
        self.wait()
        rearranged_group = VGroup(*[
            VGroup(all_possible_columns[k], all_possible_columns[index_map[k]])
            for k in index_map.keys()
        ])
        select_one_from_each_set = TexText("每组只选一个向量", color=YELLOW)
        select_one_from_each_set.scale(1.2).next_to(rearranged_group, UP, buff=1)
        set_arrows = VGroup(*[
            Arrow(ORIGIN, DOWN).set_color(YELLOW).next_to(column_set, UP)
            for column_set in rearranged_group
        ])
        self.play(
            AnimationGroup(*[GrowArrow(arrow) for arrow in set_arrows], run_time=2),
            Write(select_one_from_each_set, run_time=1),
        )
        self.play(self.all_rules[2].animate.set_color(GREEN))
        self.wait()
        self.play(
            FadeOut(select_one_from_each_set),
            FadeOut(set_arrows),
            VGroup(*[column[1:] for column in all_possible_columns]).animate.fade(1),
        )
        self.wait()
        # Now change selection to satisfy the last condition
        extra_group = rearranged_group[-1]
        extra_group_text = TexText("多出一组", color=RED)
        extra_group_text.scale(0.8).next_to(extra_group, UP)
        extra_group_cross = Cross(extra_group)
        extra_group_cross.set_stroke(width=5)
        self.play(Write(extra_group_text), ShowCreation(extra_group_cross))
        self.wait()
        self.play(
            FadeOut(rearranged_group[-1]),
            FadeOut(extra_group_text), FadeOut(extra_group_cross),
        )
        self.wait()
        for start_ind, end_ind in [(-5, -1), (2, 4)]:
            swap_column_sets = rearranged_group[start_ind:end_ind]
            self.play(
                AnimationGroup(*[
                    Swap(*column_set.submobjects)
                    for column_set in swap_column_sets
                ], lag_ratio=0.05, run_time=2)
            )
            self.wait()
            for column_set in swap_column_sets:
                column_set.submobjects.reverse()
        # That's how we construct a proper weighing matrix
        weighing_matrix = WeighingMatrix(
            np.array([
                np.array(column_set[0].entries)
                for column_set in rearranged_group[:-1]
            ]).T
        )
        weighing_matrix.set_color(YELLOW).move_to(0.5 * DOWN)
        self.play(
            AnimationGroup(*[
                ReplacementTransform(column_set[0].elements, target_column)
                for column_set, target_column in zip(
                    rearranged_group[:-1], weighing_matrix.get_columns()
                )
            ], lag_ratio=0.05, run_time=2),
            FadeOut(
                VGroup(*[column_set[1] for column_set in rearranged_group[:-1]])
            ),
        )
        self.play(
            Write(weighing_matrix.get_brackets()),
            self.all_rules.animate.set_color(GREEN),
            run_time=1,
        )
        self.wait()
        self.play(
            self.all_rules.animate.shift(4 * UP),
            weighing_matrix.animate.to_edge(UP, buff=0.5).set_color(WHITE),
            run_time=2,
        )
        self.wait()


class FromMatrixToReality(Scene):
    def construct(self):
        # Match with the last scene
        weighing_matrix = WeighingMatrix([
            [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1],
            [0, 1, -1, -1, 0, 0, 0, 1, -1, -1, 1, 1],
            [1, 0, -1, 1, 0, 1, -1, 0, -1, 1, 0, -1],
        ])
        weighing_matrix.to_edge(UP, buff=0.5)
        self.add(weighing_matrix)
        # Couldn't resist...
        matrix_to_reality = VGroup(*[
            TexText("Matrix"),
            Arrow(ORIGIN, 2 * DOWN),
            TexText("Reality"),
        ])
        matrix_to_reality.arrange(DOWN).set_color(GREY)
        matrix_to_reality.center()
        for AnimationType in (FadeIn, FadeOut):
            self.play(AnimationType(matrix_to_reality))
            self.wait()
        # Finally back to pan balances
        pbs = VGroup(*[PanBalance(max_tilt_angle=PI / 20) for k in range(3)])
        pbs.arrange(RIGHT, buff=1.5).move_to(1.5 * DOWN).set_width(FRAME_WIDTH - 1)
        pbs.shift(4 * DOWN)
        self.play(pbs.animate.shift(4 * UP))
        self.wait()
        coins = VGroup(*[
            VGroup(*[Coin(k, radius=0.25) for k in range(1, 13)])
            for l in range(3)
        ])
        weighing_texts = VGroup(*[
            TexText(text, color=GREY)
            for text in ("第一次称量", "第二次称量", "第三次称量")
        ])
        # Put all coins in place according to the matrix
        for row_ind, coin_set in enumerate(coins):
            for col_ind, coin in enumerate(coin_set):
                coin.generate_target()
                coin.scale(0).move_to(weighing_matrix.get_entry(row_ind, col_ind))
        all_coin_set_list = []
        for pb_ind, (pb, text) in enumerate(zip(pbs, weighing_texts)):
            text.scale(0.6).next_to(pb, DOWN, buff=1)
            left_coins, right_coins, idle_coins = VGroup(), VGroup(), VGroup()
            for coin_ind, coin in enumerate(coins[pb_ind]):
                entry = weighing_matrix.get_entry(pb_ind, coin_ind).tex_string
                coin_target = coin.target
                if entry == "-1":
                    left_coins.add(coin_target)
                elif entry == "1":
                    right_coins.add(coin_target)
                else:
                    idle_coins.add(coin_target)
            pb.put_coins_on_left_pan(left_coins, arrangement=[2, 2], add_updater=True)
            pb.put_coins_on_right_pan(right_coins, arrangement=[2, 2], add_updater=True)
            idle_coins.arrange(RIGHT, buff=COIN_BUFF).next_to(pb, DOWN, buff=0.1)
            for coin_set in (left_coins, right_coins, idle_coins):
                all_coin_set_list.append(coin_set)
        for coin_set, text in zip(coins, weighing_texts):
            self.play(
                AnimationGroup(*[
                    MoveToTarget(coin)
                    for coin in coin_set
                ], lag_ratio=0.05, run_time=2),
                FadeIn(text, run_time=1),
            )
            self.wait()
        self.remove(coins)
        for coin_set in all_coin_set_list:
            self.add(coin_set)
        # Example 1: coin 7 is a heavier counterfeit
        self.play(
            pbs[0].get_tilt_right_animation("hide"),
            pbs[1].get_wiggle_animation("hide"),
            pbs[2].get_tilt_left_animation("hide"),
            run_time=2,
        )
        self.wait()
        pb_states_texts = VGroup(*[
            TexText(text, color=YELLOW).scale(0.8).next_to(pb, UP, buff=1)
            for text, pb in zip(["向右倾斜", "平衡", "向左倾斜"], pbs)
        ])
        self.play(FadeIn(pb_states_texts, shift=1.5 * UP))
        self.wait()
        pb_states_numbers = VGroup(*[
            Tex(str(number), color=YELLOW).move_to(pb_state_text)
            for number, pb_state_text in zip([1, 0, -1], pb_states_texts)
        ])
        self.play(FadeTransform(pb_states_texts, pb_states_numbers))
        self.wait()
        result_vector = VVector(1, 0, -1)
        result_vector.set_color(YELLOW)
        result_vector.next_to(weighing_matrix, DOWN, buff=0.4)
        self.play(
            FadeIn(result_vector.get_brackets()[0], shift=6.5 * RIGHT),
            FadeIn(result_vector.get_brackets()[1], shift=6.5 * LEFT),
            ReplacementTransform(pb_states_numbers, result_vector.elements)
        )
        self.wait()
        seventh_column = weighing_matrix.get_columns()[6]
        self.play(
            result_vector.animate.set_color(BLUE_E).next_to(seventh_column, DOWN, buff=0.4),
            seventh_column.animate.set_color(BLUE_E),
        )
        self.wait()
        counterfeit_text = TexText("7号是偏重的假币", color=BLUE_E)
        counterfeit_text.move_to(result_vector)
        counterfeit_7_coins = VGroup(*[
            HeavierCoin(7, radius=0.25).move_to(all_coin_set_list[set_ind][coin_ind])
            for (set_ind, coin_ind) in [(1, 2), (5, 3), (6, 1)]
        ])
        coins.save_state()
        self.play(
            FadeOut(result_vector, shift=LEFT),
            FadeIn(counterfeit_text, shift=LEFT),
            FadeIn(counterfeit_7_coins),
        )
        self.wait()
        self.play(
            weighing_matrix.animate.set_color(WHITE),
            FadeOut(counterfeit_text),
            FadeOut(counterfeit_7_coins),
        )
        self.wait()
        # Example 2: coin 5 is a lighter counterfeit
        self.play(
            pbs[0].get_tilt_left_animation("hide"),
            pbs[1].get_balance_animation("hide"),
            pbs[2].get_balance_animation("hide"),
            run_time=2,
        )
        result_vector = VVector(-1, 0, 0)
        result_vector.set_color(YELLOW)
        result_vector.next_to(weighing_matrix, DOWN, buff=0.4)
        self.play(FadeIn(result_vector, shift=1.5 * UP))
        self.wait()
        fifth_column = weighing_matrix.get_columns()[4]
        self.play(
            result_vector.animate.set_color(RED_E).next_to(fifth_column, DOWN, buff=0.4),
            fifth_column.animate.set_color(RED_E),
        )
        self.wait()
        counterfeit_text = TexText("5号是偏轻的假币", color=RED_E)
        counterfeit_text.move_to(result_vector)
        counterfeit_5_coins = VGroup(*[
            LighterCoin(5, radius=0.25).move_to(all_coin_set_list[set_ind][coin_ind])
            for (set_ind, coin_ind) in [(1, 0), (5, 1), (8, 1)]
        ])
        coins.save_state()
        self.play(
            FadeOut(result_vector, shift=LEFT),
            FadeIn(counterfeit_text, shift=LEFT),
            FadeIn(counterfeit_5_coins),
        )
        self.wait()
        self.play(
            weighing_matrix.animate.set_color(WHITE),
            FadeOut(counterfeit_text),
            FadeOut(counterfeit_5_coins),
        )
        self.wait()
        # Example 3: No counterfeits
        self.play(
            AnimationGroup(*[pb.get_balance_animation("hide") for pb in pbs]),
            run_time=2,
        )
        self.wait()
        real_coins = VGroup(*[
            RealCoin(k + 1, radius=0.25).move_to(coin)
            for coin_set in coins
            for k, coin in enumerate(coin_set)
        ])
        result_vector = VVector(0, 0, 0)
        result_vector.set_color(GREEN)
        result_vector.next_to(weighing_matrix, DOWN, buff=0.4)
        self.play(FadeIn(result_vector, shift=1.5 * UP))
        self.wait()
        self.play(FadeIn(real_coins))
        self.wait()
        self.play(
            FadeOut(result_vector), FadeOut(real_coins),
            AnimationGroup(*[pb.get_balance_animation("show") for pb in pbs]),
        )
        self.wait()


class FinalRecap(Scene):
    def construct(self):
        title = Title("小结", include_underline=False)
        title.to_edge(UP, buff=0.5)
        screen_rect = ScreenRectangle(height=6).shift(0.5 * DOWN)
        self.add(title)
        self.wait()
        self.play(ShowCreation(screen_rect))
        self.wait()


class CoreIdeasBehindTheSolutions(Scene):
    def construct(self):
        # Intro
        entropy_text = Text("信息熵", font="STZhongSong", color=YELLOW)
        entropy_formula = Tex("S = -\\sum_{i}{p_i \\log_{2}{p_i}}")
        entropy_group = VGroup(entropy_text, entropy_formula)
        entropy_group.arrange(DOWN, buff=0.5)
        entropy_group.move_to(LEFT_SIDE / 2)
        test_matrix_text = Text("校验矩阵", font="STZhongSong", color=YELLOW)
        test_matrix_formula = Tex("H \\vec{\\mathbf{x}} = \\vec{\\mathbf{s}}")
        test_matrix_group = VGroup(test_matrix_text, test_matrix_formula)
        test_matrix_group.arrange(DOWN, buff=0.5)
        test_matrix_group.move_to(RIGHT_SIDE / 2)
        for group in (entropy_group, test_matrix_group):
            self.play(FadeIn(group, shift=UP))
            self.wait()
        self.remove(*self.mobjects)
        # Part 1 overlay
        close_prob_text = Tex(
            "p_{\\text{向左倾斜}}", " \\approx ", "p_{\\text{平衡}}", " \\approx ", "p_{\\text{向右倾斜}}",
        )
        hard_to_predict_text = TexText("预测结果困难", "，信息量较大")
        easy_to_predict_text = TexText("预测结果容易", "，信息量较小")
        for text in (close_prob_text, hard_to_predict_text, easy_to_predict_text):
            text.scale(1.25).move_to(2 * UP)
        self.play(FadeIn(close_prob_text, shift=UP))
        self.wait()
        self.play(
            FadeOut(close_prob_text, shift=UP),
            FadeIn(hard_to_predict_text[0], shift=UP),
        )
        self.wait()
        self.play(FadeIn(hard_to_predict_text[1], shift=UP))
        self.wait()
        self.play(FadeTransform(hard_to_predict_text, easy_to_predict_text))
        self.wait()
        vague_info_text = TexText("信息量的大小?", color=YELLOW)
        precise_info_text = TexText("信息熵", color=YELLOW)
        for text in (vague_info_text, precise_info_text):
            text.scale(1.25).to_edge(UP)
        self.play(FadeTransform(easy_to_predict_text, vague_info_text))
        self.wait()
        info_entropy_img = ImageMobject("info_entropy_paper.png")
        info_entropy_img.set_height(6).move_to(LEFT_SIDE / 2).to_edge(DOWN)
        self.play(FadeIn(info_entropy_img, shift=UP))
        self.wait()
        entropy_formula.move_to(RIGHT_SIDE / 3)
        entropy_formula.generate_target()
        entropy_formula.scale(0).move_to(info_entropy_img.get_center())
        self.play(
            FadeTransform(vague_info_text, precise_info_text),
            MoveToTarget(entropy_formula),
        )
        self.wait()
        full_entropy_formula = Tex(
            "S =",
            "& -", "p_{\\text{向左倾斜}}", " \\log_{2}", "{p_{\\text{向左倾斜}}} \\\\",
            "& -", "p_{\\text{平衡}}", " \\log_{2}", "{p_{\\text{平衡}}} \\\\",
            "& -", "p_{\\text{向右倾斜}}", " \\log_{2}", "{p_{\\text{向右倾斜}}}",
        )
        full_entropy_formula.move_to(RIGHT_SIDE / 3)
        full_entropy_formula[2:5:2].set_color(RED)
        full_entropy_formula[6:9:2].set_color(BLUE)
        full_entropy_formula[10::2].set_color(GREEN)
        self.play(FadeTransform(entropy_formula, full_entropy_formula))
        self.wait()
        sur_rect = SurroundingRectangle(full_entropy_formula)
        maximize_entropy_text = TexText("将$S$最大化", color=YELLOW)
        maximize_formula_text = Tex(
            "p_{\\text{向左倾斜}}", " \\approx ", "p_{\\text{平衡}}", " \\approx ",
            "p_{\\text{向右倾斜}}", " \\approx \\dfrac{1}{3}"
        )
        maximize_formula_text[0].set_color(RED)
        maximize_formula_text[2].set_color(BLUE)
        maximize_formula_text[4].set_color(GREEN)
        for text in (maximize_entropy_text, maximize_formula_text):
            text.next_to(sur_rect, DOWN)
        self.play(ShowCreation(sur_rect))
        self.wait()
        self.play(FadeIn(maximize_entropy_text, shift=2 * DOWN))
        self.wait()
        self.play(
            FadeOut(maximize_entropy_text, shift=2 * DOWN),
            FadeIn(maximize_formula_text, shift=2 * DOWN),
        )
        self.wait()
        self.remove(*self.mobjects)
        # Part 2 overlay
        hamming74_title = TexText("(7,\\,4)汉明码", color=YELLOW)
        hamming74_title.to_edge(UP)
        hamming74_matrix = WeighingMatrix([
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1],
        ])
        hamming74_vector = VVector(1, 1, 1, 1, 0, 1, 0)  # It should be (1,0,1,1,0,1,0)
        hamming74_result = VGroup(Tex("="), VVector(1, 0, 1))
        hamming74_result.arrange(RIGHT)
        hamming74_group = VGroup(hamming74_matrix, hamming74_vector, hamming74_result)
        hamming74_group.arrange(RIGHT)
        self.add(hamming74_title, hamming74_matrix, hamming74_vector)
        self.wait()
        connections = VGroup(*[
            Line(
                hamming74_matrix.get_entry(row_ind, col_ind),
                hamming74_vector.get_row(col_ind),
                stroke_color=BLUE, stroke_width=0.2,
            )
            for row_ind in range(3)
            for col_ind in range(7)
        ])
        self.play(ShowCreation(
            connections,
            lag_ratio=0.05, run_time=2,
        ))
        self.wait()
        hamming74_remark = TexText("（在二进制下）", color=GREY_A)
        hamming74_remark.scale(0.5).next_to(hamming74_result, RIGHT)
        self.play(
            FadeIn(hamming74_result, shift=LEFT),
            FadeOut(connections),
        )
        self.play(FadeIn(hamming74_remark))
        self.wait()
        second_column = hamming74_matrix.get_columns()[1]
        second_entry = hamming74_vector.get_row(1)
        entry_rect = SurroundingRectangle(second_entry, color=RED)
        self.play(
            second_column.animate.set_color(YELLOW),
            second_entry.animate.set_color(RED),
            hamming74_result[1].elements.animate.set_color(YELLOW),
            ShowCreation(entry_rect),
        )
        self.wait()
        hamming74_vector.add(entry_rect)
        hamming74_result.add(hamming74_remark)
        weighing_title = TexText("硬币称量", color=YELLOW)
        weighing_title.to_edge(UP)
        weighing_matrix = WeighingMatrix([
            [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1],
            [0, 1, -1, -1, 0, 0, 0, 1, -1, -1, 1, 1],
            [1, 0, -1, 1, 0, 1, -1, 0, -1, 1, 0, -1],
        ])
        # weighing_vector = VVector(*["m_{"+str(k)+"}" for k in range(1,13)])
        bias = [0] * 12
        bias[3] = -1
        weighing_vector = VVector(*bias)
        weighing_result = VGroup(Tex("="), VVector(0, 1, -1))
        weighing_result.arrange(RIGHT)
        weighing_group = VGroup(weighing_matrix, weighing_vector, weighing_result)
        weighing_group.arrange(RIGHT)
        for (fade_out_mob, fade_in_mob) in [
            (VGroup(hamming74_group, hamming74_title), weighing_matrix),
            (Mobject(), weighing_vector),
            (Mobject(), VGroup(weighing_result, weighing_title)),
        ]:
            self.play(
                FadeIn(fade_in_mob, shift=UP),
                FadeOut(fade_out_mob, shift=UP),
            )
            self.wait()
        fourth_column = weighing_matrix.get_columns()[3]
        fourth_entry = weighing_vector.get_row(3)
        entry_rect = SurroundingRectangle(fourth_entry, color=RED)
        self.play(
            fourth_column.animate.set_color(YELLOW),
            fourth_entry.animate.set_color(RED),
            weighing_result[1].elements.animate.set_color(YELLOW),
            ShowCreation(entry_rect),
        )
        self.wait()
        self.play(
            fourth_column.animate.set_color(WHITE),
            fourth_entry.animate.set_color(WHITE),
            weighing_result[1].elements.animate.set_color(WHITE),
            FadeOut(entry_rect),
        )
        self.wait()
        # Relations between counterfeit puzzle and balanced ternary
        balance_ternary_texts = VGroup(
            Tex("\\left\\{ -1,\\, 0,\\, 1 \\right\\}"), TexText("平衡三进制"),
        )
        balance_ternary_texts.set_color(GREEN)
        balance_ternary_texts.next_to(weighing_matrix, UP, buff=0.8)
        self.play(Write(balance_ternary_texts[0]), run_time=1)
        self.wait()
        self.play(FadeTransform(balance_ternary_texts[0], balance_ternary_texts[1]))
        self.wait()
        balance_ternary_title = balance_ternary_texts[1]
        self.play(
            weighing_title.animate.move_to(LEFT_SIDE / 2).to_edge(UP),
            balance_ternary_title.animate.move_to(RIGHT_SIDE / 2).to_edge(UP),
            FadeOut(weighing_group, shift=DOWN),
        )
        self.wait()
        weighing_list = BulletedList(
            "11枚外观相同的硬币",
            "至多2枚重量不同的假币",
            "假币的轻重情况未知",
            "假币相互独立（轻与重可能并存）",
            "由天平称量结果判断假币的情况",
            "最多称量5次",
            "遵循一般运算规则",
            buff=0.3,
        )
        weighing_list.scale(0.7).set_color(YELLOW_A)
        weighing_list[-1].set_color(GREY_A)
        temp_text = TexText("错误相互独立（$-1$与$+1$可能并存）\\\\")  # temporary hack
        balance_ternary_list = BulletedList(
            "11位全零的平衡三进制信息",
            "至多有2位出现错误",
            "错误的情况未知",
            "错误相互独立（$-1$与$+1$可能并存）",
            "由校验矩阵的校验结果判断错误的情况",
            "矩阵最多5行",
            "遵循三进制运算规则",
            buff=0.3,
        )
        balance_ternary_list.scale(0.7).set_color(GREEN_A)
        balance_ternary_list[-1].set_color(GREY_A)
        weighing_list.next_to(LEFT_SIDE, RIGHT, buff=0.6)
        weighing_list.next_to(weighing_title, DOWN, buff=0.5, coor_mask=[0, 1, 0])
        balance_ternary_list.next_to(ORIGIN, RIGHT, buff=0.6)
        balance_ternary_list.next_to(balance_ternary_title, DOWN, buff=0.5, coor_mask=[0, 1, 0])
        self.play(FadeIn(weighing_list))
        self.wait()
        self.play(FadeTransform(weighing_list.deepcopy(), balance_ternary_list))
        self.wait()
        ternary_golay_code_text = TexText("三进制完备码 \\\\ ternary Virtakallio–Golay code")
        ternary_golay_code_text.scale(0.9).set_color(GREEN)
        ternary_golay_code_text.move_to(RIGHT_SIDE / 2).to_edge(DOWN, buff=1)
        perfect_weighing_text = TexText("“近乎完美的”称量方案")
        perfect_weighing_text.scale(0.9).set_color(YELLOW)
        perfect_weighing_text.move_to(LEFT_SIDE / 2).to_edge(DOWN, buff=1)
        self.play(FadeIn(ternary_golay_code_text, shift=2 * DOWN))
        self.wait()
        self.play(FadeTransform(ternary_golay_code_text.deepcopy(), perfect_weighing_text))
        self.wait()


class ReferenceScene(Scene):
    def construct(self):
        refs = BulletedList(
            "Solution to the counterfeit coin problem and its generalization. arXiv:1005.1391 ",
            "Optimal non-adaptive solutions for the counterfeit coin problem. arXiv:1502.04896 ",
            "Weighing algorithms of classification and identification of situations. \\emph{Discrete Math. and Appl.}, 2015, 25(2): 69-81. ",
            buff=2,
        )
        ref_remarks = VGroup(*[
            TexText(text, alignment="")
            for text in (
                "经典的“12枚硬币称量问题”的解答，也有“39枚硬币称量问题”的答案。",
                "解决了硬币称量问题的常见变体，比如是否知道假币的轻重，是否需要确定假币的轻重，是否有额外的真币作为参考等。",
                "拓展到“$n$枚硬币中有$t$枚假币”的情况，给出了称量次数$m$的下界。花了大量篇幅讨论$n=11,\\,m=5,\\,t=2$的情况，并提供了两种“近乎完美的”称量方案。",
            )
        ])
        for remark, ref in zip(ref_remarks, refs):
            remark.scale(0.9)
            remark.next_to(ref, DOWN, aligned_edge=LEFT)
            remark.shift(RIGHT)
            ref.set_color(YELLOW)
            remark.set_color(BLUE_A)
        group = VGroup(refs, ref_remarks)
        group.set_width(FRAME_WIDTH - 2).center()
        self.add(group)
        self.wait(5)


class ThumbnailPart2(Scene):
    # Well, only the vector part...
    def construct(self):
        row_vec = TexMatrix(
            np.zeros((1, 12), dtype=np.int),
            h_buff=0.7
        )
        row_vec.scale(1.5)
        for element in row_vec.get_columns():
            one = Tex("1", fill_opacity=random.random() * 0.2)
            neg_one = Tex("-1", fill_opacity=random.random() * 0.2)
            for mob in (one, neg_one):
                mob.set_height(element.get_height())
                mob.move_to(element)
                self.add(mob)
        self.add(row_vec)
        self.wait()
