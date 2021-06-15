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
            Animation(self.get_pans()),
            Animation(self.get_stand()),
            Animation(self.get_axles()),
            Animation(self.get_screen()),
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

#####
# Scenes


class IntroTo12CoinsPuzzle(Scene):
    def construct(self):
        self.show_coins()
        self.show_pan_balance()
        self.show_questions()

    def show_coins(self):
        # Show 12 coins
        coins = VGroup(*[Coin(str(k)) for k in range(1, 13)])
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
        coins = VGroup(*[Coin(str(k)) for k in range(1, 13)])
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
            RealCoin(str(k + 1)).move_to(coin)
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
        coins = VGroup(*[Coin(str(k)) for k in range(1, 13)])
        coins.arrange(RIGHT, buff=COIN_BUFF)
        self.add(coins)
        self.coins = coins

    def construct(self):
        self.show_all_possibilities_and_codes()
        self.narrow_down_to_one()

    def show_all_possibilities_and_codes(self):
        # Arrange stuff
        light_coins = VGroup(*[
            LighterCoin(str(k + 1)).move_to(coin)
            for k, coin in enumerate(self.coins)
        ])
        heavy_coins = VGroup(*[
            HeavierCoin(str(k + 1)).move_to(coin)
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
        coins = VGroup(*[Coin(str(k)) for k in range(1, 13)])
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
            RealCoin(str(k + 1)).move_to(coin)
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
        coins = VGroup(*[Coin(str(k)) for k in range(1, 3)])
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
        extra_coins = VGroup(*[Coin(str(k)) for k in range(3, 9)])
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
        left_coins = VGroup(*[Coin(str(k)) for k in range(1, 5)])
        right_coins = VGroup(*[Coin(str(k)) for k in range(5, 9)])
        idle_coins = VGroup(*[Coin(str(k)) for k in range(9, 13)])
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
        weighing_coins = VGroup(*[Coin(str(k)) for k in range(1, 13)])
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
            RealCoin(str(k + 1)).move_to(coin)
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
            RealCoin(str(k + 9)).move_to(coin)
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
            RealCoin(str(ind + 1)).move_to(self.weighing_coins[ind])
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
            RealCoin(str(ind + 1)).move_to(self.weighing_coins[ind])
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
        # Now the exciting identity revealing!
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
        new_coins = VGroup(*[Coin(str(k)) for k in range(1, 40)])
        new_coins.arrange_in_grid(3, 13, buff=COIN_BUFF)
        new_coins.to_edge(DOWN)
        old_coins = VGroup(*[Coin(str(k)) for k in range(1, 13)])
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
                Coin(str(k), **self.coin_config)
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


class Thumbnail(IntroTo12CoinsPuzzle):
    pass
