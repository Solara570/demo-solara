from helpers import *

from animation.transform import *
from animation.simple_animations import *
from mobject.tex_mobject import *
from topics.fractals import *


class TohruCurve(FractalCurve):
    CONFIG = {
        "axiom"        : "FX",
        "rule"         : {
            "X" : "X+YF+",
            "Y" : "-FX-Y",
        },
        "pass_command" : ["X", "Y"],
        "scale_factor" : np.sqrt(2),
        "shift_vector" : LEFT, 
        "radius"       : 4,
        "colors"       : [ORANGE, GREEN],
        "start_step"   : RIGHT,
        "angle"        : np.pi/2,
        "order_to_stroke_width_map" : {
            3 : 3.5,
            5 : 3,
            7 : 2.5,
            10 : 2,
            13 : 1.5,
            16 : 1,
        },
        "num_submobjects" : 500,
    }

    def __init__(self, **kwargs):
        FractalCurve.__init__(self, **kwargs)
        self.rotate(-self.order*np.pi/4)
        self.shift(self.radius/2. * self.shift_vector)
        self.shift(UP)

    def expand_command_string(self, command):
        result = ""
        for letter in command:
            if letter in self.rule:
                result += self.rule[letter]
            else:
                result += letter
        return result

    def get_command_string(self):
        result = self.axiom
        for x in range(self.order):
            result = self.expand_command_string(result)
        return result

    def get_anchor_points(self):
        step = float(self.radius) * self.start_step 
        step /= (self.scale_factor**self.order)
        curr = np.zeros(3)
        result = [curr]
        for letter in self.get_command_string():
            if letter in self.pass_command:
                pass
            elif letter is "+":
                step = rotate(step, self.angle)
            elif letter is "-":
                step = rotate(step, -self.angle)
            else:
                curr = curr + step
                result.append(curr)
        return np.array(result)


class KannaCurve(TohruCurve):
    CONFIG = {
        "colors"       : [WHITE, PURPLE],
        "shift_vector" : RIGHT,
        "start_step"   : LEFT,
    }


class TwinDragon(VMobject):
    CONFIG = {
        'order' : 0,
    }
    def generate_points(self):
        self.add(TohruCurve(order = self.order))
        self.add(KannaCurve(order = self.order))
        return self


class ChannelLogo(Scene):
    def construct(self):
        # Generate transformation animations of the twin dragon curve
        animlist = list()
        fractal = TwinDragon()
        animlist.append(FadeIn(fractal, run_time = 0.5))
        for order in range(1, 19):
            new_fractal = TwinDragon(order = order)
            animlist.append(
                Transform(
                    fractal, new_fractal,
                    submobject_mode = "all_at_once",
                    run_time = 0.5,
                )
            )

        # Add the channel name 
        text = TextMobject("Solara570")
        text.scale(1.5).to_edge(DOWN, buff = 1.2)

        # Now sit back and watch
        self.play(
            Succession(*animlist),
            Write(text, rate_func = squish_rate_func(smooth, 0.1, 0.9)),
            run_time = 4.5,
        )

