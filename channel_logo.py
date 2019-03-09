#coding=utf-8

from manimlib.constants import *
from manimlib.utils.rate_functions import *

from manimlib.animation.creation import Write
from manimlib.animation.composition import Succession
from manimlib.animation.transform import Transform, ReplacementTransform
from manimlib.mobject.types.vectorized_mobject import VMobject, VGroup
from manimlib.mobject.svg.tex_mobject import TextMobject
from manimlib.once_useful_constructs.fractals import rotate, LindenmayerCurve
from manimlib.scene.scene import Scene


class TohruCurve(LindenmayerCurve):
    CONFIG = {
        "axiom"        : "FX",
        "rule"         : {
            "X" : "X+YF+",
            "Y" : "-FX-Y",
        },
        "pass_command" : ["X", "Y"],
        "scale_factor" : np.sqrt(2),
        "radius"       : 3.5,
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
        "num_submobjects" : 512,
    }

    def __init__(self, **kwargs):
        LindenmayerCurve.__init__(self, **kwargs)
        self.rotate(-self.order * np.pi/4)

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

    def get_start(self):
        return np.array(self.submobjects[0].points[0])

    def get_end(self):
        return np.array(self.submobjects[-1].points[-1])


class KannaCurve(TohruCurve):
    CONFIG = {
        "colors"       : [WHITE, PURPLE],
        "start_step"   : LEFT,
    }


class TwinDragon(VMobject):
    CONFIG = {
        'order' : 3,
    }
    def generate_points(self):
        if self.order < 0:
            return VMobject()
        tc = TohruCurve(order = self.order)
        kc = KannaCurve(order = self.order)
        kc.shift(tc.get_end() - kc.get_start())
        group = VGroup(tc, kc).center()
        self.add(group)


class ChannelLogo(Scene):
    CONFIG = {
        "max_order" : 18,
    }
    def construct(self):
        # Generate transformation animations of the twin dragon curve
        anims = list()
        fractal = VMobject()
        fractal.shift(UP)
        for order in range(-1, self.max_order+1):
            new_fractal = TwinDragon(order = order)
            new_fractal.shift(UP)
            run_time = 0.5 if order >= 0 else 0
            anims.append(
                Transform(
                    fractal, new_fractal,
                    submobject_mode = "all_at_once",
                    run_time = run_time,
                )
            )
            fractal = new_fractal

        # Add the channel name 
        text = TextMobject("Solara570")
        text.scale(2).to_edge(DOWN, buff = 1.2)

        # Now sit back and watch
        self.play(
            Succession(*anims, rate_func = smooth),
            Write(text, lag_factor = 2.5),
            run_time = 4.5,
        )

