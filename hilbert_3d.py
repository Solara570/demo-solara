from helpers import *
from animation.simple_animations import *
from animation.transform import *
from topics.fractals import *
from topics.three_dimensions import *

# Implemented the 3D version of Hilbert curve
class HilbertCurve3D(SelfSimilarSpaceFillingCurve):
    CONFIG = {
        "colors"  : [YELLOW, TEAL],
        "radius"  : 2.5,
        "offsets" : [
            RIGHT+DOWN+IN,
            LEFT+DOWN+IN,
            LEFT+DOWN+OUT,
            RIGHT+DOWN+OUT,
            RIGHT+UP+OUT,
            LEFT+UP+OUT,
            LEFT+UP+IN,
            RIGHT+UP+IN,
        ],
        "offset_to_axis_and_angle" : {
            str(RIGHT+DOWN+IN)  : [LEFT+UP+OUT  , 2*np.pi/3],
            str(LEFT+DOWN+IN)   : [RIGHT+DOWN+IN, 2*np.pi/3],
            str(LEFT+DOWN+OUT)  : [RIGHT+DOWN+IN, 2*np.pi/3],
            str(RIGHT+DOWN+OUT) : [UP           , np.pi    ],
            str(RIGHT+UP+OUT)   : [UP           , np.pi    ],
            str(LEFT+UP+OUT)    : [LEFT+DOWN+OUT, 2*np.pi/3],
            str(LEFT+UP+IN)     : [LEFT+DOWN+OUT, 2*np.pi/3],
            str(RIGHT+UP+IN)    : [RIGHT+UP+IN  , 2*np.pi/3],
        },
        "num_submobjects" : 300,
    }
    # Rewrote transform to include the rotation angle
    def transform(self, points, offset):
        copy = np.array(points)
        if str(offset) in self.offset_to_axis_and_angle:
            copy = rotate(
                copy, 
                axis = self.offset_to_axis_and_angle[str(offset)][0],
                angle = self.offset_to_axis_and_angle[str(offset)][1],
            )
        copy /= self.scale_factor,
        copy += offset*self.radius*self.radius_scale_factor
        return copy


class Show3DHilbertCurve(ThreeDScene):
    CONFIG = {
        "max_order" : 6,
    }
    def construct(self):
        # Setup
        self.set_camera_position(phi = np.pi/3, theta = 3*np.pi/4)
        self.begin_ambient_camera_rotation(np.pi/50)

        # Part 1: Increasing the order
        for order in range(1, self.max_order+1):
            if order == 1:
                fractal = HilbertCurve3D(order = 1)
                self.play(ShowCreation(fractal), run_time = 2)
                cur_order = fractal.order
            else:
                new_fractal = HilbertCurve3D(order = cur_order + 1)
                self.play(Transform(fractal, new_fractal))
                cur_order = new_fractal.order
                self.wait(2)
        self.wait(5)
        self.play(FadeOut(fractal))
        self.wait(3)

        # Part 2: Show one-touch construction
        self.play(ShowCreation(HilbertCurve3D(order = self.max_order)), run_time = 60)
        self.wait(5)

        # Part 3: Decreasing the order till it vanishes
        for k in reversed(range(1, self.max_order)):
            new_fractal = HilbertCurve3D(order = cur_order - 1)
            self.play(Transform(fractal, new_fractal))
            cur_order = new_fractal.order
            self.wait(2)
        self.play(Uncreate(fractal), run_time = 1)

        # The end
        self.stop_ambient_camera_rotation()
        self.set_camera_position(phi = 0, theta = -np.pi/2)
        author = TextMobject("@Solara570")
        author.scale(1.5)
        author.to_corner(RIGHT+DOWN)
        self.play(FadeIn(author), run_time = 1)
        self.wait(2)


class Thumbnail(ThreeDScene):
    def construct(self):
        self.set_camera_position(phi = np.pi/3, theta = 2*np.pi/5)
        self.add(HilbertCurve3D(order = 3, radius = 3))
