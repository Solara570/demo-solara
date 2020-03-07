###########################################################
#   An ancient project on various fractals. Beware that   #
#   things are not ordered and probably broken.           #
#                                                         #
#       https://www.bilibili.com/video/av10218601         #
###########################################################


from manimlib.mobject.geometry import Line, Polygon, RegularPolygon, Square, Circle
from manimlib.mobject.types.vectorized_mobject import VMobject, VGroup, VectorizedPoint
from manimlib.once_useful_constructs.fractals import *
from manimlib.animation.creation import ShowCreation
from manimlib.animation.transform import Transform
from manimlib.scene.scene import Scene


class SierpinskiCarpet(SelfSimilarFractal):
    CONFIG = {
        "num_subparts" : 8,
        "height" : 4,
        "colors" : [MAROON, PURPLE, RED],
    }
    def get_seed_shape(self):
        return RegularPolygon(n = 4, start_angle = np.pi/4)

    def arrange_subparts(self, *subparts):
        shift_vec = self.height * np.array([
            UP, UP+LEFT, LEFT, LEFT+DOWN, DOWN, DOWN+RIGHT, RIGHT, RIGHT+UP
        ])
        for part, vect in zip(subparts, shift_vec):
            part.move_to(vect)


class CantorDust(SelfSimilarFractal):
    CONFIG = {
        "num_subparts" : 4,
        "height" : 4,
        "colors" : [YELLOW],
        "shift_vec" : [UP+RIGHT, UP+LEFT, DOWN+LEFT, DOWN+RIGHT],
        
    }
    def get_seed_shape(self):
        return RegularPolygon(n = 4, start_angle = np.pi/4)

    def arrange_subparts(self, *subparts):
        for part, vect in zip(subparts, self.shift_vec):
            part.move_to(self.height * vect)


class CantorDust_SolidCenter(CantorDust):
    CONFIG = {
        "num_subparts" : 5,
        "height" : 4,
        "colors" : [BLUE],
        "shift_vec" : [UP+RIGHT, UP+LEFT, DOWN+LEFT, DOWN+RIGHT, ORIGIN],
        
    }
    def get_seed_shape(self):
        return RegularPolygon(n = 4, start_angle = np.pi/4)

    def arrange_subparts(self, *subparts):
        for part, vect in zip(subparts, self.shift_vec):
            part.move_to(self.height * vect)


class PentagonalFractal(SelfSimilarFractal):
    CONFIG = {
        "num_subparts" : 5,
        "colors" : [MAROON_B, RED],
        "height" : 6,
    }
    def get_seed_shape(self):
        return RegularPolygon(n = 5, start_angle = np.pi/2)

    def arrange_subparts(self, *subparts):
        for x, part in enumerate(subparts):
            part.shift(0.95*part.get_height()*UP)
            part.rotate(2*np.pi*x/5)


class PentagonalFractalSolidCenter(SelfSimilarFractal):
    CONFIG = {
        "num_subparts" : 6,
        "colors" : [MAROON_B, RED],
        "height" : 6,
    }
    def get_seed_shape(self):
        return RegularPolygon(n = 5, start_angle = np.pi/2)

    def arrange_subparts(self, *subparts):
        for x, part in enumerate(subparts):
            part.shift(0.948*part.get_height()*UP)
            part.rotate(2*np.pi*x/5)
        subparts[5].shift(0.8955*part.get_height()*DOWN)
        subparts[5].rotate(np.pi)


class HexagonalFractal(SelfSimilarFractal):
    CONFIG = {
        "num_subparts" : 6,
        "colors" : [GREEN, BLUE],
        "height" : 6,
    }
    def get_seed_shape(self):
        return RegularPolygon(n = 6, start_angle = np.pi/2)

    def arrange_subparts(self, *subparts):
        for x, part in enumerate(subparts):
            part.shift(part.get_height()*UP)
            part.rotate(2*np.pi*x/6)


class HexagonalFractalSolidCenter(SelfSimilarFractal):
    CONFIG = {
        "num_subparts" : 7,
        "colors" : [GREEN, BLUE],
        "height" : 6,
    }
    def get_seed_shape(self):
        return RegularPolygon(n = 6, start_angle = np.pi/2)

    def arrange_subparts(self, *subparts):
        for x, part in enumerate(subparts):
            part.shift(part.get_height()*UP)
            part.rotate(2*np.pi*x/6)
        subparts[6].shift(subparts[6].get_height()*DOWN)


class WonkyHexagonFractal(SelfSimilarFractal):
    CONFIG = {
        "num_subparts" : 7
    }
    def get_seed_shape(self):
        return RegularPolygon(n=6)

    def arrange_subparts(self, *subparts):
        for i, piece in enumerate(subparts):
            piece.rotate(i*np.pi/12)
        p1, p2, p3, p4, p5, p6, p7 = subparts
        center_row = VGroup(p1, p4, p7)
        center_row.arrange_submobjects(RIGHT, buff = 0)
        for p in p2, p3, p5, p6:
            p.scale_to_fit_width(p1.get_width())
        p2.move_to(p1.get_top(), DOWN+LEFT)
        p3.move_to(p1.get_bottom(), UP+LEFT)
        p5.move_to(p4.get_top(), DOWN+LEFT)
        p6.move_to(p4.get_bottom(), UP+LEFT)


class CircularFractal(SelfSimilarFractal):
    CONFIG = {
        "num_subparts" : 3,
        "colors" : [GREEN, BLUE, GREY]
    }
    def get_seed_shape(self):
        return Circle()

    def arrange_subparts(self, *subparts):
        if not hasattr(self, "been_here"):
            self.num_subparts = 3+self.order
            self.been_here = True
        for i, part in enumerate(subparts):
            theta = np.pi/self.num_subparts
            part.next_to(
                ORIGIN, UP,
                buff = self.height/(2.5*np.tan(theta))
            )
            part.rotate(i*2*np.pi/self.num_subparts)
        self.num_subparts -= 1


class HilbertCurve(SelfSimilarSpaceFillingCurve):
    CONFIG = {
        "offsets" : [
            LEFT+DOWN,
            LEFT+UP,
            RIGHT+UP,
            RIGHT+DOWN,
        ],
        "offset_to_rotation_axis" : {
            str(LEFT+DOWN)  : RIGHT+UP,
            str(RIGHT+DOWN) : RIGHT+DOWN,
        },
     }


class PeanoCurve(SelfSimilarSpaceFillingCurve):
    CONFIG = {
        "colors" : [PURPLE, TEAL],
        "offsets" : [
            LEFT+DOWN,
            LEFT,
            LEFT+UP,
            UP,
            ORIGIN,
            DOWN,
            RIGHT+DOWN,
            RIGHT,
            RIGHT+UP,
        ],
        "offset_to_rotation_axis" : {
            str(LEFT)   : UP,       
            str(UP)     : RIGHT,    
            str(ORIGIN) : LEFT+UP,  
            str(DOWN)   : RIGHT, 
            str(RIGHT)  : UP,   
        },
        "scale_factor" : 3,
        "radius_scale_factor" : 2.0/3,
    }


class FlowSnake(LindenmayerCurve):
    CONFIG = {
        "colors" : [YELLOW, GREEN],
        "axiom"       : "A",
        "rule" : {
            "A" : "A-B--B+A++AA+B-",
            "B" : "+A-BB--B-A++A+B",
        },
        "radius"       : 6, #TODO, this is innaccurate
        "scale_factor" : np.sqrt(7),
        "start_step"   : RIGHT,
        "angle"        : -np.pi/3,
    }
    def __init__(self, **kwargs):
        LindenmayerCurve.__init__(self, **kwargs)
        self.rotate(-self.order*np.pi/9)


class SierpinskiArrowheadCurve(LindenmayerCurve):
    CONFIG = {
        "colors" : [RED, WHITE, RED],
        "axiom" : "B",
        "rule" : {
            "A" : "+B-A-B+",
            "B" : "-A+B+A-",
        },
        "radius"       : 6, #TODO, this is innaccurate
        "scale_factor" : 2,
        "start_step"   : RIGHT,
        "angle"        : -np.pi/3,
        "order_to_stroke_width_map" : {
            3 : 3,
            6 : 2,
            10 : 1,
        },
    }
    def __init__(self, **kwargs):
        LindenmayerCurve.__init__(self, **kwargs)
        self.shift((0.7 + (1- 1./(self.order + 1)) * 0.1) * DOWN)


class SierpinskiBoundary(LindenmayerCurve):
    CONFIG = {
        "colors" : [PURPLE, MAROON, PURPLE],
        "axiom"  : "F-G-G",
        "rule"  : {
            "F" : "F-G+F+G-F",
            "G" : "GG",
        },
        "radius" : 6,
        "scale_factor" : 2,
        "start_step" : LEFT,
        "angle"  : 2*np.pi/3,
    }


class KochSnowFlake(LindenmayerCurve):
    CONFIG = {
        "colors" : [BLUE_D, WHITE, BLUE_D],
        "axiom"        : "A--A--A--",
        "rule"         : {
            "A" : "A+A--A+A"
        },
        "radius"       : 4,
        "scale_factor" : 3,
        "start_step"   : RIGHT,
        "angle"        : np.pi/3,
        "order_to_stroke_width_map" : {
            3 : 3,
            5 : 2,
            6 : 1,
        },
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        self.scale_factor = 2*(1+np.cos(self.angle))
        LindenmayerCurve.__init__(self, **kwargs)
        self.center()
        if not self.order:
            self.shift(self.radius * DOWN * 2./15.)


class KochAntiSnowFlake(KochSnowFlake):
    CONFIG = {
        "axiom"  : "A++A++A",
        "colors" : [YELLOW, YELLOW, GOLD, YELLOW],
        "radius" : 5,
        "order_to_stroke_width_map" : {
            3 : 3,
            5 : 2,
            6 : 1,
        },
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        self.scale_factor = 2*(1+np.cos(self.angle))
        LindenmayerCurve.__init__(self, **kwargs)
        self.center()


class QuadraticKoch(LindenmayerCurve):
    CONFIG = {
        "colors" : [YELLOW, WHITE, MAROON_B],
        "axiom"        : "A",
        "rule"         : {
            "A" : "A+A-A-AA+A+A-A"
        },
        "radius"       : 4,
        "scale_factor" : 4,
        "start_step"   : RIGHT,
        "angle"        : np.pi/2
    }
    def __init__(self, **kwargs):
        LindenmayerCurve.__init__(self, **kwargs)
        self.center()


class QuadraticKochIsland(QuadraticKoch):
    CONFIG = {
        "colors" : [YELLOW, WHITE, MAROON_B, WHITE, YELLOW],
        "axiom" : "A+A+A+A",
        "order_to_stroke_width_map" : {
            2 : 3,
            3 : 2,
            4 : 1,
        },
    }
    def __init__(self, **kwargs):
        LindenmayerCurve.__init__(self, **kwargs)
        self.center()


class LongSegmentCurve(LindenmayerCurve):
    CONFIG = {
        "colors" : [GREEN, WHITE, BLUE],
        "axiom"        : "A",
        "rule"         : {
            "A" : "A-A+A+AAA-AA+AA-A-AAA-AA+AAAA+A+AAA-AA+AAA-A-AAAA-AA+AAA+A+AA-AA+AAA-A-A+A"
        },
        "radius"       : 8,
        "scale_factor" : 10,
        "start_step"   : RIGHT,
        "angle"        : np.pi/2,
        "order_to_stroke_width_map" : {
            1 : 3,
            2 : 2,
            3 : 1,
        },
    }
    def __init__(self, **kwargs):
        LindenmayerCurve.__init__(self, **kwargs)
        self.center()


class LongSegmentIsland(LongSegmentCurve):
    CONFIG = {
        "colors" : [GREEN, WHITE, BLUE, WHITE, GREEN],
        "axiom" : "A-A-A-A",
        "radius" : 4,
    }


## Lindenmayer curve with pass command
class LindenmayerCurveWithPass(FractalCurve):
    CONFIG = {
        "axiom"        : "A",
        "rule"         : {},
        "pass_command" : [],
        "scale_factor" : 2,
        "radius"       : 3,
        "start_step"   : RIGHT,
        "angle"        : np.pi/2,
    }

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
            jump_flag = False
            for pm in self.pass_command:
                if (letter == pm):
                    jump_flag = True
                    break
            if (jump_flag):
                continue
            if letter is "+":
                step = rotate(step, self.angle)
            elif letter is "-":
                step = rotate(step, -self.angle)
            else:
                curr = curr + step
                result.append(curr)
        return np.array(result) - center_of_mass(result)


class QuadraticTypeI(LindenmayerCurve):
    CONFIG = {
        "colors" : [RED, YELLOW, RED],
        "axiom"        : "A",
        "rule"         : {
            "A" : "A-A+A+A-A"
        },
        "radius"       : 10,
        "scale_factor" : 3,
        "start_step"   : RIGHT,
        "angle"        : np.pi/2,
        "order_to_stroke_width_map" : {
            2 : 3,
            4 : 2,
            6 : 1,
        },
    }


class QuadraticTypeIIsland(QuadraticTypeI):
    CONFIG = {
        "colors" : [RED, YELLOW, RED, YELLOW, RED],
        "axiom"  : "A+A+A+A",
        "radius" : 3.5,
    }
    def __init__(self, **kwargs):
        QuadraticTypeI.__init__(self, **kwargs)
        self.center()


class QuadraticTypeISquare(QuadraticTypeI):
    CONFIG = {
        "colors" : [RED, YELLOW, RED, YELLOW, RED],
        "axiom"  : "A-A-A-A",
        "radius" : 6,
    }
    def __init__(self, **kwargs):
        QuadraticTypeI.__init__(self, **kwargs)
        self.center()


class HilbertCurve_LSystem(LindenmayerCurveWithPass):
    CONFIG = {
        "axiom"        : "A",
        "rule"         : {
            "A" : "-BF+AFA+FB-",
            "B" : "+AF-BFB-FA+",
        },
        "pass_command" : ["A", "B"],
        "radius"       : 6,
        "scale_factor" : 2,
        "start_step"   : RIGHT,
        "angle"        : -np.pi/2,
        "order_to_stroke_width_map" : {
            4 : 3,
        },
    }


class MooreCurve_LSystem(LindenmayerCurveWithPass):
    CONFIG = {
        "colors"       : [RED, GREEN, RED, GREEN, RED],
        "axiom"        : "LFL+F+LFL",
        "rule"         : {
            "L" : "-RF+LFL+FR-",
            "R" : "+LF-RFR-FL+",
        },
        "pass_command" : ["L", "R"],
        "radius"       : 3,
        "scale_factor" : 2,
        "start_step"   : UP,
        "angle"        : np.pi/2,
        "order_to_stroke_width_map" : {
            2 : 3,
            4 : 2.5,
            6 : 2,
            8 : 1.5,
            10 : 1,
        },
    }

class MooreSquare_LSystem(MooreCurve_LSystem):
    CONFIG = {
        "axiom"        : "LFL+F+LFL+F",
        "colors" : [GREEN, BLUE, GREEN],
    }

class DragonCurve_LSystem(LindenmayerCurveWithPass):
    CONFIG = {
        "axiom"        : "FX",
        "rule"         : {
            "X" : "X+YF+",
            "Y" : "-FX-Y",
        },
        "pass_command" : ["X", "Y"],
        "radius"       : 6,
        "scale_factor" : np.sqrt(2),
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
    }

    def __init__(self, **kwargs):
        LindenmayerCurveWithPass.__init__(self, **kwargs)
        self.rotate(-self.order*np.pi/4)


class TwinDragonCurve(DragonCurve_LSystem):
    CONFIG = {
        "axiom"  : "FX+FX+",
        "colors" : [RED, GREEN, BLUE],
        "rule"   : {
            "X" : "X+YF",
            "Y" : "FX-Y",
        },
        "radius" : 5,
    }


class Terdragon(LindenmayerCurve):
    CONFIG = {
        "colors" : [RED, GREEN, BLUE],
        "axiom"        : "A",
        "rule"         : {
            "A" : "A+A-A"
        },
        "radius"       : 10,
        "scale_factor" : np.sqrt(3),
        "start_step"   : RIGHT,
        "angle"        : 2*np.pi/3,
        "order_to_stroke_width_map" : {
            2 : 3.5,
            4 : 3,
            6 : 2.5,
            8 : 2,
            10 : 1.5,
            12 : 1,
        },
    }

    def __init__(self, **kwargs):
        LindenmayerCurve.__init__(self, **kwargs)
        self.rotate(-self.order*np.pi/6)


class Heighwaymed(LindenmayerCurveWithPass):
    CONFIG = {
        "colors" : [TEAL, BLUE, PURPLE],
        "axiom"        : "-X",
        "rule"         : {
            "X" : "X+F+Y",
            "Y" : "X-F-Y",
        },
        "pass_command" : ["X", "Y"],
        "radius"       : 4.5,
        "scale_factor" : np.sqrt(2),
        "start_step"   : RIGHT,
        "angle"        : np.pi/4,
        "order_to_stroke_width_map" : {
            3 : 3.5,
            5 : 3,
            7 : 2.5,
            10 : 2,
            13 : 1.5,
            16 : 1,
        },
    }
    def __init__(self, **kwargs):
        LindenmayerCurveWithPass.__init__(self, **kwargs)
        self.rotate((1-self.order) * np.pi/4)


class SierpinskiFillingCurve(LindenmayerCurveWithPass):
    CONFIG = {
        "colors" : [GREEN, BLUE, GREEN],
        "axiom"        : "L--F--L--F",
        "rule"         : {
            "L" : "+R-F-R+",
            "R" : "-L+F+L-",
        },
        "pass_command" : ["L", "R"],
        "radius"       : 3,
        "scale_factor" : np.sqrt(2),
        "start_step"   : RIGHT,
        "angle"        : np.pi/4,
        "order_to_stroke_width_map" : {
            3 : 3.5,
            5 : 3,
            7 : 2.5,
            9 : 2,
            11 : 1.5,
            13 : 1,
        },
    }
    def __init__(self, **kwargs):
        LindenmayerCurveWithPass.__init__(self, **kwargs)
        self.center()


class PeanoCurve_LSystem(LindenmayerCurveWithPass):
    CONFIG = {
        "colors" : [PURPLE, TEAL],
        "axiom"        : "X",
        "rule"         : {
            "X" : "XFYFX+F+YFXFY-F-XFYFX",
            "Y" : "YFXFY-F-XFYFX+F+YFXFY",
        },
        "pass_command" : ["X", "Y"],
        "radius"       : 6,
        "scale_factor" : 3,
        "start_step"   : UP,
        "angle"        : -np.pi/2,
        "order_to_stroke_width_map" : {
            3 : 3,
            6 : 2,
        },
    }


class WeirdTriangle(LindenmayerCurveWithPass):
    CONFIG = {
        "colors" : [GREEN, TEAL, BLUE],
        "axiom"        : "Q",
        "rule"         : {
            "F" : "",
            "P" : "--FR++++FS--FU",
            "Q" : "FT++FR----FS++",
            "R" : "++FP----FQ++FT",
            "S" : "FU--FP++++FQ--",
            "T" : "+FU--FP+",
            "U" : "-FQ++FT-",
        },
        "pass_command" : ["P", "Q", "R", "S", "T", "U"],
        "radius"       : 5,
        "scale_factor" : (np.sqrt(5)+1.)/2.,
        "start_step"   : RIGHT,
        "angle"        : np.pi/5,
        "order_to_stroke_width_map" : {
            3 : 2.5,
            5 : 3,
            7 : 2.5,
            10 : 2,
            13 : 1.5,
            16 : 1,
        },
    }
    def __init__(self, **kwargs):
        LindenmayerCurveWithPass.__init__(self, **kwargs)
        self.center()
#        self.shift((0.7 + (1- 1./(self.order + 1)) * 0.1) * DOWN+ RIGHT)


class MediumSegmentCurve(LindenmayerCurve):
    CONFIG = {
        "colors" : [RED, WHITE, PURPLE],
        "axiom"        : "A",
        "rule"         : {
            "A" : "-A+A-A-A+A+AA-A+A+AA+A-A-AA+AA-AA+A+A-AA-A-A+AA-A-A+A+A-A+",
        },
        "radius"       : 6.5,
        "scale_factor" : 8,
        "start_step"   : RIGHT,
        "angle"        : np.pi/2,
        "order_to_stroke_width_map" : {
            1 : 3,
            2 : 2,
            3 : 1,
        },
    }
    def __init__(self, **kwargs):
        LindenmayerCurve.__init__(self, **kwargs)
        self.center()


class MediumSegmentIsland(MediumSegmentCurve):
    CONFIG = {
        "colors" : [RED, WHITE, PURPLE, WHITE, RED],
        "axiom" : "A-A-A-A",
        "radius" : 3.5,
    }


class PentaTree(LindenmayerCurve):
    CONFIG = {
        "colors" : [RED, MAROON_A, RED],
        "axiom"  : "F-F-F-F-F",
        "rule"   : {
            "F" : "F-F++F+F-F-F",
        },
        "radius"       : 4,
        "scale_factor" : (3.+np.sqrt(5))/2.,
        "start_step"   : LEFT,
        "angle"        : 2.*np.pi/5.,
        "order_to_stroke_width_map" : {
            1 : 3.5,
            2 : 3,
            3 : 2.5,
            4 : 2,
            5 : 1.5,
            6 : 1,
        },
    }

    def __init__(self, **kwargs):
        LindenmayerCurve.__init__(self, **kwargs)
        self.rotate(-self.order*np.pi/5 - np.pi/10)
        self.center()


class TweakedKochCurve(LindenmayerCurve):
    CONFIG = {
        "colors" : [TEAL, BLUE, PURPLE],
        "axiom"        : "A",
        "rule"         : {
            "A" : "+A--A+",
            "+" : "-",
            "-" : "+",
        },
        "radius"       : 12,
        "scale_factor" : np.sqrt(3),
        "start_step"   : RIGHT,
        "angle"        : np.pi/6,
        "order_to_stroke_width_map" : {
            3 : 3.5,
            5 : 3,
            8 : 2,
            11 : 1,
        },
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        LindenmayerCurve.__init__(self, **kwargs)
        self.rotate(np.pi * (self.order+1), axis = RIGHT)


class TwinDragonClass(FractalCurve):
    CONFIG = {
        "axiom"        : "A",
        "rule"         : {},
        "pass_command" : [],
        "scale_factor" : 2,
        "radius"       : 3,
        "start_step"   : RIGHT,
        "angle"        : np.pi/2,
    }

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
            jump_flag = False
            for pm in self.pass_command:
                if (letter == pm):
                    jump_flag = True
                    break
            if (jump_flag):
                continue
            if letter is "+":
                step = rotate(step, self.angle)
            elif letter is "-":
                step = rotate(step, -self.angle)
            else:
                curr = curr + step
                result.append(curr)
        return np.array(result)


class DragonCurve_LSystem1(TwinDragonClass):
    CONFIG = {
        "colors"       : [ORANGE, GREEN],
        "axiom"        : "FX",
        "rule"         : {
            "X" : "X+YF+",
            "Y" : "-FX-Y",
        },
        "pass_command" : ["X", "Y"],
        "radius"       : 5,
        "scale_factor" : np.sqrt(2),
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
    }

    def __init__(self, **kwargs):
        TwinDragonClass.__init__(self, **kwargs)
        self.rotate(-self.order*np.pi/4)
        self.shift(self.radius/2. * LEFT)


class DragonCurve_LSystem2(DragonCurve_LSystem1):
    CONFIG = {
        "colors"       : [WHITE, PURPLE],
        "start_step"   : LEFT,
    }

    def __init__(self, **kwargs):
        TwinDragonClass.__init__(self, **kwargs)
        self.rotate(-self.order*np.pi/4)
        self.shift(self.radius/2. * RIGHT)


