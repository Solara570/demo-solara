from helpers import *
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





