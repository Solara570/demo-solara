import numpy as np
from custom.custom_helpers import *

#####
## General Functions

def generate_pattern(dim):
    """Generate a random pattern of Calisson tiling with a given dimension ``dim``."""
    pattern = np.random.randint(dim + 1, size = (dim, dim))
    # Try to make the tiling more balanced...
    for k in random.sample(range(dim), max(1, dim // 3)):
        pattern.T[k].sort()
    pattern.sort()
    pattern.T.sort()
    return pattern[-1::-1, -1::-1]

#####
## Constants
# POS = (phi, theta, distance)
# To make it more convincing, the distance is set to be HUGE.

DIAG_POS  = (np.arctan(np.sqrt(2)), np.pi/4., 1E7)
UP_POS    = (0, 0, 1E7)
FRONT_POS = (np.pi/2., 0, 1E7)
RIGHT_POS = (np.pi/2., np.pi/2., 1E7)

EG_4D_PATTERN = np.array([
    [4, 3, 2, 1],
    [4, 3, 1, 0],
    [4, 3, 1, 0],
    [3, 2, 0, 0],
])

AMM_PATTERN = np.array([
    [5, 5, 5, 4, 3],
    [5, 5, 5, 1, 0],
    [5, 5, 1, 0, 0],
    [5, 5, 1, 0, 0],
    [3, 3, 0, 0, 0],
])

MPACC_PATTERN = np.array([
    [7, 7, 7, 7, 6, 4, 3],
    [6, 6, 5, 5, 5, 2, 1],
    [6, 6, 4, 3, 2, 1, 0],
    [6, 5, 3, 2, 1, 1, 0],
    [5, 3, 2, 1, 1, 0, 0],
    [3, 2, 2, 1, 0, 0, 0],
    [3, 1, 0, 0, 0, 0, 0],
])


#####
## For CountingsDoNotChange Scene

def get_lower_bounding_array(*arrays):
    dim = np.shape(arrays[0])[0]
    lb_array = np.zeros((dim, dim), dtype = int)
    for i in range(dim):
        for j in range(dim):
            lb_array[i][j] = min(*[arr[i][j] for arr in arrays])
    return lb_array

BORDER_MAIN_PATH_ACS = [
    [9, 0], [8, 0], [8, -1], [6, -1], [6, -3], [7, -3],
    [7, -8], [6, -8], [6, -6], [4, -6], [4, -9], [2, -9],
    [2, -8], [3, -8], [3, -7], [0, -7], [0, -9], [-6, -3],
    [-6, -2], [-7, -1], [-7, 1], [-8, 2], [-8, 1], [-9, 2],
    [-9, 9], [-7, 9], [-6, 8], [-3, 8], [-2, 7], [1, 7],
    [2, 6], [0, 6], [3, 3], [4, 3], [5, 2], [6, 2],
    [7, 1], [8, 1], [9, 0]
]

BORDER_SUB_PATH_ACS = [
    [-5, 1], [-8, 4], [-8, 3], [-6, 1], [-6, -1],
    [-5, -2], [-5, 1]
]

BORDER_SUB_PATH_CTRLS = []
for k, coords in enumerate(BORDER_SUB_PATH_ACS):
    if k != len(BORDER_SUB_PATH_ACS) - 1:
        x, y = BORDER_SUB_PATH_ACS[k]
        z, w = BORDER_SUB_PATH_ACS[k+1]
        BORDER_SUB_PATH_CTRLS.append(coords)
        BORDER_SUB_PATH_CTRLS.append([2*x/3+z/3, 2*y/3+w/3])
        BORDER_SUB_PATH_CTRLS.append([x/3+2*z/3, y/3+2*w/3])
    else:
        BORDER_SUB_PATH_CTRLS.append(coords)

RIGHT_OBSOLETE_TILES = [
    (8, 8), (8, 7), (8, 6), (8, 5), (8, 4), (8, 3),
    (8, 2), (8, 1), (8, 0), (7, 8), (7, 7), (7, 6),
    (7, 5), (7, 4), (7, 3), (7, 2), (7, 1), (6, 8),
    (6, 2), (6, 1), (5, 8), (5, 7), (5, 6), (4, 8),
    (4, 7), (4, 6), (2, 7), (1, 8), (1, 7), (0, 8),
    (0, 7)
]

UP_OBSOLETE_TILES = [
    (8, 8), (8, 7), (8, 6), (8, 5), (8, 4), (8, 3),
    (8, 2), (7, 8), (7, 7), (7, 6), (7, 5), (6, 8),
    (5, 8), (5, 7), (5, 6), (4, 8), (4, 7), (4, 6),
    (3, 8), (3, 7), (3, 6), (2, 8), (2, 7), (1, 8),
]

OUT_OBSOLETE_TILES = [
    (8, 8), (8, 7), (8, 6), (7, 8), (7, 7), (6, 7),
    (6, 5), (5, 5), (4, 7), (4, 6), (4, 5)
]

OBSOLETE_TILES_INFO = [["right", (x, y)] for x, y in RIGHT_OBSOLETE_TILES] + \
                      [["up", (z, w)] for z, w in UP_OBSOLETE_TILES] + \
                      [["out", (u, v)] for u, v in OUT_OBSOLETE_TILES]

RIGHT_BOUNDING_VALUES = np.array([
    [9, 8, 7, 6, 6, 6, 6, 5, 2],
    [9, 8, 7, 6, 6, 6, 6, 5, 2],
    [9, 8, 7, 6, 6, 6, 6, 5, 2],
    [9, 8, 7, 6, 6, 6, 6, 5, 2],
    [9, 8, 7, 6, 6, 6, 6, 5, 2],
    [9, 8, 7, 6, 6, 6, 6, 5, 2],
    [9, 8, 7, 6, 6, 6, 6, 5, 2],
    [9, 8, 7, 6, 6, 6, 6, 5, 2],
    [9, 8, 7, 6, 6, 6, 6, 5, 2],
])

UP_BOUNDING_VALUES = np.array([
    [8] * 9,
    [6] * 9,
    [6] * 9,
    [6] * 9,
    [6] * 9,
    [6] * 9,
    [4] * 9,
    [0] * 9,
    [0] * 9,
])

OUT_BOUNDING_VALUES = np.array([
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 0, 0, 0, 0],
    [9, 9, 9, 9, 9, 0, 0, 0, 0],
    [9, 9, 9, 9, 9, 0, 0, 0, 0],
    [9, 9, 9, 9, 9, 0, 0, 0, 0],
    [9, 9, 9, 9, 9, 0, 0, 0, 0],
])

BOUNDING_VALUES = get_lower_bounding_array(
    RIGHT_BOUNDING_VALUES, UP_BOUNDING_VALUES, OUT_BOUNDING_VALUES,
)


#####
## For GroupTheoryViewRegionsPart Scene

SNOWFLAKE_LH = [
    [3, 0], [4, -1], [5, -1], [5, -2], [6, -3], [5, -3], [5, -4],
    [4, -3], [3, -3], [3, -4], [4, -5], [3, -5], [3, -6], [2, -5],
    [1, -5], [1, -4], [0, -3], [-1, -3], [-1, -4], [-2, -3], [-3, -3],
    [-3, -2], [-4, -1], [-3, -1],

]
SNOWFLAKE_RH = [[-x, -y] for x, y in SNOWFLAKE_LH]
SNOWFLAKE_ACS = SNOWFLAKE_LH + SNOWFLAKE_RH + [[3, 0]]

R1_ACS = [
    [2, 0], [2, -1], [3, -2], [4, -2], [4, -3], [3, -3],
    [3, -4], [2, -4], [1, -3], [2, -3], [2, -2], [0, -2],
    [0, -3], [-2, -3], [-2, -4], [-3, -4], [-3, -3], [-4, -2],
    [-3, -2], [-3, -1], [0, -1], [0, 0], [-2, 0], [-4, 2],
    [-4, 3], [-2, 3], [-2, 4], [-1, 3], [0, 3], [2, 1],
    [3, 1], [2, 2], [2, 4], [3, 4], [3, 3], [4, 2],
    [4, 0], [2, 0]
]

AVATAR_LH = [
    [3, 0], [4, 0], [5, -1], [4, -1], [4, -2], [3, -2], [3, -3],
    [2, -2], [1, -2], [1, -4], [2, -4], [2, -3], [3, -4], [3, -6],
    [2, -6], [1, -5], [1, -6], [0, -6], [0, -5], [-1, -4], [-1, -5],
    [-2, -4], [-3, -4], [-3, -3], [-2, -3], [-2, -1],
]
AVATAR_RH = [[-x, -y] for x, y in AVATAR_LH]
AVATAR_ACS = AVATAR_LH + AVATAR_RH + [[3, 0]]

R2_ACS = [
    [2, 0], [3, -1], [2, -1], [2, -2], [1, -2], [1, -4],
    [-2, -4], [-4, -2], [-4, 2], [-3, 2], [-4, 3], [-4, 4],
    [-3, 4], [-2, 3], [-2, 5], [-1, 5], [0, 4], [2, 4],
    [4, 2], [1, 2], [1, 1], [0, 2], [0, 3], [-1, 4],
    [-1, 2], [-2, 2], [-2, 1], [-3, 1], [-3, 0], [-2, -1],
    [-3, -1], [-3, -2], [-2, -3], [0, -3], [-1, -2], [-1, -1],
    [0, -2], [0, -1], [1, -1], [-1, 1], [0, 1], [1, 0], [2, 0],
]

C3_ACS = [
    [4, 0], [5, -1], [5, -4], [1, -4], [-2, -1], [-2, 1],
    [0, 1], [-1, 2], [-3, 2], [-3, -1], [0, -4], [-1, -4],
    [-4, -1], [-4, 3], [-1, 3], [1, 1], [1, -1], [2, -1],
    [2, 1], [-1, 4], [-4, 4], [-4, 5], [-1, 5], [3, 1],
    [3, -2], [1, -2], [-1, 0], [-1, -1], [1, -3], [4, -3],
    [4, 0]
]

ALL_SETTINGS = [
    (SNOWFLAKE_ACS, BLUE, [BLUE_D, BLUE_B], UL),
    (R1_ACS, GREEN, [GREEN_B, GREEN_D], RIGHT),
    (AVATAR_ACS, YELLOW, [ORANGE, GREEN, PURPLE, WHITE], UR),
    (R2_ACS, GOLD, [GOLD_D, GOLD_B], DR),
    (C3_ACS, RED, [MAROON_D, MAROON_B], LEFT)
]

#####
## Colors

DARK_GRAY = "#404040"
DARK_GREY = DARK_GRAY
LIGHT_GRAY = "#B0B0B0"
LIGHT_GREY = LIGHT_GRAY

TILE_RED, TILE_GREEN, TILE_BLUE = map(darken, [RED, GREEN, BLUE])
TILE_COLOR_SET = [TILE_GREEN, TILE_RED, TILE_BLUE]
RHOMBI_COLOR_SET = [TILE_RED, TILE_GREEN, TILE_BLUE]
L_GRADIENT_COLORS = [BLUE, GREEN, RED]


