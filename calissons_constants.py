import numpy as np
from custom.custom_helpers import *

#####
## Functions

def generate_pattern(dim):
    """Generate a random pattern of Calisson tiling with a given dimension ``dim``."""
    pattern = np.random.randint(dim + 1, size = (dim, dim))
    # Try to make the tiling more balanced...
    for k in random.sample(range(dim), max(1, dim // 3)):
        pattern.T[k].sort()
    pattern.sort()
    pattern.T.sort()
    return pattern[-1::-1, -1::-1]

def get_lower_bounding_array(*arrays):
    dim = np.shape(arrays[0])[0]
    lb_array = np.zeros((dim, dim), dtype = int)
    for i in range(dim):
        for j in range(dim):
            lb_array[i][j] = min(*[arr[i][j] for arr in arrays])
    return lb_array

#####
## Constants
# POS = (phi, theta, distance)
# To make it more convincing, the distance is set to be HUGE.

DIAG_POS  = (np.arctan(np.sqrt(2)), np.pi/4., 1E7)
UP_POS    = (0, 0, 1E7)
FRONT_POS = (np.pi/2., 0, 1E7)
RIGHT_POS = (np.pi/2., np.pi/2., 1E7)

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

BORDER_ANCHORS_COORDINATES = [
    [9, 0], [8, 0], [8, -3], [7, -3], [7, -5], [6, -5], [6, -6], [5, -6],
    [5, -7], [1, -7], [1, -8], [0, -8], [0, -9], [-6, -3], [-6, 0], [-8, 2],
    [-8, 4], [-9, 5], [-9, 9], [-8, 8], [-5, 8], [-4, 7], [-2, 7], [-1, 6],
    [1, 6], [3, 4], [4, 4], [5, 3], [6, 3], [9, 0]
]

OBSOLETE_TILES_INFO = [
    ["right", (8, 8)], ["right", (8, 7)], ["right", (8, 6)],
    ["right", (8, 5)], ["right", (8, 4)], ["right", (8, 3)],
    ["right", (8, 2)], ["right", (8, 1)], ["right", (8, 0)],
    ["right", (7, 8)], ["right", (7, 7)], ["right", (7, 6)],
    ["right", (7, 5)], ["right", (7, 4)], ["right", (7, 3)],
    ["right", (6, 8)], ["right", (6, 7)], ["right", (6, 6)],
    ["right", (6, 5)], ["right", (5, 8)], ["right", (5, 7)],
    ["right", (5, 6)], ["right", (4, 8)], ["right", (4, 7)],
    ["right", (3, 8)], ["right", (3, 7)], ["right", (2, 8)],
    ["right", (2, 7)], ["right", (1, 8)], ["right", (1, 7)],
    ["right", (0, 8)], ["up", (8, 8)], ["up", (8, 7)],
    ["up", (8, 6)], ["up", (8, 5)], ["up", (8, 4)],
    ["up", (8, 3)], ["up", (8, 2)], ["up", (8, 1)],
    ["up", (8, 0)], ["up", (7, 8)], ["up", (7, 7)],
    ["up", (7, 6)], ["up", (7, 5)], ["up", (7, 4)],
    ["up", (7, 3)], ["up", (6, 8)], ["up", (6, 7)],
    ["up", (6, 6)], ["up", (6, 5)], ["up", (5, 8)],
    ["up", (5, 7)], ["up", (4, 8)], ["up", (4, 7)],
    ["up", (3, 8)], ["out", (8, 8)], ["out", (8, 7)],
    ["out", (8, 6)], ["out", (7, 8)], ["out", (7, 7)],
    ["out", (7, 6)], ["out", (6, 8)], ["out", (6, 7)],
    ["out", (6, 6)], ["out", (5, 8)], ["out", (4, 8)],
]

FRONT_BOUNDING_VALUES = np.array([
    [9, 9, 9, 8, 7, 7, 5, 3, 0],
    [9, 9, 9, 8, 7, 7, 5, 3, 0],
    [9, 9, 9, 8, 7, 7, 5, 3, 0],
    [9, 9, 9, 8, 7, 7, 5, 3, 0],
    [9, 9, 9, 8, 7, 7, 5, 3, 0],
    [9, 9, 9, 8, 7, 7, 5, 3, 0],
    [9, 9, 9, 8, 7, 7, 5, 3, 0],
    [9, 9, 9, 8, 7, 7, 5, 3, 0],
    [9, 9, 9, 8, 7, 7, 5, 3, 0],
])

RIGHT_BOUNDING_VALUES = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8],
    [7, 7, 7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 7],
    [6, 6, 6, 6, 6, 6, 6, 6, 6],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
])

UP_BOUNDING_VALUES = np.array([
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 0],
    [9, 9, 9, 9, 9, 9, 9, 9, 0],
    [9, 9, 9, 9, 9, 9, 0, 0, 0],
    [9, 9, 9, 9, 9, 9, 0, 0, 0],
    [9, 9, 9, 9, 9, 9, 0, 0, 0],
])

BOUNDING_VALUES = get_lower_bounding_array(
    FRONT_BOUNDING_VALUES, RIGHT_BOUNDING_VALUES, UP_BOUNDING_VALUES
)

#####
## Colors

DARK_GRAY = "#404040"
DARK_GREY = DARK_GRAY
LIGHT_GRAY = "#B0B0B0"
LIGHT_GREY = LIGHT_GRAY

TILE_RED, TILE_GREEN, TILE_BLUE = map(darken, [RED, GREEN, BLUE])
TILE_COLOR_SET = [TILE_GREEN, TILE_RED, TILE_BLUE]
RHOMBI_COLOR_SET = [TILE_RED, TILE_GREEN, TILE_BLUE]


