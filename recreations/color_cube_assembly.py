from big_ol_pile_of_manim_imports import *

# Corner pairs
CORNER_PAIRS = (
    (0, 21), (3, 22), (12, 25), (15, 26),
    (48, 37), (51, 38), (60, 41), (63, 42),
)
CORNER_PAIRS_DICT = dict([
    (element, pair[(k+1)%2]) for pair in CORNER_PAIRS for k, element in enumerate(pair)
])
CORNER_ROTATION_AXES = (
    [1, -1, 0], [1, 1, 0], [1, 1, 0], [1, -1, 0],
    [1, -1, 0], [1, 1, 0], [1, 1, 0], [1, -1, 0],
)
CORNER_ROTATION_ANGLE = PI
CORNER_ROTATION_AXIS_DICT = dict()
for pair, axis in zip(CORNER_PAIRS, CORNER_ROTATION_AXES):
    for i in pair:
        CORNER_ROTATION_AXIS_DICT[i] = axis

# Face-and-edge quadruples
FACE_EDGE_QUADS = (
    (5, 16, 47, 58), (6, 19, 44, 57), (9, 28, 35, 54), (10, 31, 32, 53),
    (17, 4, 59, 46), (18, 7, 56, 45), (33, 52, 11, 30), (34, 55, 8, 29),
    (40, 61, 2, 23), (36, 49, 14, 27), (24, 13, 50, 39), (20, 1, 62, 43),
)
FACE_EDGE_QUADS_DICT = dict([
    (element, quad[(k+1)%4]) for quad in FACE_EDGE_QUADS for k, element in enumerate(quad)
])

# Color indices for cube faces
# There may be an easy way to paint these cubes other than using
# the cyclic property of indices within each pair or quadruple,
# and I don't see it :(
COLORS = [RED, YELLOW, GREEN, BLUE]
CUBE_COLOR_INDICES = {
    0:  (0, 1, 0, 1, 1, 0), ##
    21: (3, 2, 3, 2, 2, 3), #
    3:  (0, 1, 1, 0, 1, 0), ##
    22: (3, 2, 2, 3, 2, 3), #
    12: (0, 1, 0, 1, 0, 1), ##
    25: (3, 2, 3, 2, 3, 2), #
    15: (0, 1, 1, 0, 0, 1), ##
    26: (3, 2, 2, 3, 3, 2), #
    48: (1, 0, 0, 1, 1, 0), ##
    37: (2, 3, 3, 2, 2, 3), #
    51: (1, 0, 1, 0, 1, 0), ##
    38: (2, 3, 2, 3, 2, 3), #
    60: (1, 0, 0, 1, 0, 1), ##
    41: (2, 3, 3, 2, 3, 2), #
    63: (1, 0, 1, 0, 0, 1), ##
    42: (2, 3, 2, 3, 3, 2), #
    5:  (0, 3, 1, 2, 2, 1), ####
    16: (3, 2, 0, 1, 1, 0), #
    47: (2, 1, 3, 0, 0, 3), #
    58: (1, 0, 2, 3, 3, 2), #
    6:  (0, 3, 2, 1, 2, 1), ####
    19: (3, 2, 1, 0, 1, 0), #
    44: (2, 1, 0, 3, 0, 3), #
    57: (1, 0, 3, 2, 3, 2), #
    9:  (0, 3, 1, 2, 1, 2), ####
    28: (3, 2, 0, 1, 0, 1), #
    35: (2, 1, 3, 0, 3, 0), #
    54: (1, 0, 2, 3, 2, 3), #
    10: (0, 3, 2, 1, 1, 2), ####
    31: (3, 2, 1, 0, 0, 1), #
    32: (2, 1, 0, 3, 3, 0), #
    53: (1, 0, 3, 2, 2, 3), #
    17: (1, 2, 1, 2, 3, 0), ####
    4:  (0, 1, 0, 1, 2, 3), #
    59: (3, 0, 3, 0, 1, 2), #
    46: (2, 3, 2, 3, 0, 1), #
    18: (1, 2, 2, 1, 3, 0), ####
    7:  (0, 1, 1, 0, 2, 3), #
    56: (3, 0, 0, 3, 1, 2), #
    45: (2, 3, 3, 2, 0, 1), #
    33: (2, 1, 1, 2, 3, 0), ####
    52: (1, 0, 0, 1, 2, 3), #
    11: (0, 3, 3, 0, 1, 2), #
    30: (3, 2, 2, 3, 0, 1), #
    34: (2, 1, 2, 1, 3, 0), ####
    55: (1, 0, 1, 0, 2, 3), #
    8:  (0, 3, 0, 3, 1, 2), #
    29: (3, 2, 3, 2, 0, 1), #
    40: (2, 1, 0, 3, 1, 2), ####
    61: (1, 0, 3, 2, 0, 1), #
    2:  (0, 3, 2, 1, 3, 0), #
    23: (3, 2, 1, 0, 2, 3), #
    36: (2, 1, 0, 3, 2, 1), ####
    49: (1, 0, 3, 2, 1, 0), #
    14: (0, 3, 2, 1, 0, 3), #
    27: (3, 2, 1, 0, 3, 2), #
    24: (1, 2, 0, 3, 1, 2), ####
    13: (0, 1, 3, 2, 0, 1), #
    50: (3, 0, 2, 1, 3, 0), #
    39: (2, 3, 1, 0, 2, 3), #
    20: (1, 2, 0, 3, 2, 1), ####
    1:  (0, 1, 3, 2, 1, 0), #
    62: (3, 0, 2, 1, 0, 3), #
    43: (2, 3, 1, 0, 3, 2), #
}


class ColorCubeAssembly(ThreeDScene):
    CONFIG = {
        "cube_config" : {
            "side_length" : 0.5,
            "stroke_color" : GREY,
            "stroke_width" : 0.5,
            "fill_opacity" : 1,
        },
        "default_scaling_factor" : 2.5,
    }
    def construct(self):
        # Setup cubes and its colors
        cubes = VGroup(*[
            Cube(
                **self.cube_config
            ).move_to(self.cube_config["side_length"]*(i*OUT + j*UP + k*RIGHT))
            for i in range(4) for j in range(4) for k in range(4)
        ]).center()
        for k, cube in enumerate(cubes):
            color_indices = CUBE_COLOR_INDICES[k]
            for face, ci in zip(cube.submobjects, color_indices):
                face.set_fill(color = COLORS[ci])
        self.add(cubes)
        self.cubes = cubes

        # Time to transfrom!
        self.set_camera_orientation(phi = PI/3, theta = -PI/3)
        self.begin_ambient_camera_rotation(rate = 0.01)
        # Step 0: Intro
        self.play(GrowFromCenter(cubes), run_time = 2)
        self.wait()
        self.validate_result()
        self.wait()
        # Step 1: Scatter all cubes and record its position
        self.scatter_cubes()
        self.record_indices_and_positions()
        self.wait()
        # Step 2: First transformation
        # (1) Perform a rotational transformation on 16 corner cubes
        corner_anims = self.get_corner_cube_animations()
        self.play(corner_anims, run_time = 3)
        # (2) Swap all edge and face cubes
        edge_face_anims = self.get_edge_and_face_cube_animations()
        self.play(edge_face_anims, run_time = 5)
        self.wait()
        # (3) gather small cubes and validate the resulting cube
        self.gather_cubes()
        self.validate_result()
        self.wait(3)
        # Step 3: The remaining transformations in this cycle
        for k in range(3):
            self.scatter_cubes()
            self.wait()
            corner_anims = self.get_corner_cube_animations(is_rotation = k % 2)
            edge_face_anims = self.get_edge_and_face_cube_animations()
            self.play(corner_anims, edge_face_anims, run_time = 5)
            self.wait()
            self.gather_cubes()
            self.validate_result()
            self.wait(3)
        # Step 4: Another fast cycle before ending
        self.move_camera(phi = PI/3)
        for k in range(4):
            self.scatter_cubes(run_time = 1)
            corner_anims = self.get_corner_cube_animations(is_rotation = (k+1) % 2)
            edge_face_anims = self.get_edge_and_face_cube_animations()
            self.play(corner_anims, edge_face_anims, run_time = 2)
            self.wait()
            self.gather_cubes(run_time = 1)
            self.wait()
        self.wait(3)
        # The End
        self.play(ShrinkToCenter(cubes), run_time = 2)
        self.wait(3)

    def get_corner_cube_animations(self, is_rotation = True, **kwargs):
        # 'is_rotation' is a boolean:
        # True  - Rotate all corner cubes
        # False - Swap the adjacent corner cube pairs
        anims_list = []
        for cube in self.cubes:
            cube_index = self.get_curr_index_based_on_position(cube)
            if cube_index in CORNER_ROTATION_AXIS_DICT.keys():
                # If it's rotation...
                if is_rotation:
                    axis = CORNER_ROTATION_AXIS_DICT[cube_index]
                    anim = Rotating(
                        cube, radians = CORNER_ROTATION_ANGLE,
                        axis = axis, about_point = cube.get_center(),
                        rate_func = smooth, run_time = 3,
                    )
                # If it's swapping...
                else:
                    target_index = CORNER_PAIRS_DICT[cube_index]
                    target_position = self.get_position_from_index(target_index)
                    # Make sure the path arc is correct
                    camera_vec = self.get_camera_vec()
                    diff_vec = target_position - cube.get_center()
                    axis = cross(camera_vec, diff_vec)
                    # Make sure the path arcs don't crash
                    sign = 1 if 20 < target_index < 44 else -1
                    anim = ApplyMethod(
                        cube.move_to, target_position,
                        path_arc = sign*PI/2., path_arc_axis = axis, run_time = 3,
                    )
                anims_list.append(anim)
        return AnimationGroup(*anims_list, **kwargs)
        
    def get_edge_and_face_cube_animations(self, **kwargs):
        anims_list = []
        for cube in self.cubes:
            cube_index = self.get_curr_index_based_on_position(cube)
            if cube_index in FACE_EDGE_QUADS_DICT.keys():
                target_index = FACE_EDGE_QUADS_DICT[cube_index]
                target_position = self.get_position_from_index(target_index)
                camera_vec = self.get_camera_vec()
                diff_vec = target_position - cube.get_center()
                axis = cross(camera_vec, diff_vec)
                anim = ApplyMethod(
                    cube.move_to, target_position,
                    path_arc = PI/2., path_arc_axis = axis, run_time = 3,
                )
                anims_list.append(anim)
        return AnimationGroup(*anims_list, **kwargs)

    def get_camera_vec(self):
        phi = self.camera.phi_tracker.get_value()
        theta = self.camera.theta_tracker.get_value()
        return np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])

    def scatter_cubes(self, factor = None, **kwargs):
        if factor is None:
            factor = self.default_scaling_factor
        self.play(
            AnimationGroup(*[
                ApplyMethod(cube.move_to, cube.get_center() * factor)
                for cube in self.cubes
                ],
                **kwargs
            )
        )

    def gather_cubes(self, factor = None, **kwargs):
        if factor is None:
            factor = self.default_scaling_factor
        self.scatter_cubes(1/factor, **kwargs)

    def record_indices_and_positions(self):
        self.indices_to_positions = dict()
        for k, cube in enumerate(self.cubes):
            self.indices_to_positions[k] = cube.get_center()

    def get_position_from_index(self, i):
        return self.indices_to_positions[i]

    def get_curr_index_based_on_position(self, cube, thres = 1E-4):
        # The position of each cube keeps changing while the index stays constant,
        # while ruins the transformation after. So I need a way to find the "real"
        # index for each cube based on its current position.
        curr_center = cube.get_center()
        for i, pos in self.indices_to_positions.items():
            if np.abs(get_norm(curr_center - pos)) < thres:
                return i
        raise Exception()

    def validate_result(self):
        theta_tracker = self.camera.theta_tracker
        phi_tracker = self.camera.phi_tracker
        self.play(
            theta_tracker.increment_value, PI,
            phi_tracker.set_value, 2*PI/3,
            run_time = 2,
        )
        self.wait()
        self.play(
            theta_tracker.increment_value, PI,
            phi_tracker.set_value, PI/3,
            run_time = 2
        )


