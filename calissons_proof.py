#coding=utf-8

from big_ol_pile_of_manim_imports import *
from custom.custom_animations import *
from custom.custom_helpers import *
from calissons import *
from calissons_constants import *

# self.skip_animations
# self.skipping()
# self.revert_to_original_skipping_status()

#####
## Constants

CB_DARK  = "#825201"
CB_LIGHT = "#B69B4C"

#####
## Methods

def sort_coords(coords_list):
    # First coord_x, then coord_y
    coords_list.sort(key = lambda p: (p[0], p[1]))

def set_projection_orientation(ct_3d):
    ct_3d.move_to(ORIGIN)
    ct_3d.rotate(np.arctan(np.sqrt(2)), DR)
    ct_3d.rotate(-3*PI/4)

def get_ct_3d_xoy_projection(ct_3d, ct_grid = None):
    ct_2d = ct_3d.deepcopy()
    # Scale to match the tiling grid
    if ct_grid is not None:
        projection_factor = np.sqrt(6)/2.  # aka 1./np.sin(np.arctan(np.sqrt(2)))
        scale_factor = ct_grid.get_unit_size() / ct_2d.get_unit_size() * projection_factor
        ct_2d.scale(scale_factor)
    # Set the correct orientation
    set_projection_orientation(ct_2d)
    # Project onto the xOy plane
    ct_2d.apply_function(lambda p: [p[0], p[1], 0])
    ct_2d.set_shade_in_3d(False)
    return ct_2d


#####
## Animations

class Scatter(MoveToTarget):
    def __init__(self, mobject, center = None, scale_factor = 1.5, **kwargs):
        raise Warning("Scatter animation will overwrite mobject.target!")
        if center is None:
            center = mobject.get_center()
        mobject.generate_target()
        for source_mob, target_mob in zip(mobject.submobjects, mobject.target.submobjects):
            source_vector = source_mob.get_center_of_mass() - center
            shift_vector = source_vector * (scale_factor - 1)
            target_mob.shift(shift_vector)
        MoveToTarget.__init__(self, mobject, **kwargs)


#####
## Mobjects

class CalissonTilesCounter(VMobject):
    CONFIG = {
        "tile_types"  : [RRhombus, HRhombus, LRhombus],
        "tile_colors" : [TILE_RED, TILE_GREEN, TILE_BLUE],
        "tile_stroke_width" : 2,
        "tile_stroke_color" : WHITE,
        "tile_fill_opacity" : 1,
        "height" : 6,
        "matching_direction" : RIGHT,
        "matching_buff" : 1,
        "counter_buff" : 0.8,
    }
    def __init__(self, counting_arg = None, matching_tiling = None, **kwargs):
        self.matching_tiling = matching_tiling
        if matching_tiling is not None and counting_arg is None:
            counting_arg = matching_tiling.get_dimension() ** 2
        try:
            counting_arg + 0
        except:
            assert (len(counting_arg) == 3)
            self.tile_nums = counting_arg
        else:
            self.tile_nums = [counting_arg] * 3
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        pairs = VGroup(*[
            VGroup(tile_type(), TexMobject("\\times %s" % str(num)))
            for tile_type, num in zip(self.tile_types, self. tile_nums)
        ])
        for pair in pairs:
            tile, text = pair
            tile.set_stroke(width = self.tile_stroke_width, color = self.tile_stroke_color)
            tile.set_fill(opacity = self.tile_fill_opacity)
            text.set_color(tile.get_fill_color())
            text.scale(1.5)
            pair.arrange_submobjects(RIGHT)
        pairs.arrange_submobjects(
            # DOWN, index_of_submobject_to_align = 0, aligned_edge = RIGHT, buff = self.counter_buff
            DOWN, index_of_submobject_to_align = 1, aligned_edge = LEFT, buff = self.counter_buff
        )
        if self.matching_tiling is None:
            height = self.height
        else:
            height = self.matching_tiling.get_height() * 0.8
        pairs.set_height(height)
        if self.matching_tiling is not None:
            pairs.next_to(
                self.matching_tiling, 
                direction = self.matching_direction,
                buff = self.matching_buff
            )
        self.add(pairs)
        self.pairs = pairs

    def get_tiles(self):
        return VGroup(*[self.pairs[k][0] for k in range(3)])

    def get_texts(self):
        return VGroup(*[self.pairs[k][1] for k in range(3)])
    
    def get_nums(self):
        texts = self.get_texts()
        return VGroup(*[texts[k][1:] for k in range(3)])


class CalissonTilingGrid(VMobject):
    CONFIG = {
        "side_length" : 4,
        "unit_size" : 0.8,
        "grid_lines_type" : Line,
        "grid_lines_config" : {
            "stroke_width" : 1,
            "color" : GREY,
        },
        "grid_points_config" : {
            "radius" : 0.05,
            "color" : BLUE,
        },
        "grid_triangles_config" : {
            "colors" : [CB_DARK, CB_LIGHT],
            "opacity" : 0.75,
        },
        "grid_boundary_config" : {
            "stroke_width" : 5,
            "color" : BLUE,
        },
    }
    def generate_points(self):
        self.setup_grid_basics()
        self.add(self.grid_center, self.basis_vec_x, self.basis_vec_y)
        # self.grid_lines = VGroup()
        # self.grid_points = VGroup()
        # self.grid_triangles = VGroup()
        # self.boundary = VMobject()

    def setup_grid_basics(self):
        # The intent here is to bury some invisible Mobjects inside the grid,
        # which will make other methods functional (like coords_to_point())
        # even when the grid is shifted, stretched or rotated.
        self.grid_center = VectorizedPoint()
        self.basis_vec_x = Line(
            ORIGIN, self.unit_size * np.array((0., 1., 0.)),
            stroke_width = 0, color = BLACK
        )
        self.basis_vec_y = Line(
            ORIGIN, self.unit_size * np.array((np.sqrt(3)/2, 0.5, 0.)),
            stroke_width = 0, color = BLACK
        )
        # 2 variables (x, y) are enough to pinpoint locations,
        # but we need another variable z to help establish the grid.
        self.x_min, self.y_min, self.z_min = [-self.side_length] * 3
        self.x_max, self.y_max, self.z_max = [self.side_length] * 3

    def generate_grid_lines(self):
        grid_lines = VGroup()
        LineType = self.grid_lines_type
        # Grid lines along x direction - Down Left -> Up Right
        for x in range(self.x_min, self.x_max+1):
            start_point = self.coords_to_point(x, max(self.y_min, self.z_min-x))
            end_point = self.coords_to_point(x, min(self.y_max, self.z_max-x))
            line = LineType(start_point, end_point, **self.grid_lines_config)
            grid_lines.add(line)
        # Grid lines along y direction - Down -> Up
        for y in range(self.y_min, self.y_max+1):
            start_point = self.coords_to_point(max(self.x_min, self.z_min-y), y)
            end_point = self.coords_to_point(min(self.x_max, self.z_max-y), y)
            line = LineType(start_point, end_point, **self.grid_lines_config)
            grid_lines.add(line)
        # Grid lines along z direction - Up Left -> Down Right
            for z in range(self.z_min, self.z_max+1):
                valid_grid_points = list(filter(
                    self.is_valid_grid_point,
                    [(x, z-x) for x in range(self.x_min, self.x_max+1)]
                ))
                if len(valid_grid_points) >= 2:
                    start_point = self.coords_to_point(*valid_grid_points[0])
                    end_point = self.coords_to_point(*valid_grid_points[-1])
                    line = LineType(start_point, end_point, **self.grid_lines_config)
                    grid_lines.add(line)
        return grid_lines

    def generate_grid_points(self):
        grid_points = VGroup()
        valid_grid_coords = list(filter(
            self.is_valid_grid_point,
            [(x, y) for x in range(self.x_min, self.x_max+1) for y in range(self.y_min, self.y_max+1)]
        ))
        for x, y in valid_grid_coords:
            dot = Dot(self.coords_to_point(x, y), **self.grid_points_config)
            grid_points.add(dot)
        return grid_points

    def generate_grid_triangles(self):
        def get_coord_func(gp):
            return self.point_to_coords(gp.get_center())
        def get_point_func(gp_coord):
            return self.coords_to_point(*gp_coord)

        grid_triangles = VGroup()
        self.grid_triangles_coords_combs = list()
        for gp0 in self.get_grid_points():
            gp0_coords = get_coord_func(gp0)
            gp0_adjacent = self.get_adjacent_grid_coords(gp0_coords)
            gp0_adjacent.append(gp0_adjacent[0])
            for gp1_coords, gp2_coords in zip(gp0_adjacent[:-1], gp0_adjacent[1:]):
                if self.is_valid_grid_triangle(gp0_coords, gp1_coords, gp2_coords):
                    coords_comb = [gp0_coords, gp1_coords, gp2_coords]
                    sort_coords(coords_comb)
                    if coords_comb not in self.grid_triangles_coords_combs:
                        self.grid_triangles_coords_combs.append(coords_comb)
        for coords_comb in self.grid_triangles_coords_combs:
            gp0_coords, gp1_coords, gp2_coords = coords_comb
            gp0, gp1, gp2 = map(get_point_func, coords_comb)
            grid_triangle = Polygon(
                gp0, gp1, gp2, stroke_width = 0,
                fill_color = self.get_grid_triangle_color(coords_comb),
                fill_opacity = self.get_grid_triangle_opacity(),
            )
            grid_triangles.add(grid_triangle)
        return grid_triangles

    def generate_grid_boundary(self):
        anchors = [
            self.coords_to_point(x, y)
            for x, y in [
                (self.x_max, 0), (0, self.y_max), (self.x_min, self.y_max),
                (self.x_min, 0), (0, self.y_min), (self.x_max, self.y_min),
                (self.x_max, 0)
            ]
        ]
        grid_boundary = VMobject(**self.grid_boundary_config)
        grid_boundary.set_anchor_points(anchors, mode = "corners")
        return grid_boundary

    def add_boundary(self):
        self.boundary = self.generate_grid_boundary()
        self.add(self.boundary)
        return self

    def add_grid_lines(self):
        self.grid_lines = self.generate_grid_lines()
        self.add(self.grid_lines)
        return self

    def add_grid_points(self):
        self.grid_points = self.generate_grid_points()
        self.add(self.grid_points)
        return self

    def add_grid_triangles(self):
        self.grid_triangles = self.generate_grid_triangles()
        self.add(self.grid_triangles)
        return self

    def coords_to_point(self, x, y):
        basis_x, basis_y = self.get_basis()
        return self.get_grid_center() + x * basis_x + y * basis_y

    def point_to_coords(self, point):
        basis_x, basis_y = self.get_basis()
        grid_center = self.get_grid_center()
        dot_ii = (np.linalg.norm(basis_x))**2
        dot_jj = (np.linalg.norm(basis_y))**2
        dot_ij = np.dot(basis_x, basis_y)
        dot_pi = np.dot(point - grid_center, basis_x)
        dot_pj = np.dot(point - grid_center, basis_y)
        A = np.array([[dot_ii, dot_ij], [dot_ij, dot_jj]])
        b = np.array([dot_pi, dot_pj])
        x, y = np.around(np.linalg.solve(A, b), decimals = 5)
        return x, y

    def is_valid_grid_point(self, coords):
        x, y = coords
        x_in_range = self.x_min <=  x  <= self.x_max
        y_in_range = self.y_min <=  y  <= self.y_max
        z_in_range = self.z_min <= x+y <= self.z_max
        return all([x_in_range, y_in_range, z_in_range])

    def is_adjacent(self, coords_A, coords_B):
        return coords_B in self.get_adjacent_grid_coords(coords_A)

    def is_valid_grid_triangle(self, coords_A, coords_B, coords_C):
        return all([
            self.is_adjacent(coords_x, coords_y)
            for coords_x, coords_y in it.combinations([coords_A, coords_B, coords_C], 2)
        ])

    def get_grid_triangle_color(self, sorted_coords_comb):
        color1, color2 = self.grid_triangles_config["colors"]
        gp0_coords, gp1_coords, gp2_coords = sorted_coords_comb
        # If the triangle is pointing to the right -> Color 1, else -> Color 2.
        return color1 if gp1_coords[1] > gp0_coords[1] else color2

    def get_grid_triangle_opacity(self):
        return self.grid_triangles_config["opacity"]

    def get_adjacent_grid_coords(self, grid_point_coords):
        x, y = grid_point_coords
        possible_coords = [(x+1, y), (x, y+1), (x-1, y+1), (x-1, y), (x, y-1), (x+1, y-1)]
        adjacent_coords = list(filter(self.is_valid_grid_point, possible_coords))
        return adjacent_coords

    def get_basis(self):
        return self.basis_vec_x.get_vector(), self.basis_vec_y.get_vector()

    def get_unit_size(self):
        return self.unit_size

    def get_grid_center(self):
        return self.grid_center.get_center()

    def get_grid_lines(self):
        if not hasattr(self, "grid_lines"):
            self.grid_lines = self.generate_grid_lines()
        return self.grid_lines

    def get_grid_points(self):
        if not hasattr(self, "grid_points"):
            self.grid_points = self.generate_grid_points()
        return self.grid_points

    def get_grid_triangle(self, coords_comb):
        coords_comb_copy = coords_comb[:]
        sort_coords(coords_comb_copy)
        gp0_coords, gp1_coords, gp2_coords = coords_comb_copy
        if not self.is_valid_grid_triangle(gp0_coords, gp1_coords, gp2_coords):
            raise Exception("There's no such grid triangle")
        triangle_index = self.grid_triangles_coords_combs.index(coords_comb_copy)
        return self.grid_triangles[triangle_index]

    def get_grid_triangles(self):
        if not hasattr(self, "grid_triangles"):
            self.grid_triangles = self.generate_grid_triangles()
        return self.grid_triangles

    def get_grid_boundary(self):
        if not hasattr(self, "grid_boundary"):
            self.grid_boundary = self.generate_grid_boundary()
        return self.grid_boundary

    # The following methods are only used to create a better visual effect
    def get_randomized_copy(self, vmob):
        vmob_copy = vmob.deepcopy()
        mobs = vmob_copy.submobjects
        random.shuffle(mobs)
        vmob_copy.submobjects = mobs
        return vmob_copy

    def get_randomized_line_copy(self):
        return self.get_randomized_copy(self.get_grid_lines())

    def get_randomized_triangle_copy(self):
        return self.get_randomized_copy(self.get_grid_triangles())


### Need further tweaking
class CalissonTiling2D(CalissonTiling3D):
    CONFIG = {
        "dumbbell_config" : {
            "point_size" : 0.07,
            "stem_width" : 0.03,
            "outline_color" : BLACK,
            "outline_width" : 0,
        },
    }
    def __init__(self, ct_3d, ct_grid = None, **kwargs):
        if ct_grid is None:
            ct_grid = CalissonTilingGrid()
        self.ct_3d = ct_3d
        self.ct_grid = ct_grid
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        self.ct_2d = get_ct_3d_xoy_projection(self.ct_3d, self.ct_grid)
        self.border, self.tiles = self.ct_2d
        self.add(self.border, self.tiles)
        self.tiles_coords = list()
        self.tiles_types = list()
        for tile_set in self.tiles:
            tile_set_coords_list = [self.get_tile_coords(tile) for tile in tile_set]
            self.tiles_coords.append(tile_set_coords_list)

    def get_all_tile_types(self):
        return ["out", "up", "right"]

    def get_all_tile_colors(self):
        return self.tile_colors

    def get_tile_type(self, tile):
        """
            3 different tile types:
            a) "out", HRhombus, GREEN color: (x, y), (x, y+1), (x+1, y-1), (x+1, y);
                                             0 -> Upright -> Left -> Upright
            b) "up", RRhombus, RED color: (x, y), (x+1, y-1), (x+1, y), (x+2, y-1);
                                          0 -> Upleft -> Upright -> Upleft
            c) "right", LRhombus, BLUE color: (x, y), (x, y+1), (x+1, y), (x+1, y+1);
                                              0 -> Upright -> Upleft -> Upright
        """
        def are_close(x, y, thres = 1E-6):
            return abs(x - y) < thres
        all_tile_types = self.get_all_tile_types()
        thres = 1E-6
        pt0, pt1, pt2, pt3 = self.get_tile_coords(tile)
        x1, y1 = pt1
        x2, y2 = pt2
        if are_close(x2 - x1, 1) and are_close(y2 - y1, -2):
            return all_tile_types[0]    # "out", green tile
        elif are_close(x2 - x1, 0) and are_close(y2 - y1, 1):
            return all_tile_types[1]    # "up", red tile
        elif are_close(x2 - x1, 1) and are_close(y2 - y1, -1):
            return all_tile_types[2]    # "right", blue tile
        else:
            raise Exception("Error occurred when getting a tile type.")

    def sanitize_coords_combs_input(self, coords_combs):
        try:
            coords_combs[0] + 0
        except:
            return coords_combs
        else:
            return (coords_combs, )

    def get_tiles_by_coords(self, coords_combs):
        func_input = self.sanitize_coords_combs_input(coords_combs)
        avail_tiles = VGroup()
        for tile_set_coords, tile_set in zip(self.tiles_coords, self.tiles):
            for tile_coords, tile in zip(tile_set_coords, tile_set):
                if all([(x, y) in tile_coords for (x, y) in func_input]):
                    avail_tiles.add(tile)
        return avail_tiles

    def get_tile_colors(self):
        return self.tile_colors

    def get_tile_color(self, tile, use_current_color = True):
        if use_current_color:
            return tile.get_fill_color()
        else:
            all_tile_types = self.get_all_tile_types()
            tile_type = self.get_tile_type(tile)
            k = all_tile_types.index()
            return self.get_tile_color()[k]

    def get_tile_coords(self, tile):
        coords = [self.ct_grid.point_to_coords(point) for point in tile.points[:-1:3]]
        sort_coords(coords)
        return coords

    def get_tile_center_of_mass(self, tile):
        tile_coords = self.get_tile_coords(tile)
        tile_anchors = [self.ct_grid.coords_to_point(x, y) for (x, y) in tile_coords]
        center_of_mass = np.average(tile_anchors)
        return center_of_mass

    def get_tile_centers_of_triangles(self, tile):
        tile_coords = self.get_tile_coords(tile)
        tile_type = self.get_tile_type(tile)
        all_tile_types = self.get_all_tile_types()
        tile_anchors = [self.ct_grid.coords_to_point(x, y) for (x, y) in tile_coords]
        if tile_type == all_tile_types[0]:
            pt0_indices = [0, 2, 3]
            pt1_indices = [0, 1, 3]
        elif tile_type == all_tile_types[1]:
            pt0_indices = [1, 2, 3]
            pt1_indices = [0, 1, 2]
        else:
            pt0_indices = [0, 1, 2]
            pt1_indices = [1, 2, 3]
        centers_of_triangles = tuple([
            np.mean(np.array([tile_anchors[i] for i in pt0_indices]), axis = 0),
            np.mean(np.array([tile_anchors[j] for j in pt1_indices]), axis = 0),
        ])
        return centers_of_triangles

    def generate_tile_dumbbell(self, tile, color = None):
        pt0, pt1 = centers_of_triangles = self.get_tile_centers_of_triangles(tile)
        kwargs = copy.deepcopy(self.dumbbell_config)
        kwargs.update({"color" : self.get_tile_color(tile) if color is None else color})
        dumbbell = Dumbbell(pt0, pt1, **kwargs)
        return dumbbell

    def generate_all_dumbbells(self):
        self.dumbbells = VGroup()
        for tile_set in self.tiles:
            dumbbell_set = VGroup()
            for tile in tile_set:
                dumbbell_set.add(self.generate_tile_dumbbell(tile))
            self.dumbbells.add(dumbbell_set)

    def get_dumbbell_by_tile(self, tile):
        for tile_set, dumbbell_set in zip(self.get_all_tiles(), self.get_all_dumbbells()):
            if tile in tile_set.submobjects:
                k = tile_set.submobjects.index(tile)
                return dumbbell_set.submobjects[k]
        return VMobject()

    def get_tile_by_dumbbell(self, dumbbell):
        for dumbbell_set, tile_set in zip(self.get_all_dumbbells(), self.get_all_tiles()):
            if dumbbell in dumbbell_set.submobjects:
                k = dumbbell_set.submobjects.index(dumbbell)
                return tile_set.submobjects[k]
        return VMobject()

    def get_all_dumbbells(self):
        return self.dumbbells

    def get_dimension(self):
        return self.ct_3d.get_dimension()


class Dumbbell(VMobject):
    CONFIG = {
        "fill_opacity" : 1,
        "point_size" : 0.07,
        "stem_width" : 0.03,
        "outline_color" : BLACK,
        "outline_width" : 0,
    }
    def __init__(self, pt0, pt1, **kwargs):
        self.start_point = pt0
        self.end_point = pt1
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        bell0, bell1 = [
            Dot(
                point, radius = self.point_size, color = self.color, fill_opacity = 1,
                stroke_color = self.outline_color, stroke_width = self.outline_width
            )
            for point in (self.start_point, self.end_point)
        ]
        vector_n = normalize(rotate_vector(self.end_point - self.start_point, PI/2.))
        stem_vertices = [
            self.start_point + vector_n * self.stem_width / 2.,
            self.start_point - vector_n * self.stem_width / 2.,
            self.end_point - vector_n * self.stem_width / 2.,
            self.end_point + vector_n * self.stem_width / 2.
        ]
        stem = Polygon(
            *stem_vertices, color = self.color, fill_opacity = 1,
            stroke_color = self.outline_color, stroke_width = self.outline_width
        )
        self.add(stem, bell0, bell1)
        self.bell0 = bell0
        self.bell1 = bell1
        self.stem = stem

    def get_center(self):
        return np.average([self.start_point, self.end_point])

    def get_bells(self):
        return VGroup(self.bell0, self.bell1)

    def get_stem(self):
        return self.stem


class Sqrt2PWW(VMobject):
    CONFIG = {
        "colors" : [TEAL, PINK, PINK, YELLOW],
        "fill_opacities" : [0.5, 0.75, 0.75, 0.9],
        "height" : 2.5,
    }
    def generate_points(self):
        side_lengths =  [2, np.sqrt(2), np.sqrt(2), 2*np.sqrt(2)-2]
        outer_square, ul_square, dr_square, inner_square = [
            Square(side_length = sl, fill_opacity = fo, fill_color = color)
            for sl, fo, color in zip(side_lengths, self.fill_opacities, self.colors)
        ]
        ul_square.shift(outer_square.get_anchors()[0] - ul_square.get_anchors()[0])
        dr_square.shift(outer_square.get_anchors()[2] - dr_square.get_anchors()[2])
        inner_square.move_to(outer_square)
        self.add(outer_square, ul_square, dr_square, inner_square)
        self.set_height(self.height)


class Tangram(VMobject):
    CONFIG = {
        "piece_colors" : [RED_D, CB_LIGHT, BLUE_D, YELLOW_D, GREEN_E, MAROON_B, ORANGE],
        "border_width" : 5,
        "height" : 4,
    }
    def generate_points(self):
        self.tweak_piece_colors()
        pieces = [self.generate_piece(i) for i in range(7)]
        border = self.generate_border()
        self.add(*pieces, border)
        self.center()
        self.set_height(self.height)

        self.pieces = pieces
        self.border = border

    def tweak_piece_colors(self):
        if self.piece_colors != 7:
            new_colors = color_gradient(self.piece_colors, 7)
            self.piece_colors = new_colors

    def get_main_vertices(self):
        return [
            np.array([0., 0., 0.]), np.array([1., 0., 0.]), np.array([1., 0.5, 0.]),
            np.array([1., 1., 0.]), np.array([0.5, 1., 0.]), np.array([0., 1., 0.]),
            np.array([0.25, 0.75, 0.]), np.array([0.5, 0.5, 0.]), np.array([0.75, 0.25, 0.]),
            np.array([0.75, 0.75, 0.])
        ]

    def get_all_piece_indices(self):
        return [
            (0, 1, 7, 0), (0, 5, 7, 0), (2, 4, 3, 2), (4, 6, 7, 9, 4),
            (1, 2, 9, 8, 1), (4, 5, 6, 4), (8, 9, 7, 8)
        ]

    def generate_piece(self, piece_index, **kwargs):
        anchors = self.get_piece_anchors(piece_index)
        color = self.get_piece_color(piece_index)
        piece = VMobject(stroke_width = 0, fill_color = color, fill_opacity = 1)
        piece.set_anchor_points(anchors, mode = "corners")
        return piece

    def get_pieces(self):
        return self.pieces

    def get_piece(self, piece_index):
        return self.get_pieces()[piece_index]

    def get_piece_anchors(self, piece_index):
        main_vertices = self.get_main_vertices()
        all_piece_indices = self.get_all_piece_indices()
        return np.array([main_vertices[i] for i in all_piece_indices[piece_index]])

    def get_piece_color(self, piece_index):
        return self.piece_colors[piece_index]

    def generate_border(self):
        border = VMobject(stroke_width = self.border_width)
        border.set_anchor_points([ORIGIN, RIGHT, UR, UP, ORIGIN], mode = "corners")
        return border

    def get_border(self):
        return self.border


#####
## Main Scenes

class IntroToMathematicalProofs(Scene):
    def construct(self):
        self.show_fermat()
        self.show_confusing_face()

    def setting_things_up(self):
        smiley = SVGMobject("face_smiley.svg")
        sad = SVGMobject("face_sad.svg")
        self.add(smiley)
        self.wait()
        self.play(Transform(smiley, sad), run_time = 2)
        self.wait()

    def show_fermat(self):
        claim = TexMobject("p = x^2 + y^2 \\text{ iff } 4|(p-1)")
        claim.scale(1.2)
        claim.to_edge(UP)

        fermat = ImageMobject("Pierre_de_Fermat_blurred.jpg")
        fermat.scale(2.5)
        fermat.shift(DOWN)
        fermat.to_edge(LEFT)

        fermat_text = TextMobject("我想到了一个绝妙的证明，\\\\ 但是这个对话气泡太小...")
        fermat_text.scale(0.8)
        fermat_bubble = SpeechBubble()
        fermat_bubble.add_content(fermat_text)
        fermat_bubble.resize_to_content()
        fermat_bubble.scale(0.8)
        fermat_bubble.move_tip_to([-3.5, -0.5, 0])

        self.play(FadeInFromDown(fermat))
        self.wait()
        self.play(Write(claim))
        self.wait()
        self.play(BubbleCreation(fermat_bubble))
        self.wait()

    def show_confusing_face(self):
        happy_face = SVGMobject("face_smiley.svg")
        sad_face = SVGMobject("face_sad.svg")
        VGroup(happy_face, sad_face).to_corner(DR)
        face_bubble = ThoughtBubble()
        texts = ["UFD?", "理想?", "$\\mathbb{Z}[i]/(p)$?", "???"]
        face_bubble.pin_to(happy_face)
        face_bubble.write(texts[1])
        face_bubble.resize_to_content()
        face_bubble.clear()
        self.play(FadeIn(happy_face))
        self.play(BubbleGrowFromTip(face_bubble))
        face_transition = NormalAnimationAsContinualAnimation(
            Transform(happy_face, sad_face, run_time = 5)
        )
        self.add(face_transition)
        for n, text in enumerate(texts):
            text_mob = TextMobject(text)
            text_mob.move_to(face_bubble.get_bubble_center())
            self.play(FadeIn(text_mob), run_time = 0.25)
            self.wait(0.75)
            if n != len(texts)-1:
                self.play(FadeOut(text_mob), run_time = 0.25)
        self.wait()


class ExamplesOfShortAndVisualProofs(Scene):
    def construct(self):
        self.setting_things_up()
        self.show_proofs()
        self.highlight_cct()

    def setting_things_up(self):
        # Infinite prime numbers
        inf_primes = TexMobject("""
            0 < \\prod_{p} {\\sin{\\frac{\\pi}{p}}}
            = \\prod_{p} {\\sin{\\frac{\\left(2\\prod_{p'}{p'}+1\\right)\\pi}{p}}}
            = 0
        """)
        inf_primes.set_width(FRAME_WIDTH / 1.8)
        VGroup(inf_primes[:2], inf_primes[-1]).set_color(RED)
        # Zagier's one-sentence proof
        osp = TexMobject("""
            (x, \\, y, \\, z) \\mapsto
            \\begin{cases}
            (x + 2z,\\, z, \\, y - x - z), & x < y - z \\\\
            (2y - x,\\, y, \\, x - y + z), & y - z < x < 2y \\\\
            (x - 2y,\\, x - y + z, \\, y), & x > 2y
            \\end{cases}
            """,
            background_stroke_width = 0
        )
        osp.set_width(FRAME_WIDTH / 1.8)
        # The irrationality of sqrt(2)
        sqrt2 = Sqrt2PWW()
        # Calisson's Tiling Theorem
        ctt_tiling = CalissonTiling2D(
            CalissonTiling3D(
                dimension = 4, pattern = generate_pattern(4),
                enable_fill = True, tile_config = {"stroke_width" : 2}
            ),
        )
        ctt_counter = CalissonTilesCounter(matching_tiling = ctt_tiling)
        ctt = VGroup(ctt_tiling, ctt_counter)
        ctt.set_height(FRAME_HEIGHT / 2.5)
        # Move to designated location
        proof_list = [inf_primes, osp, sqrt2, ctt]
        location_list = [(-2.5, 3, 0), (2.5, 1.5, 0), (-4, -1, 0), (3, -1.5, 0)]
        for proof, location in zip(proof_list, location_list):
            proof.move_to(location)
        self.inf_primes = inf_primes
        self.osp = osp
        self.sqrt2 = sqrt2
        self.ctt = ctt

    def show_proofs(self):
        self.play(LaggedStart(FadeInFromLarge, VGroup(self.inf_primes, self.osp)))
        self.wait()
        self.play(LaggedStart(FadeInFromLarge, VGroup(self.sqrt2, self.ctt)))
        self.wait()

    def highlight_cct(self):
        sur_rect = SurroundingRectangle(self.ctt, color = YELLOW, buff = 0.25)
        self.play(ShowCreation(sur_rect))
        self.wait()
        video_rect = FullScreenRectangle(stroke_width = 0)
        mobs = [self.inf_primes, self.osp, self.sqrt2, self.ctt]
        locations = [(-10, 12, 0), (10, 6, 0), (-16, -4, 0), (0, 0, 0)]
        for mob, location in zip(mobs, locations):
            mob.generate_target()
            mob.target.move_to(location)
        self.ctt.target.scale(1.5)
        self.play(
            Transform(sur_rect, video_rect),
            FadeOut(VGroup(self.inf_primes, self.osp, self.sqrt2)),
            MoveToTarget(self.inf_primes),
            MoveToTarget(self.osp),
            MoveToTarget(self.sqrt2),
            MoveToTarget(self.ctt),
        )
        self.wait()


class CuttingEdgeOrEdgeCutting(Scene):
    def construct(self):
        self.add_an_example_tiling()
        self.show_dijkstra_portrait()
        self.show_ce_and_ec_texts()

    def add_an_example_tiling(self):
        # An example tiling
        dimension = 4
        tile_config = {"stroke_width" : 1.5}
        border_config = {"stroke_width" : 3}
        tiling_3d = CalissonTiling3D(
            dimension = dimension,
            pattern = generate_pattern(dimension),
            enable_fill = True,
            tile_config = tile_config,
        )
        tiling_2d = CalissonTiling2D(tiling_3d)
        # Counting tiles
        num_of_tiles = dimension ** 2
        tiles_counter = CalissonTilesCounter(matching_tiling = tiling_2d)
        group = VGroup(tiling_2d, tiles_counter)
        group.set_height(3.5)
        group.shift(0.5 * RIGHT)
        group.to_edge(UP)
        self.add(group)
        self.group = group

    def show_dijkstra_portrait(self):
        dijk_pic = ImageMobject("Edsger_Dijkstra.jpg")
        dijk_pic.scale(2.75)
        dijk_pic.to_corner(UL)
        dijk_text = TextMobject("Edsger W. Dijkstra")
        dijk_text.next_to(dijk_pic, DOWN)
        self.play(
            FadeInFromDown(dijk_pic),
            Write(dijk_text),
        )
        self.wait()
        self.dijk_pic = dijk_pic
        self.dijk_text = dijk_text

    def show_ce_and_ec_texts(self):
        ce = TextMobject("A",  "cutting-edge", "method")
        ec = TextMobject("An", "edge-cutting", "method")
        text_group = VGroup(ce, ec)
        text_group.scale(1.2)
        text_group.arrange_submobjects(DOWN, buff = 0.5)
        text_group.next_to(self.group, DOWN, buff = 0.8)
        ce_copy = ce.copy()
        cross = Cross(ce)
        self.play(Write(ce_copy))
        self.wait()
        self.play(ShowCreation(cross))
        self.wait()
        self.play(ApplyMethod(ce.move_to, ec), Animation(cross))
        self.wait()
        self.play(
            Transform(ce[0], ec[0]),
            Transform(ce[1][:7], ec[1][-7:], path_arc = PI/2),
            Transform(ce[1][7], ec[1][-8]),
            Transform(ce[1][-4:], ec[1][:4], path_arc = PI/2),
            Transform(ce[2], ec[2]),
            run_time = 1,
        )
        self.wait()


class CountingsDoNotChange(Scene):
    def construct(self):
        ct_grid = CalissonTilingGrid(
            side_length = 15, unit_size = 0.4,
            grid_lines_config = {"stroke_width" : 0.25, "color" : GREY}
        )
        ct_grid.add_grid_lines()
        tilings = [
            CalissonTiling2D(
                CalissonTiling3D(
                    dimension = 9,
                    pattern = get_min_array(generate_pattern(9), BOUNDING_VALUES),
                    tile_config = {"stroke_width" : 1}, enable_fill = True,
                ),
                ct_grid = ct_grid
            )
            for k in range(6)
        ]
        for tiling in tilings:
            tiling.remove_border()
            for direction, (x, y) in OBSOLETE_TILES_INFO:
                tiling.remove_tile(direction, [x, y])
            tiling.shuffle_tiles()
        new_border = VMobject(stroke_width = 5, stroke_color = YELLOW)
        new_border_anchor_points = [
            ct_grid.coords_to_point(x, y)
            for x, y in BORDER_ANCHORS_COORDINATES
        ]
        new_border.set_anchor_points(new_border_anchor_points, mode = "corners")
        init_tiling = tilings[0]
        new_tilings = VGroup(*tilings[1:])
        ct_counter = CalissonTilesCounter([57, 70, 50], tile_stroke_width = 1, height = 3)
        ct_counter.shift(3*RIGHT)
        self.play(ShowCreation(new_border))
        self.wait()
        self.play(TilesGrow(init_tiling))
        self.wait()
        new_tilings.shift(2*LEFT)
        self.play(
            ApplyMethod(
                VGroup(init_tiling, new_border).shift, 2*LEFT
            ),
            FadeInFromDown(ct_counter),
        )
        sur_rect = SurroundingRectangle(ct_counter.get_nums(), stroke_width = 3,)
        conclusion_text = TextMobject("不会改变")
        conclusion_text.next_to(sur_rect, UP)
        VGroup(sur_rect, conclusion_text).set_color(YELLOW)
        additional_anims = NormalAnimationAsContinualAnimation(
            AnimationGroup(ShowCreation(sur_rect), Write(conclusion_text)), run_time = 2
        )
        for k, new_tiling in enumerate(new_tilings):
            if k == 1:
                self.add(additional_anims)
            self.play(Transform(init_tiling, new_tiling, path_arc = PI/2., run_time = 3))
            self.wait()


class TheConclusionsAreDifferent(Scene):
    def construct(self):
        jp_title = TextMobject("刚才证明的")
        jp_text = TextMobject("镶嵌正六边形时，", "每种菱形的数目是", "不变的")
        np_title = TextMobject("需要证明的")
        np_text = TextMobject("镶嵌正六边形时，", "每种菱形的数目是", "相同的")
        for mobs in [jp_text, np_text]:
            mobs.arrange_submobjects(DOWN, aligned_edge = LEFT)
            mobs[-1].set_color(YELLOW)
        VGroup(jp_title, np_title).set_color(GREEN).scale(1.2)
        jp_title.shift(3*LEFT)
        np_title.shift(3*RIGHT)
        jp_text.next_to(jp_title, DOWN, buff = 0.8)
        np_text.next_to(np_title, DOWN, buff = 0.8)
        arrow = TexMobject("\\Rightarrow")
        arrow.scale(1.5)
        arrow.move_to((jp_text.get_center()+np_text.get_center())/2)
        jp_group = VGroup(jp_title, jp_text)
        np_group = VGroup(np_title, np_text)
        VGroup(jp_group, np_group, arrow).center()
        jp_rect = SurroundingRectangle(jp_text[-1], color = YELLOW)
        np_rect = SurroundingRectangle(np_text[-1], color = YELLOW)

        self.play(FadeInFromDown(np_group))
        self.play(ShowCreation(np_rect))
        self.wait()
        self.play(FadeInFromDown(jp_group))
        self.play(ShowCreation(jp_rect))
        self.wait()
        self.play(Write(arrow))
        self.wait()


class CTProjection(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(theta = 0, distance = 1E7)
        dim = 5
        pattern = generate_pattern(dim)
        ct_3d = CalissonTiling3D(
            dimension = dim, pattern = pattern,
            tile_config = {"stroke_width" : 3, "stroke_color" : WHITE},
        )
        ct_2d = get_ct_3d_xoy_projection(ct_3d)
        xoy_plane = SurroundingRectangle(
            ct_2d, color = GREY, fill_opacity = 1, buff = 0.5
        )
        ct_3d.set_stroke(width = 3)
        ct_3d.border.set_stroke(width = 5)
        ct_2d.set_fill(opacity = 1)
        set_projection_orientation(ct_3d)
        ct_3d.shift(2*OUT)
        ct_3d.set_color(YELLOW)
        VGroup(ct_2d, xoy_plane).shift(2*IN)
        proj_lines = VGroup(*[
            DashedLine(pt_3d, pt_2d, color = YELLOW, stroke_width = 1)
            for pt_3d, pt_2d in zip(ct_3d.border.points[:-1:3], ct_2d.border.points[:-1:3])
        ])
        self.add(xoy_plane, ct_2d, ct_3d, proj_lines)
        self.wait()
        self.move_camera(phi = PI/3, theta = PI/24, run_time = 5)
        self.begin_ambient_camera_rotation()
        self.wait(60)


#####
## Thumbnail

class Thumbnail(Scene):
    def construct(self):
        dim = 20
        pattern = generate_pattern(dim)
        ct_grid = CalissonTilingGrid()
        ct_3d = CalissonTiling3D(
            dimension = dim, pattern = pattern, enable_fill = True
        )
        ct_2d = CalissonTiling2D(
            ct_3d, ct_grid, dumbbell_config = {"point_size" : 0.1, "stem_width" : 0.05}
        )
        ct_2d.generate_all_dumbbells()
        tiles = ct_2d.get_all_tiles()
        for tile_set in tiles:
            for tile in tile_set:
                opacity = self.interpolate_by_horiz_position(tile, 0, 1)
                stroke_width = self.interpolate_by_horiz_position(tile, 0, 5)
                tile.set_fill(opacity = opacity)
                tile.set_stroke(width = stroke_width)
        dumbbells = ct_2d.get_all_dumbbells()
        self.add(ct_2d, dumbbells)
        self.wait()

    def interpolate_by_horiz_position(self, mob, min_val, max_val, inflection = 15.0):
        # x = mob.get_center()[0]
        x = mob.get_center_of_mass()[0]
        L, R = LEFT_SIDE[0], RIGHT_SIDE[0]
        alpha = 1 - smooth(np.clip((x-L)/(R-L), 0, 1), inflection)
        interpolate_val = min_val + (max_val - min_val) * alpha
        return interpolate_val


