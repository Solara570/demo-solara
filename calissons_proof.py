#coding=utf-8

from big_ol_pile_of_manim_imports import *
from custom.custom_animations import *
from custom.custom_helpers import *
from custom.custom_mobjects import *
from calissons import *
from calissons_constants import *

#####
## Constants

CB_DARK  = "#825201"
CB_LIGHT = "#B69B4C"
MAGENTA = "#CC00FF"
CYAN = "#00FFE0"


#####
## Methods
def are_close(x, y, thres = 1E-6):
    return abs(x - y) <= thres

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
    # Project onto the xoy plane
    ct_2d.apply_function(lambda p: [p[0], p[1], 0])
    ct_2d.set_shade_in_3d(False)
    return ct_2d

def get_grid_triangle_center(grid_triangle):
    return np.apply_along_axis(np.mean, 0, grid_triangle.get_anchors()[:-1])

def get_max_integer_under(x):
    x_int = int(x)
    return x_int-1 if x_int>x else x_int


#####
## Animations

class Scatter(MoveToTarget):
    def __init__(self, mobject, center = None, scale_factor = 1.5, **kwargs):
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

class TilesCounter(VMobject):
    CONFIG = {
        "tile_types"  : [RRhombus, HRhombus, LRhombus],
        "tile_colors" : [TILE_RED, TILE_GREEN, TILE_BLUE],
        "tile_stroke_width" : 2,
        "tile_stroke_color" : WHITE,
        "tile_fill_opacity" : 1,
        "height" : 2,
        "matching_height" : True,
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
        for k, pair in enumerate(pairs):
            tile, text = pair
            tile.set_stroke(width = self.tile_stroke_width, color = self.tile_stroke_color)
            tile.set_fill(opacity = self.tile_fill_opacity)
            text.set_color(self.tile_colors[k])
            text.scale(1.8)
            pair.arrange_submobjects(RIGHT)
        pairs.arrange_submobjects(
            DOWN, index_of_submobject_to_align = 1, aligned_edge = LEFT, buff = self.counter_buff
        )
        if self.matching_tiling is not None and self.matching_height:
            self.height = self.matching_tiling.get_height() * 0.8
        pairs.set_height(self.height)
        if self.matching_tiling is not None:
            pairs.next_to(
                self.matching_tiling, 
                direction = self.matching_direction,
                buff = self.matching_buff
            )
        self.add(pairs)
        self.pairs = pairs

    def get_pairs(self):
        return self.pairs

    def get_tiles(self):
        return VGroup(*[self.pairs[k][0] for k in range(3)])

    def get_mult_signs(self):
        return VGroup(*[self.pairs[k][1][0] for k in range(3)])

    def get_nums(self):
        return VGroup(*[self.pairs[k][1][1:] for k in range(3)])

    def get_pair(self, n):
        return self.get_pairs()[n]

    def get_tile(self, n):
        return self.get_tiles()[n]

    def get_mult_sign(self, n):
        return self.get_mult_signs()[n]

    def get_num(self, n):
        return self.get_nums()[n]

    def get_pair_elements(self, n):
        return VGroup(self.get_tile(n), self.get_mult_sign(n), self.get_num(n))


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
        self.generate_grid_lines()
        self.generate_grid_points()
        self.generate_grid_triangles()
        self.generate_grid_boundary()

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
        self.add(self.grid_center, self.basis_vec_x, self.basis_vec_y)

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
        self.grid_lines = grid_lines

    def generate_grid_points(self):
        grid_points = VGroup()
        valid_grid_coords = list(filter(
            self.is_valid_grid_point,
            [(x, y) for x in range(self.x_min, self.x_max+1) for y in range(self.y_min, self.y_max+1)]
        ))
        for x, y in valid_grid_coords:
            dot = Dot(self.coords_to_point(x, y), **self.grid_points_config)
            grid_points.add(dot)
        self.grid_points = grid_points

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
        self.grid_triangles = grid_triangles

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
        self.grid_boundary = grid_boundary

    def add_grid_boundary(self):
        self.generate_grid_boundary()
        self.add(self.grid_boundary)
        return self

    def add_grid_lines(self):
        self.generate_grid_lines()
        self.add(self.grid_lines)
        return self

    def add_grid_points(self):
        self.generate_grid_points()
        self.add(self.grid_points)
        return self

    def add_grid_triangles(self):
        self.generate_grid_triangles()
        self.add(self.grid_triangles)
        return self

    def coords_to_point(self, x, y):
        basis_x, basis_y = self.get_basis()
        return self.get_grid_center() + x * basis_x + y * basis_y

    def point_to_coords(self, point):
        basis_x, basis_y = self.get_basis()
        grid_center = self.get_grid_center()
        dot_ii = np.dot(basis_x, basis_x)
        dot_jj = np.dot(basis_y, basis_y)
        dot_ij = np.dot(basis_x, basis_y)
        dot_pi = np.dot(point - grid_center, basis_x)
        dot_pj = np.dot(point - grid_center, basis_y)
        A = np.array([[dot_ii, dot_ij], [dot_ij, dot_jj]])
        b = np.array([dot_pi, dot_pj])
        x, y = np.around(np.linalg.solve(A, b), decimals = 6)
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
        return color1 if gp1_coords[1] < gp0_coords[1] else color2

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
            self.generate_grid_lines()
        return self.grid_lines

    def get_grid_points(self):
        if not hasattr(self, "grid_points"):
            self.generate_grid_points()
        return self.grid_points

    def get_grid_triangle_containing_point(self, point):
        x, y = self.point_to_coords(point)
        x_int, y_int = get_max_integer_under(x), get_max_integer_under(y)
        if (x+y) < (x_int+y_int+1):
            coords_combs = [(x_int, y_int), (x_int+1, y_int), (x_int, y_int+1)]
        else:
            coords_combs = [(x_int+1, y_int), (x_int, y_int+1), (x_int+1, y_int+1)]
        return self.get_grid_triangle(coords_combs)

    def get_grid_triangle_on_line(self, line, thres = 1e-4):
        sp, ep = line.get_start_and_end()
        all_triangles = self.get_grid_triangles()
        valid_triangles_list = []
        for triangle in all_triangles:
            counter = 0
            for vertex in triangle.get_anchors()[:-1]:
                vec1, vec2 = sp-vertex, ep-vertex
                if (np.dot(vec1, vec2) < thres) and (get_norm(np.cross(vec1, vec2)) < thres):
                    counter += 1
            if counter == 2:
                valid_triangles_list.append(triangle)
        return VGroup(*valid_triangles_list)

    def get_grid_triangles_not_on_line(self, line, thres = 1e-4):
        all_triangles = self.get_grid_triangles()
        valid_triangles = self.get_grid_triangle_on_line(line, thres)
        invalid_triangles_list = list(set(all_triangles.submobjects) - set(valid_triangles.submobjects))
        return VGroup(*invalid_triangles_list)

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
            self.generate_grid_triangles()
        return self.grid_triangles

    def get_grid_boundary(self):
        if not hasattr(self, "grid_boundary"):
            self.generate_grid_boundary()
        return self.grid_boundary

    # The following methods are only used to create a better visual effect
    def get_randomized_copy(self, vmob):
        vmob_copy = vmob.deepcopy()
        mobs = vmob_copy.submobjects
        random.shuffle(mobs)
        vmob_copy.submobjects = mobs
        return vmob_copy

    def get_randomized_point_copy(self):
        return self.get_randomized_copy(self.get_grid_points())

    def get_randomized_line_copy(self):
        return self.get_randomized_copy(self.get_grid_lines())

    def get_randomized_triangle_copy(self):
        return self.get_randomized_copy(self.get_grid_triangles())


class CalissonTiling2D(CalissonTiling3D):
    CONFIG = {
        "enable_dumbbells" : True,
        "dumbbell_config" : {
            "point_size" : 0.07,
            "stem_width" : 0.03,
            "outline_color" : BLACK,
            "outline_width" : 0,
        },
    }
    def __init__(self, ct_3d, ct_grid = None, **kwargs):
        VMobject.__init__(self, **kwargs)
        side_length = ct_3d.get_dimension()
        if ct_grid is None:
            self.ct_grid = CalissonTilingGrid(side_length = side_length)
            self.move_to(ORIGIN)
        else:
            self.ct_grid = CalissonTilingGrid(side_length = side_length, unit_size = ct_grid.get_unit_size())
            self.move_to(ct_grid.get_grid_center())
        self.add(self.ct_grid)
        self.ct_3d = ct_3d
        self.ct_2d = get_ct_3d_xoy_projection(self.ct_3d, self.ct_grid)
        self.border, self.tiles = self.ct_2d
        self.add(self.border, self.tiles)
        self.tiles_coords = list()
        self.dumbbells_coords = list()
        for tile_set in self.tiles:
            tile_set_coords_list = tuple(self.get_tile_coords(tile) for tile in tile_set)
            self.tiles_coords.append(tile_set_coords_list)
            dumbbell_set_coords = tuple(self.get_tile_bells_center_coords(tile) for tile in tile_set)
            self.dumbbells_coords.append(dumbbell_set_coords)
        self.tiles_coords = tuple(self.tiles_coords)
        self.dumbbells_coords = tuple(self.dumbbells_coords)
        if self.enable_dumbbells:
            self.add(self.get_all_dumbbells())

    def get_all_tiles(self):
        return self.tiles

    def get_all_tile_types(self):
        return ["out", "up", "right"]

    def get_all_tile_colors(self):
        return self.tile_colors

    def get_all_tile_coords(self):
        all_tile_coords = list()
        for tile_set_coords in self.tiles_coords:
            for tile_coords in tile_set_coords:
                all_tile_coords.append(tile_coords)
        return tuple(all_tile_coords)

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
        all_tile_types = self.get_all_tile_types()
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
        if len(avail_tiles.submobjects) > 0:
            return avail_tiles
        else:
            raise Exception("Tiles not found.")
    
    def get_dumbbell_by_tile_coords(self, tile_coords):
        tile = self.get_tiles_by_coords(tile_coords)[0]
        dumbbell = self.get_dumbbell_by_tile(tile)
        return dumbbell

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
        return tuple(coords)

    def get_tile_center_of_mass(self, tile):
        tile_coords = self.get_tile_coords(tile)
        tile_anchors = [self.ct_grid.coords_to_point(x, y) for (x, y) in tile_coords]
        center_of_mass = np.average(tile_anchors)
        return center_of_mass

    def get_tile_bells_center_points(self, tile):
        tile_coords = self.get_tile_coords(tile)
        tile_type = self.get_tile_type(tile)
        all_tile_types = self.get_all_tile_types()
        tile_anchors = [self.ct_grid.coords_to_point(x, y) for (x, y) in tile_coords]
        if tile_type == all_tile_types[0]:
            pt0_indices = [0, 2, 3]
            pt1_indices = [0, 1, 3]
        else:
            pt0_indices = [0, 1, 2]
            pt1_indices = [1, 2, 3]
        bells_center_points = tuple([
            np.mean(np.array([tile_anchors[i] for i in pt0_indices]), axis = 0),
            np.mean(np.array([tile_anchors[j] for j in pt1_indices]), axis = 0),
        ])
        return bells_center_points

    def get_tile_bells_center_coords(self, tile):
        bells_center_points = self.get_tile_bells_center_points(tile)
        return tuple(map(self.ct_grid.point_to_coords, bells_center_points))

    def generate_tile_dumbbell(self, tile, color = None):
        pt0, pt1 = bells_center_points = self.get_tile_bells_center_points(tile)
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
        return self.dumbbells

    def get_dumbbell_by_tile(self, tile):
        for tile_set, dumbbell_set in zip(self.get_all_tiles(), self.get_all_dumbbells()):
            if tile in tile_set.submobjects:
                k = tile_set.submobjects.index(tile)
                return dumbbell_set.submobjects[k]
        raise Exception("Dumbbell not found")

    def get_tile_by_dumbbell(self, dumbbell):
        for dumbbell_set, tile_set in zip(self.get_all_dumbbells(), self.get_all_tiles()):
            if dumbbell in dumbbell_set.submobjects:
                k = dumbbell_set.submobjects.index(dumbbell)
                return tile_set.submobjects[k]
        raise Exception("Tile not found")

    def get_all_dumbbells(self):
        if not hasattr(self, "dumbbells"):
            self.dumbbells = self.generate_all_dumbbells()
        return self.dumbbells

    def get_dimension(self):
        return self.ct_3d.get_dimension()

    def get_random_tile(self):
        i = random.randint(0, 2)
        j = random.randint(0, self.get_dimension()**2-1)
        return self.tiles[i][j]


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
        bells = VGroup(*[
            Dot(
                point, radius = self.point_size, color = self.color, fill_opacity = 1,
                stroke_color = self.outline_color, stroke_width = self.outline_width
            )
            for point in (self.start_point, self.end_point)
        ])
        vector_l = normalize(self.end_point - self.start_point)
        vector_n = rotate_vector(vector_l, PI/2.)
        stem_vertices = [
            self.start_point + vector_l * self.point_size + vector_n * self.stem_width / 2.,
            self.start_point + vector_l * self.point_size - vector_n * self.stem_width / 2.,
            self.end_point - vector_l * self.point_size - vector_n * self.stem_width / 2.,
            self.end_point - vector_l * self.point_size + vector_n * self.stem_width / 2.
        ]
        stem = Polygon(
            *stem_vertices, color = self.color, fill_opacity = 1,
            stroke_color = self.outline_color, stroke_width = self.outline_width
        )
        self.add(stem, bells)
        self.bells = bells
        self.stem = stem

    def get_stem(self):
        return self.stem

    def get_bells(self, reverse = False):
        start_bell, end_bell = self.bells
        if reverse:
            return VGroup(end_bell, start_bell)
        else:
            return VGroup(start_bell, end_bell)

    def get_start_bell(self, reverse = False):
        return self.get_bells(reverse = reverse)[0]

    def get_end_bell(self, reverse = False):
        return self.get_bells(reverse = reverse)[1]

    def get_start_and_end_points(self, reverse = False):
        return [bell.get_center() for bell in self.get_bells(reverse = reverse)]

    def get_start_point(self, reverse = False):
        return self.get_start_bell(reverse = reverse).get_center()

    def get_end_point(self, reverse = False):
        return self.get_end_bell(reverse = reverse).get_center()

    def get_arrow(self, reverse = False):
        start_point, end_point = self.get_start_and_end_points(reverse = reverse)
        color = self.get_stem().get_stroke_color()
        arrow = Arrow(
            start_point, end_point, color = color,
            max_tip_length_to_length_ratio = 0.4, buff = 0,
        )
        if reverse:
            arrow.reverse_points()
        return VGroup(VectorizedPoint(start_point), arrow, VectorizedPoint(end_point))


class CalissonTilingDifference(object):
    def __init__(self, ct_2d_A, ct_2d_B):
        self.ct_2d_A = ct_2d_A
        self.ct_2d_B = ct_2d_B

    def get_tiles_coords(self):
        return tuple([ct.get_all_tile_coords() for ct in [self.ct_2d_A, self.ct_2d_B]])

    def get_same_tiles_coords(self):
        all_tiles_coords_A, all_tiles_coords_B = self.get_tiles_coords()
        return tuple(set(all_tiles_coords_A) & set(all_tiles_coords_B))

    def get_same_tiles(self):
        same_tiles_coords = self.get_same_tiles_coords()
        same_tiles_A = VGroup(*[
            self.ct_2d_A.get_tiles_by_coords(tile_coords) for tile_coords in same_tiles_coords
        ])
        same_tiles_B =VGroup(*[
            self.ct_2d_B.get_tiles_by_coords(tile_coords) for tile_coords in same_tiles_coords
        ])
        return VGroup(same_tiles_A, same_tiles_B)

    def get_same_dumbbells(self):
        same_tiles_coords = self.get_same_tiles_coords()
        same_dumbbells_A = VGroup(*[
            self.ct_2d_A.get_dumbbell_by_tile_coords(tile_coords) for tile_coords in same_tiles_coords
        ])
        same_dumbbells_B = VGroup(*[
            self.ct_2d_B.get_dumbbell_by_tile_coords(tile_coords) for tile_coords in same_tiles_coords
        ])
        return VGroup(same_dumbbells_A, same_dumbbells_B)

    def get_different_tiles_coords(self):
        all_tiles_coords_A, all_tiles_coords_B = self.get_tiles_coords()
        same_tiles_coords = self.get_same_tiles_coords()
        diff_tiles_coords_A = tuple(set(all_tiles_coords_A) - set(same_tiles_coords))
        diff_tiles_coords_B = tuple(set(all_tiles_coords_B) - set(same_tiles_coords))
        return diff_tiles_coords_A, diff_tiles_coords_B

    def get_different_tiles(self):
        diff_tiles_coords_A, diff_tiles_coords_B = self.get_different_tiles_coords()
        diff_tiles_A = VGroup(*[
            self.ct_2d_A.get_tiles_by_coords(tile_coords) for tile_coords in diff_tiles_coords_A
        ])
        diff_tiles_B = VGroup(*[
            self.ct_2d_B.get_tiles_by_coords(tile_coords) for tile_coords in diff_tiles_coords_B
        ])
        return VGroup(diff_tiles_A, diff_tiles_B)

    def get_different_dumbbells(self):
        diff_tiles_coords_A, diff_tiles_coords_B = self.get_different_tiles_coords()
        diff_dumbbells_A = VGroup(*[
            self.ct_2d_A.get_dumbbell_by_tile_coords(tile_coords) for tile_coords in diff_tiles_coords_A
        ])
        diff_dumbbells_B = VGroup(*[
            self.ct_2d_B.get_dumbbell_by_tile_coords(tile_coords) for tile_coords in diff_tiles_coords_B
        ])
        return VGroup(diff_dumbbells_A, diff_dumbbells_B)

    def get_loops_tiles_coords(self):
        def transfer_element(element, src_array, dst_array):
            dst_array.append(element)
            src_array.remove(element)

        def is_sharing_a_triangle(vals):
            tile_coords_A, tile_coords_B = vals
            return (len(set(tuple(tile_coords_A)) & set(tuple(tile_coords_B))) == 3)

        def find_next_tile_coords(prev_tile_coords, all_tile_coords):
            array = [(prev_tile_coords, tile_coords) for tile_coords in all_tile_coords]
            results = list(filter(is_sharing_a_triangle, array))
            return results[0][-1] if len(results) > 0 else None

        def move_first_element_to_last(array):
            array.append(array.pop(0))

        diff_tiles_coords_A, diff_tiles_coords_B = list(map(list, self.get_different_tiles_coords()))
        all_diff_tiles_coords = diff_tiles_coords_A + diff_tiles_coords_B
        loops_tiles_coords = list()
        new_loop_tiles_coords = list()
        # Forming new loops until all elements in 'all_diff_tiles_coords' are extracted.
        while (len(all_diff_tiles_coords) > 0):
            # Start a new loop
            if len(new_loop_tiles_coords) == 0:
                prev_tile_coords = all_diff_tiles_coords[0]
                transfer_element(prev_tile_coords, all_diff_tiles_coords, new_loop_tiles_coords)
            # Keep appending new 'tile_coords' to this new loop.
            next_tile_coords = find_next_tile_coords(prev_tile_coords, all_diff_tiles_coords)
            while (next_tile_coords is not None):
                transfer_element(next_tile_coords, all_diff_tiles_coords, new_loop_tiles_coords)
                prev_tile_coords = next_tile_coords
                next_tile_coords = find_next_tile_coords(prev_tile_coords, all_diff_tiles_coords)
            # Make sure all new loops start with a tile in 'self.ct_2d_A'.
            if new_loop_tiles_coords[0] in diff_tiles_coords_B:
                move_first_element_to_last(new_loop_tiles_coords)
            loops_tiles_coords.append(tuple(new_loop_tiles_coords))
            new_loop_tiles_coords = list()
        return tuple(loops_tiles_coords)

    def get_loops_tiles(self):
        loops_tiles = VGroup()
        loops_tiles_coords = self.get_loops_tiles_coords()
        for loop_tiles_coords in loops_tiles_coords:
            new_loop_tiles = VGroup()
            for k, tile_coords in enumerate(loop_tiles_coords):
                if (k % 2 == 0):    # This tile is in 'self.ct_2d_A'
                    new_loop_tiles.add(self.ct_2d_A.get_tiles_by_coords(tile_coords))
                else:               # This tile is in 'self.ct_2d_B'
                    new_loop_tiles.add(self.ct_2d_B.get_tiles_by_coords(tile_coords))
            loops_tiles.add(new_loop_tiles)
        return loops_tiles

    def get_loops_dumbbells(self):
        loops_dumbbells = VGroup()
        loops_tiles_coords = self.get_loops_tiles_coords()
        for loop_tiles_coords in loops_tiles_coords:
            new_loop_dumbbells = VGroup()
            for k, tile_coords in enumerate(loop_tiles_coords):
                if (k % 2 == 0):    # This dumbbell is in 'self.ct_2d_A'
                    new_loop_dumbbells.add(self.ct_2d_A.get_dumbbell_by_tile_coords(tile_coords))
                else:               # This dumbbell is in 'self.ct_2d_B'
                    new_loop_dumbbells.add(self.ct_2d_B.get_dumbbell_by_tile_coords(tile_coords))
            loops_dumbbells.add(new_loop_dumbbells)
        return loops_dumbbells

    def get_loop_reverse_flags(self, loop_dumbbells):
        def swap(array_with_2_elements):
            array_with_2_elements.append(array_with_2_elements.pop(0))

        def are_close_in_space(pt_1, pt_2, thres = 3E-6):
            return np.linalg.norm(pt_1 - pt_2) <= thres

        reverse_flags = list()
        dumbbells_start_and_end_points = tuple(map(Dumbbell.get_start_and_end_points, loop_dumbbells))
        for k, points in enumerate(dumbbells_start_and_end_points):
            if k == 0:      # First pair of points
                this_start, this_end = dumbbells_start_and_end_points[0]
                next_start, next_end = dumbbells_start_and_end_points[1]
                if not (are_close_in_space(this_end, next_start) or are_close_in_space(this_end, next_end)):
                    swap(dumbbells_start_and_end_points[0])
                    reverse_flags.append(False)
                else:
                    reverse_flags.append(True)
            else:           # Other pairs of points (which have a previous element)
                prev_start, prev_end = dumbbells_start_and_end_points[k-1]
                this_start, this_end = dumbbells_start_and_end_points[k]
                if not (are_close_in_space(this_start, prev_end)):
                    swap(dumbbells_start_and_end_points[k])
                    reverse_flags.append(False)
                else:
                    reverse_flags.append(True)
        return reverse_flags

    def get_loops_bells(self):
        loops_dumbbells = self.get_loops_dumbbells()
        loops_bells = VGroup()
        for loop_dumbbells in loops_dumbbells:
            loop_bells = VGroup()
            reverse_flags = self.get_loop_reverse_flags(loop_dumbbells)
            num_of_pairs = len(reverse_flags)//2
            for k in range(num_of_pairs):
                this_reverse_flag = reverse_flags[2*k]
                prev_reverse_flag = reverse_flags[2*k-1]
                next_reverse_flag = reverse_flags[2*k+1]
                this_dumbbell = loop_dumbbells[2*k]
                prev_dumbbell = loop_dumbbells[2*k-1]
                next_dumbell = loop_dumbbells[2*k+1]
                loop_bells.add(VGroup(*[
                    this_dumbbell.get_start_bell(reverse = this_reverse_flag),
                    prev_dumbbell.get_end_bell(reverse = prev_reverse_flag),
                ]))
                loop_bells.add(VGroup(*[
                    this_dumbbell.get_end_bell(reverse = this_reverse_flag),
                    next_dumbell.get_start_bell(reverse = next_reverse_flag)
                ]))
            loops_bells.add(loop_bells)
        return loops_bells

    def get_loop_arrows(self, loop_dumbbells):
        reverse_flags = self.get_loop_reverse_flags(loop_dumbbells)
        return VGroup(*[
            dumbbell.get_arrow(reverse = reverse_flag)
            for dumbbell, reverse_flag in zip(loop_dumbbells, reverse_flags)
        ])

    def get_loops_arrows(self):
        return VGroup(*[
            self.get_loop_arrows(loop_dumbbells)
            for loop_dumbbells in self.get_loops_dumbbells()
        ])

    def get_loop_arrows_crossing_vertical_line(self, loop_arrows, line):
        x_thres = (line.get_start()[0] + line.get_end()[0]) / 2.
        valid_arrows = VGroup()
        for k, loop_arrow in enumerate(loop_arrows):
            x1, x2 = loop_arrow.get_left()[0], loop_arrow.get_right()[0]
            if (x1-x_thres)*(x2-x_thres) <= 0:
                valid_arrows.add(loop_arrow)
        return valid_arrows


class CalissonRing(VMobject):
    CONFIG = {
        "colors" : [TILE_RED, TILE_GREEN, TILE_BLUE],
        "rhombus_config": {
            "side_length" : 0.4,
            "stroke_width" : 2,
            "mark_paths_closed" : True,
        },
    }
    def __init__(self, row = 3, col = 5, **kwargs):
        self.row = row
        self.col = col
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        color_a, color_b, color_c = self.colors
        row, col = self.row, self.col
        tile_set_b = VGroup(*[HRhombus(**self.rhombus_config) for k in range(row*col)])
        tile_set_b.set_fill(opacity = 1, color = color_b)
        anchors = tile_set_b[0].rhombus.get_anchors()
        vec_a = anchors[1] - anchors[0]
        vec_b = anchors[-2] - anchors[0]
        for i in range(row):
            for j in range(col):
                index = i*col+j
                shift_vec = i*vec_a + j*vec_b
                tile_set_b[index].shift(shift_vec)
        tile_set_c = tile_set_b.deepcopy().set_fill(color = color_c)
        tile_set_c.rotate(PI/3., about_point = tile_set_b[col-1].get_right())
        tile_set_a = tile_set_b.deepcopy().set_fill(color = color_a)
        tile_set_a.rotate(-PI/3., about_point = tile_set_b[-col].get_left())
        self.add(tile_set_a, tile_set_b, tile_set_c)


class Sqrt2PWW(VMobject):
    CONFIG = {
        "colors" : [GREY, GREEN, GREEN, BLUE],
        "fill_opacities" : [0.75, 0.75, 0.75, 0.9],
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
        self.squares = VGroup(outer_square, ul_square, dr_square, inner_square)
        self.add(self.squares)
        self.set_height(self.height)


class ImageWithRemark(Mobject):
    CONFIG = {
        "image_width" : 3,
        "text_width" : None,
        "text_position" : DOWN,
        "text_aligned_edge" : ORIGIN,
        "text_buff" : 0.2,
    }
    def __init__(self, image_filename, remark_text, **kwargs):
        self.image_filename = image_filename
        self.remark_text = remark_text
        Mobject.__init__(self, **kwargs)

    def generate_points(self):
        image = ImageMobject(self.image_filename)
        remark = TextMobject(self.remark_text)
        if self.image_width is not None:
            image.set_width(self.image_width)
        if self.text_width is not None:
            remark.set_width(self.text_width)
        remark.next_to(
            image, self.text_position,
            aligned_edge = self.text_aligned_edge, buff = self.text_buff
        )
        self.add(image)
        self.add(remark)
        self.center()
        self.image = image
        self.remark = remark

    def get_image(self):
        return self.image

    def get_remark(self):
        return self.remark


class ChessBoard(VMobject):
    CONFIG = {
        "height" : 5,
    }
    def __init__(self, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.chess_pieces = []
        self.add_board()
        self.add_border()
        self.add_labels()
        self.set_height(self.height)

    def add_board(self):
        board = VGroup(*[
            Square(
                side_length = 0.8,
                stroke_width = 0, fill_opacity = 1,
                fill_color = CB_DARK if (i+j)%2!=0 else CB_LIGHT
            )
            for i in range(8) for j in range(8)
        ])
        board.arrange_submobjects_in_grid(8, 8, buff = 0)
        self.add(board)
        self.board = board

    def add_border(self):
        border = Square(side_length = self.board.get_height())
        border.move_to(self.board.get_center())
        self.add(border)
        self.border = border

    def get_square(self, position = "a1"):
        l1, l2 = position.lower()
        row = ord(l1) - 97
        column = int(l2) - 1
        return self.board[(7-column)*8+row]

    def add_labels(self):
        numeral_labels = VGroup(*[
            TextMobject(str(i+1)).next_to(self.get_square("a"+str(i+1)), LEFT)
            for i in range(8)
        ])
        alphabetical_labels = TextMobject(*[chr(97+i) for i in range(8)])
        alphabetical_labels.next_to(self.border, DOWN, buff = 0.15)
        for i, label in enumerate(alphabetical_labels):
            label.next_to(self.get_square(chr(97+i)+"1"), DOWN, coor_mask = [1, 0, 0])
        self.add(numeral_labels, alphabetical_labels)
        self.numeral_labels = numeral_labels
        self.alphabetical_labels = alphabetical_labels

    def add_piece(self, position, camp_letter, piece_letter):
        piece = ChessPiece(camp_letter.upper(), piece_letter.upper())
        piece.set_height(self.border.get_height()/8)
        piece.move_to(self.get_square(position))
        self.chess_pieces.append(piece)
        self.add(piece)

    def get_pieces(self):
        return Group(*self.chess_pieces)

    def get_labels(self):
        return VGroup(*(self.numeral_labels.submobjects + self.alphabetical_labels.submobjects))

    def get_border(self):
        return self.border


class ChessPiece(ImageMobject):
    def __init__(self, camp_letter, piece_letter, **kwargs):
        png_filename = "ChessPiece_" + camp_letter + piece_letter + ".png"
        ImageMobject.__init__(self, png_filename)


class HexagonalGrid(VMobject):
    CONFIG = {
        "n" : 4,
        "side_length" : 0.45,
    }
    def generate_points(self):
        hex_unit = RegularPolygon(n = 6, color = WHITE, stroke_width = 3, start_angle = PI/6.)
        hex_unit.set_height(self.side_length*2)
        row_mobs = []
        for k in range(self.n):
            # Generate each row
            row_hexs = VGroup(*[hex_unit.deepcopy() for p in range(k+1)])
            row_hexs.arrange_submobjects(RIGHT, buff = 0)
            if k > 0:
                prev_row_hexs = row_mobs[-1]
                row_hexs.move_to(prev_row_hexs.get_center() + 1.5 * self.side_length * DOWN)
            row_mobs.append(row_hexs)
        for mobs in row_mobs:
            for mob in mobs:
                self.add(mob)
        self.center()


class Domino(Rectangle):
    CONFIG = {
        "fill_opacity" : 1,
        "fill_color" : TILE_GREEN,
        "stroke_width" : 3,
        "stroke_color" : WHITE,
        "angle" : 0,
    }
    def __init__(self, side_length = 0.7, **kwargs):
        Rectangle.__init__(self, height = side_length, width = 2*side_length, **kwargs)
        self.rotate(self.angle)

class HDomino(Domino):
    pass

class VDomino(Domino):
    CONFIG = {
        "fill_color" : TILE_RED,
        "angle" : PI/2.
    }


class Hexomino(Polygon):
    CONFIG = {
        "fill_opacity" : 1,
        "fill_color" : TILE_GREEN,
        "stroke_width" : 3,
        "stroke_color" : WHITE,
        "angle" : 0,
    }
    def __init__(self, side_length = 0.45, **kwargs):
        hex_l = RegularPolygon(n = 6, color = WHITE, stroke_width = 3, start_angle = PI/6.)
        hex_l.set_height(side_length*2)
        hex_r = hex_l.deepcopy()
        hex_r.rotate(PI).next_to(hex_l, RIGHT, buff = 0)
        vertices = np.append(hex_l.get_anchors()[:-1], hex_r.get_anchors()[1:-1])
        vertices = vertices.reshape(-1, 3)
        Polygon.__init__(self, *vertices, **kwargs)
        self.rotate(self.angle)

class HHexomino(Hexomino):
    pass

class RHexomino(Hexomino):
    CONFIG = {
        "fill_color" : TILE_RED,
        "angle" : PI/3.,
    }

class LHexomino(Hexomino):
    CONFIG = {
        "fill_color" : TILE_BLUE,
        "angle" : -PI/3.,
    }


#####
## Main Scenes

class PWWIntroScene(Scene):
    CONFIG = {
        "connection_arrow_colors" : [ORANGE, YELLOW, ORANGE],
    }
    def construct(self):
        self.show_sqrt2pww_details()
        self.show_calisson_intro()

    def show_sqrt2pww_details(self):
        arrow = Arrow(2.5*LEFT, 2.5*RIGHT)
        arrow.set_color(self.connection_arrow_colors)
        dashed_arrow = DashedMobject(arrow).set_stroke(width = 1)
        sqrt2_pww = Sqrt2PWW().next_to(arrow, LEFT, buff = 0.5)
        sqrt2_claim = TextMobject("$\\sqrt{2}$是无理数").scale(1.2).next_to(arrow, RIGHT, buff = 0.5)
        self.play(
            FadeInFromDown(VGroup(sqrt2_pww, dashed_arrow, sqrt2_claim)),
            submobject_mode = "lagged_start",
        )
        self.wait()
        # Show the hidden geometric relation
        outer_square, ul_square, dr_square, inner_square = sqrt2_pww.squares
        ul_copy = VGroup(ul_square, inner_square).deepcopy()
        dr_copy = VGroup(dr_square, inner_square).deepcopy()
        outer_copy = sqrt2_pww.deepcopy()
        geo_relation = VGroup(ul_copy, TexMobject("+").scale(2), dr_copy, TexMobject("=").scale(2), outer_copy)
        geo_relation.arrange_submobjects(RIGHT)
        geo_relation[::2].set_stroke(width = 2.5)
        geo_relation.set_width(dashed_arrow.get_width()*0.6)
        geo_relation.next_to(dashed_arrow, UP, buff = 0.5)
        # Show the hidden algebraic relation
        text_a, text_b, text_amb, text_2bma = texts = VGroup(*[
            TexMobject(text) for text in ("a", "b", "a-b", "2b-a")
        ])
        dl_small_square = Square(
            side_length = outer_square.get_height() - dr_square.get_height()
        ).next_to(ul_square, DOWN, aligned_edge = LEFT, buff = 0)
        brace_a, brace_b, brace_amb, brace_2bma = braces = VGroup(*[
            Brace(square, direction, buff = 0.05).put_at_tip(text_mob.scale(factor), buff = 0.1)
            for square, direction, text_mob, factor in zip(
                    [outer_square, ul_square, dl_small_square, inner_square],
                    [UP, LEFT, LEFT, DOWN],
                    texts,
                    [1, 1, 1, 0.7]
                )
        ])
        alg_relation = TexMobject("\\sqrt{2} = ", "\\dfrac{a}{b} = \\dfrac{2b-a}{a-b}")
        alg_relation.next_to(geo_relation, UP, buff = 0.5)
        sur_rect = SurroundingRectangle(alg_relation[-1], color = RED)
        remark = TextMobject("$\\sqrt{2}$没有最简分数形式！")
        remark.set_color(RED).scale(0.8).next_to(sur_rect, UP)
        self.play(Write(geo_relation), run_time = 2)
        self.wait()
        self.play(
            *[GrowFromCenter(brace) for brace in (brace_a, brace_b, brace_amb, brace_2bma)],
            run_time = 1,
        )
        self.play(
            *[Write(text) for text in (text_a, text_b, text_amb, text_2bma)],
            run_time = 2,
        )
        self.wait()
        self.play(Write(alg_relation), run_time = 2)
        self.play(ShowCreation(sur_rect), run_time = 1)
        self.play(Write(remark), run_time = 1)
        self.wait()
        self.play(GrowArrow(arrow), run_time = 2)
        self.wait()
        # Remove everything for the next scene
        self.arrow = arrow.deepcopy()
        self.dashed_arrow = dashed_arrow
        sqrt2_group = VGroup(
            sqrt2_pww, arrow, sqrt2_claim, braces, texts, geo_relation,
            alg_relation, sur_rect, remark,
        )
        self.play(ApplyMethod(sqrt2_group.shift, 8*UP), run_time = 1.5)
        self.wait()

    def show_calisson_intro(self):
        # Show two dots as ideas
        dot_left, dot_right = dots = VGroup(*[
            Dot(radius = 0.2).next_to(self.dashed_arrow, direction, buff = 0.5)
            for direction in (LEFT, RIGHT)
        ])
        self.play(*[GrowFromCenter(dot) for dot in dots])
        self.wait()
        # Show convoluted path between ideas
        conv_path = SVGMobject(
            file_name = "conv_path.svg",
            stroke_width = 10, stroke_color = RED, fill_opacity = 0
        )
        conv_path.set_width(self.dashed_arrow.get_width())
        conv_path_anim = NormalAnimationAsContinualAnimation(
            ShowCreation(conv_path, run_time = 60, rate_func = linear)
        )
        self.add(conv_path_anim)
        self.wait(5)
        # Show easy path between ideas
        easy_path = Arrow(
            dot_left.get_center(), dot_right.get_center(),
            use_rectangular_stem = False, path_arc = -PI/2, buff = 0.5,
        )
        easy_path.get_tip().shift(0.04*DR)
        easy_path.set_color(self.connection_arrow_colors)
        self.play(ShowCreation(easy_path, run_time = 1.5))
        self.wait(3)
        # Show Calisson tiling ideas
        ct_3d = CalissonTiling3D(
            dimension = 5, pattern = MAA_PATTERN,
            tile_config = {"stroke_width" : 1}, enable_fill = False,
        )
        ct_2d = CalissonTiling2D(ct_3d, enable_dumbbells = False).set_height(4)
        ct_counter = TilesCounter(
            matching_tiling = ct_2d, tile_fill_opacity = 0,
        )
        ct_counter.set_height(3.5).set_color(WHITE)
        ct_2d.next_to(self.dashed_arrow, LEFT, buff = 0.5)
        ct_counter.next_to(self.dashed_arrow, RIGHT, buff = 0.5)
        for mob, dot in zip([ct_2d, ct_counter], dots):
            mob.generate_target()
            mob.scale(0).move_to(dot.get_center())
            dot.generate_target()
            dot.target.scale(0)
        self.play(
            *[MoveToTarget(mob) for mob in (ct_2d, ct_counter, dot_left, dot_right)],
            ApplyMethod(easy_path.shift, 1.2*UP)
        )
        self.wait(4)


class ShowCalissonTilingProblem(Scene):
    def construct(self):
        dim = 3
        ct_grid = CalissonTilingGrid(
            side_length = dim, unit_size = 0.8, grid_lines_type = DashedLine,
            grid_boundary_config = {"color" : WHITE},
            grid_lines_config = {"stroke_width" : 2, "color" : GREY},
        )
        ct_grid.add_boundary()
        ct_grid.shift(4*RIGHT)
        ct_counter = TilesCounter([dim**2]*3, tile_stroke_width = 3)
        ct_counter.shift(3.5*LEFT)
        tiles = ct_counter.get_tiles()
        mult_signs, nums = ct_counter.get_mult_signs(), ct_counter.get_nums()
        # Scale ct_counter to match relative size
        scale_factor = ct_grid.get_width() / (tiles.get_width() * dim)
        ct_counter.scale(scale_factor)
        # Show hexagon, calissons and grid lines inside them
        self.play(ShowCreation(ct_grid), run_time = 2)
        self.wait()
        tiles.save_state()
        tiles.set_fill(opacity = 0)
        self.play(ShowCreation(tiles, submobject_mode = "lagged_start"), run_time = 2)
        self.wait()
        ct_grid.generate_grid_lines()
        grid_lines_copy = ct_grid.get_randomized_line_copy()
        self.play(
            ShowCreation(grid_lines_copy, submobject_mode = "lagged_start"),
            *[ShowCreation(tile.get_refline()) for tile in tiles],
            run_time = 5,
        )
        self.wait()
        # Take these, make this - SpaceChem reference, but it's a bad one...
        arrow = Arrow(1.5*LEFT, 1.5*RIGHT, buff = 0)
        tile_sur_rect = SurroundingRectangle(tiles, buff = 0.3).flip()
        grid_sur_rect = SurroundingRectangle(ct_grid, buff = 0.3)
        take_these_text = TextMobject("用这些...", color = YELLOW).next_to(tile_sur_rect, RIGHT).shift(0.5*UP)
        make_this_text = TextMobject("...拼这个", color = YELLOW).next_to(grid_sur_rect, LEFT).shift(0.5*UP)
        self.play(ShowCreation(tile_sur_rect), Write(take_these_text), run_time = 1)
        self.wait()
        self.play(
            ReplacementTransform(tile_sur_rect, grid_sur_rect),
            ReplacementTransform(take_these_text, make_this_text),
            GrowArrow(arrow),
            run_time = 1
        )
        self.wait()
        self.play(FadeOut(grid_sur_rect), FadeOut(make_this_text))
        self.wait()
        # Show requirements for the tiling
        reqs = VGroup(*[TextMobject(text, color = GREEN) for text in ["1. 完整覆盖", "2. 互不重叠"]])
        reqs.arrange_submobjects(DOWN).next_to(arrow, UP)
        self.play(Write(reqs[0]))
        self.wait()
        self.play(Write(reqs[1]))
        self.wait()
        reqs_sur_rect = SurroundingRectangle(reqs)
        tiling_text = TextMobject("“镶嵌”")
        tiling_text.next_to(reqs_sur_rect, UP).set_color(YELLOW)
        self.play(ShowCreation(reqs_sur_rect))
        tilings = [
            CalissonTiling2D(
                CalissonTiling3D(
                    dim = 3, pattern = generate_pattern(3),
                    enable_fill = True,
                ),
                ct_grid = ct_grid, enable_dumbbells = False,
                tile_config = {"stroke_width" : 3},
            )
            for k in range(3)
        ]
        example_tiling = tilings[0]
        example_tiling.set_fill(opacity = 0)
        example_tiling.shuffle_tiles()
        self.play(Write(tiling_text), ApplyMethod(grid_lines_copy.set_stroke, {"width" : 1}))
        self.play(
            FadeOut(VGroup(tiling_text, reqs, reqs_sur_rect)),
            TilesGrow(example_tiling), 
            run_time = 2,
        )
        self.wait()
        # Show different tiling methods
        another_tiling = tilings[1]
        another_tiling.save_state()
        another_tiling.set_fill(opacity = 0)
        self.play(ReplacementTransform(example_tiling, another_tiling, path_arc = PI/2., run_time = 3))
        self.wait()
        # Count and color each type of calisson...
        another_tiling_copy = another_tiling.deepcopy()
        self.play(*[
                ReplacementTransform(tile_set, target_tile)
                for tile_set, target_tile in zip(
                    another_tiling_copy.tiles, [tiles[1], tiles[0], tiles[2]]
                )
            ],
            *[FadeOut(tile.get_refline()) for tile in tiles],
            Restore(another_tiling),
            Restore(tiles),
            Write(VGroup(mult_signs, nums)),
            submobject_mode = "lagged_start",
            run_time = 3,
        )
        self.wait()
        # ...then show they are always the same
        yet_another_tiling = tilings[2]
        yet_another_tiling.shuffle_tiles()
        self.play(ReplacementTransform(another_tiling, yet_another_tiling, path_arc = PI/3., run_time = 3))
        self.wait()
        nums_sur_rect = SurroundingRectangle(nums)
        nums_text = TextMobject("相等！", color = YELLOW)
        nums_text.next_to(nums_sur_rect, UP)
        self.play(ShowCreation(nums_sur_rect))
        self.play(Write(nums_text))
        self.wait()


class ShowCalissonTilingSources(Scene):
    def construct(self):
        # Show authors
        david = ImageWithRemark("Guy_David.jpg", "Guy David")
        tomei = ImageWithRemark("Carlos_Tomei.jpg", "Carlos Tomei", image_width = 2.5)
        potraits = Group(david, tomei)
        potraits.arrange_submobjects(DOWN, buff = 0.5)
        potraits.to_edge(LEFT)
        self.play(LaggedStart(FadeInFromDown, potraits, lag_ratio = 0.6))
        self.wait()
        # Show the original article on MAA
        maa_notes = ImageWithRemark(
            "MAA_notes_on_calissons.png",
            "\\emph{Amer. Math. Monthly}, \\\\ 1989, 96(5): 429-431",
            text_position = UP, image_width = 3.8, text_width = 3.2,
        )
        self.play(FadeInFrom(maa_notes, direction = LEFT))
        self.wait()
        # Show the simplified version in PWW
        pww_page = ImageWithRemark(
            "PWW_on_calissons.png",
            "\\emph{Proof Without Words: \\\\ Exercises in Visual Thinking \\\\} pp. 142",
            text_position = UP, image_width = 3.8, text_width = 3.2,
        )
        pww_page.next_to(maa_notes, RIGHT, aligned_edge = DOWN, buff = 0.8)
        self.play(FadeInFrom(pww_page, direction = LEFT))
        self.wait()
        # Zoom in on the proof part
        proof_part = ImageMobject("PWW_on_calissons_proof_part.png")
        pww_page_image = pww_page.get_image()
        proof_part.set_width(pww_page_image.get_width())
        proof_part.next_to(
            0.546875*pww_page_image.get_critical_point(DL) + \
            0.453125*pww_page_image.get_critical_point(UL), # A tricky overlay
            DR, buff = 0
        )
        self.add(proof_part)
        proof_part.generate_target()
        proof_part.target.set_height(5)
        proof_part.target.center()
        self.play(FadeOut(Group(david, tomei, maa_notes, pww_page)))
        self.play(MoveToTarget(proof_part))
        self.wait()
        proof_sur_rect = SurroundingRectangle(proof_part, color = RED)
        proof_sur_rect.scale(0.08).shift(3.71*LEFT+2.24*UP)
        figure_sur_rect = SurroundingRectangle(proof_part, color = RED)
        figure_sur_rect.scale(0.8).shift(0.07*RIGHT+0.21*DOWN)
        self.play(ShowCreation(proof_sur_rect))
        self.wait()
        self.play(ReplacementTransform(proof_sur_rect, figure_sur_rect))
        self.wait()
        self.play(FadeOut(figure_sur_rect))
        self.wait()


class Demonstrate3DProof(ThreeDScene):
    def construct(self):
        ct_3d = CalissonTiling3D(
            dimension = 5, pattern = MAA_PATTERN, enable_fill = True, height = 2.5,
            tile_config = {"stroke_width" : 3,}
        )
        ct_grid_overlay = CalissonTilingGrid(unit_size = 0.39)
        ct_grid_overlay.move_to(2.16*RIGHT+0.16*DOWN)       # Yet another tricky overlay
        ct_2d_overlay = CalissonTiling2D(ct_3d, ct_grid_overlay)
        ct_2d = CalissonTiling2D(ct_3d).set_height(5*np.sqrt(6)/3.)
        self.add(ct_2d_overlay)
        # Starting from the end of the previous scene
        proof_part = ImageMobject("PWW_on_calissons_proof_part.png")
        proof_part.set_height(5)
        self.add(proof_part)
        # Reveal a more 'vectorized' proof
        self.play(FadeOut(proof_part))
        self.wait()
        self.play(ReplacementTransform(ct_2d_overlay, ct_2d))
        self.wait()
        # Switch from 2d to 3d
        self.set_camera_orientation(*DIAG_POS)
        self.remove(ct_2d)
        self.add(ct_3d)
        # Wiggle the camera
        nudge = TAU / 30
        phi, theta, distance = DIAG_POS
        self.move_camera(phi - nudge, theta, run_time = 1)
        angle_tracker = ValueTracker(TAU/4)
        self.camera.phi_tracker.add_updater(
            lambda t: t.set_value(phi - np.sin(angle_tracker.get_value())*nudge)
        )
        self.camera.theta_tracker.add_updater(
            lambda t: t.set_value(theta + np.cos(angle_tracker.get_value())*nudge)
        )
        self.play(angle_tracker.set_value, 5*TAU/4, run_time = 3)
        self.camera.phi_tracker.clear_updaters()
        self.camera.theta_tracker.clear_updaters()
        self.wait()
        self.move_camera(*DIAG_POS, run_time = 1)
        self.wait()
        # Switch back to 2d and replicate
        self.set_camera_orientation(*DEFAULT_POS)
        self.remove(ct_3d)
        self.add(ct_2d)
        ct_2d_left = ct_2d.deepcopy().move_to(LEFT_SIDE*2/3.)
        ct_2d_right = ct_2d.deepcopy().move_to(RIGHT_SIDE*2/3.)
        self.play(
            ReplacementTransform(ct_2d.deepcopy(), ct_2d_left),
            ReplacementTransform(ct_2d.deepcopy(), ct_2d_right),
        )
        self.wait()


class Demonstrate3DProofFromFront(ThreeDScene):
    CONFIG = {
        "camera_position" : FRONT_POS,
        "tile_color" : TILE_RED,
        "text_color" : RED,
        "RhombusType" : RRhombus,
    }
    def setup(self):
        self.set_camera_orientation(*DIAG_POS)
        ct_3d = CalissonTiling3D(
            dimension = 5, pattern = MAA_PATTERN, enable_fill = True, height = 2.5,
            tile_config = {"stroke_width" : 3,}
        )
        self.add(ct_3d)
        self.ct_3d = ct_3d

    def construct(self):
        braces = VGroup(*[Brace(self.ct_3d, direction) for direction in (LEFT, DOWN)])
        texts = VGroup(*[TexMobject("n").set_color(self.text_color) for i in range(2)])
        for brace, text in zip(braces, texts):
            brace.put_at_tip(text)
        tile = self.RhombusType()
        tile.set_fill(color = self.tile_color, opacity = 1)
        tile.set_stroke(width = 3)
        tile.scale(0.6)
        counting_text = VGroup(TexMobject("\\times n^2").set_color(self.text_color)).scale(2)
        VGroup(tile, counting_text).arrange_submobjects(RIGHT).shift(3*UP)
        counting_text.shift(0.15*UP)
        self.camera.add_fixed_in_frame_mobjects(braces, texts, tile, counting_text)
        self.play(DrawBorderThenFill(tile), run_time = 1)
        self.wait()
        self.move_camera(*self.camera_position, run_time = 5)
        self.wait()
        self.play(*[GrowFromCenter(brace) for brace in braces])
        self.play(*[Write(text) for text in texts])
        self.wait()
        self.play(FadeInFromDown(counting_text))
        self.wait()

        
class Demonstrate3DProofFromUp(Demonstrate3DProofFromFront):
    CONFIG = {
        "camera_position" : UP_POS,
        "tile_color" : TILE_GREEN,
        "text_color" : GREEN,
        "RhombusType" : HRhombus,
    }


class Demonstrate3DProofFromRight(Demonstrate3DProofFromFront):
    CONFIG = {
        "camera_position" : RIGHT_POS,
        "tile_color" : TILE_BLUE,
        "text_color" : BLUE,
        "RhombusType" : LRhombus,
    }


class HardToInterpretWithRigor2DPart(ThreeDScene):
    CONFIG = {
        "arrow_position" : RIGHT_SIDE/2.,
    }
    def setup(self):
        self.ct_3d = CalissonTiling3D(
            dimension = 5, pattern = MAA_PATTERN, enable_fill = True, height = 2.5,
            tile_config = {"stroke_width" : 3,}
        )
        self.ct_2d = CalissonTiling2D(self.ct_3d).set_height(5*np.sqrt(6)/3.)
        arrow = TexMobject("\\rightarrow").scale(3).move_to(self.arrow_position)
        q_mark = TexMobject("?").scale(1.5).next_to(arrow, UP, buff = 0)
        self.add_fixed_in_frame_mobjects(arrow, q_mark)

    def construct(self):
        plane = Rectangle(
            width = 4, height = 5,
            stroke_width = 1, fill_color = LIGHT_GREY, fill_opacity = 0.7,
        )
        plane.move_to(self.ct_2d)
        self.add_foreground_mobjects(self.ct_2d)
        self.play(DrawBorderThenFill(plane), run_time = 1)
        self.wait()
        self.move_camera(PI/4, -PI*5/9)
        self.begin_ambient_camera_rotation(rate = 0.03)
        self.wait(20)


class HardToInterpretWithRigor3DPart(HardToInterpretWithRigor2DPart):
    CONFIG = {
        "arrow_position" : LEFT_SIDE/2.,
    }
    def construct(self):
        self.set_camera_orientation(*DIAG_POS)
        self.add(self.ct_3d)
        self.wait()
        self.wait()     # Sync with 2D part
        self.move_camera(PI/3, PI/9)
        self.begin_ambient_camera_rotation(rate = 0.03)
        self.wait(20)


class RigorousYetArcaneProofBasedOn3D(Scene):
    def construct(self):
        self.add_the_proof()
        self.add_figure()
        self.scroll_to_reveal_the_proof()
        self.show_the_counter_example()

    def add_the_proof(self):
        quote_marks = TextMobject("``", "''").scale(3)
        quote_marks[0].to_corner(UL)
        quote_marks[1].center().shift(2.5*DOWN)
        up_rect = Rectangle(
            height = 0.7, width = FRAME_WIDTH,
            stroke_width = 0, fill_opacity = 1, fill_color = BLACK,
        ).to_edge(UP, buff = 0)
        down_rect = Rectangle(
            height = 1.5, width = FRAME_WIDTH,
            stroke_width = 0, fill_opacity = 1, fill_color = BLACK,
        ).to_edge(DOWN, buff = 0)
        proof = TextMobject(
            """将图中$A$点的坐标定为$(n,n,0)$，根据图中的路径， \\\\
            确定其他顶点的坐标。\\\\ """,
            """假设某个顶点$W$的坐标为$(a,b,c)$，从$W$出发沿着 \\\\
            路径走到顶点$U$，则$U$的坐标由如下规则确定：\\\\""",
            """(1) $WU \\parallel AF$，如果$U$在$W$上方，就将$U$的坐标 \\\\
            定为$(a,b-1,c)$，否则定为$(a,b+1,c)$；\\\\""",
            """(2) $WU \\parallel BC$，如果$U$在$W$上方，就将$U$的坐标 \\\\
            定为$(a,b,c+1)$，否则定为$(a,b,c-1)$；\\\\""",
            """(3) $WU \\parallel AB$，如果$U$在$W$上方，就将$U$的坐标 \\\\
            定为$(a-1,b,c+1)$，否则定为$(a+1,b,c)。$\\\\""",
            """这是一套良定的规则，证明留给读者自行完成。\\\\""",
            """令$s$为菱形在六边形边界上的一条边，“$s$-链”由\\\\
            一系列菱形构成。$s$-链上的第一个菱形以$s$为边，\\\\
            第$(k+1)$个菱形与第$k$个菱形共用一条边，且这条\\\\
            公共边与$s$平行。$s$三种方向，分别记为$d_1,\\,d_2$和$d_3$。\\\\
            如果$s$的方向为$d_i$，我们称这条$s$-链是“$d_i$类”的。\\\\""",
            """利用之前定义的三维坐标，不难证明如下结论：\\\\""",
            """(1) $s$-链从边界的一边延伸到对边；\\\\""",
            """(2) $s$-链上所有菱形的长对角线都不平行于$s$；\\\\""",
            """(3) 不同类的$s$-链会共用菱形，同类的$s$-链则不会\\\\
            共用菱形；\\\\"""
            """(4) 所有落在两个不同类$s$-链上的菱形的朝向一致。\\\\"""
            """每类$s$-链都有$n$条，所以对于两类$s$-链而言，它们\\\\
            至少会相交于$n$个相同朝向的菱形。而六边形的\\\\
            总菱形数为$3n^2$（简单用面积算一下就知道了），\\\\
            所以每种菱形的个数必然是$n^2$。证毕。""",
            alignment = "",
        )
        proof.arrange_submobjects(DOWN, aligned_edge = LEFT, buff = 0.3)
        proof[5].set_color(YELLOW)
        proof[-4].set_color(YELLOW)
        proof_width = (quote_marks[1][0].get_left()[0] - quote_marks[0][1].get_right()[0]) * 0.95
        proof.set_width(proof_width)
        proof.next_to(quote_marks[0][-1], RIGHT)
        proof.next_to(down_rect.get_top(), DOWN, coor_mask = [0, 1, 0], buff = 0.1)
        self.add(proof)
        self.add(up_rect, down_rect)
        self.quote_marks = quote_marks
        self.proof = proof

    def add_figure(self):
        ct_3d = CalissonTiling3D(
            dimension = 5, pattern = MAA_PATTERN, enable_fill = True,
            tile_config = {"stroke_width" : 1.5,}
        )
        ct_2d = CalissonTiling2D(ct_3d).set_height(3.5)
        ct_2d.shift(3*RIGHT)
        ct_2d.to_edge(UP, buff = 0.8)
        self.add(ct_2d)
        # Add labels of vertices
        circle = Circle(start_angle = -PI/2)
        circle.surround(ct_2d, buffer_factor = 0.9)
        labels = VGroup(*[
            TexMobject(c).scale(0.7).move_to(circle.point_from_proportion(k/6))
            for k, c in enumerate("ABCDEF")
        ])
        self.play(LaggedStart(Write, labels, lag_ratio = 0.4))
        self.wait()
        self.ct_2d = ct_2d
        self.labels = labels

    def scroll_to_reveal_the_proof(self):
        self.play(Write(self.quote_marks))
        rate = 0.9
        self.proof.add_updater(lambda mob, dt: mob.shift(rate * dt * UP))

    def show_the_counter_example(self):
        self.wait(8)
        counter_example = CalissonRing()
        counter_example.set_stroke(width = 2)
        counter_example.match_height(self.ct_2d)
        counter_example.move_to(self.ct_2d)
        question_mark = TexMobject("?", color = YELLOW).scale(5).move_to(self.quote_marks)
        sur_rect = SurroundingRectangle(
            self.proof, stroke_width = 0, fill_color = BLACK, fill_opacity = 0.6, buff = 0,
        )
        self.play(
            FadeOut(VGroup(self.ct_2d, self.labels)), FadeInFromDown(counter_example),
            FadeIn(sur_rect), Write(question_mark),
            run_time = 1.5,
        )
        self.wait(5)
        self.play(
            FadeIn(self.ct_2d),
            FadeOut(VGroup(question_mark, self.quote_marks, counter_example)),
            ApplyMethod(sur_rect.set_fill, {"opacity" : 1}),
        )
        self.remove(self.proof)
        self.wait(3)


class CuttingEdgeOrEdgeCutting(Scene):
    def construct(self):
        self.add_an_example_tiling()
        self.show_dijkstra_portrait()
        self.show_ce_and_ec_texts()

    def add_an_example_tiling(self):
        # An example tiling
        tiling_3d = CalissonTiling3D(
            dimension = 5, pattern = MAA_PATTERN, enable_fill = True,
            tile_config = {"stroke_width" : 1.5},
        )
        tiling_2d = CalissonTiling2D(tiling_3d)
        tiling_2d.set_height(3.5)
        tiling_2d.shift(3*RIGHT)
        tiling_2d.to_edge(UP, buff = 0.8)
        self.add(tiling_2d)
        self.tiling_2d = tiling_2d

    def show_dijkstra_portrait(self):
        dijkstra = ImageWithRemark(
            "Edsger_Dijkstra.jpg", "Edsger W. Dijkstra", image_width = 4,
        )
        dijkstra.to_edge(LEFT)
        dijkstra_potrait = dijkstra.get_image()
        dijkstra_text = dijkstra.get_remark()
        self.play(FadeInFromDown(dijkstra_potrait), Write(dijkstra_text))
        self.wait()

    def show_ce_and_ec_texts(self):
        ce = TextMobject("A",  "cutting-edge", "method?")
        ec = TextMobject("An", "edge-cutting", "method!")
        text_group = VGroup(ce, ec)
        text_group.scale(1.2)
        text_group.arrange_submobjects(DOWN, buff = 0.5)
        text_group.next_to(self.tiling_2d, DOWN, buff = 0.8)
        ce_copy = ce.deepcopy()
        cross = Cross(ce)
        self.play(Write(ce_copy))
        self.add(ce)
        self.wait()
        self.play(ShowCreation(cross))
        self.wait()
        self.play(ApplyMethod(ce.move_to, ec))
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


class PauseAndThinkWhenNecessary(Scene):
    def construct(self):
        lightbulb = SVGMobject("Incandescent_LightBulb.svg", height = 2)
        lightbulb.move_to(2*LEFT)
        self.play(DrawBorderThenFill(lightbulb), run_time = 1)
        self.wait()
        bubble = ThoughtBubble()
        question = TextMobject("?")
        bubble.add_content(question)
        bubble.resize_to_content()
        question.scale(2)
        VGroup(bubble, question).move_to(2*RIGHT)
        self.play(BubbleCreation(bubble), run_time = 1)
        self.wait()
        pause_button = PauseButton(color = RED)
        play_button = PlayButton(color = GREEN)
        VGroup(pause_button, play_button).scale(1.5).fade(0.8)
        self.play(FadeIn(pause_button))
        self.wait()
        ambient_light = AmbientLight(
            source_point = VectorizedPoint(lightbulb[1].get_center()),
            radius = 250, num_levels = 20000,
            opacity_function = lambda r: 1.0 / (r + 1.0)**1.2,
        )
        exclaimation = TexMobject("!")
        exclaimation.scale(2)
        exclaimation.move_to(question)
        self.play(
            SwitchOn(ambient_light),
            ApplyMethod(lightbulb[1].set_color, YELLOW),
            Transform(question, exclaimation),
            run_time = 1,
        )
        self.wait(0.25)
        self.remove(pause_button)
        self.add(play_button)
        self.wait()


class ExploreTheTriangularGrid(Scene):
    CONFIG = {
        "random_seed" : 570,
        "grid_unit_size": 1.1,
    }
    def setup(self):
        ct_grid = CalissonTilingGrid(
            side_length = 8, unit_size = self.grid_unit_size,
            grid_lines_config = {"stroke_width" : 2, "stroke_color" : WHITE},
            grid_triangles_config = {"opacity" : 1,}
        )
        self.ct_grid = ct_grid
        self.ct_grid_lines = self.ct_grid.get_grid_lines()
        self.ct_grid_triangles = self.ct_grid.get_grid_triangles()

    def construct(self):
        self.show_triangular_grid()
        self.show_calisson_and_hexagon_examples()
        self.demonstrate_two_types_of_triangles()

    def show_triangular_grid(self):
        ct_grid = self.ct_grid
        ct_grid_lines = ct_grid.get_randomized_line_copy()
        ct_grid_triangles = ct_grid.get_randomized_triangle_copy()
        self.play(LaggedStart(ShowCreation, ct_grid_lines), run_time = 5)
        self.wait()
        self.ct_grid_lines = ct_grid_lines
        self.ct_grid_triangles = ct_grid_triangles

    def show_calisson_and_hexagon_examples(self):
        eg_r, eg_h, eg_l = eg_calissons = VGroup(*[
            RhombusType(side_length = self.grid_unit_size).move_to(self.ct_grid.coords_to_point(*coords))
            for RhombusType, coords in zip(
                [RRhombus, HRhombus, LRhombus], [(4, -4.5), (2.5, -5), (0.5, -4.5)]
            )
        ])
        eg_calissons.set_fill(opacity = 1)
        eg_hexagon = self.ct_grid.get_grid_boundary().deepcopy()
        eg_hexagon.set_fill(color = GREY, opacity = 1)
        eg_hexagon.set_stroke(width = 5, color = WHITE)
        eg_hexagon.scale(0.375)
        eg_hexagon.next_to(self.ct_grid.coords_to_point(0.5, -1), RIGHT, buff = 0)
        self.ct_grid_lines.save_state()
        self.play(
            self.ct_grid_lines.set_stroke, {"width" : 0.5,},
            LaggedStart(DrawBorderThenFill, eg_calissons),
        )
        self.play(DrawBorderThenFill(eg_hexagon))
        self.wait()
        # Highlight grid lines
        ct_grid_lines_copy = self.ct_grid_lines.deepcopy()
        ct_grid_lines_copy.set_stroke(width = 5, color = YELLOW)
        self.add_foreground_mobjects(eg_calissons, eg_hexagon)
        self.play(ShowPassingFlash(ct_grid_lines_copy), time_width = 0.3, run_time = 2)
        self.remove_foreground_mobjects(eg_calissons, eg_hexagon)
        self.wait()
        self.play(FadeOut(VGroup(eg_hexagon, eg_calissons)))
        self.wait()

    def demonstrate_two_types_of_triangles(self):
        triangles = VGroup(*[
            self.ct_grid.get_grid_triangle(coords_comb)
            for coords_comb in ([(0, -1), (1, -1), (1, -2)], [(0, 1), (-1, 1), (-1, 2)])
        ])
        # Match up the orientation of the arrow's tip
        triangle_l, triangle_r = triangles.deepcopy()
        triangles[0].rotate(-PI*2/3.).move_to(triangle_l.get_center())
        triangles[1].flip().rotate(PI/3.).move_to(triangle_r.get_center())
        arrows = VGroup(*[
            Vector(
                direction, color = triangle.get_color(), rectangular_stem_width = 0.1,
                preserve_tip_size_when_scaling = False,
            ).next_to(triangle, UP).scale(1.5)
            for direction, triangle in zip([LEFT, RIGHT], triangles)
        ])
        for triangle, arrow in zip(triangles, arrows):
            self.play(DrawBorderThenFill(triangle))
            self.play(ShowCreation(arrow.stem), ReplacementTransform(triangle.deepcopy(), arrow.get_tip()))
            self.wait()
        self.play(FadeOut(arrows))
        self.wait()
        self.play(LaggedStart(FadeIn, self.ct_grid_triangles), run_time = 5)
        self.wait()


class ASimilarityToChessBoard(ExploreTheTriangularGrid):
    def setup(self):
        super().setup() 
        self.add(self.ct_grid_lines, self.ct_grid_triangles)

    def construct(self):
        self.add_chess_board()
        self.show_chess_board()
        self.show_property_no_1()
        self.show_property_no_2()

    def add_chess_board(self):
        bg_rect = Rectangle(
            height = FRAME_HEIGHT, width = FRAME_WIDTH/2., stroke_width = 0,
            fill_color = BLACK, fill_opacity = 1,
        )
        bg_rect.next_to(RIGHT_SIDE, RIGHT, buff = 0)
        chess_board = ChessBoard()
        chess_board.shift(FRAME_WIDTH * 0.75 * RIGHT)
        chess_board.to_edge(DOWN, buff = 0.2)
        chess_triplets = [
            ("A4", "B", "K"), ("C4", "W", "K"),
            ("A8", "W", "B"), ("E1", "W", "B"),
            ("B2", "W", "P"), ("D4", "W", "P"), ("E5", "W", "P"),
            ("E2", "B", "P"), ("E6", "B", "P"), ("H4", "B", "P"),
            ("H5", "B", "P"), ("H6", "B", "P"), ("H7", "B", "P"),
            ("D8", "B", "N"), ("G5", "B", "N"), ("F1", "B", "B"),
        ]
        for triplet in chess_triplets:
            chess_board.add_piece(*triplet)
        self.add(bg_rect, chess_board)
        self.bg_rect = bg_rect
        self.chess_board = chess_board

    def show_chess_board(self):
        self.play(
            ApplyMethod(Group(self.bg_rect, self.chess_board).shift, FRAME_WIDTH/2.*LEFT),
        )
        self.wait()
        border = self.chess_board.get_border()
        labels = self.chess_board.get_labels()
        pieces = self.chess_board.get_pieces()
        self.play(
            ApplyMethod(border.fade, 1),
            *[ApplyMethod(label.scale, 0) for label in labels],
            *[ApplyMethod(piece.scale, 0) for piece in pieces],
        )
        self.wait()
        pattern_text = TextMobject("“棋盘式染色”")
        pattern_text.set_color(WHITE)
        pattern_text.next_to(self.bg_rect.get_top(), DOWN)
        self.play(Write(pattern_text))
        self.wait()
        self.pattern_text = pattern_text

    def show_property_no_1(self):
        prop_1 = TextMobject("深色格", "与", "浅色格", "相邻")
        prop_1[0].set_color(CB_DARK)
        prop_1[2].set_color(CB_LIGHT)
        prop_1.next_to(self.pattern_text, DOWN, buff = 0.7)
        prop_1_remark = TextMobject("（“相邻”：有公共边）").scale(0.5)
        prop_1_remark.next_to(prop_1, DOWN, buff = 0.2)
        self.play(FadeInFromDown(VGroup(prop_1, prop_1_remark)))
        self.wait()
        self.play(Swap(prop_1[0], prop_1[2]))
        self.wait()
        self.play(FadeOut(VGroup(prop_1, prop_1_remark)))
        self.wait()

    def show_property_no_2(self):
        prop_2 = TextMobject("$\\rightarrow$".join(["深", "浅", "深", "浅", "$\\cdots$"]))
        prop_2[:-3:4].set_color(CB_DARK)
        prop_2[2:-3:4].set_color(CB_LIGHT)
        for arrow in prop_2[1:-3:2]:
            arrow.scale(0.8)
        prop_2.scale(1.2)
        prop_2.next_to(self.pattern_text, DOWN, buff = 0.7)
        sequence_tri = [RIGHT, DR, RIGHT, UR, RIGHT, DR, DL, LEFT, DL, DR]
        sequence_sqr = [RIGHT, DOWN, LEFT, DOWN, RIGHT, RIGHT, RIGHT, UP, RIGHT, UP]
        vertices_tri = self.get_path_vertices_on_triangular_grid(5*LEFT+ 1.2*UP, sequence_tri)
        path_tri = self.get_path_by_vertices(vertices_tri)
        path_tri.set_color(CYAN)
        vertices_sqr = self.get_path_vertices_on_square_grid("b6", sequence_sqr)
        path_sqr = self.get_path_by_vertices(vertices_sqr)
        path_sqr.set_color(CYAN)
        circle_tri = Circle(color = CYAN, stroke_width = 8, radius = 0.70).move_to(path_tri[0].get_start())
        circle_sqr = Circle(color = CYAN, stroke_width = 8, radius = 0.45).move_to(path_sqr[0].get_start())
        self.play(ShowCreation(circle_tri), ShowCreation(circle_sqr))
        self.wait()
        for k in range(len(sequence_tri)):
            if k <= 3:
                text = prop_2[2*k]
                text.generate_target()
                text.move_to(circle_tri.get_center()).scale(0)
                self.play(MoveToTarget(text))
                self.wait()
                self.play(
                    GrowArrow(path_tri[k]),
                    GrowArrow(path_sqr[k]),
                    ApplyMethod(circle_tri.move_to, vertices_tri[k+1]),
                    ApplyMethod(circle_sqr.move_to, vertices_sqr[k+1]),
                    Write(prop_2[2*k+1]),
                )
                self.wait()
            if k == 3:
                self.play(Write(prop_2[-3:]), run_time = 0.5)
            if k > 3:
                self.play(
                    GrowArrow(path_tri[k]),
                    GrowArrow(path_sqr[k]),
                    ApplyMethod(circle_tri.move_to, vertices_tri[k+1]),
                    ApplyMethod(circle_sqr.move_to, vertices_sqr[k+1]),
                )
        self.wait()
        # Remove unnecessary stuff
        self.play(
            FadeOut(VGroup(circle_tri, path_tri)),
            ApplyMethod(
                Group(
                    self.bg_rect, self.pattern_text, prop_2, self.chess_board, circle_sqr, path_sqr
                ).shift, FRAME_WIDTH/2. * RIGHT   
            ),
        )
        self.wait()

    def get_path_vertices_on_triangular_grid(self, starting_point, sequence):
        starting_triangle = self.ct_grid.get_grid_triangle_containing_point(starting_point)
        vertices = [get_grid_triangle_center(starting_triangle)]
        step_size = starting_triangle.get_height() / np.sqrt(3)
        direction_to_step_dict = {
            str(LEFT) : step_size*LEFT,
            str(RIGHT) : step_size*RIGHT,
            str(UL) : step_size*rotate_vector(LEFT, -PI/3.),
            str(UR) : step_size*rotate_vector(RIGHT, PI/3.),
            str(DL) : step_size*rotate_vector(LEFT, PI/3.),
            str(DR) : step_size*rotate_vector(RIGHT, -PI/3.),
        } 
        for direction in sequence:
            step = direction_to_step_dict[str(direction)]
            vertices.append(vertices[-1] + step)
        return vertices

    def get_path_vertices_on_square_grid(self, starting_position, sequence):
        starting_square = self.chess_board.get_square(starting_position)
        vertices = [starting_square.get_center()]
        step_size = starting_square.get_height()
        for direction in sequence:
            vertices.append(vertices[-1] + direction * step_size)
        return vertices
        
    def get_path_by_vertices(self, vertices):
        return VGroup(*[
            Arrow(vertices[k], vertices[k+1], buff = 0)
            for k in range(len(vertices)-1)
        ])


class ADifferenceToChessBoard(ASimilarityToChessBoard):
    def construct(self):
        self.remove(self.ct_grid_lines)
        all_grid_lines = self.ct_grid.get_grid_lines()
        grid_line_v = all_grid_lines[25]
        grid_line_v.set_color(CYAN).set_stroke(width = 8)
        valid_triangles = self.ct_grid.get_grid_triangle_on_line(grid_line_v)
        invalid_triangles = self.ct_grid.get_grid_triangles_not_on_line(grid_line_v)
        invalid_triangles.save_state()
        # Show a vertical grid line and all triangles that land on it
        self.play(ShowCreation(grid_line_v))
        self.remove(grid_line_v)
        self.add_foreground_mobjects(grid_line_v)
        self.wait()
        self.play(
            Indicate(valid_triangles, color = CYAN, scale_factor = 1),
            ApplyMethod(invalid_triangles.fade, 0.8)
        )
        self.wait()
        text_l = TextMobject("深棕色", color = CB_DARK).move_to(2*LEFT)
        text_r = TextMobject("浅棕色", color = CB_LIGHT).move_to(2*RIGHT)
        arrow_l = Arrow(ORIGIN, 2*LEFT, color = CB_DARK).next_to(text_l, UP)
        arrow_r = Arrow(ORIGIN, 2*RIGHT, color = CB_LIGHT).next_to(text_r, UP)
        self.play(*[GrowArrow(arrow) for arrow in (arrow_l, arrow_r)], run_time = 1)
        self.play(*[Write(text) for text in (text_l, text_r)], run_time = 1)
        self.wait()
        self.remove_foreground_mobjects(grid_line_v)
        self.play(FadeOut(VGroup(grid_line_v, text_l, text_r, arrow_l, arrow_r)), Restore(invalid_triangles))
        self.wait()
        # Show examples of the other 2 directions
        for eg_line in (all_grid_lines[7], all_grid_lines[42]):
            eg_line.set_color(CYAN).set_stroke(width = 8)
            valid_triangles = self.ct_grid.get_grid_triangle_on_line(eg_line)
            invalid_triangles = self.ct_grid.get_grid_triangles_not_on_line(eg_line)
            invalid_triangles.save_state()
            self.play(ShowCreation(eg_line))
            self.add_foreground_mobjects(eg_line)
            self.play(
                Indicate(valid_triangles, color = CYAN, scale_factor = 1),
                ApplyMethod(invalid_triangles.fade, 0.8)
            )
            self.wait()
            self.remove_foreground_mobjects(eg_line)
            self.play(FadeOut(eg_line), Restore(invalid_triangles))
            self.wait()


class AQuickRecapOnTriangularGrid(ASimilarityToChessBoard):
    def setup(self):
        super().setup()
        self.remove(self.ct_grid_lines)
        self.wait()

    def construct(self):
        self.add_property_texts()
        self.show_property_no_1()
        self.show_property_no_2()
        self.show_property_no_3()

    def add_property_texts(self):
        bg_rect = Rectangle(
            height = FRAME_HEIGHT, width = FRAME_WIDTH/2., stroke_width = 0,
            fill_color = BLACK, fill_opacity = 1,
        )
        bg_rect.next_to(LEFT_SIDE, LEFT, buff = 0)
        prop_title = TextMobject("正三角形网格的性质").set_color(YELLOW)
        prop_texts = VGroup(*[
            TextMobject("1. “棋盘式染色”"),
            TextMobject("2. 路径上的格子颜色深浅相间"),
            TextMobject("3. 网格线两侧紧贴的格子颜色不同")
        ]).arrange_submobjects(DOWN, aligned_edge = LEFT, buff = 0.4).scale(0.75)
        prop_title.next_to(bg_rect.get_top(), DOWN)
        prop_texts.shift(1.5*UP).next_to(bg_rect.get_left(), RIGHT, coor_mask = [1,0,0])
        self.play(VGroup(bg_rect, prop_title, prop_texts).shift, FRAME_WIDTH/2.*RIGHT)
        self.bring_to_front(VGroup(bg_rect, prop_title, prop_texts))
        self.wait()
        self.bg_rect = bg_rect
        self.prop_title = prop_title
        self.prop_texts = prop_texts

    def show_property_no_1(self):
        sur_rect = SurroundingRectangle(self.prop_texts[0])
        self.play(FadeIn(sur_rect))
        self.wait(2)
        self.sur_rect = sur_rect

    def show_property_no_2(self):
        sequence = [RIGHT, DR, DL, DR, RIGHT, DR, DL, LEFT, DL, LEFT, UL]
        vertices = self.get_path_vertices_on_triangular_grid(3.5*RIGHT+ 1.7*UP, sequence)
        path = self.get_path_by_vertices(vertices)
        path.set_color(CYAN)
        all_triangles = self.ct_grid.get_grid_triangles()
        in_triangles_list = [self.ct_grid.get_grid_triangle_containing_point(vertex) for vertex in vertices]
        in_triangles = VGroup(*in_triangles_list)
        out_triangles_list = list(set(all_triangles.submobjects) - set(in_triangles_list))
        out_triangles = VGroup(*out_triangles_list)
        out_triangles.save_state()
        self.add_foreground_mobjects(self.bg_rect, self.prop_title, self.prop_texts, self.sur_rect)
        self.play(
            ShowCreation(path),
            Transform(self.sur_rect, SurroundingRectangle(self.prop_texts[1])),
            ApplyMethod(out_triangles.fade, 0.7),
        )
        self.wait(2)
        self.play(FadeOut(path), Restore(out_triangles))
        self.wait()

    def show_property_no_3(self):
        all_grid_lines = self.ct_grid.get_grid_lines()
        grid_line_v = all_grid_lines[29]
        grid_line_v.set_color(CYAN).set_stroke(width = 8)
        valid_triangles = self.ct_grid.get_grid_triangle_on_line(grid_line_v)
        invalid_triangles = self.ct_grid.get_grid_triangles_not_on_line(grid_line_v)
        invalid_triangles.save_state()
        arrow_l = Vector(LEFT, color = CB_DARK).next_to(grid_line_v, LEFT, buff = 0.7)
        arrow_r = Vector(RIGHT, color = CB_LIGHT).next_to(grid_line_v, RIGHT, buff = 0.7)
        self.add(grid_line_v)
        self.play(
            ShowCreation(grid_line_v),
            ApplyMethod(invalid_triangles.fade, 0.7),
            Transform(self.sur_rect, SurroundingRectangle(self.prop_texts[2]))
        )
        self.play(GrowArrow(arrow_l), GrowArrow(arrow_r))
        self.wait(2)


class AlternateWayToExpressTiling(Scene):
    def setup(self):
        ct_grid = CalissonTilingGrid(
            side_length = 5, unit_size = 0.65,
            grid_triangles_config = {"opacity" : 0.4},
            grid_boundary_config = {"color" : WHITE},
        )
        ct_3d = CalissonTiling3D(
            dimension = 5, pattern = MAA_PATTERN, enable_fill = True,
            tile_config = {"stroke_width" : 2},
        )
        ct_2d = CalissonTiling2D(ct_3d, ct_grid)
        self.ct_grid = ct_grid
        self.ct_3d = ct_3d
        self.ct_2d = ct_2d

    def construct(self):
        ct_grid = self.ct_grid
        ct_3d = self.ct_3d
        ct_2d = self.ct_2d
        ct_grid_boundary = ct_grid.get_grid_boundary()
        ct_grid_triangles = ct_grid.get_grid_triangles()
        ct_grid_triangles_copy = ct_grid_triangles.deepcopy().set_fill(opacity = 0.6)
        center_dots = VGroup(*[
            Dot(get_grid_triangle_center(triangle), color = triangle.get_color())
            for triangle in ct_grid_triangles_copy
        ])
        center_dots_connections = VGroup()
        for dumbbell_set in ct_2d.get_all_dumbbells():
            for dumbbell in dumbbell_set:
                center_dots_connections.add(dumbbell.get_stem().deepcopy())
        random.shuffle(center_dots_connections.submobjects)
        for mobs in (center_dots, center_dots_connections):
            mobs.set_background_stroke(width = 2)
        center_dots_connections.set_color(GREY)
        self.add(ct_grid_boundary, ct_grid_triangles, ct_2d)
        self.wait()
        self.play(FadeIn(ct_grid_triangles_copy))
        self.wait()
        self.play(
            ReplacementTransform(ct_grid_triangles_copy, center_dots),
            submobject_mode = "lagged_start", run_time = 5,
        )
        self.wait()
        self.add_foreground_mobjects(center_dots)
        center_dots_connections.generate_target()
        for mob in center_dots_connections:
            mob.set_background_stroke(width = 0)
            mob.scale(0)
        self.play(
            MoveToTarget(center_dots_connections),
            submobject_mode = "lagged_start", run_time = 5,
        )
        self.wait()
        self.remove_foreground_mobjects(center_dots)
        self.play(FadeOut(VGroup(center_dots, center_dots_connections)))
        self.wait()
        ct_2d_copy = ct_2d.deepcopy()
        self.play(TilesShrink(ct_2d))
        self.wait()
        self.play(TilesGrow(ct_2d_copy))
        self.wait()


class RandomlyChooseTwoToCompare(AlternateWayToExpressTiling):
    CONFIG = {
        "random_seed" : (5**7)**0,
    }
    def setup(self):
        super().setup()
        self.ct_2d.remove(self.ct_2d.get_all_dumbbells())
        self.add(self.ct_2d)
        self.choosed_patterns = [generate_pattern(5), generate_pattern(5)]

    def construct(self):
        indices = [20, 24]
        small_ct_grid = CalissonTilingGrid(unit_size = 0.3)
        patterns = [generate_pattern(5) for k in range(45)]
        patterns[indices[0]], patterns[indices[1]] = self.choosed_patterns
        patterns[22] = MAA_PATTERN
        tilings = VGroup(*[
            CalissonTiling2D(
                CalissonTiling3D(
                    dimension = 5, pattern = pattern, enable_fill = True,
                    border_config = {"stroke_width" : 3},
                    tile_config = {"stroke_width" : 0.5, "stroke_color" : WHITE},
                ),
                enable_dumbbells = False, ct_grid = small_ct_grid,
            )
            for pattern in patterns
        ])
        tilings.arrange_submobjects_in_grid(5, 9, buff = 0.7)
        tilings.set_height(FRAME_HEIGHT*1.1)
        choosed_tilings = VGroup(tilings[indices[0]], tilings[indices[1]])
        circles = VGroup(*[
            Circle(color = YELLOW, stroke_width = 10).set_height(tiling.get_height()*1.1).move_to(tiling)
            for tiling in choosed_tilings
        ])
        other_tilings = VGroup(*list(set(tilings.submobjects) - set(choosed_tilings.submobjects)))
        # Show a whole bunch of different tilings
        self.ct_2d.target = tilings[22]
        self.play(
            MoveToTarget(self.ct_2d, run_time = 1),
            LaggedStart(FadeIn, tilings, run_time = 3),
        )
        self.remove(self.ct_2d)
        self.wait()
        # and randomly bring 2 to compare
        self.play(
            *[ShowPassingFlash(circle, time_width = 0.4) for circle in circles],
        )
        self.wait()
        for tiling, direction in zip(choosed_tilings, [LEFT, RIGHT]):
            tiling.generate_target()
            tiling.target.set_height(4.8).center().to_edge(direction)
        self.play(
            *[MoveToTarget(tiling) for tiling in choosed_tilings],
            ApplyMethod(other_tilings.fade, 1),
        )
        self.wait()


class CompareTwoDifferentTilings(MovingCameraScene):
    CONFIG = {
        "random_seed" : (5**7)**0,
    }
    def construct(self):
        self.setup_tilings()
        self.show_some_identical_calissons()
        self.add_remark_and_change_representation()
        self.bring_dumbbells_together()
        self.show_the_reason_of_two_edges_only()
        self.but_why_loops()
        self.color_dumbbells_by_their_source()
        self.zoom_in_on_the_difference()
        self.change_to_arrow_representation()
        self.demonstrate_cutting_the_loop()
        self.generalize_to_all_loops_and_grid_lines()

    def setup_tilings(self):
        patterns = [generate_pattern(5), generate_pattern(5)]
        ct_grid = CalissonTilingGrid(
            side_length = 5, unit_size = 2.4/5,
            grid_boundary_config = {"stroke_width" : 3, "color" : WHITE},
            grid_triangles_config = {"opacity" : 0.4},
        )
        ct_grid.add_grid_boundary()
        ct_grid.add_grid_triangles()
        ct_2d_A, ct_2d_B = ct_2ds = VGroup(*[
            CalissonTiling2D(
                CalissonTiling3D(
                    dimension = 5, pattern = pattern, enable_fill = True,
                    border_config = {"stroke_width" : 3},
                    tile_config = {"stroke_width" : 0.5, "stroke_color" : WHITE},
                ),
                ct_grid,
                dumbbell_config = {"point_size" : 0.05, "stem_width" : 0.03}
            )
            for pattern in patterns
        ])
        text_A, text_B = texts = VGroup(*[TextMobject(c, sheen_direction = DOWN) for c in ("A", "B")])
        texts.set_color(L_GRADIENT_COLORS)
        for ct_2d, text, direction in zip(ct_2ds, texts, [LEFT, RIGHT]):
            ct_2d.to_edge(direction)
            text.scale(2)
            text.next_to(ct_2d, UP, buff = 0.5)
        self.add(ct_2ds)
        self.wait()
        self.ct_grid = ct_grid
        self.ct_2d_A, self.ct_2d_B = self.ct_2ds = ct_2ds
        self.text_A, self.text_B = self.texts = texts
        self.ct_diff = CalissonTilingDifference(ct_2d_A, ct_2d_B)

    def show_some_identical_calissons(self):
        connection_config = {"color" : YELLOW, "stroke_width" : 8, "radius" : 0.5}
        same_tiles = self.ct_diff.get_same_tiles()
        line = Line(LEFT, RIGHT, **connection_config)
        text = TextMobject("相同朝向，相同位置", color = YELLOW)
        circle_A = Circle(**connection_config).move_to(same_tiles[0][0])
        circle_B = Circle(**connection_config).move_to(same_tiles[1][0])
        circle_B.add_updater(
            lambda m: m.move_to(circle_A.get_center() + self.ct_2d_B.get_center() - self.ct_2d_A.get_center())
        )
        line.add_updater(lambda l: l.put_start_and_end_on(circle_A.get_right(), circle_B.get_left()))
        text.add_updater(lambda t: t.next_to(line, UP))
        connection_group = VGroup(circle_A, circle_B, line, text)
        # Show the first example
        same_tiles[0][0].save_state()
        same_tiles[1][0].save_state()
        self.play(
            FadeIn(connection_group),
            ApplyMethod(same_tiles[0][0].set_color, YELLOW),
            ApplyMethod(same_tiles[1][0].set_color, YELLOW),
        )
        self.add_foreground_mobjects(connection_group)
        self.wait(2)
        self.play(Restore(same_tiles[0][0]), Restore(same_tiles[1][0]))
        self.wait()
        # Show a few more examples
        num_of_pairs = 5
        indices = random.sample(range(len(same_tiles[0])), num_of_pairs)
        for i in indices:
            tile_A, tile_B = same_tiles[0][i], same_tiles[1][i]
            tile_A.save_state()
            tile_B.save_state()
            self.play(
                ApplyMethod(circle_A.move_to, tile_A.get_center()),
                ApplyMethod(tile_A.set_color, YELLOW),
                ApplyMethod(tile_B.set_color, YELLOW),
            )
            self.wait()
            self.play(Restore(tile_A), Restore(tile_B))
            self.wait()
        self.remove_foreground_mobjects(connection_group)
        self.play(FadeOut(connection_group))
        self.wait()

    def add_remark_and_change_representation(self):
        # Add names for tilings
        self.play(LaggedStart(FadeInFromDown, VGroup(self.text_A, self.text_B)))
        self.wait()
        # Change to 'dumbbell' representation
        self.play(TilesShrink(self.ct_2d_A), TilesShrink(self.ct_2d_B), run_time = 3)
        self.wait()
        # Add grid for comparison
        self.play(FadeInFromDown(self.ct_grid))
        self.wait()

    def bring_dumbbells_together(self):
        # Classify dumbbells and make style match during the animation
        sd = self.ct_diff.get_same_dumbbells()
        sd_copy = sd.deepcopy()
        sd_copy.add_updater(lambda m: m.match_style(sd))
        dd = self.ct_diff.get_different_dumbbells()
        dd_copy = dd.deepcopy()
        dd_copy.add_updater(lambda m: m.match_style(dd))
        shift_length = (self.ct_2d_B.get_center()[0] - self.ct_2d_A.get_center()[0])/2.
        for mob, direction in zip([sd[0], sd[1], dd[0], dd[1]], [RIGHT, LEFT, RIGHT, LEFT]):
            mob.generate_target()
            mob.target.shift(shift_length * direction)
        # Bring the same dumbbells together
        self.add(sd_copy)
        self.play(MoveToTarget(sd[0]), MoveToTarget(sd[1]), run_time = 3)
        self.wait()
        sd.generate_target()
        sd.target.set_color(GREY).fade(0.7)
        self.play(MoveToTarget(sd))
        self.wait()
        # Bring the different dumbbells together
        self.add(dd_copy)
        self.play(MoveToTarget(dd[0]), MoveToTarget(dd[1]), run_time = 3)
        self.wait()
        # Indicate the loops
        loops = self.ct_diff.get_loops_dumbbells()
        loops_colors = color_gradient([YELLOW, CYAN], len(loops))
        loops.generate_target()
        for loop, color in zip(loops.target, loops_colors):
            loop.set_color(color)
        self.text_A.save_state()
        self.text_A.generate_target()
        self.text_A.target.set_color([YELLOW, CYAN])
        self.text_B.save_state()
        self.text_B.generate_target()
        self.text_B.target.set_color([YELLOW, CYAN])
        self.play(*[MoveToTarget(mob) for mob in (loops, self.text_A, self.text_B)], run_time = 2)
        self.wait()
        self.play(
            *[Indicate(loop, color = color, scale_factor = 1.1) for loop, color in zip(loops, loops_colors)],
            submobject_mode = "lagged_start", lag_factor = 2, run_time = 3,
        )
        self.wait()
        self.sd, self.sd_copy = sd, sd_copy
        self.dd, self.dd_copy = dd, dd_copy
        self.loops = loops

    def show_the_reason_of_two_edges_only(self):
        eg_dd_A = self.loops[5][0]
        eg_dd_A_copy = self.dd_copy[0][19]
        eg_dd_B = self.loops[5][1]
        eg_dd_B_copy = self.dd_copy[1][19]
        eg_dd_A.save_state()
        eg_dd_B.save_state()
        self.bring_to_front(eg_dd_A, eg_dd_B)
        circle_A = Circle(color = RED)
        # Show the example dot
        dot = eg_dd_A.get_start_bell().deepcopy()
        dot.set_color(RED).scale(1.2)
        triangle = self.ct_grid.get_grid_triangle_containing_point(dot.get_center())
        self.play(FocusOn(dot), FadeInFromLarge(dot, scale_factor = 10), run_time = 1.5)
        self.wait()
        self.play(Indicate(triangle, color = RED))
        self.wait()
        # Show A's contribution
        radius = get_norm(eg_dd_A.get_start_bell().get_center() - eg_dd_A.get_end_bell().get_center())*0.9
        circle_A = Circle(radius = radius, color = RED, stroke_width = 5)
        circle_A.move_to(eg_dd_A_copy)
        arrow_A = Arrow(
            eg_dd_A_copy.get_center(), eg_dd_A.get_center(), color = RED,
            use_rectangular_stem = False, path_arc = PI/2., buff = 0.25,
        )
        arrow_A.get_tip().shift(0.03*UR)
        self.play(ShowCreation(circle_A))
        self.play(ShowCreation(arrow_A), ApplyMethod(eg_dd_A.set_color, RED))
        self.wait()
        # Show B's contribution
        circle_B = circle_A.deepcopy().move_to(eg_dd_B_copy)
        arrow_B = Arrow(
            eg_dd_B_copy.get_center(), eg_dd_B.get_center(), color = RED,
            use_rectangular_stem = False, path_arc = -PI/2., buff = 0.25,
        )
        arrow_B.get_tip().shift(0.03*UL)
        self.play(ShowCreation(circle_B))
        self.play(ShowCreation(arrow_B), ApplyMethod(eg_dd_B.set_color, RED))
        self.wait()
        # Remove unnecessary stuff
        self.play(
            Restore(eg_dd_A), Restore(eg_dd_B),
            FadeOut(VGroup(dot, circle_A, circle_B, arrow_A, arrow_B))
        )
        self.wait()

    def but_why_loops(self):
        question = TextMobject("所有点都连接两条边$\\Rightarrow$多条回路?", color = RED)
        question.next_to(self.ct_grid, DOWN, buff = 0.3)
        self.play(FadeInFrom(question, UP))
        self.wait()
        self.remove(question)

    def color_dumbbells_by_their_source(self):
        self.play(
            ApplyMethod(self.text_A.set_color, MAGENTA), ApplyMethod(self.dd[0].set_color, MAGENTA),
            ApplyMethod(self.text_B.set_color, CYAN), ApplyMethod(self.dd[1].set_color, CYAN),
        )
        # Tweak the color for bells
        self.dd_copy.clear_updaters()
        loops_bells = self.ct_diff.get_loops_bells()
        loops_bells.generate_target()
        for bell_pairs in loops_bells:
            for bell_pair in bell_pairs:
                for bell in bell_pair:
                    bell.set_color(GRAY)
                    bell.fade(0.2)
        self.play(MoveToTarget(loops_bells, run_time = 3))
        self.wait()

    def zoom_in_on_the_difference(self):
        self.camera.frame.save_state()
        self.b_side_mobs = VGroup(self.sd_copy[1], self.dd_copy[1], self.ct_2d_B.get_border())
        self.play(
            FadeOut(self.b_side_mobs),
            self.camera.frame.set_height, self.ct_grid.get_height() * 1.1,
            self.camera.frame.move_to, VGroup(self.ct_grid, self.ct_2d_B),
            run_time = 3,
        )
        self.wait()
        zoom_text_A, zoom_text_B = zoom_texts = VGroup(*[
            TextMobject(c, color = color)
            for c, color in zip(["A", "B"], [MAGENTA, CYAN])
        ])
        zoom_texts.arrange_submobjects(RIGHT, buff = 1)
        zoom_texts.next_to(self.b_side_mobs.get_top(), DOWN, buff = 0.1)
        reminder = TextMobject("A和B的不同部分", "$\\rightarrow$", "回路")
        reminder.arrange_submobjects(RIGHT, buff = 0.2)
        reminder.scale(0.75)
        reminder[0][0].set_color(MAGENTA), reminder[0][2].set_color(CYAN)
        reminder.next_to(zoom_texts, DOWN, aligned_edge = UP, buff = 0.5)
        self.play(LaggedStart(FadeInFromDown, zoom_texts))
        self.wait()
        self.play(FadeInFromDown(reminder))
        self.wait()
        self.play(FadeOut(reminder), FadeOut(zoom_texts))
        self.wait()

    def change_to_arrow_representation(self):
        # Choose the longest loop for example
        loops_dumbbells = self.ct_diff.get_loops_dumbbells()
        loops_dumbbells.save_state()
        spec_loop_dumbbells = loops_dumbbells[3]
        other_loop_dumbbells = VGroup(*[loops_dumbbells[k] for k in (0, 1, 2, 4, 5)])
        self.play(
            FocusOn(spec_loop_dumbbells), Indicate(spec_loop_dumbbells, scale_factor = 1),
            run_time = 1,
        )
        self.wait()
        # Change representation for the last time
        loops_arrows = self.ct_diff.get_loops_arrows()
        spec_loop_arrows = loops_arrows[3]
        other_loop_arrows = VGroup(*[loops_arrows[k] for k in (0, 1, 2, 4, 5)])
        other_loop_arrows.save_state()
        self.play(
            ReplacementTransform(loops_dumbbells, loops_arrows, submobject_mode = "lagged_start"),
            run_time = 3,
        )
        self.play(other_loop_arrows.fade, 0.7, run_time = 1)
        self.wait()
        # Remind that calisson, dumbbell and arrow are identical
        rm_calisson = HRhombus(side_length = 0.5)
        rm_calisson.set_stroke(width = 2).set_fill(opacity = 1, color = GREEN)
        rm_dumbbell = Dumbbell(0.25*LEFT, 0.25*RIGHT).set_color(GREEN)
        rm_arrow = VGroup(
            Vector(0.5*LEFT).set_color(GREEN),
            TextMobject("或").scale(0.4),
            Vector(0.5*RIGHT).set_color(GREEN),
        ).arrange_submobjects(DOWN, buff = 0.1)
        rm_group = VGroup(rm_calisson, TextMobject("="), rm_dumbbell, TextMobject("="), rm_arrow)
        rm_group.arrange_submobjects(RIGHT, buff = 0.3)
        rm_group.next_to(self.ct_grid, RIGHT, buff = 0.5)
        self.play(LaggedStart(FadeInFromDown, rm_group))
        self.wait()
        self.play(FadeOut(rm_group))
        self.wait()
        # Focus on the loop edges and grid triangles it passes.
        # 'Indicate' just won't do its job on grid_triangles, which is infuriating.
        # So screw it.
        spec_loop_arrows_copy = spec_loop_arrows.deepcopy()
        spec_loop_arrows_copy.submobjects.reverse()
        repeat_times = 3
        for k in range(repeat_times):
            self.play(
                ApplyMethod(spec_loop_arrows_copy.set_color, YELLOW),
                rate_func = there_and_back, submobject_mode = "lagged_start", run_time = 2,
            )
            self.wait()
        self.remove(spec_loop_arrows_copy)
        # And find the hidden connection
        td = self.ct_grid.get_grid_triangles()[1].deepcopy()
        tl = self.ct_grid.get_grid_triangles()[0].deepcopy()
        aa = Vector(0.3*RIGHT).set_color(MAGENTA)
        ab = Vector(0.3*RIGHT).set_color(CYAN)
        ta = TextMobject("A").scale(0.5).set_color(MAGENTA)
        tb = TextMobject("B").scale(0.5).set_color(CYAN)
        for triangle in (td, tl):
            triangle.scale(0.7)
            triangle.set_fill(opacity = 1)
        relation = VGroup(*[
            mob.deepcopy()
            for mob in [td, aa, tl, ab, td, aa, tl, ab, td, aa, TexMobject("\\cdots").scale(0.5)]
        ])
        relation.arrange_submobjects(RIGHT, buff = 0.1).move_to(self.b_side_mobs.get_center())
        for arrow, text in zip(relation[1:-1:2], [ta, tb, ta, tb, ta]):
            mob = text.deepcopy()
            mob.next_to(arrow, UP, buff = 0.1)
            arrow.add(mob)
        self.play(FadeInFrom(relation, LEFT))
        self.wait()
        sur_rect_config = {"stroke_width" : 3, "buff" : 0.05}
        sur_rects_A = VGroup(*[SurroundingRectangle(mob, **sur_rect_config) for mob in (relation[0:3], relation[4:7])])
        sur_rects_B = VGroup(*[SurroundingRectangle(mob, **sur_rect_config) for mob in (relation[2:5], relation[6:9])])
        self.play(ShowCreation(sur_rects_A), submobject_mode = "all_at_once", run_time = 2)
        self.wait()
        self.play(ReplacementTransform(sur_rects_A, sur_rects_B), run_time = 2)
        self.wait()
        self.play(FadeOutAndShift(relation, LEFT), FadeOutAndShift(sur_rects_B, LEFT))
        self.wait()
        self.relation = relation
        self.loops_dumbbells = loops_dumbbells
        self.loops_arrows = loops_arrows
        self.spec_loop_arrows = spec_loop_arrows
        self.other_loop_arrows = other_loop_arrows

    def demonstrate_cutting_the_loop(self):
        line_length = self.ct_grid.get_height()*1.2
        vert_grid_lines = self.ct_grid.get_grid_lines()[12:21]
        for line in vert_grid_lines:
            line.set_color(YELLOW).set_stroke(width = 3)
            line.scale(line_length/line.get_length())
        eg_grid_line = vert_grid_lines[7]
        eg_grid_line.set_color(YELLOW)
        # Show grid line
        self.play(FocusOn(eg_grid_line), ShowCreation(eg_grid_line), run_time = 1)
        self.wait()
        # Show arrows that cross the grid line
        crossing_arrows = self.ct_diff.get_loop_arrows_crossing_vertical_line(self.spec_loop_arrows, eg_grid_line).deepcopy()
        crossing_arrows.save_state()
        circles = VGroup(*[
            Circle(radius = arrow.get_width()*0.6, color = YELLOW).move_to(arrow)
            for arrow in crossing_arrows
        ])
        self.play(
            Indicate(crossing_arrows, scale_factor = 1), ShowCreationThenDestruction(circles),
            submobject_mode = "lagged_start", run_time = 5,
        )
        # Move stuff to the space on the right for a better view
        grid_triangles = VGroup(*[
            VGroup(
                self.ct_grid.get_grid_triangle_containing_point(arrow.get_left()),
                self.ct_grid.get_grid_triangle_containing_point(arrow.get_right()),
            )
            for arrow in crossing_arrows
        ])
        moving_group = VGroup(grid_triangles, eg_grid_line, crossing_arrows).deepcopy()
        moving_group.generate_target()
        moving_group.target.move_to(self.b_side_mobs.get_center()).shift(0.4*DOWN)
        moving_group.target[::2].scale(1.2)
        moving_group.target[0].set_fill(opacity = 0.8)
        moving_group.target[1].scale(0.7)
        self.play(MoveToTarget(moving_group))
        self.wait(3)
        # Two types of crossing the line
        grid_triangles_copy, eg_grid_line_copy, crossing_arrows_copy = moving_group
        circles = VGroup(*[
            Circle(radius = arrow.get_width()*0.6, color = YELLOW).move_to(arrow)
            for arrow in crossing_arrows_copy
        ])
        to_left_arrows = crossing_arrows_copy[::2]
        to_left_circles = circles[::2].set_color(CYAN)
        to_right_arrows = crossing_arrows_copy[1::2]
        to_right_circles = circles[1::2].set_color(MAGENTA)
        self.play(ShowCreation(to_right_circles))
        self.wait()
        self.play(ShowCreation(to_left_circles))
        self.wait()
        moving_group.save_state()
        to_left_text = TextMobject("从右到左\\\\2次").set_color(CYAN).scale(0.65)
        to_left_text.next_to(eg_grid_line_copy.get_end(), LEFT)
        to_right_text = TextMobject("从左到右\\\\2次").set_color(MAGENTA).scale(0.65)
        to_right_text.next_to(eg_grid_line_copy.get_end(), RIGHT)
        self.play(
            FadeInFrom(to_left_text, RIGHT), FadeInFrom(to_right_text, LEFT),
            ApplyMethod(VGroup(to_left_arrows, to_left_circles).shift, LEFT),
            ApplyMethod(VGroup(to_right_arrows, to_right_circles).shift, RIGHT),
            run_time = 1,
        )
        self.wait()
        self.play(
            FadeOutAndShift(to_left_circles, RIGHT), FadeOutAndShift(to_right_circles, LEFT),
            Restore(moving_group)
        )
        self.wait()
        # Connection between direction and color-changing
        to_left_color_arrow = Vector(LEFT, color = CB_DARK)
        to_right_color_arrow = Vector(RIGHT, color = CB_LIGHT)
        to_left_color_arrow.next_to(to_right_arrows[0].get_left(), DOWN, buff = 0.4, aligned_edge = RIGHT)
        to_right_color_arrow.next_to(to_right_arrows[0].get_right(), DOWN, buff = 0.4, aligned_edge = LEFT)
        self.play(GrowArrow(to_left_color_arrow), GrowArrow(to_right_color_arrow))
        self.wait()
        to_left_color_text = TextMobject("从浅到深", color = CB_DARK).scale(0.65)
        to_right_color_text = TextMobject("从深到浅", color = CB_LIGHT).scale(0.65)
        to_left_color_text.next_to(to_left_text, UP, buff = 0.1)
        to_right_color_text.next_to(to_right_text, UP, buff = 0.1)
        self.play(ReplacementTransform(to_right_color_arrow, to_right_color_text))
        self.wait()
        self.play(ReplacementTransform(to_left_color_arrow, to_left_color_text))
        self.wait()
        # Connection between color-changing and dumbbell's source
        to_left_source_text = TextMobject("B", color = CYAN).scale(0.8)
        to_right_source_text = TextMobject("A", color = MAGENTA).scale(0.8)
        to_left_source_text.move_to(to_left_color_text)
        to_right_source_text.move_to(to_right_color_text)
        self.play(ReplacementTransform(to_right_color_text, to_right_source_text))
        self.wait()
        self.play(ReplacementTransform(to_left_color_text, to_left_source_text))
        self.wait()
        left_group = VGroup(to_left_source_text, to_left_text)
        left_sur_rect = SurroundingRectangle(left_group)
        right_group = VGroup(to_right_source_text, to_right_text)
        right_sur_rect = SurroundingRectangle(right_group)
        upper_group = VGroup(left_group, right_group)
        self.play(
            ShowCreationThenDestruction(left_sur_rect),
            ShowCreationThenDestruction(right_sur_rect)
        )
        self.wait()
        # Add a few reminders for better illustration
        reminder_1 = TextMobject("1. 从左到右的箭头数", "=", "从右到左的箭头数")
        reminder_2 = TextMobject("2. 从左到右", "=", "从深到浅", "=", "来自A")
        reminder_3 = TextMobject("3. 从右到左", "=", "从浅到深", "=", "来自B")
        reminders = VGroup(reminder_1, reminder_2, reminder_3).scale(0.5)
        reminders.arrange_submobjects(DOWN, aligned_edge = LEFT)
        reminders.next_to(self.ct_grid, RIGHT).shift(UP)
        reminder_1_arrow = Vector(0.5*DOWN).next_to(reminder_1[1], UP, buff = 0.1)
        reminder_1_remark = TextMobject("回路的性质", color = YELLOW)
        reminder_1_remark.scale(0.4).next_to(reminder_1_arrow, UP, buff = 0.1)
        reminder_2_arrow = Vector(0.3*UP).next_to(reminder_3[1], DOWN, buff = 0.1)
        reminder_2_remark = TextMobject("正三角形\\\\网格的性质", color = YELLOW)
        reminder_2_remark.scale(0.4).next_to(reminder_2_arrow, DOWN, buff = 0.1)
        reminder_3_arrow = Vector(1.0*UP).next_to(reminder_3[3], DOWN, buff = 0.1)
        reminder_3_remark = TextMobject("回路+网格的性质", color = YELLOW)
        reminder_3_remark.scale(0.4).next_to(reminder_3_arrow, DOWN, buff = 0.1)
        reminder_3_figure = self.relation.deepcopy()
        reminder_3_figure.move_to(self.b_side_mobs).next_to(reminder_3_remark, DOWN, coor_mask = [0, 1, 0])
        reminders.add(
            reminder_1_arrow, reminder_2_arrow, reminder_3_arrow,
            reminder_1_remark, reminder_2_remark, reminder_3_remark, reminder_3_figure,
        )
        self.play(FadeOut(moving_group), FadeOut(upper_group), FadeInFromDown(reminders))
        self.wait()
        self.play(FadeOut(eg_grid_line), FadeOut(reminders), run_time = 1)
        self.wait()
        self.vert_grid_lines = vert_grid_lines
        self.source_text_B = self.to_left_source_text = to_left_source_text
        self.source_text_A = self.to_right_source_text = to_right_source_text

    def generalize_to_all_loops_and_grid_lines(self):
        self.play(Restore(self.other_loop_arrows), run_time = 1)
        self.wait()
        self.play(LaggedStart(ShowCreationThenDestruction, self.vert_grid_lines.deepcopy()), run_time = 2)
        self.wait()
        # Gather different arrows
        arrows_A = VGroup()
        arrows_B = VGroup()
        source_texts = [self.source_text_A, self.source_text_B]
        for loop_set in (self.spec_loop_arrows, self.other_loop_arrows):
            for loop_arrows in loop_set:
                for line in self.vert_grid_lines:
                    valid_arrows = self.ct_diff.get_loop_arrows_crossing_vertical_line(loop_arrows, line)
                    for arrow in valid_arrows:
                        color = arrow[1].get_color()
                        if color == Color(MAGENTA):
                            arrows_A.add(arrow.deepcopy())
                        elif color == Color(CYAN):
                            arrows_B.add(arrow.deepcopy())
        for arrows, source_text in zip([arrows_A, arrows_B], source_texts):
            arrows.generate_target()
            arrows.target.arrange_submobjects(DOWN, buff = 0.12)
            arrows.target.next_to(source_text, DOWN)
        self.play(
            *[Write(text, run_time = 1) for text in (self.source_text_A, self.source_text_B)],
            *[MoveToTarget(arrows, run_time = 3) for arrows in (arrows_A, arrows_B)],
        )
        self.wait()
        # Gather same dumbbells
        dumbbells_A = VGroup()
        dumbbells_B = VGroup()
        for line in self.vert_grid_lines:
            valid_dumbbells = self.ct_diff.get_loop_arrows_crossing_vertical_line(self.sd[0], line)
            for dumbbell in valid_dumbbells:
                dumbbells_A.add(dumbbell)
                dumbbells_B.add(dumbbell.deepcopy())
        for dumbbells, arrows, source_text in zip([dumbbells_A, dumbbells_B], [arrows_A, arrows_B], source_texts):
            dumbbells.generate_target()
            arrows.generate_target()
            dumbbells.target.arrange_submobjects(DOWN, buff = 0.12)
            dumbbells.target.set_fill(opacity = 0.8)
            group_target = VGroup(dumbbells.target, arrows.target)
            group_target.arrange_submobjects(RIGHT, aligned_edge = UP)
            group_target.next_to(source_text, DOWN)
        self.play(
            *[MoveToTarget(mobs, run_time = 3) for mobs in (arrows_A, arrows_B, dumbbells_A, dumbbells_B)],
        )
        self.wait()
        # Finally, change back to calisson representation
        calissons_A = VGroup(*[
            HRhombus(side_length = 0.2, rhombus_config = {"stroke_width" : 1}).move_to(mob.get_center())
            for mob in (dumbbells_A.submobjects + arrows_A.submobjects)
        ])
        calissons_A.set_fill(opacity = 1)
        calissons_A.next_to(self.source_text_A, DOWN)
        calissons_B = calissons_A.deepcopy()
        calissons_B.next_to(self.source_text_B, DOWN)
        self.play(
            *[ShrinkToCenter(db) for dbs in (dumbbells_A, dumbbells_B) for db in dbs ],
            *[ShrinkToCenter(ar) for ars in (arrows_A, arrows_B) for ar in ars],
            *[GrowFromCenter(clssn) for clssns in (calissons_A, calissons_B) for clssn in clssns],
            submobject_mode = "lagged_start",
            run_time = 3,
        )
        calissons_A_sur_rect = SurroundingRectangle(calissons_A)
        calissons_B_sur_rect = SurroundingRectangle(calissons_B)
        same_number_text = TextMobject("数目相同", color = YELLOW).scale(0.8)
        same_number_text.next_to(VGroup(calissons_A_sur_rect, calissons_B_sur_rect), DOWN)
        self.play(ShowCreation(calissons_A_sur_rect), ShowCreation(calissons_B_sur_rect), run_time = 1)
        self.play(Write(same_number_text), run_time = 2)
        self.wait()


class Why2RegularMeansAllLoops(Scene):
    CONFIG = {
        "dot_config" : {"color" : WHITE, "radius" : 0.1},
        "line_config" : {"color": BLUE,"stroke_width" : 4},
        "circle_config" : {"radius" : 0.2, "stroke_width" : 5, "color" : RED,},
        "random_seed" : 5-7*0,
    }
    def setup(self):
        self.connections = {}

    def construct(self):
        # Setup title
        title = TextMobject("所有点都连接两条边", "$\\Rightarrow$", "多条回路")
        title.arrange_submobjects(RIGHT, buff = 0.5)
        title.scale(1.2).set_color(YELLOW).to_edge(UP)
        title[1].scale(1.5)
        self.add(title[0])
        # Setup dots
        num_of_dots = 18
        dots_centers = [self.get_random_point() for k in range(num_of_dots)]
        dots_centers.sort(key = lambda p: p[0])
        dots_centers[4] += DOWN
        dots = VGroup(*[Dot(center, **self.dot_config) for center in dots_centers])
        self.add_foreground_mobjects(dots)
        self.dots = dots
        # Show the first example
        circle_5, circle_4, circle_0 = [self.get_circle_around(n) for n in (5, 4, 0)]
        con_5_0, con_5_4 = self.get_connection(5, 0), self.get_connection(5, 4)
        self.play(FocusOn(circle_5), ShowCreation(circle_5), run_time = 1)
        self.wait()
        self.play(
            ShowCreation(con_5_0), ShowCreation(con_5_4),
            ReplacementTransform(circle_5, circle_4), ReplacementTransform(circle_5.deepcopy(), circle_0)
        )
        self.wait()
        # Either they connect together...
        circle_4.save_state(), circle_0.save_state()
        con_4_0 = self.get_connection(4, 0)
        self.play(
            ApplyMethod(circle_4.fade, 1), ApplyMethod(circle_0.fade, 1),
            ShowCreation(con_4_0),
        )
        self.wait()
        # or they connect to other points and then converge
        self.play(Restore(circle_4), Restore(circle_0), Uncreate(con_4_0))
        self.wait()
        con_0_2, con_2_1, con_4_3, con_3_1 = [
            self.get_connection(i, j)
            for (i, j) in [(0, 2), (2, 1), (4, 3), (3, 1)]
        ]
        circle_2, circle_3, circle_1 = [
            self.get_circle_around(i) for i in (2, 3, 1)
        ]
        self.play(
            ShowCreation(con_0_2), ShowCreation(con_4_3),
            ReplacementTransform(circle_0, circle_2), ReplacementTransform(circle_4, circle_3)
        )
        self.wait()
        circle_2.generate_target(), circle_3.generate_target()
        circle_2.target.move_to(circle_1).fade(1)
        circle_3.target.move_to(circle_1).fade(1)
        self.play(
            ShowCreation(con_2_1), ShowCreation(con_3_1),
            MoveToTarget(circle_2), MoveToTarget(circle_3)
        )
        self.wait()
        # Fade out a closed loop
        first_closed_loop = VGroup(con_5_0, con_0_2, con_2_1, con_5_4, con_4_3, con_3_1, self.dots[:6])
        self.play(ApplyMethod(first_closed_loop.set_color, GRAY))
        self.wait()
        # Do the same for the remaining points
        loop_indices_2 = [11, 13, 17, 16, 11]
        loop_indices_3 = [7, 8, 6, 9, 12, 14, 15, 10, 7]
        other_closed_loops = VGroup()
        for loop_indices in (loop_indices_2, loop_indices_3):
            cl = VGroup()
            for i in range(len(loop_indices)-1):
                cl.add(self.get_connection(loop_indices[i], loop_indices[i+1]))
            other_closed_loops.add(cl)
        self.play(
            ShowCreation(other_closed_loops[0], run_time = 1.5),
            ShowCreation(other_closed_loops[1], run_time = 3),
        )
        self.wait()
        other_closed_loops.add(self.dots[6:])
        self.play(ApplyMethod(other_closed_loops.set_color, GRAY))
        self.remove_foreground_mobjects(self.dots)
        self.wait()
        # Flash a formal version of this idea
        bg_rect = FullScreenRectangle(stroke_width = 0, fill_color = BLACK, fill_opacity = 1)
        formal_proof = ImageWithRemark(
            "two_regular_means_cycles.png",
            """
            MIT OpenCourseWare \\\\
            \\emph{Combinatorics: The Fine Art of Counting} \\\\
            Week 9 Lecture Notes – Graph Theory
            """,
            image_width = 10, text_width = 5, text_position = UP,
        )
        self.add(bg_rect, formal_proof)
        self.wait()
        self.remove(bg_rect, formal_proof)
        self.wait()
        self.play(FadeInFromDown(title[1:]))
        self.wait(3)

    def get_random_point(self):
        x = random.uniform(-5, 5)
        y = random.uniform(-3, 2)
        return x*RIGHT + y*UP

    def get_connection(self, i, j):
        if (i, j) in self.connections.keys():
            return self.connections[(i, j)]
        else:
            line = Line(self.dots[i].get_center(), self.dots[j].get_center(), **self.line_config)
            self.connections[(i, j)] = line
            return line

    def get_circle_around(self, i):
        return Circle(**self.circle_config).move_to(self.dots[i].get_center())


class NumberOfEachCalissonWontChange(CompareTwoDifferentTilings):
    def construct(self):
        super().setup_tilings()
        ct_counter = TilesCounter(["?"]*3, tile_stroke_width = 1, height = 3)
        ct_counter.move_to(ORIGIN)
        arrow_l = Arrow(ct_counter.get_left(), self.ct_2d_A.get_right(), color = WHITE)
        arrow_r = Arrow(ct_counter.get_right(), self.ct_2d_B.get_left(), color = WHITE)
        counter_group = VGroup(ct_counter, arrow_l, arrow_r)
        self.play(FadeInFromDown(counter_group))
        self.wait()
        nums_sur_rect = SurroundingRectangle(ct_counter.get_nums())
        nums_text = TextMobject("不会改变", color = YELLOW)
        nums_text.next_to(nums_sur_rect, UP)
        self.play(ShowCreation(nums_sur_rect))
        self.play(Write(nums_text))
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
                    pattern = get_lower_bounding_array(generate_pattern(9), BOUNDING_VALUES),
                    tile_config = {"stroke_width" : 1}, enable_fill = True,
                ),
                ct_grid,
                enable_dumbbells = False,
            )
            for k in range(6)
        ]
        for tiling in tilings:
            tiling.remove_border()
            for direction, (x, y) in OBSOLETE_TILES_INFO:
                tiling.remove_tile(direction, [x, y])
            tiling.shuffle_tiles()
        new_border = VMobject(
            fill_color = YELLOW, fill_opacity = 0.2,
            stroke_width = 5, stroke_color = YELLOW
        )
        new_border_anchor_points = [ct_grid.coords_to_point(x, y) for x, y in BORDER_MAIN_PATH_ACS]
        new_border.set_anchor_points(new_border_anchor_points, mode = "corners")
        subpath_control_points = [ct_grid.coords_to_point(z, w) for z, w in BORDER_SUB_PATH_CTRLS]
        new_border.add_subpath(subpath_control_points)
        init_tiling = tilings[0]
        new_tilings = VGroup(*tilings[1:])
        ct_counter = TilesCounter([57, 70, 50], tile_stroke_width = 1, height = 3)
        ct_counter.shift(3*RIGHT)
        self.play(DrawBorderThenFill(new_border), run_time = 2)
        self.bring_to_back(new_border)
        self.wait()
        self.play(TilesGrow(init_tiling), run_time = 4)
        self.wait()
        new_tilings.shift(2*LEFT)
        self.play(
            ApplyMethod(
                VGroup(new_border, init_tiling).shift, 2*LEFT,
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
        arrow.move_to((jp_text.get_center() + np_text.get_center()) / 2)
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


class FinalRotationTrick(Scene):
    CONFIG = {
        "tiling_buff" : 0.3,
        "tiling_horiz_bias" : 5,
        "counter_config" :{
            "matching_direction" : DOWN,
            "matching_buff" : 0.4,
            "tile_fill_opacity" : 0,
            "height" : 3,
        }
    }
    def construct(self):
        self.show_initial_tiling()
        self.rotate_initial_tiling()
        self.two_ways_to_count()
        self.merge_tilings()
        self.rearrange_and_qed()

    def show_initial_tiling(self):
        ct_grid = CalissonTilingGrid(unit_size = 0.4)
        ct_3d = CalissonTiling3D(
            dimension = 5, pattern = MAA_PATTERN,
            tile_colors = [BLACK] * 3,
            tile_config = {"stroke_width" : 1, "fill_opacity" : 0},
        )
        init_tiling = CalissonTiling2D(ct_3d, ct_grid, enable_dumbbells = False)
        init_tiling.to_edge(UP, buff = self.tiling_buff)
        init_tiling.shift(self.tiling_horiz_bias * LEFT)
        self.play(ShowCreation(init_tiling, submobject_mode = "lagged_start"))
        self.wait()
        unknown_counter, init_counter = [
            TilesCounter(array, matching_tiling = init_tiling, **self.counter_config)
            for array in (["?", "?", "?"], ["a", "b", "c"])
        ]
        self.play(FadeIn(unknown_counter))
        self.wait()
        self.play(
            ReplacementTransform(unknown_counter, init_counter, submobject_mode = "lagged_start"),
            run_time = 2
        )
        self.wait()
        self.init_tiling = init_tiling
        self.init_counter = init_counter

    def rotate_initial_tiling(self):
        init_tiling = self.init_tiling
        init_counter = self.init_counter
        fin_tiling = self.init_tiling.deepcopy()
        fin_counter = self.init_counter.deepcopy()
        fin_tiling.shift(2 * self.tiling_horiz_bias * RIGHT)
        fin_tiling.rotate(PI/3.)
        fin_counter.next_to(fin_tiling, DOWN, buff = self.get_counter_buff())
        arrow = Arrow(init_tiling.get_right(), fin_tiling.get_left())
        arrow.move_to(self.get_center_of_mobs(init_tiling, fin_tiling))
        text = TextMobject("逆时针 \\\\ 旋转$60^{\\circ}$")
        text.next_to(arrow, UP, buff = 0.2)
        rotate_group = VGroup(arrow, text)
        med_tiling = init_tiling.deepcopy()
        med_tiling.move_to(self.get_center_of_mobs(init_tiling, fin_tiling))
        self.play(ReplacementTransform(init_tiling.deepcopy(), med_tiling))
        self.wait()
        self.play(
            ReplacementTransform(med_tiling, fin_tiling, path_arc = PI/3.), 
            GrowArrow(arrow),
            SpinInFromNothing(text, path_arc = PI/3.),
            run_time = 2,
        )
        self.wait()
        self.rotate_group = rotate_group
        self.fin_tiling = fin_tiling
        self.fin_counter = fin_counter

    def two_ways_to_count(self):
        # 1. Forget the past, use the conclusion we just drew.
        new_arrow, new_text = new_rotate_group = self.rotate_group.deepcopy()
        self.play(self.rotate_group.fade, 0.8)
        self.wait()
        self.bring_to_back(self.rotate_group)
        self.bring_to_front(self.init_tiling, self.fin_tiling)
        self.play(
            Indicate(self.fin_tiling.get_border(), scale_factor = 1.1),
            Indicate(self.init_tiling.get_border(), scale_factor = 1.1),
        )
        self.wait()
        self.play(ReplacementTransform(self.init_counter.deepcopy(), self.fin_counter))
        self.wait()
        # 2. Take advantage of the rotation trick
        med_tiling = self.init_tiling.deepcopy()
        med_tiling.next_to(self.fin_tiling, LEFT)
        med_counter = self.init_counter.deepcopy()
        med_counter.next_to(med_tiling, DOWN, buff = self.get_counter_buff())

        def arrow_update(arrow):
            arrow.put_start_and_end_on(
                self.init_tiling.get_right() + MED_SMALL_BUFF * RIGHT,
                med_tiling.get_left() + MED_SMALL_BUFF * LEFT
            )
        def text_update(text):
            text.next_to(new_arrow, UP)
        arrow_update(new_arrow)
        text_update(new_text)
        new_rotate_group.save_state()
        new_rotate_group.fade(0.8)
        self.play(
            ReplacementTransform(self.init_tiling.deepcopy(), med_tiling),
            ReplacementTransform(self.init_counter.deepcopy(), med_counter),
            ReplacementTransform(self.rotate_group, new_rotate_group),
            run_time = 3,
        )
        self.wait()

        new_arrow.add_updater(arrow_update)
        new_text.add_updater(text_update)
        rotated_med_counter = TilesCounter(
            ["c", "a", "b"],
            tile_colors = [TILE_BLUE, TILE_RED, TILE_GREEN],
            matching_tiling = med_tiling, **self.counter_config
        )
        self.play(
            Rotate(med_tiling, PI/3., rate_func = smooth),
            AnimationGroup(*[
                Rotate(old_tile, PI/3., rate_func = smooth)
                for old_tile in med_counter.get_tiles()
            ]),
            ApplyMethod(new_rotate_group.restore),
            run_time = 3,
        )
        self.wait()
        pair_indices = ([0, 1], [1, 2], [2, 0])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    ApplyMethod(med_counter.get_tile(i).move_to, rotated_med_counter.get_tile(j)),
                    ApplyMethod(med_counter.get_mult_sign(i).move_to, rotated_med_counter.get_mult_sign(j)),
                    ApplyMethod(med_counter.get_num(i).move_to, rotated_med_counter.get_num(j)),
                )
                for i, j in pair_indices
            ]),
            run_time = 2
        )
        self.wait()
        self.new_rotate_group = new_rotate_group
        self.med_tiling = med_tiling
        self.med_counter = med_counter
        self.rotated_med_counter = rotated_med_counter
    
    def merge_tilings(self):
        tiling_mid_point = self.get_center_of_mobs(self.med_tiling, self.fin_tiling)
        counter_mid_point = self.get_center_of_mobs(self.med_counter, self.fin_counter)
        self.play(
            ApplyMethod(self.med_tiling.move_to, tiling_mid_point),
            ApplyMethod(self.fin_tiling.move_to, tiling_mid_point),
            ApplyMethod(self.med_counter.next_to, counter_mid_point, LEFT),
            ApplyMethod(self.fin_counter.next_to, counter_mid_point, RIGHT),
            run_time = 2,
        )
        self.wait()
        # <Sleight of hands>
        self.rotated_med_counter.move_to(self.med_counter)
        self.remove(self.med_counter)
        self.add(self.rotated_med_counter)
        # </Sleight of hands>

    def rearrange_and_qed(self):
        equations = VGroup(*[
            TexMobject(
                text,
                tex_to_color_map = dict(zip(["a", "b", "c"], [TILE_RED, TILE_GREEN, TILE_BLUE]))
            )
            for text in ["c = a", "a = b", "b = c"]
        ])
        equations.arrange_submobjects(DOWN, buff = 0.35)
        equations.scale(1.5)
        equations.next_to(self.fin_tiling, DOWN, buff = 0.7)
        for k in range(3):
            med_tile, med_mult_sign, med_num = self.rotated_med_counter.get_pair_elements(k)
            fin_tile, fin_mult_sign, fin_num = self.fin_counter.get_pair_elements(k)
            lhs, equal_sign, rhs = equations[k]
            sur_rect = SurroundingRectangle(VGroup(med_tile, fin_num))
            self.play(ShowCreationThenDestruction(sur_rect))
            self.wait()
            self.play(
                FadeOut(VGroup(med_tile, med_mult_sign, fin_tile, fin_mult_sign)),
                Write(equal_sign),
                ReplacementTransform(med_num, lhs),
                ReplacementTransform(fin_num, rhs),
            )
            self.wait()
        for mob in (self.init_tiling, self.init_counter, equations):
            mob.generate_target()
        equations.target.scale(1.2)
        eq1_shift_vec = 0.3 * UL
        eq2_shift_vec = equations.target[1][-1].get_center() - equations.target[2][0].get_center()
        equations.target[0].fade(1)
        equations.target[1].shift(eq1_shift_vec)
        equations.target[2].shift(eq1_shift_vec + eq2_shift_vec)
        self.play(
            FadeOut(self.fin_tiling),
            FadeOut(self.med_tiling),
            FadeOut(self.new_rotate_group),
            MoveToTarget(equations),
        )
        self.wait()
        self.init_counter.target.next_to(self.init_tiling, RIGHT, buff = 0.3)
        init_target_group = VGroup(self.init_tiling.target, self.init_counter.target)
        init_target_group.move_to(2*LEFT)
        self.play(
            MoveToTarget(self.init_tiling),
            MoveToTarget(self.init_counter),
        )
        self.wait()
        qed_symbol = QEDSymbol()
        qed_symbol.to_corner(DR)
        self.play(Write(qed_symbol), run_time = 2)
        self.wait()

    def get_counter_buff(self):
        return self.counter_config["matching_buff"]

    def get_center_of_mobs(self, *mobs):
        return np.mean([mob.get_center() for mob in mobs], axis = 0)


class ABriefSummary(Scene):
    def construct(self):
        frame = PictureInPictureFrame(height = 5)
        title = TextMobject("小结").scale(1.5)
        group = VGroup(title, frame)
        group.arrange_submobjects(DOWN, buff = 0.4).to_edge(UP)
        self.add(group)
        self.wait()


class DijkstrasOriginalNote(Scene):
    def construct(self):
        self.demonstrate_using_animations()
        self.show_original_notes()
        self.show_epilogue()

    def demonstrate_using_animations(self):
        ctt_anim = CalissonTiling2D(
            CalissonTiling3D(
                dimension = 5, pattern = generate_pattern(5), enable_fill = True,
                tile_config = {"stroke_width" : 3}
            ),
            CalissonTilingGrid(unit_size = 0.5), enable_dumbbells = False,
        )
        ctt_anim.move_to((LEFT_SIDE + ORIGIN) / 2.)
        note_svg = SVGMobject("Note.svg")
        note_svg.move_to((RIGHT_SIDE + ORIGIN) / 2.)
        note_svg.set_height(ctt_anim.get_height() * 0.8)
        note_author = TextMobject("Edsger W. Dijkstra", background_stroke_width = 0)
        note_author.set_width(1.5)
        note_author.next_to(note_svg[2], UP, aligned_edge = RIGHT, buff = 0.3)
        arrow = Arrow(ctt_anim.get_right(), note_svg.get_left(), color = WHITE)
        note_group = VGroup(note_svg, note_author)
        question = TexMobject("?")
        question.scale(2)
        question.next_to(arrow, UP)
        ctt_anim_group = VGroup(ctt_anim, note_group, arrow, question)
        self.add(ctt_anim_group)
        self.wait()
        self.ctt_anim_group = ctt_anim_group
        self.ewd_text = VGroup(*[note_author[k] for k in (0, 6, 8)])

    def show_original_notes(self):
        upper_notes = Group(*[
            ImageMobject("%s_EWD1055_%s.png" % (str(k), str(k)))
            for k in range(5)
        ])
        lower_notes = Group(*[
            ImageMobject("%s_EWD1055c_%s.png" % (str(k+5), str(k)))
            for k in range(5)
        ])
        for notes in (upper_notes, lower_notes):
            notes.arrange_submobjects(RIGHT)
            notes.set_height(2.5)
        upper_title = TextMobject("EWD1055")
        upper_group = Group(upper_title, upper_notes)
        lower_title = TextMobject("EWD1055c")
        lower_group = Group(lower_title, lower_notes)
        for group in (upper_group, lower_group):
            group.arrange_submobjects(DOWN, aligned_edge = LEFT)
        sep_line = DashedLine(upper_group.get_left(), upper_group.get_right())
        ewd1055_group = Group(upper_group, sep_line, lower_group)
        ewd1055_group.arrange_submobjects(DOWN)
        ewd1055_group.to_edge(UP, buff = 0.25)
        self.play(
            FadeOut(self.ctt_anim_group),
            ReplacementTransform(self.ewd_text.deepcopy(), upper_title[:3]),
            Write(upper_title[3:]),
            ReplacementTransform(self.ewd_text.deepcopy(), lower_title[:3]),
            Write(lower_title[3:]),
            ShowCreation(sep_line),
            run_time = 2,
        )
        self.play(
            FadeInFromDown(upper_notes), FadeInFromDown(lower_notes),
            submobject_mode = "lagged_start", run_time = 3,
        )
        self.wait()
        self.upper_notes = upper_notes
        self.fade_group = Group(upper_title, lower_title, sep_line, upper_notes[:-2], lower_notes)

    def show_epilogue(self):
        page_3, page_4 = self.upper_notes[-2], self.upper_notes[-1]
        page_group = Group(page_3, page_4)
        page_group.generate_target()
        page_group.target.arrange_submobjects(DOWN, buff = 0.1)
        page_group.target.scale(1.5)
        page_group.target.to_edge(RIGHT, buff = 1)
        self.play(FadeOut(self.fade_group), MoveToTarget(page_group))
        self.wait()
        page_rect = Rectangle(height = 3.7, width = 3, stroke_width = 5, color = RED)
        page_rect.move_to(page_group)
        page_rect.shift(page_3.get_height() * 0.05 * DOWN)
        self.play(ShowCreation(page_rect))
        self.wait()
        text_rect = page_rect.deepcopy()
        text_rect.generate_target()
        text_rect.target.set_height(6, stretch = True)
        text_rect.target.set_width(8, stretch = True)
        text_rect.target.to_edge(LEFT, buff = 1).shift(0.5*UP)
        self.play(MoveToTarget(text_rect))
        text = TextMobject(
            "David和Tomei处理这个问题的方式不尽人意。原话是这样的： \\\\",
            """“证明的思路是把问题化成三维空间中的直观结论。我们不给出证明的 \\\\
            全部细节，原因有两点，一是我们不想破坏基础直观思想的简洁，二是 \\\\
            这个借助图片证明的结论并不能立刻用精确的数学语言描述......”\\\\""",
            "接着，他们就给出了一个复杂的伪证。\\\\",
            """把平面图形理解成简单的三维结构，这样的思路是有误导性的。举个 \\\\
            例子，我们把结论从正六边形推广到下面的环状图形：\\\\""",
            "埃舍尔会很高兴的。",
            alignment = "",
        ).arrange_submobjects(DOWN, aligned_edge = LEFT, buff = 0.5).set_width(7.5)
        text.next_to(text_rect.get_top(), DOWN, buff = 0.4)
        text[-1].shift(2*DOWN)
        rings = VGroup(CalissonRing(2, 4).flip().rotate(PI/6.), CalissonRing(2, 4).rotate(-PI/6.))
        rings.arrange_submobjects(RIGHT, buff = 1.5).set_height(1.6)
        rings.set_fill(opacity = 0).set_stroke(width = 2)
        rings.move_to(text_rect)
        rings.next_to(text[-1], UP, coor_mask = [0, 1, 0])
        text_group = VGroup(text, rings)
        self.play(FadeIn(text_group))
        self.wait()


class GroupTheoryViewTilingPart(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(theta = 0, distance = 1E7)
        dim = 5
        pattern = generate_pattern(dim)
        ct_3d = CalissonTiling3D(
            dimension = dim, pattern = pattern,
            tile_config = {"stroke_width" : 3, "stroke_color" : WHITE},
        )
        ct_2d = get_ct_3d_xoy_projection(ct_3d)
        xoy_plane = SurroundingRectangle(ct_2d, color = GREY, fill_opacity = 1, buff = 0.5)
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


# https://en.wikipedia.org/wiki/Presentation_of_a_group
# Lozenge group L is isomorphic to Z^3.
class GroupTheoryViewExplanationPart(Scene):
    def construct(self):
        relation = VGroup(*[TexMobject(text) for text in ("L", "=", "\\mathbb{Z}^3")])
        relation.scale(2)
        text_L, text_equal, text_Z3 = relation
        text_L.set_color(L_GRADIENT_COLORS)
        text_L.set_sheen_direction(DOWN)
        text_equal.rotate(PI/2.)
        text_Z3.set_color(YELLOW)
        relation.arrange_submobjects(UP, buff = 0.3)
        rep_L_text = TexMobject("""
            \\left< a , \\, b , \\, c \\, | \\right.
            \\, & \\left. a b a^{-1} b^{-1} \\right. \\\\
            = & \\left. b c b^{-1} c^{-1} \\right. \\\\
            = & \\left. c a c^{-1} a^{-1} = \\mathbf{1} \\right>
        """)
        rep_Z3_text = TexMobject("""
            \\left< x , \\, y , \\, z \\, | \\right.
            \\, & \\left.  x y x^{-1} y^{-1} \\right. \\\\
            = & \\left. y z y^{-1} z^{-1} \\right. \\\\
            = & \\left. z x z^{-1} x^{-1} = \\mathbf{1} \\right>
        """)
        VGroup(rep_L_text, rep_Z3_text).scale(0.75)
        rep_L_text.next_to(text_L, DOWN, buff = 0.4)
        rep_Z3_text.next_to(text_Z3, UP, buff = 0.4)
        rep_L_rect = SurroundingRectangle(
            rep_L_text, sheen_direction = UP, stroke_color = L_GRADIENT_COLORS,
            fill_color = L_GRADIENT_COLORS, fill_opacity = 0.1
        )
        rep_Z3_rect = SurroundingRectangle(
            rep_Z3_text, stroke_color = YELLOW,
            fill_color = YELLOW, fill_opacity = 0.1
        )
        rep_L = VGroup(rep_L_rect, rep_L_text)
        rep_Z3 = VGroup(rep_Z3_rect, rep_Z3_text)
        # L = Z^3 growing animation
        relation.generate_target()
        relation.fade(1)
        text_equal.scale(0)
        text_L.move_to(text_equal)
        text_Z3.move_to(text_equal)
        self.play(MoveToTarget(relation), run_time = 5)
        self.wait()
        # Group representations of L and Z^3 growing animation
        rep_L.generate_target()
        rep_Z3.generate_target()
        VGroup(rep_L, rep_Z3).fade(1)
        VGroup(rep_L, rep_Z3).scale(0)
        rep_L.move_to(text_L)
        rep_Z3.move_to(text_Z3)
        self.play(MoveToTarget(rep_L), MoveToTarget(rep_Z3), run_time = 2)
        self.wait()


class GroupTheoryView2DGenerators(Scene):
    def construct(self):
        axes_2d = VGroup(*[
            Vector(rotate_vector(RIGHT, PI/2. - TAU/6*k), color = WHITE)
            for k in range(6)
        ])
        texts_2d = VGroup(*[
            TexMobject(element).shift(axis.get_vector() * 1.4)
            for axis, element in zip(axes_2d, ["a", "b", "c", "a^{-1}", "b^{-1}", "c^{-1}"])
        ])
        system_2d = VGroup(axes_2d, texts_2d)
        plane_2d = SurroundingRectangle(axes_2d, color = GREY, fill_opacity = 0.4, buff = 0.8)
        generators_2d = VGroup(plane_2d, system_2d)
        generators_2d.set_height(FRAME_HEIGHT - 2 * MED_SMALL_BUFF)
        self.play(FadeIn(generators_2d), run_time = 2)
        self.wait()


class GroupTheoryView3DGenerators(ThreeDScene):
    def construct(self):
        axes_3d = VGroup(*[
            Vector(direction, color = WHITE)
            for direction in (RIGHT, UP, OUT, LEFT, DOWN, IN)
        ])
        VGroup(axes_3d[2], axes_3d[-1]).rotate_about_origin(PI/2., OUT)
        texts_3d = VGroup(*[
            TexMobject(element) for element in ["x", "y", "z", "x^{-1}", "y^{-1}", "z^{-1}"]
        ])
        factors_3d = [1.3, 1.2, 1.2, 1.5, 1.4, 1.2]
        for text, axis, factor in zip(texts_3d, axes_3d, factors_3d):
            text.rotate_about_origin(PI/2., axis = RIGHT)
            text.rotate_about_origin(PI/2., axis = OUT)
            text.move_to(axis.get_vector() * factor)
        generators_3d = VGroup(axes_3d, texts_3d)
        generators_3d.scale(3)
        self.set_camera_orientation(phi = PI/3, theta = PI/10, distance = 1E7)
        self.play(FadeIn(generators_3d), run_time = 2)
        self.begin_ambient_camera_rotation(rate = 0.015)
        self.wait(30)


class GroupTheoryViewRegionsPart(Scene):
    def construct(self):
        ct_grid = CalissonTilingGrid(
            side_length = 15, unit_size = 0.6,
            grid_lines_type = DashedLine,
            grid_lines_config = {"stroke_width" : 0.5, "stroke_color" : GREY},
            grid_triangles_config = {"opacity" : 0}
        )
        ct_grid.add_grid_lines()
        ct_grid.add_grid_triangles()
        text = TextMobject("能否用$\\left\\{ \\quad, \\quad\\;, \\quad \\right\\}$镶嵌?")
        text.scale(1.2)
        text.to_edge(UP)
        text_rect = SurroundingRectangle(
            text, stroke_width = 0,
            fill_color = BLACK, fill_opacity = 0.5,
        )
        text_tiles = VGroup(*[
            RhombusType(rhombus_config = {"fill_opacity" : 1, "stroke_width" : 1}).scale(0.5)
            for RhombusType in (RRhombus, HRhombus, LRhombus)
        ])
        text_tiles[0].next_to(text[3], RIGHT, buff = 0.15)
        text_tiles[2].next_to(text[6], LEFT, buff = 0.15)
        text_tiles[1].move_to((text_tiles[0].get_center() + text_tiles[2].get_center()) / 2.)
        self.add(ct_grid)
        self.add(text_rect, text, text_tiles)
        mobs = VGroup()
        for acs, stroke_color, fill_color, sheen_direction in ALL_SETTINGS:
            mob = VMobject(
                stroke_width = 5, stroke_color = stroke_color,
                sheen_direction = sheen_direction,
                fill_color = fill_color, fill_opacity = 0.6
            )
            mob_anchor_points = [ct_grid.coords_to_point(x, y) for x, y in acs]
            mob.set_anchor_points(mob_anchor_points, mode = "corners")
            shift_vector = ct_grid.coords_to_point(-1, 0)
            mob.shift(shift_vector)
            mobs.add(mob)
        for k, mob in enumerate(mobs):
            if k == 0:
                self.play(DrawBorderThenFill(mob), run_time = 2)
            else:
                self.play(Transform(mobs[0], mob), run_time = 2)
            self.wait(2)
        self.wait()


class ConnectionsToOtherFields(Scene):
    def construct(self):
        # Original Tiling
        ct_grid = CalissonTilingGrid(unit_size = 0.4)
        ct_3d = CalissonTiling3D(
            dimension = 5, pattern = MAA_PATTERN, enable_fill = True,
            tile_config = {"stroke_width" : 2},
        )
        ct_2d = CalissonTiling2D(
            ct_3d, ct_grid,
            dumbbell_config = {"point_size" : 0.06, "stem_width" : 0.03}
        )
        ct_2d.move_to(ORIGIN)
        # Group Theory: Isomorphism - the core of the famous 3D proof (up)
        gt_iso = TexMobject("L", "=", "\\mathbb{Z}^3")
        gt_iso.scale(2.2)
        gt_iso[0].set_color(L_GRADIENT_COLORS)
        gt_iso[0].set_sheen_direction(DOWN)
        # Algebra: proof by constructing an invariant (up left corner)
        rhombi = VGroup(*[
            RhombusType(rhombus_config = {"stroke_width" : 3,"fill_opacity" : 1})
            for RhombusType in (RRhombus, HRhombus, LRhombus)
        ])
        values = TexMobject("+1", "0", "-1", color = BLACK)
        alg_invar = VGroup()
        for rhombus, value in zip(rhombi, values):
            value.arrange_submobjects(RIGHT, buff = 0.1)
            value.set_height(0.3)
            value.move_to(rhombus)
            alg_invar.add(VGroup(rhombus, value))
        alg_invar.arrange_submobjects(RIGHT)
        # Linear Algebra: Matrix - proof using system of linear equations (down left corner)
        la_matrix = Matrix(
            np.array([
                ["1", "1", "1"],
                ["{1 \\over 2}", "-1", "{1 \\over 2}"],
                ["-{\\sqrt{3} \\over 2}", "0", "{\\sqrt{3} \\over 2}"],
            ]),
            v_buff = 1.4, h_buff = 1.2,
            element_alignment_corner = ORIGIN
        )
        la_matrix.scale(0.8)
        # Combinatorics: Plane Partition (up right corner)
        comb_pp = VGroup()
        for row in MAA_PATTERN:
            for element in row:
                comb_pp.add(VGroup(
                    Square(side_length = 0.5, stroke_width = 1),
                    TexMobject(str(element))
                ))
        comb_pp.arrange_submobjects_in_grid(5, 5, buff = 0)
        comb_pp_rect = SurroundingRectangle(
            comb_pp, stroke_width = 5, stroke_color = WHITE, buff = 0
        )
        comb_pp.add(comb_pp_rect)
        # Statistical Mechanics: Dimer Model (down right corner)
        sm_dimer = ct_2d.deepcopy()
        sm_dimer.get_border().set_stroke(width = 3)
        sm_dimer.remove(sm_dimer.get_all_tiles())
        sm_dimer.add(sm_dimer.get_all_dumbbells())
        # Arrange those stuffs to make it looks pleasing
        ct_2d.shift(DOWN)
        gt_iso.to_edge(UP)
        left_group = Group(alg_invar, la_matrix)
        right_group = Group(comb_pp, sm_dimer)
        for group, direction, buff in zip([left_group, right_group], [LEFT, RIGHT], [1.5, 0.4]):
            group.arrange_submobjects(DOWN, aligned_edge = -direction, buff = buff)
            group.to_edge(direction)
        mobs = VGroup(gt_iso, alg_invar, la_matrix, comb_pp, sm_dimer)
        start_points = np.array([1.2*UP, 1.5*LEFT+0.5*UP, 2*LEFT-1*UP, 1.5*RIGHT+0.5*UP, 2*RIGHT-1*UP])
        end_points = np.array([2.4*UP, 3*LEFT+2*UP, 3*LEFT-1.5*UP, 3*RIGHT+2*UP, 3*RIGHT-1.5*UP])
        arrows = VGroup(*[
            Arrow(start_point, end_point, buff = 0)
            for start_point, end_point in zip(start_points, end_points)
        ])
        self.add(ct_2d, gt_iso,  arrows[0])
        self.wait()
        for mob, arrow in zip(mobs[1:], arrows[1:]):
            self.play(
                GrowArrow(arrow),
                Write(mob, submobject_mode = "all_at_once"),
                run_time = 1)
            self.wait()
        self.play(Group(*self.mobjects).shift, 8*UP, run_time = 2)
        self.wait()


class FinalProblem(Scene):
    def construct(self):     
        # Add titles
        sqr_title = TextMobject("正方形网格").set_color(YELLOW)
        sqr_title.move_to(LEFT_SIDE/2.).to_edge(UP, buff = 0.2)
        hex_title = TextMobject("正六边形网格").set_color(YELLOW)
        hex_title.move_to(RIGHT_SIDE/2.).to_edge(UP, buff = 0.2)
        sep_line = DashedLine(TOP, BOTTOM, color = LIGHT_GRAY)
        self.play(ShowCreation(sep_line), Write(sqr_title), run_time = 1)
        self.play(Write(hex_title))
        self.wait()
        # Setup square grid and its 1x2 units
        sqr_grid = VGroup(*[Square(side_length = 0.7, stroke_width = 3) for k in range(16)])
        sqr_grid.arrange_submobjects_in_grid(4, 4, buff = 0)
        domino_horiz = VGroup(sqr_grid[0], sqr_grid[1]).deepcopy()
        domino_horiz.set_fill(opacity = 1, color = TILE_GREEN)
        domino_vert = VGroup(sqr_grid[0], sqr_grid[4]).deepcopy()
        domino_vert.set_fill(opacity = 1, color = TILE_RED)
        sqr_units = VGroup(domino_vert, domino_horiz).scale(0.7).set_stroke(width = 2)
        sqr_units.arrange_submobjects(DOWN, buff = 0.5)
        sqr_group = VGroup(sqr_grid, sqr_units)
        sqr_group.arrange_submobjects(RIGHT, buff = 1).next_to(sqr_title, DOWN, buff = 0.4)
        # Setup hexagonal grid and its 1x2 units
        hex_grid = HexagonalGrid()
        hexomino_ur = VGroup(hex_grid[0], hex_grid[1]).deepcopy()
        hexomino_ur.set_fill(opacity = 1, color = TILE_RED)
        hexomino_horiz = VGroup(hex_grid[1], hex_grid[2]).deepcopy()
        hexomino_horiz.set_fill(opacity = 1, color = TILE_GREEN)
        hexomino_dr = VGroup(hex_grid[0], hex_grid[2]).deepcopy()
        hexomino_dr.set_fill(opacity = 1, color = TILE_BLUE)
        hex_units = VGroup(hexomino_ur, hexomino_horiz, hexomino_dr).scale(0.5).set_stroke(width = 2)
        hex_units.arrange_submobjects(DOWN, buff = 0.3)
        hex_group = VGroup(hex_grid, hex_units)
        hex_group.arrange_submobjects(RIGHT, buff = 1).next_to(hex_title, DOWN, buff = 0.4)
        self.play(ShowCreation(sqr_grid), ShowCreation(hex_grid))
        self.wait()
        self.play(FadeInFrom(sqr_units, LEFT), FadeInFrom(hex_units, LEFT))
        self.wait()
        # Tile the grid
        sqr_tiles_pairs = [
            (HDomino, [0, 1]), (HDomino, [2, 3]), (VDomino, [4, 8]), (VDomino, [5, 9]),
            (HDomino, [6, 7]), (HDomino, [10, 11]), (HDomino, [12, 13]), (HDomino, [14, 15]),
        ]
        sqr_tiles = VGroup(*[
            DominoType().move_to(Group(sqr_grid[i], sqr_grid[j]).get_center())
            for DominoType, (i, j) in sqr_tiles_pairs
        ])
        hex_tiles_pairs = [
            (RHexomino, [0, 1]), (HHexomino, [3, 4]), (LHexomino, [2, 5]),
            (HHexomino, [6, 7]), (HHexomino, [8, 9]),
        ]
        hex_tiles = VGroup(*[
            HexominoType().move_to(Group(hex_grid[i], hex_grid[j]).get_center())
            for HexominoType, (i, j) in hex_tiles_pairs
        ])
        self.play(
            LaggedStart(GrowFromCenter, sqr_tiles),
            LaggedStart(GrowFromCenter, hex_tiles),
            run_time = 1,
        )
        self.wait()
        # Show the count
        sqr_counts = VGroup(*[
            TexMobject(f"\\times {k}").next_to(mob, RIGHT).set_color(mob[0].get_fill_color())
            for k, mob in zip([2, 6], sqr_units)
        ])
        hex_counts = VGroup(*[
            TexMobject(f"\\times {k}").next_to(mob, RIGHT).set_color(mob[0].get_fill_color())
            for k, mob in zip([1, 3, 1], hex_units)
        ])
        self.play(Write(sqr_counts), Write(hex_counts))
        self.wait()
        # Show different tilings and different count
        new_sqr_grid, new_sqr_units = new_sqr_group = VGroup(sqr_grid, sqr_units).deepcopy()
        new_sqr_group.shift(3.4*DOWN)
        new_hex_grid, new_hex_units = new_hex_group = VGroup(hex_grid, hex_units).deepcopy()
        new_hex_group.shift(3.3*DOWN)
        new_sqr_tiles_pairs = [
            (HDomino, [0, 1]), (VDomino, [2, 6]), (VDomino, [3, 7]), (HDomino, [4, 5]),
            (VDomino, [8, 12]), (VDomino, [9, 13]), (HDomino, [10, 11]), (HDomino, [14, 15]),
        ]
        new_sqr_tiles = VGroup(*[
            DominoType().move_to(Group(new_sqr_grid[i], new_sqr_grid[j]).get_center())
            for DominoType, (i, j) in new_sqr_tiles_pairs
        ])
        new_hex_tiles_pairs = [
            (LHexomino, [0, 2]), (LHexomino, [1, 4]), (RHexomino, [3, 6]),
            (HHexomino, [7, 8]), (LHexomino, [5, 9]),
        ]
        new_hex_tiles = VGroup(*[
            HexominoType().move_to(Group(new_hex_grid[i], new_hex_grid[j]).get_center())
            for HexominoType, (i, j) in new_hex_tiles_pairs
        ])
        new_sqr_counts = VGroup(*[
            TexMobject(f"\\times {k}").next_to(mob, RIGHT).set_color(mob[0].get_fill_color())
            for k, mob in zip([4, 4], new_sqr_units)
        ])
        new_hex_counts = VGroup(*[
            TexMobject(f"\\times {k}").next_to(mob, RIGHT).set_color(mob[0].get_fill_color())
            for k, mob in zip([1, 1, 3], new_hex_units)
        ])
        new_group = VGroup(
            new_sqr_units, new_hex_units, new_sqr_tiles, new_hex_tiles,
            new_sqr_counts, new_hex_counts,
        )
        self.play(FadeInFrom(new_group, UP))
        self.wait()
        # Done
        screen_rect = FullScreenFadeRectangle()
        question_mark = TextMobject("?", color = YELLOW).scale(7)
        self.play(FadeIn(screen_rect), Write(question_mark))
        self.wait()


class OutroScene(Scene):
    def construct(self):
        logo = ImageMobject("logo.png").scale(0.5)
        author = TextMobject("@Solara570").scale(0.7)
        author.to_corner(DR)
        logo.next_to(author, UP)
        self.add(logo, author)
        self.wait()
        # A tiling idea for the outro transition, but it's too bright and distracting.
        # So screw it as well.
        
        # width, height = logo.get_width(), logo.get_height()
        # for i in range(-4, 5):
        #     for j in range(-9, 10):
        #         new_logo = logo.deepcopy()
        #         new_logo.shift(0.6*i * width*RIGHT + 0.75*j * height*UP)
        #         self.add(new_logo)
        # self.wait()


#####
## Thumbnail

class Thumbnail(Scene):
    def construct(self):
        dim = 24
        pattern = generate_pattern(dim)
        ct_grid = CalissonTilingGrid(unit_size = 0.5)
        ct_3d = CalissonTiling3D(
            dimension = dim, pattern = pattern, enable_fill = True
        )
        ct_2d = CalissonTiling2D(
            ct_3d, ct_grid, dumbbell_config = {"point_size" : 0.06, "stem_width" : 0.03}
        )
        tiles = ct_2d.get_all_tiles()
        dumbbells = ct_2d.get_all_dumbbells()
        for tile_set in tiles:
            for tile in tile_set:
                x = tile.get_center_of_mass()[0]
                opacity = self.interpolate_fill_opacity_by_x(x, 0, 1)
                stroke_width = self.interpolate_stroke_and_fade_by_x(x, 0, 3)
                tile.set_fill(opacity = opacity)
                tile.set_stroke(width = stroke_width)
        for dumbbell_set in dumbbells:
            for dumbbell in dumbbell_set:
                x = dumbbell.get_center_of_mass()[0]
                fade_factor = self.interpolate_stroke_and_fade_by_x(x, 0, 1)
                dumbbell.fade(fade_factor)
        sep_lines = VGroup(*[
            DashedLine(TOP, BOTTOM, color = YELLOW).move_to(position)
            for position in (LEFT_SIDE/3, RIGHT_SIDE/3)
        ])
        titles = VGroup(*[
            TextMobject(text, color = YELLOW).scale(3.5).move_to(position)
            for text, position in zip(["2D...", "3D?", "2D!"], [LEFT_SIDE*2/3, ORIGIN, RIGHT_SIDE*2/3])
        ]).to_edge(UP, buff = 0.3)
        titles_bg_rect = BackgroundRectangle(titles).scale(1.6)
        self.add(ct_2d, dumbbells, titles_bg_rect, sep_lines, titles)
        self.wait()

    def interpolate_stroke_and_fade_by_x(self, x, min_val, max_val, inflection = 10.0):
        R = RIGHT_SIDE[0]
        if x <= R/6:
            return max_val
        elif x>= R/2:
            return min_val
        else:
            alpha = 1 - smooth(np.clip(3*x/R-1/2, 0, 1))
            return min_val + (max_val - min_val) * alpha

    def interpolate_fill_opacity_by_x(self, x, min_val, max_val, inflection = 10.0):
        return self.interpolate_stroke_and_fade_by_x(np.abs(x), min_val, max_val, inflection)







