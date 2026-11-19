import numpy as np
import itertools as it
import cmath
from manimlib import *


# Constants
INF_THRESHOLD = 1e6
ZERO_THRESHOLD = 1e-6
ISOTROPIC_METRIC = np.array([
    [0, -1 / 2, 0, 0],
    [-1 / 2, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])
F = ISOTROPIC_METRIC

CIRCLE_COLORS = [RED, GREEN, BLUE, ORANGE]
SPHERE_COLORS = [RED, GREEN, BLUE, ORANGE, PINK]
CURVATURE_T2C = dict([
    (f"k_{{{i + 1}}}", color)
    for i, color in enumerate(SPHERE_COLORS)
])
CURVATURE_T2C["k_{n+2}"] = SPHERE_COLORS[-1]
RELATION_COLORS = {
    "outer": PURPLE_E,      # 相离 (<-1)
    "tangent_out": GOLD_B,  # 外切 (-1)
    "intersect": PURPLE_C,  # 相交 ((-1, 1))
    "tangent_in": GOLD_B,   # 内切 (1)
    "inner": PURPLE_A,      # 内含 (>1)
}

CMU_TEXT_CONFIG = {"font": "CMU Serif"}


# Functions
def get_circle_intersections(c1, c2):
    p1 = c1.get_center()
    p2 = c2.get_center()
    r1 = c1.get_radius()
    r2 = c2.get_radius()

    d_vec = p2 - p1
    d = np.linalg.norm(d_vec[:2])
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []
    u = d_vec / d
    u_perp = np.array([-u[1], u[0], 0.0])

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0.0, r1**2 - a**2))
    mid = p1 + a * u
    if np.isclose(h, 0.0):
        return [mid]
    return [mid + h * u_perp, mid - h * u_perp]


def get_tangent_circle_centers(k1, k2, k3):
    r1, r2, r3 = 1.0 / k1, 1.0 / k2, 1.0 / k3
    d12 = r1 + r2
    d13 = r1 + r3
    d23 = r2 + r3
    c1 = ORIGIN
    c2 = np.array([d12, 0.0, 0.0])
    x3 = (d12**2 + d13**2 - d23**2) / (2.0 * d12)
    y3 = np.sqrt(max(0.0, d13**2 - x3**2))
    c3 = np.array([x3, y3, 0.0])
    return c1, c2, c3


def get_angle_arc(vertex, p1, p2, radius=0.3, **kwargs):
    v1 = p1 - vertex
    v2 = p2 - vertex
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    d_angle = (angle2 - angle1) % TAU
    if d_angle > PI:
        start_angle = angle2
        angle_span = TAU - d_angle
    else:
        start_angle = angle1
        angle_span = d_angle

    radius = min(radius, np.linalg.norm(v1) * 0.5, np.linalg.norm(v2) * 0.5)
    arc = Arc(
        start_angle=start_angle,
        angle=angle_span,
        radius=radius,
        arc_center=vertex,
        **kwargs
    )
    return arc


def get_angle_label(vertex, p1, p2, radius=0.4, label_tex="\\theta", **kwargs):
    v1 = p1 - vertex
    v2 = p2 - vertex
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    d_angle = (angle2 - angle1) % TAU
    if d_angle > PI:
        start_angle = angle2
        angle_span = TAU - d_angle
    else:
        start_angle = angle1
        angle_span = d_angle
    bisector_angle = start_angle + angle_span / 2.0
    bisector_unit = np.array([np.cos(bisector_angle), np.sin(bisector_angle), 0.0])
    label_dist = radius * 1.5
    label_pos = vertex + bisector_unit * label_dist
    label = Tex(label_tex, **kwargs)
    scale_factor = float(np.clip(angle_span / 1.0, 0, 0.85))
    label.scale(scale_factor)
    label.move_to(label_pos)
    return label


def get_line_label(line, label_tex, buff=0.3, rotate=True, invert=False, **kwargs):
    label = Tex(label_tex, **kwargs)
    unit_v = line.get_unit_vector()
    normal = np.array([-unit_v[1], unit_v[0], 0.0])
    if invert:
        normal *= -1
    label.move_to(line.get_center() + normal * buff)

    if rotate:
        angle = line.get_angle()
        if (PI / 2 < angle <= 3 * PI / 2) or (-3 * PI / 2 <= angle < -PI / 2):
            angle += PI
        label.rotate(angle)

    return label


def get_vector_product(v1, v2):
    return v1.T @ ISOTROPIC_METRIC @ v2


def get_circle_product(c1, c2):
    v1 = c1.get_pedoe_vector()
    v2 = c2.get_pedoe_vector()
    return get_vector_product(v1, v2)


def sum_of_pedoe_vectors(vector1, vector2, weight1=0.3, weight2=0.7, monic=False):
    v1, v2 = np.array(vector1), np.array(vector2)
    if monic:
        reg_v1, reg_v2 = v1 / v1[0], v2 / v2[0]
        return weight1 * reg_v1 + weight2 * reg_v2
    else:
        return weight1 * v1 + weight2 * v2


def geom_to_pedoe_vector(center, radius):
    # Given parameters of a geometry object (dot, line or circle), get its Pedoe vector
    cx, cy = center[:2]
    if radius >= INF_THRESHOLD or radius == float('inf'):
        # Line type: a horizontal line passing through 'center'
        return np.array([0, 2 * cy, 0, 1])
    elif abs(radius) < ZERO_THRESHOLD:
        # Dot type: a dot at 'center'
        return np.array([1, cx**2 + cy**2, cx, cy])
    else:
        # Circle type: a circle centered at 'center' with radius 'radius'
        return np.array([
            1 / radius,
            (cx**2 + cy**2 - radius**2) / radius,
            cx / radius,
            cy / radius
        ])


def analyze_raw_vector(raw_vector):
    # Parse a given vector 'raw_vector', return its characteristics as a Pedoe vector
    # shape_type, center, radius, and pedoe_vector (normalized)
    v1, v2, v3, v4 = raw_vector

    # 1. Line type
    if np.abs(v1) < ZERO_THRESHOLD:
        norm_factor = np.hypot(v3, v4)
        if norm_factor < ZERO_THRESHOLD:
            raise Warning("This is a Line at Infinity, can't be drawn.")
            return {
                "shape_type": "line_at_infinity",
                "center": np.inf,
                "radius": np.inf,
                "pedoe_vector": np.array([0., 0., 0., 1.]),
            }

        cx = (v2 * v3) / (2 * norm_factor**2)
        cy = (v2 * v4) / (2 * norm_factor**2)
        center = np.array([cx, cy, 0])
        pedoe_vector = np.array(raw_vector) / norm_factor
        return {
            "shape_type": "line",
            "center": center,
            "radius": np.inf,
            "pedoe_vector": pedoe_vector,
        }

    # 2. Circle or dot type
    center = np.array([v3 / v1, v4 / v1, 0])
    D = v3**2 + v4**2 - v1 * v2    # D = self inner product, should be 1

    if D > ZERO_THRESHOLD:
        # 2.1 Real circle type, with a positive first entry
        pedoe_vector = np.array(raw_vector) / np.sqrt(D) * np.sign(v1)
        radius = 1 / pedoe_vector[0]
        return {
            "shape_type": "real_circle",
            "center": center,
            "radius": radius,
            "pedoe_vector": pedoe_vector,
        }

    elif D < -ZERO_THRESHOLD:
        # 2.2 Imaginary circle type, with a positive first entry
        pedoe_vector = np.array(raw_vector) / np.sqrt(-D) * np.sign(v1)
        radius = 1 / pedoe_vector[0]
        return {
            "shape_type": "imaginary_circle",
            "center": center,
            "radius": radius,
            "pedoe_vector": pedoe_vector,
        }

    else:
        # 2.3 Dot type, with first entry 1
        pedoe_vector = np.array(raw_vector) / v1
        return {
            "shape_type": "dot",
            "center": center,
            "radius": 0.0,
            "pedoe_vector": pedoe_vector,
        }


def get_line_geometry(pedoe_vector):
    # Get the center and direction vector from a line-type Pedoe vector
    v1, v2, v3, v4 = pedoe_vector
    cx = (v2 * v3) / (2 * (v3**2 + v4**2))
    cy = (v2 * v4) / (2 * (v3**2 + v4**2))

    center = np.array([cx, cy, 0.0])
    direction = np.array([-v4, v3, 0.0])
    return center, direction


def generate_descartes_matrix(random_seed=None):
    # Generate a random Descartes' configuration and return its 4-by-4 matrix
    rng = random.Random(random_seed)
    config_type = rng.choice(['external', 'enclosing'])

    while True:
        r1, r2, r3 = [rng.uniform(1.0, 10.0) for _ in range(3)]
        k1, k2, k3 = 1 / r1, 1 / r2, 1 / r3

        root_term = 2 * np.sqrt(k1 * k2 + k2 * k3 + k3 * k1)

        if config_type == 'external':
            k4 = k1 + k2 + k3 + root_term
            break
        else:
            k4 = k1 + k2 + k3 - root_term
            if k4 < 0:
                break

    r4 = 1 / k4

    c1 = 0.0 + 0.0j
    c2 = (r1 + r2) + 0.0j

    d = r1 + r2
    x3 = ((r1 + r3)**2 - (r2 + r3)**2 + d**2) / (2 * d)
    y3 = np.sqrt(max(0, (r1 + r3)**2 - x3**2))
    c3 = x3 + y3 * 1.0j

    w1, w2, w3 = k1 * c1, k2 * c2, k3 * c3
    sqrt_term_complex = cmath.sqrt(w1 * w2 + w2 * w3 + w3 * w1)

    c4_plus = (w1 + w2 + w3 + 2 * sqrt_term_complex) / k4
    c4_minus = (w1 + w2 + w3 - 2 * sqrt_term_complex) / k4

    if np.isclose(abs(c4_plus - c1), abs(r1 + r4), atol=1e-5):
        c4 = c4_plus
    else:
        c4 = c4_minus

    radii = [r1, r2, r3, r4]
    centers = [c1, c2, c3, c4]

    # Random affine transformation
    scale = rng.uniform(0.5, 3.0)
    angle = rng.uniform(0, 2 * np.pi)
    translation = rng.uniform(-10, 10) + rng.uniform(-10, 10) * 1.0j

    # Fill out the C, the matrix of the configuration
    C = np.zeros((4, 4))

    for i in range(4):
        new_r = radii[i] * scale
        new_c = centers[i] * scale * cmath.exp(1j * angle) + translation

        x = new_c.real
        y = new_c.imag

        C[:, i] = [
            1 / new_r,
            (x**2 + y**2 - new_r**2) / new_r,
            x / new_r,
            y / new_r
        ]

    return C


# Animations
class DirectionalFlash(Flash):
    def __init__(
        self,
        point,
        direction=RIGHT,
        spread_angle=90 * DEGREES,
        num_lines=7,
        **kwargs,
    ):
        self.direction = direction
        self.spread_angle = spread_angle
        super().__init__(point, num_lines=num_lines, **kwargs)

    def create_lines(self):
        lines = VGroup()
        center_angle = angle_of_vector(self.direction)

        if self.num_lines == 1 or self.spread_angle == 0:
            angles = [center_angle]
        else:
            angles = np.linspace(
                center_angle - self.spread_angle / 2,
                center_angle + self.spread_angle / 2,
                self.num_lines
            )

        for angle in angles:
            line = Line(self.point, self.point + self.line_length * RIGHT)
            line.shift(self.flash_radius * RIGHT)
            line.rotate(angle, about_point=self.point)
            lines.add(line)

        lines.set_color(self.color)
        lines.set_stroke(width=self.line_stroke_width)
        return lines


# Mobjects
class RemarkText(TexText):
    def __init__(
        self, mob, text,
        direction=DOWN, aligned_edge=ORIGIN,
        scale_factor=0.6, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        color=GREY,
        **kwargs
    ):
        super().__init__(text, **kwargs)
        self.scale(scale_factor)
        self.next_to(mob, direction=direction, aligned_edge=aligned_edge, buff=buff)
        self.set_color(color)


class DecimalEntry(DecimalNumber):
    def __init__(self, number=0, num_decimal_places=4, edge_to_fix=RIGHT, **kwargs):
        super().__init__(
            show_ellipsis=False,
            num_decimal_places=num_decimal_places,
            edge_to_fix=edge_to_fix,
            text_config=CMU_TEXT_CONFIG,
            **kwargs,
        )
        self.set_value(number)


class FadeCoordinatePlane(NumberPlane):
    def __init__(
        self,
        x_range=(-8.0, 8.0, 1.0),
        y_range=(-4.0, 4.0, 1.0),
        include_labels=True,
        **kwargs,
    ):
        super().__init__(x_range=x_range, y_range=y_range, **kwargs)
        self.background_lines.set_style(stroke_color=GREY_D)
        self.faded_lines.set_style(stroke_color=GREY_E)
        if include_labels:
            text_config = {"font": "CMU Serif"}
            self.add_coordinate_labels(text_config=text_config)
        self.set_opacity(0.8)


class LeftSideCoordinatePlane(FadeCoordinatePlane):
    def __init__(
        self,
        x_range=(-4, 4, 1),
        y_range=(-4, 4, 1),
        **kwargs
    ):
        super().__init__(x_range=x_range, y_range=y_range, **kwargs)
        shift_vec = LEFT_SIDE - self.c2p(self.x_range[0], 0)
        self.shift(shift_vec)


class FadeComplexPlane(ComplexPlane):
    def __init__(
        self,
        x_range=(-8.0, 8.0, 1.0),
        y_range=(-4.0, 4.0, 1.0),
        **kwargs,
    ):
        super().__init__(x_range=x_range, y_range=y_range)
        self.background_lines.set_style(stroke_color=GREY_D)
        self.faded_lines.set_style(stroke_color=GREY_E)
        text_config = {"font": "CMU Serif"}
        self.add_coordinate_labels(font_size=24, text_config=text_config)
        self.set_opacity(0.8)


class LeftSideComplexPlane(FadeComplexPlane):
    def __init__(self, x_range=(-4, 4, 1), y_range=(-4, 4, 1), **kwargs):
        super().__init__(
            x_range=x_range,
            y_range=y_range,
            **kwargs,
        )
        shift_vec = LEFT_SIDE - self.c2p(-4, 0)
        self.shift(shift_vec)


class FakeMatrix(VGroup):
    def __init__(self, text, side_length=3, scale_factor=2, **kwargs):
        super().__init__(**kwargs)
        matrix_frame = Square(side_length=side_length)
        content = Tex(text)
        content.scale(scale_factor)
        self.add(matrix_frame, content)
        self.content = content
        self.matrix_frame = matrix_frame

    def get_content(self):
        return self.content

    def get_matrix_frame(self):
        return self.matrix_frame


class InnerProductRelationLine(VGroup):
    def __init__(
        self,
        length=None,
        unit_size=1.5,
        x_min=-3.5,
        x_max=3.5,
        initial_val=0.0,
        axis_label_text="\\langle \\mathbf{c}_{1}, \\mathbf{c}_{2} \\rangle",
        relation_colors=None,
        pointer_color=GOLD_B,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.x_min = x_min
        self.x_max = x_max
        self.val_tracker = ValueTracker(initial_val)
        self.axis_label_text = axis_label_text
        self.pointer_color = pointer_color

        total_units = self.x_max - self.x_min
        if length is not None:
            self.length = length
            self.unit_size = length / total_units
        else:
            self.unit_size = unit_size
            self.length = total_units * unit_size

        self.relation_colors = dict(RELATION_COLORS)
        if relation_colors:
            self.relation_colors.update(relation_colors)

        self._build_axis()
        self._build_regions_and_ticks()
        self._build_slider()

    def val_to_point(self, val):
        unit_vector = self.unit_point.get_center() - self.center_point.get_center()
        return self.center_point.get_center() + unit_vector * val

    def _build_axis(self):
        self.center_point = VectorizedPoint(ORIGIN)
        self.unit_point = VectorizedPoint(RIGHT * self.unit_size)
        p_left = self.val_to_point(self.x_min)
        p_neg1 = self.val_to_point(-1)
        p_pos1 = self.val_to_point(1)
        p_right = self.val_to_point(self.x_max)

        tip_l = ArrowTip(fill_color=self.relation_colors["outer"], fill_opacity=1).scale(0.8)
        tip_l.rotate(PI).move_to(p_left)

        tip_r = ArrowTip(fill_color=self.relation_colors["inner"], fill_opacity=1).scale(0.8)
        tip_r.move_to(p_right)

        axis_label = Tex(self.axis_label_text, font_size=36, fill_color=self.pointer_color)
        axis_label.next_to(tip_r, RIGHT, buff=0.25)

        seg_outer = Line(p_left, p_neg1, stroke_width=5, color=self.relation_colors["outer"])
        seg_intersect = Line(p_neg1, p_pos1, stroke_width=5, color=self.relation_colors["intersect"])
        seg_inner = Line(p_pos1, p_right, stroke_width=5, color=self.relation_colors["inner"])

        self.axis_group = VGroup(
            self.center_point, self.unit_point,
            seg_outer, seg_intersect, seg_inner,
            tip_l, tip_r,
            axis_label,
        )
        self.add(self.axis_group)

    def _build_regions_and_ticks(self):
        p_neg1 = self.val_to_point(-1)
        p_pos1 = self.val_to_point(1)

        # 统一对称刻度线
        tick_neg1 = Line(UP * 0.16, DOWN * 0.16, stroke_width=4, color=WHITE).move_to(p_neg1)
        tick_pos1 = Line(UP * 0.16, DOWN * 0.16, stroke_width=4, color=WHITE).move_to(p_pos1)

        label_neg1 = Tex("-1", fill_color=self.relation_colors["tangent_out"])
        label_neg1.next_to(tick_neg1, DOWN, buff=0.2)
        label_pos1 = Tex("1", fill_color=self.relation_colors["tangent_in"])
        label_pos1.next_to(tick_pos1, DOWN, buff=0.2)

        sub_neg1 = TexText("外切", font_size=30, fill_color=self.relation_colors["tangent_out"])
        sub_neg1.next_to(label_neg1, DOWN, buff=0.25)
        sub_pos1 = TexText("内切", font_size=30, fill_color=self.relation_colors["tangent_in"])
        sub_pos1.next_to(label_pos1, DOWN, buff=0.25)

        reg_outer = TexText("相离", font_size=30, fill_color=self.relation_colors["outer"])
        reg_outer.move_to((self.val_to_point(self.x_min) * 2 + p_neg1) / 3)

        reg_intersect = TexText("相交", font_size=30, fill_color=self.relation_colors["intersect"])
        reg_intersect.move_to((p_neg1 + p_pos1) / 2)

        reg_inner = TexText("内含", font_size=30, fill_color=self.relation_colors["inner"])
        reg_inner.move_to((p_pos1 + self.val_to_point(self.x_max) * 2) / 3)

        text_labels = VGroup(reg_outer, sub_neg1, reg_intersect, sub_pos1, reg_inner)
        for label in (reg_outer, reg_intersect, reg_inner):
            label.match_y(sub_neg1)

        self.static_annotations = VGroup(
            tick_neg1, tick_pos1,
            label_neg1, label_pos1,
            text_labels,
        )
        self.add(self.static_annotations)

    def _build_slider(self):
        self.pointer = RegularPolygon(
            n=3,
            start_angle=-PI / 2,
            fill_opacity=1,
            fill_color=self.pointer_color,
            stroke_width=0,
        )
        self.pointer.scale(0.2)
        self.pointer.shift(OUT * 0.1)

        def update_pointer(mob):
            val = self.val_tracker.get_value()
            target_pt = self.val_to_point(val)
            tip = mob.get_vertices()[0]
            mob.shift(target_pt - tip)

        self.pointer.add_updater(update_pointer)
        self.add(self.pointer)

    def get_all_texts(self):
        label_neg1, label_pos1, text_labels = self.static_annotations[-3:]
        axis_label = self.axis_group[-1]
        return VGroup(
            label_neg1, label_pos1, axis_label,
            *text_labels.submobjects,
        )

    def get_value(self):
        return self.val_tracker.get_value()

    def set_value(self, val):
        self.val_tracker.set_value(val)
        return self

    def get_value_tracker(self):
        return self.val_tracker

    def get_unit_size(self):
        if hasattr(self, "axis_group"):
            center_point, unit_point = self.axis_group[:2]
        else:
            center_point = VectorizedPoint(ORIGIN)
            unit_point = VectorizedPoint(RIGHT * self.unit_size)
        return np.linalg.norm(unit_point.get_center() - center_point.get_center())

    def animate_to_value(self, target_val, **kwargs):
        return self.val_tracker.animate(**kwargs).set_value(target_val)


class TangentFourCircles(VGroup):
    # Mainly for demonstration
    def __init__(
        self, p1, p2, p3,
        fourth_type="smaller",  # or "bigger"
        stroke_width=4,
        fill_opacity=0,
        circle_colors=CIRCLE_COLORS,
        n_components=128,
        eps=1e-8,
        tangency_tolerance=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.base_points = [
            self.to_vectorized_point(p1),
            self.to_vectorized_point(p2),
            self.to_vectorized_point(p3),
        ]
        self.fourth_type = fourth_type
        self.stroke_width = stroke_width
        self.fill_opacity = fill_opacity
        self.circle_colors = circle_colors
        self.n_components = n_components
        self.eps = eps
        self.tangency_tolerance = tangency_tolerance

        self.current_centers = None
        self.current_radii = None
        self.current_curvatures = None
        self.fourth_center = VectorizedPoint(ORIGIN)

        self.is_valid = True
        self.fourth_is_valid = True

        # Generate 4 circles
        self.circles = VGroup()
        for color in self.circle_colors:
            circle = Circle(radius=1, n_components=self.n_components)
            circle.set_color(color)
            circle.set_stroke(width=self.stroke_width)
            circle.set_fill(opacity=self.fill_opacity)
            self.circles.add(circle)
        self.add(self.circles)
        # Refresh and update
        self.refresh()
        self.add_updater(lambda m: m.refresh())

    def to_vectorized_point(self, p):
        if isinstance(p, VectorizedPoint):
            return p
        arr = np.array(p, dtype=float)

        # [x, y] -> VectorizedPoint([x, y, 0])
        # [x, y, z] -> VectorizedPoint([x, y, z])
        if arr.shape == (2,):
            arr = np.array([arr[0], arr[1], 0.0])
        if arr.shape != (3,):
            raise ValueError("Point must be a VectorizedPoint or a 2D/3D coordinate.")

        return VectorizedPoint(arr)

    def is_bigger_type(self):
        return self.fourth_type == "bigger"

    def set_fourth_type(self, fourth_type):
        self.fourth_type = fourth_type
        self.refresh()
        return self

    def get_base_centers(self):
        # Return base centers and force them on xy-plane
        centers = []
        for point in self.base_points:
            center = point.get_center().copy()
            center[2] = 0
            centers.append(center)
        return centers

    def compute_first_three_radii(self, c1, c2, c3):
        d12 = np.linalg.norm(c1 - c2)
        d23 = np.linalg.norm(c2 - c3)
        d13 = np.linalg.norm(c1 - c3)
        r1 = (d12 + d13 - d23) / 2
        r2 = (d12 + d23 - d13) / 2
        r3 = (d13 + d23 - d12) / 2
        return r1, r2, r3

    def compute_fourth_curvature(self, r1, r2, r3):
        k1 = 1.0 / r1
        k2 = 1.0 / r2
        k3 = 1.0 / r3
        S = k1 + k2 + k3
        Q = (k1 * k2 + k2 * k3 + k3 * k1)
        root = np.sqrt(max(Q, 0))

        if self.is_bigger_type():
            # For a bigger fourth circle, we choose the negative sign of root
            k4 = S - 2 * root
        else:
            k4 = S + 2 * root

        if abs(k4) < self.eps:
            return None

        return k1, k2, k3, k4

    def compute_fourth_center(self, c1, c2, c3, k1, k2, k3, k4):
        # Use mixed Descartes' Theorem to solve z4 without sqrt
        S = (k1 + k2 + k3)
        T = (k1 * c1 + k2 * c2 + k3 * c3)
        U = (k1**2 * c1 + k2**2 * c2 + k3**2 * c3)

        denominator = (k4 * (k4 - S))
        if abs(denominator) < self.eps:
            return None

        c4 = ((S + k4) * T - 2 * U) / denominator
        c4 = c4.copy()
        c4[2] = 0
        return c4

    def validate_fourth_circle(self, centers, radii, curvatures):
        # Make sure the fourth circle is indeed tangent to the other 3
        c1, c2, c3, c4 = centers
        r1, r2, r3, r4 = radii
        k1, k2, k3, k4 = curvatures
        errors = []
        for ci, ri in zip([c1, c2, c3], [r1, r2, r3]):
            actual = np.linalg.norm(c4 - ci)
            if k4 > 0:
                expected = r4 + ri
            else:
                expected = r4 - ri
                if expected < 0:
                    return False
            errors.append(abs(actual - expected))
        scale = max(1.0, r1, r2, r3, r4)
        tolerance = (self.tangency_tolerance * scale)
        return max(errors) < tolerance

    def compute_configuration(self):
        # Normally returns 'centers', 'radii' and 'curvatures'
        # If first 3 circles are degenerate: returns None, None, None
        # If fourth circle doesn't exist: returns all None for only the last one
        c1, c2, c3 = self.get_base_centers()
        r1, r2, r3 = self.compute_first_three_radii(c1, c2, c3)

        # If first 3 circles are degenerate
        if min(r1, r2, r3) <= self.eps:
            return None, None, None

        curvature_data = self.compute_fourth_curvature(r1, r2, r3)
        # If fourth circle doesn't exist (no valid curvature)
        if curvature_data is None:
            k1, k2, k3 = 1 / r1, 1 / r2, 1 / r3
            return (
                [c1, c2, c3, None],
                [r1, r2, r3, None],
                [k1, k2, k3, None],
            )

        k1, k2, k3, k4 = curvature_data
        c4 = self.compute_fourth_center(c1, c2, c3, k1, k2, k3, k4)
        # If fourth circle doesn't exist (no valid center)
        if c4 is None:
            return (
                [c1, c2, c3, None],
                [r1, r2, r3, None],
                [k1, k2, k3, None],
            )
        r4 = abs(1.0 / k4)
        # Return a valid configuration
        centers = [c1, c2, c3, c4]
        radii = [r1, r2, r3, r4]
        curvatures = [k1, k2, k3, k4]
        return centers, radii, curvatures

    def set_circle_geometry(self, index, center, radius):
        color = self.circle_colors[index]
        new_circle = Circle(radius=radius, n_components=self.n_components)
        new_circle.move_to(center)
        new_circle.set_stroke(color, width=self.stroke_width)
        new_circle.set_fill(color, opacity=self.fill_opacity)
        self.circles[index].become(new_circle)
        return self

    def hide_circle(self, index):
        self.circles[index].set_opacity(0)
        return self

    def refresh(self):
        centers, radii, curvatures = self.compute_configuration()

        # If first 3 circles are degenerate - hide everything
        if centers is None:
            self.is_valid = False
            self.fourth_is_valid = False
            for i in range(4):
                self.hide_circle(i)
            self.current_centers = None
            self.current_radii = None
            self.current_curvatures = None
            return self

        # If not, always refresh first 3 circles
        self.is_valid = True
        for i in range(3):
            self.set_circle_geometry(i, centers[i], radii[i])

        # If fourth circle doesn't exist - hide it
        if centers[3] is None:
            self.fourth_is_valid = False
            self.hide_circle(3)
        else:
            self.fourth_is_valid = True
            self.set_circle_geometry(3, centers[3], radii[3])
            self.fourth_center.move_to(centers[3])

        # Save current data
        self.current_centers = centers
        self.current_radii = radii
        self.current_curvatures = curvatures

        return self

    def update_centers_from_circles(self):
        for base_point, circle in zip(self.base_points, self.get_circles()[:-1]):
            base_point.move_to(circle.get_center())
        self.fourth_center.move_to(self.get_circle(3).get_center())

    def get_circle(self, index):
        return self.circles[index]

    def get_circles(self):
        return self.circles

    def get_centers(self):
        return self.current_centers

    def get_radii(self):
        return self.current_radii

    def get_curvatures(self):
        return self.current_curvatures

    def get_tangent_points(self):
        centers = self.get_centers()
        curvatures = self.get_curvatures()
        assert all(c is not None for c in centers)
        assert all(k is not None for k in curvatures)

        signed_radii = [1.0 / k for k in curvatures]

        pairs = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3)
        ]

        tangent_points = []
        for i, j in pairs:
            ci = centers[i]
            cj = centers[j]
            si = signed_radii[i]
            sj = signed_radii[j]
            assert(abs(si + sj) >= self.eps)
            tangent_point = (sj * ci + si * cj) / (si + sj)
            tangent_point = tangent_point.copy()
            tangent_point[2] = 0
            tangent_points.append(tangent_point)
        return tangent_points


class PedoeVectorMobject(MobjectMatrix):
    def __init__(self, values=[-1, -1, -1, -1], height=2.5, **kwargs):
        super().__init__(
            VGroup(*(DecimalEntry(value) for value in values)),
            n_cols=1,
            element_alignment_corner=RIGHT,
            height=height,
            **kwargs,
        )

    def set_values(self, new_values):
        for entry, new_value in zip(self.get_entries(), new_values):
            entry.set_value(new_value)
        # self.get_entries().arrange(DOWN, aligned_edge=RIGHT, coor_mask=[1, 0, 0], center=False)
        # max_x_point = self.get_entries().get_right()
        # for entry in self.get_entries():
        #     entry.next_to(max_x_point, LEFT, buff=0, coor_mask=[1, 0, 0])
        # self.get_entries().move_to(self.get_brackets().get_center(), coor_mask=[1, 0, 0])
        return self

    def get_values(self):
        return np.array([entry.get_value() for entry in self.get_entries()])


class VCircle(VMobject):
    def __init__(self, plane, raw_vector=None, center=None, radius=None, **kwargs):
        super().__init__()
        self.plane = plane

        if raw_vector is not None:
            geom_data = analyze_raw_vector(raw_vector)
        elif center is not None and radius is not None:
            generated_pedoe = geom_to_pedoe_vector(center, radius)
            geom_data = analyze_raw_vector(generated_pedoe)
        else:
            raise Exception("Parameters are not enough.")

        self.shape_type = geom_data["shape_type"]
        center = geom_data["center"]
        radius = geom_data["radius"]
        pedoe_vector = geom_data["pedoe_vector"]

        # Initialize geometry
        c_pos = self.plane.c2p(*center)
        if self.shape_type == "line":
            # Line type
            _, direction = get_line_geometry(pedoe_vector)
            start_local = center + direction * 20
            end_local = center - direction * 20
            start_pos = self.plane.c2p(*start_local)
            end_pos = self.plane.c2p(*end_local)
            geom = Line(start=start_pos, end=end_pos, **kwargs)
        elif self.shape_type == "dot":
            # Dot type
            geom = Dot(point=c_pos, **kwargs)
        else:
            # Circle type
            edge_local = center + np.array([radius, 0.0, 0.0])
            p_edge = self.plane.c2p(*edge_local)
            actual_r = np.linalg.norm(p_edge - c_pos)
            geom = Circle(radius=actual_r, n_components=128, **kwargs)
            geom.move_to(c_pos)
            if self.shape_type == "imaginary_circle":
                # Virtual circle type
                geom = DashedVMobject(geom, num_dashes=24)

        self.add(geom)
        self.geom = geom

    def get_shape(self):
        return self.geom

    def get_shape_type(self):
        return self.shape_type

    def get_pedoe_vector(self):
        geom = self.geom

        if self.shape_type == "line":
            # Line type
            c_start = np.array(self.plane.p2c(geom.get_start()))
            c_end = np.array(self.plane.p2c(geom.get_end()))
            direction = c_end - c_start
            norm_d = np.linalg.norm(direction)
            if norm_d < 1e-12:
                return np.array([0., 0., 0., 1.])
            direction = direction / norm_d
            nx, ny = -direction[1], direction[0]
            v3, v4 = nx, ny
            v2 = 2.0 * (nx * p1[0] + ny * p1[1])
            return np.array([0.0, v2, v3, v4])
        elif self.shape_type == "dot":
            # Dot type (Explicit)
            c = np.array(self.plane.p2c(geom.get_center()))
            cx, cy = c[0], c[1]
            return np.array([1.0, cx**2 + cy**2, cx, cy])
        else:
            c = np.array(self.plane.p2c(geom.get_center()))
            edge = np.array(self.plane.p2c(geom.get_right()))
            cx, cy = c[0], c[1]
            r = np.linalg.norm(edge - c)
            if r < ZERO_THRESHOLD:
                # Dot type (Implicit)
                return np.array([1.0, cx**2 + cy**2, cx, cy])
            if self.shape_type == "real_circle":
                # Real circle type
                return np.array([
                    1 / r,
                    (cx**2 + cy**2 - r**2) / r,
                    cx / r,
                    cy / r
                ])
            else:
                # Virtual circle type
                return np.array([
                    1 / r,
                    (cx**2 + cy**2 + r**2) / r,
                    cx / r,
                    cy / r
                ])

    def get_center_coords(self):
        return analyze_raw_vector(self.get_pedoe_vector())["center"]

    def get_radius(self):
        return analyze_raw_vector(self.get_pedoe_vector())["radius"]

    def get_product_with(self, other):
        return get_circle_product(self, other)


class GlowCircle(Mobject):
    """One GLSL quad that glows around an existing Circle.

    Parameters:
        circle: An existing Circle (or a compatible subclass).
        sigma: Light falloff size in Manim scene coordinates (0.06-0.16).
        intensity: Glow brightness multiplier (0.6-1.6 typical).
        color: Override glow hue. Defaults to circle stroke color.
        follow: Bind one lightweight updater to the source circle.
        padding_sigmas: The quad extends this many sigma units beyond rim.
    """
    shader_folder = os.path.join(Path(__file__).resolve().parent, "circle_glow_shader")
    render_primitive = moderngl.TRIANGLE_STRIP

    def __init__(
        self,
        circle,
        sigma=0.11,
        intensity=1.2,
        color=None,
        follow=True,
        padding_sigmas=9.0,
        **kwargs
    ):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if padding_sigmas < 4.0:
            raise ValueError("padding_sigmas should be >= 4 for soft edges")
        self.source_circle = circle
        self.sigma = float(sigma)
        self.intensity = float(intensity)
        self.padding_sigmas = float(padding_sigmas)
        self.follow_color = color is None
        self._last_state = None
        self._last_color = circle.get_stroke_color() if color is None else color

        super().__init__(color=self._last_color, opacity=1.0, **kwargs)

        # Updating at the *source* avoids a one-frame lag when source Circle
        # already has geometric updaters. Add other circle updaters first.
        if follow:
            def source_updater(mob):
                self.sync_to_circle()
            self._source_updater = source_updater
            circle.add_updater(source_updater)

    def init_uniforms(self):
        super().init_uniforms()
        self.uniforms.update({
            "glow_center": np.array([0.0, 0.0], dtype=np.float32),
            "glow_radius": 1.0,
            "glow_sigma": self.sigma,
            "glow_intensity": self.intensity,
        })

    def init_points(self):
        self.sync_to_circle(force=True)

    def sync_to_circle(self, force=False):
        c = self.source_circle
        center = np.asarray(c.get_center(), dtype=float)
        radius = float(c.get_radius())

        if self.follow_color:
            new_color = c.get_stroke_color()
            if new_color != self._last_color:
                self.set_color(new_color)
                self._last_color = new_color

        if not (np.all(np.isfinite(center)) and np.isfinite(radius)) or radius <= 0:
            if self.get_num_points():
                self.set_points(np.zeros((0, 3), dtype=np.float32))
            self._last_state = None
            return self

        state = (float(center[0]), float(center[1]), float(center[2]), radius)
        if not force and self._last_state == state:
            return self
        self._last_state = state
        # set_uniform refreshes ManimGL's shader batch identity as well.
        self.set_uniform(
            glow_center=np.array(center[:2], dtype=np.float32),
            glow_radius=radius,
        )

        # Triangle strip vertex order: BL, BR, TL, TR.
        half = radius + self.padding_sigmas * self.sigma
        x, y, z = center
        points = np.array([
            [x - half, y - half, z],
            [x + half, y - half, z],
            [x - half, y + half, z],
            [x + half, y + half, z],
        ], dtype=np.float32)
        self.set_points(points)
        return self

    def set_glow(self, sigma=None, intensity=None):
        if sigma is not None:
            if sigma <= 0:
                raise ValueError("sigma must be positive")
            self.sigma = float(sigma)
            self.uniforms["glow_sigma"] = self.sigma
            self.sync_to_circle(force=True)
        if intensity is not None:
            self.intensity = float(intensity)
            self.uniforms["glow_intensity"] = self.intensity
        # Mobject.set_uniform manages the shader batch identity.
        self.refresh_shader_wrapper_id()
        return self


# Things about references that borrowed from previous projects
class Reference(VGroup):
    name_color = BLUE
    author_color = BLUE_A
    pub_color = GREY_A
    doi_color = GREY_A
    info_color = GOLD_A

    def __init__(
        self,
        name='Name',
        authors='Authors',
        pub='Publication',
        doi='doi',
        info='',
        **kwargs,
    ):
        super().__init__(**kwargs)

        textext_config = {'alignment': ''}
        text_config = {'font': 'CMU Sans Serif'}

        text_types = [TexText, TexText, TexText, Text, TexText]
        text_configs = [
            textext_config if text_type == TexText else text_config
            for text_type in text_types
        ]
        texts = [name, authors, pub, doi, info]
        colors = [self.name_color, self.author_color, self.pub_color, self.doi_color, self.info_color]
        scale_factors = [1, 0.7, 0.6, 0.6, 0.7]
        texs = VGroup(*[
            TextType(text, **config).scale(factor).set_color(color)
            for TextType, config, text, color, factor in zip(
                text_types, text_configs, texts, colors, scale_factors,
            )
        ])
        texs.arrange(DOWN, aligned_edge=LEFT)
        texs[2:-1].shift(0.5 * RIGHT)
        texs.scale(0.8)
        self.add(*texs)


class ReferenceScene(Scene):
    pause_time_between_pages = 5

    def construct(self):
        refs = self.get_references()
        if len(refs) == 0:
            return
        for ref, y_pos in zip(refs, it.cycle([3.2, -0.5])):
            ref.center().next_to(y_pos * UP, DOWN, buff=0).to_edge(LEFT, buff=1)
        num_of_groups = (len(refs) + 1) // 2
        ref_groups = [
            VGroup(*refs[2 * k:2 * k + 2])
            for k in range(num_of_groups)
        ]
        curr_group = None
        for group in ref_groups:
            if not curr_group:
                self.play(FadeIn(group), run_time=0.5)
            else:
                self.play(FadeOut(curr_group), run_time=0.5)
                self.play(FadeIn(group), run_time=0.5)
            curr_group = group
            self.wait(self.pause_time_between_pages)
        self.play(FadeOut(ref_groups[-1]), run_time=0.5)
        self.wait()

    def get_references(self):
        return ()


# Scenes
class IntroToDescartesLinAlgProof(Scene):
    def construct(self):
        # What will you think of when you see 'linear algebra'
        linalg_text = TexText("线性代数")
        linalg_text.set_color(YELLOW).scale(2)
        self.play(Write(linalg_text))
        self.wait()

        mat = [
            [3, 2],
            [-2, 1],
        ]
        matrix = TexMatrix(mat)
        matrix.get_column(0).set_color(GREEN)
        matrix.get_column(1).set_color(RED)
        eq_matrix = matrix.copy()
        arrow = Vector(UR)
        arrow.set_color(YELLOW)
        arrow.save_state()
        VGroup(matrix, arrow).arrange(RIGHT, buff=1.5).scale(1.5).center()
        self.play(
            FadeOut(linalg_text),
            FadeIn(matrix, shift=LEFT),
            FadeIn(arrow, shift=RIGHT),
        )
        self.wait()

        eq_one = Tex("1").set_color(YELLOW)
        eq_plus = Tex("+")
        eq_equal = Tex("=")
        col1 = TexMatrix([[3], [-2]])
        col1.get_columns().set_color(GREEN)
        col2 = TexMatrix([[2], [1]])
        col2.get_columns().set_color(RED)
        eq_start_vec = TexMatrix([[1], [1]])
        eq_start_vec.get_columns().set_color(YELLOW)
        eq_end_vec = TexMatrix([[5], [-1]])
        eq_end_vec.get_columns().set_color(YELLOW)
        equation = VGroup(
            eq_matrix, eq_start_vec, eq_equal.copy(),
            eq_one.copy(), col1, eq_plus, eq_one.copy(), col2,
            eq_equal.copy(), eq_end_vec
        )
        equation.arrange(RIGHT).to_corner(UL, buff=0.2)
        back_rect = BackgroundRectangle(equation, buff=0.2)
        plane = NumberPlane(
            y_range=(-8.0, 8.0, 1.0),
            faded_line_ratio=0,
        )
        basis_i = Vector(RIGHT).set_color(GREEN)
        basis_j = Vector(UP).set_color(RED)
        self.add(plane, back_rect, matrix)
        self.play(
            FadeIn(back_rect),
            ShowCreation(plane, lag_ratio=0.05),
            ReplacementTransform(matrix, equation[0]),
            Restore(arrow),
            GrowFromPoint(eq_start_vec, arrow.get_center()),
            GrowArrow(basis_i),
            GrowArrow(basis_j),
            run_time=2,
        )
        self.wait()

        trans_plane = plane.copy()
        plane.set_opacity(0.2)
        arrow_shadow = arrow.copy().set_opacity(0)
        arrow.add_updater(lambda m: m.put_start_and_end_on(*arrow_shadow.get_start_and_end()))
        basis_i_shadow = basis_i.copy().set_opacity(0)
        basis_i.add_updater(lambda m: m.put_start_and_end_on(*basis_i_shadow.get_start_and_end()))
        basis_j_shadow = basis_j.copy().set_opacity(0)
        basis_j.add_updater(lambda m: m.put_start_and_end_on(*basis_j_shadow.get_start_and_end()))
        self.add(
            trans_plane, plane, back_rect, equation,
            arrow, basis_i, basis_j,
        )
        self.play(
            *[
                ApplyMethod(mob.apply_matrix, mat)
                for mob in (trans_plane, arrow_shadow, basis_i_shadow, basis_j_shadow)
            ],
            Write(equation[2:]),
            run_time=3,
        )
        self.wait()

        # We'll handle 'round' with linear algebra
        fade_group = VGroup(
            trans_plane, back_rect, equation,
            arrow, basis_i, basis_j,
        )
        circle = Circle()
        circle.move_to(LEFT_SIDE / 2)
        pedoe_vector = MobjectMatrix(
            [Tex(text) for text in ["k", "t", "kx", "ky"]],
            n_cols=1,
        )
        pedoe_vector.move_to(RIGHT_SIDE / 2)
        left_side_rect = FullScreenFadeRectangle(fill_opacity=1)
        left_side_rect.stretch(0.5, 0).to_edge(LEFT, buff=0)
        self.play(
            *[FadeOut(mob, run_time=1) for mob in fade_group],
            FadeIn(left_side_rect, run_time=1),
            AnimationGroup(
                GrowFromCenter(circle),
                GrowFromCenter(pedoe_vector),
                run_time=2,
            )
        )
        self.remove(arrow_shadow, basis_i_shadow, basis_j_shadow)
        self.wait()

        # Most seemingly unrelated objects ...
        circle.save_state()
        circle.generate_target()
        circle.target.scale(1.5).set_color(YELLOW)
        self.play(MoveToTarget(circle))
        self.wait()

        # ... can also be put in a linear algebra world
        pedoe_vector.generate_target()
        pedoe_vector.target.scale(1.5).set_color(YELLOW)
        arrow = Arrow(
            circle.target.get_right(), pedoe_vector.target.get_left(),
            thickness=5, buff=0.5,
        )
        arrow.set_color(YELLOW)
        self.play(
            Restore(circle),
            GrowArrow(arrow),
            MoveToTarget(pedoe_vector),
        )
        self.wait()

        # Descartes' Theorem
        # 1. Left side
        config_mat = generate_descartes_matrix(random_seed=570)
        c1, c2, c3, c4 = c_group = VGroup(
            VCircle(plane, raw_vector=config_mat[:, ind]).set_color(color)
            for ind, color in zip(range(4), CIRCLE_COLORS)
        )
        c_group.set_height(FRAME_HEIGHT * 0.75).move_to(LEFT_SIDE / 2)
        for c in c_group[1:]:
            c.save_state()
            c.set_opacity(0)
            c.scale(1e-2 if c is not c4 else 1.5)
        # 2. Right side
        descartes_relation = VGroup(
            Tex(text, t2c=CURVATURE_T2C, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2}",
                "= 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)"
            ]
        )
        descartes_relation.arrange(DOWN, aligned_edge=RIGHT)
        descartes_title = VGroup(
            TexText("笛卡尔定理"),
            TexText("(Descartes' Theorem)"),
        )
        descartes_title.set_color(YELLOW)
        descartes_title[1].scale(0.6)
        descartes_title.arrange(DOWN, buff=0.1)
        descartes_group = VGroup(descartes_title, descartes_relation)
        descartes_group.arrange(DOWN)
        descartes_group.scale(0.9)
        descartes_group.move_to((TOP + RIGHT_SIDE) / 2)
        notation_k = TexText("($k$：有向曲率)")
        notation_k.scale(0.75).set_color(GREY)
        notation_k.move_to(RIGHT_SIDE / 2)
        soddy_gosset_relation = VGroup(
            Tex(text, t2c=CURVATURE_T2C, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + \\cdots + k_{n+2} \\right)^{2}",
                "= n \\left( k_{1}^{2} + k_{2}^{2} + \\cdots + k_{n+2}^{2} \\right)"
            ]
        )
        soddy_gosset_relation[1][-5].set_color(WHITE)  # color fixing
        soddy_gosset_relation[1][-2].set_color(SPHERE_COLORS[-1])  # color fixing
        soddy_gosset_relation.arrange(DOWN, aligned_edge=RIGHT)
        soddy_gosset_title = VGroup(
            TexText("索迪-戈赛特定理"),
            TexText("(Soddy-Gosset Theorem)"),
        )
        soddy_gosset_title.set_color(YELLOW)
        soddy_gosset_title[1].scale(0.6)
        soddy_gosset_title.arrange(DOWN, buff=0.1)
        soddy_gosset_group = VGroup(soddy_gosset_title, soddy_gosset_relation)
        soddy_gosset_group.arrange(DOWN)
        soddy_gosset_group.scale(0.9)
        soddy_gosset_group.move_to((BOTTOM + RIGHT_SIDE) / 2)

        self.play(
            AnimationGroup(
                ReplacementTransform(circle, c1[0]),
                *[Restore(c) for c in c_group[1:]],
                lag_ratio=0.1,
                run_time=1,
            ),
            AnimationGroup(
                FadeOut(arrow),
                FadeOut(pedoe_vector),
                FadeIn(descartes_group),
                FadeIn(notation_k),
                run_time=1,
            )
        )
        self.add(c1, c2, c3, c4)
        self.wait()

        # And Soddy-Gosset Theorem, with Blender-rendered scene
        self.play(
            FadeIn(soddy_gosset_group),
            FadeOut(c_group),
            run_time=1,
        )
        self.wait()

        # Prerequisite knowledge
        basics_title = TexText("你需要了解矩阵的...")
        basics_title.set_color(BLUE_D)
        basics_list = BulletedList(
            "乘法运算规则",
            "转置",
            "逆",
            "初等变换",
        )
        basics_list.scale(0.8)
        basics_group = VGroup(basics_title, basics_list)
        basics_group.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        basics_group.move_to(LEFT_SIDE / 2)
        self.play(
            AnimationGroup(
                FadeIn(basics_title, shift=RIGHT),
                *[FadeIn(text, shift=RIGHT) for text in basics_list],
                lag_ratio=0.05,
                run_time=1.5,
            ),
        )
        self.wait()
        for text in basics_list:
            text.generate_target()
            text.target.scale(1.2, about_edge=LEFT)
            text.target.set_color(TEAL)
        self.play(
            AnimationGroup(
                *[MoveToTarget(text, rate_func=there_and_back) for text in basics_list],
                lag_ratio=0.05,
                run_time=1.5,
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                FadeOut(basics_title, shift=LEFT),
                *[FadeOut(text, shift=LEFT) for text in basics_list],
                lag_ratio=0.05,
                run_time=1.5,
            ),
        )
        self.wait()


class WhatIsDescartesTheoremP1(Scene):
    def construct(self):
        # Setup
        circle = Circle(radius=2)
        center_dot = Dot()
        center_dot.add_updater(lambda m: m.move_to(circle.get_center()))
        example_group = VGroup(center_dot, circle)
        example_group.set_color(WHITE)
        example_group.shift(DOWN)
        self.add(example_group)
        self.wait()

        # Circle, radius and curvature
        radial_line = DashedLine()
        radial_line.add_updater(
            lambda m: m.become(
                DashedLine(center_dot.get_center(), circle.get_top())
                .set_color(YELLOW)
            )
        )
        radial_line.update().suspend_updating()
        self.play(
            Succession(
                GrowFromPoint(radial_line, center_dot.get_center(), run_time=1),
                Rotating(
                    radial_line,
                    angle=2 * PI,
                    about_point=center_dot.get_center(),
                    rate_func=smooth,
                    run_time=1.5,
                ),
            ),
        )
        radial_line.resume_updating()
        self.wait()

        value_text_config = {
            "num_decimal_places": 4,
            "text_config": CMU_TEXT_CONFIG,
        }
        radius_text = TexText("半径$r = $")
        radius_value = DecimalNumber(0, **value_text_config)
        radius_value.add_updater(
            lambda m: m.set_value(circle.get_height() / 2)
            .next_to(radius_text, RIGHT)
        )
        radius_group = VGroup(radius_text, radius_value)
        radius_group.set_color(YELLOW)
        curvature_text = TexText("曲率$k = \\dfrac{1}{r} = $")
        curvature_value = DecimalNumber(0, **value_text_config)
        curvature_value.add_updater(
            lambda m: m.set_value(1 / radius_value.get_value())
            .next_to(curvature_text, RIGHT)
        )
        curvature_group = VGroup(curvature_text, curvature_value)
        curvature_group.set_color(GOLD)
        group = VGroup(radius_group, curvature_group)
        group.arrange(DOWN, aligned_edge=LEFT)
        group.to_edge(UP)

        r_tex = Tex("r")
        r_tex.set_color(YELLOW)
        r_tex.add_updater(lambda m: m.next_to(radial_line, RIGHT))
        r_tex.update().suspend_updating()
        self.play(
            FadeIn(radius_group, shift=0.5 * UP),
            Write(r_tex),
        )
        r_tex.resume_updating()
        self.wait()
        self.play(
            FadeIn(curvature_group, shift=0.5 * UP),
        )
        self.wait()

        circle.save_state()
        self.play(
            ApplyMethod(circle.scale, 3, {"about_point": circle.get_top()}),
            run_time=2,
        )
        self.wait()
        self.play(
            Restore(circle),
            run_time=2,
        )
        fade_mobs = Group(*self.mobjects)
        fade_mobs.remove(circle)
        for mob in fade_mobs:
            mob.clear_updaters()
        self.wait()

        # Now imagine 4 pairwise tangent (externally) circles
        tangent_circles = TangentFourCircles(
            ORIGIN,
            UP,
            1.5 * RIGHT,
        )
        tangent_circles.suspend_updating()
        tangent_circles.rotate(PI / 5).center().scale(2)
        tangent_circles.update_centers_from_circles()
        tangent_circles.resume_updating()
        c1, c2, c3, c4 = tangent_circles.get_circles()
        for c in c2, c3, c4:
            c.save_state()
            c.scale(0)
            c.set_opacity(0)

        self.add(c2, c3, c4)
        self.play(
            FadeOut(fade_mobs, run_time=1),
            ReplacementTransform(circle, c1, run_time=2),
            *[Restore(c, run_time=2) for c in (c2, c3, c4)],
        )
        self.add(tangent_circles)
        self.wait()

        four_circles_text = TexText("“四圆互切”")
        four_circles_text.scale(1.2)
        four_circles_text.set_color(YELLOW)
        four_circles_text.to_edge(UP)
        self.play(Write(four_circles_text))
        self.wait()

        # 6 tangent points should be different
        tangent_points = tangent_circles.get_tangent_points()
        self.play(
            *[
                Flash(p, line_length=0.05, flash_radius=0.1, line_stroke_width=3)
                for p in tangent_points
            ],
            run_time=2,
        )
        self.wait()

        # There're plenty of such configuration, but one relation always holds
        curvature_notes = VGroup(
            TexText(f"曲率$k_{sub + 1}=$").set_color(color)
            for sub, color in enumerate(CIRCLE_COLORS)
        )
        curvature_notes.arrange(DOWN, aligned_edge=LEFT)
        curvature_notes.next_to((2, 0, 0), RIGHT, buff=0)
        curvature_values = VGroup(
            DecimalNumber(
                0, show_ellipsis=True,
                num_decimal_places=5, text_config=CMU_TEXT_CONFIG,
            )
            .set_color(color)
            .next_to(note, RIGHT)
            for color, note in zip(CIRCLE_COLORS, curvature_notes)
        )

        def make_temp_updater(circle):
            def updater(mob):
                mob.set_value(2 / circle.get_height())
            return updater
        for value, note, circle in zip(curvature_values, curvature_notes, tangent_circles.get_circles()):
            value.add_updater(make_temp_updater(circle))

        p1, p2, p3 = base_points = tangent_circles.base_points
        target_positions = [
            (-3.8, -2, 0),
            (-4.8, 0.5, 0),
            (-1.3, 0, 0),
        ]
        for p, pos in zip(base_points, target_positions):
            p.generate_target()
            p.target.move_to(pos)
        self.play(
            *[MoveToTarget(p) for p in base_points],
            *[FadeIn(note[:-1], shift=1.5 * LEFT) for note in curvature_notes],
            run_time=3,
        )
        self.wait()
        curvature_values.update().suspend_updating()
        self.play(
            *[FadeIn(note[-1]) for note in curvature_notes],
            FadeIn(curvature_values),
        )
        curvature_values.resume_updating()
        self.wait()

        # Wiggle a bit
        anchors = [p.get_center().copy() for p in base_points]
        wiggle_params = [
            # ax, ay, wx, wy
            (1.00, 0.75, -0.73, 1.07),
            (0.82, 1.00, 0.91, -0.67),
            (0.93, 0.81, 0.61, 0.83),
        ]
        amplitude = 0.2
        duration = 10
        fade_time = 2

        def get_envelope(t):
            # Fade-in part
            if t < fade_time:
                alpha = t / fade_time
                return smooth(alpha)
            # Fade-out part
            if t > duration - fade_time:
                alpha = (duration - t) / fade_time
                return smooth(max(0, alpha))
            return 1.0

        def update_base_points(mob, dt):
            mob.increment_value(dt)
            t = mob.get_value()
            for point, anchor, par in zip(base_points, anchors, wiggle_params):
                envelope = get_envelope(t)
                ax, ay, wx, wy = par
                dx = envelope * amplitude * ax * np.sin(wx * t)
                dy = envelope * amplitude * ay * np.sin(wy * t)
                point.move_to(anchor + dx * RIGHT + dy * UP)

        wiggle_movement_driver = ValueTracker(0)
        wiggle_movement_driver.add_updater(update_base_points)
        self.add(wiggle_movement_driver)
        self.wait(duration)
        self.remove(wiggle_movement_driver)
        tangent_circles.refresh()

        # Show Descartes' Theorem
        relation = Tex(
            """
            \\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2}
            = 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)
            """,
            t2c=CURVATURE_T2C,
            use_labelled_svg=False,
        )
        relation.to_edge(UP)
        self.play(FadeTransform(four_circles_text, relation))
        self.wait()

        # Origin of the theorem
        cover_rect = FullScreenFadeRectangle(fill_opacity=1)
        cover_rect.next_to(relation, DOWN)
        descartes_portrait = ImageMobject("Rene_Descartes.jpg")
        theorem_origin = ImageMobject("DescartesTheoremOrigin.png")
        elisabeth_portrait = ImageMobject("1636_Elisabeth_of_Bohemia.jpg")
        image_group = Group(
            descartes_portrait, theorem_origin, elisabeth_portrait,
        )
        for image in image_group:
            image.set_height(4)
        image_group.arrange(RIGHT, buff=0.5)
        image_group.center()
        image_notes = [
            "勒内·笛卡尔",
            "\\emph{Oeuvres de Descartes. Correspondence, IV}",
            "伊丽莎白公主"
        ]
        image_remarks = VGroup(
            RemarkText(ref, text)
            for ref, text in zip(image_group, image_notes)
        )
        descartes_remark, theorem_remark, elisabeth_remark = image_remarks
        self.play(
            FadeIn(cover_rect, run_time=1),
            AnimationGroup(
                *[
                    FadeIn(image, shift=UP)
                    for image in (descartes_portrait, elisabeth_portrait, theorem_origin)
                ],
                lag_ratio=0.3,
                run_time=2,
            ),
            AnimationGroup(
                *[
                    FadeIn(remark, shift=UP)
                    for remark in (descartes_remark, elisabeth_remark, theorem_remark)
                ],
                lag_ratio=0.3,
                run_time=2,
            ),
        )
        self.remove(tangent_circles, curvature_notes, curvature_values, cover_rect)
        self.wait()

        example_rect = Rectangle()
        example_rect.set_color(PINK)
        example_rect.set_width(theorem_origin.get_width() / 2, stretch=True)
        example_rect.set_height(theorem_origin.get_width() / 7, stretch=True)
        example_rect.next_to(theorem_origin.get_right(), LEFT, buff=0)
        example_rect.shift(0.1 * DOWN)
        self.play(ShowCreationThenDestruction(example_rect), run_time=2)
        self.wait()

        # Original form involves a lot of radii, which is a bit messy
        RADII_T2C = dict([
            (radius, color)
            for radius, color in zip(list("defx"), CIRCLE_COLORS)
        ])
        lhs = VGroup(
            Tex(text, t2c=RADII_T2C)
            for text in ["ddeeff", "+ ddeexx", "+ ddffxx", "+ eeffxx"]
        )
        equal_sign = Tex("=")
        rhs = VGroup(
            Tex(text, t2c=RADII_T2C)
            for text in [
                "2deffxx + 2deeffx",
                "+ 2deefxx + 2ddeffx",
                "+ 2ddefxx + 2ddeefx",
            ]
        )
        for part in (lhs, rhs):
            part.arrange(DOWN, aligned_edge=RIGHT)
        old_relation = VGroup(lhs, equal_sign, rhs)
        old_relation.arrange(RIGHT, buff=0.7)
        old_relation.next_to(descartes_portrait, RIGHT, buff=0.7)
        old_relation_remark = TexText(
            "（$d$, $e$, $f$, $x$分别为四个圆的\\underline{半径}）",
            t2c=RADII_T2C,
        )
        old_relation_remark.next_to(old_relation, DOWN, buff=0.5)
        old_relation.generate_target()
        old_relation.set_width(theorem_origin.get_width() / 4)
        old_relation.next_to(example_rect, UP, buff=0.6)
        old_relation.set_opacity(0)
        self.play(
            AnimationGroup(
                *[
                    FadeOut(mob)
                    for mob in (
                        theorem_origin, theorem_remark,
                        elisabeth_portrait, elisabeth_remark,
                    )
                ],
            ),
            MoveToTarget(old_relation),
            run_time=1,
        )
        self.wait()
        self.play(FadeIn(old_relation_remark), run_time=1)
        self.wait()

        # So the theorem was named after him
        alt_relation = Tex(
            """
            \\left( \\dfrac{1}{d} + \\dfrac{1}{e} + \\dfrac{1}{f} + \\dfrac{1}{x} \\right)^{2}
            = 2 \\left( \\dfrac{1}{d^2} + \\dfrac{1}{e^2} + \\dfrac{1}{f^2} + \\dfrac{1}{x^2} \\right)
            """,
            t2c=RADII_T2C,
            use_labelled_svg=False,
        )
        relation_name = TexText("笛卡尔定理：")
        relation_name.set_color(YELLOW)
        named_relation = VGroup(relation_name, alt_relation)
        named_relation.arrange(RIGHT, buff=0.5)
        named_relation.to_edge(UP)
        self.play(
            ReplacementTransform(descartes_remark[3:6].copy(), relation_name[:3]),
            Write(relation_name[3:], stroke_color=YELLOW),
            ReplacementTransform(relation[:13:3], alt_relation[:17:4]),
            ReplacementTransform(relation[13:16], alt_relation[17:20]),
            ReplacementTransform(relation[16::4], alt_relation[20::5]),
            *[
                FadeTransform(relation[3 * step + 1:3 * step + 3], alt_relation[4 * step + 1:4 * step + 4])
                for step in range(4)
            ],
            *[
                FadeTransform(relation[4 * step + 17:4 * step + 20:2], alt_relation[5 * step + 21:5 * step + 24])
                for step in range(4)
            ],
            *[
                ReplacementTransform(relation[4 * step + 18], alt_relation[5 * step + 24])
                for step in range(4)
            ],
            run_time=2,
        )
        self.wait()

        # This relation was rediscovered multiple times
        fade_descartes_group = Group(
            descartes_portrait, descartes_remark,
            old_relation, old_relation_remark,
        )

        sangaku_algorithm = ImageMobject("ShinpekiSanpo_1789.png")
        beecroft_proof = ImageMobject("Beecroft_1842.png")
        soddy_poem = ImageMobject("PoemSoddy_1936.png")
        gosset_poem = ImageMobject("PoemGosset_1936.png")
        sg_poem = Group(soddy_poem, gosset_poem)
        gosset_poem.set_width(soddy_poem.get_width() / 2)
        sg_poem.arrange(DOWN)
        rediscovery_images = Group(sangaku_algorithm, beecroft_proof, sg_poem)
        for image in rediscovery_images:
            image.set_height(4.5)
        rediscovery_images.arrange(RIGHT, buff=0.8)
        rediscovery_remarks = VGroup(
            RemarkText(mob, year, scale_factor=0.6)
            for mob, year in zip(
                [sangaku_algorithm, beecroft_proof, gosset_poem],
                [
                    "1789年 \\\\ 神壁算法 \\\\[0.5em] （图源：京都大学理学 \\\\ 研究科数学教室）",
                    "1842年 \\\\ \\emph{Properties of Circles \\\\ in Mutual Contact}",
                    "1936年 \\\\ \\emph{The Kiss Precise}",
                ]
            )
        )
        rediscovery_group = Group(rediscovery_images, rediscovery_remarks)
        rediscovery_group.shift(UP)

        author_extra = TexText("Frederick Soddy, Thorold Gosset")
        author_extra.scale(0.6)
        author_extra.match_color(rediscovery_remarks[-1])
        author_extra.next_to(rediscovery_remarks[-1], DOWN)
        year_extra = TexText("1936-1937年")
        year_ref = rediscovery_remarks[-1][:5]
        year_extra.match_color(year_ref[0])
        year_extra.match_height(year_ref)
        year_extra.move_to(year_ref)

        sg_poem.remove(gosset_poem)
        self.play(
            FadeOut(named_relation, shift=UP),
            FadeOut(fade_descartes_group, shift=UP),
            FadeIn(rediscovery_group, shift=UP),
        )
        self.wait()

        # The poem in 1936 is the most famous one
        self.play(
            Indicate(soddy_poem),
            Indicate(rediscovery_remarks[-1]),
        )
        self.wait()

        SODDY_GOSSET_POEM = [
            [
                "双唇相吻寻常事，何必三角费思量。",
                "四圆相切若同框，彼此相拥不相让。",
                "排列无非分两类，三围一圆或反向：",
                "若是三圆围一圆，外部相触理应当；",
                "若是一个吞三个，大圆内侧吻三趟。",
            ],
            [
                "四圆聚首齐相看，个头越小弧越卷。",
                "圆心至缘量半径，取其倒数便为弯。",
                "纵使欧氏亦无言，如今法则已通关。",
            ],
            [
                "直线弯度定为零，内凹弯度负号生。",
                "四弯平方加一起，和之平方折半成。",
            ],
            [
                "若看球体更风流，密切测量却犯愁。",
                "球中亲吻添热闹，两对之外又添球。",
                "正负零值同前理，每球环顾吻四周。",
                "五弯总和之平方，三倍平方和相投。",
            ],
            [
                "莫将眼界囿于此，圆圈球体嫌太小。",
                "且向高维空间探，超平超弯任逍遥。",
                "$n$维空间超球舞，几何真理已宣告：",
                "$(n+2)$个同拥吻，各与$(n+1)$相靠。",
                "弯度总和之平方，$n$倍平方和正好。",
            ]
        ]

        poem_title = TexText("精确之吻")
        poem_title.set_color(YELLOW)
        left_verses = VGroup(
            TexText(" \\\\[0.5em] ".join(texts))
            for texts in SODDY_GOSSET_POEM[:3]
        )
        left_verses.arrange(DOWN, buff=1)
        right_verses = VGroup(
            TexText(" \\\\[0.5em] ".join(texts))
            for texts in SODDY_GOSSET_POEM[3:]
        )
        right_verses.arrange(DOWN, buff=1)
        right_verses.next_to(left_verses, RIGHT, buff=1, aligned_edge=UP)
        verses = VGroup(left_verses, right_verses)
        verses.set_width(FRAME_WIDTH * 0.5)

        verses.to_edge(LEFT, buff=0.5)
        poem_title.next_to(verses, UP, buff=0.5)

        translators = TexText("（翻译: Gemini 3.1 Pro, GPT-5.6 Sol）")
        translators.set_color(GREY)
        translators.scale(0.4)
        translators.next_to(verses, DOWN, aligned_edge=LEFT, buff=0.5)

        self.play(
            *[
                FadeOut(mob)
                for mob in (rediscovery_images[:-1], rediscovery_remarks[:-1])
            ],
            FadeIn(author_extra[:len("FrederickSoddy")]),
            FadeIn(poem_title),
            FadeIn(left_verses),
            FadeIn(right_verses[0]),
            FadeIn(translators),
        )
        self.wait()

        highlight_color = BLUE_B
        underline_config = {
            "stretch_factor": 1,
            "buff": 0.05,
            "stroke_color": highlight_color,
        }
        underline_2d = Underline(left_verses[-1], **underline_config)
        underline_3d = Underline(right_verses[0], **underline_config)
        self.play(
            ShowCreationThenDestruction(underline_2d),
            ShowCreationThenDestruction(underline_3d),
            left_verses[-1][-16:].animate.set_color(highlight_color),
            right_verses[0][-16:].animate.set_color(highlight_color),
            run_time=2,
        )
        self.wait()

        # And Gosset added another verse
        self.play(
            ReplacementTransform(year_ref[:-1], year_extra[:4]),
            ReplacementTransform(VectorizedPoint(year_ref[-1].get_left()), year_extra[4:-1]),
            ReplacementTransform(year_ref[-1], year_extra[-1]),
            FadeIn(gosset_poem),
            FadeIn(author_extra[len("FrederickSoddy"):]),
            FadeIn(right_verses[1]),
        )
        sg_poem.add(gosset_poem)
        self.wait()
        underline_nd = Underline(right_verses[1], **underline_config)
        self.play(
            ShowCreationThenDestruction(underline_nd),
            right_verses[1][-16:].animate.set_color(highlight_color),
            run_time=1.5,
        )
        self.wait()

        # And here comes Soddy-Gosset theorem
        soddy_portrait = ImageMobject("Frederick_Soddy.jpg")
        gosset_portrait = ImageMobject("unnamed.jpg")
        gosset_portrait.set_opacity(0.5)
        sg_portraits = Group(soddy_portrait, gosset_portrait)
        sg_portraits.arrange(RIGHT)
        sg_portraits.match_width(author_extra)
        sg_portraits.next_to(author_extra, UP)

        sg_theorem = VGroup(
            Tex(text, t2c=CURVATURE_T2C, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + \\cdots + k_{n+2} \\right)^{2}",
                "= n \\left( k_{1}^{2} + k_{2}^{2} + \\cdots + k_{n+2}^{2} \\right)"
            ]
        )
        sg_theorem[1][-5].set_color(WHITE)  # color fixing
        sg_theorem[1][-2].set_color(SPHERE_COLORS[-1])  # color fixing
        sg_theorem.arrange(DOWN, aligned_edge=RIGHT)
        sg_title = TexText("索迪-戈赛特定理")
        sg_title.set_color(YELLOW)
        sg_group = VGroup(sg_title, sg_theorem)
        sg_group.arrange(DOWN)
        sg_group.scale(0.9)
        sg_group.next_to(sg_portraits, UP, buff=0.5)
        sg_group.generate_target()
        sg_group.set_opacity(0).match_width(right_verses[1])
        sg_group.move_to(right_verses[1])

        self.play(
            FadeOut(sg_poem),
            FadeOut(year_extra),
            FadeOut(rediscovery_remarks[-1][5:]),
            MoveToTarget(sg_group),
            run_time=2,
        )
        self.wait()
        self.play(FadeIn(soddy_portrait, shift=UP))
        self.wait()
        self.play(FadeIn(gosset_portrait, shift=UP))
        self.wait()


class WhatIsDescartesTheoremP2(Scene):
    def construct(self):
        # Setup a typical configuration
        curvatures_group = VGroup(
            VGroup(
                TexText(f"有向曲率$k_{{{i + 1}}} = {k}$").set_color(color)
                for i, (k, color) in enumerate(
                    zip(k_list, CIRCLE_COLORS)
                )
            )
            for k_list in ([2, 2, 3, 15], [2, 2, 3, 1], [2, 2, 3, -1])
        )
        curvatures_external, bad_curvatures_enclosing, curvatures_enclosing = curvatures_group
        for curvatures in curvatures_group:
            for text in curvatures:
                text[:2].set_opacity(0)
            curvatures.arrange(DOWN, aligned_edge=LEFT)
            curvatures.next_to(RIGHT_SIDE / 2, UP, buff=0.5)
        for text in curvatures_enclosing:
            text[:2].set_opacity(1)
            text[:2].set_color(YELLOW)
        relation_external = Tex(
            "\\left( 2 + 2 + 3 + 15 \\right)^{2} \\\\ = 2 \\left( 2^{2} + 2^{2} + 3^{2} + 15^{2} \\right)"
        )
        bad_relation_enclosing = Tex(
            "\\left( 2 + 2 + 3 + 1 \\right)^{2} \\\\ = 2 \\left( 2^{2} + 2^{2} + 3^{2} + 1^{2} \\right)"
        )
        relation_enclosing = Tex(
            "\\left( 2 + 2 + 3 + (-1) \\right)^{2} \\\\ = 2 \\left( 2^{2} + 2^{2} + 3^{2} + (-1)^{2} \\right)"
        )
        relation_group = VGroup(relation_external, bad_relation_enclosing, relation_enclosing)
        for relation in relation_group:
            relation.next_to(RIGHT_SIDE / 2, DOWN, buff=0.5)
        for relation, relation_indices in zip(
            relation_group,
            (
                [[1, 14], [3, 17], [5, 20], [7, 8, 23, 24]],
                [[1, 13], [3, 16], [5, 19], [7, 22]],
                [[1, 16], [3, 19], [5, 22], [7, 8, 9, 10, 25, 26, 27, 28]],
            ),
        ):
            for indices, color in zip(relation_indices, CIRCLE_COLORS):
                for ind in indices:
                    relation[ind].set_color(color)
        circles_external = TangentFourCircles(
            1 / 2 * LEFT, 1 / 2 * RIGHT, 2 / 3 * UP,
            fourth_type="smaller",
        )
        circles_external.suspend_updating()
        circles_external.set_width(6)
        circles_external.to_edge(LEFT, buff=1)
        circles_external.next_to(curvatures_external.get_top(), DOWN, buff=0, coor_mask=[0, 1, 0])
        circles_external.update_centers_from_circles()
        circles_external.refresh()
        circles_external.resume_updating()
        p1, p2, p3 = circles_external.get_base_centers()
        self.add(circles_external, curvatures_external, relation_external)
        self.wait()

        # But there's a different type of configuration - enclosing
        circles_enclosing = TangentFourCircles(
            p1, p2, p3, fourth_type="bigger",
        )
        for mob in (circles_external, circles_enclosing):
            mob.suspend_updating()
        self.play(
            AnimationGroup(
                FadeOut(curvatures_external[-1][-1]),
                FadeOut(relation_external[8]),
                FadeOut(relation_external[24]),
                run_time=1,
            ),
            AnimationGroup(
                *[
                    ReplacementTransform(source, target)
                    for source, target in zip(curvatures_external[:-1], bad_curvatures_enclosing[:-1])
                ],
                ReplacementTransform(curvatures_external[-1][:-1], bad_curvatures_enclosing[-1]),
                ReplacementTransform(relation_external[:8], bad_relation_enclosing[:8]),
                ReplacementTransform(relation_external[9:24], bad_relation_enclosing[8:23]),
                ReplacementTransform(relation_external[25:], bad_relation_enclosing[23:]),
                ReplacementTransform(circles_external, circles_enclosing),
                run_time=2,
            )
        )
        self.remove(curvatures_external, relation_external)
        self.add(bad_curvatures_enclosing, bad_relation_enclosing)
        self.wait()

        # Absurd result!
        question_mark = TexText("?????")
        question_mark.scale(1.5)
        question_mark.set_color(YELLOW)
        question_mark.next_to(bad_relation_enclosing, DOWN)
        self.play(GrowFromCenter(question_mark))
        self.wait()

        # We need to tweak a bit to make it work
        question_mark_ref = VGroup(text[:2] for text in curvatures_enclosing)
        question_mark.generate_target()
        question_mark.target.set_opacity(0)
        question_mark.target.move_to(question_mark_ref)
        question_mark.target.match_width(question_mark_ref)
        self.play(
            *[Write(text[:2], stroke_color=YELLOW) for text in curvatures_enclosing],
            *[
                ReplacementTransform(source[2:], target[2:])
                for source, target in zip(bad_curvatures_enclosing[:-1], curvatures_enclosing[:-1])
            ],
            ReplacementTransform(bad_curvatures_enclosing[-1][2:-1], curvatures_enclosing[-1][2:-2]),
            ReplacementTransform(VectorizedPoint(bad_curvatures_enclosing[-1][-1].get_left()), curvatures_enclosing[-1][-2]),
            ReplacementTransform(bad_curvatures_enclosing[-1][-1], curvatures_enclosing[-1][-1]),
            *[
                FadeInFromPoint(relation_enclosing[ind], relation_enclosing[ind].get_center())
                for ind in (7, 8, 10, 25, 26, 28)
            ],
            ReplacementTransform(bad_relation_enclosing[:7], relation_enclosing[:7]),
            ReplacementTransform(bad_relation_enclosing[7], relation_enclosing[9]),
            ReplacementTransform(bad_relation_enclosing[8:22], relation_enclosing[11:25]),
            ReplacementTransform(bad_relation_enclosing[22], relation_enclosing[27]),
            ReplacementTransform(bad_relation_enclosing[23:], relation_enclosing[29:]),
            MoveToTarget(question_mark),
            run_time=2,
        )
        self.remove(bad_curvatures_enclosing, bad_relation_enclosing, question_mark)
        self.add(curvatures_enclosing, relation_enclosing)
        self.wait()

        c1, c2, c3, c4 = circles_enclosing.get_circles()
        text1, text2, text3, text4 = curvatures_enclosing
        tangent_points = [
            circles_enclosing.get_tangent_points()[ind]
            for ind in (2, 4, 5)
        ]
        self.play(
            AnimationGroup(
                Indicate(c4, color=None, scale_factor=1.1),
                AnimationGroup(
                    *[
                        DirectionalFlash(
                            tp, direction=circles_enclosing.get_center() - tp,
                            color=c4.get_color(),
                        )
                        for tp in tangent_points
                    ],
                ),
                lag_ratio=1,
                run_time=2,
            ),
            text4.animate.set_color(CIRCLE_COLORS[-1]),
            *[text.animate.set_color(GREY).set_opacity(0.5) for text in (text1, text2, text3)],
        )
        self.wait()
        self.play(
            *[Indicate(c, color=None, scale_factor=0.9) for c in (c1, c2, c3)],
            *[
                text.animate.set_color(color).set_opacity(1)
                for text, color in zip(curvatures_enclosing[:-1], CIRCLE_COLORS[:-1])
            ],
            text4.animate.set_color(GREY).set_opacity(0.5),
        )
        self.wait()

        # Many questions appear
        descartes_relation = VGroup(
            Tex(text, t2c=CURVATURE_T2C, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2}",
                "= 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)"
            ]
        )
        descartes_relation.arrange(DOWN, aligned_edge=RIGHT)
        descartes_relation.move_to(relation_enclosing)

        self.play(
            ReplacementTransform(relation_enclosing[0], descartes_relation[0][0]),
            FadeTransform(relation_enclosing[1], descartes_relation[0][1:3]),
            ReplacementTransform(relation_enclosing[2], descartes_relation[0][3]),
            FadeTransform(relation_enclosing[3], descartes_relation[0][4:6]),
            ReplacementTransform(relation_enclosing[4], descartes_relation[0][6]),
            FadeTransform(relation_enclosing[5], descartes_relation[0][7:9]),
            ReplacementTransform(relation_enclosing[6], descartes_relation[0][9]),
            FadeTransform(relation_enclosing[7:11], descartes_relation[0][10:12]),
            ReplacementTransform(relation_enclosing[11:13], descartes_relation[0][12:]),
            ReplacementTransform(relation_enclosing[13:16], descartes_relation[1][:3]),
            FadeTransform(relation_enclosing[16], descartes_relation[1][3:6:2]),
            ReplacementTransform(relation_enclosing[17], descartes_relation[1][4]),
            ReplacementTransform(relation_enclosing[18], descartes_relation[1][6]),
            FadeTransform(relation_enclosing[19], descartes_relation[1][7:10:2]),
            ReplacementTransform(relation_enclosing[20], descartes_relation[1][8]),
            ReplacementTransform(relation_enclosing[21], descartes_relation[1][10]),
            FadeTransform(relation_enclosing[22], descartes_relation[1][11:14:2]),
            ReplacementTransform(relation_enclosing[23], descartes_relation[1][12]),
            ReplacementTransform(relation_enclosing[24], descartes_relation[1][14]),
            FadeTransform(relation_enclosing[25:29], descartes_relation[1][15:18:2]),
            ReplacementTransform(relation_enclosing[29], descartes_relation[1][16]),
            ReplacementTransform(relation_enclosing[30], descartes_relation[1][18]),
            text4.animate.set_color(CIRCLE_COLORS[-1]).set_opacity(1),
            run_time=1,
        )
        self.remove(relation_enclosing)
        self.add(descartes_relation)
        self.wait()
        self.play(FlashAround(descartes_relation))
        self.wait()
        self.play(FlashAround(text4))
        self.wait()
        soddy_gosset_relation = VGroup(
            Tex(text, t2c=CURVATURE_T2C, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + \\cdots + k_{n+2} \\right)^{2}",
                "= n \\left( k_{1}^{2} + k_{2}^{2} + \\cdots + k_{n+2}^{2} \\right)"
            ]
        )
        soddy_gosset_relation[1][-5].set_color(WHITE)  # color fixing
        soddy_gosset_relation[1][-2].set_color(SPHERE_COLORS[-1])  # color fixing
        soddy_gosset_relation.arrange(DOWN, aligned_edge=RIGHT)
        soddy_gosset_relation.move_to(curvatures_enclosing)
        two_relations = VGroup(descartes_relation, soddy_gosset_relation)
        self.play(
            FadeOut(curvatures_enclosing),
            FadeIn(soddy_gosset_relation, shift=UP),
        )
        self.play(FlashAround(two_relations))
        self.wait()

        # The secret is to treat circles as vectors
        circle = Circle()
        arrow = Vector(RIGHT)
        vector = MobjectMatrix(
            [Tex(text) for text in ["k", "t", "kx", "ky"]],
            n_cols=1,
        )
        secret = VGroup(circle, arrow, vector)
        secret.arrange(RIGHT, buff=0.5)
        secret.move_to(RIGHT_SIDE / 2)
        secret.set_color(YELLOW)
        self.play(
            FadeOut(two_relations),
            GrowFromCenter(secret),
        )
        self.wait()

        # Transform to the next scene
        circles_enclosing.suspend_updating()
        new_c1, new_c2 = Circle(radius=1), Circle(radius=2)
        for c, color, pos in zip(
            [new_c1, new_c2],
            CIRCLE_COLORS[:2],
            [2 * LEFT, 2 * RIGHT],
        ):
            c.set_color(color)
            c.move_to(pos)
        for c in (c3, c4):
            c.generate_target()
            c.target.set_opacity(0)
            c.target.set_stroke(width=0)
            c.target.scale(0)
        self.play(
            FadeOut(secret, shift=RIGHT, run_time=1),
            AnimationGroup(
                *[MoveToTarget(c) for c in (c3, c4)],
                ReplacementTransform(c2, new_c2),
                ReplacementTransform(c1, new_c1),
                lag_ratio=0.2,
                run_time=2,
            ),
        )
        self.wait()


class InnerProductOfTwoCircles(Scene):
    def construct(self):
        # Setup to match previous scene
        c1, c2 = Circle(radius=1), Circle(radius=2)
        for c, color, pos in zip(
            [c1, c2],
            CIRCLE_COLORS[:2],
            [2 * LEFT, 2 * RIGHT],
        ):
            c.set_color(color)
            c.move_to(pos)
        self.add(c1, c2)

        # Suppose there are only 2 circles C1 and C2
        text_c1 = TexText("$C_1$ \\\\[0.2em] 半径$r_1$")
        text_c2 = TexText("$C_2$ \\\\[0.2em] 半径$r_2$")
        for text, circle in zip([text_c1, text_c2], [c1, c2]):
            text.next_to(circle, UP)
            text.to_edge(UP)
            text.match_color(circle)
        self.play(
            *[FadeIn(text, shift=UP) for text in (text_c1, text_c2)],
            lag_ratio=0.1,
            run_time=1,
        )
        self.wait()

        # We have their radii and the distance between two centers
        dot_c1 = Dot().match_color(c1)
        dot_c2 = Dot().match_color(c2)
        dot_c1.add_updater(lambda m: m.move_to(c1.get_center()))
        dot_c2.add_updater(lambda m: m.move_to(c2.get_center()))
        distance_line = DashedLine().set_color(YELLOW)
        distance_line.add_updater(
            lambda m: m.become(
                DashedLine(c1.get_center(), c2.get_center())
                .set_color(YELLOW)
            )
        )
        distance_brace = Brace(distance_line)
        distance_text = TexText("圆心距$d$").match_color(distance_line)
        distance_brace.add_updater(
            lambda m: m.become(
                Brace(distance_line).match_color(distance_line).put_at_tip(distance_text)
            )
        )

        motion_main_group = VGroup(c1, c2, text_c1, text_c2)
        motion_update_group = VGroup(distance_line, distance_brace, dot_c1, dot_c2)
        for mob in motion_update_group:
            mob.update()
            mob.suspend_updating()
        self.play(
            *[GrowFromCenter(mob) for mob in motion_update_group],
            Write(distance_text),
        )
        for mob in motion_update_group:
            mob.resume_updating()
        self.wait()

        # r1, r2 and d determine the position relation of two circles
        position_relations = VGroup(
            TexText(f"{text}：")
            for text in ["外离", "外切", "相交", "内切", "内含"]
        )
        position_relations.arrange(DOWN, aligned_edge=RIGHT, buff=0.85)
        position_relations.to_corner(UL)
        self.play(
            motion_main_group.animate.to_edge(RIGHT),
            FadeIn(position_relations[0][:-1]),
        )
        self.wait()
        for ind, (amount, relation) in enumerate(zip([0.5, 0.4, 0.6, 0.2], position_relations[1:])):
            anims = [
                c1.animate.shift(amount * RIGHT),
                c2.animate.shift(amount * LEFT),
                FadeIn(relation[:-1]),
            ]
            if ind < 2:
                anims.append(text_c1.animate.shift(amount * RIGHT))
                anims.append(text_c2.animate.shift(amount * LEFT))
            self.play(
                *anims,
                run_time=0.5,
            )
            self.wait(0.5)

        # Each corresponds to a distance relation
        distance_t2c = {
            "r_{1}": c1.get_color(),
            "r_{2}": c2.get_color(),
            "d": distance_line.get_color(),
        }
        distance_relations = VGroup(
            Tex(text, t2c=distance_t2c)
            for text in [
                "d > r_{1} + r_{2}",
                "d = r_{1} + r_{2}",
                "\\left| r_{1} - r_{2} \\right| < d < r_{1} + r_{2}",
                "\\left| r_{1} - r_{2} \\right| = d",
                "\\left| r_{1} - r_{2} \\right| > d",
            ]
        )
        for d_relation, pos_relation in zip(distance_relations, position_relations):
            d_relation.next_to(pos_relation, RIGHT, buff=0.4)
        for mob in distance_relations[:2]:
            mob.match_x(distance_relations[2], direction=RIGHT)
        for mob in distance_relations[3:]:
            mob.match_x(distance_relations[2], direction=LEFT)
        self.play(
            *[FadeIn(relation[-1]) for relation in position_relations],
            FadeIn(distance_relations),
            run_time=2,
        )
        self.wait()

        # We start from the intersect case
        c1.generate_target()
        c2.generate_target()
        c2.target.to_edge(RIGHT)
        c1.target.next_to(c2.target.get_center(), LEFT, buff=1.2)
        text_c1.generate_target()
        text_c2.generate_target()
        text_c1.target.next_to(c1.target, UP, coor_mask=[1, 0, 0])
        text_c2.target.next_to(c2.target, UP, coor_mask=[1, 0, 0])
        self.play(
            FlashAround(VGroup(distance_relations[2], position_relations[2]))
        )
        self.wait()
        self.play(
            FadeOut(distance_relations[:2]),
            FadeOut(position_relations[:2]),
            FadeOut(distance_relations[3:]),
            FadeOut(position_relations[3:]),
            *[MoveToTarget(mob) for mob in (c1, c2, text_c1, text_c2)]
        )
        self.wait()

        # Prepare for a bunch of manipulation
        fade_shift = 0.5 * DOWN
        intersect_0 = distance_relations[2]
        intersect_manipulation = VGroup(
            Tex(
                text,
                t2c=distance_t2c, use_labelled_svg=False,
            )
            for text in (
                "\\left( r_{1} - r_{2} \\right)^2 < d^2 < \\left( r_{1} + r_{2} \\right)^2",
                "r_{1}^2 - 2 r_{1} r_{2} + r_{2}^2 < d^2 < r_{1}^2 + 2 r_{1} r_{2} + r_{2}^2",
                "- 2 r_{1} r_{2} < d^2 - r_{1}^2 - r_{2}^2 < 2 r_{1} r_{2}",
                "- 1 < \\dfrac{d^2 - r_{1}^2 - r_{2}^2}{2 r_{1} r_{2}} < 1",
                "- 1 < \\dfrac{r_{1}^2 + r_{2}^2 - d^2}{2 r_{1} r_{2}} < 1",
            )
        )
        intersect_1, intersect_2, intersect_3, intersect_4, intersect_5 = intersect_manipulation
        for text in intersect_manipulation:
            text.next_to(intersect_0.get_left(), RIGHT, buff=0)

        # 1. Add in square
        source, target = intersect_0, intersect_1
        self.play(
            ReplacementTransform(source[:7], target[:7]),
            FadeIn(target[7], shift=fade_shift),
            ReplacementTransform(source[7:9], target[8:10]),
            FadeIn(target[10], shift=fade_shift),
            ReplacementTransform(source[9], target[11]),
            FadeIn(target[12], shift=fade_shift),
            ReplacementTransform(source[10:], target[13:18]),
            FadeIn(target[18:], shift=fade_shift),
            run_time=2,
        )
        self.remove(source)
        self.add(target)
        self.wait()

        # 2. simplify the square
        source, target = intersect_1, intersect_2
        motion_update_group.suspend_updating()
        motion_update_group.add(distance_text)
        self.play(
            FadeTransform(source[:8], target[:13]),
            ReplacementTransform(source[8:12], target[13:17]),
            FadeTransform(source[12:], target[17:]),
            FadeOut(motion_main_group, shift=2 * RIGHT),
            FadeOut(motion_update_group, shift=2 * RIGHT),
            run_time=2,
        )
        self.remove(source)
        self.add(target)
        self.wait()

        # 3. Minus (r1^2 + r2^2)
        source, target = intersect_2, intersect_3
        self.play(
            AnimationGroup(
                FadeOut(source[:3], shift=fade_shift),
                FadeOut(source[9:13], shift=fade_shift),
                FadeOut(source[17:21], shift=fade_shift),
                FadeOut(source[26:], shift=fade_shift),
                run_time=1,
            ),
            AnimationGroup(
                ReplacementTransform(source[3:9], target[:6]),
                ReplacementTransform(source[13:16], target[6:9]),
                FadeIn(target[9:17], shift=fade_shift),
                ReplacementTransform(source[16], target[17]),
                ReplacementTransform(source[21:26], target[18:]),
                run_time=2,
            )
        )
        self.remove(source)
        self.add(target)
        self.wait()

        # 4. Divided by (2r1r2)
        source, target = intersect_3, intersect_4
        self.play(
            ReplacementTransform(source[0], target[0]),
            FadeTransform(source[1:6], target[1]),
            ReplacementTransform(source[6], target[2]),
            ReplacementTransform(source[7:17], target[3:13]),
            FadeIn(target[13:19], shift=-fade_shift),
            ReplacementTransform(source[17], target[19]),
            FadeTransform(source[18:], target[20]),
            run_time=2,
        )
        self.remove(source)
        self.add(target)
        self.wait()

        # 5. Invert the sign
        path_arc = PI / 4
        source, target = intersect_4, intersect_5
        self.play(
            FadeTransform(source[:2], target[20], path_arc=3 * path_arc),
            ReplacementTransform(source[2], target[19], path_arc=3 * path_arc),
            ReplacementTransform(source[3:5], target[11:13], path_arc=path_arc),
            ReplacementTransform(source[5], VectorizedPoint(target[3].get_left()), path_arc=path_arc),
            ReplacementTransform(source[6:9], target[3:6], path_arc=path_arc),
            ReplacementTransform(source[9], target[6], path_arc=path_arc),
            ReplacementTransform(source[10:13], target[7:10], path_arc=path_arc),
            ReplacementTransform(VectorizedPoint(source[3].get_left()), target[10], path_arc=path_arc),
            ReplacementTransform(source[13:19], target[13:19]),
            ReplacementTransform(source[19], target[2], path_arc=3 * path_arc),
            FadeTransform(source[20], target[:2], path_arc=3 * path_arc),
            run_time=2,
        )
        self.remove(source)
        self.add(target)
        self.wait()
        self.play(FlashAround(target[3:19]))
        self.wait()

        # Same goes for other relations
        inner_product_relations = VGroup(
            Tex(text, t2c=distance_t2c, use_labelled_svg=False)
            for text in [
                "-1 > \\dfrac{r_{1}^2 + r_{2}^2 - d^2}{2 r_{1} r_{2}}",
                "-1 = \\dfrac{r_{1}^2 + r_{2}^2 - d^2}{2 r_{1} r_{2}}",
                "-1 < \\dfrac{r_{1}^2 + r_{2}^2 - d^2}{2 r_{1} r_{2}} < 1",
                "\\dfrac{r_{1}^2 + r_{2}^2 - d^2}{2 r_{1} r_{2}} = 1",
                "\\dfrac{r_{1}^2 + r_{2}^2 - d^2}{2 r_{1} r_{2}} > 1",
            ]
        )
        for ip_relation, pos_relation in zip(inner_product_relations, position_relations):
            ip_relation.scale(0.8)
            ip_relation.next_to(pos_relation, RIGHT, buff=0.4)
        for mob in inner_product_relations[:2]:
            mob.match_x(inner_product_relations[2], direction=LEFT)
        for mob in inner_product_relations[3:]:
            mob.match_x(inner_product_relations[2], direction=RIGHT)

        self.play(
            FadeIn(position_relations[:2]),
            FadeIn(position_relations[3:]),
            FadeIn(inner_product_relations[:2]),
            ReplacementTransform(intersect_5, inner_product_relations[2]),
            FadeIn(inner_product_relations[3:]),
        )
        self.wait()

        inner_product_name = TexText(
            "$C_1$与$C_2$的内积",
            t2c={
                "C_1": c1.get_color(),
                "C_2": c2.get_color(),
            }
        )
        inner_product_formula = Tex(
            "\\dfrac{r_{1}^2 + r_{2}^2 - d^2}{2 r_{1} r_{2}}",
            t2c=distance_t2c, use_labelled_svg=False,
        )
        inner_product_formula.move_to((RIGHT_SIDE + BOTTOM) / 2)
        inner_product_name.next_to(inner_product_formula, UP, buff=0.5)

        ipr, ipf = inner_product_relations, inner_product_formula
        self.play(
            ReplacementTransform(ipr[0][3:].copy(), ipf),
            ReplacementTransform(ipr[1][3:].copy(), ipf),
            ReplacementTransform(ipr[2][3:-2].copy(), ipf),
            ReplacementTransform(ipr[3][:-2].copy(), ipf),
            ReplacementTransform(ipr[4][:-2].copy(), ipf),
            run_time=3,
        )
        self.wait()
        self.play(Write(inner_product_name))
        self.wait()

        # We only use the name of 'inner product'
        vector_space = TexText("向量空间 ?")
        certain_properties = TexText("$+$ 特定性质 ?")
        arrow = Vector(DOWN)
        arrow.set_color(BLUE)
        arrow.next_to(inner_product_name, UP, buff=0.5)
        problems = VGroup(vector_space, certain_properties)
        problems.match_color(arrow)
        problems.arrange(DOWN, aligned_edge=RIGHT)
        problems.next_to(arrow, UP, buff=0.5)
        self.play(
            Write(vector_space[:-1], stroke_color=arrow.get_color()),
            GrowArrow(arrow),
        )
        self.wait()
        self.play(
            Write(certain_properties[:-1], stroke_color=arrow.get_color()),
            inner_product_name[-2:].animate.set_color(arrow.get_color()),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    FadeIn(text[-1], shift=0.5 * LEFT)
                    for text in (vector_space, certain_properties)
                ],
                lag_ratio=0.1,
                run_time=1.5,
            )
        )
        self.wait()

        # But it does have a geometry meaning
        c2.scale(0.8)
        VGroup(c1, c2).to_edge(UP)
        dot_c1.resume_updating()
        dot_c2.resume_updating()
        self.play(
            FadeOut(problems),
            ApplyMethod(arrow.scale, 0, {"about_point": arrow.get_bottom()}),
            *[FadeIn(mob) for mob in (c1, c2, dot_c1, dot_c2)],
        )
        self.remove(arrow)
        self.wait()

        self.play(FlashAround(inner_product_relations), run_time=2)
        self.wait()
        self.play(FlashAround(inner_product_formula), run_time=2)
        self.wait()

        intersection_point = always_redraw(
            lambda: Dot(
                get_circle_intersections(c1, c2)[0]
            )
            .match_color(arrow)
        )
        triangle = always_redraw(
            lambda: Polygon(
                dot_c1.get_center(),
                dot_c2.get_center(),
                intersection_point.get_center(),
                stroke_width=2,
                fill_opacity=0.3,
            )
            .match_color(arrow)
        )
        angle = always_redraw(
            lambda: get_angle_arc(
                intersection_point.get_center(),
                dot_c1.get_center(),
                dot_c2.get_center(),
                stroke_width=2,
            )
            .match_color(arrow)
        )
        label = always_redraw(
            lambda: get_angle_label(
                intersection_point.get_center(),
                dot_c1.get_center(),
                dot_c2.get_center(),
            )
            .match_color(arrow)
        )
        angle_group = VGroup(intersection_point, triangle, angle, label)
        for mob in angle_group:
            mob.update().suspend_updating()
        intersection_point.generate_target()
        intersection_point.scale(8).set_opacity(0)
        self.play(
            MoveToTarget(intersection_point),
            FadeIn(triangle),
        )
        self.wait()

        # A triangle with side lengths r1, r2 and d
        r1_text = get_line_label(
            Line(intersection_point, dot_c1),
            "r_1",
            invert=True,
        ).match_color(c1)
        r2_text = get_line_label(
            Line(intersection_point, dot_c2),
            "r_2",
        ).match_color(c2)
        d_text = get_line_label(
            Line(dot_c1, dot_c2),
            "d",
            invert=True,
        ).match_color(distance_line)
        self.play(
            *[
                Write(mob, stroke_color=mob.get_color())
                for mob in (r1_text, r2_text, d_text)
            ]
        )
        self.wait()

        # The inner product is cosine of the angle at the intersection point
        cosine_t2c = distance_t2c.copy()
        cosine_t2c["\\cos \\theta"] = arrow.get_color()
        cosine_formula = Tex(
            "\\cos \\theta = \\dfrac{r_{1}^2 + r_{2}^2 - d^2}{2 r_{1} r_{2}}",
            t2c=cosine_t2c, use_labelled_svg=False,
        )
        cosine_formula.move_to(inner_product_formula)
        self.play(
            Write(cosine_formula[:4], stroke_color=arrow.get_color()),
            GrowFromCenter(cosine_formula[4]),
            ReplacementTransform(inner_product_formula, cosine_formula[5:]),
            GrowFromPoint(angle, intersection_point.get_center()),
            GrowFromPoint(label, intersection_point.get_center()),
            run_time=2,
        )
        self.wait()
        self.play(
            *[FadeOut(mob) for mob in (r1_text, r2_text, d_text)],
        )
        for mob in angle_group:
            mob.update().resume_updating()
        self.wait()

        # External tangent = inner product of -1
        left_group = VGroup(position_relations, inner_product_relations)
        external_part = VGroup(position_relations[1], inner_product_relations[1])
        top_mask_external = FullScreenFadeRectangle(fill_opacity=0.9)
        top_mask_external.set_width(left_group.get_width() * 1.3, stretch=True)
        top_mask_external.move_to(left_group)
        top_mask_external.next_to(external_part, UP, coor_mask=[0, 1, 0])
        bottom_mask_external = top_mask_external.copy()
        bottom_mask_external.next_to(external_part, DOWN, coor_mask=[0, 1, 0])
        cosine_external = Tex("= \\cos \\pi")
        cosine_external[1:].match_color(arrow)
        cosine_external.scale(0.6)
        cosine_external.next_to(external_part[1], RIGHT)
        self.play(
            FadeIn(cosine_external),
            FadeIn(top_mask_external),
            FadeIn(bottom_mask_external),
            c2.animate.next_to(c1, RIGHT, buff=0),
            run_time=2,
        )
        self.wait()

        # Internal tangent = inner product of 1
        internal_part = VGroup(position_relations[3], inner_product_relations[3])
        top_mask_internal = top_mask_external.copy()
        top_mask_internal.next_to(internal_part, UP, coor_mask=[0, 1, 0])
        bottom_mask_internal = top_mask_internal.copy()
        bottom_mask_internal.next_to(internal_part, DOWN, coor_mask=[0, 1, 0])
        cosine_internal = Tex("\\cos 0 =")
        cosine_internal[:-1].match_color(arrow)
        cosine_internal.scale(0.6)
        cosine_internal.next_to(internal_part[1], LEFT)
        self.play(
            ReplacementTransform(cosine_external[0].copy(), cosine_internal[-1]),
            ReplacementTransform(cosine_external[1:].copy(), cosine_internal[:-1]),
            ReplacementTransform(top_mask_external, top_mask_internal),
            ReplacementTransform(bottom_mask_external, bottom_mask_internal),
            c2.animate.shift(c1.get_width() * 0.99999 * LEFT),
            run_time=3,
        )
        self.wait()

        self.play(
            FadeOut(top_mask_internal),
            FadeOut(bottom_mask_internal),
            c2.animate.shift(c1.get_width() * 0.4 * RIGHT),
            run_time=3,
        )
        self.wait()

        # Setup for the next scene
        for mob in angle_group:
            mob.suspend_updating()
        angle_group.remove(dot_c1, dot_c2)
        c1.generate_target()
        c1.target.set_height(3.4)
        c1.target.move_to(1.8 * RIGHT + 0.7 * DOWN)
        c2.generate_target()
        c2.target.set_height(4.7)
        c2.target.move_to(4.2 * RIGHT + 1.3 * UP)
        left_group.add(cosine_external, cosine_internal)
        self.play(
            AnimationGroup(
                FadeOut(
                    VGroup(
                        left_group, angle_group,
                        inner_product_name, cosine_formula[:5]
                    )
                ),
                run_time=1,
            ),
            AnimationGroup(
                cosine_formula[5:].animate.scale(1.5).move_to(LEFT_SIDE / 2),
                *[MoveToTarget(mob) for mob in (c1, c2)],
                run_time=2,
            ),
        )
        self.wait()


class SimplifyTheInnerProduct(Scene):
    def construct(self):
        # Setup to match previous scene
        c1, c2 = [Circle(n_components=128) for _ in range(2)]
        for c, color in zip([c1, c2], CIRCLE_COLORS[:2]):
            c.set_color(color)
        c1.set_height(3.4)
        c1.move_to(1.8 * RIGHT + 0.7 * DOWN)
        c2.set_height(4.7)
        c2.move_to(4.2 * RIGHT + 1.3 * UP)
        dot_c1 = Dot(c1.get_center()).match_color(c1)
        dot_c2 = Dot(c2.get_center()).match_color(c2)
        variables_t2c = {
            "r_{1}": c1.get_color(),
            "x_{1}": c1.get_color(),
            "y_{1}": c1.get_color(),
            "r_{2}": c2.get_color(),
            "x_{2}": c2.get_color(),
            "y_{2}": c2.get_color(),
            "{d}": YELLOW,
        }
        inner_product_formula = Tex(
            "\\dfrac{r_{1}^2 + r_{2}^2 - {d}^2}{2 r_{1} r_{2}}",
            t2c=variables_t2c, use_labelled_svg=False,
        )
        inner_product_formula.save_state()
        inner_product_formula.move_to(LEFT_SIDE / 2)
        inner_product_formula.scale(1.5)
        setup_group = VGroup(inner_product_formula, c1, c2, dot_c1, dot_c2)
        self.add(setup_group)

        # Center distance d is troublesome, so we use a coordinate system
        p1, p2 = dot_c1.get_center(), dot_c2.get_center()
        pm = (p1 + p2) / 2
        line_part1, line_part2 = Line(pm, p1), Line(pm, p2)
        for line in (line_part1, line_part2):
            line.set_color(YELLOW)
        for k in range(2):
            self.play(
                FlashAround(inner_product_formula[8:10]),
                ShowCreationThenDestruction(line_part1),
                ShowCreationThenDestruction(line_part2),
            )
            self.wait()

        plane = FadeCoordinatePlane(
            x_range=(-16.0, 16.0, 1.0),
            y_range=(-8.0, 8.0, 1.0),
            include_labels=False,
        )
        self.add(plane, setup_group)
        self.play(
            inner_product_formula.animate.move_to((TOP + LEFT_SIDE) / 2),
            FadeIn(plane),
        )
        self.wait()
        coord_c1 = Tex("(x_{1},y_{1})").match_color(c1)
        coord_c1.next_to(dot_c1, DOWN)
        coord_c2 = Tex("(x_{2},y_{2})").match_color(c2)
        coord_c2.next_to(dot_c2, DOWN)
        self.play(
            AnimationGroup(
                GrowFromPoint(coord_c1, dot_c1.get_center()),
                GrowFromPoint(coord_c2, dot_c2.get_center()),
                lag_ratio=0.1,
            ),
        )
        self.wait()

        # We don't care about how the plane is placed
        plane.generate_target()
        plane.target.rotate(-0.570 / 2).shift(0.570 * DL).scale(2 * 0.570)
        self.play(
            MoveToTarget(plane, rate_func=there_and_back),
            run_time=2,
        )
        self.wait()

        # Expand the inner product
        simplify_0 = inner_product_formula
        simplify_manipulation = VGroup(
            Tex(
                text,
                t2c=variables_t2c, use_labelled_svg=False,
            )
            for text in (
                "\\dfrac{r_{1}^2 + r_{2}^2 - \\left( x_{1} - x_{2} \\right)^2 - \\left( y_{1} - y_{2} \\right)^2}{2 r_{1} r_{2}}",
                "\\dfrac{r_{1}^2 + r_{2}^2 - x_{1}^2 + 2x_{1}x_{2} - x_{2}^2 - y_{1}^2 + 2y_{1}y_{2} - y_{2}^2}{2 r_{1} r_{2}}",
            )
        )
        simplify_1, simplify_2 = simplify_manipulation
        simplify_2.save_state()
        for text in simplify_manipulation:
            text.set_width(FRAME_WIDTH * 0.4)
            text.move_to((BOTTOM + LEFT_SIDE) / 2)

        # 1. Replace d with x1, x2, y1, y2
        source, target = simplify_0, simplify_1
        self.play(
            ReplacementTransform(source[:7].copy(), target[:7]),
            FadeTransform(source[7:10].copy(), target[7:25]),
            ReplacementTransform(source[10:].copy(), target[25:]),
            run_time=2,
        )
        self.add(target)
        self.wait()

        # 2. More expansion
        source, target = simplify_1, simplify_2
        self.play(
            ReplacementTransform(source[:7], target[:7]),
            FadeTransform(source[7:16], target[7:21]),
            FadeTransform(source[16:25], target[21:35]),
            ReplacementTransform(source[25:], target[35:]),
            run_time=2,
        )
        self.remove(source)
        self.add(target)
        self.wait()

        # It looks more intimidating, but is easier to deal with
        geometry_group = VGroup(plane, c1, c2, dot_c1, dot_c2, coord_c1, coord_c2)
        simplify_0.generate_target()
        simplify_2.generate_target()
        for mob in (simplify_0, simplify_2):
            mob.target.match_height(mob.saved_state)
        simplify_0.target.move_to(TOP / 3)
        simplify_2.target.move_to(BOTTOM / 3)
        equal_sign = Tex("=").rotate(PI / 2)
        equal_sign.generate_target()
        equal_sign.scale(0).set_opacity(0).move_to(LEFT_SIDE / 2)
        self.add(geometry_group, simplify_0, simplify_2, equal_sign)
        temp_group = VGroup(simplify_0, simplify_2, equal_sign)
        self.play(
            *[MoveToTarget(mob) for mob in temp_group],
            FadeOut(geometry_group, shift=6 * RIGHT),
            run_time=2,
        )
        self.wait()

        mixed_color = [c1.get_color(), c2.get_color()] * 5
        self.play(
            FlashAround(simplify_0[7:10]),
            FlashAround(simplify_2[7:35], color=mixed_color),
            run_time=3,
        )
        self.wait()

        for mob in temp_group:
            mob.generate_target()
        simplify_2.target.move_to(TOP / 2)
        shift_vec = simplify_2.target.get_center() - simplify_2.get_center()
        for mob in (simplify_0, equal_sign):
            mob.target.shift(shift_vec).set_opacity(0)
        self.play(
            *[MoveToTarget(mob) for mob in temp_group],
            run_time=2,
        )
        self.remove(simplify_0, equal_sign)
        self.wait()

        more_simplify_manipulation = VGroup(
            Tex(
                text,
                t2c=variables_t2c, use_labelled_svg=False,
            )
            for text in (
                """
                - \\dfrac{x_{1}^2 + y_{1}^2 - r_{1}^2}{2 r_{1} r_{2}}
                - \\dfrac{x_{2}^2 + y_{2}^2 - r_{2}^2}{2 r_{1} r_{2}}
                + \\dfrac{2 x_{1} x_{2}}{2 r_{1} r_{2}}
                + \\dfrac{2 y_{1} y_{2}}{2 r_{1} r_{2}}
                """,
                """
                - \\dfrac{1}{2} \\dfrac{x_{1}^2 + y_{1}^2 - r_{1}^2}{r_{1}} \\dfrac{1}{r_{2}}
                - \\dfrac{1}{2} \\dfrac{1}{r_{1}} \\dfrac{x_{2}^2 + y_{2}^2 - r_{2}^2}{r_{2}}
                + 1 \\dfrac{x_{1}}{r_{1}} \\dfrac{x_{2}}{r_{2}}
                + 1 \\dfrac{y_{1}}{r_{1}} \\dfrac{y_{2}}{r_{2}}
                """,
            )
        )
        simplify_3, simplify_4 = more_simplify_manipulation
        simplify_3.move_to(ORIGIN)
        simplify_4.move_to(BOTTOM / 2)
        equal_sign_3 = Tex("=").next_to(simplify_3, LEFT)
        equal_sign_4 = equal_sign_3.copy().next_to(simplify_4, LEFT)

        # 3. Separate the terms
        source, target = simplify_2, simplify_3
        source_copy = source.copy()
        source.save_state()
        source.set_color(GREY).set_opacity(0.4)
        self.add(source_copy)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    ReplacementTransform(source_copy[11:17], target[36:42]),
                    ReplacementTransform(source_copy[35].copy(), target[42]),
                    ReplacementTransform(source_copy[36:].copy(), target[43:48]),
                ),
                AnimationGroup(
                    ReplacementTransform(source_copy[25:31], target[48:54]),
                    ReplacementTransform(source_copy[35].copy(), target[54]),
                    ReplacementTransform(source_copy[36:].copy(), target[55:]),
                ),
                lag_ratio=0.25,
            ),
            run_time=3,
        )
        self.wait()
        self.play(
            AnimationGroup(
                AnimationGroup(
                    ReplacementTransform(VectorizedPoint(source_copy[0].get_left()), target[8]),
                    ReplacementTransform(source_copy[:3], target[9:12]),
                    ReplacementTransform(source_copy[35].copy(), target[12]),
                    ReplacementTransform(source_copy[7], target[0]),
                    ReplacementTransform(source_copy[8:11], target[1:4]),
                    ReplacementTransform(source_copy[21:25], target[4:8]),
                    ReplacementTransform(source_copy[36:].copy(), target[13:18]),
                ),
                AnimationGroup(
                    ReplacementTransform(source_copy[3:7], target[26:30]),
                    ReplacementTransform(source_copy[17:21], target[18:22]),
                    ReplacementTransform(source_copy[31:35], target[22:26]),
                    ReplacementTransform(source_copy[35], target[30]),
                    ReplacementTransform(source_copy[36:], target[31:36]),
                ),
                lag_ratio=0.25,
            ),
            run_time=3,
        )
        self.wait()
        self.remove(source_copy)
        self.wait()
        self.play(
            *[FlashAround(target[ind]) for ind in (0, 18)],
            lag_ratio=0.2,
            run_time=1,
        )
        self.wait()

        # 4. Separate by subscripts
        source, target = simplify_3, simplify_4
        source_copy = source.copy()
        source.save_state()
        source.set_color(GREY).set_opacity(0.4)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    ReplacementTransform(source_copy[0], target[0]),
                    ReplacementTransform(VectorizedPoint(source_copy[12].get_left()), target[1]),
                    ReplacementTransform(source_copy[12].copy(), target[2]),
                    ReplacementTransform(source_copy[13], target[3]),
                    ReplacementTransform(source_copy[1:12], target[4:15]),
                    ReplacementTransform(source_copy[12].copy(), target[15]),
                    ReplacementTransform(source_copy[14:16], target[16:18]),
                    ReplacementTransform(VectorizedPoint(source_copy[12].get_right()), target[18]),
                    ReplacementTransform(source_copy[12], target[19]),
                    ReplacementTransform(source_copy[16:18], target[20:22]),
                ),
                AnimationGroup(
                    ReplacementTransform(source_copy[18], target[22]),
                    ReplacementTransform(VectorizedPoint(source_copy[30].get_left()), target[23]),
                    ReplacementTransform(source_copy[30].copy(), target[24]),
                    ReplacementTransform(source_copy[31], target[25]),
                    ReplacementTransform(VectorizedPoint(source_copy[30].get_left()), target[26]),
                    ReplacementTransform(source_copy[30].copy(), target[27]),
                    ReplacementTransform(source_copy[32:34], target[28:30]),
                    ReplacementTransform(source_copy[19:30], target[30:41]),
                    ReplacementTransform(source_copy[30], target[41]),
                    ReplacementTransform(source_copy[34:36], target[42:44]),
                ),
                AnimationGroup(
                    ReplacementTransform(source_copy[36], target[44]),
                    ReplacementTransform(source_copy[37], target[45]),
                    ReplacementTransform(source_copy[43], target[45]),
                    ReplacementTransform(source_copy[38:40], target[46:48]),
                    ReplacementTransform(source_copy[42].copy(), target[48]),
                    ReplacementTransform(source_copy[44:46], target[49:51]),
                    ReplacementTransform(source_copy[40:42], target[51:53]),
                    ReplacementTransform(source_copy[42], target[53]),
                    ReplacementTransform(source_copy[46:48], target[54:56]),
                ),
                AnimationGroup(
                    ReplacementTransform(source_copy[48], target[56]),
                    ReplacementTransform(source_copy[49], target[57]),
                    ReplacementTransform(source_copy[55], target[57]),
                    ReplacementTransform(source_copy[50:52], target[58:60]),
                    ReplacementTransform(source_copy[54].copy(), target[60]),
                    ReplacementTransform(source_copy[56:58], target[61:63]),
                    ReplacementTransform(source_copy[52:54], target[63:65]),
                    ReplacementTransform(source_copy[54], target[65]),
                    ReplacementTransform(source_copy[58:], target[66:]),
                ),
                lag_ratio=0.25,
            ),
            run_time=5,
        )
        self.remove(source_copy)
        self.wait()
        self.play(
            *[Restore(mob) for mob in (simplify_2, simplify_3)],
            *[Write(mob) for mob in (equal_sign_3, equal_sign_4)],
        )
        self.wait()

        # Its structure becomes clearer
        simplify_4.generate_target()
        simplify_4.target.center()
        shift_vec = simplify_4.target.get_center() - simplify_4.get_center()
        title = TexText(
            "圆$C_{1}$与圆$C_{2}$的内积",
            t2c={
                "圆$C_{1}$": c1.get_color(),
                "圆$C_{2}$": c2.get_color(),
            },
            use_labelled_svg=False,
        )
        title.to_edge(UP)
        for mob in (simplify_2, simplify_3, equal_sign_3, equal_sign_4):
            mob.generate_target()
            mob.target.shift(shift_vec).set_opacity(0)
        self.play(
            *[
                MoveToTarget(mob)
                for mob in (
                    simplify_2, simplify_3, simplify_4,
                    equal_sign_3, equal_sign_4,
                )
            ],
            FadeIn(title, shift=shift_vec),
        )
        self.wait()

        # Four terms have the same structure - constant, property from c1 and property from c2
        simplify_4.save_state()
        terms_constants = VGroup(
            VGroup(simplify_4[start:end])
            for start, end in zip(
                [0, 22, 44, 56],
                [4, 26, 46, 58],
            )
        )
        terms_c1 = VGroup(
            VGroup(simplify_4[start:end])
            for start, end in zip(
                [4, 26, 46, 58],
                [18, 30, 51, 63],
            )
        )
        terms_c2 = VGroup(
            VGroup(simplify_4[start:end])
            for start, end in zip(
                [18, 30, 51, 63],
                [22, 44, 56, 68],
            )
        )
        self.play(
            AnimationGroup(
                *[FlashAround(mob, color=BLUE) for mob in terms_constants],
                lag_ratio=0.1,
            ),
            terms_constants.animate.set_color(BLUE).set_opacity(1),
            terms_c1.animate.set_color(GREY).set_opacity(0.5),
            terms_c2.animate.set_color(GREY).set_opacity(0.5),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[FlashAround(mob, color=c1.get_color()) for mob in terms_c1],
                lag_ratio=0.1,
            ),
            terms_constants.animate.set_color(GREY).set_opacity(0.5),
            terms_c1.animate.set_color(c1.get_color()).set_opacity(1),
            terms_c2.animate.set_color(GREY).set_opacity(0.5),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[FlashAround(mob, color=c2.get_color()) for mob in terms_c2],
                lag_ratio=0.1,
            ),
            terms_constants.animate.set_color(GREY).set_opacity(0.5),
            terms_c1.animate.set_color(GREY).set_opacity(0.5),
            terms_c2.animate.set_color(c2.get_color()).set_opacity(1),
        )
        self.wait()

        # Properties of a circle have 4 types
        simplify_4.generate_target()
        simplify_4.target.next_to(title, DOWN, buff=0.5)
        shift_vec = simplify_4.target.get_center() - simplify_4.get_center()
        properties_c1 = terms_c1.copy()
        properties_c2 = terms_c2.copy()
        properties_c1.generate_target()
        properties_c2.generate_target()
        properties_c1.target.next_to(simplify_4.target, DOWN, buff=1, coor_mask=[0, 1, 0])
        properties_c1.target.set_color(c1.get_color())
        properties_c1.target.set_opacity(1)
        properties_c2.target.next_to(properties_c1.target, DOWN, buff=0.5, coor_mask=[0, 1, 0])
        properties_c2.target.set_color(c2.get_color())
        properties_c2.target.set_opacity(1)
        self.play(
            terms_constants.animate.shift(shift_vec).set_color(WHITE).set_opacity(1),
            terms_c1.animate.shift(shift_vec).set_color(c1.get_color()).set_opacity(1),
            terms_c2.animate.shift(shift_vec).set_color(c2.get_color()).set_opacity(1),
            MoveToTarget(properties_c1),
            MoveToTarget(properties_c2),
            run_time=2,
        )
        self.wait()
        properties_group = VGroup(properties_c1, properties_c2)
        properties_rect = SurroundingRectangle(properties_group, buff=0.5)
        properties_text = TexText("圆的属性")
        properties_text.set_color(YELLOW)
        properties_text.set_backstroke()
        properties_text.next_to(properties_rect.get_corner(DL), UR)
        self.add(properties_rect, properties_c1, properties_c2, properties_text)
        self.play(
            ShowCreation(properties_rect),
            Write(properties_text, stroke_color=YELLOW),
            run_time=1,
        )
        self.wait()

        properties_rect.generate_target()
        properties_rect.target.stretch_to_fit_width(FRAME_WIDTH)
        properties_rect.target.stretch_to_fit_height(FRAME_HEIGHT)
        properties_rect.target.set_opacity(0).set_stroke(width=0)
        properties_rect.target.center()
        properties_text.generate_target()
        properties_text.target.center().to_edge(UP)
        old_title_group = VGroup(title, simplify_4)
        old_title_group.generate_target()
        old_title_group.target.next_to(TOP, UP).set_opacity(0)

        t1, k1, kx1, ky1 = properties_c1
        k2, t2, kx2, ky2 = properties_c2
        all_properties = VGroup(
            k1, t1, kx1, ky1,
            k2, t2, kx2, ky2,
        )
        all_properties.generate_target()
        all_properties.target.arrange_in_grid(
            4, 2, fill_rows_first=False,
            h_buff=0.5,
        )
        all_properties.target.to_edge(RIGHT)
        for ind, mob in enumerate(all_properties):
            mob.generate_target()
            mob.target.become(all_properties.target[ind])

        # End of the scene, simplify to match the next scene
        self.play(
            *[
                MoveToTarget(mob)
                for mob in (
                    properties_rect, properties_text, old_title_group,
                )
            ],
            AnimationGroup(
                *[MoveToTarget(mob) for mob in all_properties],
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.wait()


class CirclePropertiesAndLinearAlgebra(Scene):
    def construct(self):
        # Setup to match the previous scene
        title = TexText("圆的属性")
        title.set_color(YELLOW)
        title.to_edge(UP)
        grid = VGroup(
            Tex(text)
            for text in [
                "\\dfrac{1}{r_{1}}", "\\dfrac{x_{1}^2 + y_{1}^2 - r_{1}^2}{r_{1}}", "\\dfrac{x_{1}}{r_{1}}", "\\dfrac{y_{1}}{r_{1}}",
                "\\dfrac{1}{r_{2}}", "\\dfrac{x_{2}^2 + y_{2}^2 - r_{2}^2}{r_{2}}", "\\dfrac{x_{2}}{r_{2}}", "\\dfrac{y_{2}}{r_{2}}",
            ]
        )
        k1, t1, kx1, ky1, k2, t2, kx2, ky2 = grid
        props_c1, props_c2 = grid[:4], grid[4:]
        props_c1.set_color(CIRCLE_COLORS[0])
        props_c2.set_color(CIRCLE_COLORS[1])
        grid.arrange_in_grid(
            4, 2, fill_rows_first=False,
            h_buff=0.5,
        )
        grid.to_edge(RIGHT)
        self.add(title, grid)

        # More setup for revealing properties
        props_full_text = Tex(
            """
            \\begin{tabular}{@{} r @{\\hspace{0.3em}} l @{\\hspace{0.3em}} l @{}}
            曲率 & $k$ & $= \\dfrac{1}{r}$ \\\\
            反演曲率 & $t$ & $= \\dfrac{x^2 + y^2 - r^2}{r}$ \\\\
            约化横坐标 & $kx$ & $= \\dfrac{x}{r}$ \\\\
            约化纵坐标 & $ky$ & $= \\dfrac{y}{r}$
            \\end{tabular}
            """
        )
        prop_k, prop_t = props_full_text[:7], props_full_text[7:23]
        prop_kx, prop_ky = props_full_text[23:34], props_full_text[34:]
        for mob in (prop_k[:3], prop_t[:5], prop_kx[:7], prop_ky[:7]):
            mob.match_color(title)
        props_all = VGroup(prop_k, prop_t, prop_kx, prop_ky)
        props_all.next_to(grid, LEFT, buff=0.5)
        for prop, ref in zip(props_all, props_c2):
            prop.next_to(ref, LEFT, coor_mask=[0, 1, 0])
        props_all.to_edge(LEFT)

        # 1. Curvature (k = 1/r)
        k_prop_pair = [
            [0, 4], [1, 5], [2, 6],
        ]
        self.play(
            AnimationGroup(
                *[
                    ReplacementTransform(k1[k_ind].copy(), props_full_text[prop_ind])
                    for k_ind, prop_ind in k_prop_pair
                ],
            ),
            AnimationGroup(
                *[
                    ReplacementTransform(k2[k_ind].copy(), props_full_text[prop_ind])
                    for k_ind, prop_ind in k_prop_pair
                ],
            ),
        )
        self.wait()
        self.play(
            Write(props_full_text[:3], stroke_color=title.get_color()),
            GrowFromCenter(props_full_text[3]),
        )
        self.wait()

        # 2. Co-curvature (t = (x^2+y^2-r^2)/r)
        t_prop_pair = [
            [0, 13], [1, 14], [3, 15], [4, 16], [5, 17],
            [7, 18], [8, 19], [9, 20], [11, 21], [12, 22],
        ]
        self.play(
            AnimationGroup(
                *[
                    ReplacementTransform(t1[t_ind].copy(), props_full_text[prop_ind])
                    for t_ind, prop_ind in t_prop_pair
                ],
            ),
            AnimationGroup(
                *[
                    ReplacementTransform(t2[t_ind].copy(), props_full_text[prop_ind])
                    for t_ind, prop_ind in t_prop_pair
                ],
            ),
        )
        self.wait()
        self.play(
            Write(props_full_text[7:12], stroke_color=title.get_color()),
            GrowFromCenter(props_full_text[12]),
        )
        self.wait()
        self.play(
            FlashAround(prop_t[4:], color=title.get_color()),
        )
        self.wait()

        # 3-4. Reduced coordinates
        kx_prop_pair = [
            [0, 31], [2, 32], [3, 33],
        ]
        ky_prop_pair = [
            [0, 42], [2, 43], [3, 44],
        ]
        self.play(
            AnimationGroup(
                AnimationGroup(
                    AnimationGroup(
                        *[
                            ReplacementTransform(kx1[kx_ind].copy(), props_full_text[prop_ind])
                            for kx_ind, prop_ind in kx_prop_pair
                        ],
                    ),
                    AnimationGroup(
                        *[
                            ReplacementTransform(kx2[kx_ind].copy(), props_full_text[prop_ind])
                            for kx_ind, prop_ind in kx_prop_pair
                        ],
                    ),
                ),
                AnimationGroup(
                    AnimationGroup(
                        *[
                            ReplacementTransform(ky1[ky_ind].copy(), props_full_text[prop_ind])
                            for ky_ind, prop_ind in ky_prop_pair
                        ],
                    ),
                    AnimationGroup(
                        *[
                            ReplacementTransform(ky2[ky_ind].copy(), props_full_text[prop_ind])
                            for ky_ind, prop_ind in ky_prop_pair
                        ],
                    ),
                ),
                lag_ratio=0.2,
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            Write(props_full_text[23:30], stroke_color=title.get_color()),
            Write(props_full_text[34:41], stroke_color=title.get_color()),
            GrowFromCenter(props_full_text[30]),
            GrowFromCenter(props_full_text[41]),
        )
        self.wait()

        # Wrap everything in a vector
        vectors = VGroup(
            MobjectMatrix(
                [
                    Tex(text)
                    for text in [
                        f"k_{{{i}}}",
                        f"t_{{{i}}}",
                        f"k_{{{i}}} x_{{{i}}}",
                        f"k_{{{i}}} y_{{{i}}}"
                    ]
                ],
                n_cols=1,
            ).set_color(color)
            for i, color in zip([1, 2], CIRCLE_COLORS[:2])
        )
        vector_c1, vector_c2 = vectors
        k1, t1, kx1, ky1 = vector_c1.get_entries()
        k2, t2, kx2, ky2 = vector_c2.get_entries()
        for vector, ref in zip(vectors, [props_c1, props_c2]):
            vector.move_to(ref)
        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        *[
                            ReplacementTransform(source, target)
                            for source, target in zip(props, vector)
                        ],
                        FadeIn(vector.get_brackets()[0], shift=RIGHT, scale=0.5),
                        FadeIn(vector.get_brackets()[1], shift=LEFT, scale=0.5),
                    )
                    for props, vector in ([props_c1, vector_c1], [props_c2, vector_c2])
                ],
                lag_ratio=0.5,
                run_time=3,
            ),
            FadeOut(
                title,
                run_time=2,
            ),
        )
        self.wait()

        notes = VGroup()
        for i, vector in enumerate(vectors):
            note = TexText(f"圆$C_{{{i + 1}}}$的 \\\\ 向量表示 \\\\[0.5em] $\\mathbf{{c}}_{{{i + 1}}}$")
            note.next_to(vector, UP, buff=0.5)
            note[:3].match_color(vector)
            note[-2:].match_color(vector)
            note.set_opacity(0)
            notes.add(note)
        note_c1, note_c2 = notes
        vector_group = VGroup(notes, vectors)
        vector_group.generate_target()
        vector_group.target.set_opacity(1)
        vector_group.target.center().to_edge(RIGHT)
        self.add(vector_group)
        self.play(MoveToTarget(vector_group))
        self.wait()

        # Circle -> Vector
        plane = LeftSideCoordinatePlane(include_labels=False)
        c1 = Circle(radius=1.7, arc_center=plane.c2p(-1.570, 0.570))
        c2 = Circle(radius=2.3, arc_center=plane.c2p(1.45, -0.45))
        dot_c1 = Dot(c1.get_center())
        dot_c2 = Dot(c2.get_center())
        name_c1 = TexText("圆$C_{1}$").next_to(c1, UP).to_edge(UP)
        name_c2 = TexText("圆$C_{2}$").next_to(c2, UP).to_edge(UP)
        info_c1 = TexText("圆心$(x_{1},y_{1})$ \\\\ 半径$r_{1}$")
        info_c1.scale(0.8).next_to(dot_c1, DOWN)
        info_c2 = TexText("圆心$(x_{2},y_{2})$ \\\\ 半径$r_{2}$")
        info_c2.scale(0.8).next_to(dot_c2, DOWN)
        for mob in (c1, dot_c1, name_c1, info_c1):
            mob.match_color(vector_c1)
        for mob in (c2, dot_c2, name_c2, info_c2):
            mob.match_color(vector_c2)
        info_c1.set_backstroke()
        info_c2.set_backstroke()

        geometry_group = VGroup(
            plane, c1, c2, dot_c1, dot_c2,
            name_c1, name_c2, info_c1, info_c2,
        )
        vector_cover_rect = Rectangle(
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=0.9,
        )
        geometry_cover_rect = vector_cover_rect.copy()
        for rect, group in zip(
            [vector_cover_rect, geometry_cover_rect],
            [vector_group, geometry_group],
        ):
            rect.set_width(group.get_width() * 1.05, stretch=True)
            rect.set_height(group.get_height() * 1.05, stretch=True)
            rect.move_to(group)
        self.play(
            FadeOut(props_full_text),
            FadeIn(geometry_group),
            FadeIn(vector_cover_rect),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    Indicate(mob, color=None)
                    for mob in (info_c1, info_c2)
                ],
                lag_ratio=0.5,
            ),
            run_time=2,
        )
        self.wait()
        self.play(FadeOut(vector_cover_rect))
        self.wait()

        # Vector -> Circle
        self.play(FadeIn(geometry_cover_rect))
        self.wait()
        center_c1 = info_c1[:-4].copy()
        center_c2 = info_c2[:-4].copy()
        radius_c1 = info_c1[-4:].copy()
        radius_c2 = info_c2[-4:].copy()
        temp_group = VGroup(radius_c1, radius_c2, center_c1, center_c2)
        for mob in temp_group:
            mob.set_backstroke(width=0)
        self.play(
            TransformFromCopy(k1, radius_c1),
            TransformFromCopy(k2, radius_c2),
            run_time=2,
        )
        self.wait()
        self.play(
            TransformFromCopy(VGroup(kx1, ky1), center_c1),
            TransformFromCopy(VGroup(kx2, ky2), center_c2),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeOut(geometry_cover_rect, rate_func=squish_rate_func(smooth, 0, 0.6)),
            FadeOut(temp_group, rate_func=squish_rate_func(smooth, 0.4, 1)),
            run_time=2,
        )
        self.wait()

        # All properties come from the inner product
        geometry_group.remove(name_c1, name_c2)
        ip_title = TexText(
            "圆$C_{1}$与圆$C_{2}$的内积",
            t2c={
                "圆$C_{1}$": c1.get_color(),
                "圆$C_{2}$": c2.get_color(),
            },
            use_labelled_svg=False,
        )
        ip_title.move_to(plane)
        ip_title.next_to(note_c1[:-2], LEFT, coor_mask=[0, 1, 0])
        orig_formula = Tex(
            """
            - \\dfrac{1}{2} \\dfrac{x_{1}^2 + y_{1}^2 - r_{1}^2}{r_{1}} \\dfrac{1}{r_{2}}
                - \\dfrac{1}{2} \\dfrac{1}{r_{1}} \\dfrac{x_{2}^2 + y_{2}^2 - r_{2}^2}{r_{2}}
                + 1 \\dfrac{x_{1}}{r_{1}} \\dfrac{x_{2}}{r_{2}}
                + 1 \\dfrac{y_{1}}{r_{1}} \\dfrac{y_{2}}{r_{2}}
            """,
            use_labelled_svg=False,
        )
        orig_formula.next_to(ip_title, DOWN, buff=0.3)
        orig_formula.set_width(plane.get_width() * 0.95)
        orig_terms_constants = VGroup(
            VGroup(orig_formula[start:end])
            for start, end in zip([0, 22, 44, 56], [4, 26, 46, 58])
        )
        orig_terms_c1 = VGroup(
            VGroup(orig_formula[start:end])
            for start, end in zip([4, 26, 46, 58], [18, 30, 51, 63])
        )
        orig_terms_c2 = VGroup(
            VGroup(orig_formula[start:end])
            for start, end in zip([18, 30, 51, 63], [22, 44, 56, 68])
        )
        orig_terms_c1.match_color(c1)
        orig_terms_c2.match_color(c2)
        self.play(
            ReplacementTransform(name_c1, ip_title[:3]),
            ReplacementTransform(name_c2, ip_title[4:7]),
            Write(VGroup(ip_title[3], ip_title[7:])),
            GrowFromCenter(orig_formula),
            FadeOut(geometry_group),
        )
        self.wait()

        # Rewrite the formula in properties
        prop_formula = Tex(
            """
            - \\dfrac{1}{2} t_{1} k_{2} - \\dfrac{1}{2} k_{1} t_{2}
            + k_{1} x_{1} k_{2} x_{2} + k_{1} y_{1} k_{2} y_{2}
            """,
            use_labelled_svg=False,
        )
        prop_formula.match_width(orig_formula)
        prop_formula.next_to(orig_formula, DOWN, buff=0.8)
        prop_terms_constants = VGroup(
            prop_formula[:4], prop_formula[8:12],
            prop_formula[16], prop_formula[25],
        )
        prop_terms_c1 = VGroup(
            prop_formula[4:6], prop_formula[12:14],
            prop_formula[17:21], prop_formula[26:30],
        )
        prop_terms_c2 = VGroup(
            prop_formula[6:8], prop_formula[14:16],
            prop_formula[21:25], prop_formula[30:],
        )
        prop_terms_c1.match_color(c1)
        prop_terms_c2.match_color(c2)
        ordered_sources = VGroup(
            *[
                mob.copy()
                for group in zip(orig_terms_constants, orig_terms_c1, orig_terms_c2)
                for mob in group
            ]
        )
        ordered_targets = VGroup(
            *[
                mob
                for group in zip(prop_terms_constants, prop_terms_c1, prop_terms_c2)
                for mob in group
            ]
        )
        self.add(ordered_sources)
        orig_formula.set_backstroke(width=0).set_color(GREY).set_opacity(0.5)
        self.play(
            AnimationGroup(
                *[
                    FadeTransform(source, target)
                    for source, target in zip(ordered_sources, ordered_targets)
                ],
                lag_ratio=0.1,
            ),
            run_time=3,
        )
        self.wait()

        # We use properties from now on
        prop_formula.generate_target()
        prop_formula.target.move_to(orig_formula)
        self.play(
            FadeOut(orig_formula),
            MoveToTarget(prop_formula),
        )
        self.wait()

        # Recall dot product
        color_c1, color_c2 = CIRCLE_COLORS[:2]
        color_special = YELLOW
        dot_t2c = dict([
            ("\\mathbf{c}_{1}", color_c1), ("k_{1}", color_c1), ("t_{1}", color_c1),
            ("x_{1}", color_c1), ("y_{1}", color_c1),
            ("\\mathbf{c}_{2}", color_c2), ("k_{2}", color_c2), ("t_{2}", color_c2),
            ("x_{2}", color_c2), ("y_{2}", color_c2),
            ("圆$C_{1}$", color_c1), ("圆$C_{2}$", color_c2),
            ("\\mathbf{I}", color_special), ("\\mathbf{F}", color_special),
        ])
        dot_formula = Tex(
            """
            & \\mathbf{c}_{1}^{\\mathrm{T}} \\mathbf{c}_{2} \\\\
            = {} & \\mathbf{c}_{1} \\cdot \\mathbf{c}_{2} \\\\
            = {} & k_{1} k_{2} + t_{1} t_{2}
            + k_{1} x_{1} k_{2} x_{2} + k_{1} y_{1} k_{2} y_{2}
            """,
            t2c=dot_t2c,
            use_labelled_svg=False,
        )
        dot_formula.match_width(prop_formula)
        dot_formula.next_to(LEFT_SIDE / 2, DOWN)
        dot_formula.next_to(prop_formula, DOWN, coor_mask=[1, 0, 0])
        self.play(
            TransformFromCopy(note_c1[-2:], dot_formula[6:8]),
            TransformFromCopy(note_c2[-2:], dot_formula[9:11]),
            TransformFromCopy(k1, dot_formula[12:14]),
            TransformFromCopy(k2, dot_formula[14:16]),
            TransformFromCopy(t1, dot_formula[17:19]),
            TransformFromCopy(t2, dot_formula[19:21]),
            TransformFromCopy(kx1, dot_formula[22:26]),
            TransformFromCopy(kx2, dot_formula[26:30]),
            TransformFromCopy(ky1, dot_formula[31:35]),
            TransformFromCopy(ky2, dot_formula[35:]),
            AnimationGroup(
                *[
                    GrowFromCenter(dot_formula[ind])
                    for ind in (8, 11, 16, 21, 30)
                ],
                lag_ratio=0.2,
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            TransformFromCopy(dot_formula[6], dot_formula[0]),
            TransformFromCopy(dot_formula[8], dot_formula[1]),
            TransformFromCopy(dot_formula[7], dot_formula[2]),
            TransformFromCopy(dot_formula[9:11], dot_formula[3:5]),
            GrowFromCenter(dot_formula[5]),
        )
        self.add(dot_formula)
        self.wait()

        # Same and different parts
        same_color, diff_color = BLUE, PINK
        prop_same_part, prop_diff_part = prop_formula[17:], prop_formula[:16]
        prop_same_rect, prop_diff_rect = VGroup(
            SurroundingRectangle(ref).set_color(color)
            for ref, color in zip(
                (prop_formula[17:], prop_formula[:16]),
                (same_color, diff_color),
            )
        )
        dot_same_part, dot_diff_part = dot_formula[22:], dot_formula[12:21]
        dot_same_rect, dot_diff_rect = VGroup(
            SurroundingRectangle(ref).set_color(color)
            for ref, color in zip(
                (dot_formula[22:], dot_formula[12:21]),
                (same_color, diff_color),
            )
        )
        self.play(
            ShowCreation(prop_same_rect),
            ShowCreation(dot_same_rect),
        )
        self.wait()
        self.play(
            ReplacementTransform(prop_same_rect, prop_diff_rect),
            ReplacementTransform(dot_same_rect, dot_diff_rect),
        )
        self.wait()
        self.play(
            FadeOut(prop_diff_rect),
            FadeOut(dot_diff_rect),
        )
        self.wait()

        # Insert an identity matrix
        dot_id_formula = Tex(
            """
            & \\mathbf{c}_{1}^{\\mathrm{T}} \\mathbf{I} \\mathbf{c}_{2} \\\\
            = {} & \\mathbf{c}_{1} \\cdot \\mathbf{c}_{2} \\\\
            = {} & k_{1} k_{2} + t_{1} t_{2}
            + k_{1} x_{1} k_{2} x_{2} + k_{1} y_{1} k_{2} y_{2}
            """,
            t2c=dot_t2c,
            use_labelled_svg=False,
        )
        dot_id_formula.match_width(dot_formula)
        dot_id_formula.move_to(dot_formula)
        self.play(
            ReplacementTransform(dot_formula[:3], dot_id_formula[:3]),
            GrowFromCenter(dot_id_formula[3]),
            FocusOn(dot_id_formula[3], color=color_special),
            ReplacementTransform(dot_formula[3:], dot_id_formula[4:]),
            run_time=2,
        )
        self.wait()

        # 1. Expand it out
        linalg_group_scale_factor = 0.8
        linalg_group_buff = 0.8
        matrix_product_buff = 0.5
        inner_product_target = TexText(
            """
            目标内积
            $ = - \\dfrac{1}{2} t_{1} k_{2} - \\dfrac{1}{2} k_{1} t_{2}
            + k_{1} x_{1} k_{2} x_{2} + k_{1} y_{1} k_{2} y_{2} $
            """,
            t2c=dot_t2c,
            use_labelled_svg=False,
        )
        inner_product_target.to_edge(UP).to_edge(LEFT)
        inner_product_target.to_corner(UL)

        vector_c1_trans_1 = MobjectMatrix(
            [Tex(text) for text in ["k_{1}", "t_{1}", "k_{1} x_{1}", "k_{1} y_{1}"]],
            n_rows=1, height=None, h_buff=0.3,
        )
        vector_c1_trans_1.set_color(color_c1)
        matrix_1 = TexMatrix(
            np.identity(4, dtype=int),
            element_alignment_corner=ORIGIN,
            h_buff=0.8, v_buff=0.6,
            bracket_v_buff=0.5,
        )
        matrix_1.set_color(color_special)
        vector_c2_1 = MobjectMatrix(
            [Tex(text) for text in ["k_{2}", "t_{2}", "k_{2} x_{2}", "k_{2} y_{2}"]],
            n_cols=1, height=None,
        )
        vector_c2_1.set_color(color_c2)
        linalg_group_1 = VGroup(vector_c1_trans_1, matrix_1, vector_c2_1)
        linalg_group_1.arrange(RIGHT)
        linalg_group_1.scale(linalg_group_scale_factor)
        linalg_group_1.center().next_to(inner_product_target, DOWN, buff=linalg_group_buff, coor_mask=[0, 1, 0])
        matrix_product_1 = TexText(
            """
            矩阵乘积
            $ = k_{1} k_{2} + t_{1} t_{2} + k_{1} x_{1} k_{2} x_{2} + k_{1} y_{1} k_{2} y_{2}$
            """,
            t2c=dot_t2c,
            use_labelled_svg=False,
        )
        matrix_product_1.next_to(linalg_group_1, DOWN, buff=matrix_product_buff)
        matrix_product_1.to_edge(LEFT)

        self.play(
            AnimationGroup(
                FadeOut(notes),
                run_time=1,
            ),
            AnimationGroup(
                # Target part
                FadeTransform(ip_title[:8], inner_product_target[:2]),
                ReplacementTransform(ip_title[8:], inner_product_target[2:4]),
                GrowFromPoint(inner_product_target[4], prop_formula.get_left()),
                ReplacementTransform(prop_formula, inner_product_target[5:]),
                # Current result part
                FadeTransform(dot_id_formula[6:12], matrix_product_1[:4]),
                ReplacementTransform(dot_id_formula[12:], matrix_product_1[4:]),
                # Vector part
                *[
                    AnimationGroup(
                        FadeTransform(source.get_brackets()[0], target.get_brackets()[0]),
                        ReplacementTransform(source.get_entries(), target.get_entries()),
                        FadeTransform(source.get_brackets()[1], target.get_brackets()[1]),
                    )
                    for source, target in zip(
                        (vector_c1, vector_c2),
                        (vector_c1_trans_1, vector_c2_1),
                    )
                ],
                # Matrix part
                GrowFromPoint(matrix_1, dot_id_formula[3].get_center()),
                # Misc part
                dot_id_formula[:3].animate.set_opacity(0).move_to(vector_c1_trans_1),
                dot_id_formula[3].animate.set_opacity(0).move_to(matrix_1),
                dot_id_formula[4:6].animate.set_opacity(0).move_to(vector_c2_1),
                run_time=3,
            ),
        )
        self.remove(dot_id_formula)
        self.wait()

        # 2. Swap first two rows
        swap_path_arc_2 = PI / 2
        vector_c1_trans_2 = MobjectMatrix(
            [Tex(text) for text in ["k_{1}", "t_{1}", "k_{1} x_{1}", "k_{1} y_{1}"]],
            n_rows=1, height=None, h_buff=0.3,
        )
        vector_c1_trans_2.set_color(color_c1)
        matrix_2 = TexMatrix(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            element_alignment_corner=ORIGIN,
            h_buff=0.8, v_buff=0.6,
            bracket_v_buff=0.5,
        )
        matrix_2.set_color(color_special)
        vector_c2_2 = MobjectMatrix(
            [Tex(text) for text in ["k_{2}", "t_{2}", "k_{2} x_{2}", "k_{2} y_{2}"]],
            n_cols=1, height=None,
        )
        vector_c2_2.set_color(color_c2)
        linalg_group_2 = VGroup(vector_c1_trans_2, matrix_2, vector_c2_2)
        linalg_group_2.arrange(RIGHT)
        linalg_group_2.scale(linalg_group_scale_factor)
        linalg_group_2.center().next_to(inner_product_target, DOWN, buff=linalg_group_buff, coor_mask=[0, 1, 0])
        matrix_product_2 = TexText(
            """
            矩阵乘积
            $ = t_{1} k_{2} + k_{1} t_{2} + k_{1} x_{1} k_{2} x_{2} + k_{1} y_{1} k_{2} y_{2}$
            """,
            t2c=dot_t2c,
            use_labelled_svg=False,
        )
        matrix_product_2.next_to(linalg_group_2, DOWN, buff=matrix_product_buff)
        matrix_product_2.to_edge(LEFT)
        self.play(
            # No change part
            ReplacementTransform(vector_c1_trans_1, vector_c1_trans_2),
            ReplacementTransform(vector_c2_1, vector_c2_2),
            # Matrix part
            ReplacementTransform(matrix_1.get_brackets(), matrix_2.get_brackets()),
            ReplacementTransform(matrix_1.get_row(0), matrix_2.get_row(1), path_arc=swap_path_arc_2),
            ReplacementTransform(matrix_1.get_row(1), matrix_2.get_row(0), path_arc=swap_path_arc_2),
            ReplacementTransform(matrix_1.get_row(2), matrix_2.get_row(2)),
            ReplacementTransform(matrix_1.get_row(3), matrix_2.get_row(3)),
            # Current result part
            ReplacementTransform(matrix_product_1[:5], matrix_product_2[:5]),
            ReplacementTransform(matrix_product_1[5:7], matrix_product_2[10:12], path_arc=swap_path_arc_2),
            ReplacementTransform(matrix_product_1[7:10], matrix_product_2[7:10]),
            ReplacementTransform(matrix_product_1[10:12], matrix_product_2[5:7], path_arc=swap_path_arc_2),
            ReplacementTransform(matrix_product_1[12:], matrix_product_2[12:]),
            run_time=2,
        )
        self.wait()

        # 3. Add in coefficients
        vector_c1_trans_3 = MobjectMatrix(
            [Tex(text) for text in ["k_{1}", "t_{1}", "k_{1} x_{1}", "k_{1} y_{1}"]],
            n_rows=1, height=None, h_buff=0.3,
        )
        vector_c1_trans_3.set_color(color_c1)
        matrix_3 = TexMatrix(
            [
                [0, "-\\tfrac{1}{2}", 0, 0],
                ["-\\tfrac{1}{2}", 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            element_alignment_corner=ORIGIN,
            h_buff=0.5, v_buff=0.3,
            bracket_v_buff=0.5,
        )
        matrix_3.set_color(color_special)
        vector_c2_3 = MobjectMatrix(
            [Tex(text) for text in ["k_{2}", "t_{2}", "k_{2} x_{2}", "k_{2} y_{2}"]],
            n_cols=1, height=None,
        )
        vector_c2_3.set_color(color_c2)
        linalg_group_3 = VGroup(vector_c1_trans_3, matrix_3, vector_c2_3)
        linalg_group_3.arrange(RIGHT)
        linalg_group_3.scale(linalg_group_scale_factor)
        linalg_group_3.center().next_to(inner_product_target, DOWN, buff=linalg_group_buff, coor_mask=[0, 1, 0])
        matrix_product_3 = TexText(
            """
            矩阵乘积
            $ = - \\dfrac{1}{2} t_{1} k_{2} - \\dfrac{1}{2} k_{1} t_{2}
            + k_{1} x_{1} k_{2} x_{2} + k_{1} y_{1} k_{2} y_{2}$
            """,
            t2c=dot_t2c,
            use_labelled_svg=False,
        )
        matrix_product_3.next_to(linalg_group_3, DOWN, buff=matrix_product_buff)
        matrix_product_3.to_edge(LEFT)
        self.play(
            # No change part
            ReplacementTransform(vector_c1_trans_2, vector_c1_trans_3),
            ReplacementTransform(vector_c2_2, vector_c2_3),
            # Matrix part
            ReplacementTransform(matrix_2.get_brackets(), matrix_3.get_brackets()),
            ReplacementTransform(matrix_2.get_row(0)[0], matrix_3.get_row(0)[0]),
            FadeOut(matrix_2.get_row(0)[1]),
            FadeTransform(inner_product_target[13:17].copy(), matrix_3.get_row(0)[1]),
            ReplacementTransform(matrix_2.get_row(0)[2:], matrix_3.get_row(0)[2:]),
            FadeOut(matrix_2.get_row(1)[0]),
            FadeTransform(inner_product_target[5:9].copy(), matrix_3.get_row(1)[0]),
            ReplacementTransform(matrix_2.get_row(1)[1:], matrix_3.get_row(1)[1:]),
            ReplacementTransform(matrix_2.get_row(2), matrix_3.get_row(2)),
            ReplacementTransform(matrix_2.get_row(3), matrix_3.get_row(3)),
            # Current result part
            ReplacementTransform(matrix_product_2[:5], matrix_product_3[:5]),
            FadeTransform(VectorizedPoint(matrix_product_2[5].get_left()), matrix_product_3[5:9]),
            ReplacementTransform(matrix_product_2[5:9], matrix_product_3[9:13]),
            FadeTransform(matrix_product_2[9], matrix_product_3[13:17]),
            ReplacementTransform(matrix_product_2[10:], matrix_product_3[17:]),
            run_time=2,
        )
        self.remove(linalg_group_2)
        self.add(linalg_group_3)
        self.wait()

        # We finally reach the target
        final_product_1 = TexText(
            """
            目标内积 = 矩阵乘积
            $ = - \\dfrac{1}{2} t_{1} k_{2} - \\dfrac{1}{2} k_{1} t_{2}
            + k_{1} x_{1} k_{2} x_{2} + k_{1} y_{1} k_{2} y_{2}$
            """,
            t2c=dot_t2c,
            use_labelled_svg=False,
        )
        final_product_1.to_edge(UP)
        self.play(
            ReplacementTransform(inner_product_target[:5], final_product_1[:5]),
            ReplacementTransform(inner_product_target[5:], final_product_1[10:]),
            ReplacementTransform(matrix_product_3, final_product_1[5:]),
            linalg_group_3.animate.move_to(DOWN),
            run_time=2,
        )
        self.wait()

        # Compare with -1 and 1
        linalg_group_3.save_state()
        linalg_group_3.generate_target()
        linalg_group_3.target.set_opacity(0.05)
        relation_line = InnerProductRelationLine(
            pointer_color=WHITE,
            axis_label_text="\\text{内积}",
        )

        base_time = self.time
        relation_line.add_updater(
            lambda line: line.set_value(
                0.4 + 2.1 * np.sin(PI / 2 * (self.time - base_time))
            )
        )
        self.play(
            MoveToTarget(linalg_group_3),
            FadeIn(relation_line),
        )
        self.wait(5)
        self.play(
            Restore(linalg_group_3),
            FadeOut(relation_line),
        )
        self.wait()

        # Metric Matrix - F
        label_matF = TexText("度量矩阵 \\\\[0.5em] $\\mathbf{F}$")
        label_matF.set_color(color_special)
        label_matF.next_to(matrix_3, UP)
        self.play(FadeIn(label_matF, shift=UP))
        self.wait()

        label_c1_trans = Tex("\\mathbf{c}_{1}^{\\mathrm{T}}")
        label_c1_trans[::2].set_color(color_c1)
        label_c1_trans.next_to(vector_c1_trans_3, UP)
        label_c2 = Tex("\\mathbf{c}_{2}")
        label_c2.set_color(color_c2)
        label_c2.next_to(vector_c2_3, UP)
        for label in (label_c1_trans, label_c2):
            label.next_to(label_matF, aligned_edge=DOWN, coor_mask=[0, 1, 0])

        # Rewrite a bit
        final_product_2 = TexText(
            """
            目标内积 = $\\mathbf{c}_{1}^{\\mathrm{T}} \\mathbf{F} \\mathbf{c}_{2}$
            $ = - \\dfrac{1}{2} t_{1} k_{2} - \\dfrac{1}{2} k_{1} t_{2}
            + k_{1} x_{1} k_{2} x_{2} + k_{1} y_{1} k_{2} y_{2}$
            """,
            t2c=dot_t2c,
            use_labelled_svg=False,
        )
        final_product_2.to_edge(UP)

        self.play(
            FadeIn(label_c1_trans, shift=UP),
            FadeIn(label_c2, shift=UP),
            ReplacementTransform(final_product_1[:5], final_product_2[:5]),
            FadeOut(final_product_1[5:9], shift=UP),
            FadeIn(final_product_2[5:11], shift=UP),
            ReplacementTransform(final_product_1[9:], final_product_2[11:]),
            run_time=1,
        )
        self.remove(final_product_1)
        self.add(final_product_2)
        self.wait()

        # Fully rewritten the inner product
        final_product_3 = Tex(
            """
            \\langle \\mathbf{c}_{1}, \\mathbf{c}_{2} \\rangle
            = \\mathbf{c}_{1}^{\\mathrm{T}} \\mathbf{F} \\mathbf{c}_{2}
            = - \\dfrac{1}{2} t_{1} k_{2} - \\dfrac{1}{2} k_{1} t_{2}
            + k_{1} x_{1} k_{2} x_{2} + k_{1} y_{1} k_{2} y_{2}
            """,
            t2c=dot_t2c,
            use_labelled_svg=False,
        )
        final_product_3.to_edge(UP)

        self.play(
            AnimationGroup(
                FlashAround(label_c1_trans[::2], color=color_c1),
                FlashAround(label_c2, color=color_c2),
                lag_ratio=0.3,
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeOut(final_product_2[:4], shift=UP),
            ReplacementTransform(final_product_2[4:], final_product_3[7:]),
            TransformFromCopy(label_c1_trans[::2], final_product_3[1:3], path_arc=-PI / 3),
            TransformFromCopy(label_c2, final_product_3[4:6], path_arc=-PI / 3),
            FadeIn(final_product_3[:7:3], shift=UP),
            run_time=2,
        )
        self.remove(final_product_2)
        self.add(final_product_3)
        self.wait()
        self.play(FadeOut(label_matF[:-1]))
        label_matF = label_matF[-1]
        self.add(label_matF)
        self.wait()


class FewExamplesAndASpecialCase(Scene):
    def construct(self):
        # Setup
        color_c1, color_c2 = CIRCLE_COLORS[:2]
        color_special = YELLOW
        basic_t2c = dict([
            ("\\mathbf{c}_{1}", color_c1),
            ("\\mathbf{c}_{2}", color_c2),
            ("\\mathbf{F}", color_special),
            ("{d}", color_special),
            ("r_{1}", color_c1),
            ("r_{2}", color_c2),
            ("{r}", color_c1),
            ("{0}", color_special),
        ])
        plane = LeftSideCoordinatePlane(
            x_range=(-2, 5, 1),
            y_range=(-6, 6, 1),
            unit_size=0.8,
        )
        plane.shift(plane.c2p(-0.5, 0) - plane.c2p(0, 0))
        c1 = VCircle(plane, center=(0, 0), radius=1)
        c1.set_fill(opacity=0.1)
        dot_c1 = Dot(c1.get_center())
        label_c1 = Tex("C_{1}")
        label_c1.scale(0.8)
        label_c1.move_to(c1.get_corner(UL))
        geom_group_c1 = VGroup(c1, dot_c1, label_c1)
        geom_group_c1.set_color(color_c1)
        c2 = VCircle(plane, center=(2, 0), radius=2)
        c2.set_fill(opacity=0.1)
        dot_c2 = Dot(c2.get_center())
        label_c2 = Tex("C_{2}")
        label_c2.scale(0.8)
        label_c2.add_updater(lambda mob: mob.move_to(c2.get_corner(UR)))
        geom_group_c2 = VGroup(c2, dot_c2, label_c2)
        geom_group_c2.set_color(color_c2)

        vector_name_c1 = VGroup(
            Tex("\\mathbf{c}_{1} ="),
            MobjectMatrix(
                [Tex(text) for text in ["k_{1}", "t_{1}", "k_{1} x_{1}", "k_{1} y_{1}"]],
                n_cols=1,
            ),
            Tex("="),
        )
        vector_name_c1.set_color(color_c1)
        vector_name_c2 = VGroup(
            Tex("\\mathbf{c}_{2} ="),
            MobjectMatrix(
                [Tex(text) for text in ["k_{2}", "t_{2}", "k_{2} x_{2}", "k_{2} y_{2}"]],
                n_cols=1,
            ),
            Tex("="),
        )
        vector_name_c2.set_color(color_c2)
        vector_names = VGroup(vector_name_c1, vector_name_c2)
        for vector_name in vector_names:
            vector_name.arrange(RIGHT)
            vector_name.set_height(2)
        vector_names.arrange(RIGHT, buff=2.5)
        vector_names.next_to(plane, RIGHT).to_edge(UP)

        vector_c1 = MobjectMatrix(
            [Tex(text) for text in ["1", "-1", "0", "0"]],
            n_cols=1,
        )
        vector_c1.set_height(2)
        vector_c1.next_to(vector_name_c1, RIGHT)
        vector_c1.set_color(color_c1)
        vector_c2 = PedoeVectorMobject()
        vector_c2.set_height(2)
        vector_c2.next_to(vector_name_c2, RIGHT)
        vector_c2.set_color(color_c2)
        vector_c2.add_updater(
            lambda mob: mob.set_values(c2.get_pedoe_vector())
        )
        vectors = VGroup(vector_c1, vector_c2)

        product_prefix = VGroup(
            Tex(
                """
                \\langle \\mathbf{c}_{1}, \\mathbf{c}_{2} \\rangle =
                \\mathbf{c}_{1}^{\\mathrm{T}} \\mathbf{F} \\mathbf{c}_{2} =
                """,
                t2c=basic_t2c,
                use_labelled_svg=False,
            ),
            TexMatrix(
                [[1, -1, 0, 0]],
                h_buff=0.3,
            ),
            TexMatrix(
                [
                    [0, "-\\tfrac{1}{2}", 0, 0],
                    ["-\\tfrac{1}{2}", 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                element_alignment_corner=ORIGIN,
                h_buff=0.5, v_buff=0.3,
                bracket_v_buff=0.5,
            ),
        )
        product_prefix[1].set_color(color_c1)
        product_prefix[2].set_color(color_special)
        product_prefix.arrange(RIGHT)
        product_prefix.set_height(2)
        product_prefix.next_to(vector_names, DOWN, aligned_edge=LEFT, buff=0.5)
        product_c2_part = vector_c2.copy()
        product_c2_part.next_to(product_prefix, RIGHT)

        result_equal_sign = Tex("=")
        ref = product_prefix[0][-1]
        scale_factor = ref.get_width() / result_equal_sign.get_width()
        result_equal_sign.scale(scale_factor)
        result_equal_sign.next_to(ref, DOWN)
        result_equal_sign.next_to(product_prefix, DOWN, coor_mask=[0, 1, 0], buff=0.5)
        result = DecimalEntry(0, edge_to_fix=LEFT)
        result.scale(scale_factor * 1.5)
        result.next_to(result_equal_sign, RIGHT)

        def update_result(mob):
            mob.set_value(get_circle_product(c1, c2))
            curr_ip = mob.get_value()
            if abs(curr_ip - 1) < 1e-6:
                relation_type = "tangent_in"
            elif abs(curr_ip + 1) < 1e-6:
                relation_type = "tangent_out"
            elif -1 < curr_ip < 1:
                relation_type = "intersect"
            elif curr_ip > 1:
                relation_type = "inner"
            else:
                relation_type = "outer"
            mob.set_color(RELATION_COLORS[relation_type])
        result.add_updater(update_result)

        relation_text = VGroup(
            TexText(text).set_color(RELATION_COLORS[relation_type])
            for text, relation_type in zip(
                ["（内切）", "（外切）", "（相交）", "（内含）", "（外离）"],
                ["tangent_in", "tangent_out", "intersect", "inner", "outer"],
            )
        )
        relation_text.scale(scale_factor * 1.5)
        relation_text.next_to(result, RIGHT, buff=0.4)

        def update_relation_text(mob):
            mob.set_opacity(0)
            curr_ip = result.get_value()
            if abs(curr_ip - 1) < 1e-6:
                ind = 0
                relation_type = "tangent_in"
            elif abs(curr_ip + 1) < 1e-6:
                ind = 1
                relation_type = "tangent_out"
            elif -1 < curr_ip < 1:
                ind = 2
                relation_type = "intersect"
            elif curr_ip > 1:
                ind = 3
                relation_type = "inner"
            else:
                ind = 4
                relation_type = "outer"
            mob[ind].set_opacity(1)
        relation_text.add_updater(update_relation_text)

        setup_group = VGroup(
            plane, geom_group_c1, geom_group_c2,
            vector_names, product_prefix,
            vector_c1, vector_c2, product_c2_part,
            result_equal_sign, result,
            relation_text,
        )
        static_group = VGroup(dot_c1, c1)
        motion_group = VGroup(dot_c2, c2)
        self.add(setup_group)
        self.bring_to_front(geom_group_c1, geom_group_c2)
        self.wait()

        # Introduce c1, c2 and their inner product
        setup_group.suspend_updating()
        self.play(
            Indicate(static_group),
            Animation(motion_group),
        )
        self.wait()
        self.play(Indicate(motion_group))
        self.wait()
        setup_group.resume_updating()
        self.play(
            AnimationGroup(
                FlashAround(VGroup(vector_name_c1, vector_c1), color=color_c1),
                FlashAround(VGroup(vector_name_c2, vector_c2), color=color_c2),
                lag_ratio=0.3,
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            FlashUnder(VGroup(result, relation_text), color=RELATION_COLORS["intersect"]),
        )
        self.wait()

        # Case 1: externally tangent
        unit_right = plane.c2p(1, 0) - plane.c2p(0, 0)
        self.play(
            ApplyMethod(
                motion_group.shift,
                unit_right,
                rate_func=lambda t: overshoot(t, 1.5),
            ),
            run_time=2,
        )
        self.wait()

        # Case 2: internally tangent
        self.play(
            ApplyMethod(
                motion_group.shift,
                -2 * unit_right,
                rate_func=lambda t: overshoot(t, 1.5),
            ),
            run_time=3,
        )
        self.wait()

        # Special case: coincidence
        c2.generate_target()
        c2.target.match_height(c1).move_to(c1)
        self.play(
            dot_c2.animate.move_to(dot_c1.get_center()),
            MoveToTarget(c2),
            run_time=2,
        )
        self.wait()

        ip_orig = Tex(
            """
            \\langle \\mathbf{c}_{1}, \\mathbf{c}_{2} \\rangle
            = \\dfrac{ r_{1}^2 + r_{2}^2 - {d}^2}{2 r_{1} r_{2}}
            """,
            t2c=basic_t2c, use_labelled_svg=False,
        )
        ip_coincide = Tex(
            """
            \\langle \\mathbf{c}_{1}, \\mathbf{c}_{1} \\rangle
            = \\dfrac{ r_{1}^2 + r_{1}^2 - {0}^2}{2 r_{1} r_{1}}
            """,
            t2c=basic_t2c, use_labelled_svg=False,
        )

        ref_point = plane.c2p(0, -2)
        for ip in (ip_orig, ip_coincide):
            ip.scale(0.7)
            ip.next_to(ref_point, RIGHT)
        alt_label_c2 = label_c1.copy().move_to(label_c2)
        alt_vector_name_c2 = vector_name_c1.copy().move_to(vector_name_c2)
        alt_prefix = Tex(
            """
                \\langle \\mathbf{c}_{1}, \\mathbf{c}_{1} \\rangle =
                \\mathbf{c}_{1}^{\\mathrm{T}} \\mathbf{F} \\mathbf{c}_{1} =
                """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        alt_prefix.match_height(product_prefix[0])
        alt_prefix.move_to(product_prefix[0])
        relation_coincide = TexText("（重合）")
        relation_coincide.match_height(relation_text)
        relation_coincide.match_color(relation_text[0])
        relation_coincide.move_to(relation_text)
        self.play(GrowFromPoint(ip_orig, ref_point))
        self.wait()
        relation_text.suspend_updating()
        self.play(
            FadeTransform(label_c2, alt_label_c2),
            motion_group.animate.set_color(color_c1),
            FadeTransform(vector_name_c2, alt_vector_name_c2),
            vector_c2.animate.set_color(color_c1),
            FadeTransform(product_prefix[0], alt_prefix),
            product_c2_part.animate.set_color(color_c1),
            FadeTransform(ip_orig, ip_coincide),
            FadeTransform(relation_text, relation_coincide),
            run_time=2,
        )
        self.wait()

        result_group = VGroup(setup_group[-3:-1], relation_coincide)
        coincide_group = VGroup(ip_coincide, result_group)
        coincide_group.generate_target()
        coincide_group.target.arrange(RIGHT)
        right_space_center = (plane.get_right() + RIGHT_SIDE) / 2
        coincide_group.target.move_to(right_space_center)
        coincide_group.target.next_to(product_prefix, DOWN, buff=0.5, coor_mask=[0, 1, 0])
        self.play(MoveToTarget(coincide_group))
        self.wait()


class WhyDoWeTreatCirclesAsVectors(Scene):
    def construct(self):
        # Setup plane
        color_c1, color_c2 = CIRCLE_COLORS[:2]
        color_special = YELLOW
        basic_t2c = dict([
            ("\\mathbf{c}_{1}", color_c1),
            ("\\mathbf{c}_{2}", color_c2),
            ("\\mathbf{F}", color_special),
            ("{d}", color_special),
            ("r_{1}", color_c1),
            ("r_{2}", color_c2),
            ("{r}", color_c1),
            ("{0}", color_special),
        ])
        plane = LeftSideCoordinatePlane(
            x_range=(-2, 5, 1),
            y_range=(-6, 6, 1),
            unit_size=0.8,
        )
        plane.shift(plane.c2p(-0.5, 0) - plane.c2p(0, 0))
        right_space_center = (plane.get_right() + RIGHT_SIDE) / 2
        self.add(plane)

        # Setup circle movement
        period = 30.0
        omega = TAU / period
        left_limit = -FRAME_WIDTH / 2 + 0.15
        right_limit = left_limit + 4.5
        top_limit = FRAME_HEIGHT / 2 - 0.18
        bottom_limit = -FRAME_HEIGHT / 2 + 0.5
        edge_buff = 0.10

        def smootherstep(x):
            x = np.clip(x, 0, 1)
            return 6 * x**5 - 15 * x**4 + 10 * x**3

        def map_to_interval(u, low, high,):
            if high <= low:
                return 0.5 * (low + high)
            alpha = (u + 1) / 2
            return interpolate(low, high, alpha)

        def get_relation_parameter(t):
            t = t % period
            keyframes = [
                # time, q (relation parameter)
                # ---------- external ----------
                (0.0, 1.35),
                # approach external tangency
                (3.8, 1.00),
                # external tangency hold
                (4.9, 1.00),
                # ---------- intersection ----------
                # internal tangency
                (9.2, 0.00),
                # internal tangency hold
                (10.3, 0.00),
                # ---------- containment ----------
                # clearly contained
                (14.0, -0.18),
                # return to internal tangency
                (16.7, 0.00),
                # internal tangency hold
                (17.8, 0.00),
                # ---------- intersection ----------
                # external tangency again
                (22.0, 1.00),
                # external tangency hold
                (23.1, 1.00),
                # ---------- external ----------
                # exactly match beginning
                (30.0, 1.35),
            ]
            for i in range(len(keyframes) - 1):
                t1, q1 = keyframes[i]
                t2, q2 = keyframes[i + 1]
                if t1 <= t <= t2:
                    # Constant section
                    if q1 == q2:
                        return q1
                    alpha = ((t - t1) / (t2 - t1))
                    alpha = smootherstep(alpha)
                    return interpolate(q1, q2, alpha)
            return keyframes[-1][1]

        def get_radii(t):
            s = omega * t
            r1 = 1.20
            + 0.18 * np.sin(s + 0.30)
            + 0.05 * np.sin(2 * s - 0.70)
            r2 = 0.55
            + 0.10 * np.sin(s - 1.10)
            + 0.04 * np.cos(2 * s + 0.40)
            return r1, r2

        def get_direction(t):
            s = omega * t
            theta = 0.52 * np.sin(s + 0.40)
            + 0.20 * np.sin(2 * s - 1.20)
            + 0.08 * np.sin(3 * s + 0.60)

            return np.array([np.cos(theta), np.sin(theta), 0.0])

        def get_drift(t):
            s = omega * t
            ux = 0.58 * np.sin(s + 0.15)
            + 0.27 * np.sin(2 * s - 1.0)
            + 0.15 * np.cos(3 * s + 0.45)
            uy = 0.55 * np.sin(s - 0.85)
            + 0.30 * np.cos(2 * s + 0.60)
            + 0.15 * np.sin(3 * s - 0.30)
            return ux, uy

        def get_state(t):
            t = t % period
            r1, r2 = get_radii(t)
            q = get_relation_parameter(t)
            outer_distance = r1 + r2
            inner_distance = abs(r1 - r2)
            d = (inner_distance + q * (outer_distance - inner_distance))
            d = max(d, 0.04)
            direction = get_direction(t)
            offset_1 = -0.30 * d * direction
            offset_2 = 0.70 * d * direction
            x_low = max(
                left_limit + r1 + edge_buff - offset_1[0],
                left_limit + r2 + edge_buff - offset_2[0],
            )
            x_high = min(
                right_limit - r1 - edge_buff - offset_1[0],
                right_limit - r2 - edge_buff - offset_2[0],
            )
            y_low = max(
                bottom_limit + r1 + edge_buff - offset_1[1],
                bottom_limit + r2 + edge_buff - offset_2[1],
            )
            y_high = min(
                top_limit - r1 - edge_buff - offset_1[1],
                top_limit - r2 - edge_buff - offset_2[1],
            )
            ux, uy = get_drift(t)
            center_x = map_to_interval(ux, x_low, x_high)
            center_y = map_to_interval(uy, y_low, y_high)
            common_center = np.array([center_x, center_y, 0.0])
            c1 = common_center + offset_1
            c2 = common_center + offset_2
            return (c1, r1, c2, r2)

        driver = VectorizedPoint()
        driver.time = 0.0

        def update_driver(mob, dt):
            mob.time = (mob.time + dt) % period
        driver.add_updater(update_driver)

        ghost_plane = NumberPlane()
        circle_1 = VCircle(ghost_plane, center=(0, 0), radius=1)

        def update_circle_1(m):
            c1, r1, _, _ = get_state(driver.time)
            new_circle = VCircle(ghost_plane, center=c1, radius=r1)
            new_circle.set_stroke(RED, width=5)
            new_circle.set_fill(RED, opacity=0.1)
            m.become(new_circle)
        circle_1.add_updater(update_circle_1)

        circle_2 = VCircle(ghost_plane, center=(0, 0), radius=1)

        def update_circle_2(m):
            _, _, c2, r2 = get_state(driver.time)
            new_circle = VCircle(ghost_plane, center=c2, radius=r2)
            new_circle.set_stroke(GREEN, width=5)
            new_circle.set_fill(GREEN, opacity=0.1)
            m.become(new_circle)
        circle_2.add_updater(update_circle_2)

        self.add(driver)
        self.add(circle_1, circle_2)

        # Setup summary
        inner_product_formula = Tex(
            """
            \\text{内积} \\langle \\mathbf{c}_{1}, \\mathbf{c}_{2} \\rangle & =
            \\dfrac{ r_{1}^2 + r_{2}^2 - {d}^2}{2 r_{1} r_{2}} \\\\[0.5em]
            & = \\mathbf{c}_{1}^{\\mathrm{T}} \\mathbf{F} \\mathbf{c}_{2} \\\\[0em]
            & = \\begin{bmatrix} k_{1} & t_{1} & k_{1} x_{1} & k_{1} y_{1} \\end{bmatrix}
            \\begin{bmatrix} 0 & -\\tfrac{1}{2} & 0 & 0 \\\\ -\\tfrac{1}{2} & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & 1 \\end{bmatrix}
            \\begin{bmatrix} k_{2} \\\\ t_{2} \\\\ k_{2} x_{2} \\\\ k_{2} y_{2} \\end{bmatrix}
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        inner_product_formula.set_width(FRAME_WIDTH * 0.6)
        inner_product_formula.move_to(right_space_center)
        inner_product_formula.to_edge(UP)
        inner_product_formula[34:48].set_color(color_c1)
        inner_product_formula[48:78].set_color(color_special)
        inner_product_formula[78:].set_color(color_c2)

        self_inner_product = Tex(
            """
            \\langle \\mathbf{c}_{1}, \\mathbf{c}_{1} \\rangle =
            \\langle \\mathbf{c}_{2}, \\mathbf{c}_{2} \\rangle =
            1
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        self_inner_product.move_to(right_space_center)
        self_inner_product.next_to(3 * DOWN, UP, buff=0, coor_mask=[0, 1, 0])

        def get_inner_product(t):
            c1, r1, c2, r2 = get_state(t)
            d = np.linalg.norm(c1 - c2)
            return (r1**2 + r2**2 - d**2) / (2 * r1 * r2)

        relation_line = InnerProductRelationLine(
            unit_size=1.2,
            x_min=-2.5,
            x_max=2.5,
            initial_val=0.0,
            axis_label_text="\\text{内积}",
            pointer_color=WHITE,
        )
        relation_line.move_to(right_space_center)
        relation_line.next_to(self_inner_product, UP, buff=0.8, coor_mask=[0, 1, 0])
        relation_line.add_updater(
            lambda line: line.set_value(
                get_inner_product(driver.time)
            )
        )
        relation_rect = Rectangle()
        text_labels = relation_line.static_annotations[-1]

        def update_relation_rect(rect):
            # reg_outer, sub_neg1, reg_intersect, sub_pos1, reg_inner
            value = relation_line.get_value()
            if abs(value - 1) < 1e-4:
                ind = 3
            elif abs(value + 1) < 1e-4:
                ind = 1
            elif value < -1:
                ind = 0
            elif value > 1:
                ind = 4
            else:
                ind = 2
            target = text_labels[ind]
            rect.surround(target)
            rect.set_color(target.get_fill_color())
        relation_rect.add_updater(update_relation_rect)
        relation_line.add_updater(
            lambda line: line.pointer.match_color(relation_rect)
        )
        self.add(relation_line, relation_rect)
        self.add(self_inner_product)
        self.add(inner_product_formula)

        # Cycle animation
        self.wait(period)

        # Few indications
        geom_form = inner_product_formula[10:26]
        matrix_form = VGroup(
            inner_product_formula[27:32],
            inner_product_formula[34:],
        )
        relation_group = VGroup(relation_line, relation_rect)
        algebra_group = VGroup(inner_product_formula, relation_line, self_inner_product)
        for mob in (geom_form, matrix_form, relation_group, self_inner_product, algebra_group):
            self.play(FlashAround(mob), run_time=2)
            self.wait()


class MassProducedInnerProduct(Scene):
    def construct(self):
        # Setup
        color_c1, color_c2, color_c3, color_c4 = CIRCLE_COLORS
        color_special = YELLOW
        basic_t2c = {
            key: color
            for i, color in enumerate(CIRCLE_COLORS, start=1)
            for key in [
                f"\\mathbf{{c}}_{{{i}}}",
                f"k_{{{i}}}",
                f"t_{{{i}}}",
                f"x_{{{i}}}",
                f"y_{{{i}}}",
                f"{{{i}}}",
            ]
        }
        basic_t2c["\\mathbf{F}"] = color_special
        formula_template = Tex(
            """
            \\langle \\mathbf{c}_{1}, \\mathbf{c}_{2} \\rangle
            = \\mathbf{c}_{1}^{\\mathrm{T}} \\mathbf{F} \\mathbf{c}_{2}
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        formula_template.scale(2.0)
        static_part = VGroup(
            formula_template[ind]
            for ind in (0, 3, 6, 7, 9, 11)
        )
        lhs_left = formula_template[1:3]
        bracket_left_slot = lhs_left.get_center()
        lhs_right = formula_template[4:6]
        bracket_right_slot = lhs_right.get_center()
        rhs_left = formula_template[8:11:2]
        vector_left_slot = rhs_left.get_center()
        rhs_right = formula_template[12:]
        vector_right_slot = rhs_right.get_center()
        self.add(formula_template)
        self.wait()

        # The inner product is structured
        shift_vec = 0.8 * UP
        curr_left_ind = 1
        curr_right_ind = 2

        def make_c(ind):
            return Tex(
                f"\\mathbf{{c}}_{{{ind}}}",
                t2c=basic_t2c,
                use_labelled_svg=False,
            ).scale(2.0)

        def change_expression(left_index=None, right_index=None, run_time=1, extra_anims=None):
            nonlocal lhs_left, lhs_right, rhs_left, rhs_right
            nonlocal curr_left_ind, curr_right_ind
            anims = []
            if (left_index is not None) and (left_index != curr_left_ind):
                new_lhs_left = make_c(left_index).move_to(bracket_left_slot)
                new_rhs_left = make_c(left_index).move_to(vector_left_slot)
                for anim in [
                    FadeOut(lhs_left, shift=-shift_vec),
                    FadeOut(rhs_left, shift=-shift_vec),
                    FadeIn(new_lhs_left, shift=-shift_vec),
                    FadeIn(new_rhs_left, shift=-shift_vec),
                ]:
                    anims.append(anim)
                lhs_left = new_lhs_left
                rhs_left = new_rhs_left
                curr_left_ind = left_index
            if (right_index is not None) and (right_index != curr_right_ind):
                new_lhs_right = make_c(right_index).move_to(bracket_right_slot)
                new_rhs_right = make_c(right_index).move_to(vector_right_slot)
                for anim in [
                    FadeOut(lhs_right, shift=shift_vec),
                    FadeOut(rhs_right, shift=shift_vec),
                    FadeIn(new_lhs_right, shift=shift_vec),
                    FadeIn(new_rhs_right, shift=shift_vec),
                ]:
                    anims.append(anim)
                lhs_right = new_lhs_right
                rhs_right = new_rhs_right
                curr_right_ind = right_index
            if extra_anims is not None:
                for anim in extra_anims:
                    anims.append(anim)
            self.play(*anims, run_time=run_time)

        pool = [(1, 3), (1, 4), (3, 2), (4, 2), (3, 3), (3, 1)]
        for k, (left_index, right_index) in enumerate(pool):
            if k == 2:
                ref = formula_template[11]
                arrow = Vector(DOWN)
                arrow.next_to(ref, UP)
                no_change_text = TexText("不会改变")
                no_change_text.next_to(arrow, UP)
                extra_group = VGroup(arrow, no_change_text)
                extra_group.set_color(color_special)
                extra_anims = [FadeInFromPoint(extra_group, ref.get_top())]
            else:
                extra_anims = None
            change_expression(left_index, right_index, extra_anims=extra_anims)
            self.wait()
        curr_formula = VGroup(static_part, lhs_left, lhs_right, rhs_left, rhs_right, extra_group)

        # Here are 4 circles, and we want to know the relationship of every possible pair of circles
        initial_circles_data = [
            {"center": (-5.0, 0.0), "radius": 2.00},
            {"center": (-3.8, 0.0), "radius": 0.80},
            {"center": (-6.25, 1.1), "radius": 1.15},
            {"center": (-5.4, -0.9), "radius": 0.60},
        ]
        ghost_plane = NumberPlane()
        four_circles = VGroup(
            VCircle(
                ghost_plane,
                center=data["center"],
                radius=data["radius"],
            ).set_color(color)
            for data, color in zip(initial_circles_data, CIRCLE_COLORS)
        )
        four_circles.to_edge(LEFT)
        four_circles.save_state()

        four_circles.set_height(FRAME_HEIGHT - 3).center()
        self.play(
            FadeOut(curr_formula, rate_func=squish_rate_func(smooth, 0, 0.4)),
            AnimationGroup(
                *[
                    GrowFromCenter(circle)
                    for circle in four_circles
                ],
                lag_ratio=0.1,
                rate_func=squish_rate_func(smooth, 0.2, 1)
            ),
            run_time=2,
        )
        self.add(four_circles)
        self.wait()

        c1, c2, c3, c4 = four_circles
        vectors = VGroup(
            VGroup(
                Tex(f"\\mathbf{{c}}_{{{i + 1}}} = "),
                MobjectMatrix(
                    [
                        Tex(text)
                        for text in [
                            f"k_{{{i + 1}}}",
                            f"t_{{{i + 1}}}",
                            f"k_{{{i + 1}}} x_{{{i + 1}}}",
                            f"k_{{{i + 1}}} y_{{{i + 1}}}"
                        ]
                    ],
                    n_cols=1,
                )
            ).arrange(RIGHT).set_color(color)
            for i, color in enumerate(CIRCLE_COLORS)
        )
        vectors.arrange(RIGHT, buff=1)
        vectors.set_width(8)
        vectors.move_to(2.2 * RIGHT)
        vectors.to_edge(UP)
        self.play(
            Restore(four_circles, rate_func=squish_rate_func(smooth, 0, 0.5)),
            AnimationGroup(
                *[
                    FadeIn(vector, shift=UP)
                    for vector in vectors
                ],
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.wait()

        # Configuration matrix C
        string_table_C = np.array([
            ["k_{1}", "k_{2}", "k_{3}", "k_{4}"],
            ["t_{1}", "t_{2}", "t_{3}", "t_{4}"],
            ["k_{1} x_{1}", "k_{2} x_{2}", "k_{3} x_{3}", "k_{4} x_{4}"],
            ["k_{1} y_{1}", "k_{2} y_{2}", "k_{3} y_{3}", "k_{4} y_{4}"],
        ])
        name_C = TexText("构型矩阵 $\\mathbf{C}$ =")
        name_C[:-1].set_color(color_special)
        matrix_C = TexMatrix(
            string_table_C,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        config_group = VGroup(name_C, matrix_C)
        config_group.arrange(RIGHT)
        config_group.match_width(vectors)
        config_group.next_to(vectors, DOWN, buff=1)
        matrix_C.generate_target()
        matrix_C.move_to(config_group)
        self.play(
            AnimationGroup(
                *[
                    TransformFromCopy(
                        vectors[ind][1].get_column(0), matrix_C.get_column(ind),
                        rate_func=squish_rate_func(smooth, 0, 0.7),
                    )
                    for ind in range(4)
                ],
                lag_ratio=0.05,
            ),
            AnimationGroup(
                FadeIn(matrix_C.get_brackets()[0], shift=RIGHT, rate_func=squish_rate_func(smooth, 0.3, 1)),
                FadeIn(matrix_C.get_brackets()[1], shift=LEFT, rate_func=squish_rate_func(smooth, 0.3, 1)),
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            Write(name_C, stroke_color=color_special),
            MoveToTarget(matrix_C),
        )
        self.wait()

        # Structure of C is also very neat
        self.play(
            AnimationGroup(
                *[
                    FlashAround(col)
                    for col in matrix_C.get_columns()
                ],
                lag_ratio=0.05,
            ),
            run_time=3,
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    FlashAround(row)
                    for row in matrix_C.get_rows()
                ],
                lag_ratio=0.05,
            ),
            run_time=3,
        )
        self.wait()

        # First important take of this video: C^T F C = G
        matrix_C_trans = TexMatrix(
            string_table_C.T,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        matrix_C_copy = TexMatrix(
            string_table_C,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        matrix_F = TexMatrix(
            [
                [0, "-\\tfrac{1}{2}", 0, 0],
                ["-\\tfrac{1}{2}", 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            element_alignment_corner=ORIGIN,
            h_buff=0.5, v_buff=0.3,
            bracket_v_buff=0.4,
        )
        matrix_equal_sign = Tex("=")
        matrix_G = TexMatrix(
            [
                [
                    f"\\langle \\mathbf{{c}}_{{{row}}}, \\mathbf{{c}}_{{{col}}} \\rangle"
                    for col in range(1, 5)
                ]
                for row in range(1, 5)
            ],
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        matrix_G_unknown = TexMatrix(
            [["?"] * 4] * 4,
            element_alignment_corner=ORIGIN,
            h_buff=1.7, v_buff=0.65,
            bracket_h_buff=0.8, bracket_v_buff=0.25,
        )
        matrix_F.set_color(color_special)

        # Left hand side C^T F C
        lhs = VGroup(matrix_C_trans, matrix_F, matrix_C_copy)
        lhs.arrange(RIGHT)
        backup_lhs = lhs.copy()
        lhs.set_width(FRAME_WIDTH - 1)
        top_name_C_trans = Tex("\\mathbf{C}^{\\mathrm{T}}")
        top_name_F = Tex("\\mathbf{F}")
        top_name_C = Tex("\\mathbf{C}")
        top_names = VGroup(top_name_C_trans, top_name_F, top_name_C)
        top_names.set_color(color_special)
        for name, matrix in zip(top_names, lhs):
            name.next_to(matrix, UP)
            name.to_edge(UP, buff=1.5)
        for name in top_names[:-1]:
            name.match_y(top_names[-1], DOWN)

        rf_group2 = squish_rate_func(smooth, 0.1, 0.7)
        rf_group3 = squish_rate_func(smooth, 0.5, 1)
        name_C_copy = name_C[-2].copy()
        self.play(
            AnimationGroup(
                *[
                    FadeOut(mob)
                    for mob in (four_circles, vectors, name_C[:-2], name_C[-1])
                ],
                run_time=0.5,
            ),
            AnimationGroup(
                name_C_copy.animate(rate_func=rf_group2).match_height(top_name_C_trans[0]).move_to(top_name_C_trans[0]),
                ReplacementTransform(
                    VectorizedPoint(name_C[-2].get_center()), top_name_C_trans[1],
                    rate_func=rf_group2,
                ),
                ReplacementTransform(
                    matrix_C.get_brackets().copy(), matrix_C_trans.get_brackets(),
                    rate_func=rf_group2,
                ),
                *[
                    ReplacementTransform(
                        matrix_C.get_row(row)[col].copy(), matrix_C_trans.get_column(row)[col],
                        rate_func=rf_group2
                    )
                    for col in range(4)
                    for row in range(4)
                ],
                name_C[-2].animate(rate_func=rf_group2).match_height(top_name_C).move_to(top_name_C),
                ReplacementTransform(matrix_C, matrix_C_copy, rate_func=rf_group2),
                run_time=3,
            ),
            AnimationGroup(
                FadeInFromPoint(top_name_F, top_name_F.get_center(), rate_func=rf_group3),
                FadeInFromPoint(matrix_F, matrix_F.get_center(), rate_func=rf_group3),
                run_time=3,
            ),
        )
        self.wait()
        self.remove(name_C, name_C_copy)
        self.add(top_names)
        self.wait()

        # The result is also a 4x4 matrix
        full_formula = VGroup(backup_lhs.copy(), matrix_equal_sign, matrix_G)
        full_formula.arrange(RIGHT)
        full_formula.match_width(lhs)
        lhs.generate_target()
        lhs.target.become(full_formula[0])
        top_names.generate_target()
        for name, matrix in zip(top_names.target, lhs.target):
            name.next_to(matrix, UP, coor_mask=[1, 0, 0])
        top_equal_sign = Tex("=")
        top_name_G = Tex("\\mathbf{G}")
        for name, ref in zip((top_equal_sign, top_name_G), (matrix_equal_sign, matrix_G)):
            name.set_color(color_special)
            name.next_to(top_name_C, RIGHT)
            name.next_to(ref, UP, coor_mask=[1, 0, 0])
        matrix_G_unknown.match_height(matrix_G).match_width(matrix_G, stretch=True)
        matrix_G_unknown.move_to(matrix_G)
        self.play(
            MoveToTarget(lhs),
            MoveToTarget(top_names),
            FadeInFromPoint(VGroup(matrix_equal_sign, matrix_G_unknown), RIGHT_SIDE),
            FadeInFromPoint(top_equal_sign, RIGHT_SIDE + top_equal_sign.get_center()[1] * UP),
            FadeInFromPoint(top_name_G, RIGHT_SIDE + top_name_G.get_center()[1] * UP),
        )
        self.wait()

        # And we call it "relation matrix G"
        relation_matrix_name = TexText("关系矩阵")
        relation_matrix_name.set_color(color_special)
        relation_matrix_name.next_to(top_name_G, UP)
        self.play(
            Write(relation_matrix_name, stroke_color=color_special),
        )
        self.wait()

        orig_t2c = {
            "\\mathbf{c}_{i}": TEAL,
            "{i}": TEAL,
            "\\mathbf{c}_{j}": PURPLE,
            "{j}": PURPLE,
        }
        relation_matrix_orig = TexText(
            """第${i}$行第${j}$列的元素 \\\\
            $\\left[ \\mathbf{G} \\right]_{ {i} {j} } = \\langle \\mathbf{c}_{i}, \\mathbf{c}_{j} \\rangle$
            """,
            t2c=orig_t2c,
            use_labelled_svg=False,
        )
        relation_matrix_orig.match_width(matrix_G)
        relation_matrix_orig.next_to(matrix_G, DOWN, aligned_edge=RIGHT, buff=0.5)
        self.play(FadeInFromPoint(relation_matrix_orig, matrix_G.get_bottom()))
        self.wait()

        # An example would help understand
        top_names = VGroup(
            top_name_C_trans, top_name_F, top_name_C,
            top_equal_sign, VGroup(relation_matrix_name, top_name_G),
        )
        top_names.save_state()
        top_names.saved_state[-1][0].set_opacity(0)
        self.add(top_names)
        self.play(
            top_names.animate.set_color(GREY).set_opacity(0.2),
        )
        self.wait()

        row, col = 3, 2
        row_ind, col_ind = row - 1, col - 1
        other_row_indices, other_col_indices = list(range(4)), list(range(4))
        other_row_indices.remove(row_ind)
        other_col_indices.remove(col_ind)
        for matrix in (matrix_C_trans, matrix_F, matrix_C_copy):
            matrix.save_state()
        row_text = TexText(
            f"第{{{row}}}个圆的行向量 \\\\ $\\mathbf{{c}}_{{{row}}}^{{\\mathrm{{T}}}}$",
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        F_text = VGroup(top_name_F.copy()).set_opacity(1).set_color(color_special)
        col_text = TexText(
            f"第{{{col}}}个圆的列向量 \\\\ $\\mathbf{{c}}_{{{col}}}$",
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        result_text = TexText(
            f"""
            第{{{row}}}行第{{{col}}}列的元素 \\\\
            $ \\mathbf{{c}}_{{{row}}}^{{\\mathrm{{T}}}} \\mathbf{{F}} \\mathbf{{c}}_{{{col}}} $
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        example_texts = VGroup(row_text, F_text, col_text, result_text)
        example_refs = VGroup(matrix_C_trans, matrix_F, matrix_C_copy, matrix_G)
        for text, ref in zip(example_texts, example_refs):
            text.scale(0.7)
            text.next_to(ref, UP)
            text.next_to(example_refs[-1], UP, coor_mask=[0, 1, 0])
        self.play(
            AnimationGroup(
                *[
                    matrix.get_row(ind).animate.set_color(GREY).set_opacity(0.2)
                    for ind in other_row_indices
                    for matrix in (matrix_C_trans, matrix_G_unknown)
                ],
                FadeInFromPoint(row_text, matrix_C_trans.get_row(row_ind).get_center()),
                run_time=1,
            ),
            AnimationGroup(
                FlashAround(matrix_C_trans.get_row(row_ind), color=CIRCLE_COLORS[row_ind]),
                FlashAround(matrix_G_unknown.get_row(row_ind), color=GREY),
                run_time=2,
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    matrix.get_column(ind).animate.set_color(GREY).set_opacity(0.2)
                    for ind in other_col_indices
                    for matrix in (matrix_C_copy, matrix_G_unknown)
                ],
                FadeInFromPoint(col_text, matrix_C_copy.get_column(col_ind).get_center()),
                run_time=1,
            ),
            AnimationGroup(
                FlashAround(matrix_C_copy.get_column(col_ind), color=CIRCLE_COLORS[col_ind]),
                FlashAround(matrix_G_unknown.get_column(col_ind), color=GREY),
                run_time=2,
            ),
        )
        self.wait()
        self.play(
            ReplacementTransform(top_name_F.copy().set_opacity(0), example_texts[1]),
        )
        self.wait()
        F_text_copy = F_text.copy()
        self.play(
            TransformFromCopy(row_text[:2], result_text[:2], path_arc=-PI / 5),
            TransformFromCopy(row_text[5], result_text[2], path_arc=-PI / 5),
            TransformFromCopy(col_text[:2], result_text[3:5], path_arc=-PI / 5),
            TransformFromCopy(col_text[5], result_text[5], path_arc=-PI / 5),
            FadeInFromPoint(result_text[6:-6], result_text[6:-6].get_center()),
            TransformFromCopy(row_text[-3:], result_text[-6:-3]),
            ApplyMethod(F_text_copy.move_to, result_text[-3]),
            TransformFromCopy(col_text[-2:], result_text[-2:]),
            run_time=2,
        )
        self.remove(F_text_copy)
        self.add(result_text)
        self.wait()
        matrix_product_copy = result_text[-6:].copy()
        self.play(
            FadeTransform(matrix_product_copy, matrix_G.get_row(row_ind)[col_ind]),
            FadeOut(matrix_G_unknown.get_row(row_ind)[col_ind]),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    FadeOut(mob)
                    for mob in [row_text, F_text, col_text, result_text]
                ],
                run_time=1,
            ),
            AnimationGroup(
                Restore(matrix_C_trans),
                Restore(matrix_C_copy),
                Restore(top_names),
                run_time=2,
            ),
        )
        self.wait()
        fade_anims = [
            FadeTransform(matrix_G_unknown.get_row(r)[c], matrix_G.get_row(r)[c])
            for c in range(4)
            for r in range(4)
            if (r, c) != (row_ind, col_ind)
        ]
        random.shuffle(fade_anims)
        self.play(
            AnimationGroup(*fade_anims, lag_ratio=0.05),
            ReplacementTransform(matrix_G_unknown.get_brackets(), matrix_G.get_brackets()),
            run_time=3,
        )
        self.remove(matrix_G_unknown)
        self.add(matrix_G)
        self.wait()
        self.play(FadeOut(relation_matrix_orig))
        self.wait()

        # Notes for all matrices
        notes = VGroup(
            TexText(text)
            for text in [
                "所有圆的 \\\\ 行向量",
                "共用的 \\\\ 度量矩阵",
                "所有圆的 \\\\ 列向量",
                "两两组合的 \\\\ 所有内积",
            ]
        )
        note_C_trans, note_F, note_C, note_G = notes
        for note, ref in zip(notes, example_refs):
            note.set_color(GREY)
            note.scale(0.8)
            note.next_to(ref, DOWN, buff=0.5)
        for note in notes[:-1]:
            note.match_y(notes[-1], UP)
        for mob in (VGroup(note_C_trans, note_C), note_G, note_F):
            self.play(FadeIn(mob, shift=DOWN))
            self.wait()
        self.play(
            AnimationGroup(
                *[
                    Indicate(matrix, scale_factor=1.1, color=None)
                    for matrix in example_refs
                ],
                lag_ratio=0.5,
            ),
            run_time=2,
        )
        self.wait()
        self.play(FadeOut(notes))
        self.wait()

        # Special ops for square matrices: transpose and inverse
        inverse_lhs = VGroup(Tex("("), Tex(")^{-1}"))
        inverse_lhs.set_color(color_special)
        inverse_lhs[0].next_to(top_name_C_trans, LEFT, buff=0.3)
        inverse_lhs[0].match_y(top_name_F)
        inverse_lhs[1].next_to(top_name_C, RIGHT, buff=0.3)
        inverse_lhs[1].match_y(inverse_lhs[0], DOWN)
        temp = Tex("{\\mathbf{G}}^{-1}")
        temp.set_color(color_special)
        temp.align_to(top_name_G, DL)
        inverse_rhs = temp[-2:]
        middle_group = VGroup(matrix_C_trans, matrix_F, matrix_C_copy, matrix_equal_sign, matrix_G)

        self.play(FlashAround(top_name_C_trans, buff=0.2))
        self.wait()

        # Product inverse = Inverted inverse product
        rule_t2c = {
            f"\\mathbf{{{alpha}}}": color
            for alpha, color in zip(list("XYZ"), CIRCLE_COLORS)
        }
        rule_t2c["^{-1}"] = color_special
        inverse_rule = Tex(
            """
            \\left( \\mathbf{X} \\mathbf{Y} \\mathbf{Z} \\right)^{-1} =
            \\mathbf{Z}^{-1} \\mathbf{Y}^{-1} \\mathbf{X}^{-1}
            """,
            t2c=rule_t2c,
            use_labelled_svg=False,
        )
        inverse_rule.scale(1.5).shift(DOWN)
        self.play(
            FadeOut(middle_group),
            FadeIn(inverse_rule[:7]),
        )
        self.wait()
        self.play(
            AnimationGroup(
                TransformFromCopy(
                    inverse_rule[source_ind], inverse_rule[target_ind],
                    path_arc=PI / 2,
                )
                for source_ind, target_ind in [(1, 14), (2, 11), (3, 8)]
            ),
            AnimationGroup(
                TransformFromCopy(
                    inverse_rule[5:7], inverse_rule[target_ind:target_ind + 2],
                    path_arc=-PI / 3,
                )
                for target_ind in [9, 12, 15]
            ),
            FadeInFromPoint(inverse_rule[7], inverse_rule[7].get_center()),
            run_time=2,
        )
        self.add(inverse_rule)
        self.wait()

        # If G is invertible, then C is invertible as well
        inv_note_G = TexText("可逆").to_edge(UP)
        inv_note_C_trans, inv_note_C = inv_note_G.copy(), inv_note_G.copy()
        inv_notes = VGroup(inv_note_C_trans, inv_note_C, inv_note_G)
        inv_notes.next_to(top_names[0], UP)
        ref_names = VGroup(top_name_C_trans, top_name_C, top_name_G)
        for note, name in zip(inv_notes, ref_names):
            note.set_color(color_special)
            note.match_x(name)
        self.play(GrowFromPoint(inv_note_G, top_name_G.get_center()))
        self.wait()
        self.play(
            *[
                TransformFromCopy(inv_note_G, note)
                for note in (inv_note_C_trans, inv_note_C)
            ],
        )
        self.wait()
        self.play(
            *[
                FadeIn(mob, shift=DOWN, scale=0.5, run_time=2)
                for mob in (inverse_lhs[0], inverse_lhs[1], inverse_rhs)
            ],
            *[
                FadeOutToPoint(mob, mob.get_center(), run_time=1)
                for mob in inv_notes
            ],
            FadeOut(inverse_rule, run_time=1),
        )
        self.wait()

        # Take inverse of both sides and expand it out
        expanded_formula = Tex(
            """
            \\mathbf{C} \\mathbf{C}^{-1} \\mathbf{F}^{-1} \\left( {\\mathbf{C}^{\\mathrm{T}}} \\right)^{-1}
            \\mathbf{C}^{\\mathrm{T}} = \\mathbf{C} \\mathbf{G}^{-1} \\mathbf{C}^{\\mathrm{T}}
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        expanded_formula.scale(1.5).shift(DOWN)
        expanded_formula.set_color(color_special)
        top_name_C_ghost = top_name_C.copy()
        top_name_F_ghost = top_name_F.copy()
        top_name_G_ghost = top_name_G.copy()
        top_equal_sign_ghost = top_equal_sign.copy()
        top_left_paren_ghost = inverse_lhs[0].copy()
        ghosts = VGroup(
            top_name_C_ghost, top_name_F_ghost, top_name_G_ghost,
            top_equal_sign_ghost, top_left_paren_ghost,
        )

        self.play(
            AnimationGroup(
                top_name_C_ghost.animate.replace(expanded_formula[1]),
                top_name_F_ghost.animate.replace(expanded_formula[4]),
                TransformFromCopy(top_name_C_trans, expanded_formula[8:10]),
                AnimationGroup(
                    top_equal_sign_ghost.animate.replace(expanded_formula[15]),
                    top_name_G_ghost.animate.replace(expanded_formula[17]),
                ),
                AnimationGroup(
                    TransformFromCopy(inverse_lhs[1][-2:], expanded_formula[2:4]),
                    TransformFromCopy(inverse_lhs[1][-2:], expanded_formula[5:7]),
                    top_left_paren_ghost.animate.replace(expanded_formula[7], stretch=True),
                    ReplacementTransform(inverse_lhs[1].copy(), expanded_formula[10:13]),
                    ReplacementTransform(inverse_rhs.copy(), expanded_formula[18:20]),
                ),
                lag_ratio=0.05,
            ),
            run_time=3,
        )
        self.remove(ghosts)
        self.add(expanded_formula[1:13], expanded_formula[15], expanded_formula[17:20])
        self.wait()

        # Multiply some matrices for reduction
        color_reduction = TEAL
        color_nil = GREY
        reduction_group = VGroup(
            expanded_formula[0], expanded_formula[16],
            expanded_formula[13:15], expanded_formula[20:]
        )
        reduction_group.set_color(color_reduction)

        self.play(
            AnimationGroup(
                AnimationGroup(
                    *[FadeIn(mob, shift=DOWN) for mob in reduction_group[:2]],
                ),
                AnimationGroup(
                    *[FadeIn(mob, shift=DOWN)for mob in reduction_group[2:]],
                ),
                lag_ratio=0.4,
            ),
            run_time=2,
        )
        self.wait()
        left_part = expanded_formula[:4]
        left_rect = SurroundingRectangle(left_part).set_color(color_nil)
        left_group = VGroup(left_part, left_rect)
        right_part = expanded_formula[7:15]
        right_rect = SurroundingRectangle(right_part).set_color(color_nil)
        right_group = VGroup(right_part, right_rect)
        self.play(
            left_part.animate.set_color(color_nil),
            right_part.animate.set_color(color_nil),
            ShowCreation(left_rect),
            ShowCreation(right_rect),
            expanded_formula[15:].animate.set_color(color_special),
        )
        self.wait()
        self.play(
            FadeOutToPoint(left_group, left_group.get_center()),
            FadeOutToPoint(right_group, right_group.get_center()),
        )
        self.wait()

        # And we get a 'brand new' formula
        new_formula = Tex("\\mathbf{C} \\mathbf{G}^{-1} \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}")
        new_formula.set_color(color_special)
        new_formula.scale(1.5).shift(DOWN)
        new_text = TexText("“全新”公式 ?")
        new_text.next_to(new_formula, UP)
        self.play(
            AnimationGroup(
                ReplacementTransform(expanded_formula[16:], new_formula[:6], path_arc=PI / 2),
                ReplacementTransform(expanded_formula[15], new_formula[6]),
                ReplacementTransform(expanded_formula[4:7], new_formula[7:], path_arc=PI / 2),
                run_time=1,
            ),
            Write(new_text[:-1], run_time=2),
            AnimationGroup(
                FadeOut(inverse_lhs),
                FadeOut(inverse_rhs),
            ),
        )
        self.remove(expanded_formula)
        self.add(new_formula)
        self.wait()

        # But why is it brand new?
        self.play(FadeInFromPoint(new_text[-1], new_text[-1].get_center()))
        self.wait()
        circle_relation = Tex("\\mathbf{C}^{\\mathrm{T}} \\mathbf{F} \\mathbf{C} = \\mathbf{G}")
        property_relation = Tex("\\mathbf{C} \\mathbf{G}^{-1} \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}")
        for relation, side in zip([circle_relation, property_relation], [LEFT_SIDE, RIGHT_SIDE]):
            relation.set_color(color_special)
            relation.move_to(side / 2)
            relation.to_edge(UP, buff=0.5)
        new_text.generate_target()
        new_text.target.next_to(property_relation, UP)
        new_text.target.set_opacity(0)
        matrix_C = TexMatrix(
            string_table_C,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        self.play(
            AnimationGroup(
                top_name_C_trans.animate.replace(circle_relation[:2]),
                top_name_F.animate.replace(circle_relation[2]),
                top_name_C.animate.replace(circle_relation[3]),
                top_equal_sign.animate.replace(circle_relation[4]),
                top_name_G.animate.replace(circle_relation[5]),
                MoveToTarget(new_text),
                ReplacementTransform(new_formula, property_relation),
                run_time=2,
            ),
            FadeInFromPoint(
                matrix_C, matrix_C.get_center(),
                rate_func=squish_rate_func(smooth, 0.4, 1),
                run_time=3,
            ),
        )
        self.remove(top_names)
        self.add(circle_relation)
        self.wait()


class InterpretBothFormulae(Scene):
    def construct(self):
        # Setup to match the previous scene
        color_c1, color_c2, color_c3, color_c4 = CIRCLE_COLORS
        color_special = YELLOW
        color_note = GREY_B
        basic_t2c = {
            key: color
            for i, color in enumerate(CIRCLE_COLORS, start=1)
            for key in [
                f"\\mathbf{{c}}_{{{i}}}",
                f"k_{{{i}}}",
                f"t_{{{i}}}",
                f"x_{{{i}}}",
                f"y_{{{i}}}",
                f"{{{i}}}",
            ]
        }
        basic_t2c["\\mathbf{F}"] = color_special
        circle_relation = Tex("\\mathbf{C}^{\\mathrm{T}} \\mathbf{F} \\mathbf{C} = \\mathbf{G}")
        property_relation = Tex("\\mathbf{C} \\mathbf{G}^{-1} \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}")
        for relation, side in zip([circle_relation, property_relation], [LEFT_SIDE, RIGHT_SIDE]):
            relation.set_color(color_special)
            relation.move_to(side / 2)
            relation.to_edge(UP, buff=0.5)
            relation.save_state()
        string_table_C = np.array([
            ["k_{1}", "k_{2}", "k_{3}", "k_{4}"],
            ["t_{1}", "t_{2}", "t_{3}", "t_{4}"],
            ["k_{1} x_{1}", "k_{2} x_{2}", "k_{3} x_{3}", "k_{4} x_{4}"],
            ["k_{1} y_{1}", "k_{2} y_{2}", "k_{3} y_{3}", "k_{4} y_{4}"],
        ])
        matrix_C = TexMatrix(
            string_table_C,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        self.add(circle_relation, property_relation, matrix_C)

        # Interpret the original formula
        C_trans_orig = TexMatrix(
            string_table_C.T,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        F_orig, G_orig = VGroup(
            FakeMatrix(
                text,
                side_length=3,
                scale_factor=2,
            )
            for text in ("\\mathbf{F}", "\\mathbf{G}")
        )
        C_orig = TexMatrix(
            string_table_C,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        equal_orig = Tex("=")
        orig_group = VGroup(C_trans_orig, F_orig, C_orig, equal_orig, G_orig)
        for matrix in (F_orig, G_orig):
            matrix.set_color(color_special)
        orig_group.arrange(RIGHT)
        orig_group.set_width(FRAME_WIDTH - 1)
        self.play(
            AnimationGroup(
                property_relation.animate.set_color(GREY).set_opacity(0.2),
                run_time=1,
            ),
            AnimationGroup(
                ReplacementTransform(matrix_C, C_orig),
                *[
                    ReplacementTransform(
                        matrix_C.get_row(row)[col].copy(), C_trans_orig.get_column(row)[col],
                    )
                    for col in range(4)
                    for row in range(4)
                ],
                ReplacementTransform(matrix_C.get_brackets().copy(), C_trans_orig.get_brackets()),
                run_time=2,
            ),
            AnimationGroup(
                *[
                    FadeInFromPoint(
                        mob, mob.get_center(),
                        rate_func=squish_rate_func(smooth, 0.5, 1),
                    )
                    for mob in (F_orig, G_orig, equal_orig)
                ],
                run_time=2,
            ),
        )
        self.wait()

        # Rows of C^T and columns of C are all properties of one circle
        underline_config = {
            "stroke_color": color_note,
            "stretch_factor": 1,
        }
        C_trans_orig_note = TexText("行： \\\\ 某个圆的所有属性")
        C_orig_note = TexText("列： \\\\ 某个圆的所有属性")
        G_orig_note = TexText("任意两个圆的 \\\\ 位置关系")
        orig_notes = VGroup(C_trans_orig_note, C_orig_note, G_orig_note)
        orig_refs = VGroup(C_trans_orig, C_orig, G_orig)
        orig_notes.next_to(orig_refs, DOWN)
        for note, ref in zip(orig_notes, orig_refs):
            note.set_color(color_note)
            note.scale(0.8)
            note.match_x(ref)
        C_trans_orig_underline = Underline(C_trans_orig_note[2:5], **underline_config)
        C_orig_underline = Underline(C_orig_note[2:5], **underline_config)
        self.play(
            AnimationGroup(
                FadeInFromPoint(
                    VGroup(C_trans_orig_note, C_trans_orig_underline),
                    C_trans_orig.get_bottom(),
                ),
                FadeInFromPoint(
                    VGroup(C_orig_note, C_orig_underline),
                    C_orig.get_bottom(),
                ),
                run_time=1,
            ),
            AnimationGroup(
                *[
                    FlashAround(C_trans_orig.get_row(row_ind), color=color)
                    for row_ind, color in enumerate(CIRCLE_COLORS)
                ],
                lag_ratio=0.1,
                run_time=3,
            ),
            AnimationGroup(
                *[
                    FlashAround(C_orig.get_column(col_ind), color=color)
                    for col_ind, color in enumerate(CIRCLE_COLORS)
                ],
                lag_ratio=0.1,
                run_time=3,
            ),
        ),
        self.wait()
        # We can get relation of any two circles
        self.play(Indicate(F_orig))
        self.wait()
        self.play(
            *[
                FadeTransform(mob.copy(), G_orig_note)
                for mob in (C_trans_orig_note, F_orig, C_orig_note)
            ]
        )
        self.wait()

        # Interpret the inverted formula
        C_new = TexMatrix(
            string_table_C,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        Ginv_new, Finv_new = VGroup(
            FakeMatrix(
                text,
                side_length=3,
                scale_factor=2,
            )
            for text in ("\\mathbf{G}^{-1}", "\\mathbf{F}^{-1}")
        )
        C_trans_new = TexMatrix(
            string_table_C.T,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        equal_new = Tex("=")
        new_group = VGroup(C_new, Ginv_new, C_trans_new, equal_new, Finv_new)
        for matrix in (Ginv_new, Finv_new):
            matrix.set_color(color_special)
        new_group.arrange(RIGHT)
        new_group.set_width(FRAME_WIDTH - 1)

        # Rows of C and columns of C^T are one property of all circles
        C_new_note = TexText("行： \\\\ 所有圆的某个属性")
        C_trans_new_note = TexText("列： \\\\ 所有圆的某个属性")
        Finv_new_note = TexText("任意两个属性的 \\\\ 约束关系")
        new_notes = VGroup(C_new_note, C_trans_new_note, Finv_new_note)
        new_refs = VGroup(C_new, C_trans_new, Finv_new)
        new_notes.next_to(new_refs, DOWN)
        for note, ref in zip(new_notes, new_refs):
            note.set_color(color_note)
            note.scale(0.8)
            note.match_x(ref)
        C_trans_new_underline = Underline(C_trans_new_note[-4:], **underline_config)
        C_new_underline = Underline(C_new_note[-4:], **underline_config)
        C_trans_orig_underline.generate_target()
        C_trans_orig_underline.target.match_x(C_new_note[2:5])
        C_orig_underline.generate_target()
        C_orig_underline.target.match_x(C_trans_new_note[2:5])
        self.play(
            AnimationGroup(
                circle_relation.animate.set_color(GREY).set_opacity(0.2),
                Restore(property_relation),
                FadeOut(G_orig_note),
                run_time=1,
            ),
            AnimationGroup(
                ReplacementTransform(C_orig, C_new, path_arc=PI * 2 / 3),
                ReplacementTransform(C_trans_orig, C_trans_new, path_arc=PI * 2 / 3),
                *[
                    AnimationGroup(
                        ReplacementTransform(source[:2], target[:2]),
                        ReplacementTransform(source[2:4], target[-4:-2], path_arc=PI / 2),
                        ReplacementTransform(source[4:-4], target[4:-4]),
                        ReplacementTransform(source[-4:-2], target[2:4], path_arc=PI / 2),
                        ReplacementTransform(source[-2:], target[-2:]),
                    )
                    for source, target in [
                        (C_orig_note, C_trans_new_note),
                        (C_trans_orig_note, C_new_note),
                    ]
                ],
                *[
                    MoveToTarget(underline)
                    for underline in (C_trans_orig_underline, C_orig_underline)
                ],
                run_time=3,
            ),
            AnimationGroup(
                FadeTransform(F_orig, Ginv_new),
                FadeTransform(G_orig, Finv_new),
                ReplacementTransform(equal_orig, equal_new),
                run_time=2,
            ),
        )
        self.wait()

        # Now changing the perspective...
        self.play(
            AnimationGroup(
                *[
                    FlashAround(C_new.get_row(row_ind), color=CIRCLE_COLORS * 2)
                    for row_ind in range(4)
                ],
                lag_ratio=0.1,
                run_time=3,
            ),
            AnimationGroup(
                *[
                    FlashAround(C_trans_new.get_column(col_ind), color=CIRCLE_COLORS * 2)
                    for col_ind in range(4)
                ],
                lag_ratio=0.1,
                run_time=3,
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    ReplacementTransform(source, target, path_arc=PI / 3)
                    for source, target in [
                        (C_trans_orig_underline, C_new_underline),
                        (C_orig_underline, C_trans_new_underline),
                    ]
                ],
                run_time=2,
            )
        )
        self.wait()
        # We can get relation of any two properties
        self.play(Indicate(Ginv_new))
        self.wait()
        self.play(
            *[
                FadeTransform(mob.copy(), Finv_new_note)
                for mob in (C_new_note, Ginv_new, C_trans_new_note)
            ]
        )
        self.wait()

        # Finally, the meaning of both formulae
        meaning_orig = TexText("任意两个圆的位置关系")
        meaning_new = TexText("任意两个属性的约束关系")
        meaning_orig.next_to(circle_relation, DOWN)
        meaning_new.next_to(property_relation, DOWN)
        self.play(
            Write(meaning_orig),
            Restore(circle_relation),
        )
        self.wait()
        self.play(
            Write(meaning_new),
        )
        self.wait()

        circle_group = VGroup(circle_relation, meaning_orig)
        property_group = VGroup(property_relation, meaning_new)
        for group in (circle_group, property_group):
            group.generate_target()
            group.target[0].scale(1.25)
            group.target.arrange(DOWN)
            group.target.match_x(group)
        arrow = VGroup(
            TexText("（取逆）"),
            Tex("{} \\iff {}"),
        )
        arrow[0].set_width(arrow[1].get_width() * 0.8)
        arrow.arrange(DOWN, buff=0.1)
        arrow.set_color(color_special)
        arrow.generate_target()
        source_point = (circle_group.get_center() + property_group.get_center()) / 2
        arrow.scale(0.01).set_opacity(0).move_to(source_point)
        target_point = (circle_group.target[0].get_center() + property_group.target[0].get_center()) / 2
        arrow.target.shift(target_point - arrow.target[1].get_center())
        fade_group = VGroup(
            new_group, C_new_note, C_trans_new_note, Finv_new_note,
            C_new_underline, C_trans_new_underline,
        )
        self.play(
            AnimationGroup(
                *[
                    MoveToTarget(group)
                    for group in (circle_group, property_group)
                ],
                MoveToTarget(arrow),
                run_time=2,
            ),
            FadeOut(fade_group, run_time=1),
        )
        self.wait()


class ProvingDescartesTheorem(Scene):
    def construct(self):
        # Setup
        color_c1, color_c2, color_c3, color_c4 = CIRCLE_COLORS
        color_special = YELLOW
        color_note = GREY_B
        color_F_nonzero, color_F_zero = TEAL, WHITE
        color_G_diag, color_G_off_diag = PURPLE_B, MAROON_B
        basic_t2c = {
            key: color
            for i, color in enumerate(CIRCLE_COLORS, start=1)
            for key in [
                f"\\mathbf{{c}}_{{{i}}}",
                f"k_{{{i}}}",
                f"t_{{{i}}}",
                f"x_{{{i}}}",
                f"y_{{{i}}}",
                f"{{{i}}}",
            ]
        }
        basic_t2c["\\mathbf{F}"] = color_special
        string_table_C = np.array([
            ["k_{1}", "k_{2}", "k_{3}", "k_{4}"],
            ["t_{1}", "t_{2}", "t_{3}", "t_{4}"],
            ["k_{1} x_{1}", "k_{2} x_{2}", "k_{3} x_{3}", "k_{4} x_{4}"],
            ["k_{1} y_{1}", "k_{2} y_{2}", "k_{3} y_{3}", "k_{4} y_{4}"],
        ])
        string_table_ktrans = string_table_C[0:1].copy()
        string_table_F = [
            [0, "-\\tfrac{1}{2}", 0, 0],
            ["-\\tfrac{1}{2}", 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        string_table_Finv = np.array([
            [0, -2, 0, 0],
            [-2, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=int)
        string_table_G = [
            [
                1 if (row == col) else -1
                for col in range(1, 5)
            ]
            for row in range(1, 5)
        ]
        string_table_Ginv = [
            [
                "\\tfrac{1}{4}" if (row == col) else "-\\tfrac{1}{4}"
                for col in range(1, 5)
            ]
            for row in range(1, 5)
        ]

        tangent_circles = TangentFourCircles(
            ORIGIN,
            1.2 * UP,
            RIGHT + 0.4 * UP,
        )
        tangent_circles.suspend_updating()
        tangent_circles.rotate(PI / 10).center().scale(2)
        tangent_circles.to_edge(LEFT)
        tangent_circles.update_centers_from_circles()

        matrix_G_note = TexText("关系矩阵 $\\mathbf{G}$")
        matrix_G_note.set_color(color_special)
        matrix_G_unknown = TexMatrix(
            [
                [
                    f"\\langle \\mathbf{{c}}_{{{row}}}, \\mathbf{{c}}_{{{col}}} \\rangle"
                    for col in range(1, 5)
                ]
                for row in range(1, 5)
            ],
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        matrix_G_filled = TexMatrix(
            string_table_G,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
            v_buff=0.6,
        )
        for entry in matrix_G_filled.get_entries():
            if entry.tex_string == "1":
                entry.set_color(color_G_diag)
            else:
                entry.set_color(color_G_off_diag)
        matrix_G_unknown.set_width(FRAME_WIDTH * 0.4)
        matrix_G_group = VGroup(matrix_G_note, matrix_G_unknown)
        matrix_G_group.arrange(DOWN)
        matrix_G_group.to_edge(RIGHT)
        matrix_G_filled.match_height(matrix_G_unknown)
        matrix_G_filled.move_to(matrix_G_unknown)
        arrow = Tex("\\overset{?}{\\Rightarrow}")
        arrow.scale(1.5)
        arrow.move_to((matrix_G_group.get_left() + tangent_circles.get_right()) / 2)

        self.add(tangent_circles, arrow, matrix_G_group)
        self.wait()

        # Every diagonal element is 1 (coincident)
        for matrix in (matrix_G_filled, matrix_G_unknown):
            matrix.save_state()
        diag_entries_unknown, diag_entries_filled = [
            VGroup(
                matrix.get_row(ind)[ind]
                for ind in range(4)
            )
            for matrix in (matrix_G_unknown, matrix_G_filled)
        ]
        off_diag_entries_unknown, off_diag_entries_filled = [
            VGroup(
                matrix.get_row(row_ind)[col_ind]
                for row_ind in range(4)
                for col_ind in range(4)
                if row_ind != col_ind
            )
            for matrix in (matrix_G_unknown, matrix_G_filled)
        ]
        for source, target in zip(matrix_G_filled.get_brackets(), matrix_G_unknown.get_brackets()):
            source.move_to(target)
        for source, target in zip(matrix_G_filled.get_entries(), matrix_G_unknown.get_entries()):
            source.move_to(target)
        diag_note = TexText("圆与自身的内积$=1$")
        off_diag_note = TexText("外切圆之间的内积$=-1$")
        for note in (diag_note, off_diag_note):
            note.scale(0.8)
            note.next_to(matrix_G_unknown, DOWN)
            note.set_color(color_note)
        self.play(
            off_diag_entries_unknown.animate.set_opacity(0.1),
            Write(diag_note),
        )
        self.wait()
        anims = [
            FadeTransform(source, target)
            for source, target in zip(diag_entries_unknown, diag_entries_filled)
        ]
        random.shuffle(anims)
        self.play(
            AnimationGroup(
                *anims,
                lag_ratio=0.05,
                run_time=1,
            ),
        )
        self.wait()

        # Every off-diagonal element is -1 (externally tangent)
        self.play(
            off_diag_entries_unknown.animate.set_opacity(1),
            diag_entries_filled.animate.set_opacity(0.1),
            FadeTransform(diag_note, off_diag_note),
        )
        self.wait()
        anims = [
            FadeTransform(source, target)
            for source, target in zip(off_diag_entries_unknown, off_diag_entries_filled)
        ]
        random.shuffle(anims)
        self.play(
            AnimationGroup(
                *anims,
                lag_ratio=0.05,
                run_time=2,
            ),
        )
        self.wait()

        # We have filled the relation matrix G
        self.play(
            FadeOut(off_diag_note),
            diag_entries_filled.animate.set_opacity(1),
            FadeOut(arrow[0]),
        )
        arrow = arrow[1:]
        self.remove(matrix_G_unknown)
        self.add(matrix_G_filled)
        self.wait()
        self.play(Restore(matrix_G_filled))
        self.wait()

        # Metric matrix F is known
        matrix_F = TexMatrix(
            string_table_F,
            element_alignment_corner=ORIGIN,
            h_buff=0.5, v_buff=0.3,
        )
        for entry in matrix_F.get_entries():
            if entry.tex_string == "0":
                entry.set_color(color_F_zero)
            else:
                entry.set_color(color_F_nonzero)
        matrix_F.match_height(matrix_G_filled)
        matrix_F_note = TexText("度量矩阵 $\\mathbf{F}$")
        matrix_F_note.set_color(color_special)
        matrix_F_group = VGroup(matrix_F_note, matrix_F)
        matrix_F_group.arrange(DOWN)
        matrix_F_group.move_to(LEFT_SIDE / 2)
        matrix_G_group = VGroup(matrix_G_note, matrix_G_filled)
        matrix_G_group.generate_target()
        matrix_G_group.target.move_to(RIGHT_SIDE / 2)
        self.play(
            FadeIn(matrix_F_group),
            MoveToTarget(matrix_G_group),
            FadeOut(arrow),
            FadeOut(tangent_circles),
        )
        self.wait()

        # Calculate their inverses to make use of the brand-new formula
        relation = Tex(
            """
            \\mathbf{C} \\mathbf{G}^{-1} \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}
            \\quad \\text{（任意两个属性的约束关系）}
            """
        )
        relation_formula, relation_note = relation[:10], relation[10:]
        relation_formula.set_color(color_special)
        relation_note.set_color(color_note)
        relation.to_edge(UP)

        text_F = Tex("\\mathbf{F} = ")
        text_Finv = Tex("\\mathbf{F}^{-1} = ")
        text_G = Tex("\\mathbf{G} = ")
        text_Ginv = Tex("\\mathbf{G}^{-1} = ")
        for text in (text_F, text_Finv, text_G, text_Ginv):
            text[:-1].set_color(color_special)
        matrix_Finv = TexMatrix(
            string_table_Finv,
            element_alignment_corner=ORIGIN,
            h_buff=0.5, v_buff=0.55,
        )
        matrix_G = matrix_G_filled
        matrix_Ginv = TexMatrix(
            string_table_Ginv,
            element_alignment_corner=ORIGIN,
            h_buff=0.5, v_buff=0.3,
        )
        for ref, matrix in [(matrix_F, matrix_Finv), (matrix_G, matrix_Ginv)]:
            matrix.match_width(ref)
            matrix.match_height(ref, stretch=True)
            matrix.next_to(ref, DOWN)
            for ref_entry, matrix_entry in zip(ref.get_entries(), matrix.get_entries()):
                matrix_entry.match_color(ref_entry)

        matrix_F.generate_target()
        matrix_G.generate_target()
        result_F_group = VGroup(text_F, matrix_F.target)
        result_G_group = VGroup(text_G, matrix_G.target)
        result_Finv_group = VGroup(text_Finv, matrix_Finv)
        result_Ginv_group = VGroup(text_Ginv, matrix_Ginv)
        result_groups = VGroup(result_F_group, result_Finv_group, result_G_group, result_Ginv_group)
        directions = [UL, DL, UR, DR]
        frame = FullScreenRectangle()
        for group, direction in zip(result_groups, directions):
            group.arrange(RIGHT)
            group.move_to(frame.get_corner(direction) / 2)
        for inv_group, orig_group in zip(result_groups[1::2], result_groups[::2]):
            inv_group.next_to(orig_group, DOWN, buff=0.5)
            inv_group.match_x(orig_group, RIGHT)
        self.play(
            AnimationGroup(
                FadeOut(matrix_F_note[:-1], shift=LEFT),
                FadeOut(matrix_G_note[:-1], shift=LEFT),
                run_time=1,
            ),
            AnimationGroup(
                FadeInFromPoint(text_F[1], matrix_F.get_left()),
                ReplacementTransform(matrix_F_note[-1], text_F[0]),
                MoveToTarget(matrix_F),
                FadeInFromPoint(text_G[1], matrix_G.get_left()),
                ReplacementTransform(matrix_G_note[-1], text_G[0]),
                MoveToTarget(matrix_G),
                run_time=2,
            ),
        )
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeTransform(text_F.copy(), text_Finv),
                    FadeTransform(matrix_F.copy(), matrix_Finv),
                ),
                AnimationGroup(
                    FadeTransform(text_G.copy(), text_Ginv),
                    FadeTransform(matrix_G.copy(), matrix_Ginv),
                ),
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.wait()

        # Back to the formula, write out its full form
        result_F_group = VGroup(text_F, matrix_F)
        result_G_group = VGroup(text_G, matrix_G)
        self.play(
            FadeOut(result_F_group),
            FadeOut(result_G_group),
            FadeIn(relation, shift=DOWN),
        )
        self.wait()

        matC_1 = TexMatrix(
            string_table_C,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        matC_2 = matC_1.copy()
        matGinv_1 = TexMatrix(string_table_Ginv)
        matCtrans_1 = TexMatrix(
            string_table_C.T,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        matCtrans_2 = matCtrans_1.copy()
        equal_1 = Tex("=")
        equal_2 = equal_1.copy()
        matFinv_1 = TexMatrix(string_table_Finv)
        group_1 = VGroup(matC_1, matGinv_1, matCtrans_1, equal_1, matFinv_1)
        group_1.arrange(RIGHT)
        group_1.set_width(FRAME_WIDTH - 1)
        top_names = VGroup(
            Tex(text).set_color(color_special)
            for text in ["\\mathbf{C}", "\\mathbf{G}^{-1}", "\\mathbf{C}^{\\mathrm{T}}", "=", "\\mathbf{F}^{-1}"]
        )
        top_names.next_to(group_1, UP, buff=0.5)
        for name, ref in zip(top_names, group_1):
            name.match_x(ref)
        self.play(
            AnimationGroup(
                FadeOut(text_Finv[-1]),
                FadeOut(text_Ginv[-1]),
                run_time=1,
            ),
            AnimationGroup(
                ReplacementTransform(text_Finv[:-1], top_names[4]),
                ReplacementTransform(text_Ginv[:-1], top_names[1]),
                ReplacementTransform(matrix_Finv, matFinv_1, path_arc=PI / 2),
                ReplacementTransform(matrix_Ginv, matGinv_1, path_arc=PI / 2),
                run_time=2,
            )
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        *[FadeInFromPoint(mob, mob.get_center()) for mob in mob_group],
                        lag_ratio=0.1,
                    )
                    for mob_group in (
                        (matC_1, matCtrans_1, equal_1),
                        (top_names[0], top_names[2], top_names[3]),
                    )
                ],
            ),
            run_time=2,
        )
        self.wait()

        # Multiply 4 to get rid of fractions
        matGinvt4_2 = TexMatrix(string_table_G)
        matFinvt4_2 = TexMatrix(string_table_Finv * 4)
        group_2 = VGroup(matC_2, matGinvt4_2, matCtrans_2, equal_2, matFinvt4_2)
        group_2.arrange(RIGHT)
        group_2.match_width(group_1)
        group_2.move_to(group_1)

        t4_Ginv = Tex("4 \\times")
        t4_Ginv.scale(1.5).set_color(color_special)
        t4_Ginv.next_to(matGinv_1, DOWN, buff=0.5)
        t4_Ginv.set_opacity(0)
        t4_Finv = t4_Ginv.copy()
        t4_Finv.match_x(matFinv_1)

        def opacity_func(alpha):
            if alpha < 0.25:
                t = alpha / 0.25
                return 3 * t**2 - 2 * t**3
            elif alpha <= 0.75:
                return 1.0
            else:
                t = (alpha - 0.75) / 0.25
                val = 1.0 - (3 * t**2 - 2 * t**3)
                return max(0.0, val)

        start_t4_Ginv = np.array(t4_Ginv.get_center())
        shift_t4_Ginv = (matGinvt4_2.get_center() - start_t4_Ginv)

        def update_t4_Ginv(mob, alpha):
            mob.move_to(start_t4_Ginv + smooth(alpha) * shift_t4_Ginv)
            mob.set_opacity(0.4 * opacity_func(alpha))

        start_t4_Finv = np.array(t4_Finv.get_center())
        shift_t4_Finv = (matFinvt4_2.get_center() - start_t4_Finv)

        def update_t4_Finv(mob, alpha):
            mob.move_to(start_t4_Finv + smooth(alpha) * shift_t4_Finv)
            mob.set_opacity(0.5 * opacity_func(alpha))

        self.play(
            UpdateFromAlphaFunc(t4_Ginv, update_t4_Ginv, rate_func=linear),
            UpdateFromAlphaFunc(t4_Finv, update_t4_Finv, rate_func=linear),
            FadeTransform(matGinv_1, matGinvt4_2),
            FadeTransform(matFinv_1, matFinvt4_2),
            *[
                ReplacementTransform(source, target)
                for source, target in zip(
                    [matC_1, matCtrans_1, equal_1],
                    [matC_2, matCtrans_2, equal_2],
                )
            ],
            FadeOut(top_names),
            run_time=2,
        )
        self.remove(t4_Ginv, t4_Finv)
        self.remove(matC_1, matCtrans_1, equal_1)
        self.add(matC_2, matCtrans_2, equal_2)
        self.wait()

        # Focus on important parts - curvature
        matC_imp = VGroup(*[
            matC_2.get_entries()[i]
            for i in range(4)
        ])
        matC_nil = VGroup(*[
            matC_2.get_entries()[i]
            for i in range(4, 16)
        ])
        matC_rect = SurroundingRectangle(matC_imp)
        matCtrans_imp = VGroup(*[
            matCtrans_2.get_entries()[i]
            for i in range(16)
            if i % 4 == 0
        ])
        matCtrans_nil = VGroup(*[
            matCtrans_2.get_entries()[i]
            for i in range(16)
            if i % 4 != 0
        ])
        matCtrans_rect = SurroundingRectangle(matCtrans_imp)
        matFinvt4_imp = matFinvt4_2.get_entries()[0]
        matFinvt4_nil = matFinvt4_2.get_entries()[1:]
        matFinvt4_rowrect = SurroundingRectangle(matFinvt4_2.get_row(0))
        matFinvt4_colrect = SurroundingRectangle(matFinvt4_2.get_column(0))
        temp = SurroundingRectangle(matFinvt4_2.get_entries())
        matFinvt4_rowrect.match_width(temp, stretch=True)
        matFinvt4_rowrect.match_x(temp, LEFT).match_y(temp, UP)
        matFinvt4_colrect.match_height(temp, stretch=True)
        matFinvt4_colrect.match_x(temp, LEFT).match_y(temp, UP)
        descartes_title = TexText("笛卡尔定理（曲率-曲率的约束关系）")
        descartes_title.set_color(color_special)
        descartes_title[5:].set_color(color_note)
        descartes_title.move_to(relation)
        rects = VGroup(matC_rect, matCtrans_rect, matFinvt4_rowrect, matFinvt4_colrect)
        self.play(
            *[
                mob.animate.set_color(GREY).set_opacity(0.2)
                for mob in (matC_nil, matCtrans_nil, matFinvt4_nil)
            ],
            *[
                ShowCreation(rect)
                for rect in rects
            ],
            *[
                mob.animate.set_color(color_special)
                for mob in (matGinvt4_2, matFinvt4_imp)
            ],
            run_time=3,
        )
        self.wait()
        self.play(
            FadeOut(rects),
        )
        self.wait()

        # Clean out redundant terms
        self.play(
            FadeOut(relation[:10]),
            ReplacementTransform(relation[10], descartes_title[5]),
            FadeTransform(relation[11:17], descartes_title[6:11]),
            ReplacementTransform(relation[17:], descartes_title[11:]),
        )
        self.wait()
        vec_k_trans = TexMatrix(
            string_table_ktrans,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        mat_Q = TexMatrix(
            string_table_G,
            v_buff=0.6,
        )
        vec_k = TexMatrix(
            string_table_ktrans.T,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        equal_zero = Tex("= 0")
        descartes_group = VGroup(vec_k_trans, mat_Q, vec_k, equal_zero)
        descartes_group.arrange(RIGHT)
        descartes_group.set_width(FRAME_WIDTH * 0.6)
        self.remove(matC_2, matCtrans_2)
        self.add(
            matC_imp, matC_nil, matC_2.get_brackets(),
            matCtrans_imp, matCtrans_nil, matCtrans_2.get_brackets(),
        )
        self.play(
            AnimationGroup(
                FadeTransform(matC_2.get_brackets()[0], vec_k_trans.get_brackets()[0]),
                FadeTransform(matC_2.get_brackets()[1], vec_k_trans.get_brackets()[1]),
                ReplacementTransform(matC_imp, vec_k_trans.get_entries()),
                ReplacementTransform(matGinvt4_2, mat_Q),
                ReplacementTransform(matCtrans_2.get_brackets()[0], vec_k.get_brackets()[0]),
                ReplacementTransform(matCtrans_2.get_brackets()[1], vec_k.get_brackets()[1]),
                ReplacementTransform(matCtrans_imp, vec_k.get_entries()),
                ReplacementTransform(equal_2[0], equal_zero[0]),
                ReplacementTransform(matFinvt4_imp[0], equal_zero[1]),
                run_time=2,
            ),
            AnimationGroup(
                *[FadeOut(nil) for nil in (matC_nil, matCtrans_nil, matFinvt4_nil)],
                FadeOut(matFinvt4_2.get_brackets()),
                run_time=1,
            ),
        )
        self.remove(matC_2, matCtrans_2)
        self.wait()

        # And we proved the Descartes' theorem (quadratic form)!
        self.play(Write(descartes_title[:5], stroke_color=color_special))
        self.add(descartes_title)
        self.wait()

        expand_t2c = {
            key: color
            for i, color in enumerate(CIRCLE_COLORS, start=1)
            for key in [
                f"k_{{{i}}}",
            ]
        }
        expand_notes = VGroup(
            TexText(text)
            for text in (
                "(1) 按矩阵乘法（或二次型）展开：",
                "(2) 移项，两边同时补充一套平方和：",
                "(3) 右边恰好是4项之和的完全平方公式，合并之后得到：",
            )
        )
        expand_formulae = VGroup(
            Tex(text, t2c=expand_t2c, use_labelled_svg=False)
            for text in (
                """
                k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2}
                - 2 k_{1} k_{2} - 2 k_{1} k_{3} - 2 k_{1} k_{4}
                - 2 k_{2} k_{3} - 2 k_{2} k_{4} - 2 k_{3} k_{4} = 0
                """,
                """
                2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right) =
                k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2}
                + 2 k_{1} k_{2} + 2 k_{1} k_{3} + 2 k_{1} k_{4}
                + 2 k_{2} k_{3} + 2 k_{2} k_{4} + 2 k_{3} k_{4}
                """,
                """
                2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right) =
                \\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2}
                """,
            )
        )
        expand_notes.set_color(GOLD_A)
        expand_groups = VGroup(
            VGroup(note, formula)
            for note, formula in zip(expand_notes, expand_formulae)
        )
        for group in expand_groups:
            group.arrange(DOWN, aligned_edge=LEFT)
        expand_groups.set_width(FRAME_WIDTH - 1)
        expand_groups.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        expand_groups.next_to(3 * DOWN, UP)

        descartes_title.generate_target()
        descartes_group.generate_target()
        descartes_title.target[5:].set_opacity(0)
        descartes_title.target.to_corner(UL)
        descartes_group.target.set_width(FRAME_WIDTH / 2)
        descartes_group.target.to_corner(UR)
        self.play(
            AnimationGroup(
                MoveToTarget(descartes_title),
                MoveToTarget(descartes_group),
                run_time=1,
            ),
            AnimationGroup(
                *[FadeIn(group, shift=UP) for group in expand_groups],
                lag_ratio=0.1,
                run_time=2,
            ),
        )
        self.wait()
        self.play(
            FlashAround(expand_formulae[-1]),
            run_time=2,
        )
        self.wait()


class TinyFixForInternallyTangentCase(Scene):
    def construct(self):
        color_c1, color_c2, color_c3, color_c4 = CIRCLE_COLORS
        color_special = YELLOW
        color_note = GREY_B
        color_extra = PURPLE_B
        basic_t2c = {
            key: color
            for i, color in enumerate(CIRCLE_COLORS, start=1)
            for key in [
                f"\\mathbf{{c}}_{{{i}}}",
                f"k_{{{i}}}",
                f"t_{{{i}}}",
                f"x_{{{i}}}",
                f"y_{{{i}}}",
            ]
        }
        basic_t2c["?"] = color_extra

        string_table_G = [
            [
                1 if (row == col) else -1
                for col in range(1, 5)
            ]
            for row in range(1, 5)
        ]
        string_table_Gprime = [
            [
                1 if ((row == col) or (row == 4) or (col == 4)) else -1
                for col in range(1, 5)
            ]
            for row in range(1, 5)
        ]
        string_table_Gprime_unknown = [
            ["1", "-1", "-1", "?"],
            ["-1", "1", "-1", "?"],
            ["-1", "-1", "1", "?"],
            ["?", "?", "?", "1"],
        ]
        string_table_R = np.diag([1, 1, 1, -1]).astype(int)

        # Setup tangent circles
        in_tangent_circles = TangentFourCircles(
            ORIGIN,
            1.2 * UP,
            RIGHT + 0.4 * UP,
            fourth_type="bigger",
        )
        in_tangent_circles.suspend_updating()
        in_tangent_circles.rotate(PI / 10).center().scale(2)
        in_tangent_circles.move_to(LEFT_SIDE / 2)
        in_tangent_circles.update_centers_from_circles()
        in_tangent_circles.refresh()
        c1, c2, c3, c4 = in_tangent_circles.get_circles()
        base_centers = in_tangent_circles.get_base_centers()
        ex_tangent_circles = TangentFourCircles(*base_centers)
        ex_tangent_circles.suspend_updating()
        self.add(ex_tangent_circles)

        # Setup matrices
        arrow = Tex("\\overset{?}{\\Rightarrow}")
        arrow.scale(1.5)
        arrow[0].set_opacity(0)
        matrix_G_note = TexText("关系矩阵 $\\mathbf{G}$")
        matrix_G = TexMatrix(
            string_table_G,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        matrix_G_group = VGroup(matrix_G_note, matrix_G)
        matrix_Gprime_note = TexText("新的关系矩阵 $\\mathbf{G}'$")
        matrix_Gprime = TexMatrix(
            string_table_Gprime,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        matrix_Gprime_group = VGroup(matrix_Gprime_note, matrix_Gprime)
        for group in (matrix_G_group, matrix_Gprime_group):
            group.arrange(DOWN)
            group.move_to(RIGHT_SIDE / 2)
            group[0].set_color(color_special)
        matrix_Gprime_unknown = TexMatrix(
            string_table_Gprime_unknown,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        for mob, ref in zip(matrix_Gprime_unknown.get_brackets(), matrix_Gprime.get_brackets()):
            mob.replace(ref)
        for mob, ref in zip(matrix_Gprime_unknown.get_entries(), matrix_Gprime.get_entries()):
            mob.move_to(ref)
            pts = mob.family_members_with_points()
            if pts:
                ref.match_color(pts[0])
        self.add(matrix_G_group, arrow)
        self.wait()

        # What if one is internally tangent to others?
        matrix_G.save_state()
        ex_tangent_circles.save_state()
        self.play(
            ReplacementTransform(ex_tangent_circles, in_tangent_circles),
            arrow.animate.set_opacity(1),
            FadeIn(matrix_Gprime_note[:2], shift=0.5 * DOWN),
            ReplacementTransform(matrix_G_note, matrix_Gprime_note[2:-1]),
            FadeIn(matrix_Gprime_note[-1], shift=0.5 * DOWN),
            ReplacementTransform(matrix_G.get_brackets(), matrix_Gprime_unknown.get_brackets()),
            *[
                ReplacementTransform(source, target)
                for source, target in zip(matrix_G.get_entries(), matrix_Gprime_unknown.get_entries())
            ],
            run_time=2,
        )
        self.remove(matrix_G)
        matrix_G.restore()
        self.wait()

        tangent_points = [
            in_tangent_circles.get_tangent_points()[ind]
            for ind in (2, 4, 5)
        ]
        self.play(
            AnimationGroup(
                Indicate(c4, color=None),
                AnimationGroup(
                    *[
                        DirectionalFlash(
                            tp, direction=in_tangent_circles.get_center() - tp,
                            color=color_c4,
                        )
                        for tp in tangent_points
                    ],
                ),
                lag_ratio=1,
                run_time=2,
            ),
        )
        self.wait()
        self.play(
            ReplacementTransform(matrix_Gprime_unknown, matrix_Gprime),
            arrow[0].animate.set_opacity(0),
            run_time=1,
        )
        arrow = arrow[1:]
        self.wait()

        # The inverse of this new relation matrix
        self.play(
            matrix_Gprime.animate.center(),
            matrix_Gprime_note.animate.center().to_edge(UP),
            FadeOut(VGroup(in_tangent_circles, arrow), shift=3 * LEFT),
        )
        self.wait()
        matR_t2c = {
            "-1": PURPLE_B,
            "0": GREY_E,
        }
        matrix_R_left, matrix_R_right = VGroup(
            TexMatrix(
                string_table_R,
                tex_config={
                    "t2c": matR_t2c,
                    "use_labelled_svg": False,
                },
            )
            for _ in range(2)
        )
        matrix_R_left.next_to(matrix_Gprime, LEFT)
        matrix_R_right.next_to(matrix_Gprime, RIGHT)
        matrix_G_mid = TexMatrix(
            [
                ["1", "-1", "-1", "1"],
                ["-1", "1", "-1", "1"],
                ["-1", "-1", "1", "1"],
                ["-1", "-1", "-1", "-1"],
            ],
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        matrix_G_mid.get_column(3).set_color(color_extra)
        matrix_G.center()
        top_R_left = Tex("\\mathbf{R}")
        top_G_mid = Tex("\\mathbf{G}")
        top_R_right = Tex("\\mathbf{R}")
        top_names = VGroup(top_R_left, top_G_mid, top_R_right)
        matrices = VGroup(matrix_R_left, matrix_G, matrix_R_right)
        for name, matrix in zip(top_names, matrices):
            name.next_to(matrix, UP)
            name.set_color(color_extra)
        top_G_mid.set_color(color_special)
        bottom_G_mid = TexText("（四圆两两外切的关系矩阵）")
        bottom_G_mid.set_color(color_note)
        bottom_G_mid.scale(0.7)
        bottom_G_mid.next_to(matrix_G, DOWN)
        self.remove(matrix_Gprime)
        self.add(matrix_Gprime.get_brackets(), matrix_Gprime.get_entries())
        self.play(
            FadeIn(top_R_left, shift=LEFT),
            FadeIn(matrix_R_left, shift=LEFT),
            FadeTransform(matrix_Gprime.get_brackets(), matrix_G_mid.get_brackets()),
            *[
                (FadeTransform if (source.tex_string != target.tex_string) else ReplacementTransform)(source, target)
                for source, target in zip(matrix_Gprime.get_entries(), matrix_G_mid.get_entries())
            ],
            FlashAround(matrix_Gprime.get_row(3), color=color_extra),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeIn(top_R_right, shift=RIGHT),
            FadeIn(matrix_R_right, shift=RIGHT),
            FadeIn(top_G_mid),
            FadeIn(bottom_G_mid),
            FadeTransform(matrix_G_mid.get_brackets(), matrix_G.get_brackets()),
            *[
                (FadeTransform if (source.tex_string != target.tex_string) else ReplacementTransform)(source, target)
                for source, target in zip(matrix_G_mid.get_entries(), matrix_G.get_entries())
            ],
            FlashAround(matrix_G_mid.get_column(3), color=color_extra),
            run_time=2,
        )
        self.wait()

        matrix_t2c = {
            "^{-1}": color_special,
            "^{\\mathrm{T}}": color_special,
            "\\mathbf{C}": color_special,
            "\\mathbf{F}": color_special,
            "\\mathbf{G}": color_special,
            "'": color_special,
            "\\mathbf{R}": color_extra,
            "\\mathbf{R}^{-1}": color_extra,
            "\\mathbf{R}^{\\mathrm{T}}": color_extra,
        }

        matGprime_formula = Tex(
            "\\mathbf{G}' = \\mathbf{R} \\mathbf{G} \\mathbf{R}",
            t2c=matrix_t2c,
            use_labelled_svg=False,
        )
        inverse_formula = Tex(
            "\\left( \\mathbf{G}' \\right)^{-1} = \\mathbf{R}^{-1} \\mathbf{G}^{-1} \\mathbf{R}^{-1}",
            t2c=matrix_t2c,
            use_labelled_svg=False,
        )
        left_center = matrix_R_left.get_center()
        right_center = VGroup(matrix_G, matrix_R_right).get_center()
        inverse_formula.move_to(right_center).to_edge(UP)
        matGprime_formula.move_to(left_center).match_y(inverse_formula)
        matrix_Gprime_note.generate_target()
        shift_vec = matGprime_formula[0].get_center() - matrix_Gprime_note[-2].get_center()
        matrix_Gprime_note.target[:-2].set_opacity(0)
        matrix_Gprime_note.target.shift(shift_vec)
        matrix_Gprime_note.target[-1].move_to(matGprime_formula[1].get_center())
        self.play(
            MoveToTarget(matrix_Gprime_note),
            ReplacementTransform(VectorizedPoint(top_R_left.get_left()), matGprime_formula[2]),
            TransformFromCopy(top_R_left[0], matGprime_formula[3]),
            TransformFromCopy(top_G_mid[0], matGprime_formula[4]),
            TransformFromCopy(top_R_right[0], matGprime_formula[5]),
            run_time=2,
        )
        self.remove(matrix_Gprime_note)
        self.add(matGprime_formula)
        self.wait()
        self.play(
            FadeTransform(matGprime_formula.copy(), inverse_formula),
        )
        self.wait()

        # Deal with brand-new formula once more...
        formulae = VGroup(
            Tex(text, t2c=matrix_t2c, use_labelled_svg=False)
            for text in [
                "\\mathbf{C} \\left( \\mathbf{G}' \\right)^{-1} \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}",
                """
                \\mathbf{C} \\mathbf{R}^{-1} \\mathbf{G}^{-1} \\mathbf{R}^{-1}
                \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}
                """,
                """
                \\mathbf{C} \\mathbf{R} \\mathbf{G}^{-1} \\mathbf{R}^{\\mathrm{T}}
                \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}
                """,
                """
                \\left( \\mathbf{C} \\mathbf{R} \\right) \\mathbf{G}^{-1}
                \\left( \\mathbf{C} \\mathbf{R} \\right)^{\\mathrm{T}} = \\mathbf{F}^{-1}
                """,
            ]
        )
        orig_formula, sub_formula_1, sub_formula_2, internal_formula = formulae
        for formula in formulae:
            formula.move_to(right_center)
        self.play(
            FadeIn(orig_formula),
            *[
                FadeOut(mob)
                for mob in (top_G_mid, bottom_G_mid, matrix_G, top_R_right, matrix_R_right)
            ],
        )
        self.wait()

        # Substitute (G')^(-1)
        self.play(
            ReplacementTransform(orig_formula[0], sub_formula_1[0]),
            FadeTransform(orig_formula[1:-6], sub_formula_1[1:-6]),
            TransformFromCopy(inverse_formula[-9:], sub_formula_1[1:-6]),
            ReplacementTransform(orig_formula[-6:], sub_formula_1[-6:]),
        )
        self.wait()

        # Matrix R has some special properties
        matR_property = Tex(
            "\\mathbf{R}^{-1} = \\mathbf{R}^{\\mathrm{T}} = \\mathbf{R}",
            t2c=matrix_t2c,
            use_labelled_svg=False,
        )
        matR_property.next_to(matrix_R_left, DOWN, buff=0.5)
        self.play(Write(matR_property, stroke_color=color_extra))
        self.wait()
        self.play(
            FocusOn(sub_formula_2[2]),
            FocusOn(sub_formula_2[6]),
            ReplacementTransform(sub_formula_1[:2], sub_formula_2[:2]),
            FadeOutToPoint(sub_formula_1[2:4], sub_formula_2[1].get_center()),
            ReplacementTransform(sub_formula_1[4:8], sub_formula_2[2:6]),
            FadeTransform(sub_formula_1[8:10], sub_formula_2[6]),
            ReplacementTransform(sub_formula_1[10:], sub_formula_2[7:]),
            run_time=2,
        )
        self.wait()

        # Group C with R
        self.play(
            AnimationGroup(
                *[
                    FadeIn(internal_formula[ind], shift=0.5 * UP)
                    for ind in (0, 3, 7, 10)
                ],
                rate_func=squish_rate_func(smooth, 0.5, 1),
            ),
            ReplacementTransform(sub_formula_2[:2], internal_formula[1:3]),
            ReplacementTransform(sub_formula_2[2:5], internal_formula[4:7]),
            ReplacementTransform(sub_formula_2[5], internal_formula[9], path_arc=PI / 2),
            ReplacementTransform(sub_formula_2[6], internal_formula[11], path_arc=-PI / 2),
            ReplacementTransform(sub_formula_2[7], internal_formula[8]),
            ReplacementTransform(sub_formula_2[8:], internal_formula[11:]),
            run_time=2,
        )
        self.wait()

        # Compare both cases
        full_t2c = basic_t2c.copy()
        full_t2c.update(matrix_t2c)
        full_t2c["{-{}}"] = color_extra
        string_table_C = np.array([
            ["k_{1}", "k_{2}", "k_{3}", "k_{4}"],
            ["t_{1}", "t_{2}", "t_{3}", "t_{4}"],
            ["k_{1} x_{1}", "k_{2} x_{2}", "k_{3} x_{3}", "k_{4} x_{4}"],
            ["k_{1} y_{1}", "k_{2} y_{2}", "k_{3} y_{3}", "k_{4} y_{4}"],
        ])
        string_table_CR = np.array([
            ["k_{1}", "k_{2}", "k_{3}", "{-{}} k_{4}"],
            ["t_{1}", "t_{2}", "t_{3}", "{-{}} t_{4}"],
            ["k_{1} x_{1}", "k_{2} x_{2}", "k_{3} x_{3}", "{-{}} k_{4} x_{4}"],
            ["k_{1} y_{1}", "k_{2} y_{2}", "k_{3} y_{3}", "{-{}} k_{4} y_{4}"],
        ])

        external_note = TexText("无内切的情况")
        internal_note = TexText("有内切的情况")
        external_note.move_to(LEFT_SIDE / 2).to_edge(UP)
        internal_note.move_to(RIGHT_SIDE / 2).to_edge(UP)
        internal_formula.generate_target()
        internal_formula.target.next_to(internal_note, DOWN)
        external_formula = Tex(
            """
            \\left( \\mathbf{C} \\right) \\mathbf{G}^{-1}
            \\left( \\mathbf{C} \\right)^{\\mathrm{T}} = \\mathbf{F}^{-1}
            """,
            t2c=matrix_t2c,
            use_labelled_svg=False,
        )
        external_formula.next_to(external_note, DOWN)
        external_core = VGroup(
            Tex("\\mathbf{C} =", t2c=full_t2c, use_labelled_svg=False),
            TexMatrix(
                string_table_C,
                tex_config={
                    "t2c": full_t2c,
                    "use_labelled_svg": False,
                },
            )
        )
        external_core.arrange(RIGHT)
        external_core.move_to(LEFT_SIDE / 2)
        internal_core = VGroup(
            Tex("\\mathbf{C} \\mathbf{R} =", t2c=full_t2c, use_labelled_svg=False),
            TexMatrix(
                string_table_CR,
                tex_config={
                    "t2c": full_t2c,
                    "use_labelled_svg": False,
                },
            )
        )
        for core, pos in zip([external_core, internal_core], [LEFT_SIDE / 2, RIGHT_SIDE / 2]):
            core.arrange(RIGHT)
            core.set_width(FRAME_WIDTH / 2 - 2)
            core.move_to(pos)
        internal_core.match_height(external_core)

        self.play(
            FadeOut(VGroup(matGprime_formula, top_R_left, matrix_R_left, matR_property), shift=3 * LEFT),
            FadeOut(inverse_formula, shift=UP),
            *[
                FadeIn(mob)
                for mob in (external_note, internal_note, external_formula)
            ],
            MoveToTarget(internal_formula),
            run_time=2,
        )
        self.wait()

        # Multiply R on the right changes the sign of the last column
        self.play(
            AnimationGroup(
                FlashUnder(external_formula[:3], color=color_special),
                FlashUnder(external_formula[6:9], color=color_special),
                FlashUnder(internal_formula[:4], color=color_extra),
                FlashUnder(internal_formula[7:11], color=color_extra),
                run_time=2,
            ),
            AnimationGroup(
                *[
                    FadeIn(mob)
                    for mob in (external_core, internal_core)
                ],
                run_time=1,
            ),
        )
        self.wait()
        self.play(
            FlashAround(internal_core[1].get_column(3), color=color_extra),
            run_time=2,
        )
        self.wait()

        # The negative sign will be carried to the end
        external_theorem = VGroup(
            Tex(text, t2c=full_t2c, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2}",
                "= 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)"
            ]
        )
        internal_theorem = VGroup(
            Tex(text, t2c=full_t2c, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + k_{3} + ({-{}}k_{4}) \\right)^{2}",
                "= 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + ({-{}}k_{4})^{2} \\right)"
            ]
        )
        external_arrow = Tex("\\Rightarrow")
        external_arrow.rotate(-PI / 2)
        internal_arrow = external_arrow.copy()

        cores = VGroup(external_core, internal_core)
        arrows = VGroup(external_arrow, internal_arrow)
        theorems = VGroup(external_theorem, internal_theorem)
        for theorem, arrow, core in zip(theorems, arrows, cores):
            theorem.arrange(DOWN, aligned_edge=RIGHT)
            theorem.scale(0.9)
            theorem.next_to(3 * DOWN, UP).match_x(core)
            arrow.next_to(theorem, UP)
            core.generate_target()
            core.target.next_to(arrow, UP)

        self.play(
            FadeOut(external_formula),
            FadeOut(internal_formula),
            MoveToTarget(external_core),
            MoveToTarget(internal_core),
            GrowFromPoint(external_arrow, external_core.get_bottom()),
            GrowFromPoint(external_theorem, external_core.get_bottom()),
            GrowFromPoint(internal_arrow, internal_core.get_bottom()),
            GrowFromPoint(internal_theorem, internal_core.get_bottom()),
        )
        self.wait()
        result = internal_theorem[1].get_parts_by_tex("({-{}}k_{4})")
        self.play(
            *[
                FlashAround(mob, color=color_extra)
                for theorem_part in internal_theorem
                for mob in theorem_part.get_parts_by_tex("({-{}}k_{4})")
            ],
        )
        self.wait()

        # Bring back 4 tangent circles
        ex_tangent_circles.restore()
        ex_tangent_circles.set_height(3.5).next_to(external_note, DOWN, buff=0.5)
        ex_tangent_circles.update_centers_from_circles()
        ex_tangent_circles.refresh()
        in_tangent_circles.set_height(3.5).next_to(internal_note, DOWN, buff=0.5)
        in_tangent_circles.update_centers_from_circles()
        in_tangent_circles.refresh()

        orig_k_note = TexText("（$k$：曲率）")
        new_k_note = TexText("（$k$：有向曲率）")
        for note in (orig_k_note, new_k_note):
            note.set_color(color_note)
            note.scale(0.8)
            note.match_y(in_tangent_circles, DOWN)
        new_k_note[-5:-3].set_color(color_extra)
        self.play(
            FadeIn(ex_tangent_circles),
            FadeIn(in_tangent_circles),
            FadeOut(VGroup(external_core, external_arrow)),
            FadeOut(VGroup(internal_core, internal_arrow)),
            FadeInFromPoint(orig_k_note, orig_k_note.get_center()),
        )
        self.wait()

        # From curvature to signed curvature
        internal_theorem_alt = external_theorem.copy()
        internal_theorem_alt.move_to(internal_theorem)
        _, _, _, c4 = in_tangent_circles.get_circles()
        tangent_points = [
            in_tangent_circles.get_tangent_points()[ind]
            for ind in (2, 4, 5)
        ]
        self.play(
            AnimationGroup(
                Indicate(c4, color=None, scale_factor=1.1),
                AnimationGroup(
                    *[
                        DirectionalFlash(
                            tp, direction=in_tangent_circles.get_center() - tp,
                            color=color_c4,
                        )
                        for tp in tangent_points
                    ],
                ),
                lag_ratio=1,
                run_time=2,
            ),
        )
        self.wait()

        self.play(
            ReplacementTransform(orig_k_note[:3], new_k_note[:3]),
            ReplacementTransform(orig_k_note[-3:], new_k_note[-3:]),
            ReplacementTransform(
                VGroup(
                    *[internal_theorem[0][ind] for ind in (10, 11, 14)],
                    *[internal_theorem[1][ind] for ind in (15, 16, 19)],
                ),
                new_k_note[-5:-3],
                path_arc=PI / 4,
            ),
            ReplacementTransform(internal_theorem[0][:10], internal_theorem_alt[0][:10]),
            ReplacementTransform(internal_theorem[0][12:14], internal_theorem_alt[0][10:12]),
            ReplacementTransform(internal_theorem[0][15:], internal_theorem_alt[0][12:]),
            ReplacementTransform(internal_theorem[1][:15], internal_theorem_alt[1][:15]),
            ReplacementTransform(internal_theorem[1][17], internal_theorem_alt[1][15]),
            ReplacementTransform(internal_theorem[1][20], internal_theorem_alt[1][16]),
            ReplacementTransform(internal_theorem[1][18], internal_theorem_alt[1][17]),
            ReplacementTransform(internal_theorem[1][21], internal_theorem_alt[1][18]),
            run_time=2,
        )
        self.wait()

        # Descartes' Theorem proved (again)
        descartes_note = TexText("笛卡尔定理：")
        descartes_note.set_color(color_special)
        descartes_content = Tex(
            """
            \\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2}
            = 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)
            """,
            t2c=full_t2c,
            use_labelled_svg=False,
        )
        descartes_title = VGroup(descartes_note, descartes_content)
        descartes_title.arrange(RIGHT)
        descartes_title.to_edge(UP)
        new_k_note.generate_target()
        # new_k_note.target.next_to(descartes_title, DOWN, aligned_edge=RIGHT)
        new_k_note.target.next_to(descartes_title, DOWN)
        new_k_note.target.set_color(color_note)
        ex_k1, ex_k2, ex_k3, ex_k4 = ex_ks = [3, 6, 7, 34]
        ex_p1, ex_p2, ex_p3 = get_tangent_circle_centers(ex_k1, ex_k2, ex_k3)
        external_example = TangentFourCircles(ex_p1, ex_p2, ex_p3)
        external_example.suspend_updating()
        in_k1, in_k2, in_k3, in_k4 = in_ks = [10, 15, 19, -6]
        in_p1, in_p2, in_p3 = get_tangent_circle_centers(in_k1, in_k2, in_k3)
        internal_example = TangentFourCircles(in_p1, in_p2, in_p3, fourth_type="bigger")
        internal_example.suspend_updating()

        examples = VGroup(external_example, internal_example)
        random.seed(570)
        for example, pos in zip(examples, [LEFT_SIDE / 2, RIGHT_SIDE / 2]):
            example.suspend_updating()
            example.rotate(random.random() * 2 * PI)
            example.set_height(5)
            example.move_to(pos)
            example.next_to(new_k_note.target, DOWN, coor_mask=[0, 1, 0])
            example.update_centers_from_circles()
            example.refresh()
        sep_line = Underline(descartes_title)
        sep_line.rotate(PI / 2).match_height(external_example)
        sep_line.set_color(color_note).set_opacity(0.5)
        sep_line.match_y(examples)

        self.play(
            AnimationGroup(
                FadeOut(external_note),
                FadeOut(internal_note),
                run_time=1,
            ),
            AnimationGroup(
                FadeInFromPoint(descartes_note, descartes_note.get_center()),
                ReplacementTransform(external_theorem[0], descartes_content[:14]),
                ReplacementTransform(external_theorem[1], descartes_content[14:]),
                ReplacementTransform(internal_theorem_alt[0], descartes_content[:14]),
                ReplacementTransform(internal_theorem_alt[1], descartes_content[14:]),
                MoveToTarget(new_k_note),
                ReplacementTransform(ex_tangent_circles, external_example),
                ReplacementTransform(in_tangent_circles, internal_example),
                FadeInFromPoint(sep_line, sep_line.get_center()),
                run_time=2,
            ),
        )
        self.wait()

        # All curvatures are integers
        ex_c1, ex_c2, ex_c3, ex_c4 = ex_cs = external_example.get_circles()
        ex_texts = VGroup(
            Tex(f"k_{{{i}}}={ki}").scale(0.7).set_color(color).move_to(circle)
            for i, (ki, color, circle) in enumerate(
                zip(ex_ks, CIRCLE_COLORS, ex_cs),
                start=1,
            )
        )
        ex_texts[-1].next_to(external_example.get_corner(UL), DR)
        ex_arrow = CurvedArrow(
            ex_texts[-1].get_bottom() + 0.2 * DOWN,
            ex_c4.get_corner(DL) + 0.1 * LEFT,
        )
        ex_arrow.match_color(ex_texts[-1])
        in_c1, in_c2, in_c3, in_c4 = in_cs = internal_example.get_circles()
        in_texts = VGroup(
            Tex(f"k_{{{i}}}={ki}").scale(0.7).set_color(color).move_to(circle)
            for i, (ki, color, circle) in enumerate(
                zip(in_ks, CIRCLE_COLORS, in_cs),
                start=1,
            )
        )
        in_texts[-1].next_to(internal_example.get_bottom(), UP, buff=0.5)
        in_texts[-1].match_x(in_texts[-2])
        self.play(
            *[
                FadeIn(mob)
                for mob in (ex_texts, in_texts, ex_arrow)
            ],
        )
        self.wait()

        # Seems cheesy
        curr_mobs = Group(*self.mobjects)
        frames = VGroup(ScreenRectangle() for _ in range(2))
        frames.arrange(RIGHT, buff=1)
        frames.set_width(FRAME_WIDTH - 1)
        frame_notes = VGroup(
            TexText(text)
            for text in ("初等变换求逆", "暴力硬算拼凑")
        )
        frame_pics = Group(
            ImageMobject(img_name)
            for img_name in ("DescartesProof-inverse.png", "DescartesProof-expand.png")
        )
        for note, pic, frame in zip(frame_notes, frame_pics, frames):
            frame.set_stroke(color=color_note)
            note.set_color(color_note)
            note.next_to(frame, UP)
            pic.match_height(frame)
            pic.move_to(frame)
        Group(frame_notes, frames, frame_pics).center()
        special_treatment = TexText("特殊处理?")
        special_treatment.set_color(color_special)
        special_treatment.to_edge(UP)
        self.play(
            FadeOut(curr_mobs),
            *[
                FadeIn(mob)
                for mob in (frames, frame_notes, frame_pics, special_treatment)
            ],
        )
        self.wait()

        # To higher dimensions
        full_t2c["k_{n+2}"] = SPHERE_COLORS[-1]
        higher_text = TexText("高维空间")
        higher_text.move_to(special_treatment).match_color(special_treatment)
        higher_core = VGroup(
            Tex(text)
            for text in [
                "\\mathbf{C}^{\\mathrm{T}} \\mathbf{F} \\mathbf{C} = \\mathbf{G}",
                "\\mathbf{C} \\mathbf{G}^{-1} \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}",
            ]
        )
        higher_core.arrange(DOWN)
        higher_core.set_color(color_special)
        higher_arrow = Tex("\\Rightarrow")
        higher_arrow.rotate(-PI / 2)
        soddy_gosset_theorem = VGroup(
            Tex(text, t2c=full_t2c, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + \\cdots + k_{n+2} \\right)^{2}",
                "= n \\left( k_{1}^{2} + k_{2}^{2} + \\cdots + k_{n+2}^{2} \\right)"
            ]
        )
        soddy_gosset_theorem.arrange(DOWN, aligned_edge=RIGHT)
        soddy_gosset_theorem[1][-5].set_color(WHITE)  # color fixing
        soddy_gosset_theorem[1][-2].set_color(SPHERE_COLORS[-1])  # color fixing
        higher_group = VGroup(
            higher_core, higher_arrow, soddy_gosset_theorem
        )
        higher_group.arrange(DOWN, buff=0.5)
        higher_group.move_to(VGroup(frame_notes[0], frames[0]))
        self.play(
            *[
                FadeOut(mob)
                for mob in (frame_pics, frames, frame_notes)
            ],
            FadeTransform(special_treatment, higher_text),
            FadeIn(higher_group),
        )
        self.wait()
        self.play(FlashAround(higher_core))
        self.wait()


class ChangesInHigherDimensions(Scene):
    def construct(self):
        # Setup
        color_c1, color_c2, color_c3, color_c4 = CIRCLE_COLORS
        color_special = YELLOW
        color_note = GREY_B
        basic_t2c = {
            key: color
            for i, color in enumerate(CIRCLE_COLORS, start=1)
            for key in [
                f"C_{{{i}}}",
                f"\\mathbf{{c}}_{{{i}}}",
                f"k_{{{i}}}",
                f"t_{{{i}}}",
                f"x_{{{i}}}",
                f"y_{{{i}}}",
            ]
        }
        pos_note = TexText(
            "$C_{1}$与$C_{2}$ \\\\ 位置关系",
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        pos_relations = VGroup(
            TexText(text)
            for text in ["外离", "外切", "相交", "内切", "内含"]
        )
        pos_group = VGroup(pos_note, pos_relations)
        ip_note = TexText(
            "$\\langle \\mathbf{c}_{1}, \\mathbf{c}_{2} \\rangle$ \\\\ 内积",
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        ip_relations = VGroup(
            Tex(text)
            for text in ["(-\\infty, -1)", "-1", "(-1, 1)", "1", "(1, +\\infty)"]
        )
        ip_group = VGroup(ip_note, ip_relations)
        for group in (pos_group, ip_group):
            note, relations = group
            relations.arrange(DOWN, buff=0.3)
            relations.next_to(note, DOWN, buff=0.8)
            relations[1::2].set_color(color_special)
        ip_group.next_to(pos_group, RIGHT, buff=1)
        ip_note.match_y(pos_note)
        for mob, ref in zip(ip_relations, pos_relations):
            mob.match_y(ref)
        grid_group = VGroup(pos_group, ip_group)
        grid_group.center().to_edge(LEFT)
        sep_line = Line(LEFT, RIGHT)
        sep_line.set_color(color_note)
        sep_line.set_width(grid_group.get_width() * 1.05)
        sep_line.move_to(grid_group)
        sep_line.move_to((ip_note.get_bottom() + ip_relations.get_top()) / 2, coor_mask=[0, 1, 0])
        grid_group.add(sep_line)
        ip_note[:-2].set_opacity(0)
        self.add(grid_group)

        # Linear algebra part
        for key in ("C", "\\mathbf{c}", "r", "k", "t", "x", "y", "z", "w"):
            basic_t2c[f"{key}_{{i}}"] = TEAL
            basic_t2c[f"{key}_{{j}}"] = PURPLE
        basic_t2c["(x_{i}, y_{i})"] = basic_t2c["C_{i}"]
        basic_t2c["{d}"] = color_special
        basic_t2c["\\mathbf{F}"] = color_special
        circle_info_title = TexText(
            "圆$C_{i}$",
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        circle_info_content = VGroup(
            TexText(
                text,
                t2c=basic_t2c,
                use_labelled_svg=False,
            )
            for text in [
                "半径$r_{i}$",
                "圆心$(x_{i}, y_{i})$",
            ]
        )
        circle_info_content.arrange(DOWN, aligned_edge=LEFT)
        circle_vector_title = TexText(
            "向量$\\mathbf{c}_{i}$",
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        circle_vector_content = MobjectMatrix(
            [Tex(text) for text in ["k_{i}", "t_{i}", "k_{i} x_{i}", "k_{i} y_{i}"]],
            n_cols=1,
        )
        circle_vector_content.scale(0.8)
        circle_vector_content.set_color(basic_t2c["C_{i}"])
        arrow_title = Tex("\\leftrightarrow")
        VGroup(circle_info_title, arrow_title, circle_vector_title).arrange(RIGHT, buff=0.6)
        circle_vector_content.next_to(circle_vector_title, DOWN)
        circle_info_content.next_to(circle_info_title, DOWN)
        circle_info_content.match_y(circle_vector_content)
        circle_group = VGroup(
            circle_info_title, circle_info_content,
            arrow_title,
            circle_vector_title, circle_vector_content,
        )

        # Inner product part
        inner_product_formula = Tex(
            """
            \\dfrac{r_{i}^2 + r_{j}^2 - {d}^2}{2 r_{i} r_{j}} =
            \\begin{matrix}
              \\langle \\mathbf{c}_{i}, \\mathbf{c}_{j} \\rangle \\\\
              \\text{内积}
            \\end{matrix}
            = \\mathbf{c}_{i}^{\\mathrm{T}} \\mathbf{F} \\mathbf{c}_{j}
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        linalg_group = VGroup(circle_group, inner_product_formula)
        linalg_group.arrange(DOWN, buff=0.8)
        linalg_group.to_edge(RIGHT)
        self.wait()
        self.play(
            FadeIn(linalg_group, shift=2 * LEFT),
            ip_note.animate.set_opacity(1),
        )
        self.wait()

        # Many things won't change
        self.play(FlashAround(inner_product_formula))
        self.wait()
        self.play(FlashAround(VGroup(pos_relations, ip_relations)))
        self.wait()

        # Switch to higher dimension
        string_table_F_lowD = [
            [0, "-\\tfrac{1}{2}", 0, 0],
            ["-\\tfrac{1}{2}", 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        string_table_F = [
            [0, "-\\tfrac{1}{2}", 0, 0, "\\cdots", 0],
            ["-\\tfrac{1}{2}", 0, 0, 0, "\\cdots", 0],
            [0, 0, 1, 0, "\\cdots", 0],
            [0, 0, 0, 1, "\\cdots", 0],
            ["\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\ddots", "\\vdots"],
            [0, 0, 0, 0, "\\cdots", 1],
        ]
        highD_vector_title = circle_vector_title.copy()
        highD_vector_temp = circle_vector_content.copy()
        highD_vector_content = MobjectMatrix(
            [
                Tex(text)
                for text in ["k_{i}", "t_{i}", "k_{i} x_{i}", "k_{i} y_{i}", "\\vdots", "k_{i} w_{i}"]
            ],
            n_cols=1,
        )
        highD_vector_title.to_edge(UP)
        highD_vector_title.match_x(LEFT_SIDE / 2)
        for vector in (highD_vector_temp, highD_vector_content):
            vector.match_color(highD_vector_temp)
            vector.next_to(highD_vector_title, DOWN)
        temp_matF = inner_product_formula[-3]
        highD_metric_title = TexText("度量矩阵$\\mathbf{F}$")
        highD_metric_title[-1].set_color(color_special)
        highD_metric_title.to_edge(UP)
        highD_metric_title.match_x(RIGHT_SIDE / 2).match_y(highD_vector_title)
        highD_metric_temp = TexMatrix(string_table_F_lowD, element_alignment_corner=ORIGIN)
        highD_metric_content = TexMatrix(string_table_F, element_alignment_corner=ORIGIN)
        for matrix in (highD_metric_temp, highD_metric_content):
            matrix.scale(0.8)
            matrix.set_color(color_special)
            matrix.next_to(highD_metric_title, DOWN)
        highD_metric_temp.set_color(color_special)
        highD_vector_content.match_height(highD_metric_content, about_edge=UP)
        self.play(
            AnimationGroup(
                ReplacementTransform(circle_vector_title, highD_vector_title),
                ReplacementTransform(circle_vector_content, highD_vector_temp),
                FadeInFromPoint(highD_metric_title[:-1], temp_matF.get_left()),
                ReplacementTransform(temp_matF, highD_metric_title[-1]),
                FadeInFromPoint(highD_metric_temp, temp_matF.get_bottom()),
                run_time=2,
            ),
            AnimationGroup(
                *[
                    FadeOut(mob)
                    for mob in [
                        grid_group, arrow_title,
                        circle_info_title, circle_info_content,
                        inner_product_formula[-2:], inner_product_formula[:-3],
                    ]
                ],
                run_time=1,
            ),
        )
        self.remove(*self.mobjects)
        self.add(
            highD_vector_title, highD_vector_temp,
            highD_metric_title, highD_metric_temp,
        )
        self.wait()

        # Modification 1: the vector needs more reduced coordinates
        more_coord_text = TexText(
            "$\\leftarrow$ 需要更多的坐标描述球心",
            t2c={"更多的坐标": highD_vector_content.get_color()}
        )
        more_coord_text.next_to(highD_vector_temp, RIGHT, aligned_edge=DOWN, buff=0.5)
        texts = VGroup(
            TexText(text)
            for text in ["$\\leftarrow$ 曲率", "$\\leftarrow$ 反演曲率", "球心的约化坐标"]
        )
        orig_curv_text, inv_curv_text, reduced_coord_text = texts
        for mob, row_ind in zip(texts[:-1], [0, 1]):
            ref = highD_metric_content.get_row(row_ind)
            mob.next_to(highD_vector_content, RIGHT)
            mob.match_y(ref)
        brace = Brace(highD_vector_content.get_entries()[2:], direction=RIGHT)
        brace.next_to(highD_vector_content, RIGHT, coor_mask=[1, 0, 0])
        brace.put_at_tip(reduced_coord_text)
        VGroup(texts, brace).set_color(color_note)
        self.play(
            FadeOut(VGroup(highD_metric_title, highD_metric_temp)),
            FadeIn(more_coord_text),
        )
        self.wait()
        self.remove(highD_vector_temp)
        self.add(highD_vector_temp.get_brackets(), highD_vector_temp.get_entries())
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(more_coord_text),
                    *[
                        FadeTransform(source, target)
                        for source, target in zip(
                            highD_vector_temp.get_brackets(), highD_vector_content.get_brackets()
                        )
                    ],
                    ReplacementTransform(highD_vector_temp.get_entries(), highD_vector_content.get_entries()[:4]),
                    FadeInFromPoint(highD_vector_content.get_entries()[4:], highD_vector_temp.get_bottom()),
                ),
                AnimationGroup(
                    FadeIn(mob)
                    for mob in (texts, brace)
                ),
                lag_ratio=1,
            ),
            run_time=2,
        )
        self.wait()

        # Modification 2: the metric matrix needs to match the size
        vector_group = VGroup(highD_vector_title, highD_vector_content, texts, brace)
        element_anims = []
        for row in range(6):
            for col in range(6):
                if row < 4 and col < 4:
                    source = highD_metric_temp.get_row(row)[col]
                elif row < 4:
                    source = VectorizedPoint(highD_metric_temp.get_row(row)[3].get_right())
                elif col < 4:
                    source = VectorizedPoint(highD_metric_temp.get_row(3)[col].get_bottom())
                else:
                    source = VectorizedPoint(highD_metric_temp.get_row(3)[3].get_corner(DR))
                target = highD_metric_content.get_row(row)[col]
                element_anims.append(ReplacementTransform(source, target))
        self.play(
            vector_group.animate.to_edge(LEFT, buff=1),
            FadeIn(VGroup(highD_metric_title, highD_metric_temp)),
        )
        self.wait()
        self.remove(highD_metric_temp)
        self.add(highD_metric_temp.get_brackets(), highD_metric_temp.get_entries())
        self.play(
            *[
                FadeTransform(source, target)
                for source, target in zip(
                    highD_metric_temp.get_brackets(), highD_metric_content.get_brackets()
                )
            ],
            *element_anims,
        )
        self.wait()

        # F is a block diagonal matrix
        cover_rect = Rectangle(
            stroke_width=0,
            fill_color=MAROON_B, fill_opacity=0.3,
        )
        rects = VGroup([cover_rect.copy() for _ in range(4)])
        vector_top_rect, vector_bottom_rect, matrix_top_rect, matrix_bottom_rect = rects
        vector_top_rect.surround(highD_vector_content.get_entries()[:2])
        vector_bottom_rect.surround(highD_vector_content.get_entries()[2:])
        matrix_top_rect.surround(
            VGroup(
                highD_metric_content.get_row(0)[:2],
                highD_metric_content.get_row(1)[:2],
            )
        )
        matrix_bottom_rect.surround(
            VGroup(
                highD_metric_content.get_row(2)[2:],
                highD_metric_content.get_row(5)[2:],
            )
        )
        self.play(
            FadeInFromPoint(vector_top_rect, vector_top_rect.get_center()),
            FadeInFromPoint(matrix_top_rect, matrix_top_rect.get_center()),
        )
        self.wait()
        self.play(
            FadeInFromPoint(vector_bottom_rect, vector_bottom_rect.get_center()),
            FadeInFromPoint(matrix_bottom_rect, matrix_bottom_rect.get_center()),
        )
        self.wait()

        # Inner product is still a vector-metric-vector sandwich
        prop_formula = Tex(
            """
            \\langle \\mathbf{c}_{i}, \\mathbf{c}_{j} \\rangle
            = \\mathbf{c}_{i}^{\\mathrm{T}} \\mathbf{F} \\mathbf{c}_{j}
            = - \\dfrac{1}{2} t_{i} k_{j} - \\dfrac{1}{2} k_{i} t_{j}
            + k_{i} x_{i} k_{j} x_{j} + k_{i} y_{i} k_{j} y_{j} + \\cdots + k_{i} w_{i} k_{j} w_{j}
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        prop_formula.set_width(FRAME_WIDTH - 1)
        prop_formula.next_to(3 * DOWN, UP)
        curr_mobs = VGroup(*self.mobjects)
        curr_mobs.generate_target()
        curr_mobs.target.set_height(4.8).center().to_edge(UP)
        self.add(curr_mobs)
        self.play(
            FadeIn(prop_formula, shift=0.5 * UP),
            MoveToTarget(curr_mobs),
        )
        self.wait()

        # Now on to multiple spheres
        subscripts = ("1", "2", "3", "4", "n+2")
        sphere_vectors_list = []
        for key in subscripts:
            elements = [
                f"k_{{{key}}}",
                f"t_{{{key}}}",
                f"k_{{{key}}} x_{{{key}}}",
                f"k_{{{key}}} y_{{{key}}}",
                "\\vdots",
                f"k_{{{key}}} w_{{{key}}}",
            ]
            v = MobjectMatrix([Tex(text)for text in elements], n_cols=1)
            sphere_vectors_list.append(v)
        sphere_vectors_list.insert(4, Tex("\\cdots"))
        sphere_vectors = VGroup(*sphere_vectors_list)
        sphere_vectors.arrange(RIGHT, buff=0.5).to_edge(UP)
        sphere_vectors.match_height(highD_vector_content)
        sphere_vectors.match_y(highD_vector_content, UP)
        spaced_colors = SPHERE_COLORS[:]
        spaced_colors.insert(4, WHITE)
        for vector, color in zip(sphere_vectors, spaced_colors):
            vector.set_color(color)
            vector.generate_target()
            vector.set_opacity(0).move_to(highD_vector_content)

        self.play(
            AnimationGroup(
                *[MoveToTarget(mob) for mob in sphere_vectors],
                lag_ratio=0.05,
                run_time=2,
            ),
            AnimationGroup(
                *[
                    FadeOut(mob)
                    for mob in [
                        highD_vector_title, highD_vector_content,
                        texts, brace,
                        highD_metric_title, highD_metric_content,
                        vector_top_rect, vector_bottom_rect,
                        matrix_top_rect, matrix_bottom_rect,
                    ]
                ],
                run_time=1,
            ),
        )
        self.wait()
        self.add(sphere_vectors)
        self.wait()

        # Arrange them to get configuration matrix C
        string_table_C = [
            ["k_{1}", "k_{2}", "k_{3}", "k_{4}", "\\cdots", "k_{n+2}"],
            ["t_{1}", "t_{2}", "t_{3}", "t_{4}", "\\cdots", "t_{n+2}"],
            ["k_{1} x_{1}", "k_{2} x_{2}", "k_{3} x_{3}", "k_{4} x_{4}", "\\cdots", "k_{n+2} x_{n+2}"],
            ["k_{1} y_{1}", "k_{2} y_{2}", "k_{3} y_{3}", "k_{4} y_{4}", "\\cdots", "k_{n+2} y_{n+2}"],
            ["\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\ddots", "\\vdots"],
            ["k_{1} w_{1}", "k_{2} w_{2}", "k_{3} w_{3}", "k_{4} w_{4}", "\\cdots", "k_{n+2} w_{n+2}"],
        ]
        matrix_C = TexMatrix(
            string_table_C,
            h_buff=0.1,
            element_alignment_corner=ORIGIN,
        )
        matrix_C.match_height(sphere_vectors)
        matrix_C.match_y(sphere_vectors, UP)
        for col_ind, color in enumerate(spaced_colors):
            matrix_C.get_column(col_ind).set_color(color)
        matrix_C_name = TexText("构型矩阵$\\mathbf{C}$")
        matrix_C_name.set_color(color_special)
        matrix_C_name.to_edge(UP)
        self.play(
            AnimationGroup(
                *[FadeOut(vector.get_brackets()[0])for vector in sphere_vectors[1:-2]],
                *[FadeOut(vector.get_brackets()[1])for vector in sphere_vectors[:-2]],
                FadeOut(sphere_vectors[-1].get_brackets()[0]),
                run_time=1,
            ),
            AnimationGroup(
                Write(matrix_C_name, stroke_color=color_special),
                ReplacementTransform(sphere_vectors[0].get_brackets()[0], matrix_C.get_brackets()[0]),
                ReplacementTransform(sphere_vectors[-1].get_brackets()[1], matrix_C.get_brackets()[1]),
                *[
                    ReplacementTransform(vector.get_entries(), matrix_C.get_column(ind))
                    for ind, vector in enumerate(sphere_vectors)
                    if ind != 4
                ],
                *[
                    TransformFromCopy(sphere_vectors[4], entry)
                    for entry in matrix_C.get_column(4)[:-1]
                ],
                ReplacementTransform(sphere_vectors[4], matrix_C.get_column(4)[-1]),
                run_time=2,
            ),
        )
        self.wait()

        # On to matrix multiplication
        position_relation = Tex("\\mathbf{C}^{\\mathrm{T}} \\mathbf{F} \\mathbf{C} = \\mathbf{G}")
        property_relation = Tex("\\mathbf{C} \\mathbf{G}^{-1} \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}")
        position_relation.move_to(LEFT_SIDE / 2)
        property_relation.move_to(RIGHT_SIDE / 2)
        for relation in (position_relation, property_relation):
            relation.set_color(color_special)
            relation.match_y(prop_formula)
        self.play(
            AnimationGroup(
                FadeOut(prop_formula[:8]),
                FadeOut(prop_formula[10]),
                FadeOut(prop_formula[13]),
                FadeOut(prop_formula[15:]),
                run_time=1,
            ),
            AnimationGroup(
                ReplacementTransform(prop_formula[8], position_relation[0]),
                ReplacementTransform(prop_formula[9], position_relation[1]),
                ReplacementTransform(prop_formula[11], position_relation[2]),
                ReplacementTransform(prop_formula[12], position_relation[3]),
                ReplacementTransform(prop_formula[14], position_relation[4]),
                ReplacementTransform(VectorizedPoint(prop_formula[15].get_left()), position_relation[5]),
                run_time=2,
            ),
        )
        self.wait()

        # Take the inverse
        path_arc = -PI / 4
        self.play(
            TransformFromCopy(position_relation[:2], property_relation[4:6], path_arc=path_arc),
            TransformFromCopy(position_relation[2], property_relation[7], path_arc=path_arc),
            TransformFromCopy(VectorizedPoint(position_relation[2].get_corner(UR)), property_relation[8:], path_arc=path_arc),
            TransformFromCopy(position_relation[4], property_relation[6]),
            TransformFromCopy(position_relation[3], property_relation[0], path_arc=path_arc),
            TransformFromCopy(position_relation[5], property_relation[1], path_arc=path_arc),
            TransformFromCopy(VectorizedPoint(position_relation[5].get_corner(UR)), property_relation[2:4], path_arc=path_arc),
            run_time=2,
        )
        self.wait()

        # Core meaning doesn't change
        matrix_C_group = VGroup(matrix_C, matrix_C_name)
        self.add(matrix_C_group)
        position_meaning = TexText("任意两个超球的位置关系")
        property_meaning = TexText("任意两个属性的约束关系")
        relations = VGroup(position_relation, property_relation)
        meanings = VGroup(position_meaning, property_meaning)
        for relation, meaning in zip(relations, meanings):
            relation.generate_target()
            relation.target.match_y(ORIGIN)
            meaning.next_to(relation.target, UP)
        matrix_C_group.generate_target()
        matrix_C_group.target.next_to(meanings, UP, coor_mask=[0, 1, 0])
        matrix_C_group.target.set_opacity(0)
        self.play(
            MoveToTarget(position_relation),
            MoveToTarget(property_relation),
            MoveToTarget(matrix_C_group),
        )
        self.remove(matrix_C_group)
        self.wait()
        self.play(
            FadeInFromPoint(position_meaning, position_relation.get_top()),
            FadeInFromPoint(property_meaning, property_relation.get_top()),
        )
        self.wait()
        self.clear()
        self.add(
            position_relation, position_meaning,
            property_relation, property_meaning,
        )

        # Matrix G has the same structure
        string_table_G = [
            [
                "1" if (row == col) else "-1"
                for col in range(1, 7)
            ]
            for row in range(1, 7)
        ]
        string_table_G[4] = ["\\vdots"] * 6
        for row in string_table_G:
            row[4] = "\\cdots"
        string_table_G[4][4] = "\\ddots"
        matrix_G = TexMatrix(string_table_G)
        matrix_G.set_height(4)
        matrix_G.move_to(RIGHT_SIDE / 2 + DOWN)
        matrix_G_name = TexText("关系矩阵$\\mathbf{G}$")
        matrix_G_name.set_color(color_special)
        matrix_G_name.next_to(matrix_G, UP)
        curr_mobs = VGroup(*self.mobjects)
        curr_mobs.generate_target()
        curr_mobs.target.to_edge(UP)
        curr_mobs.target.set_opacity(0.2)
        self.play(
            FadeIn(matrix_G),
            FadeIn(matrix_G_name),
            MoveToTarget(curr_mobs),
        )
        self.wait()

        # Diagonal and off-diagnol entries
        diag_entries = VGroup()
        off_diag_entries = VGroup()
        for entry in matrix_G.get_entries():
            ts = entry.tex_string
            if ts in ["1", "\\ddots"]:
                diag_entries.add(entry)
            else:
                off_diag_entries.add(entry)
        color_lit = RED_B
        color_dark = GREY_E
        diag_note = TexText("超球与自身的内积$=1$")
        off_diag_note = TexText("不同超球之间的内积$=-1$")
        for note in (diag_note, off_diag_note):
            note.next_to(matrix_G, LEFT, buff=1)
            note.set_color(color_lit)

        self.play(
            diag_entries.animate.set_color(color_lit),
            off_diag_entries.animate.set_color(color_dark),
            FadeIn(diag_note),
        )
        self.wait()
        self.play(
            off_diag_entries.animate.set_color(color_lit),
            diag_entries.animate.set_color(color_dark),
            FadeTransform(diag_note, off_diag_note),
        )
        self.wait()

        # A picture for internally tangent case
        internal_image = ImageMobject("TangentSpheres-Inner.png")
        internal_note = TexText("有向曲率 \\\\ $k < 0$")
        internal_note.scale(0.7)
        internal_note.set_color(SPHERE_COLORS[-2])
        internal_note.next_to(internal_image.get_corner(UR), DOWN, buff=0)
        internal_group = Group(internal_image, internal_note)
        internal_group.move_to(off_diag_note)
        self.play(
            FadeOut(off_diag_note),
            FadeIn(internal_group),
        )
        self.wait()
        self.play(
            diag_entries.animate.set_color(WHITE),
            off_diag_entries.animate.set_color(WHITE),
        )
        self.wait()

        # The result so far
        matrix_F_name = TexText("度量矩阵 $\\mathbf{F}$")
        matrix_F_name.set_color(color_special)
        matrix_F_name.move_to(LEFT_SIDE / 2)
        matrix_F_name.match_y(matrix_G_name)
        matrix_F = TexMatrix(string_table_F, element_alignment_corner=ORIGIN)
        matrix_F.move_to(LEFT_SIDE / 2)
        matrix_F.match_height(matrix_G)
        matrix_F.match_y(matrix_G)
        title_line = Underline(relations)
        title_line.set_width(FRAME_WIDTH)
        title_line.next_to(relations, DOWN).match_x(ORIGIN)
        self.play(
            FadeInFromPoint(title_line, title_line.get_center()),
            FadeIn(matrix_F),
            FadeIn(matrix_F_name),
            FadeOut(internal_group),
            relations.animate.set_opacity(1),
            meanings.animate.set_opacity(1),
        )
        self.wait()

        # The order of the matrix is very high
        matF_grouped_entries = VGroup(*[VGroup() for _ in range(6)])
        matG_grouped_entries = VGroup(*[VGroup() for _ in range(6)])
        for row in range(6):
            for col in range(6):
                ind = max(row, col)
                matF_entry = matrix_F.get_row(row)[col]
                matF_grouped_entries[ind].add(matF_entry)
                matG_entry = matrix_G.get_row(row)[col]
                matG_grouped_entries[ind].add(matG_entry)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    *[
                        AnimationGroup(
                            *[
                                Indicate(entry, scale_factor=1.5, color=RED)
                                for entry in entries
                            ]
                        )
                        for entries in grouped_entries
                    ],
                    lag_ratio=0.2,
                )
                for grouped_entries in (matF_grouped_entries, matG_grouped_entries)
            ),
            run_time=3,
        )
        self.wait()

        # Time for some special technique
        matrix_F_group = VGroup(matrix_F_name, matrix_F)
        matrix_G_group = VGroup(matrix_G_name, matrix_G)
        self.add(matrix_F_group, matrix_G_group)
        for group in (matrix_F_group, matrix_G_group):
            group.generate_target()
            group.target.to_edge(UP)
        matrix_G_group.target.set_opacity(0)
        self.play(
            MoveToTarget(matrix_F_group),
            MoveToTarget(matrix_G_group),
            FadeOut(
                VGroup(
                    title_line, position_relation, position_meaning,
                    property_relation, property_meaning,
                ),
                shift=2 * UP,
            ),
        )
        self.wait()

        # F is block diagonal, we only need to calculate the inverse of a 2x2 matrix
        off_diag_blocks = VGroup(
            matrix_F.get_row(0)[2:], matrix_F.get_row(1)[2:],
            matrix_F.get_column(0)[2:], matrix_F.get_column(1)[2:],
        )
        matF_top_rect = matrix_top_rect
        matF_bottom_rect = matrix_bottom_rect
        matrix_top_rect.surround(
            VGroup(
                matrix_F.get_row(0)[:2],
                matrix_F.get_row(1)[:2],
            )
        )
        matrix_bottom_rect.surround(
            VGroup(
                matrix_F.get_row(2)[2:],
                matrix_F.get_row(5)[2:],
            )
        )
        self.play(
            off_diag_blocks.animate.set_opacity(0.3),
            FadeIn(matrix_top_rect),
            FadeIn(matrix_bottom_rect),
        )
        self.wait()

        string_table_Finv = [
            [0, "-2", 0, 0, "\\cdots", 0],
            ["-2", 0, 0, 0, "\\cdots", 0],
            [0, 0, 1, 0, "\\cdots", 0],
            [0, 0, 0, 1, "\\cdots", 0],
            ["\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\ddots", "\\vdots"],
            [0, 0, 0, 0, "\\cdots", 1],
        ]
        matrix_Finv_name = Tex("\\mathbf{F}^{-1}")
        matrix_Finv_name.set_color(color_special)
        matrix_Finv_name.move_to(RIGHT_SIDE / 2).to_edge(UP)
        matrix_Finv = TexMatrix(string_table_Finv, element_alignment_corner=ORIGIN)
        matrix_Finv.match_height(matrix_F)
        matrix_Finv.next_to(matrix_Finv_name, DOWN)
        for mob, ref in zip(matrix_Finv.get_entries(), matrix_F.get_entries()):
            mob.match_style(ref)
        matFinv_top_rect = matrix_top_rect.copy()
        matFinv_bottom_rect = matrix_bottom_rect.copy()
        matFinv_top_rect.surround(
            VGroup(
                matrix_Finv.get_row(0)[:2],
                matrix_Finv.get_row(1)[:2],
            )
        )
        matFinv_bottom_rect.surround(
            VGroup(
                matrix_Finv.get_row(2)[2:],
                matrix_Finv.get_row(5)[2:],
            )
        )
        self.play(
            TransformFromCopy(matrix_F.get_brackets(), matrix_Finv.get_brackets()),
            *[
                ReplacementTransform(source.copy().set_opacity(0), target)
                for source, target in zip(matrix_F.get_entries(), matrix_Finv.get_entries())
            ],
            ReplacementTransform(matF_top_rect.copy().set_opacity(0), matFinv_top_rect),
            ReplacementTransform(matF_bottom_rect.copy().set_opacity(0), matFinv_bottom_rect),
            TransformFromCopy(matrix_F_name[-1], matrix_Finv_name[0]),
            ReplacementTransform(VectorizedPoint(matrix_F_name.get_corner(UR)), matrix_Finv_name[1:]),
            run_time=2,
        )
        self.wait()
        self.play(
            *[
                FlashAround(mob)
                for mob in (matF_top_rect, matFinv_top_rect)
            ],
        )
        self.wait()

        # G is structured in a peculiar way, and can be written as 2I-J
        curr_mobs = VGroup(*self.mobjects)
        matrix_G.scale(0.8, about_edge=UP)

        matIJ_t2c = {
            "2": TEAL,
            "\\ddots": TEAL,
            "1": PURPLE,
            "0": GREY_D,
            "\\cdots": GREY_D,
            "\\vdots": GREY_D,
        }
        string_table_2I = [
            [2, 0, 0, 0, "\\cdots", 0],
            [0, 2, 0, 0, "\\cdots", 0],
            [0, 0, 2, 0, "\\cdots", 0],
            [0, 0, 0, 2, "\\cdots", 0],
            ["\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\ddots", "\\vdots"],
            [0, 0, 0, 0, "\\cdots", 2],
        ]
        string_table_J = [
            [1, 1, 1, 1, "\\cdots", 1],
            [1, 1, 1, 1, "\\cdots", 1],
            [1, 1, 1, 1, "\\cdots", 1],
            [1, 1, 1, 1, "\\cdots", 1],
            ["\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\ddots", "\\vdots"],
            [1, 1, 1, 1, "\\cdots", 1],
        ]
        matrix_2I = TexMatrix(
            string_table_2I,
            tex_config={
                "t2c": matIJ_t2c,
                "use_labelled_svg": False,
            },
            element_alignment_corner=ORIGIN,
        )
        matrix_J = TexMatrix(
            string_table_J,
            tex_config={
                "t2c": matIJ_t2c,
                "use_labelled_svg": False,
            },
            element_alignment_corner=ORIGIN,
        )
        minus, equal = Tex("-"), Tex("=")
        matrix_J.get_entries().set_color(matIJ_t2c["1"])
        for matrix in (matrix_2I, matrix_J):
            matrix.match_height(matrix_G)
        matrix_G.generate_target()
        decompose_group = VGroup(matrix_2I, minus, matrix_J, equal, matrix_G.target)
        decompose_group.arrange(RIGHT).match_y(matrix_G)
        matrix_G.move_to(matrix_G.target)
        matrix_G_name.match_x(matrix_G)
        matrix_2I_name = TexText("数量阵$2 \\mathbf{I}$")
        matrix_2I_name.next_to(matrix_2I, UP)
        matrix_2I_name.set_color(matIJ_t2c["2"])
        matrix_J_name = TexText("全一矩阵$\\mathbf{J}$")
        matrix_J_name.next_to(matrix_J, UP)
        matrix_J_name.set_color(matIJ_t2c["1"])
        for name in (matrix_2I_name, matrix_J_name):
            name.match_y(matrix_G_name)
        self.play(
            FadeOut(curr_mobs),
            matrix_G_group.animate.set_opacity(1),
        )
        self.add(matrix_G_group)
        self.wait()

        matG_diag_entries = VGroup()
        matG_off_diag_entries = VGroup()
        for row in range(6):
            for col in range(6):
                entry = matrix_G.get_row(row)[col]
                if row == col:
                    matG_diag_entries.add(entry)
                else:
                    matG_off_diag_entries.add(entry)
        color_diag = interpolate_color(WHITE, matIJ_t2c["2"], 0.8)
        color_off_diag = interpolate_color(WHITE, matIJ_t2c["1"], 0.8)
        self.play(
            matG_diag_entries.animate.set_color(color_diag),
            matG_off_diag_entries.animate.set_color(color_off_diag),
            AnimationGroup(
                *[
                    FadeIn(matrix, shift=UP)
                    for matrix in (matrix_2I, matrix_J)
                ],
                lag_ratio=0.02,
            ),
            AnimationGroup(
                *[
                    FadeIn(name, shift=DOWN)
                    for name in (matrix_2I_name, matrix_J_name)
                ],
                lag_ratio=0.02,
            ),
            AnimationGroup(
                *[
                    FadeInFromPoint(symbol, symbol.get_center())
                    for symbol in (minus, equal)
                ],
                lag_ratio=0.02,
            ),
        )
        self.wait()

        # {I, J} is closed under matrix multiplication
        comb_t2c = {
            "\\mathbf{G}": color_special,
            "\\mathbf{G}^{-1}": color_special,
            "\\mathbf{I}": matIJ_t2c["2"],
            "\\mathbf{J}": matIJ_t2c["1"],
            "\\mathbf{1}": matIJ_t2c["1"],
        }
        decompose_group = VGroup(matrix_2I, minus, matrix_J, equal, matrix_G)
        close_property = VGroup(
            Tex(text, t2c=comb_t2c, use_labelled_svg=False)
            for text in [
                "\\text{乘积封闭：}",
                """
                \\mathbf{I} \\mathbf{I} & = \\mathbf{I} \\\\
                \\mathbf{J} \\mathbf{I} & = \\mathbf{J}
                """,
                """
                \\mathbf{I} \\mathbf{J} & = \\mathbf{J} \\\\
                \\mathbf{J} \\mathbf{J} & = (n+2) \\mathbf{J}
                """,
            ]
        )
        close_property.arrange(RIGHT, buff=0.8)
        close_property[-1].match_y(close_property[-2], UP)
        close_property.next_to(decompose_group[:-2], DOWN, buff=0.8)
        close_property_rect = SurroundingRectangle(close_property, buff=0.2)
        close_property_rect.set_color(color_note)
        self.play(
            AnimationGroup(
                *[Write(mob) for mob in close_property],
                lag_ratio=0.1,
                run_time=2,
            ),
            ShowCreation(close_property_rect),
        )
        self.wait()

        # If G is a linear combination of I and J, then G^(-1) also is
        combs = VGroup(
            Tex(text, t2c=comb_t2c, use_labelled_svg=False)
            for text in [
                "\\mathbf{G} = 2 \\mathbf{I} - \\mathbf{J}",
                "\\mathbf{G}^{-1} = u \\mathbf{I} + v \\mathbf{J}",
                "\\mathbf{G}^{-1} = \\dfrac{1}{2} \\mathbf{I} - \\dfrac{1}{2n} \\mathbf{J}",
                "\\mathbf{G}^{-1} = \\dfrac{1}{2} \\mathbf{I} - \\dfrac{1}{2n} \\mathbf{1} \\mathbf{1}^{\\mathrm{T}}",
            ]
        )
        comb_G, comb_Ginv_unknown, comb_Ginv_mid, comb_Ginv_final = combs
        ref_equal = combs[0].get_part_by_tex("=")
        for eq in combs[1:]:
            eq.next_to(combs[0], DOWN)
            target_equal = eq.get_part_by_tex("=")
            eq.shift((ref_equal.get_x() - target_equal.get_x()) * RIGHT)
        combs.next_to(matrix_G, DOWN)
        combs.match_y(close_property, UP)
        self.play(
            TransformFromCopy(matrix_G_name[-1], comb_G[0]),
            TransformFromCopy(equal[0], comb_G[1]),
            TransformFromCopy(matrix_2I_name[-2:], comb_G[2:4]),
            TransformFromCopy(minus[0], comb_G[4]),
            TransformFromCopy(matrix_J_name[-1], comb_G[5]),
            run_time=2,
        )
        self.wait()
        self.play(
            TransformFromCopy(comb_G[0], comb_Ginv_unknown[0]),
            ReplacementTransform(VectorizedPoint(comb_G[0].get_corner(UR)), comb_Ginv_unknown[1:3]),
            TransformFromCopy(comb_G[1], comb_Ginv_unknown[3]),
            FadeTransform(comb_G[2].copy(), comb_Ginv_unknown[4]),
            TransformFromCopy(comb_G[3:5], comb_Ginv_unknown[5:7]),
            FadeInFromPoint(comb_Ginv_unknown[7], comb_G[5].get_left()),
            TransformFromCopy(comb_G[5], comb_Ginv_unknown[8]),
        )
        self.wait()

        # Solve u and v using inverse definition
        solution = Tex(
            """
            \\mathbf{G} \\mathbf{G}^{-1} & = \\mathbf{I} \\\\
            (2 \\mathbf{I} - \\mathbf{J}) (u \\mathbf{I} + v \\mathbf{J}) & = \\mathbf{I} \\\\
            2u \\mathbf{I} + 2v \\mathbf{J} - u \\mathbf{J} - (n+2)v \\mathbf{J} & = \\mathbf{I} \\\\
            (2u - 1) \\mathbf{I} - (u + nv) \\mathbf{J} & = \\mathbf{O} \\\\[0.5em]
            \\text{解得：} u = \\dfrac{1}{2} ,\\, v = -\\dfrac{1}{2n}
            """,
            t2c=comb_t2c,
            use_labelled_svg=False,
        )
        solution.set_height(4)
        solution.move_to(decompose_group[:-1])
        solution.match_y(decompose_group, DOWN)
        self.play(
            *[
                FadeOut(mob)
                for mob in (decompose_group[:-1], matrix_2I_name, matrix_J_name)
            ],
            FadeIn(solution),
        )
        self.wait()

        coeff_u = solution[-11:-8]
        coeff_v = solution[-5:]
        self.play(
            ReplacementTransform(comb_Ginv_unknown[:4], comb_Ginv_mid[:4]),
            FadeOutToPoint(comb_Ginv_unknown[4], comb_Ginv_mid[4:7].get_center()),
            TransformFromCopy(coeff_u, comb_Ginv_mid[4:7]),
            ReplacementTransform(comb_Ginv_unknown[5], comb_Ginv_mid[7]),
            FadeOutToPoint(comb_Ginv_unknown[6:8], comb_Ginv_mid[-6:-1].get_center()),
            TransformFromCopy(coeff_v, comb_Ginv_mid[-6:-1]),
            ReplacementTransform(comb_Ginv_unknown[-1], comb_Ginv_mid[-1]),
            run_time=2,
        )
        self.wait()

        # Back to all-one matrix J
        matrix_J.scale(1.5, about_edge=UP)
        matrix_J_group = VGroup(matrix_J_name, matrix_J)
        matrix_J_group.to_edge(LEFT)
        self.play(
            *[
                FadeOut(mob)
                for mob in (
                    solution, close_property, close_property_rect,
                    matrix_G_name, matrix_G, combs[0],
                )
            ],
            FadeIn(matrix_J_group)
        )
        self.wait()
        vec_equal = Tex("=")
        vec_ones = MobjectMatrix(
            [Tex("1" if ind != 4 else "\\vdots") for ind in range(6)],
            n_cols=1,
        )
        vec_ones.match_height(matrix_J)
        vec_ones_trans = MobjectMatrix(
            [Tex("1" if ind != 4 else "\\cdots") for ind in range(6)],
            n_rows=1,
            height=0.7,
        )
        scale_factor = vec_ones.get_entries()[0].get_height() / vec_ones_trans.get_entries()[0].get_height()
        vec_ones_trans.scale(scale_factor)
        vec_group = VGroup(vec_equal, vec_ones, vec_ones_trans)
        vec_group.arrange(RIGHT)
        vec_group.next_to(matrix_J, RIGHT)
        for vec in (vec_ones, vec_ones_trans):
            vec.get_entries().match_color(matrix_J_name)
        vec_ones_names = VGroup(
            TexText("全一列向量$\\mathbf{1}$"),
            TexText("全一行向量$\\mathbf{1}^{\\mathrm{T}}$"),
        )
        vec_ones_names.match_color(matrix_J_name)
        vec_ones_names.arrange(RIGHT, buff=1)
        vec_ones_names.next_to(vec_group[1:], UP)
        vec_ones_names.match_y(matrix_J_name)

        # For simplicity, we rewrite J as (1 1^T)
        Ginv_rect = SurroundingRectangle(comb_Ginv_final, buff=0.2)
        Ginv_rect.set_color(color_special)
        self.play(*[FadeIn(mob) for mob in (vec_ones_names[0], vec_ones)])
        self.wait()
        self.play(*[FadeIn(mob) for mob in (vec_ones_names[1], vec_ones_trans, vec_equal)])
        self.wait()
        self.play(
            ReplacementTransform(comb_Ginv_mid[:-1], comb_Ginv_final[:-3]),
            FadeOutToPoint(comb_Ginv_mid[-1], comb_Ginv_final[-3:].get_center()),
            TransformFromCopy(vec_ones_names[0][-1], comb_Ginv_final[-3]),
            TransformFromCopy(vec_ones_names[1][-2:], comb_Ginv_final[-2:]),
            run_time=2,
        )
        self.wait()
        self.play(ShowCreation(Ginv_rect))
        self.wait()


class ProvingSoddyGossetTheorem(Scene):
    def construct(self):
        # Setup brand-new formula
        color_c1, color_c2, color_c3, color_c4 = CIRCLE_COLORS
        color_special = YELLOW
        color_note = GREY_B
        color_id = TEAL
        color_one = PURPLE

        property_relation = Tex("\\mathbf{C} \\mathbf{G}^{-1} \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}")
        property_relation.scale(2)
        property_relation.set_color(color_special)
        property_group = VGroup(
            property_relation[start:end]
            for start, end in [(0, 1), (1, 4), (4, 6), (6, 7), (7, 10)]
        )
        self.add(property_group)
        self.wait()

        # Expand configuration matrix and F^(-1)
        string_table_Finv = [
            [0, "-2", 0, 0, "\\cdots", 0],
            ["-2", 0, 0, 0, "\\cdots", 0],
            [0, 0, 1, 0, "\\cdots", 0],
            [0, 0, 0, 1, "\\cdots", 0],
            ["\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\ddots", "\\vdots"],
            [0, 0, 0, 0, "\\cdots", 1],
        ]
        matrix_Finv = TexMatrix(
            string_table_Finv,
            element_alignment_corner=ORIGIN,
        )
        string_table_C = np.array([
            ["k_{1}", "k_{2}", "k_{3}", "k_{4}", "\\cdots", "k_{n+2}"],
            ["t_{1}", "t_{2}", "t_{3}", "t_{4}", "\\cdots", "t_{n+2}"],
            ["k_{1} x_{1}", "k_{2} x_{2}", "k_{3} x_{3}", "k_{4} x_{4}", "\\cdots", "k_{n+2} x_{n+2}"],
            ["k_{1} y_{1}", "k_{2} y_{2}", "k_{3} y_{3}", "k_{4} y_{4}", "\\cdots", "k_{n+2} y_{n+2}"],
            ["\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\ddots", "\\vdots"],
            ["k_{1} w_{1}", "k_{2} w_{2}", "k_{3} w_{3}", "k_{4} w_{4}", "\\cdots", "k_{n+2} w_{n+2}"],
        ])
        string_table_Ctrans = np.array(string_table_C.T)
        mask_cdots = (string_table_Ctrans == "\\cdots")
        mask_vdots = (string_table_Ctrans == "\\vdots")
        string_table_Ctrans[mask_cdots] = "\\vdots"
        string_table_Ctrans[mask_vdots] = "\\cdots"
        matrix_C = TexMatrix(
            string_table_C,
            h_buff=0,
            element_alignment_corner=ORIGIN,
        )
        matrix_Ctrans = TexMatrix(
            string_table_Ctrans,
            h_buff=0,
            element_alignment_corner=ORIGIN,
        )
        Ginv_text = Tex("\\mathbf{G}^{-1}")
        equal_text = Tex("=")
        matrix_group = VGroup(matrix_C, Ginv_text, matrix_Ctrans, equal_text, matrix_Finv)
        matrix_group.arrange(RIGHT)
        matrix_group.set_width(FRAME_WIDTH - 1)
        self.play(
            *[
                FadeTransform(source, target)
                for source, target in zip(property_group[::2], matrix_group[::2])
            ],
            *[
                ReplacementTransform(source, target)
                for source, target in zip(property_group[1::2], matrix_group[1::2])
            ],
            run_time=2,
        )
        self.wait()

        # Focus on important parts - curvature... again
        matC_imp = VGroup(*[
            matrix_C.get_entries()[i]
            for i in range(6)
        ])
        matC_nil = VGroup(*[
            matrix_C.get_entries()[i]
            for i in range(6, 36)
        ])
        matC_rect = SurroundingRectangle(matC_imp)
        matCtrans_imp = VGroup(*[
            matrix_Ctrans.get_entries()[i]
            for i in range(36)
            if i % 6 == 0
        ])
        matCtrans_nil = VGroup(*[
            matrix_Ctrans.get_entries()[i]
            for i in range(36)
            if i % 6 != 0
        ])
        matCtrans_rect = SurroundingRectangle(matCtrans_imp)
        matFinv_imp = matrix_Finv.get_entries()[0]
        matFinv_nil = matrix_Finv.get_entries()[1:]
        matFinv_rowrect = SurroundingRectangle(matrix_Finv.get_row(0))
        matFinv_colrect = SurroundingRectangle(matrix_Finv.get_column(0))
        temp = SurroundingRectangle(matrix_Finv.get_entries())
        matFinv_rowrect.match_width(temp, stretch=True)
        matFinv_rowrect.match_x(temp, LEFT).match_y(temp, UP)
        matFinv_colrect.match_height(temp, stretch=True)
        matFinv_colrect.match_x(temp, LEFT).match_y(temp, UP)
        self.remove(matrix_C, matrix_Ctrans)
        self.add(
            matC_imp, matC_nil, matrix_C.get_brackets(),
            matCtrans_imp, matCtrans_nil, matrix_Ctrans.get_brackets(),
        )
        rects = VGroup(matC_rect, matCtrans_rect, matFinv_rowrect, matFinv_colrect)
        matC_note = TexText("第1行 \\\\ 所有的曲率")
        matCtrans_note = TexText("第1列 \\\\ 所有的曲率")
        matFinv_note = TexText("第1行第1列 \\\\ 矩阵元 0")
        notes = VGroup(matC_note, matCtrans_note, matFinv_note)
        note_refs = VGroup(matrix_C, matrix_Ctrans, matrix_Finv)
        for note, ref in zip(notes, note_refs):
            note.scale(0.7)
            note.set_color(color_special)
            note.next_to(ref, UP, buff=0.5)
        self.play(
            *[
                mob.animate.set_color(color_special)
                for mob in (matC_imp, matCtrans_imp, matFinv_imp)
            ],
            *[
                mob.animate.set_color(GREY).set_opacity(0.2)
                for mob in (matC_nil, matCtrans_nil, matFinv_nil)
            ],
            *[
                ShowCreation(rect)
                for rect in rects
            ],
            *[
                FadeInFromPoint(note, note.get_bottom())
                for note in notes
            ],
            run_time=2,
        )
        self.wait()

        # Clean out redundant terms... again
        string_table_k = string_table_Ctrans.T[0:1].T.copy()
        string_table_ktrans = string_table_C[0:1].copy()
        vec_k_trans = TexMatrix(string_table_ktrans)
        mat_Q = Tex("\\mathbf{G}^{-1}")
        vec_k = TexMatrix(string_table_k)
        equal_zero = Tex("= 0")
        soddy_gosset_group = VGroup(vec_k_trans, mat_Q, vec_k, equal_zero)
        soddy_gosset_group.arrange(RIGHT)
        soddy_gosset_group.set_width(FRAME_WIDTH * 0.8)
        self.play(
            AnimationGroup(
                FadeTransform(matrix_C.get_brackets()[0], vec_k_trans.get_brackets()[0]),
                FadeTransform(matrix_C.get_brackets()[1], vec_k_trans.get_brackets()[1]),
                ReplacementTransform(matC_imp, vec_k_trans.get_entries()),
                ReplacementTransform(Ginv_text, mat_Q),
                ReplacementTransform(matrix_Ctrans.get_brackets()[0], vec_k.get_brackets()[0]),
                ReplacementTransform(matrix_Ctrans.get_brackets()[1], vec_k.get_brackets()[1]),
                ReplacementTransform(matCtrans_imp, vec_k.get_entries()),
                ReplacementTransform(equal_text[0], equal_zero[0]),
                ReplacementTransform(matFinv_imp[0], equal_zero[1]),
                run_time=2,
            ),
            AnimationGroup(
                *[FadeOut(nil) for nil in (matC_nil, matCtrans_nil, matFinv_nil)],
                FadeOut(matrix_Finv.get_brackets()),
                FadeOut(rects),
                FadeOut(notes),
                run_time=0.5,
            ),
        )
        self.wait()

        # Introduce curvature vector k
        vector_k_note = TexText("曲率向量$\\mathbf{k}$")
        vector_k_note.set_color(color_special)
        vector_k_note.next_to(vec_k, UP)
        self.play(
            FadeIn(vector_k_note),
            vec_k.animate.set_color(color_special),
        )
        self.wait()

        # Simplified quadratic form
        quad_form = Tex("\\mathbf{k}^{\\mathrm{T}} \\mathbf{G}^{-1} \\mathbf{k} = 0")
        quad_form.set_color(color_special)
        quad_form.scale(1.5)
        quad_form.move_to(LEFT_SIDE * 0.35)
        vector_k_note.generate_target()
        vec_k.generate_target()
        targets = VGroup(vector_k_note.target, vec_k.target)
        targets.move_to(RIGHT_SIDE * 0.4).to_edge(UP)
        targets.set_color(color_note)
        self.play(
            FadeOutToPoint(vec_k_trans, quad_form[:2].get_center()),
            FadeOutToPoint(vec_k.copy(), quad_form[5].get_center()),
            *[MoveToTarget(mob) for mob in (vector_k_note, vec_k)],
            ReplacementTransform(vector_k_note[-1].copy(), quad_form[0]),
            ReplacementTransform(VectorizedPoint(vector_k_note.get_corner(UR)), quad_form[1]),
            ReplacementTransform(mat_Q, quad_form[2:5]),
            ReplacementTransform(vector_k_note[-1].copy(), quad_form[5]),
            ReplacementTransform(equal_zero, quad_form[6:]),
            run_time=2,
        )
        self.wait()
        vector_k_group = VGroup(vector_k_note, vec_k)

        # We have the expression for G^(-1)
        soddy_t2c = {
            "\\mathbf{G}": color_special,
            "\\mathbf{G}^{-1}": color_special,
            "\\mathbf{I}": color_id,
            "\\mathbf{1}": color_one,
        }
        Ginv_text = Tex(
            "\\mathbf{G}^{-1} = \\dfrac{1}{2} \\mathbf{I} - \\dfrac{1}{2n} \\mathbf{1} \\mathbf{1}^{\\mathrm{T}}",
            t2c=soddy_t2c,
            use_labelled_svg=False,
        )
        Ginv_arrow = Arrow(ORIGIN, 1.5 * DOWN)
        quad_form_Ginv = quad_form[2:5]
        Ginv_arrow.next_to(quad_form_Ginv, UP)
        Ginv_arrow.set_color(color_special)
        Ginv_text.next_to(Ginv_arrow, UP)
        Ginv_rect = SurroundingRectangle(Ginv_text, buff=0.25)
        quad_form.generate_target()
        quad_form.target.set_color(WHITE)
        quad_form.target[2:5].set_color(color_special)
        vector_ones = TexMatrix(
            [["1"], ["1"], ["1"], ["1"], ["\\vdots"], ["1"]],
            v_buff=0.44,
            bracket_h_buff=0.5,
        )
        vector_ones_note = TexText("全一向量$\\mathbf{1}$")
        vector_ones_group = VGroup(vector_ones_note, vector_ones)
        vector_ones_group.set_color(color_note)
        vector_ones_group.arrange(DOWN)
        vector_ones_group.to_edge(UP).to_edge(RIGHT, buff=0.15)
        self.play(
            MoveToTarget(quad_form),
            *[
                FadeIn(mob)
                for mob in (Ginv_arrow, Ginv_text, Ginv_rect, vector_ones_group)
            ]
        )
        self.wait()

        # A bunch of simplification
        simp_forms = VGroup(
            Tex(
                text,
                t2c=soddy_t2c,
                use_labelled_svg=False,
            )
            for text in (
                """
                \\mathbf{k}^{\\mathrm{T}}
                \\left( \\dfrac{1}{2} \\mathbf{I} - \\dfrac{1}{2n} \\mathbf{1} \\mathbf{1}^{\\mathrm{T}} \\right)
                \\mathbf{k} = 0
                """,
                """
                \\mathbf{k}^{\\mathrm{T}}
                \\left( n \\mathbf{I} - \\mathbf{1} \\mathbf{1}^{\\mathrm{T}} \\right)
                \\mathbf{k} = 0
                """,
                """
                n \\mathbf{k}^{\\mathrm{T}} \\mathbf{k} =
                \\mathbf{k}^{\\mathrm{T}} \\mathbf{1} \\mathbf{1}^{\\mathrm{T}} \\mathbf{k}
                """,
                """
                n \\left( \\mathbf{k}^{\\mathrm{T}} \\mathbf{k} \\right) =
                \\left( \\mathbf{k}^{\\mathrm{T}} \\mathbf{1} \\right)
                \\left( \\mathbf{1}^{\\mathrm{T}} \\mathbf{k} \\right)
                """,
                """
                n \\left( \\mathbf{k} \\cdot \\mathbf{k} \\right) =
                \\left( \\mathbf{k} \\cdot \\mathbf{1} \\right)
                \\left( \\mathbf{1} \\cdot \\mathbf{k} \\right)
                """,
            )
        )
        for form in simp_forms:
            form.save_state()
            form.scale(1.3)
            form.move_to(quad_form)

        simp_form_1, simp_form_2, simp_form_3, simp_form_4, simp_form_5 = simp_forms
        # 1. Replace G^(-1)
        self.play(
            AnimationGroup(
                FadeOut(Ginv_rect),
                FadeOut(Ginv_arrow),
                FadeOut(Ginv_text[:4]),
                run_time=0.5,
            ),
            AnimationGroup(
                ReplacementTransform(quad_form[:2], simp_form_1[:2]),
                ReplacementTransform(VectorizedPoint(quad_form[2:5].get_left()), simp_form_1[2]),
                FadeOutToPoint(quad_form[2:5], quad_form[2:5].get_center()),
                ReplacementTransform(Ginv_text[4:], simp_form_1[3:15]),
                ReplacementTransform(VectorizedPoint(quad_form[2:5].get_right()), simp_form_1[15]),
                ReplacementTransform(quad_form[5:], simp_form_1[16:]),
                run_time=1,
            ),
        )
        self.remove(quad_form)
        self.add(simp_form_1)
        self.wait()

        # 2. Multiply 2n to remove fractions
        self.play(
            ReplacementTransform(simp_form_1[:3], simp_form_2[:3]),
            FadeTransform(simp_form_1[3:6], simp_form_2[3]),
            ReplacementTransform(simp_form_1[6:8], simp_form_2[4:6]),
            FadeOutToPoint(simp_form_1[8:12], simp_form_2[6].get_left()),
            ReplacementTransform(simp_form_1[12:], simp_form_2[6:]),
            run_time=1,
        )
        self.remove(simp_form_1)
        self.add(simp_form_2)
        self.wait()

        # 3. Rearrange
        path_arc = PI
        self.play(
            ReplacementTransform(simp_form_2[3], simp_form_3[0], path_arc=path_arc),
            ReplacementTransform(simp_form_2[:2], simp_form_3[1:3], path_arc=path_arc),
            ReplacementTransform(simp_form_2[4], VectorizedPoint(simp_form_3[3].get_left()), path_arc=path_arc),
            ReplacementTransform(simp_form_2[10], simp_form_3[3], path_arc=path_arc),
            ReplacementTransform(simp_form_2[11], simp_form_3[4]),
            ReplacementTransform(simp_form_2[:2].copy(), simp_form_3[5:7], path_arc=path_arc),
            ReplacementTransform(simp_form_2[6:9], simp_form_3[7:10], path_arc=path_arc),
            ReplacementTransform(simp_form_2[10].copy(), simp_form_3[10], path_arc=path_arc),
            *[FadeOut(simp_form_2[ind]) for ind in (2, 5, 9, 12)],
            run_time=2,
        )
        self.remove(simp_form_2)
        self.add(simp_form_3)
        self.wait()

        # 4. Group
        for form in (simp_form_4, simp_form_5):
            shift_vec = simp_form_3[0].get_y() - form[0].get_y()
            form.shift(shift_vec)
            for ind in (2, 4, 8, 15):
                form[ind].set_color(color_special)
        self.play(
            vector_k_group.animate.set_color(color_special),
            vector_ones_group.animate.set_color(color_one),
            *[FadeIn(simp_form_4[ind], shift=UP) for ind in (1, 5, 7, 11, 12, 16)],
            *[
                ReplacementTransform(simp_form_3[ind_3], simp_form_4[ind_4])
                for ind_3, ind_4 in zip(
                    range(len(simp_form_3)),
                    [0, 2, 3, 4, 6, 8, 9, 10, 13, 14, 15],
                )
            ],
        )
        self.remove(simp_form_3)
        self.add(simp_form_4)
        self.wait()

        # 5. Rewrite as dot product
        self.play(ReplacementTransform(simp_form_4, simp_form_5))
        self.wait()

        # Expand all dot products
        soddy_t2c["\\mathbf{k}"] = color_special
        k_dot_k = Tex(
            """
            \\mathbf{k} \\cdot \\mathbf{k} =
            k_{1}^{2} + k_{2}^{2} + \\cdots + k_{n+2}^{2}
            """,
            t2c=soddy_t2c,
            use_labelled_svg=False,
        )
        k_dot_one = Tex(
            """
            \\mathbf{k} \\cdot \\mathbf{1} = \\mathbf{1} \\cdot \\mathbf{k} =
            k_{1} + k_{2} + \\cdots + k_{n+2}
            """,
            t2c=soddy_t2c,
            use_labelled_svg=False,
        )
        dot_products = VGroup(k_dot_k, k_dot_one)
        dot_products.arrange(DOWN)
        dot_products.match_x(simp_form_5)
        dot_products.shift(0.5 * DOWN)
        self.play(
            simp_form_5.animate.shift(TOP / 2),
            FadeIn(k_dot_k, shift=0.5 * UP),
        )
        self.wait()
        self.play(FadeIn(k_dot_one, shift=0.5 * UP))
        self.wait()

        # Calculate LHS and RHS
        lhs = Tex(
            """
            n \\left( \\mathbf{k} \\cdot \\mathbf{k} \\right) =
            n \\left( k_{1}^{2} + k_{2}^{2} + \\cdots + k_{n+2}^{2} \\right)
            """,
            t2c=soddy_t2c,
            use_labelled_svg=False,
        )
        rhs = Tex(
            """
            \\left( \\mathbf{k} \\cdot \\mathbf{1} \\right) \\left( \\mathbf{1} \\cdot \\mathbf{k} \\right) =
            \\left( k_{1} + k_{2} + \\cdots + k_{n+2} \\right)^{2}
            """,
            t2c=soddy_t2c,
            use_labelled_svg=False,
        )
        lhs.move_to(k_dot_k)
        rhs.move_to(k_dot_one)

        self.play(
            ReplacementTransform(VectorizedPoint(k_dot_k[0].get_left()), lhs[:2]),
            ReplacementTransform(k_dot_k[:3], lhs[2:5]),
            ReplacementTransform(VectorizedPoint(k_dot_k[2].get_right()), lhs[5]),
            ReplacementTransform(k_dot_k[3], lhs[6]),
            ReplacementTransform(VectorizedPoint(k_dot_k[4].get_left()), lhs[7:9]),
            ReplacementTransform(k_dot_k[4:], lhs[9:26]),
            ReplacementTransform(VectorizedPoint(k_dot_k[16:].get_right()), lhs[26]),
        )
        self.wait()
        self.play(
            ReplacementTransform(VectorizedPoint(k_dot_one[0].get_left()), rhs[0]),
            ReplacementTransform(k_dot_one[:3], rhs[1:4]),
            ReplacementTransform(k_dot_one[3:4], rhs[4:6]),
            ReplacementTransform(k_dot_one[4:7], rhs[6:9]),
            ReplacementTransform(VectorizedPoint(k_dot_one[6].get_right()), rhs[9]),
            ReplacementTransform(k_dot_one[7], rhs[10]),
            ReplacementTransform(VectorizedPoint(k_dot_one[8].get_left()), rhs[11]),
            ReplacementTransform(k_dot_one[8:], rhs[12:26]),
            ReplacementTransform(VectorizedPoint(k_dot_one[18:].get_right()), rhs[26:]),
        )
        self.wait()

        # And there's Soddy-Gosset theorem
        temp = VGroup(lhs, rhs)
        soddy_gosset_theorem = VGroup(
            Tex(text, t2c=CURVATURE_T2C, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + \\cdots + k_{n+2} \\right)^{2}",
                "= n \\left( k_{1}^{2} + k_{2}^{2} + \\cdots + k_{n+2}^{2} \\right)"
            ]
        )
        soddy_gosset_theorem.arrange(DOWN, aligned_edge=RIGHT)
        soddy_gosset_theorem[1][-5].set_color(WHITE)  # color fixing
        soddy_gosset_theorem[1][-2].set_color(SPHERE_COLORS[-1])  # color fixing
        soddy_gosset_theorem.scale(1.2)
        self.play(
            *[FadeOut(mob) for mob in (simp_form_5, vector_k_group, vector_ones_group)],
            FadeOut(lhs[:6], shift=LEFT),
            ReplacementTransform(lhs[6:], soddy_gosset_theorem[1]),
            FadeOut(rhs[:11], shift=LEFT),
            ReplacementTransform(rhs[11:], soddy_gosset_theorem[0]),
        )
        self.wait()
        soddy_gosset_title = TexText("索迪-戈赛特定理")
        soddy_gosset_title.set_color(color_special)
        soddy_gosset_title.to_edge(UP)
        soddy_gosset_note = TexText("$n$维空间中$(n+2)$个互切超球的曲率关系")
        soddy_gosset_note.scale(0.8)
        soddy_gosset_note.set_color(YELLOW_B)
        soddy_gosset_note.next_to(soddy_gosset_title, DOWN)
        soddy_gosset_group = VGroup(soddy_gosset_title, soddy_gosset_note)
        soddy_gosset_group.set_color(color_special)
        self.play(FadeIn(soddy_gosset_group, shift=DOWN))
        self.wait()

        # n=3 -> What the poem said
        sg_3d_note = TexText("三维空间中的五球互切")
        sg_3d_note.set_color(color_special)
        sg_3d_note.move_to(LEFT_SIDE / 2).to_edge(UP)
        sg_3d_formula = VGroup(
            Tex(text, t2c=CURVATURE_T2C, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + k_{3} + k_{4} + k_{5} \\right)^{2}",
                "= 3 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} + k_{5}^{2} \\right)"
            ]
        )
        sg_3d_formula.arrange(DOWN, aligned_edge=RIGHT)
        sg_3d_formula.next_to(sg_3d_note, DOWN)
        self.play(
            FadeOut(soddy_gosset_title, shift=UP),
            FadeTransform(soddy_gosset_note, sg_3d_note),
            FadeTransform(soddy_gosset_theorem, sg_3d_formula),
        )
        self.wait()

        # n=2 -> Descartes' theorem
        sg_2d_note = TexText("二维空间中的四圆互切")
        sg_2d_note.set_color(color_special)
        sg_2d_note.move_to(RIGHT_SIDE / 2).to_edge(UP)
        sg_2d_formula = VGroup(
            Tex(text, t2c=CURVATURE_T2C, use_labelled_svg=False)
            for text in [
                "\\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2}",
                "= 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)"
            ]
        )
        sg_2d_formula.arrange(DOWN, aligned_edge=RIGHT)
        sg_2d_formula.next_to(sg_2d_note, DOWN)
        p1, p2, p3 = get_tangent_circle_centers(10, 15, 19)
        sg_2d_example = TangentFourCircles(p1, p2, p3, fourth_type="bigger")
        sg_2d_example.suspend_updating()
        sg_2d_example.set_height(3.5)
        sg_2d_example.next_to(sg_2d_formula, DOWN, buff=0.5)
        self.play(
            FadeTransform(sg_3d_note.copy(), sg_2d_note),
            FadeTransform(sg_3d_formula.copy(), sg_2d_formula),
            FadeIn(sg_2d_example),
        )
        self.wait()

        # Descartes' theorem comes naturally from vector k and 1
        common_backend = simp_form_5
        common_backend.match_height(common_backend.saved_state)
        common_backend.move_to(LEFT_SIDE / 2)
        common_backend.match_y(sg_2d_formula)
        sg_2d_arrow = Tex("\\Rightarrow")
        sg_2d_arrow.match_y(common_backend)
        self.play(
            FadeOut(sg_3d_note),
            FadeOut(sg_3d_formula),
            FadeIn(common_backend),
            FadeInFromPoint(sg_2d_arrow, sg_2d_arrow.get_left()),
        )
        self.wait()

        # Soddy-Gosset theorem comes from the same framework, which explains the similarity
        target_group = VGroup(sg_2d_note, sg_2d_formula, sg_2d_arrow, common_backend)
        for mob in target_group:
            mob.generate_target()
        common_backend.target.center().to_edge(UP)
        sg_2d_arrow.target.rotate(-PI / 4).next_to(common_backend.target, DR)
        sg_2d_note.target.move_to(RIGHT_SIDE / 2)
        sg_2d_note.target.next_to(sg_2d_arrow.target, DOWN, coor_mask=[0, 1, 0], buff=0.5)
        sg_2d_formula.target.next_to(sg_2d_note.target, DOWN)

        sg_nd_arrow = sg_2d_arrow.target.copy()
        sg_nd_arrow.flip(about_point=ORIGIN)
        sg_nd_note = TexText("$n$维空间中的$(n+2)$球互切")
        sg_nd_note.set_color(color_special)
        sg_nd_note.move_to(sg_3d_note).match_y(sg_2d_note.target)
        sg_nd_formula = soddy_gosset_theorem
        sg_nd_formula.match_height(sg_2d_formula.target)
        sg_nd_formula.move_to(sg_3d_formula).match_y(sg_2d_formula.target)

        self.play(
            FadeOut(sg_2d_example, shift=DOWN),
            AnimationGroup(
                *[MoveToTarget(mob) for mob in target_group],
                *[
                    FadeInFromPoint(mob, common_backend.get_corner(DL))
                    for mob in (sg_nd_arrow, sg_nd_note, sg_nd_formula)
                ],
                run_time=2,
            ),
        )
        self.wait()
        similar_results = VGroup(
            sg_nd_arrow, sg_nd_note, sg_nd_formula,
            sg_2d_arrow, sg_2d_note, sg_2d_formula,
        )

        vector_group = VGroup(
            Tex("\\mathbf{c} ="),
            MobjectMatrix(
                [
                    Tex(text)
                    for text in ["k", "t", "kx", "ky", "\\vdots"]
                ],
                n_cols=1,
            ),
        )
        vector_group.arrange(RIGHT)
        ip_group = VGroup(
            Tex(
                """
                \\langle \\mathbf{c}_{1}, \\mathbf{c}_{2} \\rangle
                = \\mathbf{c}_{1}^{\\mathrm{T}} \\mathbf{F} \\mathbf{c}_{2}
                """,
            ),
            Tex("\\text{外切} \\leftrightarrow -1"),
            Tex("\\text{内切} \\leftrightarrow 1"),
        )
        ip_group.arrange(DOWN)
        ip_group[0].shift(0.8 * UP)
        mat_group = VGroup(
            Tex("\\mathbf{C}^{\\mathrm{T}} \\mathbf{F} \\mathbf{C} = \\mathbf{G}"),
            Tex("\\mathbf{C} \\mathbf{G}^{-1} \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}"),
            Tex("\\mathbf{G}^{-1} = \\dfrac{1}{2} \\mathbf{I} - \\dfrac{1}{2n} \\mathbf{1} \\mathbf{1}^{\\mathrm{T}}"),
        )
        mat_group[:2].set_color(color_special)
        mat_group.arrange(DOWN)
        mat_group[-1].shift(0.8 * DOWN)
        framework = VGroup(vector_group, ip_group, mat_group)
        framework.arrange(RIGHT, buff=1.5)
        framework.set_height(2.8)
        framework.to_edge(UP)
        self.play(
            AnimationGroup(
                *[FadeIn(mob, shift=DOWN) for mob in framework],
                lag_ratio=0.1,
                run_time=2,
            ),
            AnimationGroup(
                similar_results.animate.shift(2.5 * DOWN),
                common_backend.animate.shift(2.5 * DOWN).set_opacity(0),
                run_time=1.5,
            ),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[Indicate(mob, scale_factor=1, color=MAROON_B) for mob in framework],
                lag_ratio=0.3,
                run_time=2,
            ),
        )
        self.wait()

        # The only difference: dimension n
        only_difference = mat_group[-1][-4]
        self.play(
            FocusOn(only_difference, color=color_special),
            Indicate(only_difference, scale_factor=2, color=color_special),
            run_time=2,
        )
        self.wait()


class TwoVariantsInTwoDimensionalCase(Scene):
    def construct(self):
        # Setup
        color_c1, color_c2, color_c3, color_c4 = CIRCLE_COLORS
        color_special = YELLOW
        color_note = GREY_B
        basic_t2c = {
            key: color
            for i, color in enumerate(CIRCLE_COLORS, start=1)
            for key in [
                f"\\mathbf{{c}}_{{{i}}}",
                f"k_{{{i}}}",
                f"t_{{{i}}}",
                f"x_{{{i}}}",
                f"y_{{{i}}}",
                f"z_{{{i}}}",
                f"{{{i}}}",
            ]
        }
        basic_t2c["\\mathbf{F}"] = color_special
        string_table_C = np.array([
            ["k_{1}", "k_{2}", "k_{3}", "k_{4}"],
            ["t_{1}", "t_{2}", "t_{3}", "t_{4}"],
            ["k_{1} x_{1}", "k_{2} x_{2}", "k_{3} x_{3}", "k_{4} x_{4}"],
            ["k_{1} y_{1}", "k_{2} y_{2}", "k_{3} y_{3}", "k_{4} y_{4}"],
        ])
        string_table_F = [
            [0, "-\\tfrac{1}{2}", 0, 0],
            ["-\\tfrac{1}{2}", 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        string_table_Finv = np.array([
            [0, -2, 0, 0],
            [-2, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=int)

        # Back to 2d case
        matrix_C = TexMatrix(
            string_table_C,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        matrix_Ginv = VGroup(
            Tex("\\mathbf{G}^{-1}"),
            Square(side_length=1.5),
        )
        matrix_Ginv.set_height(3)
        matrix_Ginv.set_color(color_special)
        matrix_Ctrans = TexMatrix(
            string_table_C.T,
            tex_config={
                "t2c": basic_t2c,
                "use_labelled_svg": False,
            },
        )
        equal_sign = Tex("=")
        matrix_Finv = TexMatrix(string_table_Finv)
        matrix_group = VGroup(matrix_C, matrix_Ginv, matrix_Ctrans, equal_sign, matrix_Finv)
        matrix_group.arrange(RIGHT)
        matrix_group.set_width(FRAME_WIDTH - 1)
        matrix_group.move_to(1.5 * UP)
        top_names = VGroup(
            Tex(text).set_color(color_special)
            for text in ["\\mathbf{C}", "\\mathbf{G}^{-1}", "\\mathbf{C}^{\\mathrm{T}}", "=", "\\mathbf{F}^{-1}"]
        )
        top_names.next_to(matrix_group, UP, buff=0.5)
        for name, ref in zip(top_names, matrix_group):
            name.match_x(ref)
        self.add(matrix_group, top_names)
        self.wait()

        # We only used partial information to derive Descartes' theorem
        for mob in matrix_group:
            mob.save_state()
            mob.generate_target()
        fade_group = VGroup(
            *matrix_C.target.get_entries()[4:],
            *[matrix_Ctrans.target.get_entries()[ind] for ind in range(16) if ind % 4 != 0],
            *matrix_Finv.target.get_entries()[1:],
        )
        fade_group.set_opacity(0.1).set_color(GREY)
        descartes_theorem = Tex(
            """
            \\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2}
            = 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)
            """,
            t2c=CURVATURE_T2C,
            use_labelled_svg=False,
        )
        descartes_theorem.next_to(matrix_group, DOWN, buff=1)
        self.play(
            *[MoveToTarget(mob) for mob in matrix_group],
            FadeIn(descartes_theorem, shift=DOWN),
        )
        self.wait()
        self.play(
            *[Restore(mob) for mob in matrix_group],
            FadeOut(descartes_theorem),
        )
        self.wait()

        # For terms related to coordinates, we have their names as well
        sep_line = Underline(matrix_group, stretch_factor=1)
        sep_line.set_color(GREY)
        sep_line.center()
        string_table_kx = string_table_C[2:3].T.copy()
        string_table_ky = string_table_C[3:4].T.copy()
        vector_kx = VGroup(
            Tex("\\mathbf{k}_{x} = "),
            TexMatrix(
                string_table_kx,
                tex_config={
                    "t2c": basic_t2c,
                    "use_labelled_svg": False,
                },
            ),
        )
        vector_ky = VGroup(
            Tex("\\mathbf{k}_{y} = "),
            TexMatrix(
                string_table_ky,
                tex_config={
                    "t2c": basic_t2c,
                    "use_labelled_svg": False,
                },
            ),
        )
        vector_group = VGroup(vector_kx, vector_ky)
        for vector in vector_group:
            vector[0][:-1].set_color(color_special)
            vector.arrange(RIGHT)
            vector.set_height(2)
        note_kx = Tex("\\text{约化横坐标向量} \\mathbf{k}_{x}")
        note_ky = Tex("\\text{约化纵坐标向量} \\mathbf{k}_{y}")
        note_group = VGroup(note_kx, note_ky)
        note_group.set_color(color_special)
        note_group.scale(0.7)
        note_group.arrange(RIGHT, buff=0.5)
        note_group.next_to(sep_line, DOWN, aligned_edge=LEFT)
        for mob, ref in zip(vector_group, note_group):
            mob.next_to(ref, DOWN)

        self.play(
            GrowFromCenter(sep_line),
            *[Write(note, stroke_color=color_special) for note in note_group],
            *[Write(vector[0], stroke_color=color_special) for vector in vector_group],
            *[
                FadeInFromPoint(bracket, bracket.get_center())
                for vector in vector_group
                for bracket in vector[1].get_brackets()
            ],
            TransformFromCopy(matrix_Ctrans.get_column(2), vector_kx[1].get_entries()),
            TransformFromCopy(matrix_Ctrans.get_column(3), vector_ky[1].get_entries()),
            run_time=2,
        )
        self.wait()

        # We can read 4 relations right away
        part = VGroup(
            *[
                matrix_Finv.get_row(row)[col]
                for row in (2, 3)
                for col in (2, 3)
            ]
        )
        self.play(FlashAround(part))
        self.wait()

        four_relations = VGroup(
            Tex(text) for text in [
                "\\mathbf{k}^{\\mathrm{T}}_{x} \\mathbf{G}^{-1} \\mathbf{k}_{x} = 1",
                "\\mathbf{k}^{\\mathrm{T}}_{x} \\mathbf{G}^{-1} \\mathbf{k}_{y} = 0",
                "\\mathbf{k}^{\\mathrm{T}}_{y} \\mathbf{G}^{-1} \\mathbf{k}_{x} = 0",
                "\\mathbf{k}^{\\mathrm{T}}_{y} \\mathbf{G}^{-1} \\mathbf{k}_{y} = 1",
            ]
        )
        for relation in four_relations:
            relation[:-2].set_color(color_special)
        four_relations.scale(0.8)
        four_relations.arrange_in_grid(2, 2, h_buff_ratio=0.3, v_buff_ratio=0.7)
        four_relations.move_to((RIGHT_SIDE + BOTTOM) / 2)

        relation_count = 0
        for row in (2, 3):
            for col in (2, 3):
                for mat in (matrix_C, matrix_Ctrans, matrix_Finv):
                    mat.generate_target()
                    mat.target.get_entries().set_opacity(0.1)
                part_C = VGroup(matrix_C.target.get_entries()[row * 4:(row + 1) * 4])
                part_Ctrans = VGroup(matrix_Ctrans.target.get_entries()[col::4])
                part_Finv = matrix_Finv.target.get_entries()[4 * row + col]
                parts = VGroup(part_C, part_Ctrans, part_Finv)
                for part in parts:
                    part.set_opacity(1)
                self.play(
                    *[MoveToTarget(mat) for mat in (matrix_C, matrix_Ctrans, matrix_Finv)],
                    *[FlashAround(part) for part in parts],
                    FadeIn(four_relations[relation_count], shift=LEFT)
                )
                self.wait()
                relation_count += 1

        # We may integrate these relations using complex number
        plane = ComplexPlane(y_range=(-2, 2, 1), faded_line_ratio=1)
        plane.move_to(TOP / 2)
        plane.add_coordinate_labels(text_config=CMU_TEXT_CONFIG, font_size=24)
        plane.set_opacity(0.7)
        self.play(
            *[FadeOut(mob) for mob in (matrix_group, sep_line, top_names)],
            FadeIn(plane),
            four_relations.animate.set_color(WHITE),
        )
        self.wait()

        eg_dot = Dot(plane.c2p(-0.57, -0.22))
        eg_circle = Circle(radius=1.5, arc_center=eg_dot.get_center())
        eg_note = Tex("\\text{圆心复坐标} z = x + \\mathrm{i}y")
        eg_note.next_to(eg_circle, RIGHT, aligned_edge=DOWN)
        eg_group = VGroup(eg_dot, eg_circle, eg_note)
        eg_group.set_color(color_special)
        self.play(
            FadeInFromPoint(eg_dot, eg_dot.get_center()),
            FadeInFromPoint(eg_circle, eg_circle.get_center()),
        )
        self.wait()
        self.play(FadeInFromPoint(eg_note[:-5], eg_dot.get_center()))
        self.wait()
        self.play(Write(eg_note[-5:], stroke_color=color_special))
        self.wait()

        string_table_kz = np.array([["k_{1} z_{1}", "k_{2} z_{2}", "k_{3} z_{3}", "k_{4} z_{4}"]]).T
        note_kz = Tex("\\text{约化复坐标向量} \\mathbf{k}_{z}")
        vector_kz = VGroup(
            Tex("\\mathbf{k}_{z} = \\mathbf{k}_{x} + \\mathrm{i} \\mathbf{k}_{y} = "),
            TexMatrix(
                string_table_kz,
                tex_config={
                    "t2c": basic_t2c,
                    "use_labelled_svg": False,
                },
            ),
        )
        note_kz.move_to(RIGHT_SIDE / 2)
        note_kz.match_height(note_ky).match_y(note_ky)
        note_kz.set_color(color_special)
        vector_kz.arrange(RIGHT)
        vector_kz.match_height(vector_ky)
        vector_kz.next_to(note_kz, DOWN).match_y(vector_ky)
        vector_kz[0][:-1].set_color(color_special)
        self.play(
            FadeOut(four_relations, run_time=1),
            AnimationGroup(
                FadeTransform(note_kx.copy(), note_kz),
                FadeTransform(note_ky.copy(), note_kz),
                FadeTransform(vector_kx[0].copy(), vector_kz[0]),
                FadeTransform(vector_ky[0].copy(), vector_kz[0]),
                FadeTransform(vector_kx[1].copy(), vector_kz[1]),
                FadeTransform(vector_ky[1].copy(), vector_kz[1]),
                run_time=2,
            )
        )
        self.wait()

        # What is the relation of kz and itself?
        note_kz_relation = TexText("$\\mathbf{k}_{z}$与自身的约束关系?")
        arrow_kz_relation = Arrow(ORIGIN, DOWN)
        arrow_kz_relation.next_to(note_kz, UP)
        note_kz_relation.next_to(arrow_kz_relation, UP)
        self.play(
            FadeOut(VGroup(plane, eg_dot, eg_circle, eg_note)),
            FadeIn(VGroup(arrow_kz_relation, note_kz_relation)),
        )
        self.wait()

        key_relation = Tex("\\mathbf{k}_{z}^{\\mathrm{T}} \\mathbf{G}^{-1} \\mathbf{k}_{z}")
        key_relation.next_to(note_kz_relation, UP)
        key_relation.set_color(color_special)
        self.play(FadeIn(key_relation[3:6], shift=DOWN))
        self.wait()
        self.play(
            FadeIn(key_relation[:3], shift=RIGHT),
            FadeIn(key_relation[6:], shift=LEFT),
            FadeOut(note_kz_relation[-1], shift=DOWN)
        )
        note_kz_relation = note_kz_relation[:-1]
        self.wait()

        # It has the same structure as the relations we just read out
        four_relations.generate_target()
        four_relations.target.move_to(TOP / 2)
        four_relations.target.match_x(VGroup(note_kx, note_ky))
        four_relations.set_opacity(0)
        self.add(four_relations)
        self.play(MoveToTarget(four_relations))
        self.wait()

        # Simple algebra manipulation
        derivation = Tex(
            """
            \\mathbf{k}_{z}^{\\mathrm{T}} \\mathbf{G}^{-1} \\mathbf{k}_{z}
            = & \\thickspace \\left( \\mathbf{k}_{x} + \\mathrm{i} \\mathbf{k}_{y} \\right)^{\\mathrm{T}} \\mathbf{G}^{-1} \\left( \\mathbf{k}_{x} + \\mathrm{i} \\mathbf{k}_{y} \\right) \\\\
            = & \\thickspace \\mathbf{k}_{x}^{\\mathrm{T}} \\mathbf{G}^{-1} \\mathbf{k}_{x} + \\mathrm{i} \\mathbf{k}_{x}^{\\mathrm{T}} \\mathbf{G}^{-1} \\mathbf{k}_{y} \\\\
              & \\thickspace {} + \\mathrm{i} \\mathbf{k}_{y}^{\\mathrm{T}} \\mathbf{G}^{-1} \\mathbf{k}_{x} + \\mathrm{i}^{2} \\mathbf{k}_{y}^{\\mathrm{T}} \\mathbf{G}^{-1} \\mathbf{k}_{y} \\\\
            = & \\thickspace 1 + \\mathrm{i} \\cdot 0 + \\mathrm{i} \\cdot 0 - 1 \\\\
            = & \\thickspace 0
            """,
            use_labelled_svg=False,
        )
        derivation.move_to(TOP / 2)
        derivation.match_x(note_kz)
        derivation.set_width(FRAME_WIDTH / 2 - 1)
        parts = VGroup(
            derivation[:8], derivation[8:29], derivation[29:69],
            derivation[69:81], derivation[81:]
        )
        for mob in (parts[0], parts[-1]):
            mob.set_color(color_special)
        self.play(
            ReplacementTransform(key_relation, parts[0]),
            FadeOutToPoint(note_kz_relation, note_kz.get_top()),
            FadeOutToPoint(arrow_kz_relation, note_kz.get_top()),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[FadeIn(part, shift=UP) for part in parts[1:]],
                lag_ratio=0.2,
            ),
            run_time=2,
        )
        self.wait()

        # Expand this relation
        note_expansion = VGroup(
            TexText(text)
            for text in [
                "(1) 代入$\\mathbf{G}^{-1}$：",
                "(2) 展开并整理：",
                "(3) 用点积改写：",
            ]
        )
        expansion = VGroup(
            Tex(text)
            for text in [
                "\\mathbf{k}_{z}^{\\mathrm{T}} \\mathbf{G}^{-1} \\mathbf{k}_{z} = 0",
                "\\mathbf{k}_{z}^{\\mathrm{T}} \\left( \\dfrac{1}{2} \\mathbf{I} - \\dfrac{1}{4} \\mathbf{1} \\mathbf{1}^{\\mathrm{T}} \\right) \\mathbf{k}_{z} = 0",
                "2 \\mathbf{k}_{z}^{\\mathrm{T}} \\mathbf{k}_{z} = \\mathbf{k}_{z}^{\\mathrm{T}} \\mathbf{1} \\mathbf{1}^{\\mathrm{T}} \\mathbf{k}_{z}",
                """
                2 \\left( \\mathbf{k}_{z} \\cdot \\mathbf{k}_{z} \\right)
                = \\left( \\mathbf{k}_{z} \\cdot \\mathbf{1} \\right) \\left( \\mathbf{1} \\cdot \\mathbf{k}_{z} \\right)
                """,
            ]
        )
        expansion[0].set_color(color_special)
        expansion.arrange(DOWN, aligned_edge=LEFT)
        note_expansion.set_color(GOLD_A)
        for mob, ref in zip(note_expansion, expansion[1:]):
            mob.next_to(ref, LEFT, buff=0.3)
        group_expansion = VGroup(expansion, note_expansion)
        group_expansion.set_width(FRAME_WIDTH * 0.6)
        group_expansion.to_corner(UL)
        key_relation = Tex("\\mathbf{k}_{z}^{\\mathrm{T}} \\mathbf{G}^{-1} \\mathbf{k}_{z} = 0")
        key_relation.set_color(color_special)
        key_relation.move_to(expansion[0]).match_height(expansion[0])
        group_kz = VGroup(note_kz, vector_kz)
        group_kz.generate_target()
        group_kz.target.scale(8 / 7).to_corner(UR)
        group_kz.target[1][0].set_color(WHITE)
        note_Ginv = TexText("$\\mathbf{G}^{-1}$的表达式")
        note_Ginv.set_color(color_special)
        formula_Ginv = Tex("\\mathbf{G}^{-1} = \\dfrac{1}{2} \\mathbf{I} - \\dfrac{1}{2n} \\mathbf{1} \\mathbf{1}^{\\mathrm{T}}")
        group_Ginv = VGroup(note_Ginv, formula_Ginv)
        group_Ginv.arrange(DOWN)
        group_Ginv.scale(0.9)
        group_Ginv.next_to(group_kz.target, DOWN, buff=1)

        self.play(
            *[
                FadeOut(mob, shift=3 * LEFT)
                for mob in (four_relations, vector_group, note_group)
            ],
            MoveToTarget(group_kz),
            FadeIn(group_Ginv, shift=UP),
            ReplacementTransform(parts[0], key_relation[:-2]),
            ReplacementTransform(parts[-1], key_relation[-2:]),
            FadeOutToPoint(parts[1:-1], derivation.get_corner(UR)),
        )
        self.wait()
        self.play(
            ReplacementTransform(key_relation, expansion[0]),
            AnimationGroup(
                *[
                    FadeIn(VGroup(note, formula), shift=UP)
                    for note, formula in zip(note_expansion, expansion[1:])
                ],
                lag_ratio=0.2,
                run_time=2,
            )
        )
        self.wait()

        # And we have complex Descartes' theorem
        arrow = Tex("\\Rightarrow")
        arrow.rotate(-PI / 2)
        arrow.next_to(group_expansion, DOWN)
        complex_theorem = Tex(
            """
            & \\left( k_{1} z_{1}  + k_{2} z_{2} + k_{3} z_{3} + k_{4} z_{4} \\right)^{2} \\\\
            = & \\thickspace 2 \\left[ (k_{1} z_{1})^{2} + (k_{2} z_{2})^{2} + (k_{3} z_{3})^{2} + (k_{4} z_{4})^{2} \\right]
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        for ind in (21, 31, 39, 47, 55):
            complex_theorem[ind].set_color(WHITE)
        complex_theorem.match_width(group_expansion)
        complex_theorem.next_to(arrow, DOWN)
        self.play(
            FadeInFromPoint(arrow, group_expansion.get_bottom()),
            FadeIn(complex_theorem, shift=0.8 * DOWN)
        )
        self.wait()

        # Compare with the original Descartes' theorem
        original_theorem = Tex(
            """
            & \\left( k_{1}  + k_{2} + k_{3} + k_{4} \\right)^{2} \\\\
            = & \\thickspace 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        original_theorem.match_height(complex_theorem)
        original_theorem.move_to(complex_theorem)
        for ind in (13, 18, 22, 26, 30):
            original_theorem[ind].set_color(WHITE)

        expansion_original = VGroup(
            Tex(text)
            for text in [
                "\\mathbf{k}^{\\mathrm{T}} \\mathbf{G}^{-1} \\mathbf{k} = 0",
                "\\mathbf{k}^{\\mathrm{T}} \\left( \\dfrac{1}{2} \\mathbf{I} - \\dfrac{1}{4} \\mathbf{1} \\mathbf{1}^{\\mathrm{T}} \\right) \\mathbf{k} = 0",
                "2 \\mathbf{k}^{\\mathrm{T}} \\mathbf{k} = \\mathbf{k}^{\\mathrm{T}} \\mathbf{1} \\mathbf{1}^{\\mathrm{T}} \\mathbf{k}",
                """
                2 \\left( \\mathbf{k} \\cdot \\mathbf{k} \\right)
                = \\left( \\mathbf{k} \\cdot \\mathbf{1} \\right) \\left( \\mathbf{1} \\cdot \\mathbf{k} \\right)
                """,
            ]
        )
        scale_factor = expansion[0][0].get_height() / expansion_original[0][0].get_height()
        for mob, ref in zip(expansion_original, expansion):
            mob.scale(scale_factor)
            mob.move_to(ref).match_x(ref, LEFT)
            mob.match_color(ref)
        string_table_k = np.array([["k_{1}", "k_{2}", "k_{3}", "k_{4}"]]).T
        note_k = Tex("\\text{曲率向量} \\mathbf{k}")
        vector_k = VGroup(
            Tex("\\mathbf{k} = "),
            TexMatrix(
                string_table_k,
                tex_config={
                    "t2c": basic_t2c,
                    "use_labelled_svg": False,
                },
            ),
        )
        vector_k.arrange(RIGHT)
        group_k = VGroup(note_k, vector_k)
        note_k.match_color(note_kz)
        for mob, ref in zip(group_k, group_kz):
            mob.match_height(ref).move_to(ref)

        everything_complex = VGroup(expansion, group_kz, complex_theorem)
        everything_original = VGroup(expansion_original, group_k, original_theorem)
        self.play(
            FadeOut(everything_complex),
            FadeIn(everything_original),
        )
        self.wait()
        self.play(
            FadeOut(everything_original),
            FadeIn(everything_complex),
        )
        self.wait()

        original_theorem_name = TexText("笛卡尔定理：")
        complex_theorem_name = TexText("复笛卡尔定理：")
        for name in (original_theorem_name, complex_theorem_name):
            name.set_color(color_special)
        complex_theorem.generate_target()
        group_complex = VGroup(complex_theorem_name, complex_theorem.target)
        group_original = VGroup(original_theorem_name, original_theorem)
        for group in (group_complex, group_original):
            group.arrange(RIGHT, buff=0.5)
        group_original.next_to(group_complex, UP, buff=0.8)
        original_theorem_name.match_x(complex_theorem_name, RIGHT)
        original_theorem.match_x(complex_theorem.target, LEFT)
        self.play(
            FadeOut(VGroup(group_kz, group_Ginv), shift=4 * RIGHT),
            FadeOut(VGroup(note_expansion, expansion, arrow), shift=2 * UP),
            FadeInFromPoint(complex_theorem_name, complex_theorem.get_left()),
            MoveToTarget(complex_theorem),
        )
        group_complex = VGroup(complex_theorem_name, complex_theorem)
        self.wait()
        self.play(FadeIn(group_original, shift=DOWN))
        self.wait()

        # How we construct the plane is irrelavent
        bg_rect = BackgroundRectangle(group_complex)
        bg_rect.set_width(FRAME_WIDTH)
        bg_rect.set_height(2, stretch=True)
        bg_rect.to_edge(UP, buff=0)
        bg_rect.generate_target()
        bg_rect.move_to(group_complex).set_opacity(0)
        plane = LeftSideComplexPlane(x_range=(-6, 4, 1), y_range=(-7, 7, 1))
        tangent_circles = TangentFourCircles(
            ORIGIN,
            UP,
            1.5 * RIGHT,
            n_components=256,
        )
        tangent_circles.suspend_updating()
        tangent_circles.rotate(PI / 5).scale(2)
        tangent_circles.next_to(bg_rect.target, DOWN).match_x(plane.n2p(0))
        tangent_circles.update_centers_from_circles()
        dots = VGroup(
            SmallDot(circle.get_center()).match_color(circle)
            for circle in tangent_circles.get_circles()
        )
        note_curvature = VGroup(
            Tex(f"k_{{{i}}} = ").set_color(color)
            for i, color in enumerate(CIRCLE_COLORS, start=1)
        )
        note_center = VGroup(
            Tex(f"z_{{{i}}} = ").set_color(color)
            for i, color in enumerate(CIRCLE_COLORS, start=1)
        )
        note_curvature.arrange(DOWN, buff=0.5)
        note_center.next_to(note_curvature, RIGHT, buff=2)
        for mob, ref in zip(note_center, note_curvature):
            mob.match_y(ref)
        curvatures = VGroup(
            DecimalNumber(
                2.0 / np.linalg.norm(circle.get_right() - circle.get_left()),
                text_config=CMU_TEXT_CONFIG,
            )
            for circle in tangent_circles.get_circles()
        )
        for curv, note in zip(curvatures, note_curvature):
            curv.next_to(note, RIGHT)
            curv.match_color(note)
            curv.suspend_updating()
        centers = VGroup(
            DecimalNumber(0, text_config=CMU_TEXT_CONFIG)
            for dot in dots
        )
        for center, note, dot in zip(centers, note_center, dots):
            center.next_to(note, RIGHT)
            center.match_color(note)
            center.add_updater(lambda mob, dot=dot: mob.set_value(plane.p2n(dot.get_center())))
            center.update().suspend_updating()
        centers.shift(0.05 * UP)
        group_value = VGroup(note_curvature, curvatures, note_center, centers)
        group_value.scale(0.8)
        group_value.next_to(plane, RIGHT)

        def get_equation_value():
            w = [
                k.get_value() * z.get_value()
                for k, z in zip(curvatures, centers)
            ]
            lhs = sum(w) ** 2
            rhs = 2 * sum(x ** 2 for x in w)
            assert(abs(lhs - rhs) < 1e-4)
            return lhs
        note_equation = Tex("\\text{左边} = \\text{右边} = ")
        note_equation.set_color(color_special)
        value_equation = DecimalNumber(0, text_config=CMU_TEXT_CONFIG)
        value_equation.match_color(note_equation)
        value_equation.add_updater(lambda mob: mob.set_value(get_equation_value()))
        value_equation.update().suspend_updating()
        group_equation = VGroup(note_equation, value_equation)
        group_equation.arrange(RIGHT)
        group_equation.scale(0.8)
        group_equation.next_to(group_value, DOWN, buff=0.8)
        self.play(
            *[
                FadeIn(mob, shift=2 * UP)
                for mob in (
                    plane, tangent_circles, dots, note_curvature, note_center, note_equation,
                    curvatures, centers, value_equation,
                )
            ],
            FadeOut(group_original, shift=2 * UP),
            MoveToTarget(bg_rect),
            group_complex.animate.move_to(bg_rect.target),
        )
        self.wait()
        for center in centers:
            center.resume_updating()
        value_equation.resume_updating()

        # Wiggle the plane a bit
        base_plane = plane.copy()
        origin = plane.n2p(0).copy()
        elapsed = [0.0]
        period = 20.0
        fade_time = 2.0

        def update_plane(mob, dt):
            elapsed[0] += dt
            s = np.clip(elapsed[0], 0, period)
            if s < fade_time:
                u = s**2 / (2 * fade_time)
            elif s > period - fade_time:
                u = period - fade_time - (period - s)**2 / (2 * fade_time)
            else:
                u = s - fade_time / 2
            t = TAU * u / (period - fade_time)
            mob.become(base_plane)

            angle = 20 * DEGREES * np.sin(t)
            mob.rotate(angle, about_point=origin)

            dx = 0.5 * np.sin(2 * t)
            dy = 0.4 * np.sin(t)
            mob.shift(dx * RIGHT + dy * UP)

        plane.add_updater(update_plane)
        self.wait(period)
        for mobject in self.mobjects:
            mobject.suspend_updating()
        self.wait()

        # Cross-term theorem and a hint
        self.remove(*self.mobjects)
        group_complex.next_to(group_original, DOWN, coor_mask=[0, 1, 0], buff=0.5)
        new_original_theorem_name = VGroup(
            TexText("笛卡尔定理："),
            TexText("（曲率-曲率）"),
        )
        new_complex_theorem_name = VGroup(
            TexText("复笛卡尔定理："),
            TexText("（约化复坐标-约化复坐标）"),
        )
        cross_theorem_name = VGroup(
            TexText("?????定理："),
            TexText("（曲率-约化复坐标）"),
        )
        cross_theorem = Tex(
            """
            & \\left( k_{1} + k_{2} + k_{3} + k_{4} \\right) \\left( k_{1} z_{1} + k_{2} z_{2} + k_{3} z_{3} + k_{4} z_{4} \\right) \\\\
            = & \\thickspace 2 \\left( k_{1}^{2} z_{1} + k_{2}^{2} z_{2} + k_{3}^{2} z_{3} + k_{4}^{2} z_{4} \\right)
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        for ind in (38, 44, 50, 56):
            cross_theorem[ind].set_color(WHITE)
        for name in (new_original_theorem_name, new_complex_theorem_name, cross_theorem_name):
            name[0].set_color(color_special)
            name[1].set_color(GOLD_A)
            name[1].scale(0.6)
            name.arrange(DOWN, aligned_edge=RIGHT)
        new_original_theorem_name.next_to(original_theorem_name.get_right(), LEFT, buff=0)
        new_complex_theorem_name.next_to(complex_theorem_name.get_right(), LEFT, buff=0)
        group_original = VGroup(new_original_theorem_name, original_theorem)
        group_complex = VGroup(new_complex_theorem_name, complex_theorem)
        group_cross = VGroup(cross_theorem_name, cross_theorem)
        group_cross.arrange(RIGHT)
        group_cross.next_to(group_complex, DOWN, buff=0.5)
        cross_theorem.match_x(complex_theorem, LEFT)
        cross_theorem_name.match_x(new_complex_theorem_name, RIGHT)
        cross_theorem.scale(0.85, about_edge=LEFT)
        self.add(group_original, group_complex)
        self.wait()
        self.play(FadeIn(cross_theorem_name))
        self.wait()
        self.play(FadeInFromPoint(cross_theorem, cross_theorem.get_left()))
        self.wait()

        group_hint = VGroup(top_names, matrix_group)
        for mob in (matrix_C, matrix_Ctrans, matrix_Finv):
            mob.set_opacity(1)
        self.play(
            FadeOut(group_original),
            FadeOut(group_complex),
            FadeIn(group_hint),
            FadeIn(sep_line),
        )
        self.wait()
        self.play(
            *[
                FlashAround(
                    VGroup(
                        matrix_Finv.get_entries()[ind]
                        for ind in indices
                    ),
                )
                for indices in ([2, 3], [8, 12])
            ],
        )
        self.wait()


class OutroSceneDescartes(Scene):
    def construct(self):
        frame_rect = ScreenRectangle()
        frame_rect.set_height(6)
        title = TexText("总结")
        title.next_to(frame_rect, UP)
        self.add(title, frame_rect)
        self.wait()


class OutroSceneDescartesAppendix(Scene):
    def construct(self):
        # Setup
        basic_t2c = {
            key: color
            for i, color in enumerate(CIRCLE_COLORS, start=1)
            for key in [
                f"\\mathbf{{c}}_{{{i}}}",
                f"k_{{{i}}}",
                f"z_{{{i}}}",
            ]
        }
        basic_t2c["k_{n+2}"] = SPHERE_COLORS[-1]

        descartes_name = TexText("笛卡尔定理：")
        complex_name = TexText("复笛卡尔定理：")
        soddy_gosset_name = TexText("索迪-戈赛特定理：")
        descartes_theorem = Tex(
            """
            & \\left( k_{1}  + k_{2} + k_{3} + k_{4} \\right)^{2} \\\\
            = & \\thickspace 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        complex_theorem = Tex(
            """
            & \\left( k_{1} z_{1}  + k_{2} z_{2} + k_{3} z_{3} + k_{4} z_{4} \\right)^{2} \\\\
            = & \\thickspace 2 \\left[ (k_{1} z_{1})^{2} + (k_{2} z_{2})^{2} + (k_{3} z_{3})^{2} + (k_{4} z_{4})^{2} \\right]
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        soddy_gosset_theorem = Tex(
            """
            & \\left( k_{1} + k_{2} + \\cdots + k_{n+2} \\right)^{2} \\\\
            = & \\thickspace n \\left( k_{1}^{2} + k_{2}^{2} + \\cdots + k_{n+2}^{2} \\right)
            """,
            t2c=basic_t2c,
            use_labelled_svg=False,
        )
        soddy_gosset_theorem[-5].set_color(WHITE)  # color fixing
        soddy_gosset_theorem[-2].set_color(SPHERE_COLORS[-1])  # color fixing
        descartes_group = VGroup(descartes_name, descartes_theorem)
        complex_group = VGroup(complex_name, complex_theorem)
        soddy_gosset_group = VGroup(soddy_gosset_name, soddy_gosset_theorem)
        groups = VGroup(descartes_group, complex_group, soddy_gosset_group)
        descartes_group.arrange(RIGHT)
        groups.arrange(DOWN, buff=0.8)
        for group in groups:
            group[0].set_color(YELLOW)
            group[0].match_x(descartes_group[0], RIGHT)
            group[1].match_x(descartes_group[1], LEFT)
        groups.center()
        self.add(groups)
        self.wait()
        for group in groups:
            self.play(FlashAround(group))
            self.wait()
        common_backend = VGroup(
            Tex("\\mathbf{C}^{\\mathrm{T}} \\mathbf{F} \\mathbf{C} = \\mathbf{G}"),
            Tex("\\mathbf{C} \\mathbf{G}^{-1} \\mathbf{C}^{\\mathrm{T}} = \\mathbf{F}^{-1}"),
        )
        common_backend.arrange(RIGHT, buff=2)
        rect = SurroundingRectangle(common_backend, buff=0.3)
        common_backend.add(rect)
        common_backend.to_edge(UP)
        common_backend.set_color(YELLOW)
        groups.generate_target()
        groups.target.scale(0.8, about_edge=DOWN)
        self.play(
            FadeIn(common_backend, shift=DOWN),
            MoveToTarget(groups),
        )
        self.wait()


class AboutCircleCocurvature(Scene):
    def construct(self):
        # Setup
        circle = Circle(radius=1.5, n_components=256)
        circle_info = VGroup(
            TexText("曲率$k$"),
            TexText("圆心坐标$(x, y)$"),
        )
        circle_info.arrange(DOWN, aligned_edge=LEFT)
        circle_group = VGroup(circle, circle_info)
        circle_group.arrange(DOWN, buff=0.5)
        circle_group.move_to(LEFT_SIDE / 3)
        circle_group.set_color(GREEN)
        vector = MobjectMatrix(
            [Tex(text) for text in ["k", "t", "kx", "ky"]],
            n_cols=1,
        )
        vector.move_to(RIGHT_SIDE / 3)
        self.add(circle_group, vector)
        self.wait()
        element_t = vector.get_entries()[1]
        self.play(
            FlashAround(element_t, buff=0.3),
            element_t.animate.set_color(YELLOW),
        )
        self.wait()

        cocurv_def = Tex(
            """
            \\text{反演曲率} \\thickspace t
            & = \\dfrac{x^2 + y^2 - r^2}{r} \\\\
            & = k \\left( x^2 + y^2 \\right) - \\dfrac{1}{k}
            """
        )
        cocurv_def[:5].set_color(YELLOW)
        group = VGroup(circle_group, vector)
        group.generate_target()
        group.target.arrange(RIGHT, buff=1).to_edge(LEFT, buff=0.8)
        cocurv_def.next_to(group.target, RIGHT)
        temp = cocurv_def[0].copy()
        temp.match_y(element_t)
        cocurv_def.shift(temp.get_center() - cocurv_def[0].get_center())
        self.play(
            MoveToTarget(group),
            FadeIn(cocurv_def, shift=4 * LEFT),
        )
        self.wait()

        # Prepare for inversion
        plane = FadeCoordinatePlane(unit_size=2)
        unit_circle = Circle(radius=plane.get_x_unit_size(), n_components=256)
        unit_circle.set_color(YELLOW)
        self.add(plane, circle_group, vector)
        circle_group.generate_target()
        circle_group.target.scale(0.8).shift(0.7 * RIGHT)
        circle_name = TexText("圆$C$")
        circle_name.match_color(circle)
        circle_name.next_to(circle_group.target, UP)
        self.play(
            FadeInFromPoint(circle_name, circle.get_top()),
            FadeOut(cocurv_def),
            FadeIn(plane),
            MoveToTarget(circle_group),
            vector.animate(path_arc=PI / 2).scale(0.8).set_color(GREEN).next_to(LEFT_SIDE, UR, buff=0.5),
        )
        self.wait()
        self.play(
            FadeInFromPoint(unit_circle, unit_circle.get_center()),
        )
        self.wait()

        unit_center = unit_circle.get_center()
        unit_radius = 2

        def invert_points(points, center=unit_center, radius=unit_radius):
            diff = points - center
            dists = np.linalg.norm(diff, axis=1, keepdims=True)
            dists = np.clip(dists, 1e-6, None)
            new_points = center + diff * (radius**2 / (dists**2))
            return new_points

        inverted_circle = circle.copy().apply_points_function(invert_points, about_edge=None)
        inverted_circle.set_color(RED)
        self.play(
            TransformFromCopy(circle, inverted_circle, run_time=3),
        )
        self.wait()

        # Inverted object is also a circle
        inverted_name = TexText("圆$C'$")
        inverted_name.match_color(inverted_circle)
        inverted_name.next_to(inverted_circle, UP, aligned_edge=LEFT)
        self.play(FadeInFromPoint(inverted_name[1:], inverted_circle.get_center()))
        self.wait()
        self.play(FadeInFromPoint(inverted_name[0], inverted_circle.get_center()))
        self.wait()
        inverted_info = VGroup(
            TexText("曲率$|t|$"),
            TexText("圆心坐标$\\left(\\tfrac{kx}{t}, \\tfrac{ky}{t} \\right)$"),
        )
        inverted_info.arrange(DOWN, aligned_edge=LEFT)
        inverted_info.scale(0.7).match_color(inverted_circle)
        inverted_info.next_to(inverted_circle, DOWN, aligned_edge=LEFT)
        self.play(FadeInFromPoint(inverted_info, inverted_circle.get_bottom()))
        self.wait()


class AboutCircleDual(Scene):
    def construct(self):
        core_example = TangentFourCircles(
            UP,
            ORIGIN,
            1.2 * RIGHT,
        )
        core_example.suspend_updating()
        core_example.rotate(PI / 5).center().scale(3)
        core_example.update_centers_from_circles()
        core_example.resume_updating()
        c1, c2, c3, c4 = orig_circles = core_example.get_circles()
        base_points = core_example.base_points
        self.add(core_example)
        self.wait()

        # Wiggle a bit
        anchors = [p.get_center().copy() for p in base_points]
        wiggle_params = [
            # ax, ay, wx, wy
            (1.00, 0.75, -0.73, 1.07),
            (0.82, 1.00, 0.91, -0.67),
            (0.93, 0.81, 0.61, 0.83),
        ]
        amplitude = 0.2
        duration = 10
        fade_time = 2

        def get_envelope(t):
            # Fade-in part
            if t < fade_time:
                alpha = t / fade_time
                return smooth(alpha)
            # Fade-out part
            if t > duration - fade_time:
                alpha = (duration - t) / fade_time
                return smooth(max(0, alpha))
            return 1.0

        def update_base_points(mob, dt):
            mob.increment_value(dt)
            t = mob.get_value()
            for point, anchor, par in zip(base_points, anchors, wiggle_params):
                envelope = get_envelope(t)
                ax, ay, wx, wy = par
                dx = envelope * amplitude * ax * np.sin(wx * t)
                dy = envelope * amplitude * ay * np.sin(wy * t)
                point.move_to(anchor + dx * RIGHT + dy * UP)

        wiggle_movement_driver = ValueTracker(0)
        wiggle_movement_driver.add_updater(update_base_points)
        self.add(wiggle_movement_driver)
        self.wait(duration)
        self.remove(wiggle_movement_driver)
        core_example.refresh()

        # A configuration has 4 circles and 6 tangent points
        orig_names = VGroup(
            TexText(f"$C_{{{i}}}$").set_color(color)
            for i, color in enumerate(CIRCLE_COLORS, start=1)
        )
        orig_names[-1].scale(0.5)
        for name, circle in zip(orig_names, orig_circles):
            name.move_to(circle)
        self.play(
            AnimationGroup(
                *[FadeInFromPoint(name, name.get_center()) for name in orig_names],
                lag_ratio=0.05,
            )
        )
        self.wait()

        tangent_points = core_example.get_tangent_points()
        dots = VGroup(
            Dot(p).set_color(YELLOW)
            for p in tangent_points
        )
        self.play(
            AnimationGroup(
                AnimationGroup(
                    *[
                        Flash(p, line_length=0.05, flash_radius=0.1, line_stroke_width=3)
                        for p in tangent_points
                    ],
                ),
                AnimationGroup(
                    *[FadeInFromPoint(dot, dot.get_center()) for dot in dots],
                    rate_func=squish_rate_func(smooth, 0.5, 1),
                ),
                run_time=2,
            )
        )
        self.wait()

        # Construct dual configuration
        def circle_to_pedoe_vector(circle):
            center = circle.get_center()
            radius = circle.get_width() / 2
            return geom_to_pedoe_vector(center, radius)

        def configuration_to_matrix(tangent_circles):
            circles = tangent_circles.get_circles()
            vectors = [circle_to_pedoe_vector(circle) for circle in circles]
            matrix = np.column_stack(vectors)
            return matrix

        def get_dual_circles(tangent_circles):
            orig_circles = tangent_circles.get_circles()
            orig_matrix = configuration_to_matrix(tangent_circles)
            dual_matrix = orig_matrix @ (0.5 * np.ones((4, 4)) - np.eye(4))
            dual_vectors = dual_matrix.T
            dual_circles = VGroup()
            for vector, circle in zip(dual_vectors, orig_circles):
                info = analyze_raw_vector(vector)
                shape_type = info["shape_type"]
                center = info["center"]
                assert(shape_type == "real_circle")
                dual = Circle(radius=info["radius"], n_components=256).move_to(center)
                dual.match_color(circle)
                dual_circles.add(dual)
            return dual_circles

        dual_circles = get_dual_circles(core_example)
        dual_names = VGroup(
            TexText(f"$D_{{{i}}}$").set_color(color)
            for i, color in enumerate(CIRCLE_COLORS, start=1)
        )
        for name, circle in zip(dual_names, dual_circles):
            name.scale(0.8)
            name.move_to(circle)
        dual_names[-1].next_to(dual_circles[-1], RIGHT)

        # Explain the rules with circle C1
        dim_opacity = 0.2

        def get_on_and_off_dots(dots, circle, tol=1e-4):
            center = circle.get_center()
            radius = circle.get_width() / 2
            on_dots = VGroup()
            off_dots = VGroup()
            for point, dot in zip(tangent_points, dots):
                distance = np.linalg.norm(point - center)
                if np.isclose(distance, radius, atol=tol, rtol=0):
                    on_dots.add(dot)
                else:
                    off_dots.add(dot)
            return on_dots, off_dots

        on_dots, off_dots = get_on_and_off_dots(dots, c1)
        self.play(
            *[circle.animate.set_stroke(opacity=dim_opacity) for circle in orig_circles[1:]],
            *[name.animate.set_fill(opacity=dim_opacity) for name in orig_names[1:]],
            *[dot.animate.set_fill(opacity=dim_opacity) for dot in on_dots],
        )
        self.wait()
        self.play(
            *[FadeInFromPoint(mob, mob.get_center()) for mob in (dual_circles[0], dual_names[0])]
        )
        self.wait()

        # Same goes for c2, c3 and c4
        for ind, c in enumerate(orig_circles[1:], start=1):
            on_dots, off_dots = get_on_and_off_dots(dots, c)
            self.play(
                *[
                    circle.animate.set_stroke(opacity=dim_opacity if i != ind else 1)
                    for i, circle in enumerate(orig_circles)
                ],
                *[
                    name.animate.set_fill(opacity=dim_opacity if i != ind else 1)
                    for i, name in enumerate(orig_names)
                ],
                *[
                    dot.animate.set_fill(opacity=dim_opacity if dot in on_dots else 1)
                    for dot in dots
                ],
                *[FadeOut(mob) for mob in (dual_circles[ind - 1], dual_names[ind - 1])],
                *[FadeIn(mob) for mob in (dual_circles[ind], dual_names[ind])],
            )
            self.wait()

        # Here are the questions
        note_dual_tangency = TexText("对偶构型四圆互切")
        note_dual_tangency.next_to(dual_circles, UP, buff=0.5)
        note_involution = Tex("\\text{对偶构型} \\xrightarrow{\\text{对偶}} \\text{原始构型}")
        note_involution.next_to(orig_circles, UP)
        notes = VGroup(note_dual_tangency, note_involution)
        notes.set_color(YELLOW)
        self.play(
            FadeIn(note_dual_tangency, shift=UP),
            orig_circles[-1].animate.set_stroke(opacity=dim_opacity),
            orig_names[-1].animate.set_fill(opacity=dim_opacity),
            *[dot.animate.set_fill(opacity=1) for dot in dots],
            *[FadeIn(mob) for mob in (dual_circles[:-1])],
            *[FadeIn(mob) for mob in (dual_names[:-1])],
        )
        self.wait()
        self.play(
            *[circle.animate.set_stroke(opacity=1) for circle in orig_circles],
            *[circle.animate.set_stroke(opacity=dim_opacity) for circle in dual_circles],
            *[name.animate.set_fill(opacity=1) for name in orig_names],
            *[name.animate.set_fill(opacity=dim_opacity) for name in dual_names],
            FadeTransform(note_dual_tangency, note_involution),
        )
        self.wait()

        # Add complex plane for reference
        geometry_group = VGroup(
            dual_circles, dual_names, dots,
            orig_circles, orig_names,
        )
        geometry_group.generate_target()
        geometry_group.target.scale(0.9, about_edge=DOWN).to_edge(LEFT, buff=0.5)
        geometry_group.target[2].set_opacity(0)
        plane = FadeComplexPlane()
        self.add(plane, geometry_group)
        self.play(
            FadeIn(plane),
            FadeOut(note_involution, shift=UP),
            MoveToTarget(geometry_group),
        )
        geometry_group = VGroup(dual_circles, dual_names, orig_circles, orig_names)
        self.add(geometry_group)
        self.wait()

        circle_properties = VGroup(
            TexText("圆$C_i$：有向曲率$k_i$，圆心复坐标$z_i$"),
            TexText("圆$D_i$：有向曲率$b_i$，圆心复坐标$w_i$"),
        )
        circle_properties.arrange(DOWN, aligned_edge=LEFT)
        circle_properties.set_width(FRAME_WIDTH / 2 - 1)
        circle_properties.to_corner(UL)
        for text in circle_properties:
            for mob in (text[:3], text[8:10], text[-2:]):
                mob.set_color(YELLOW)
        self.play(FadeIn(circle_properties[0], shift=UP))
        self.wait()
        self.play(
            FadeIn(circle_properties[1], shift=UP),
            *[circle.animate.set_stroke(opacity=dim_opacity) for circle in orig_circles],
            *[circle.animate.set_stroke(opacity=1) for circle in dual_circles],
            *[name.animate.set_fill(opacity=dim_opacity) for name in orig_names],
            *[name.animate.set_fill(opacity=1) for name in dual_names],
        )
        self.wait()

        # More homework for fun (with a key hint!)
        homework = VGroup(
            Tex(f"({ind + 1}) \\thickspace " + text)
            for ind, text in enumerate([
                "\\sum_{i}{k_i b_i} = 0",
                "\\sum_{i}{k_i z_i b_i w_i} = 0",
                "\\sum_{i}{k_i} = \\sum_{i}{b_i}",
                "\\sum_{i}{k_i z_i} = \\sum_{i}{b_i w_i}",
                "\\sum_{i}{k_i^2} = \\sum_{i}{b_i^2}",
                "\\sum_{i}{(k_i z_i)^2} = \\sum_{i}{(b_i w_i)^2}",
                "& k_i + b_i \\text{和} k_i z_i + b_i w_i \\\\ & \\text{都与下标$i$无关}",
            ])
        )
        homework.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        homework.set_height(6.5).to_corner(UR)
        rect = BackgroundRectangle(homework)
        rect.set_height(FRAME_HEIGHT).set_width(5)
        rect.center().to_edge(RIGHT, buff=0)
        self.play(
            *[circle.animate.set_stroke(opacity=1) for circle in orig_circles],
            *[name.animate.set_fill(opacity=1) for name in orig_names],
            FadeIn(rect),
            AnimationGroup(
                *[FadeIn(line, shift=LEFT) for line in homework],
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.wait()

        key_hint = VGroup(
            Tex("\\mathbf{D} = \\mathbf{C} \\mathbf{H}"),
            Tex(
                """
            \\text{其中} \\mathbf{H} = \\dfrac{1}{2} \\mathbf{J} - \\mathbf{I}
            = \\dfrac{1}{2} 
            \\begin{bmatrix}
            -1 & 1 & 1 & 1 \\\\
            1 & -1 & 1 & 1 \\\\
            1 & 1 & -1 & 1 \\\\
            1 & 1 & 1 & -1
            \\end{bmatrix}
            """
            ),
        )
        key_hint.set_color(GOLD_A)
        key_hint.arrange(DOWN, buff=0.5)
        key_hint.match_width(geometry_group)
        key_hint.move_to(geometry_group).match_x(circle_properties, LEFT)
        self.play(
            *[FadeOut(mob) for mob in (plane, geometry_group)],
            FadeIn(key_hint),
        )
        self.wait()


class InterestingTopicsToExplore(Scene):
    def construct(self):
        sep_lines = VGroup(
            Underline(TexText("test"))
            .rotate(PI / 2).set_height(FRAME_HEIGHT)
            .set_color(GREY_B).set_opacity(0.5)
            .move_to(side / 3)
            for side in (LEFT_SIDE, RIGHT_SIDE)
        )
        titles = VGroup(
            TexText(text)
            for text in [
                "圆与线性代数的更多联系",
                "同一个圆的不同向量表示",
                "弯曲空间中的笛卡尔定理",
            ]
        )
        titles.set_color(YELLOW).scale(0.75)
        for title, pos in zip(titles, [LEFT_SIDE * 2 / 3, ORIGIN, RIGHT_SIDE * 2 / 3]):
            title.move_to(pos).to_edge(UP)
        connections = BulletedList(
            "秩的性质 \\\\ 为什么最多四圆互切?",
            "惯性定理 \\\\ 四个圆可以两两正交吗? \\\\ 哪些四圆构型是允许的?",
            "线性组合 \\\\ $a_{1} \\mathbf{c}_{1} + a_{2} \\mathbf{c}_{2}$代表着什么?",
            "线性变换 \\\\ 如何用矩阵表示几何变换?"
        )
        for item in connections:
            item[:5].set_color(TEAL)
        connections.set_width(FRAME_WIDTH / 3 - 0.5)
        connections.match_x(titles[0])
        lie_vector = Tex(
            """
            \\begin{bmatrix}
            \\dfrac{1+x^2+y^2-r^2}{2} \\\\[1.5em]
            \\dfrac{1-x^2-y^2+r^2}{2} \\\\[1.5em]
            x \\\\[1em]
            y \\\\[1em]
            \\pm r
            \\end{bmatrix}
            """,
        )
        lie_vector.set_height(4.5)
        VGroup(lie_vector[:16], lie_vector[44:]).scale(1.1)
        spherical_image = ImageMobject("Spherical_Descartes_Example.png")
        spherical_image.set_height(3)
        spherical_theorem = VGroup(
            Tex(
                """
            & \\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2} \\\\
            = & \\thickspace 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right) + 4
            """,
                t2c=CURVATURE_T2C,
                use_labelled_svg=False,
            ),
            TexText("（有向曲率$k = \\cot r$）"),
        )
        spherical_theorem[0][-2:].set_color(YELLOW)
        spherical_theorem[1].set_color(GREY)
        spherical_theorem.match_width(connections)
        spherical_theorem.arrange(DOWN)
        spherical_group = Group(spherical_image, spherical_theorem)
        spherical_group.arrange(DOWN)
        spherical_group.match_x(titles[-1])
        self.add(titles, sep_lines)
        self.wait()
        self.play(
            AnimationGroup(
                *[FadeIn(item, shift=RIGHT) for item in connections],
                lag_ratio=0.05,
            ),
            FadeIn(lie_vector, shift=UP),
            AnimationGroup(
                *[FadeIn(item, shift=LEFT) for item in spherical_group],
                lag_ratio=0.05,
            ),
        )
        self.wait()


class ReferenceDescartesLinAlg(ReferenceScene):
    def get_references(self):
        kocik_short_ref = Reference(
            name='Proof of Descartes circle formula and its generalization clarified',
            authors='Jerzy Kocik',
            pub='arXiv:1910.09174, 2019.',
            doi='https://arxiv.org/pdf/1910.09174',
            info='''
            非常简洁的三页论文，用线性代数的方法证明笛卡尔定理。可以与这期视频直接对照阅读。\\\\
            向量和内积的定义方式跟视频中给的有点区别，稍微留意一下。
            ''',
        )

        kocik_long_ref = Reference(
            name='A theorem on circle configurations',
            authors='Jerzy Kocik',
            pub='arXiv:0706.0372, 2007.',
            doi='https://arxiv.org/pdf/0706.0372',
            info='''
            上面那篇短文所依据的详细版本。系统地建立一般四圆构型的矩阵恒等式，点明了笛卡尔定理只是个特例。\\\\
            此外还讨论了阿波罗尼奥斯问题与高维中的多球互切构型。
            ''',
        )

        beyond_ref = Reference(
            name='Beyond the Descartes Circle Theorem',
            authors='Jeffrey C. Lagarias, Colin L. Mallows and Allan R. Wilks',
            pub='The American Mathematical Monthly, 2002, 109(4): 338-361.',
            doi='https://doi.org/10.1080/00029890.2002.11920896',
            info='''
            笛卡尔定理只讲了四个互切圆的大小关系，那么圆心的位置关系是什么呢？\\\\
            在更高维的空间中，有没有类似笛卡尔定理的结论呢？\\\\
            如果我们处理的不再是欧氏几何，而是球面几何或者双曲几何呢？\\\\
            这篇文章给出了笛卡尔定理的一系列推广，结论也都是优美的矩阵恒等式。
            ''',
        )

        cvla_ref = Reference(
            name='Circles, Vectors, and Linear Algebra',
            authors='Richard E. Pfiefer and Cathleen Van Hook',
            pub='Mathematics Magazine, 1993, 66(2): 75-86.',
            doi='https://doi.org/10.2307/2691113',
            info='''
            用四维向量表示圆（以及点和直线）之后，能做的事情可就太多了。\\\\
            这篇文章重新证明了梅涅劳斯定理、托勒密定理、蒙日三圆定理、牛顿线定理、费尔巴哈定理
            等几何结论，而背后贯穿着一个简单的思想：“线性相关”。\\\\
            这期视频的标题也来源于此。
            ''',
        )

        beecroft_ref = Reference(
            name='Properties of Circles in Mutual Contact',
            authors='Philip Beecroft',
            pub="The Lady's and Gentleman's Diary, 1842, pp. 91-96.",
            doi='https://en.wikisource.org/wiki/Properties_of_circles_in_mutual_contact',
            info='''
            通过四圆互切的原始构型和对偶构型的关系来证明笛卡尔定理，还得到了涉及两组圆曲率的恒等式。\\\\
            视频结尾的对偶构型的思考题灵感来自这篇文章。
            ''',
        )

        coxeter_ref = Reference(
            name='The Problem of Apollonius',
            authors='H. S. M. Coxeter',
            pub='The American Mathematical Monthly, 1968, 75(1): 5-15.',
            doi='https://doi.org/10.2307/2315097',
            info='''
            对Beecroft的证明做了重新整理和简化，借助三角形的内切圆和旁切圆来证明笛卡尔定理。
            ''',
        )

        levrie_ref = Reference(
            name="A Straightforward Proof of Descartes's Circle Theorem",
            authors='Paul Levrie',
            pub='The Mathematical Intelligencer, 2019, 41(3): 24-27.',
            doi='https://doi.org/10.1007/s00283-019-09883-x',
            info='''
            借助海伦公式和初等代数给出的一个很短的笛卡尔定理证明。
            ''',
        )

        pedoe_ref = Reference(
            name='On a Theorem in Geometry',
            authors='Daniel Pedoe',
            pub='The American Mathematical Monthly, 1967, 74(6): 627-640.',
            doi='https://doi.org/10.2307/2314247',
            info='''
            从对称性猜测公式，再通过反演、Cayley–Menger行列式、Grassmann演算等方法研究和证明笛卡尔定理的思路。\\\\
            如果再用矩阵和向量重新组织Pedoe的最后一种证明，就跟视频里展示的方法很接近了。
            ''',
        )

        return (
            kocik_short_ref, kocik_long_ref,
            beyond_ref, cvla_ref,
            beecroft_ref, coxeter_ref,
            levrie_ref, pedoe_ref,
        )


class ThumbnailDescartes(Scene):
    def construct(self):
        title = TexText("笛卡尔定理")
        title.set_color(YELLOW).scale(2)
        theorem = Tex(
            """
            & \\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2} \\\\
            = & \\thickspace 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)
            """,
            t2c=CURVATURE_T2C,
            use_labelled_svg=False,
        )
        theorem.scale(1.2)
        theorem.next_to(title, DOWN)
        text_group = VGroup(title, theorem)
        text_group.arrange(DOWN, buff=0.8)
        text_group.set_width(FRAME_WIDTH / 2 - 1)
        text_group.move_to(LEFT_SIDE / 2)

        curvs = [10, 15, 19, -6]
        p1, p2, p3 = get_tangent_circle_centers(*curvs[:-1])
        circles = TangentFourCircles(p1, p2, p3, fourth_type="bigger")
        circles.suspend_updating()
        circles.set_width(FRAME_WIDTH / 2 - 1)
        circles.move_to(RIGHT_SIDE / 2)

        curv_texts = VGroup(
            Tex(f"k_{{{i}}}={ki}").scale(0.7).set_color(color).move_to(circle)
            for i, (ki, color, circle) in enumerate(
                zip(curvs, CIRCLE_COLORS, circles.get_circles()),
                start=1,
            )
        )
        for text in curv_texts:
            text.scale(0.8)
        curv_texts[-1].next_to(circles.get_bottom(), UP, buff=0.5)
        curv_texts.set_opacity(0.5)

        glows = Group()
        for circle in circles.get_circles():
            circle.set_fill(opacity=0.05)
            glow = GlowCircle(
                circle,
                sigma=0.15,
                intensity=0.2,
            )
            glows.add(glow)

        focus = circles.get_center()
        bg = FullScreenRectangle()
        bg.set_fill("#020711", opacity=1)
        bg.set_stroke(width=0)

        n_steps = 128
        gradient = VGroup()
        for i in range(n_steps, 0, -1):
            circle = Circle(radius=8.5 * i / n_steps, n_components=128)
            circle.set_fill("#183A60", opacity=0.002)
            circle.set_stroke(width=0)
            circle.move_to(focus)
            gradient.add(circle)

        grid = VGroup()
        step = 0.7
        segment = 0.25
        falloff = 3.2
        for x in np.arange(-FRAME_WIDTH / 2, FRAME_WIDTH / 2 + step, step):
            for y in np.arange(-FRAME_HEIGHT / 2, FRAME_HEIGHT / 2, segment):
                p1 = np.array([x, y, 0])
                p2 = np.array([x, y + segment, 0])
                mid = (p1 + p2) / 2
                d = np.linalg.norm(mid - focus)
                alpha = 0.13 * np.exp(-(d / falloff) ** 2)
                grid.add(Line(
                    p1, p2,
                    stroke_color="#79ADDD",
                    stroke_width=0.8,
                    stroke_opacity=alpha,
                ))

        for y in np.arange(-FRAME_HEIGHT / 2, FRAME_HEIGHT / 2 + step, step):
            for x in np.arange(-FRAME_WIDTH / 2, FRAME_WIDTH / 2, segment):
                p1 = np.array([x, y, 0])
                p2 = np.array([x + segment, y, 0])
                mid = (p1 + p2) / 2
                d = np.linalg.norm(mid - focus)
                alpha = 0.13 * np.exp(-(d / falloff) ** 2)
                grid.add(Line(
                    p1, p2,
                    stroke_color="#79ADDD",
                    stroke_width=0.8,
                    stroke_opacity=alpha,
                ))

        self.add(bg, gradient, grid)
        self.add(text_group, circles)
        self.add(curv_texts)
        # self.add(glows)
        self.wait()


class OldThumbnailDescartes(Scene):
    def construct(self):
        circle = Circle()
        circle.set_stroke(width=5)
        circle.set_fill(opacity=0.3)

        vectors = VGroup(
            Vector(direction, thickness=5)
            for direction in (2 * UR, 2 * UL)
        )
        for vec in vectors:
            vec.center()

        linalg = TexMatrix(
            [
                [0, "-1/2", 0, 0],
                ["-1/2", 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            element_alignment_corner=ORIGIN,
            bracket_v_buff=0.5,
        )

        symbols = VGroup(circle, vectors, linalg)
        symbols.set_color(YELLOW)
        for mobs in symbols[:-1]:
            mobs.match_height(symbols[-1])
        symbols.arrange(RIGHT, buff=1)
        symbols.set_width(FRAME_WIDTH - 2)

        theorem_title = TexText("笛卡尔定理")
        theorem_content = Tex(
            """
            \\left( k_{1} + k_{2} + k_{3} + k_{4} \\right)^{2}
            = 2 \\left( k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2} \\right)
            """,
            t2c=CURVATURE_T2C,
            use_labelled_svg=False,
        )
        theorem = VGroup(theorem_title, theorem_content)
        theorem.arrange(DOWN)
        theorem.set_width(FRAME_WIDTH - 2)
        theorem.to_edge(UP, buff=0.5)
        symbols.next_to(theorem, DOWN, buff=1)

        self.add(symbols, theorem)
        self.wait()


class CipherTeaser(Scene):
    def construct(self):
        lines = [
            "置圓圓名圓名甲徑名乘加名圓加平開加用小徑",
            "曰徑乙元丙亨置乙徑丙徑貞丙利亨之而除得合",
            "術乘乘併圓圓置乘倍亨問",
            "甲徑徑圓利元徑方貞圓",
        ]
        texts = VGroup(*[
            self.fixed_width_text_line(line)
            for line in lines
        ])
        texts.arrange(DOWN)
        colors = [CIRCLE_COLORS[i] for i in [2, 1, 0, 3]]
        for text, color in zip(texts, colors):
            text.set_color(color)
        hint_t2c = {
            "\\mathbf{F}": interpolate_color(colors[0], WHITE, 0.3),
            "\\mathbf{R}": interpolate_color(colors[2], WHITE, 0.3),
        }
        hint_top = TexText("$\\mathbf{F}$ail", t2c=hint_t2c)
        hint_bottom = TexText("$\\mathbf{R}$ence", t2c=hint_t2c)
        hints = VGroup(hint_top, hint_bottom)
        hints.set_opacity(0.5)
        p1, p2, p3 = get_tangent_circle_centers(10, 15, 19)
        bg_circles = TangentFourCircles(p1, p2, p3)
        bg_circles.suspend_updating()
        bg_circles.set_height(FRAME_HEIGHT * 0.3)
        bg_circles.set_stroke(width=2, opacity=0.3)
        bg_circles.set_color(WHITE)
        c1, c2, c3, c4 = bg_circles.get_circles()
        circles = VGroup(c4, c3, c2, c1)
        circles.arrange(RIGHT, aligned_edge=DOWN)
        circles.to_corner(DR)
        circles_copy = circles.copy()
        circles_copy.flip().to_corner(DL)
        circles_bottom = VGroup(circles, circles_copy)
        circles_top = circles_bottom.copy()
        circles_top.flip(RIGHT).to_edge(UP)
        hint_top.match_y(circles_top, UP)
        hint_bottom.match_y(circles_bottom, DOWN)
        self.add(circles_top, circles_bottom, texts, hints)
        self.wait()

    def fixed_width_text_line(self, text, width=FRAME_WIDTH - 2):
        chars = VGroup(TexText(char) for char in text)
        if len(chars) == 1:
            chars[0].move_to(ORIGIN)
            return chars

        xs = np.linspace(-width / 2, width / 2, len(chars))
        for char, x in zip(chars, xs):
            char.move_to(x * RIGHT)
        return chars
