from big_ol_pile_of_manim_imports import *

COLOR_PALLETE = (
    (0.00, BLACK),
    (0.20, GREEN),
    (0.21, YELLOW),
    (0.40, RED),
    (0.60, WHITE)
)
L = len(COLOR_PALLETE) - 1

# Need further tweaking to speed up rendering time.
# Current time: 31s for one 1024*768-pixel frame.
# So slow.
class GrayScottModel(VMobject):
    CONFIG = {
        "nx": 1024,
        "ny": 768,
        # "nx": 100,
        # "ny": 100,
        "height": 6,
        "use_periodic_condition": True,
    }

    def __init__(self, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.setup()
        self.add_boundary()
        self.initialize()
        self.add_updater(lambda m, dt: m.image.become(m.get_evolved_image()))

    def setup(self):
        self.Da = 0.2097
        self.Db = 0.1050
        self.feed_rate = 0.037
        self.kill_rate = 0.060

    def get_Da(self):
        return self.Da

    def get_Db(self):
        return self.Db

    def get_feed_rate(self):
        return self.feed_rate

    def get_kill_rate(self):
        return self.kill_rate

    def add_boundary(self):
        self.boundary = Rectangle(
            width=self.nx, height=self.ny,
            stroke_width=3, stroke_color=GREY,
        )
        self.boundary.set_height(self.height)
        self.add(self.boundary)

    def initialize(self):
        self.Ca = np.ones((self.ny + 2, self.nx + 2))
        self.Cb = np.zeros((self.ny + 2, self.nx + 2))
        self.kickstart_concentration()
        self.add_image()

    def kickstart_concentration(self):
        x, y = np.meshgrid(
            np.linspace(0, 1, self.nx + 2), np.linspace(0, 1, self.ny + 2)
        )
        mask = (0.4 < x) & (x < 0.6) & (0.4 < y) & (y < 0.6)
        self.Ca[mask] = 0.50
        self.Cb[mask] = 0.25

    def add_image(self):
        init_array = self.Cb[1:-1, 1:-1]
        self.image = self.get_image_from_concetration_array(init_array)
        self.add(self.image)

    def get_image_from_pixel_array(self, pixel_array):
        reshaped_pixel_array = np.reshape(pixel_array, (self.ny, self.nx, 3))
        image = ImageMobject(reshaped_pixel_array)
        image.set_height(self.height)
        return image

    def get_empty_pixel_array(self):
        return np.zeros((self.ny, self.nx, 3), dtype=np.uint8)

    def get_pixel_array_from_concentration_array(self, concentration_array):
        pixel_array = self.get_empty_pixel_array()
        for i, row in enumerate(concentration_array):
            for j, c in enumerate(row):
                int_rgb = self.get_int_rgb_from_concentration(c)
                pixel_array[i][j][:] = int_rgb
        return pixel_array

    def get_image_from_concetration_array(self, concentration_array):
        pixel_array = self.get_pixel_array_from_concentration_array(concentration_array)
        image = self.get_image_from_pixel_array(pixel_array)
        return image

    def get_color_from_concentration(self, concentration):
        c = np.clip(concentration, 0, 1)
        if c <= COLOR_PALLETE[0][0]:
            return COLOR_PALLETE[0][1]
        elif c >= COLOR_PALLETE[L][0]:
            return COLOR_PALLETE[L][1]
        else:
            for k in range(0, L):
                if c > COLOR_PALLETE[k][0] and c <= COLOR_PALLETE[k + 1][0]:
                    color_1 = COLOR_PALLETE[k][1]
                    color_2 = COLOR_PALLETE[k + 1][1]
                    alpha = (c - COLOR_PALLETE[k][0]) / (COLOR_PALLETE[k + 1][0] - COLOR_PALLETE[k][0])
                    return interpolate_color(color_1, color_2, alpha)

    def get_int_rgb_from_concentration(self, concentration):
        color = self.get_color_from_concentration(concentration)
        return color_to_int_rgb(color)

    def update_boundary(self, full_concentration_array):
        A = full_concentration_array
        if self.use_periodic_condition:
            A[0, :] = A[-2, :]      # First outside column = Last inside column
            A[-1, :] = A[1, :]      # Last outside column = First inside column
            A[:, 0] = A[:, -2]      # First outside row = Last inside row
            A[:, -1] = A[:, 1]      # Last outside row = First inside row
        else:
            A[0, :] = A[-2, :]      # First outside column = 0
            A[-1, :] = A[1, :]      # Last outside column = 0
            A[:, 0] = A[:, -2]      # First outside row = 0
            A[:, -1] = A[:, 1]      # Last outside row = 0

    def get_laplacian(self, full_concentration_array):
        """
        The Laplacian is performed with a 3x3 convolution:
        del^2(A[i][j]) = A[i+1][j] + A[i-1][j] + A[i][j+1] + A[i][j-1] - 4*A[i][j]
        """
        A = full_concentration_array
        return A[:-2, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] - 4 * A[1:-1, 1:-1]

    def evolved_arrays(self, delta_t=1):
        """
        Evolve concentration arrays according to Gray-Scott model
        """
        Ca_inner = self.Ca[1:-1, 1:-1]
        Cb_inner = self.Cb[1:-1, 1:-1]
        # Calculate diffusion term
        diffusion_a = self.get_Da() * self.get_laplacian(self.Ca)
        diffusion_b = self.get_Db() * self.get_laplacian(self.Cb)
        # Calculate reaction term
        reaction_a = -Ca_inner * Cb_inner * Cb_inner
        reaction_b = -reaction_a
        # Calculate feed term for compound A
        feed_a = self.get_feed_rate() * (1 - Ca_inner)
        # Calculate kill term for compound B
        # Feed rate is added to make sure net kill rate is always higher than 0
        kill_b = -(self.get_feed_rate() + self.get_kill_rate()) * Cb_inner
        # Bring every term together
        Ca_inner += (diffusion_a + reaction_a + feed_a) * delta_t
        Cb_inner += (diffusion_b + reaction_b + kill_b) * delta_t
        # Update boundary conditions
        self.update_boundary(self.Ca)
        self.update_boundary(self.Cb)
        return self.Ca, self.Cb

    def get_evolved_image(self):
        self.evolved_arrays()
        return self.get_image_from_concetration_array(self.Cb[1:-1, 1:-1])


class TestGSModel(Scene):
    def construct(self):
        gs_model = GrayScottModel()
        self.add(gs_model)
        self.wait(60)


class GSModelIntro(Scene):
    def construct(self):
        self.show_title()
        self.show_construction_rule()
        self.show_parameters()

    def show_title(self):
        gs_text = TextMobject("Gray-Scott Model", color=YELLOW)
        link = TextMobject("https://pmneila.github.io/jsexp/grayscott/")
        gs_text.scale(1.5).to_corner(UL, buff=0.3)
        link.scale(0.8).next_to(gs_text, DOWN, aligned_edge=LEFT, buff=0.15)
        self.play(FadeIn(gs_text), FadeIn(link), run_time=1)
        self.wait(4)
        self.play(FadeOut(gs_text), FadeOut(link), run_time=1)
        self.wait(2)

    def show_construction_rule(self):
        text_1 = TextMobject("想象这个屏幕是一个容器，里面有两种化学物质A和B")
        text_2 = TextMobject("容器中有以下几个过程：")
        process_1 = TextMobject("1. A和B的", "扩散")
        process_2 = TextMobject("2. ", "化学反应A+2B$\\rightarrow$3B")
        process_3 = TextMobject("3. 持续而均匀地", "加入A")
        process_4 = TextMobject("4. 持续而均匀地", "移出B")
        text_3 = TextMobject("每一点根据B的浓度$c_\\text{B}$上色")
        processes = VGroup(process_1, process_2, process_3, process_4)
        colors = (YELLOW, BLUE, GREEN, RED)
        for process, color in zip(processes, colors):
            process[1].set_color(color)
        group = VGroup(
            text_1, text_2,
            process_1, process_2, process_3, process_4,
            text_3
        )
        group.arrange_submobjects(DOWN, aligned_edge=LEFT)
        group.scale(0.6).to_corner(UL, buff=0.3)
        for process in processes:
            process.shift(0.3 * RIGHT)
        self.play(FadeIn(group), run_time=1)
        self.wait(6)

        text_4 = TextMobject("两种物质的浓度$c_\\text{A}$和$c_\\text{B}$满足：")
        equations = TexMobject(
            "\\dfrac{\\partial c_\\text{A}}{\\partial t} = ",
            "D_\\text{A} \\nabla^2 c_\\text{A}",
            "-c_\\text{A} c_\\text{B}^2",
            "+f(1-c_\\text{A})",
            "\\\\",
            "\\dfrac{\\partial c_\\text{B}}{\\partial t} = ",
            "D_\\text{B} \\nabla^2 c_\\text{B}",
            "+c_\\text{A} c_\\text{B}^2",
            "-(k+f)c_\\text{B}",
        )
        equations[1::5].set_color(colors[0])
        equations[2::5].set_color(colors[1])
        equations[3].set_color(colors[2])
        equations[-1].set_color(colors[3])
        equations.scale(0.6).to_corner(DL, buff=0.3)
        text_4.scale(0.6).next_to(equations, UP, aligned_edge=LEFT)
        self.play(FadeIn(equations), FadeIn(text_4), run_time=1)
        self.wait(4)
        self.play(FadeOut(group), FadeOut(equations), FadeOut(text_4), run_time=1)
        self.wait(2)

    def show_parameters(self):
        text_1 = TextMobject("目前使用的参数：")
        da = TexMobject("D_\\text{A} = 0.2097", color=YELLOW)
        db = TexMobject("D_\\text{B} = 0.1050", color=YELLOW)
        feed_rate = TexMobject("f = 0.0200", color=GREEN)
        kill_rate = TexMobject("k = 0.0600", color=RED)
        parameters = VGroup(da, db, feed_rate, kill_rate)
        parameters.arrange_submobjects(DOWN, aligned_edge=LEFT)
        parameters.next_to(text_1, DOWN, aligned_edge=LEFT)
        group = VGroup(text_1, parameters)
        group.scale(0.8).to_corner(UL, buff=0.3)
        self.play(FadeIn(group), run_time=1)
        self.wait(5)
        self.play(FadeOut(group), run_time=1)
        self.wait()




