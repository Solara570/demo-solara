import numpy as np

from helpers import *
from mobject import Mobject
from mobject.vectorized_mobject import *
from mobject.point_cloud_mobject import *
from mobject.svg_mobject import *
from mobject.tex_mobject import *

from animation.animation import Animation
from animation.transform import *
from animation.simple_animations import *
from animation.playground import *

from topics.geometry import *
from topics.objects import *
from topics.number_line import *
from topics.three_dimensions import *
from topics.common_scenes import *

from scene import Scene
from camera import Camera

# self.skip_animations
# self.force_skipping()
# self.revert_to_original_skipping_status()


## A few animations about bubbles
class BubbleAnimation(AnimationGroup):
    CONFIG = {
        "bubble_animation_class"  : ShowCreation,
        "content_animation_class" : Write,
    }
    def __init__(self, bubble, **kwargs):
        digest_config(self, kwargs)
        Group(bubble, bubble.content).shift_onto_screen()
        create_bubble = self.bubble_animation_class(bubble)
        create_content = self.content_animation_class(Group(*bubble.content))
        AnimationGroup.__init__(
            self, create_bubble, create_content, **kwargs
        )

class BubbleCreation(BubbleAnimation):
    # Rename to make it clearer
    pass

class BubbleFadeIn(BubbleAnimation):
    CONFIG = {
        "bubble_animation_class"  : FadeIn,
        "content_animation_class" : FadeIn,
    }

class BubbleFadeOut(BubbleAnimation):
    CONFIG = {
        "bubble_animation_class"  : FadeOut,
        "content_animation_class" : FadeOut,
    }