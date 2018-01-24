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
        "bubble_animation_class"   : ShowCreation,
        "bubble_animation_args"    : [],
        "bubble_animation_kwargs"  : {},
        "content_animation_class"  : Write,
        "content_animation_args"   : [],
        "content_animation_kwargs" : {},
    }
    def __init__(self, bubble, **kwargs):
        digest_config(self, kwargs)
        create_bubble = self.bubble_animation_class(
            bubble, *self.bubble_animation_args, **self.bubble_animation_kwargs
        )
        create_content = self.content_animation_class(
            Group(*bubble.content), *self.content_animation_args, **self.content_animation_kwargs
        )
        AnimationGroup.__init__(
            self, create_bubble, create_content, **kwargs
        )

class BubbleCreation(BubbleAnimation):
    # Rename to make it clearer
    pass

class BubbleFadeIn(BubbleAnimation):
    CONFIG = {
        "bubble_animation_class"   : FadeIn,
        "content_animation_class"  : FadeIn,
    }

class BubbleFadeOut(BubbleAnimation):
    CONFIG = {
        "bubble_animation_class"   : FadeOut,
        "content_animation_class"  : FadeOut,
    }

class BubbleGrowFromPoint(BubbleAnimation):
    CONFIG = {
        "bubble_animation_class"   : GrowFromPoint,
        "bubble_animation_args"    : [ORIGIN],
        "content_animation_class"  : GrowFromPoint,
        "content_animation_args"   : [ORIGIN],
    }





