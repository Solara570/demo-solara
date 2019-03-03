#coding=utf-8

from manimlib.constants import *
from manimlib.utils.config_ops import digest_config
from manimlib.utils.rate_functions import *

from manimlib.animation.composition import AnimationGroup
from manimlib.animation.creation import ShowCreation, Write, FadeIn, FadeOut, GrowFromPoint

from manimlib.mobject.mobject import Mobject, Group
from manimlib.mobject.types.vectorized_mobject import VMobject, VGroup

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
            VGroup(*bubble.content), *self.content_animation_args, **self.content_animation_kwargs
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


class BubbleShrinkToPoint(BubbleGrowFromPoint):
    CONFIG = {
        "bubble_animation_kwargs"  : {"rate_func" : lambda t: 1-smooth(t)},
        "content_animation_kwargs" : {"rate_func" : lambda t: 1-smooth(t)},
    }


class BubbleGrowFromTip(BubbleGrowFromPoint):
    def __init__(self, bubble, **kwargs):
        self.bubble_animation_args = [bubble.get_tip()]
        self.content_animation_args = [bubble.get_tip()]
        BubbleGrowFromPoint.__init__(self, bubble, **kwargs)


class BubbleShrinkToTip(BubbleShrinkToPoint):
    def __init__(self, bubble, **kwargs):
        self.bubble_animation_args = [bubble.get_tip()]
        self.content_animation_args = [bubble.get_tip()]
        BubbleShrinkToPoint.__init__(self, bubble, **kwargs)



