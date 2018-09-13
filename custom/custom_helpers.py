#coding=utf-8

import numpy as np
import random

from constants import *
from mobject.types.vectorized_mobject import VGroup
from utils.color import rgb_to_color, color_to_rgb


def list_shuffle(l):
    """Return a shuffled copy of the original list ``l``."""
    x = l[:]
    random.shuffle(x)
    return x

def vgroup_expansion(mobs):
    """Flatten a nested VGroup object ``mobs``."""
    while any(map(lambda x: isinstance(x, VGroup), mobs)):
        expanded_mobs = []
        for mob in mobs.submobjects:
            expanded_mobs.extend(mob)
        mobs = VGroup(expanded_mobs)
    return mobs

def mobs_shuffle(mobs):
    """Shuffle the submobjects inside a VGroup object ``mobs``."""
    mobs = vgroup_expansion(mobs)
    mobs.submobjects = list_shuffle(mobs.submobjects)
    return mobs

def fit_mobject_in(content_mob, container_mob, buffer_factor = 0.6):
    width_factor = container_mob.get_width() / content_mob.get_width()
    height_factor = container_mob.get_height() / content_mob.get_height()
    scale_factor = min(width_factor, height_factor)
    content_mob.scale(scale_factor * buffer_factor)
    content_mob.move_to(container_mob)
    return content_mob

def tweak_color(color1, color2, alpha = 0.3):
    """Return a weighted-average of two colors."""
    alpha = np.clip(alpha, 0, 1)
    tweaked_rgb = alpha * color_to_rgb(color2) + (1-alpha) * color_to_rgb(color1)
    return rgb_to_color(tweaked_rgb)

def brighten(color, alpha = 0.3):
    return tweak_color(color, WHITE, alpha)

def darken(color, alpha = 0.3):
    return tweak_color(color, BLACK, alpha)



