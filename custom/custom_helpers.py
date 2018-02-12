from helpers import *
from mobject.vectorized_mobject import VGroup

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

def tweak_color(color1, color2, alpha = 0.3):
    """Return a weight average of two colors."""
    alpha = clamp(0, 1, alpha)
    tweaked_rgb = alpha * color_to_rgb(color2) + (1-alpha) * color_to_rgb(color1)
    return rgb_to_color(tweaked_rgb)

def brighten(color, alpha = 0.3):
    return tweak_color(color, WHITE, alpha)

def darken(color, alpha = 0.3):
    return tweak_color(color, BLACK, alpha)



