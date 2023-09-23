import math
import warnings
import functools

pi = math.pi

degrees = math.degrees
radians = math.radians


class SDFCADWarning(Warning):
    pass


class SDFCADAlphaQualityWarning(SDFCADWarning):
    pass


def alpha_quality(decorated_fun):
    @functools.wraps(decorated_fun)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{decorated_fun.__name__}() is alpha quality "
            f"and might give wrong results. Use with care.",
            SDFCADAlphaQualityWarning,
        )
        with warnings.catch_warnings():
            # Don't reissue nested alpha quality warnings
            warnings.simplefilter("ignore", SDFCADAlphaQualityWarning)
            return decorated_fun(*args, **kwargs)

    return wrapper
