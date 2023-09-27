import warnings
import functools


class SDFCADError(Exception):
    pass


class SDFCADInfiniteObjectError(Exception):
    """
    Error raised when an infinite object is encountered where not suitable.
    """

    pass


class SDFCADWarning(Warning):
    pass


class SDFCADAlphaQualityWarning(SDFCADWarning):
    show = True


def alpha_quality(decorated_fun):
    @functools.wraps(decorated_fun)
    def wrapper(*args, **kwargs):
        if SDFCADAlphaQualityWarning.show:
            warnings.warn(
                f"{decorated_fun.__name__}() is alpha quality "
                f"and might give wrong results. Use with care. "
                f"Hide this warning by setting sdf.errors.SDFCADAlphaQualityWarning.show=False.",
                SDFCADAlphaQualityWarning,
            )
            with warnings.catch_warnings():
                # Don't reissue nested alpha quality warnings
                warnings.simplefilter("ignore", SDFCADAlphaQualityWarning)
                return decorated_fun(*args, **kwargs)
        else:
            return decorated_fun(*args, **kwargs)

    return wrapper
