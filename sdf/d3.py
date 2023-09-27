import os
import logging
import functools
import numpy as np
import operator
import warnings
import copy

# internal modules
from . import dn, d2, ease, mesh, errors, util
from .units import units

# external modules
import scipy.optimize
from scipy.linalg import LinAlgWarning
import rich.progress

# Constants

logger = logging.getLogger(__name__)

ORIGIN = np.array((0, 0, 0))

X = np.array((1, 0, 0))
Y = np.array((0, 1, 0))
Z = np.array((0, 0, 1))

UP = Z
DOWN = -Z
RIGHT = X
LEFT = -X
BACK = Y
FRONT = -Y

# SDF Class

_ops = {}


class SDF3:
    def __init__(self, f):
        self.f = f

    def __call__(self, p):
        return self.f(p).reshape((-1, 1))

    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        raise AttributeError

    def __or__(self, other):
        return union(self, other)

    def __and__(self, other):
        return intersection(self, other)

    def __sub__(self, other):
        return difference(self, other)

    def k(self, k=None):
        newSelf = copy.deepcopy(self)
        newSelf._k = k
        return newSelf

    def generate(self, *args, **kwargs):
        return mesh.generate(self, *args, **kwargs)

    @errors.alpha_quality
    def closest_surface_point(self, point):
        def distance(p):
            # root() wants same input/output dims (yeah...)
            return np.repeat(self.f(np.expand_dims(p, axis=0)).ravel()[0], 3)

        dist = self.f(np.expand_dims(point, axis=0)).ravel()[0]
        optima = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LinAlgWarning, RuntimeWarning, UserWarning)
            )
            for method in (
                # loosely sorted by speed
                "lm",
                "broyden2",
                "df-sane",
                "hybr",
                "broyden1",
                "anderson",
                "linearmixing",
                "diagbroyden",
                "excitingmixing",
                "krylov",
            ):
                try:
                    optima[method] = (
                        opt := scipy.optimize.root(
                            distance, x0=np.array(point), method=method
                        )
                    )
                except Exception as e:
                    pass
                opt.zero_error = abs(opt.fun[0])
                opt.zero_error_rel = opt.zero_error / dist
                opt.dist_error = np.linalg.norm(opt.x - point) - dist
                opt.dist_error_rel = opt.dist_error / dist
                logger.debug(f"{method = }, {opt = }")
                # shortcut if fit is good
                if (
                    np.allclose(opt.fun, 0)
                    and abs(opt.dist_error / dist - 1) < 0.01
                    and opt.dist_error < 0.01
                ):
                    break

            def cost(m):
                penalty = (
                    # unsuccessfulness is penaltied
                    (not optima[m].success)
                    # a higher status normally means something bad
                    + abs(getattr(optima[m], "status", 1))
                    # the more we're away from zero, the worse it is
                    # „1mm of away from boundary is as bad as one status or success step”
                    + optima[m].zero_error
                    # the distance error can be quite large e.g. for non-uniform scaling,
                    # and methods often find weird points, it makes sense to compare to the SDF
                    + optima[m].dist_error
                )
                logger.debug(f"{m = :20s}: {penalty = }")
                return penalty

            best_root = optima[best_method := min(optima, key=cost)]
            closest_point = best_root.x
            if (
                best_root.zero_error > 1
                or best_root.zero_error_rel > 0.01
                or best_root.dist_error > 1
                or best_root.dist_error_rel > 0.01
            ):
                warnings.warn(
                    f"Closest surface point to {point} acc. to method {best_method!r} seems to be {closest_point}. "
                    f"The SDF there is {best_root.fun[0]} (should be 0, that's {best_root.zero_error} or {best_root.zero_error_rel*100:.2f}% off).\n"
                    f"Distance between {closest_point} and {point} is {np.linalg.norm(point - closest_point)}, "
                    f"SDF says it should be {dist} (that's {best_root.dist_error} or {best_root.dist_error_rel*100:.2f}% off)).\n"
                    f"The root finding algorithms seem to have a problem with your SDF, "
                    f"this might be caused due to operations breaking the metric like non-uniform scaling.",
                    errors.SDFCADWarning,
                )
        return closest_point

    @errors.alpha_quality
    def surface_intersection(self, start, direction=None):
        """
        ``start`` at a point, move (back or forth) along a line following a
        ``direction`` and return surface intersection coordinates.

        .. note::

            In case there is no intersection, the result *might* (not sure
            about that) return the point on the line that's closest to the
            surface 🤔.

        Args:
            start (3d vector): starting point
            direction (3d vector or None): direction to move into, defaults to
            ``-start`` (”move to origin”).

        Returns:
            3d vector: the optimized surface intersection
        """
        if direction is None:
            direction = -start

        def transform(t):
            return start + t * direction

        def distance(t):
            # root() wants same input/output dims (yeah...)
            return np.repeat(self.f(np.expand_dims(transform(t), axis=0)).ravel()[0], 3)

        dist = self.f(np.expand_dims(start, axis=0)).ravel()[0]
        optima = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LinAlgWarning, RuntimeWarning, UserWarning)
            )
            for method in (
                # loosely sorted by speed
                "lm",
                "broyden2",
                "df-sane",
                "hybr",
                "broyden1",
                "anderson",
                "linearmixing",
                "diagbroyden",
                "excitingmixing",
                "krylov",
            ):
                try:
                    optima[method] = (
                        opt := scipy.optimize.root(distance, x0=[0], method=method)
                    )
                except Exception as e:
                    pass
                opt.zero_error = abs(opt.fun[0])
                opt.zero_error_rel = opt.zero_error / dist
                opt.point = transform(opt.x[0])
                logger.debug(f"{method = }, {opt = }")
                # shortcut if fit is good
                if np.allclose(opt.fun, 0):
                    break

            def cost(m):
                penalty = (
                    # unsuccessfulness is penaltied
                    (not optima[m].success)
                    # a higher status normally means something bad
                    + abs(getattr(optima[m], "status", 1))
                    # the more we're away from zero, the worse it is
                    # „1mm of away from boundary is as bad as one status or success step”
                    + optima[m].zero_error
                )
                logger.debug(f"{m = :20s}: {penalty = }")
                return penalty

            best_root = optima[best_method := min(optima, key=cost)]
            closest_point = transform(best_root.x[0])
            if best_root.zero_error > 1 or best_root.zero_error_rel > 0.01:
                warnings.warn(
                    f"Surface intersection point from {start = } to {direction = }, acc. to method {best_method!r} seems to be {closest_point}. "
                    f"The SDF there is {best_root.fun[0]} (should be 0, that's {best_root.zero_error} or {best_root.zero_error_rel*100:.2f}% off).\n"
                    f"The root finding algorithms seem to have a problem with your SDF, "
                    f"this might be caused due to operations breaking the metric like non-uniform scaling "
                    f"or just because there is no intersection...",
                    errors.SDFCADWarning,
                )
        return closest_point

    @errors.alpha_quality
    def minimum_sdf_on_plane(self, origin, normal, return_point=False):
        """
        Find the minimum SDF distance (not necessarily the real distance if you
        have non-uniform scaling!) on a plane around an ``origin`` that points
        into the ``normal`` direction.

        Args:
            origin (3d vector): a point on the plane
            normal (3d vector): normal vector of the plane
            return_point (bool): whether to also return the closest point (on
                the plane!)

        Returns:
            float: the (minimum) distance to the plane
            float, 3d vector : distance and closest point (on the plane!) if
                ``return_point=True``
        """
        basemat = np.array(
            [(e1 := _perpendicular(normal)), (e2 := np.cross(normal, e1))]
        ).T

        def transform(t):
            return origin + basemat @ t

        def distance(t):
            return self.f(np.expand_dims(transform(t), axis=0)).ravel()[0]

        optima = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LinAlgWarning, RuntimeWarning, UserWarning)
            )
            for method in (
                "Nelder-Mead",
                "Powell",
                "CG",
                "BFGS",
                "L-BFGS-B",
                "TNC",
                "COBYLA",
                "SLSQP",
                "trust-constr",
            ):
                try:
                    optima[method] = (
                        opt := scipy.optimize.minimize(
                            distance, x0=[0, 0], method=method
                        )
                    )
                    opt.point = transform(opt.x)
                    logger.debug(f"{method = }, {opt = }")
                except Exception as e:
                    logger.error(f"{method = } error {e!r}")

        best_min = optima[min(optima, key=lambda m: optima[m].fun)]
        if return_point:
            return best_min.fun, best_min.point
        else:
            return best_min.fun

    @errors.alpha_quality
    def extent_in(self, direction):
        """
        Determine the largest distance from the origin in a given ``direction``
        that's still within the object.

        Args:
            direction (3d vector): the direction to check

        Returns:
            float: distance from origin
        """
        # create probing points along direction to check where object ends roughly
        # object ends when SDF is only increasing (not the case for infinite repetitions e.g.)
        probing_points = np.expand_dims(np.logspace(0, 9, 30), axis=1) * direction
        d = self.f(probing_points)  # get SDF value at probing points
        n_trailing_ascending = util.n_trailing_ascending_positive(d)
        if not n_trailing_ascending:
            return np.inf
        if (ratio := n_trailing_ascending / d.size) < 0.5:
            warnings.warn(
                f"extent_in({direction = !r}): "
                f"Only {n_trailing_ascending}/{d.size} ({ratio*100:.1f}%) of probed points in "
                f"{direction = } have ascending positive SDF distance values. "
                f"This can be caused by infinite objects. "
                f"Result of might be wrong. ",
                errors.SDFCADWarning,
            )
        faraway = probing_points[
            -n_trailing_ascending + 1
        ]  # choose first point after which SDF only increases
        closest_surface_point = self.closest_surface_point_to_plane(
            origin=faraway, normal=direction
        )
        extent = np.linalg.norm(closest_surface_point)
        return extent

    @errors.alpha_quality
    def closest_surface_point_to_plane(self, origin, normal):
        """
        Find the closest surface point to a plane around an ``origin`` that points
        into the ``normal`` direction.

        Args:
            origin (3d vector): a point on the plane
            normal (3d vector): normal vector of the plane

        Returns:
            3d vector : closest surface point
        """
        distance, plane_point = self.minimum_sdf_on_plane(
            origin=origin, normal=normal, return_point=True
        )
        return self.surface_intersection(start=plane_point, direction=normal)

    @errors.alpha_quality
    def move_to_positive(self, direction=Z):
        return self.translate(self.extent_in(-direction) * direction)

    def cut(self, direction=UP, point=ORIGIN, at=None, k=None, return_cutouts=False):
        """
        Split an object along a direction and return resulting parts.

        Args:
            direction (3d vector): direction to cut into
            point (3d vector): point to perform the cut at
            at (sequence of float): where to perform the cuts along the
                ``direction`` starting at ``point``. Defaults to ``[0]``,
                meaning it only splits at ``point``.
            k (float): passed to :any:`intersection`
            return_cutouts (bool): whether to return the used cutout masks as
                second value


        Returns:
            sequence of SDFs: the parts
            two sequences of SDFs: the parts and cutouts if ``return_cutouts=True``
        """
        direction = np.array(direction).reshape(3)
        direction = _normalize(direction)
        point = np.array(point).reshape(3)
        if at is None:
            at = [0]
        at = [-np.inf] + list(at) + [np.inf]
        parts = []
        cutouts = []
        for start, end in zip(at[:-1], at[1:]):
            cuts = []
            if np.isfinite(start):
                cuts.append(plane(normal=direction, point=point + start * direction))
            if np.isfinite(end):
                cuts.append(plane(normal=-direction, point=point + end * direction))
            if len(cuts) > 1:
                cutout = intersection(*cuts)
            elif len(cuts) == 1:
                cutout = cuts[0]
            else:
                cutout = self
            cutouts.append(cutout)
            parts.append(intersection(self, cutout, k=k))
        if return_cutouts:
            return parts, cutouts
        else:
            return parts

    def chamfer(self, size, at=ORIGIN, direction=Z, e=ease.linear):
        """
        Chamfer (and then cut) an object along a plane

        Args:
            size (float): size to chamfer along ``direction``.
            at (3d point): A point on the plane where to chamfer at. Defaults
                to ORIGIN.
            direction (3d vector): direction to chamfer to. Defaults to Z.
            e (ease.Easing): the easing to use. Will be scaled with ``size``.
        """
        direction = direction / np.linalg.norm(direction)
        result = self.stretch(at, at - size * direction)
        result = result.modulate_between(at + size * direction, at, e=-size * e)
        result &= plane(direction, point=at)
        return result

    def save(
        self,
        path="out.stl",
        screenshot=False,
        add_text=None,
        openscad=False,
        plot=True,
        plot_kwargs=None,
        **kwargs,
    ):
        mesh.save(path, self, **{**dict(samples=2**18), **kwargs})
        print(f"💾 Saved mesh to {path!r} ({os.stat(path).st_size} bytes)")
        if openscad:
            with open((p := f"{path}.scad"), "w") as fh:
                fh.write(
                    f"""
        import("{path}");
                """.strip()
                )
                print(f"💾 Saved OpenSCAD viewer script to {p!r}")
        if plot:
            try:
                import pyvista as pv
            except ImportError as e:
                print(
                    f"To use the plotting functionality, "
                    f"install pyvista and trame (pip install pyvista trame)"
                )
                return
            # pv.set_jupyter_backend("client")
            plotter = pv.Plotter()
            # axes = pv.Axes(show_actor=True, actor_scale=2, line_width=5)
            # plotter.camera = pv.Camera()
            # plotter.camera.position = (-1, -1, 1)
            # plotter.add_actor(axes.actor)
            plotter.enable_parallel_projection()
            m = pv.read(path)
            plotter.add_mesh(m)
            # xl, xu, yl, yu, zl, zu = m.bounds
            # plotter.add_ruler(
            #     pointa=[0, 0, 0],
            #     pointb=[max(0, xu), 0, 0],
            #     number_minor_ticks=10,
            #     title="X",
            # )
            # plotter.add_ruler(
            #     pointa=[0, 0, 0],
            #     pointb=[0, max(0, yu), 0],
            #     number_minor_ticks=10,
            #     title="Y",
            # )
            # plotter.add_ruler(
            #     pointa=[0, 0, 0],
            #     pointb=[0, 0, max(0, zu)],
            #     number_minor_ticks=10,
            #     title="Z",
            # )
            plotter.add_axes()
            if add_text:
                if isinstance(add_text, str):
                    add_text = dict(text=add_text)
                plotter.add_text(**add_text)
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=UserWarning)
                if screenshot:
                    if not isinstance(screenshot, str):
                        screenshot = f"{path}.png"
                    plotter.screenshot(screenshot)
                    print(f"🖼️ Saved screenshot to {screenshot!r}")
                plotter.show(**(plot_kwargs or {}))

    def show_slice(self, *args, **kwargs):
        return mesh.show_slice(self, *args, **kwargs)


def sdf3(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))

    return wrapper


def op3(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))

    _ops[f.__name__] = wrapper
    return wrapper


def op32(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return d2.SDF2(f(*args, **kwargs))

    _ops[f.__name__] = wrapper
    return wrapper


# Helpers


def _length(a):
    return np.linalg.norm(a, axis=1)


def _normalize(a):
    return a / np.linalg.norm(a)


def _dot(a, b):
    return np.sum(a * b, axis=1)


def _vec(*arrs):
    return np.stack(arrs, axis=-1)


def _perpendicular(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError("zero vector")
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


_min = np.minimum
_max = np.maximum

# Primitives


@sdf3
def sphere(radius=None, diameter=None, center=ORIGIN):
    if (radius is not None) == (diameter is not None):
        raise ValueError(f"Specify either radius or diameter")
    if radius is None:
        radius = diameter / 2

    def f(p):
        return _length(p - center) - radius

    return f


@sdf3
def plane(normal=UP, point=ORIGIN):
    normal = _normalize(normal)

    def f(p):
        return np.dot(point - p, normal)

    return f


@sdf3
def slab(
    x0=None,
    y0=None,
    z0=None,
    x1=None,
    y1=None,
    z1=None,
    dx=None,
    dy=None,
    dz=None,
    k=None,
):
    # How to improve this if/None madness?
    if dx is not None:
        if x0 is None:
            x0 = -dx / 2
        if x1 is None:
            x1 = dx / 2
    if dy is not None:
        if y0 is None:
            y0 = -dy / 2
        if y1 is None:
            y1 = dy / 2
    if dz is not None:
        if z0 is None:
            z0 = -dz / 2
        if z1 is None:
            z1 = dz / 2
    fs = []
    if x0 is not None:
        fs.append(plane(X, (x0, 0, 0)))
    if x1 is not None:
        fs.append(plane(-X, (x1, 0, 0)))
    if y0 is not None:
        fs.append(plane(Y, (0, y0, 0)))
    if y1 is not None:
        fs.append(plane(-Y, (0, y1, 0)))
    if z0 is not None:
        fs.append(plane(Z, (0, 0, z0)))
    if z1 is not None:
        fs.append(plane(-Z, (0, 0, z1)))
    return intersection(*fs, k=k)


@sdf3
def box(size=1, center=ORIGIN, a=None, b=None):
    if a is not None and b is not None:
        a = np.array(a)
        b = np.array(b)
        size = b - a
        center = a + size / 2
        return box(size, center)
    size = np.array(size)

    def f(p):
        q = np.abs(p - center) - size / 2
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0)

    return f


@sdf3
def rounded_box(size, radius):
    size = np.array(size)

    def f(p):
        q = np.abs(p) - size / 2 + radius
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0) - radius

    return f


@sdf3
def wireframe_box(size, thickness):
    size = np.array(size)

    def g(a, b, c):
        return _length(_max(_vec(a, b, c), 0)) + _min(_max(a, _max(b, c)), 0)

    def f(p):
        p = np.abs(p) - size / 2 - thickness / 2
        q = np.abs(p + thickness / 2) - thickness / 2
        px, py, pz = p[:, 0], p[:, 1], p[:, 2]
        qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]
        return _min(_min(g(px, qy, qz), g(qx, py, qz)), g(qx, qy, pz))

    return f


@sdf3
def torus(r1, r2):
    def f(p):
        xy = p[:, [0, 1]]
        z = p[:, 2]
        a = _length(xy) - r1
        b = _length(_vec(a, z)) - r2
        return b

    return f


@sdf3
def capsule(a, b, radius=None, diameter=None):
    if (radius is not None) == (diameter is not None):
        raise ValueError(f"Specify either radius or diameter")
    if radius is None:
        radius = diameter / 2
    a = np.array(a)
    b = np.array(b)
    ba = b - a
    babadot = np.dot(ba, ba)

    r = radius if hasattr(radius, "__call__") else lambda h: radius

    def f(p):
        pa = p - a
        h = np.clip(np.dot(pa, ba) / babadot, 0, 1).reshape((-1, 1))
        return _length(pa - np.multiply(ba, h)) - r(h.reshape(-1))

    return f


def pieslice(angle, centered=False):
    """
    Make a pie slice starting at X axis, rotated ``angle`` in mathematically
    positive direction. Infinite in Z direction.

    Args:
        angle: the angle to use
        centered: center the slice at X axis
    """
    angle = angle % units("360°")
    if angle <= units("180°"):
        s = plane(Y) & plane(-Y).rotate(angle)
    else:
        s = plane(Y) | plane(-Y).rotate(angle)
    if centered:
        s = s.rotate(-angle / 2)
    return s


@sdf3
def cylinder(radius=None, diameter=None):
    if (radius is not None) == (diameter is not None):
        raise ValueError(f"Specify either radius or diameter")
    if radius is None:
        radius = diameter / 2

    def f(p):
        return _length(p[:, [0, 1]]) - radius

    return f


@sdf3
def capped_cylinder(a, b, radius=None, diameter=None):
    if (radius is not None) == (diameter is not None):
        raise ValueError(f"Specify either radius or diameter")
    if radius is None:
        radius = diameter / 2

    a = np.array(a)
    b = np.array(b)

    def f(p):
        ba = b - a
        pa = p - a
        baba = np.dot(ba, ba)
        paba = np.dot(pa, ba).reshape((-1, 1))
        x = _length(pa * baba - ba * paba) - radius * baba
        y = np.abs(paba - baba * 0.5) - baba * 0.5
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        x2 = x * x
        y2 = y * y * baba
        d = np.where(
            _max(x, y) < 0,
            -_min(x2, y2),
            np.where(x > 0, x2, 0) + np.where(y > 0, y2, 0),
        )
        return np.sign(d) * np.sqrt(np.abs(d)) / baba

    return f


@sdf3
def rounded_cylinder(ra, rb, h):
    def f(p):
        d = _vec(_length(p[:, [0, 1]]) - ra + rb, np.abs(p[:, 2]) - h / 2 + rb)
        return _min(_max(d[:, 0], d[:, 1]), 0) + _length(_max(d, 0)) - rb

    return f


@sdf3
def capped_cone(a, b, ra, rb):
    a = np.array(a)
    b = np.array(b)

    def f(p):
        rba = rb - ra
        baba = np.dot(b - a, b - a)
        papa = _dot(p - a, p - a)
        paba = np.dot(p - a, b - a) / baba
        x = np.sqrt(papa - paba * paba * baba)
        cax = _max(0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = np.clip((rba * (x - ra) + paba * baba) / k, 0, 1)
        cbx = x - ra - f * rba
        cby = paba - f
        s = np.where(np.logical_and(cbx < 0, cay < 0), -1, 1)
        return s * np.sqrt(
            _min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba)
        )

    return f


@sdf3
def rounded_cone(r1, r2, h):
    def f(p):
        q = _vec(_length(p[:, [0, 1]]), p[:, 2])
        b = (r1 - r2) / h
        a = np.sqrt(1 - b * b)
        k = np.dot(q, _vec(-b, a))
        c1 = _length(q) - r1
        c2 = _length(q - _vec(0, h)) - r2
        c3 = np.dot(q, _vec(a, b)) - r1
        return np.where(k < 0, c1, np.where(k > a * h, c2, c3))

    return f


@sdf3
def ellipsoid(size):
    size = np.array(size)

    def f(p):
        k0 = _length(p / size)
        k1 = _length(p / (size * size))
        return k0 * (k0 - 1) / k1

    return f


@sdf3
def pyramid(h):
    def f(p):
        a = np.abs(p[:, [0, 1]]) - 0.5
        w = a[:, 1] > a[:, 0]
        a[w] = a[:, [1, 0]][w]
        px = a[:, 0]
        py = p[:, 2]
        pz = a[:, 1]
        m2 = h * h + 0.25
        qx = pz
        qy = h * py - 0.5 * px
        qz = h * px + 0.5 * py
        s = _max(-qx, 0)
        t = np.clip((qy - 0.5 * pz) / (m2 + 0.25), 0, 1)
        a = m2 * (qx + s) ** 2 + qy * qy
        b = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2
        d2 = np.where(_min(qy, -qx * m2 - qy * 0.5) > 0, 0, _min(a, b))
        return np.sqrt((d2 + qz * qz) / m2) * np.sign(_max(qz, -py))

    return f


# Platonic Solids


@sdf3
def tetrahedron(r):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        return (_max(np.abs(x + y) - z, np.abs(x - y) + z) - r) / np.sqrt(3)

    return f


@sdf3
def octahedron(r):
    def f(p):
        return (np.sum(np.abs(p), axis=1) - r) * np.tan(np.radians(30))

    return f


@sdf3
def dodecahedron(r):
    x, y, z = _normalize(((1 + np.sqrt(5)) / 2, 1, 0))

    def f(p):
        p = np.abs(p / r)
        a = np.dot(p, (x, y, z))
        b = np.dot(p, (z, x, y))
        c = np.dot(p, (y, z, x))
        q = (_max(_max(a, b), c) - x) * r
        return q

    return f


@sdf3
def icosahedron(r):
    r *= 0.8506507174597755
    x, y, z = _normalize(((np.sqrt(5) + 3) / 2, 1, 0))
    w = np.sqrt(3) / 3

    def f(p):
        p = np.abs(p / r)
        a = np.dot(p, (x, y, z))
        b = np.dot(p, (z, x, y))
        c = np.dot(p, (y, z, x))
        d = np.dot(p, (w, w, w)) - x
        return _max(_max(_max(a, b), c) - x, d) * r

    return f


# Shapes


def lerp(x1, x2, t):
    return (1 - t) * x1 + t * x2


def bezier_via_lerp(p1, p2, p3, p4, t):
    t = np.array(t).reshape(-1, 1)
    p12 = lerp(p1, p2, t)
    p23 = lerp(p2, p3, t)
    p34 = lerp(p3, p4, t)
    p1223 = lerp(p12, p23, t)
    p2334 = lerp(p23, p34, t)
    return lerp(p1223, p2334, t)


@sdf3
def bezier(
    p1=ORIGIN,
    p2=10 * X,
    p3=10 * Y,
    p4=10 * Z,
    radius=None,
    diameter=None,
    steps=20,
    k=None,
):
    """
    Generate a single bezier curve :any:`capsule_chain` from four
    control points with a fixed or variable thickness

    Args:
        p1, p2, p3, p4 (point vectors): the control points. Segment will start at p1 and end at p4.
        radius,diameter (float or callable): either a fixed number as radius/diameter or a
            callable taking a number within [0;1] and returning a radius/diameter, e.g.
            the easing function :any:`ease.linear` or
            ``radius=ease.linear.between(10,2)`` for a linear transition
            between radii/diameters.
        k (float or None): handed to :any:`capsule_chain`
    """
    if (radius is not None) == (diameter is not None):
        raise ValueError(f"Specify either radius or diameter")
    if radius is None:
        radius = diameter / 2
    if isinstance(radius, (float, int)):
        radius = ease.constant(radius)
    points = bezier_via_lerp(p1, p2, p3, p4, (t := np.linspace(0, 1, steps)))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)

    # TODO: better steps taking curvature and changing radius into account
    t_eq = np.interp(
        np.arange(0, lengths.sum() + radius.mean / 4, radius.mean / 4),
        # np.linspace(0, lengths.sum(), steps),
        np.hstack([0, np.cumsum(lengths)]),
        t,
    )
    points = bezier_via_lerp(p1, p2, p3, p4, t_eq)
    return capsule_chain(points, radius=radius, k=k)


def capsule_chain(points, radius=None, diameter=None, k=0):
    if (radius is not None) == (diameter is not None):
        raise ValueError(f"Specify either radius or diameter")
    if radius is None:
        radius = diameter / 2
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumlengths = np.hstack([0, np.cumsum(lengths)])
    relcumlengths = cumlengths / lengths.sum()
    return union(
        *[
            capsule(
                p1,
                p2,
                radius=(radius[a:b] if isinstance(radius, ease.Easing) else radius),
            )
            for p1, p2, a, b in zip(
                points[0:-1], points[1:], relcumlengths[0:-1], relcumlengths[1:]
            )
        ],
        k=k,
    )


def Thread(
    pitch=5,
    diameter=20,
    offset=1,
    left=False,
):
    """
    An infinite thread
    """
    angleperz = -(2 * np.pi) / pitch
    if left:
        angleperz *= -1
    thread = (
        cylinder(swipediameter := diameter / 2 - offset)
        .translate(X * offset)
        .twist(angleperz)
    )
    thread.diameter = diameter
    thread.pitch = pitch
    thread.offset = offset
    thread.left = left
    return thread


def Screw(
    length=40,
    head_shape=None,
    head_height=10,
    k_tip=10,
    k_head=0,
    **threadkwargs,
):
    if head_shape is None:
        head_shape = d2.hexagon(30)
    if not (thread := threadkwargs.pop("thread", None)):
        thread = Thread(**threadkwargs)
    k_tip = np.clip(k_tip, 0, min(thread.diameter, length)) or None
    k_head = np.clip(k_head, 0, thread.diameter) or None
    head = head_shape.extrude(head_height).translate(-Z * head_height / 2)
    return head | (thread & slab(z0=0) & slab(z1=length).k(k_tip)).k(k_head)


def RegularPolygonColumn(n, r=1):
    ri = r * np.cos(np.pi / n)
    return intersection(
        *[slab(y0=-ri).rotate(a, Z) for a in np.arange(0, 2 * np.pi, 2 * np.pi / n)]
    )


# Positioning


@op3
def translate(other, offset):
    def f(p):
        return other(p - offset)

    return f


@op3
def scale(other, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = (x, y, z)
    m = min(x, min(y, z))

    def f(p):
        return other(p / s) * m

    return f


def rotation_matrix(angle, axis=Z):
    """
    Euler-Rodriguez Formula for arbitrary-axis rotation:

    https://en.m.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    try:
        angle = angle.to("radians").m
    except AttributeError:
        pass
    x, y, z = _normalize(axis)
    s = np.sin(angle)
    c = np.cos(angle)
    m = 1 - c
    # code moved from mfogleman's rotate() below
    return np.array(
        [
            [m * x * x + c, m * x * y + z * s, m * z * x - y * s],
            [m * x * y - z * s, m * y * y + c, m * y * z + x * s],
            [m * z * x + y * s, m * y * z - x * s, m * z * z + c],
        ]
    ).T
    # Alternative matrix construction (slightly slower than the above):
    # kx, ky, kz = _normalize(axis)
    # K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    # return np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    # Using np.linalg.matrix_power(K,2) is not faster


@op3
def rotate(other, angle, vector=Z):
    matrix = rotation_matrix(axis=vector, angle=angle)

    def f(p):
        # np.dot(p, matrix) actually rotates *backwards*,
        # (matrix @ p) or np.dot(matrix,p) would rotate
        # forwards but that doesn't work with the shapes here.
        # In this case, rotating backwards is actually what we want:
        # we want to inverse the rotation to look up the SDF there
        return other(np.dot(p, matrix))

    return f


@op3
def rotate_to(other, a, b):
    a = _normalize(np.array(a))
    b = _normalize(np.array(b))
    dot = np.dot(b, a)
    if dot == 1:
        return other
    if dot == -1:
        return rotate(other, np.pi, _perpendicular(a))
    angle = np.arccos(dot)
    v = _normalize(np.cross(b, a))
    return rotate(other, angle, v)


@op3
def orient(other, axis):
    # quick fix that probably is a problem in rotate_to()
    axis = axis * np.array([-1, -1, 1])
    return rotate_to(other, UP, axis)


@op3
def circular_array(other, count, offset=0):
    other = other.translate(X * offset)
    da = 2 * np.pi / count

    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        d = np.hypot(x, y)
        a = np.arctan2(y, x) % da
        d1 = other(_vec(np.cos(a - da) * d, np.sin(a - da) * d, z))
        d2 = other(_vec(np.cos(a) * d, np.sin(a) * d, z))
        return _min(d1, d2)

    return f


# Alterations


@op3
def elongate(other, size):
    def f(p):
        q = np.abs(p) - size
        x = q[:, 0].reshape((-1, 1))
        y = q[:, 1].reshape((-1, 1))
        z = q[:, 2].reshape((-1, 1))
        w = _min(_max(x, _max(y, z)), 0)
        return other(_max(q, 0)) + w

    return f


@op3
def twist(other, k):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        c = np.cos(k * z)
        s = np.sin(k * z)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))

    return f


@op3
def twist_between(sdf, a, b, e=ease.in_out_cubic(2 * np.pi)):
    """
    Twist an object between two control points

    Args:
        a, b (vectors): the two control points
        e (scalar function): the angle to rotate, will be called with
            values between 0 (at control point ``a``) and 1 (at control point
            ``b``).  Its result will be used as rotation angle in radians.
    """

    # unit vector from control point a to b
    ab = (ab := b - a) / (L := np.linalg.norm(ab))

    def f(p):
        # project current point onto control direction, clip and apply easing
        angle = e(np.clip((p - a) @ ab / L, 0, 1))
        # move to origin ”-a”, then rotate, then move back ”+a”
        # create many rotation matrices (along 3rd dim) for all angles
        matrix = rotation_matrix(axis=ab, angle=-angle)
        # apply rotation matrix to points moved back to origin
        # (this is a slow Python loop, I wonder how to optmize this 🤔)
        rotated = np.array([m @ p for p, m in zip((p - a), matrix)])
        # move rotated points back to where they were
        return sdf(rotated + a)

    return f


@op3
def bend(other, k):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        c = np.cos(k * x)
        s = np.sin(k * x)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))

    return f


@op3
def bend_linear(other, p0, p1, v, e=ease.linear):
    p0 = np.array(p0)
    p1 = np.array(p1)
    v = -np.array(v)
    ab = p1 - p0

    def f(p):
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return other(p + t * v)

    return f


@op3
def bend_radial(other, r0, r1, dz, e=ease.linear):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        r = np.hypot(x, y)
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        z = z - dz * e(t)
        return other(_vec(x, y, z))

    return f


@op3
def transition_linear(f0, f1, p0=-Z, p1=Z, e=ease.linear):
    p0 = np.array(p0)
    p1 = np.array(p1)
    ab = p1 - p0

    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1

    return f


@op3
def transition_radial(f0, f1, r0=0, r1=1, e=ease.linear):
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        r = np.hypot(p[:, 0], p[:, 1])
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1

    return f


@op3
def wrap_around(other, x0, x1, r=None, e=ease.linear):
    p0 = X * x0
    p1 = X * x1
    v = -Y
    if r is None:
        r = np.linalg.norm(p1 - p0) / (2 * np.pi)

    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        d = np.hypot(x, y) - r
        d = d.reshape((-1, 1))
        a = np.arctan2(y, x)
        t = (a + np.pi) / (2 * np.pi)
        t = e(t).reshape((-1, 1))
        q = p0 + (p1 - p0) * t + v * d
        q[:, 2] = z
        return other(q)

    return f


# 3D => 2D Operations


@op32
def slice(other):
    # TODO: support specifying a slice plane
    # TODO: probably a better way to do this
    s = slab(z0=-1e-9, z1=1e-9)
    a = other & s
    b = other.negate() & s

    def f(p):
        p = _vec(p[:, 0], p[:, 1], np.zeros(len(p)))
        A = a(p).reshape(-1)
        B = -b(p).reshape(-1)
        w = A <= 0
        A[w] = B[w]
        return A

    return f


# Common

union = op3(dn.union)
difference = op3(dn.difference)
intersection = op3(dn.intersection)
blend = op3(dn.blend)
negate = op3(dn.negate)
dilate = op3(dn.dilate)
erode = op3(dn.erode)
shell = op3(dn.shell)
repeat = op3(dn.repeat)
mirror = op3(dn.mirror)
modulate_between = op3(dn.modulate_between)
stretch = op3(dn.stretch)
shear = op3(dn.shear)
