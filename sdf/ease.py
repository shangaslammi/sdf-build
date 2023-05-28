# system modules
from dataclasses import dataclass
from typing import Callable
import itertools
import functools
import warnings

# external modules
import numpy as np
import scipy.optimize


@dataclass
@functools.total_ordering
class Extremum:
    """
    Container for min and max in Easing
    """

    pos: float
    value: float

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value


@dataclass
class Easing:
    """
    A function defined on the interval [0;1]
    """

    f: Callable[float, float]
    name: str

    def modifier(decorated_fun):
        @functools.wraps(decorated_fun)
        def wrapper(self, *args, **kwargs):
            newfun = decorated_fun(self, *args, **kwargs)
            arglist = ",".join(
                itertools.chain(map(str, args), (f"{k}={v}" for k, v in kwargs.items()))
            )
            newfun.__name__ = f"{self.f.__name__}.{decorated_fun.__name__}({arglist})"
            return type(self)(f=newfun, name=newfun.__name__)

        return wrapper

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    @modifier
    def reverse(self):
        """
        Revert the function so it goes the other way round (starts at the end)
        """
        return lambda t: self.f(1 - t)

    @property
    @modifier
    def symmetric(self):
        """
        Mirror and squash function to make it symmetric
        """
        return lambda t: self.f(-2 * (np.abs(t - 0.5) - 0.5))

    @modifier
    def clip(self, min=None, max=None):
        """
        Clip function at low and/or high values
        """
        if min is None and max is None:
            min = 0
            max = 1
        return lambda t: np.clip(self.f(t), min, max)

    @modifier
    def clip_input(self, min=None, max=None):
        """
        Clip input parameter, i.e. extrapolate constantly outside the interval.
        """
        if min is None and max is None:
            min = 0
            max = 1
        return lambda t: self.f(np.clip(t, min, max))

    @property
    @modifier
    def clipped(self):
        """
        Clipped parameter and result to [0;1]
        """
        return lambda t: np.clip(self(np.clip(t, 0, 1)), 0, 1)

    @modifier
    def append(self, other, e=None):
        """
        Append another easing function and squish both into the [0;1] interval
        """
        if e is None:
            e = in_out_square

        def f(t):
            mix = e(t)
            return self.f(t * 2) * (1 - mix) + other((t - 0.5) * 2) * mix

        return f

    @modifier
    def prepend(self, other, e=None):
        """
        Prepend another easing function and squish both into the [0;1] interval
        """
        if e is None:
            e = in_out_square

        def f(t):
            mix = e(t)
            return other(t * 2) * (1 - mix) + self.f((t - 0.5) * 2) * mix

        return f

    @modifier
    def shift(self, offset):
        """
        Shift function on x-axis into positive direction by ``offset``.
        """
        return lambda t: self.f(t - offset)

    @modifier
    def multiply(self, factor):
        """
        Scale function by ``factor``
        """
        if isinstance(factor, Easing):
            return lambda t: self(t) * factor(t)
        else:
            return lambda t: factor * self.f(t)

    @modifier
    def add(self, offset):
        """
        Add ``offset`` to function
        """
        if isinstance(offset, Easing):
            return lambda t: self(t) + offset(t)
        else:
            return lambda t: self.f(t) + offset

    def __add__(self, offset):
        return self.add(offset)

    def __sub__(self, offset):
        return self.add(-offset)

    def __mul__(self, factor):
        return self.multiply(factor)

    def __rmul__(self, factor):
        return self.multiply(factor)

    def __neg__(self):
        return self.multiply(-1)

    def __truediv__(self, factor):
        return self.multiply(1 / factor)

    def __or__(self, other):
        return self.transition(other)

    def __rshift__(self, offset):
        return self.shift(offset)

    def __lshift__(self, offset):
        return self.shift(-offset)

    def __getitem__(self, index):
        if isinstance(index, Easing):
            return self.chain(index)
        if isinstance(index, slice):
            return self.zoom(
                0 if index.start is None else index.start,
                1 if index.stop is None else index.stop,
            )
        else:
            raise ValueError(
                f"{index = } has to be slice of floats or an easing function"
            )

    @modifier
    def chain(self, f=None):
        """
        Feed parameter through the given function before evaluating this function.
        """
        if f is None:
            f = self.f
        return lambda t: self.f(f(t))

    @modifier
    def zoom(self, left, right=None):
        """
        Arrange so that the interval [left;right] is moved into [0;1]
        If only one argument is given, zoom in/out by moving edges that far.
        """
        if left is not None and right is None:
            if left >= 0.5:
                raise ValueError(
                    f"{left = } is > 0.5 which doesn't make sense (bounds would cross)"
                )
            left = left
            right = 1 - left
        if left >= right:
            raise ValueError(f"{right = } bound must be greater than {left = }")
        return self.chain(linear.between(left, right)).f

    @modifier
    def between(self, left=0, right=1, e=None):
        """
        Arrange so ``f(0)==a`` and ``f(1)==b``.
        """
        f0, f1 = self.f(np.array([0, 1]))
        la = f0 - left
        lb = f1 - right
        if e is None:
            e = linear

        def f(t):
            t_ = e(t)
            return self.f(t) - (la * (1 - t_)) - lb * t_

        return f

    @modifier
    def transition(self, other, e=None):
        """
        Transiton from one easing to another
        """
        if e is None:
            e = linear

        def f(t):
            t_ = e(t)
            return self.f(t) * (1 - t_) + other(t) * t_

        return f

    @classmethod
    def function(cls, decorated_fun):
        return cls(f=decorated_fun, name=decorated_fun.__name__)

    def plot(self, *others, ax=None):
        import matplotlib.pyplot as plt  # lazy import for speed

        if ax is None:
            fig, ax_ = plt.subplots()
        else:
            ax_ = ax
        t = np.linspace(0, 1, 1000)
        funs = list(others or [])
        if isinstance(self, Easing):
            funs.insert(0, self)
        for f in funs:
            ax_.plot(t, f(t), label=getattr(f, "name", getattr(f, "__name__", str(f))))
        ax_.legend(ncol=int(np.ceil(len(ax_.get_lines()) / 10)))
        if ax is None:
            plt.show()
        return ax_

    @functools.cached_property
    def min(self):
        v = self.f(t := np.linspace(0, 1, 1000))
        approxmin = Extremum(pos=t[i := np.argmin(v)], value=v[i])
        opt = scipy.optimize.minimize(self, x0=[approxmin.pos], bounds=[(0, 1)])
        optmin = Extremum(pos=opt.x[0], value=opt.fun)
        return min(approxmin, optmin)

    @functools.cached_property
    def max(self):
        """
        Determine the maximum value
        """
        v = self.f(t := np.linspace(0, 1, 1000))
        approxmax = Extremum(pos=t[i := np.argmax(v)], value=v[i])
        opt = scipy.optimize.minimize(-self, x0=[approxmax.pos], bounds=[(0, 1)])
        optmax = Extremum(pos=opt.x[0], value=-opt.fun)
        return max(approxmax, optmax)

    @functools.cached_property
    def mean(self):
        return np.mean(self.f(np.linspace(0, 1, 1000)))

    def __call__(self, t):
        return self.f(t)


@Easing.function
def linear(t):
    return t


@Easing.function
def in_quad(t):
    return t * t


@Easing.function
def out_quad(t):
    return -t * (t - 2)


@Easing.function
def in_out_quad(t):
    u = 2 * t - 1
    a = 2 * t * t
    b = -0.5 * (u * (u - 2) - 1)
    return np.where(t < 0.5, a, b)


@Easing.function
def in_cubic(t):
    return t * t * t


@Easing.function
def out_cubic(t):
    u = t - 1
    return u * u * u + 1


@Easing.function
def in_out_cubic(t):
    u = t * 2
    v = u - 2
    a = 0.5 * u * u * u
    b = 0.5 * (v * v * v + 2)
    return np.where(u < 1, a, b)


@Easing.function
def in_quart(t):
    return t * t * t * t


@Easing.function
def out_quart(t):
    u = t - 1
    return -(u * u * u * u - 1)


@Easing.function
def in_out_quart(t):
    u = t * 2
    v = u - 2
    a = 0.5 * u * u * u * u
    b = -0.5 * (v * v * v * v - 2)
    return np.where(u < 1, a, b)


@Easing.function
def in_quint(t):
    return t * t * t * t * t


@Easing.function
def out_quint(t):
    u = t - 1
    return u * u * u * u * u + 1


@Easing.function
def in_out_quint(t):
    u = t * 2
    v = u - 2
    a = 0.5 * u * u * u * u * u
    b = 0.5 * (v * v * v * v * v + 2)
    return np.where(u < 1, a, b)


@Easing.function
def in_sine(t):
    return -np.cos(t * np.pi / 2) + 1


@Easing.function
def out_sine(t):
    return np.sin(t * np.pi / 2)


@Easing.function
def in_out_sine(t):
    return -0.5 * (np.cos(np.pi * t) - 1)


@Easing.function
def in_expo(t):
    a = np.zeros(len(t))
    b = 2 ** (10 * (t - 1))
    return np.where(t == 0, a, b)


@Easing.function
def out_expo(t):
    a = np.zeros(len(t)) + 1
    b = 1 - 2 ** (-10 * t)
    return np.where(t == 1, a, b)


@Easing.function
def in_out_expo(t):
    zero = np.zeros(len(t))
    one = zero + 1
    a = 0.5 * 2 ** (20 * t - 10)
    b = 1 - 0.5 * 2 ** (-20 * t + 10)
    return np.where(t == 0, zero, np.where(t == 1, one, np.where(t < 0.5, a, b)))


@Easing.function
def in_circ(t):
    return -1 * (np.sqrt(1 - t * t) - 1)


@Easing.function
def out_circ(t):
    u = t - 1
    return np.sqrt(1 - u * u)


@Easing.function
def in_out_circ(t):
    u = t * 2
    v = u - 2
    a = -0.5 * (np.sqrt(1 - u * u) - 1)
    b = 0.5 * (np.sqrt(1 - v * v) + 1)
    return np.where(u < 1, a, b)


@Easing.function
def in_elastic(t, k=0.5):
    u = t - 1
    return -1 * (2 ** (10.0 * u) * np.sin((u - k / 4) * (2 * np.pi) / k))


@Easing.function
def out_elastic(t, k=0.5):
    return 2 ** (-10.0 * t) * np.sin((t - k / 4) * (2 * np.pi / k)) + 1


@Easing.function
def in_out_elastic(t, k=0.5):
    u = t * 2
    v = u - 1
    a = -0.5 * (2 ** (10 * v) * np.sin((v - k / 4) * 2 * np.pi / k))
    b = 2 ** (-10 * v) * np.sin((v - k / 4) * 2 * np.pi / k) * 0.5 + 1
    return np.where(u < 1, a, b)


@Easing.function
def in_back(t):
    k = 1.70158
    return t * t * ((k + 1) * t - k)


@Easing.function
def out_back(t):
    k = 1.70158
    u = t - 1
    return u * u * ((k + 1) * u + k) + 1


@Easing.function
def in_out_back(t):
    k = 1.70158 * 1.525
    u = t * 2
    v = u - 2
    a = 0.5 * (u * u * ((k + 1) * u - k))
    b = 0.5 * (v * v * ((k + 1) * v + k) + 2)
    return np.where(u < 1, a, b)


@Easing.function
def in_bounce(t):
    return 1 - out_bounce(1 - t)


@Easing.function
def out_bounce(t):
    a = (121 * t * t) / 16
    b = (363 / 40 * t * t) - (99 / 10 * t) + 17 / 5
    c = (4356 / 361 * t * t) - (35442 / 1805 * t) + 16061 / 1805
    d = (54 / 5 * t * t) - (513 / 25 * t) + 268 / 25
    return np.where(t < 4 / 11, a, np.where(t < 8 / 11, b, np.where(t < 9 / 10, c, d)))


@Easing.function
def in_out_bounce(t):
    a = in_bounce(2 * t) * 0.5
    b = out_bounce(2 * t - 1) * 0.5 + 0.5
    return np.where(t < 0.5, a, b)


@Easing.function
def in_square(t):
    return np.heaviside(t - 1, 0)


@Easing.function
def out_square(t):
    return np.heaviside(t + 1, 0)


@Easing.function
def in_out_square(t):
    return np.heaviside(t - 0.5, 0)


def constant(x):
    return Easing(f=lambda t: np.full_like(t, x), name=f"constant({x})")


zero = constant(0)
one = constant(1)


@Easing.function
def smoothstep(t):
    t = np.clip(t, 0, 1)
    return 3 * t * t - 2 * t * t * t


def _main():
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams["axes.prop_cycle"] *= cycler(
        linestyle=["solid", "dashed", "dotted"], linewidth=[1, 2, 3]
    )
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["legend.fontsize"] = "small"
    LOCALS = globals()
    print(f"{LOCALS = }")
    fig, axes = plt.subplots(nrows=2)
    Easing.plot(
        *sorted((obj for n, obj in LOCALS.items() if isinstance(obj, Easing)), key=str),
        ax=axes[0],
    )
    Easing.plot(
        in_sine.symmetric,
        in_out_sine.symmetric.multiply(-0.6),
        linear.symmetric.multiply(-0.7),
        in_out_sine.multiply(-0.6).symmetric,
        out_sine.multiply(-0.6).reverse.symmetric.multiply(2),
        out_bounce.add(-0.5),
        ax=axes[1],
    )
    axes[0].set_title("Standard")
    axes[1].set_title("Derived")
    plt.show()


if __name__ == "__main__":
    _main()
