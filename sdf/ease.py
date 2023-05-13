import numpy as np
from dataclasses import dataclass
from typing import Callable
import itertools
import functools


@dataclass
class UnitFunction:
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
                itertools.chain(
                    map(repr, args), ("{k}={v!r}" for k, v in kwargs.items())
                )
            )
            newfun.__name__ = f"{self.f.__name__}.{decorated_fun.__name__}({arglist})"
            return type(self)(f=newfun, name=newfun.__name__)

        return wrapper

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
    def scale(self, factor):
        """
        Scale function by ``factor``
        """
        return lambda t: factor * self.f(t)

    @modifier
    def add(self, offset):
        """
        Add ``offset`` to function
        """
        return lambda t: self.f(t) + offset

    @modifier
    def between(self, a, b):
        """
        Arrange so ``f(0)==a`` and ``f(1)==b``.
        """
        raise NotImplementedError("Just a teaser that between() will be there 😉")

    @classmethod
    def function(cls, decorated_fun):
        return cls(f=decorated_fun, name=decorated_fun.__name__)

    def plot(self, *others, ax=None):
        import matplotlib.pyplot as plt  # lazy import for speed

        if ax is None:
            fig, ax = plt.subplots()
        t = np.linspace(0, 1, 100)
        print(others)
        funs = list(others or [])
        if isinstance(self, UnitFunction):
            funs.insert(0, self)
        for f in funs:
            ax.plot(t, f(t), label=getattr(f, "name", getattr(f, "__name__", str(f))))
        ax.legend()
        plt.show()

    def __call__(self, t):
        return self.f(t)


class Easing(UnitFunction):
    """
    A function mapping the interval [0;1] to [0;1]
    """

    pass


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
    return -1 * (2 ** (10 * u) * np.sin((u - k / 4) * (2 * np.pi) / k))


@Easing.function
def out_elastic(t, k=0.5):
    return 2 ** (-10 * t) * np.sin((t - k / 4) * (2 * np.pi / k)) + 1


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
    a = np.zeros(len(t))
    b = a + 1
    return np.where(t < 1, a, b)


@Easing.function
def out_square(t):
    a = np.zeros(len(t))
    b = a + 1
    return np.where(t > 0, b, a)


@Easing.function
def in_out_square(t):
    a = np.zeros(len(t))
    b = a + 1
    return np.where(t < 0.5, a, b)


def _main():
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams["axes.prop_cycle"] *= cycler(
        linestyle=["solid", "dashed", "dotted"], linewidth=[1, 2, 3]
    )
    Easing.plot(
        linear,
        in_quad,
        out_quad,
        in_out_quad,
        in_cubic,
        out_cubic,
        in_out_cubic,
        in_quart,
        out_quart,
        in_out_quart,
        in_quint,
        out_quint,
        in_out_quint,
        in_sine,
        out_sine,
        in_out_sine,
        in_expo,
        out_expo,
        in_out_expo,
        in_circ,
        out_circ,
        in_out_circ,
        in_elastic,
        out_elastic,
        in_out_elastic,
        in_back,
        out_back,
        in_out_back,
        in_bounce,
        out_bounce,
        in_out_bounce,
        in_square,
        out_square,
        in_out_square,
        in_sine.symmetric,
        in_out_sine.symmetric.scale(-0.6),
        linear.symmetric.scale(-0.7),
        in_out_sine.scale(-0.6).symmetric,
        out_sine.scale(-0.6).reverse.symmetric.scale(2),
        out_bounce.add(-0.5),
    )


if __name__ == "__main__":
    _main()
