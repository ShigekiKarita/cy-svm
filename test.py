# coding: utf-8
import numpy
import svm_wrap
import matplotlib.pyplot as plt


def surface(X):
    h = 1e-1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                            numpy.arange(y_min, y_max, h))
    return xx, yy, numpy.c_[xx.ravel(), yy.ravel()]


def plot(model, X, Y):
    plt.cla()
    xx, yy, xy = surface(X)
    Z = model.predict(xy).reshape(xx.shape)
    levels = numpy.arange(numpy.min(Z), numpy.max(Z), 1e-2)
    cmap = plt.get_cmap("bwr")
    plt.contourf(xx, yy, Z, levels, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, cmap=cmap)


def rands(num=10, dim=2, dtype = numpy.float64):
    return numpy.random.randn(num, dim).astype(dtype)


def dataset():
    xs0 = rands()
    xs1 = rands().dot(numpy.array([[3.0, 0.0], [0.0, 1.2]])) + numpy.array([3.0, 2.0])
    ts0 = -numpy.ones(len(xs0))
    ts1 = numpy.ones(len(xs1))
    xs = numpy.concatenate([xs0, xs1])
    ts = numpy.concatenate([ts0, ts1])
    return xs, ts


def test():
    s = svm_wrap.SVMWrapper()
    xs, ts = dataset()
    s.fit(xs, ts)

    success = 0.0
    for x, t in zip(xs, ts):
        y = s.predict(x)
        print("expect %f, actual %f" % (t, y))
        if numpy.sign(y) == t:
            success += 1.0
    success /= len(xs)

    print("train accuracy: %f" % success)
    plot(s, xs, ts)
    plt.show()


# endless joy
while True:
    test()
