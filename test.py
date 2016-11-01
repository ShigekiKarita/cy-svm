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
    print("predict surface")
    levels = numpy.arange(numpy.min(Z), numpy.max(Z), 1e-2)
    plt.contourf(xx, yy, Z, levels, cmap=plt.get_cmap("bwr"))
    print("plot surface")

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, cmap=plt.get_cmap("bwr"))

s = svm_wrap.SVMWrapper()
dtype = numpy.float64
num = 100
dim = 2
xs0 = numpy.random.randn(num, dim).astype(dtype)
cov = numpy.array([[3.0, 0.0], [0.0, 1.2]])
xs1 = numpy.random.randn(num, dim).astype(dtype).dot(cov) + numpy.array([3.0, 2.0])
xs = numpy.concatenate([xs0, xs1])
ts0 = -numpy.ones(len(xs0))
ts1 = numpy.ones(len(xs1))
ts = numpy.concatenate([ts0, ts1])
s.fit(xs, ts)


success = 0
for x, t in zip(xs, ts):
    y = s.predict(x)
    print("expect %f, actual %f" % (t, y))
    if numpy.sign(y) == t:
        success += 1

print("train accuracy: " + str(success / num / 2))
plot(s, xs, ts)
plt.show()