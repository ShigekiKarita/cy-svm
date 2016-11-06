# coding: utf-8
import numpy
import svm_wrap
import matplotlib.pyplot as plt


class DecisionBoundary:
    def __init__(self, num=20, dim=2):
        self.num = num
        self.dim = dim
        self.dtype = numpy.float64
        self.generate_dataset()

    def surface(self, X):
        h = 1e-1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                                numpy.arange(y_min, y_max, h))
        return xx, yy, numpy.c_[xx.ravel(), yy.ravel()]

    def plot(self, model):
        plt.cla()
        xx, yy, xy = self.surface(self.xs)
        Z = model.decision_function(xy).reshape(xx.shape)
        eps = 1.0
        a, b = numpy.min(Z) - eps, numpy.max(Z) + eps
        resolution = 1e2
        levels = numpy.arange(a, b, (b - a) / resolution)
        cmap = plt.get_cmap("bwr")
        plt.contourf(xx, yy, Z, levels, cmap=cmap)
        plt.scatter(self.xs[:, 0], self.xs[:, 1], s=100, c=self.ts, cmap=cmap)

    def rands(self):
        return numpy.random.randn(self.num, self.dim).astype(self.dtype)

    def generate_dataset(self):
        xs0 = self.rands()
        xs1 = self.rands().dot(numpy.array([[3.0, 0.0], [0.0, 1.2]])) + numpy.array([3.0, 2.0])
        ts0 = -numpy.ones(len(xs0))
        ts1 = numpy.ones(len(xs1))
        self.xs = numpy.concatenate([xs0, xs1])
        self.ts = numpy.concatenate([ts0, ts1])

    def fit(self, model, **config):
        model.fit(self.xs, self.ts, **config)
        return model

    def accuracy(self, model):
        success = 0.0
        ys = model.decision_function(self.xs)
        for y, t in zip(ys, self.ts):
            print("expect %f, actual %f" % (t, y))
            if numpy.sign(y) == t:
                success += 1.0
        success /= len(self.xs)
        print("train accuracy: %f" % success)
        return success

    def test(self, **svm_config):
        s = svm_wrap.SVMWrapper()
        self.fit(s, eps=1e-3, loop_limit=1e3, **svm_config)

        acc = self.accuracy(s)
        self.plot(s)
        return acc * 100

    def main(self):
        # harder margin
        c = 1e6
        plt.subplot(2, 2, 1)
        a = self.test(c=c, is_linear=True)
        plt.title("Linear (c={:.1E} acc={}%)".format(c, a))
        plt.subplot(2, 2, 2)
        a = self.test(c=c, is_linear=False)
        plt.title("RBF (c={:.1e} acc={}%)".format(c, a))

        # softer margin
        c = 1e-2
        plt.subplot(2, 2, 3)
        a = self.test(c=c, is_linear=True)
        plt.title("Linear (c={:.1e} acc={}%)".format(c, a))
        plt.subplot(2, 2, 4)
        a = self.test(c=c, is_linear=False)
        plt.title("RBF (c={:.1e} acc={}%)".format(c, a))
        plt.suptitle("Support Vector Machines", fontsize=20)
        plt.show()


if __name__ == '__main__':
    DecisionBoundary().main()
