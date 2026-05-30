# Hyman, J. M. (1983). Accurate monotonicity preserving cubic interpolation.
#     SIAM Journal on Scientific and Statistical Computing, 4(4), 645–654.
# original code from: https://github.com/antdvid/MonotonicCubicInterpolation

import numpy as np


class MonoSpline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = self.y.size
        self.h = self.x[1:] - self.x[:-1]
        self.m = (self.y[1:] - self.y[:-1]) / self.h
        self.a = self.y[:]
        self.b = self.compute_b()
        self.c = (3 * self.m - self.b[1:] - 2 * self.b[:-1]) / self.h
        self.d = (self.b[1:] + self.b[:-1] - 2 * self.m) / (self.h * self.h)

    def compute_b(self):
        b = np.empty(self.n)
        for i in range(1, self.n - 1):
            is_mono = self.m[i - 1] * self.m[i] > 0
            if is_mono:
                n = max(self.m[i - 1], self.m[i]) + 2 * min(self.m[i - 1], self.m[i])
                b[i] = 3 * self.m[i - 1] * self.m[i] / n
            else:
                b[i] = 0

            if is_mono and self.m[i] > 0:
                b[i] = min(max(0, b[i]), 3 * min(self.m[i - 1], self.m[i]))
            elif is_mono and self.m[i] < 0:
                b[i] = max(min(0, b[i]), 3 * max(self.m[i - 1], self.m[i]))

        n1 = self.h[0] + self.h[1]
        b[0] = ((2 * self.h[0] + self.h[1]) * self.m[0] - self.h[0] * self.m[1]) / n1
        n1 = (2 * self.h[self.n - 2] + self.h[self.n - 3]) * self.m[self.n - 2]
        n2 = self.h[self.n - 2] * self.m[self.n - 3]
        n3 = self.h[self.n - 2] + self.h[self.n - 3]
        b[self.n - 1] = (n1 - n2) / n3
        return b

    def _find_i(self, tau):
        i = np.where(tau >= self.x)[0]
        if i.size == 0:
            i = 0
        else:
            i = i[-1]
        i = min(i, self.n - 2)
        return i

    def evaluate(self, t_intrp):
        ans = []
        for tau in t_intrp:
            i = self._find_i(tau)
            t1 = tau - self.x[i]
            n2 = self.c[i] * np.power(t1, 2.0)
            n3 = self.d[i] * np.power(t1, 3.0)
            res = self.a[i] + self.b[i] * t1 + n2 + n3  # original curve r(t)
            ans.append(res)
        return ans

    def evaluate_derivative(self, t_intrp):
        ans = []
        if not hasattr(t_intrp, "__len__"):
            t_intrp = np.array([t_intrp])
        for tau in t_intrp:
            i = self._find_i(tau)
            t1 = tau - self.x[i]
            n2 = self.c[i] * np.power(t1, 2.0)
            res = self.b[i] + 2 * self.c[i] * t1 + 3 * self.d[i] * n2
            ans.append(res)
        if len(ans) == 1:
            return ans[0]
        else:
            return ans

    def evaluate_forward(self, t_intrp):
        ans = []
        for tau in t_intrp:
            i = self._find_i(tau)
            t1 = tau - self.x[i]
            t2 = 2 * tau - self.x[i]
            t3 = 3 * tau - self.x[i]
            t4 = 4 * tau - self.x[i]
            n2 = np.power(t1, 2.0)
            # d(xy)/dx:
            res = self.a[i] + self.b[i] * t2 + self.c[i] * t1 * t3 + self.d[i] * n2 * t4
            ans.append(res)
        return ans
