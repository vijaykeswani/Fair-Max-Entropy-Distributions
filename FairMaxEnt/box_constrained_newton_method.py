from time import time

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse


def _QuadraticSolver():
    """Primal path following algorithm to minimize
        1/2 x^\top A x + b^\top x
    for l \leq x \leq u
    A is assumed to be PD
    """

    A = T.matrix('A')
    b = T.vector('b')
    u = T.vector('u')
    l = T.vector('l')
    eps = T.scalar('eps')
    # test values
    Ai = A.tag.test_value = np.array([[2, -1], [-1, 2]], dtype='float32')
    b.tag.test_value = np.array([-6, 2], dtype='float32')
    u.tag.test_value = np.array([3, 1], dtype='float32')
    l.tag.test_value = np.array([0, -1], dtype='float32')
    eps.tag.test_value = 1e-8
    # answer is [3, 0.5]

    m = b.size
    # initialize the variables
    R = T.max(u - l)
    d = T.dot(A, u + l) / 2 + b

    nu = 2 ** 0.5 / (3 * R * T.sum(d ** 2) ** 0.5)
    x = (l + u) / 2.

    NU = (2. * m + (20 * m * R ** 2 * T.sum(A ** 2) ** 0.5 + 8 * m) ** 0.5 / 3.) ** 2 / eps ** 2
    alpha = 1 + 1 / (1 + 12 * (2 * m) ** 0.5)

    cutoff = eps / (T.sum(T.abs_(b)) + T.sum(T.abs_(A)) * T.max(T.maximum(T.abs_(u), T.abs_(l))))

    t = T.log(NU / nu) / T.log(alpha)
    t = ifelse(T.le(t, 0.), np.int32(0), T.cast(T.ceil(t), 'int32'))

    def inner(x, nu, A, b, alpha, u, l, cutoff):
        gradient = nu * (T.dot(A, x) + b) - 1 / (x - l) - 1 / (x - u)
        Hessian = nu * A + T.diag((1 / (x - l) ** 2 + 1 / (x - u) ** 2))
        newtonStep = - T.dot(T.nlinalg.pinv(Hessian), gradient)
        x2 = x + newtonStep
        nu2 = nu * alpha

        # if every coordinates near the boundary and closer than x, then early terminate
        du = u - x
        du2 = u - x2
        dl = x - l
        dl2 = x2 - l

        d = T.switch(T.ge(du, du2), du2, dl2)

        condition = T.lt(T.max(T.abs_(d)), cutoff)

        # if a vairalbe reachs to boundary due to finite precision
        # revert to old value
        x2 = T.switch(T.le(x2, l), x, x2)
        x2 = T.switch(T.ge(x2, u), x, x2)

        return (x2, nu2), theano.scan_module.until(condition)

    outputs, updates = theano.scan(inner, outputs_info=[x, nu], n_steps=t, non_sequences=[A, b, alpha, u, l, cutoff])

    return theano.function(inputs=[A, b, u, l, eps], outputs=[outputs[0][-1], t], updates=updates,
                           allow_input_downcast=True)


def _ProjectedGradientDescend():
    A = T.matrix('A')
    b = T.vector('b')
    u = T.vector('u')
    l = T.vector('l')
    eps = T.scalar('eps')
    # test values
    Ai = A.tag.test_value = np.array([[2, -1], [-1, 2]], dtype='float32')
    b.tag.test_value = np.array([-6, 2], dtype='float32')
    u.tag.test_value = np.array([3, 1], dtype='float32')
    l.tag.test_value = np.array([0, -1], dtype='float32')
    eps.tag.test_value = 1e-8
    # answer is [3, 0.5]

    x = -T.dot(T.nlinalg.pinv(A), b)
    x = T.switch(T.le(x, l), l, x)
    x = T.switch(T.ge(x, u), u, x)

    cutoff = eps / (T.sum(T.abs_(b)) + T.sum(T.abs_(A)) * T.max(T.maximum(T.abs_(u), T.abs_(l))))
    L = T.max(T.sum(T.abs_(A), axis=1))

    def inner(x, A, b, u, l, L, eps):
        gradient = T.dot(A, x) + b
        x2 = x - gradient / L
        x2 = T.switch(T.le(x2, l), l, x2)
        x2 = T.switch(T.ge(x2, u), u, x2)

        d = T.max(T.abs_(x - x2))
        condition = T.le(T.max(d), eps)

        return (x2, d), theano.scan_module.until(condition)

    outputs, updates = theano.scan(inner, outputs_info=[x, None], n_steps=10240, non_sequences=[A, b, u, l, L, cutoff])

    return theano.function(inputs=[A, b, u, l, eps], outputs=[outputs[0][-1], outputs[1][-1], outputs[0].shape[0]],
                           updates=updates, allow_input_downcast=True)


ProjectedGradientDescend = _ProjectedGradientDescend()


def BoxConstrainedNewtonMethod(x, f, s, R, eps, *args, **kwargs):
    """
        box constrained newton method to minimize a s-second order robust function f
        in the \ell_\infty ball of radiues R
        gradient: is a function that takes x and returns the gradient at x as vector
        Hessian: is a function that takes x and returns the Hessian at x as matrix
        (y-z) <= a(x-z) + b and x-y <= c, then y-z<= (b+ac)/(1-a)
    """
    earlyCut = kwargs.get("earlyCut", None)
    if earlyCut is None:
        earlyCut = -float('inf')
    alpha = 8 * np.exp(2) * s * R
    eps2 = eps / alpha

    t = int(-alpha * np.log(eps))
    alpha = 1 - 1 / alpha

    gamma = eps

    r = 1. / s

    x = np.array(x)

    m = x.size

    # 1/e^2
    ie2 = np.exp(-2)
    ie = np.exp(-1)
    one = r * np.ones_like(x)
    old_value = float('inf')
    # print("Starting box-constrained newton method, it will run for", t, "many iterations")
    times = []
    iterations = []
    Start = time()
    for it in range(1, t):
        u = np.minimum(one, R - x)
        l = np.maximum(-one, -R - x)
        new_value, gradient, Hessian = f(x, *args)
        if old_value - new_value <= eps:
            break
        if new_value < earlyCut:
            break
        old_value = new_value
        b = gradient
        A = Hessian * ie
        start = time()
        # y, _it = QuadraticSolver(A, b, u, l, eps2)
        y, d, _it = ProjectedGradientDescend(A, b, u, l, eps2)
        end = time()
        x += ie2 * y
        times.append(end - start)
        if it % 1000 == 0:
            print(it, t, new_value, 'runned for', _it, 'iteration and took', end - start, 'seconds, total elapsed time',
                  end - Start, 'size of x:', np.max(np.abs(x)), 'PGD last diff:', d)
    End = time()
    totalTime = End - Start
    totalIteration = it

    info = {
        'times': times,
        'iterations': iterations,
        'totalTime': totalTime,
        'totalIteration': totalIteration
    }
    return x, new_value, info