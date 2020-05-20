import itertools
import numpy as np
from matplotlib import pyplot
from scipy.optimize import fsolve

EPSILON = 1e-8

NUMBER_OF_ITERATIONS = 100

# root index to its first approximations
root_approximations = {
    0: 1,
    1: 3
}


def jacobian(p2, p3, x1, x2, x3, x4):
    return [
        [
            - p2 * np.e ** (x2 / (x2 / 1000 + 1)) - 1,
            p2 * np.e ** (x2 / (x2 / 1000 + 1)) * (x2 / (1000 * (x2 / 1000 + 1) ** 2) - 1 / (x2 / 1000 + 1)) * (x1 - 1),
            0,
            0
        ],
        [
            -12 * p2 * np.e ** (x2 / (x2 / 1000 + 1)),
            12 * p2 * np.e ** (x2 / (x2 / 1000 + 1)) * (x2 / (1000 * (x2 / 1000 + 1) ** 2) - 1 / (x2 / 1000 + 1)) * (x1 - 1) - 3,
            0,
            0
        ],
        [
            p2 / p3,
            0, -(p2 * (p3 * np.e ** (x4 / (x4 / 1000 + 1)) + 1)) / p3,
            p2 * np.e ** (x4 / (x4 / 1000 + 1)) * (x4 / (1000 * (x4 / 1000 + 1) ** 2) - 1 / (x4 / 1000 + 1)) * (x3 - 1)
        ],
        [
            0,
            p2 / p3,
            -12 * p2 * np.e ** (x4 / (x4 / 1000 + 1)),
            (p2 * (12 * p3 * np.e ** (x4 / (x4 / 1000 + 1)) * (x4 / (1000 * (x4 / 1000 + 1) ** 2) - 1 / (x4 / 1000 + 1)) * (x3 - 1) - 3)) / p3
        ]
    ]


x4s = [[], []]
p3s = [[], []]
p2s = [[], []]

for x2int in range(1, NUMBER_OF_ITERATIONS + 1):
    x2 = float(x2int) / NUMBER_OF_ITERATIONS
    x1 = x2 / 4
    p2 = x2 / (4 - x2) / (np.e ** (x2 / (x2 / 1000 + 1)))


    def sum_of_bottom_right_jacobian_part(x4):
        x3 = x4 / 4 + x2 / 6
        expo = np.e ** (x4 / (x4 / 1000 + 1))
        p3 = (x3 - x1) / (1 - x3) / expo

        jac = jacobian(p2, p3, x1, x2, x3, x4)
        sum = jac[2][2] + jac[3][3]
        return sum


    for root_index in range(len(root_approximations)):
        approximation = root_approximations[root_index]
        x4 = fsolve(sum_of_bottom_right_jacobian_part, approximation)[0]
        x3 = x4 / 4 + x2 / 6
        exp = np.e ** (x4 / (x4 / 1000 + 1))
        p3 = (x3 - x1) / (1 - x3) / exp

        jac = jacobian(p2, p3, x1, x2, x3, x4)
        j = [[jac[0][0], jac[0][1]], [jac[1][0], jac[1][1]]]
        jj = [[jac[2][2], jac[2][3]], [jac[3][2], jac[3][3]]]

        eigenvaluesj, _ = np.linalg.eig(j)
        eigenvaluesjj, _ = np.linalg.eig(jj)
        eigenvalues = list(eigenvaluesj) + list(eigenvaluesjj)

        has_conjugates = len(list(filter(
            lambda elems: ((elems[0] + elems[1]).imag == 0) and abs(elems[0].real) < EPSILON and abs(elems[1].real) < EPSILON,
            itertools.product(eigenvalues, repeat=2)
        )))
        if has_conjugates:
            x4s[root_index].append(x4)
            p2s[root_index].append(p2)
            p3s[root_index].append(p3)

        print(
            f"{x2:.3f}, {x4:.4f}, {p2:.4f}, {p3:.4f}, ",
            ", ".join(f"{eigenvalue:.4f}" for eigenvalue in eigenvalues),
            ", taken" if has_conjugates else "dropped"
        )

pyplot.subplot(211)
pyplot.xlabel("p2")
pyplot.ylabel("p3")

pyplot.vlines([0.134592, 0.201643, 0], 0, 0.3)
for i in range(len(root_approximations)):
    pyplot.plot(p2s[i], p3s[i])



x1 = 0.05
x2 = 0.2
p2 = 0.043

x4 = 3.29
x3 = x4 / 4 + x2 / 6
exp = np.e ** (x4 / (x4 / 1000 + 1))
p3 = (x3 - x1) / (1 - x3) / exp
print(p3)
jac = jacobian(p2, p3, x1, x2, x3, x4)
eigenvalues, _ = np.linalg.eig(jac)
print(", ".join(f"{eigenvalue:.4f}" for eigenvalue in eigenvalues))

pyplot.plot(p2, p3, '.')

x4 = 1.64
x3 = x4 / 4 + x2 / 6
exp = np.e ** (x4 / (x4 / 1000 + 1))
p3 = (x3 - x1) / (1 - x3) / exp
print(p3)
jac = jacobian(p2, p3, x1, x2, x3, x4)
eigenvalues, _ = np.linalg.eig(jac)
print(", ".join(f"{eigenvalue:.4f}" for eigenvalue in eigenvalues))

pyplot.plot(p2, p3, '.')

pyplot.show()
