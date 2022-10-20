import math
from scipy import constants
import matplotlib.pyplot as plt

'''
Simulates a torsion spring and plots its torque and deflection after a given impulse.
'''
class Spring:
    def __init__(self, k, l=0.17, m=6.0):
        self.k = k
        self.l = l
        self.m = m
        self.alphaplt = []
        self.forceplt = []

    def impulse(self, vt, at_angle, stept):
        time = 0
        alpha = at_angle
        i = self.m * self.l
        vk = 2 * math.cos(90-alpha) * vt * self.l
        while time <= 100:
            tm = 0.5 * self.m * constants.g * self.l * math.sin(alpha)
            tf = self.k * vk
            tg = tm - tf
            vk = vk + tg / i * stept
            alpha = alpha + vk * stept
            time += stept
            self.alphaplt.append(alpha)
            self.forceplt.append(tg)

        plt.plot(self.alphaplt)
        plt.show()
        plt.plot(self.forceplt)
        plt.show()


if __name__ == "__main__":
    s = Spring(0.75)
    s.impulse(0, 60, 0.01)
