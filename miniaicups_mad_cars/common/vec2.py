import math


class Vec2:
    def __init__(self, x=0.0, y=0.0):
        if isinstance(x, list) or isinstance(x, tuple):
            x, y = x
        self.x = x
        self.y = y

    @property
    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def __add__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        else:
            return Vec2(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x - other.x, self.y - other.y)
        else:
            return Vec2(self.x - other, self.y - other)

    def __mul__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x * other.x, self.y * other.y)
        else:
            return Vec2(self.x * other, self.y * other)

    def __div__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x / other.x, self.y / other.y)
        else:
            return Vec2(self.x / other, self.y / other)

    def __eq__(self, other):
        if isinstance(other, Vec2):
            return self.x == other.x and self.y == other.y
        else:
            return False

    def __pow__(self, power, modulo=None):
        if isinstance(power, Vec2):
            return Vec2(self.x ** power.x, self.y ** power.y)
        else:
            return Vec2(self.x ** power, self.y ** power)

    def __iter__(self):
        return iter((self.x, self.y))