from .vec2 import Vec2


class Car:
    def __init__(self, data):
        pos, self.angle, self.side, \
        (fw_pos_x, fw_pos_y, self.fw_angle), \
        (bw_pos_x, bw_pos_y, self.bw_angle) = data
        self.pos = Vec2(pos)
        self.fw_pos = Vec2(fw_pos_x, fw_pos_y)
        self.bw_pos = Vec2(bw_pos_x, bw_pos_y)


class TickStep:
    def __init__(self, data):
        self.my_car = Car(data.my_car)
        self.enemy_car = Car(data.enemy_car)
        self.deadline_pos = data.deadline_position


class NewMatchStep:
    def __init__(self, data):
        pass


