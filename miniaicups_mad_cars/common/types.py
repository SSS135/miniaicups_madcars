from .vec2 import Vec2
from .dict_ex import DictEx


class Car:
    def __init__(self, data):
        pos, self.angle, self.side, \
        (fw_pos_x, fw_pos_y, self.fw_angle), \
        (bw_pos_x, bw_pos_y, self.bw_angle) = data
        self.pos = Vec2(pos)
        self.fw_pos = Vec2(fw_pos_x, fw_pos_y)
        self.bw_pos = Vec2(bw_pos_x, bw_pos_y)


class ProtoMap:
    def __init__(self, data):
        self.external_id = data.external_id
        self.segments = data.segments


class ProtoCar:
    def __init__(self, data):
        self.car_body_poly = data.car_body_poly
        self.rear_wheel_radius = data.rear_wheel_radius
        self.front_wheel_radius = data.front_wheel_radius
        self.button_poly = data.button_poly
        self.external_id = data.external_id
        self.car_body_mass = data.car_body_mass
        self.car_body_friction = data.car_body_friction
        self.car_body_elasticity = data.car_body_elasticity
        self.max_speed = data.max_speed
        self.max_angular_speed = data.max_angular_speed
        self.drive = data.drive
        self.rear_wheel_mass = data.rear_wheel_mass
        self.rear_wheel_position = data.rear_wheel_position
        self.rear_wheel_friction = data.rear_wheel_friction
        self.rear_wheel_elasticity = data.rear_wheel_elasticity
        self.rear_wheel_joint = data.rear_wheel_joint
        self.rear_wheel_damp_position = data.rear_wheel_damp_position
        self.rear_wheel_damp_length = data.rear_wheel_damp_length
        self.rear_wheel_damp_stiffness = data.rear_wheel_damp_stiffness
        self.rear_wheel_damp_damping = data.rear_wheel_damp_damping
        self.front_wheel_mass = data.front_wheel_mass
        self.front_wheel_position = data.front_wheel_position
        self.front_wheel_friction = data.front_wheel_friction
        self.front_wheel_elasticity = data.front_wheel_elasticity
        self.front_wheel_joint = data.front_wheel_joint
        self.front_wheel_damp_position = data.front_wheel_damp_position
        self.front_wheel_damp_length = data.front_wheel_damp_length
        self.front_wheel_damp_stiffness = data.front_wheel_damp_stiffness
        self.front_wheel_damp_damping = data.front_wheel_damp_damping


class TickStep:
    def __init__(self, data):
        self.my_car = Car(data.my_car)
        self.enemy_car = Car(data.enemy_car)
        self.deadline_pos = data.deadline_position


class NewMatchStep:
    def __init__(self, data):
        self.my_lives = data.my_lives
        self.enemy_lives = data.enemy_lives
        self.proto_map = ProtoMap(DictEx(data.proto_map))
        self.proto_car = ProtoCar(DictEx(data.proto_car))


