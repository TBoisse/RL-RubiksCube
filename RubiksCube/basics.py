from enum import Enum

class RotationType(Enum):
    PRIME = 1
    DOUBLE = 2
    NORMAL = 3

class MoveType(Enum):
    UNKOWN = -1
    U = 0
    R = 1
    F = 2

def opposite_rotation(rotation_type : RotationType):
    if rotation_type == RotationType.NORMAL:
        return RotationType.PRIME
    if rotation_type == RotationType.PRIME:
        return RotationType.NORMAL
    return rotation_type
    