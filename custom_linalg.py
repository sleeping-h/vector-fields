import numpy as np
from numpy import linalg as la

def rotation_matrix_by_2_vec(a, b):
    unit_vector_a = a / np.linalg.norm(a)
    unit_vector_b = b / np.linalg.norm(b)
    cos_t = np.dot(unit_vector_a, unit_vector_b)
    sin_t = np.sqrt(1-cos_t**2)*np.sign(np.cross(unit_vector_a, unit_vector_b))
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
def rotation_matrix_by_angle(t):
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]])