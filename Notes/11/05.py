import numpy as np
import sympy as sym

theta = sym.symbols('theta')
R = sym.Matrix([[sym.cos(theta), -sym.sin(theta)], [sym.sin(theta), sym.cos(theta)]])

print(f"R transpose times R: {R.T @ R}")
det = sym.det(R)
print(f"determinant of R: {det}")