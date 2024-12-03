import sympy as sym


w1, w2, w3 = sym.symbols('w1 w2 w3')
v1, v2, v3 = sym.symbols('v1 v2 v3')

# skew symmetric matrix
w_hat = sym.Matrix([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
v = sym.Matrix([[v1], [v2], [v3]])

# cross product
cross = w_hat @ v
print(f"cross product: {cross}")