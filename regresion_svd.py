import numpy as np

# Datos históricos de ventas (2013-2022)
sales_hist_data = np.array([
    [172, 89,  18,  28],
    [185, 116, 49,  33],
    [202, 155, 98,  49],
    [225, 188, 96,  44],
    [252, 200, 148, 59],
    [286, 199, 173, 72],
    [316, 240, 204, 70],
    [342, 245, 235, 96],
    [371, 280, 266, 140],
    [402, 302, 297, 250]
])

num_years_hist = sales_hist_data.shape[0]
t_hist = np.arange(1, num_years_hist + 1)
A_design = np.vstack([np.ones(num_years_hist), t_hist]).T

print("Coeficientes de regresión usando SVD:")
for i in range(sales_hist_data.shape[1]):
    S_tipo = sales_hist_data[:, i]
    # Descomposición SVD de la matriz de diseño
    U, s, Vt = np.linalg.svd(A_design, full_matrices=False)
    # Calcular la pseudoinversa de A_design
    S_inv = np.diag(1 / s)
    A_pinv = Vt.T @ S_inv @ U.T
    # Coeficientes beta usando la pseudoinversa
    beta_svd = A_pinv @ S_tipo
    print(f"Tipo {i+1}: beta_0 = {beta_svd[0]:.2f}, beta_1 = {beta_svd[1]:.2f}")

# Puedes comparar estos coeficientes con los obtenidos por el método tradicional.