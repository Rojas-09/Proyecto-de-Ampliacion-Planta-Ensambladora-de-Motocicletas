import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Datos históricos de ventas (2013-2022) [cite: 62, 63, 64]
# Columnas: Tipo 1, Tipo 2, Tipo 3, Tipo 4
sales_hist_data = np.array([
    [172, 89,  18,  28],  # 2013 (t=1)
    [185, 116, 49,  33],  # 2014 (t=2)
    [202, 155, 98,  49],  # 2015 (t=3)
    [225, 188, 96,  44],  # 2016 (t=4)
    [252, 200, 148, 59],  # 2017 (t=5)
    [286, 199, 173, 72],  # 2018 (t=6)
    [316, 240, 204, 70],  # 2019 (t=7)
    [342, 245, 235, 96],  # 2020 (t=8)
    [371, 280, 266, 140], # 2021 (t=9)
    [402, 302, 297, 250]  # 2022 (t=10)
])

num_years_hist = sales_hist_data.shape[0]
# Vector de tiempo para datos históricos (t=1 para 2013, ..., t=10 para 2022)
t_hist = np.arange(1, num_years_hist + 1)

# Construcción de la matriz de diseño A para la regresión lineal S(t) = beta_0 + beta_1*t
# Cada fila de A es [1, t_i]
A_design = np.vstack([np.ones(num_years_hist), t_hist]).T # [cite: 46, 58]

# Cálculo de la parte (A^T A)^-1 A^T de la solución de mínimos cuadrados
# beta = (A^T A)^-1 A^T S
try:
    A_T_A_inv_A_T = np.linalg.inv(A_design.T @ A_design) @ A_design.T # [cite: 37, 59, 65]
except np.linalg.LinAlgError:
    # En caso de que A^T A sea singular, se podría usar la pseudoinversa para A directamente
    # A_plus = np.linalg.pinv(A_design)
    # Esta ruta no es necesaria aquí porque A_design tiene columnas linealmente independientes.
    print("Error: A^T A es singular. Se necesitaría la pseudoinversa de A.")
    exit()


beta_coeffs = [] # Lista para almacenar los coeficientes (beta_0, beta_1) para cada tipo
models_str = []  # Lista para almacenar las ecuaciones de los modelos
ecm_values = []  # Lista para almacenar los valores de ECM

print("Coeficientes de Regresion (beta_0, beta_1), Error Cuadratico Medio (ECM) y Calcular el Coeficiente de Determinación (R²):")
# Iterar sobre cada tipo de motocicleta (cada columna en sales_hist_data)
for i in range(sales_hist_data.shape[1]):
    S_tipo = sales_hist_data[:, i]  # Vector de ventas históricas para el tipo actual
    
    # Calcular coeficientes beta usando la fórmula de mínimos cuadrados
    beta = A_T_A_inv_A_T @ S_tipo
    beta_coeffs.append(beta)
    models_str.append(f"S{i+1}(t) = {beta[0]:.2f} + {beta[1]:.2f}t")
    
    # Calcular el Error Cuadrático Medio (ECM/MSE)
    S_pred_hist = A_design @ beta  # Valores predichos para el período histórico
    residuals = S_tipo - S_pred_hist  # Diferencias entre observado y predicho
    ecm = np.sum(residuals**2) / num_years_hist  # Suma de cuadrados de residuos / m
    ecm_values.append(ecm)
    
    # Calcular el Coeficiente de Determinación (R²)
    total_variance = np.sum((S_tipo - np.mean(S_tipo))**2)  # Suma total^2
    r_squared = 1 - (np.sum(residuals**2) / total_variance)  # Fórmula de R²
    
    print(f"Moto Tipo {i+1}: beta_0 = {beta[0]:.2f}, beta_1 = {beta[1]:.2f}, Modelo: {models_str[i]}, ECM = {ecm:.2f}, R² = {r_squared:.2f}")

# Pronósticos para los próximos 5 años: 2023-2027 (t=11 a t=15) [cite: 3, 70]
num_years_forecast = 5
# Vector de tiempo para los años de pronóstico
t_forecast_abs = np.arange(num_years_hist + 1, num_years_hist + 1 + num_years_forecast) 
# Matriz para almacenar los datos de ventas pronosticados
sales_forecast_data = np.zeros((num_years_forecast, sales_hist_data.shape[1]))

for i in range(sales_hist_data.shape[1]): # Para cada tipo de moto
    beta = beta_coeffs[i] # Coeficientes del modelo para este tipo
    # Calcular ventas pronosticadas usando la ecuación del modelo
    S_pred_forecast = beta[0] + beta[1] * t_forecast_abs
    sales_forecast_data[:, i] = np.round(S_pred_forecast) # Redondear al entero más cercano

print("\nVentas Pronosticadas (unidades) para 2023-2027:")
header = "Año (t) | " + " | ".join([f"Tipo {i+1}" for i in range(sales_hist_data.shape[1])])
print(header)
print("-" * len(header))
for j in range(num_years_forecast):
    year_label = 2022 + (j + 1) # Año calendario
    row_str = f"{year_label} ({t_forecast_abs[j]:<2}) | "
    row_str += " | ".join([f"{sales_forecast_data[j,k]:<6.0f}" for k in range(sales_hist_data.shape[1])])
    print(row_str)

# Visualización Gráfica Combinada (similar a la Figura \ref{fig:ventas_combinada} del LaTeX)
plt.figure(figsize=(12, 7)) # Tamaño de la figura
colors = ['blue', 'red', 'green', 'orange'] # Colores para cada tipo de moto
markers_hist = ['o', 's', '^', 'p'] # Marcadores para datos históricos
type_labels = ['Tipo 1', 'Tipo 2', 'Tipo 3', 'Tipo 4'] # Etiquetas para leyenda

# Vector de tiempo completo (histórico + pronóstico) para graficar líneas de modelo
t_all = np.arange(1, num_years_hist + 1 + num_years_forecast)

for i in range(sales_hist_data.shape[1]): # Para cada tipo de moto
    beta = beta_coeffs[i] # Coeficientes del modelo
    
    # Graficar datos históricos
    plt.plot(t_hist, sales_hist_data[:, i], marker=markers_hist[i], linestyle='None', color=colors[i], label=f'{type_labels[i]} Histórico')
    
    # Graficar línea de regresión (modelo ajustado) extendida sobre todo el período
    S_reg_line = beta[0] + beta[1] * t_all
    plt.plot(t_all, S_reg_line, color=colors[i], linestyle='-', label=f'{type_labels[i]} Modelo ({models_str[i]})')
    
    # Graficar datos pronosticados
    plt.plot(t_forecast_abs, sales_forecast_data[:, i], marker='d', linestyle='None', markersize=7, color=colors[i], markeredgecolor='black', label=f'{type_labels[i]} Pronóstico')

# Configuración de los ejes y etiquetas del gráfico
xtick_locs = np.arange(1, num_years_hist + 1 + num_years_forecast + 1) # Ubicaciones de las marcas en eje X
xtick_labels_year = np.arange(2013, 2013 + len(xtick_locs)) # Etiquetas de año para eje X
plt.xticks(xtick_locs, xtick_labels_year, rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel("Año (t, donde t=1 corresponde a 2013)", fontsize=11)
plt.ylabel("Unidades Vendidas", fontsize=11)
plt.title("Ventas Históricas y Pronosticadas por Tipo de Motocicleta", fontsize=13)
plt.legend(fontsize='small', ncol=2) # Mostrar leyenda
plt.grid(True, linestyle='--', alpha=0.7) # Añadir rejilla
plt.axvline(x=num_years_hist + 0.5, color='gray', linestyle=':', linewidth=1.5) # Línea vertical separando histórico de pronóstico
plt.text(num_years_hist + 0.6, plt.ylim()[1]*0.9, 'Pronóstico \u2192', fontsize=10, color='dimgray')
plt.text(num_years_hist - 0.4, plt.ylim()[1]*0.9, '\u2190 Histórico', fontsize=10, ha='right', color='dimgray')

plt.tight_layout() # Ajustar layout para evitar superposiciones
# plt.savefig("ventas_pronostico_combinado_python.png", dpi=300) # Para guardar la figura
plt.show() # Mostrar el gráfico

# Matriz de Componentes C (10 componentes, 4 tipos de moto) [cite: 80, 81, 82, 83]
# Filas: Componentes 1 a 10
# Columnas: Moto Tipo 1 a Tipo 4
C_matrix = np.array([
    [1, 1, 1, 0],  # Componente 1
    [0, 2, 1, 1],  # Componente 2
    [0, 0, 0, 1],  # Componente 3
    [0, 0, 0, 1],  # Componente 4
    [0, 0, 1, 0],  # Componente 5
    [3, 2, 0, 0],  # Componente 6
    [1, 4, 0, 0],  # Componente 7
    [5, 2, 0, 1],  # Componente 8
    [1, 1, 2, 0],  # Componente 9
    [1, 1, 0, 0]   # Componente 10
])

# Calcular requerimientos de componentes para 2023 (t=11, primer año de pronóstico)
# F_2023 es la primera fila (índice 0) de sales_forecast_data
F_2023_vector = sales_forecast_data[0, :] # Ventas pronosticadas para 2023 para los 4 tipos
R_2023_vector = C_matrix @ F_2023_vector # Producto matricial C * F_2023 [cite: 61, 85]

# Calcular requerimientos de componentes para 2025 (t=13, tercer año de pronóstico)
# F_2025 es la tercera fila (índice 2) de sales_forecast_data
F_2025_vector = sales_forecast_data[2, :] # Ventas pronosticadas para 2025 para los 4 tipos
R_2025_vector = C_matrix @ F_2025_vector

print("\nRequerimientos de Componentes Estimados:")
print("Componente | Req. 2023 (t=11) | Req. 2025 (t=13)")
print("-----------|------------------|-----------------")
for i in range(C_matrix.shape[0]): # Para cada componente
    print(f"Comp. {i+1:<6} | {R_2023_vector[i]:<16.0f} | {R_2025_vector[i]:.0f}")

# Ejemplo de cálculo detallado para Componente 1 en 2023 para verificación:
comp1_req_2023_manual = 0
for i in range(C_matrix.shape[1]): # Para cada tipo de moto
    comp1_req_2023_manual += C_matrix[0, i] * F_2023_vector[i]
print(f"\nVerificación cálculo Comp. 1 para 2023: {comp1_req_2023_manual:.0f} (igual a R_2023_vector[0])")

# Calcular las ventas totales por año
sales_total = np.sum(sales_hist_data, axis=1)  # Suma de las ventas de todos los tipos por año

# Ajustar el modelo de regresión lineal para las ventas totales
A_design_total = np.vstack([np.ones(num_years_hist), t_hist]).T
beta_total = np.linalg.inv(A_design_total.T @ A_design_total) @ A_design_total.T @ sales_total

# Calcular el ECM y R² para las ventas totales
S_pred_total = A_design_total @ beta_total
residuals_total = sales_total - S_pred_total
ecm_total = np.sum(residuals_total**2) / num_years_hist
total_variance_total = np.sum((sales_total - np.mean(sales_total))**2)
r_squared_total = 1 - (np.sum(residuals_total**2) / total_variance_total)

print(f"Modelo Total: S_total(t) = {beta_total[0]:.2f} + {beta_total[1]:.2f}t")
print(f"ECM Total: {ecm_total:.2f}, R² Total: {r_squared_total:.2f}")

# Pronósticos para los próximos 5 años (2023-2027)
t_forecast_abs = np.arange(num_years_hist + 1, num_years_hist + 6)
S_forecast_total = beta_total[0] + beta_total[1] * t_forecast_abs

print("\nPronósticos de Ventas Totales (2023-2027):")
for i, year in enumerate(range(2023, 2028)):
    print(f"Año {year}: {S_forecast_total[i]:.0f} unidades")

# Ajustar un modelo polinomial de grado 2
p = Polynomial.fit(t_hist, sales_total, deg=2)
print(f"Modelo Polinomial: {p}")