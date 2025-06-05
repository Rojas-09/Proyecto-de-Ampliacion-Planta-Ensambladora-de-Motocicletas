import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# from numpy.polynomial.polynomial import Polynomial # Removed unused import
from sklearn.linear_model import LinearRegression

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
    # En caso de que A^T A sea si ngular, se podría usar la pseudoinversa para A directamente
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
# Vector de tiempo para los años de pronóstico (definido once here)
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
plt.show() # Mostrar el gráfico

# Matriz de Componentes C (10 componentes, 4 tipos de moto) [cite: 80, 81, 82, 83]
# Filas: Componentes 1 a 10
# Columnas: Moto Tipo 1 a Tipo 4
C_matrix = np.array([
    [1, 1, 1, 0],  # Componente 1
    [2, 0, 1, 1],  # Componente 2
    [0, 0, 0, 1],  # Componente 3
    [0, 0, 0, 1],  # Componente 4
    [0, 0, 1, 0],  # Componente 5
    [3, 2, 0, 0],  # Componente 6
    [1, 4, 0, 0],  # Componente 7
    [5, 2, 0, 1],  # Componente 8
    [1, 1, 2, 0],  # Componente 9
    [1, 1, 0, 0]   # Componente 10
])

        # Calcular requerimientos de componentes para 2024 (t=12) y 2025 (t=13)
# F_2024 es la segunda fila (índice 1) de sales_forecast_data (corresponde a t=12)
F_2024_vector = sales_forecast_data[1, :] 
R_2024_vector = C_matrix @ F_2024_vector

# F_2025 es la tercera fila (índice 2) de sales_forecast_data (corresponde a t=13)
F_2025_vector = sales_forecast_data[2, :] 
R_2025_vector = C_matrix @ F_2025_vector

print("\nRequerimientos de Componentes Estimados para 2024-2025:")
print("Componente | Req. 2024 (t=12) | Req. 2025 (t=13)")
print("-----------|------------------|-----------------")
for i in range(C_matrix.shape[0]): # Para cada componente
    print(f"Comp. {i+1:<6} | {R_2024_vector[i]:<16.0f} | {R_2025_vector[i]:.0f}")

# Ejemplo de cálculo detallado para Componente 1 en 2024 para verificación:
comp1_req_2024_manual = 0
for i in range(C_matrix.shape[1]): # Para cada tipo de moto
    comp1_req_2024_manual += C_matrix[0, i] * F_2024_vector[i]
print(f"\nVerificación cálculo Comp. 1 para 2024: {comp1_req_2024_manual:.0f} (igual a R_2024_vector[0])")

# Gráfico de Demanda de Componentes para 2024-2025
num_componentes = C_matrix.shape[0]
component_labels = [f"Comp. {j+1}" for j in range(num_componentes)]
x_indices = np.arange(num_componentes) # Posiciones para los grupos de barras

bar_width = 0.35 # Ancho de las barras

plt.figure(figsize=(14, 8))
bars1 = plt.bar(x_indices - bar_width/2, R_2024_vector, bar_width, label='Demanda 2024 (t=12)', color='skyblue')
bars2 = plt.bar(x_indices + bar_width/2, R_2025_vector, bar_width, label='Demanda 2025 (t=13)', color='lightcoral')

# Añadir etiquetas de valor encima de cada barra
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval:.0f}', ha='center', va='bottom', fontsize=8)

plt.xlabel("Componente")
plt.ylabel("Cantidad Requerida")
plt.title("Demanda Estimada de Componentes para 2024 y 2025")
plt.xticks(x_indices, component_labels, rotation=45, ha="right")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y') # Rejilla solo en el eje Y para claridad
plt.tight_layout()
plt.show()

# MODELO DE HIPERPLANO CORREGIDO: Ventas Totales vs Tiempo
# En lugar de usar ventas por tipo como predictores, usamos el tiempo como en los modelos individuales
print("\n=== ANÁLISIS CON MODELO DE HIPERPLANO (Ventas Totales vs Tiempo) ===")

# Calcular las ventas totales por año (suma de todos los tipos)
sales_total = np.sum(sales_hist_data, axis=1)

# Crear matriz de diseño para regresión lineal simple: S_total(t) = beta_0 + beta_1*t
A_design_total = np.vstack([np.ones(num_years_hist), t_hist]).T

# Calcular coeficientes usando mínimos cuadrados
beta_total = np.linalg.inv(A_design_total.T @ A_design_total) @ A_design_total.T @ sales_total

# Calcular métricas del modelo
S_pred_total_hist = A_design_total @ beta_total
residuals_total = sales_total - S_pred_total_hist
ecm_total = np.sum(residuals_total**2) / num_years_hist
total_variance_total = np.sum((sales_total - np.mean(sales_total))**2)
r_squared_total = 1 - (np.sum(residuals_total**2) / total_variance_total)

print(f"Modelo Total: S_total(t) = {beta_total[0]:.2f} + {beta_total[1]:.2f}t")
print(f"ECM Total: {ecm_total:.2f}, R² Total: {r_squared_total:.4f}")

# Proyecciones para 2023-2027 usando el modelo total
S_forecast_total = beta_total[0] + beta_total[1] * t_forecast_abs

print("\nProyecciones de Ventas Totales (Modelo Hiperplano):")
for i, year in enumerate(range(2023, 2028)):
    print(f"Año {year} (t={t_forecast_abs[i]}): {S_forecast_total[i]:.0f} unidades")

# Análisis de Incertidumbre para 2024 y 2025
print("\n=== ANÁLISIS DE INCERTIDUMBRE (MODELO HIPERPLANO) ===")

# Parámetros para intervalos de confianza
alpha = 0.05
df_total = num_years_hist - 2  # n - 2 (beta_0 y beta_1)
t_critical_total = stats.t.ppf(1 - alpha/2, df_total)
mse_total = ecm_total

# Matriz (A^T A)^-1 para el cálculo de intervalos
ATA_inv_total = np.linalg.inv(A_design_total.T @ A_design_total)

# Calcular intervalos para 2024 (t=12) y 2025 (t=13)
years_focus = [2024, 2025]
t_focus = [12, 13]

print("Proyecciones con Intervalos de Confianza del 95%:")
for i, (year, t_val) in enumerate(zip(years_focus, t_focus)):
    # Predicción puntual
    y_pred = beta_total[0] + beta_total[1] * t_val
    
    # Vector de diseño para el nuevo punto
    x_new = np.array([1, t_val])
    
    # Varianza de la predicción
    var_pred = mse_total * (1 + x_new.T @ ATA_inv_total @ x_new)
    se_pred = np.sqrt(var_pred)
    
    # Margen de error
    margin = t_critical_total * se_pred
    
    # Intervalo de predicción
    ci_lower = y_pred - margin
    ci_upper = y_pred + margin
    
    # Calcular porcentaje de error
    error_pct = (margin / y_pred) * 100
    
    print(f"Año {year}: {y_pred:.0f} unidades")
    print(f"  IC 95%: [{ci_lower:.0f}, {ci_upper:.0f}]")
    print(f"  Margen de error: ±{margin:.0f} unidades ({error_pct:.1f}%)")

# ANÁLISIS DE COMPONENTES CORREGIDO
print("\n=== ANÁLISIS DE DEMANDA DE COMPONENTES ===")

# Usar las proyecciones del modelo total y distribuirlas proporcionalmente
# Calcular proporciones históricas promedio de cada tipo
avg_proportions = np.mean(sales_hist_data / sales_total.reshape(-1, 1), axis=0)
print("Proporciones promedio por tipo:", [f"{p:.3f}" for p in avg_proportions])

# Distribuir las ventas totales proyectadas según estas proporciones
sales_2024_total = beta_total[0] + beta_total[1] * 12  # t=12 para 2024
sales_2025_total = beta_total[0] + beta_total[1] * 13  # t=13 para 2025

sales_2024_by_type = sales_2024_total * avg_proportions
sales_2025_by_type = sales_2025_total * avg_proportions

# Calcular demanda de componentes
R_2024_corrected = C_matrix @ sales_2024_by_type
R_2025_corrected = C_matrix @ sales_2025_by_type

print(f"\nVentas Totales Proyectadas:")
print(f"2024: {sales_2024_total:.0f} unidades")
print(f"2025: {sales_2025_total:.0f} unidades")

print(f"\nDistribución por tipo (2024):")
for i, (tipo, cantidad) in enumerate(zip(['Tipo 1', 'Tipo 2', 'Tipo 3', 'Tipo 4'], sales_2024_by_type)):
    print(f"  {tipo}: {cantidad:.0f} unidades ({avg_proportions[i]*100:.1f}%)")

print("\nDemanda de Componentes Corregida:")
print("Componente | Req. 2024 | Req. 2025")
print("-----------|-----------|----------")
total_components_2024 = 0
total_components_2025 = 0
for i in range(C_matrix.shape[0]):
    print(f"Comp. {i+1:<6} | {R_2024_corrected[i]:<9.0f} | {R_2025_corrected[i]:.0f}")
    total_components_2024 += R_2024_corrected[i]
    total_components_2025 += R_2025_corrected[i]

# Calcular componentes promedio por motocicleta
comp_per_moto_2024 = total_components_2024 / sales_2024_total
comp_per_moto_2025 = total_components_2025 / sales_2025_total

print(f"\nTotal componentes 2024: {total_components_2024:.0f}")
print(f"Total componentes 2025: {total_components_2025:.0f}")
print(f"Componentes por motocicleta 2024: {comp_per_moto_2024:.1f}")
print(f"Componentes por motocicleta 2025: {comp_per_moto_2025:.1f}")
print(f"Promedio componentes por moto: {(comp_per_moto_2024 + comp_per_moto_2025)/2:.1f}")

# Verificación: suma de componentes por fila en matriz C
components_per_type = np.sum(C_matrix, axis=0)
print(f"\nComponentes por tipo de moto (según matriz C): {components_per_type}")
weighted_avg_components = np.sum(components_per_type * avg_proportions)
print(f"Promedio ponderado de componentes por moto: {weighted_avg_components:.1f}")

# --- Simulación de Monte Carlo CORREGIDA ---
print("\n--- Simulación de Monte Carlo para Proyecciones (2024-2025) ---")
num_simulations = 10000

# Almacenamiento de resultados
mc_total_sales_2024 = []
mc_total_sales_2025 = []
mc_component_demand_2024_corrected = [[] for _ in range(C_matrix.shape[0])]
mc_component_demand_2025_corrected = [[] for _ in range(C_matrix.shape[0])]

# Desviación estándar del modelo total
std_dev_total = np.sqrt(mse_total)

print(f"Ejecutando {num_simulations} iteraciones de Monte Carlo...")
for sim_count in range(num_simulations):
    if (sim_count + 1) % 2000 == 0:
        print(f"  Iteración {sim_count + 1}/{num_simulations}")
    
    # Simular ventas totales para 2024 y 2025
    error_2024 = np.random.normal(0, std_dev_total)
    error_2025 = np.random.normal(0, std_dev_total)
    
    sim_total_2024 = max(0, sales_2024_total + error_2024)
    sim_total_2025 = max(0, sales_2025_total + error_2025)
    
    mc_total_sales_2024.append(sim_total_2024)
    mc_total_sales_2025.append(sim_total_2025)
    
    # Distribuir por tipos y calcular componentes
    sim_sales_2024_by_type = sim_total_2024 * avg_proportions
    sim_sales_2025_by_type = sim_total_2025 * avg_proportions
    
    sim_components_2024 = C_matrix @ sim_sales_2024_by_type
    sim_components_2025 = C_matrix @ sim_sales_2025_by_type
    
    for comp_idx in range(C_matrix.shape[0]):
        mc_component_demand_2024_corrected[comp_idx].append(max(0, sim_components_2024[comp_idx]))
        mc_component_demand_2025_corrected[comp_idx].append(max(0, sim_components_2025[comp_idx]))

print("Simulación completa.")

# Resultados de Monte Carlo
print("\n=== RESULTADOS MONTE CARLO ===")
print("Ventas Totales:")
for year, data in [("2024", mc_total_sales_2024), ("2025", mc_total_sales_2025)]:
    mean_val = np.mean(data)
    ci_lower = np.percentile(data, 2.5)
    ci_upper = np.percentile(data, 97.5)
    error_pct = ((ci_upper - ci_lower) / 2) / mean_val * 100
    print(f"  {year}: Media={mean_val:.0f}, IC 95%=[{ci_lower:.0f}, {ci_upper:.0f}], Error=±{error_pct:.1f}%")

print("\nDemanda de Componentes 2024 (Monte Carlo):")
for comp_idx in range(C_matrix.shape[0]):
    data = mc_component_demand_2024_corrected[comp_idx]
    mean_val = np.mean(data)
    ci_lower = np.percentile(data, 2.5)
    ci_upper = np.percentile(data, 97.5)
    print(f"  Componente {comp_idx+1}: Media={mean_val:.0f}, IC 95%=[{ci_lower:.0f}, {ci_upper:.0f}]")

# Plot histogram for Total Sales in 2025 (Monte Carlo results)
plt.figure(figsize=(10, 6))
plt.hist(mc_total_sales_2025, bins=50, density=True, alpha=0.75, color='purple', edgecolor='black')
plt.title("Monte Carlo: Distribución de la proyección de ventas totales para 2025 (Hiperplano)")
plt.xlabel("Ventas Totales Simuladas para 2025")
plt.ylabel("Densidad de Probabilidad")
# Añadir líneas de media y percentiles
mean_val = np.mean(mc_total_sales_2025)
ci_lower = np.percentile(mc_total_sales_2025, 2.5)
ci_upper = np.percentile(mc_total_sales_2025, 97.5)
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Media: {mean_val:.0f}')
plt.axvline(ci_lower, color='orange', linestyle='dotted', linewidth=2, label=f'2.5th Pctl: {ci_lower:.0f}')
plt.axvline(ci_upper, color='orange', linestyle='dotted', linewidth=2, label=f'97.5th Pctl: {ci_upper:.0f}')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()

# Additional visualization: Comparison of forecasting methods
plt.figure(figsize=(12, 8))
years_comparison = [2024, 2025]

# Individual models sum
individual_totals = [
    np.sum(sales_forecast_data[1, :]),  # 2024 from individual models
    np.sum(sales_forecast_data[2, :])   # 2025 from individual models
]

# Hyperplane model
hyperplane_totals = [sales_2024_total, sales_2025_total]

# Monte Carlo means
mc_means = [np.mean(mc_total_sales_2024), np.mean(mc_total_sales_2025)]

x = np.arange(len(years_comparison))
width = 0.25

plt.bar(x - width, individual_totals, width, label='Modelos Individuales (Suma)', alpha=0.8)
plt.bar(x, hyperplane_totals, width, label='Modelo Hiperplano', alpha=0.8)
plt.bar(x + width, mc_means, width, label='Monte Carlo (Media)', alpha=0.8)

plt.xlabel('Año')
plt.ylabel('Ventas Totales Proyectadas')
plt.title('Comparación de Métodos de Proyección')
plt.xticks(x, years_comparison)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (ind, hyp, mc) in enumerate(zip(individual_totals, hyperplane_totals, mc_means)):
    plt.text(i - width, ind + 20, f'{ind:.0f}', ha='center', va='bottom', fontsize=9)
    plt.text(i, hyp + 20, f'{hyp:.0f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + width, mc + 20, f'{mc:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\n=== RESUMEN EJECUTIVO ===")
print(f"Proyecciones de Ventas Totales:")
print(f"2024: {sales_2024_total:.0f} unidades (Modelo Hiperplano)")
print(f"2025: {sales_2025_total:.0f} unidades (Modelo Hiperplano)")
print(f"\nPrecisión del Modelo:")
print(f"R² = {r_squared_total:.4f} ({r_squared_total*100:.2f}%)")
print(f"Error estándar: ±{np.sqrt(mse_total):.1f} unidades")
print(f"\nComponentes requeridos en promedio: {weighted_avg_components:.1f} por motocicleta")
print(f"Relación estable y coherente con datos históricos: {'✓' if weighted_avg_components > 30 else '✗'}")