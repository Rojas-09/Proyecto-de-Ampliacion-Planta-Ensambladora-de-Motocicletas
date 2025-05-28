import numpy as np
import matplotlib.pyplot as plt
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
# t_forecast_abs = np.arange(num_years_hist + 1, num_years_hist + 6) # This redundant definition is removed
S_forecast_total = beta_total[0] + beta_total[1] * t_forecast_abs # Uses the t_forecast_abs defined earlier

print("\nPronósticos de Ventas Totales (2023-2027):")
for i, year in enumerate(range(2023, 2028)):
    print(f"Año {year}: {S_forecast_total[i]:.0f} unidades")

# Modelo: sales_total ≈ x0*Tipo1 + x1*Tipo2 + x2*Tipo3 + x3*Tipo4 + intercepto

X = sales_hist_data  # Variables independientes: ventas por tipo de moto
y = sales_total      # Variable dependiente: ventas totales por año

reg = LinearRegression(fit_intercept=True)
reg.fit(X, y)
y_pred = reg.predict(X)

# Métricas del ajuste global
ecm_global = np.mean((y - y_pred) ** 2)
r2_global = reg.score(X, y)

print("\nAjuste Global por Año (hiperplano de mínimos cuadrados):")
print("Coeficientes por tipo de moto:", reg.coef_)
print("Intercepto:", reg.intercept_)
print(f"ECM Global: {ecm_global:.2f}, R² Global: {r2_global:.2f}")

# Gráfica del ajuste global
plt.figure(figsize=(10,6))
plt.plot(range(2013, 2013+len(y)), y, 'o-', label='Ventas Totales Observadas')
plt.plot(range(2013, 2013+len(y_pred)), y_pred, 's--', label='Ajuste Global (Predicho)')
plt.xlabel("Año")
plt.ylabel("Ventas Totales")
plt.title("Ajuste Global por Año (Hiperplano de Mínimos Cuadrados)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Análisis de Incertidumbre en Proyecciones 2024-2025 (Modelo por Hiperplano)
print("\nAnálisis de Incertidumbre en Proyecciones 2023-2025 (Modelo por Hiperplano):") # Título opcionalmente actualizado

from scipy import stats

# Años y valores t para las proyecciones
years_uncertainty = [2023, 2024, 2025] # Incluye 2023
t_values_uncertainty = [11, 12, 13] # Incluye t=11 para 2023

# Proyectar ventas por tipo para 2023, 2024 y 2025
sales_type_proj_uncertainty = []
for t_val in t_values_uncertainty:
    current_year_sales = []
    for k in range(sales_hist_data.shape[1]): # Para cada tipo de moto
        beta_k = beta_coeffs[k] # Coeficientes del modelo para el tipo k
        proj_sale_k = beta_k[0] + beta_k[1] * t_val
        current_year_sales.append(proj_sale_k)
    sales_type_proj_uncertainty.append(np.array(current_year_sales))

# Preparar para el cálculo del intervalo de predicción del hiperplano
# X_hyper son las ventas históricas por tipo, y es sales_total
# El modelo es reg = LinearRegression().fit(X, y)
# ECM global ya está calculado como ecm_global
mse_hyper = ecm_global
n_hyper = num_years_hist # Número de observaciones históricas
p_hyper = sales_hist_data.shape[1] # Número de predictores (tipos de moto)
df_hyper = n_hyper - p_hyper - 1 # Grados de libertad: n - (k_predictores + 1_intercepto)

# Valor t crítico para un intervalo de confianza del 95%
alpha = 0.05
t_critical_hyper = stats.t.ppf(1 - alpha / 2, df_hyper)

# Matriz de diseño usada para ajustar el modelo de hiperplano (con intercepto)
X_design_hyper = np.hstack([np.ones((n_hyper, 1)), sales_hist_data])
try:
    XTX_inv_hyper = np.linalg.inv(X_design_hyper.T @ X_design_hyper)
except np.linalg.LinAlgError:
    print("Error: La matriz X^T X para el modelo de hiperplano es singular.")
    XTX_inv_hyper = None # O usar pseudoinversa si es necesario y se maneja el error

if XTX_inv_hyper is not None:
    for i, year in enumerate(years_uncertainty):
        # Ventas proyectadas por tipo para el año actual (X_new para el hiperplano)
        x_new_features_hyper = sales_type_proj_uncertainty[i]
        
        # Predicción de ventas totales con el modelo de hiperplano
        total_sales_pred_hyper = reg.predict(x_new_features_hyper.reshape(1, -1))[0]
        
        # Construir el vector x_new con el intercepto para la fórmula del intervalo
        x_new_with_intercept_hyper = np.concatenate(([1], x_new_features_hyper))
        
        # Varianza de la predicción: MSE * (1 + x_new^T * (X^T X)^-1 * x_new)
        # Asegurarse de que x_new_with_intercept_hyper es un vector columna para el producto matricial final
        var_pred_hyper = mse_hyper * (1 + x_new_with_intercept_hyper.reshape(1, -1) @ XTX_inv_hyper @ x_new_with_intercept_hyper.reshape(-1, 1))
        se_pred_hyper = np.sqrt(var_pred_hyper[0,0]) # Extraer el escalar
        
        # Margen de error
        margin_of_error_hyper = t_critical_hyper * se_pred_hyper
        
        # Intervalo de predicción
        pi_lower_hyper = total_sales_pred_hyper - margin_of_error_hyper
        pi_upper_hyper = total_sales_pred_hyper + margin_of_error_hyper
        
        print(f"Año {year} (t={t_values_uncertainty[i]}):")
        print(f"  Ventas Totales Proyectadas (Hiperplano): {total_sales_pred_hyper:.0f} unidades")
        print(f"  Intervalo de Predicción del 95%: [{pi_lower_hyper:.0f}, {pi_upper_hyper:.0f}] unidades")
        print(f"  Margen de Error: +/- {margin_of_error_hyper:.0f} unidades")

# Graficar Proyecciones del Hiperplano con Intervalos de Incertidumbre
if XTX_inv_hyper is not None:
    plt.figure(figsize=(12, 7))
    
    # Datos históricos y ajuste del hiperplano
    plt.plot(range(2013, 2013 + len(y)), y, 'o-', label='Ventas Totales Observadas', color='blue')
    plt.plot(range(2013, 2013 + len(y_pred)), y_pred, 's--', label='Ajuste Hiperplano (Histórico)', color='green')

    # Recopilar datos de pronóstico para graficar
    forecast_years_plot = []
    forecast_sales_plot = []
    forecast_pi_lower_plot = []
    forecast_pi_upper_plot = []

    for i, year_val in enumerate(years_uncertainty):
        x_new_features_hyper = sales_type_proj_uncertainty[i]
        total_sales_pred_hyper = reg.predict(x_new_features_hyper.reshape(1, -1))[0]
        
        x_new_with_intercept_hyper = np.concatenate(([1], x_new_features_hyper))
        var_pred_hyper = mse_hyper * (1 + x_new_with_intercept_hyper.reshape(1, -1) @ XTX_inv_hyper @ x_new_with_intercept_hyper.reshape(-1, 1))
        se_pred_hyper = np.sqrt(var_pred_hyper[0,0])
        margin_of_error_hyper = t_critical_hyper * se_pred_hyper
        pi_lower_hyper = total_sales_pred_hyper - margin_of_error_hyper
        pi_upper_hyper = total_sales_pred_hyper + margin_of_error_hyper

        forecast_years_plot.append(year_val)
        forecast_sales_plot.append(total_sales_pred_hyper)
        forecast_pi_lower_plot.append(pi_lower_hyper)
        forecast_pi_upper_plot.append(pi_upper_hyper)

    # Pronósticos puntuales
    plt.plot(forecast_years_plot, forecast_sales_plot, 'D-', label='Proyección Hiperplano (2023-2025)', color='red', markersize=7)
    
    # Intervalos de predicción
    plt.fill_between(forecast_years_plot, forecast_pi_lower_plot, forecast_pi_upper_plot, color='red', alpha=0.2, label='Intervalo de Predicción 95%')
    
    plt.xlabel("Año")
    plt.ylabel("Ventas Totales")
    plt.title("Proyección de Ventas Totales con Modelo Hiperplano e Incertidumbre")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(2013, max(forecast_years_plot) + 1, 1)) # Ajustar ticks del eje X para mejor visualización
    plt.tight_layout()
    plt.show()


# --- Simulación de Monte Carlo ---
print("\n--- Simulación de Monte Carlo para Proyecciones (2023-2025) ---")
num_simulations = 10000  # Número de iteraciones de Monte Carlo

# --- Almacenamiento de Resultados de Simulación ---
# Ventas por tipo: {año: [type1_sims, type2_sims, ...], ...}
mc_sales_type_projections = {year: [[] for _ in range(sales_hist_data.shape[1])] for year in years_uncertainty}
# Ventas totales (hiperplano): {año: [totals_sims], ...}
mc_total_sales_projections = {year: [] for year in years_uncertainty}
# Demanda de componentes: {component_idx: [demand_sims_2024], ...}
mc_component_demand_2024 = [[] for _ in range(C_matrix.shape[0])]
mc_component_demand_2025 = [[] for _ in range(C_matrix.shape[0])]

# --- Desviaciones Estándar para Muestreo de Errores ---
# Desviación estándar de los residuos para modelos de tipo individual (sqrt(ECM_type))
std_dev_residuals_types = [np.sqrt(ecm) for ecm in ecm_values]
# Desviación estándar de los residuos para el modelo de hiperplano (sqrt(ECM_global))
std_dev_residuals_hyperplane = np.sqrt(ecm_global)

print(f"Ejecutando {num_simulations} iteraciones de Monte Carlo...")
for sim_count in range(num_simulations):
    if (sim_count + 1) % 1000 == 0:
        print(f"  Iteración {sim_count + 1}/{num_simulations}")

    # Almacenar ventas simuladas por tipo para la iteración actual, por año
    # {year: np.array([type1_sale, type2_sale, ...]), ...}
    current_iteration_sales_per_type_by_year = {}

    # 1. Simular Ventas para Cada Tipo de Motocicleta
    for year_idx, year_val in enumerate(years_uncertainty): # 2023, 2024, 2025
        t_val = t_values_uncertainty[year_idx] # 11, 12, 13
        simulated_sales_for_current_year_types = []

        for type_idx in range(sales_hist_data.shape[1]): # Para cada tipo de motocicleta
            beta = beta_coeffs[type_idx]
            point_prediction_type = beta[0] + beta[1] * t_val
            # Muestrear error de Normal(0, sqrt(ECM_type))
            error_type = np.random.normal(0, std_dev_residuals_types[type_idx])
            simulated_sale_type = point_prediction_type + error_type
            simulated_sale_type = max(0, simulated_sale_type) # Asegurar ventas no negativas

            mc_sales_type_projections[year_val][type_idx].append(simulated_sale_type)
            simulated_sales_for_current_year_types.append(simulated_sale_type)
        
        current_iteration_sales_per_type_by_year[year_val] = np.array(simulated_sales_for_current_year_types)

    # 2. Simular Ventas Totales usando el Modelo de Hiperplano
    for year_val in years_uncertainty: # 2023, 2024, 2025
        # La entrada para el hiperplano es el array de ventas simuladas por tipo para este año
        simulated_type_sales_for_year = current_iteration_sales_per_type_by_year[year_val]
        
        point_prediction_hyperplane = reg.predict(simulated_type_sales_for_year.reshape(1, -1))[0]
        # Muestrear error de Normal(0, sqrt(ECM_global))
        error_hyperplane = np.random.normal(0, std_dev_residuals_hyperplane)
        simulated_total_sale = point_prediction_hyperplane + error_hyperplane
        simulated_total_sale = max(0, simulated_total_sale) # Asegurar ventas no negativas

        mc_total_sales_projections[year_val].append(simulated_total_sale)

    # 3. Simular Demanda de Componentes para 2024 y 2025
    # Para 2024
    sim_sales_2024_types = current_iteration_sales_per_type_by_year[2024]
    sim_demand_2024_components = C_matrix @ sim_sales_2024_types
    for comp_idx in range(C_matrix.shape[0]):
        mc_component_demand_2024[comp_idx].append(max(0, sim_demand_2024_components[comp_idx]))

    # Para 2025
    sim_sales_2025_types = current_iteration_sales_per_type_by_year[2025]
    sim_demand_2025_components = C_matrix @ sim_sales_2025_types
    for comp_idx in range(C_matrix.shape[0]):
        mc_component_demand_2025[comp_idx].append(max(0, sim_demand_2025_components[comp_idx]))
print("Iteraciones de Monte Carlo completas.")

# --- Analizar e Imprimir Resultados de Monte Carlo ---
print("\n--- Resultados de la Simulación de Monte Carlo (Media, IC del 95% a partir de Percentiles) ---")

print("\nProyección de Ventas por Tipo (Monte Carlo):")
for year_val in sorted(mc_sales_type_projections.keys()):
    print(f"Año {year_val}:")
    for type_idx in range(sales_hist_data.shape[1]):
        sim_data = np.array(mc_sales_type_projections[year_val][type_idx])
        mean_val = np.mean(sim_data)
        ci_lower = np.percentile(sim_data, 2.5)
        ci_upper = np.percentile(sim_data, 97.5)
        print(f"  Tipo {type_idx+1}: Media={mean_val:.0f}, IC del 95%=[{ci_lower:.0f}, {ci_upper:.0f}]")

print("\nProyección de Ventas Totales - Hiperplano (Monte Carlo):")
for year_val in sorted(mc_total_sales_projections.keys()):
    sim_data = np.array(mc_total_sales_projections[year_val])
    mean_val = np.mean(sim_data)
    ci_lower = np.percentile(sim_data, 2.5)
    ci_upper = np.percentile(sim_data, 97.5)
    print(f"Año {year_val}: Media={mean_val:.0f}, IC del 95%=[{ci_lower:.0f}, {ci_upper:.0f}]")

print("\nProyección de Demanda de Componentes 2024 (Monte Carlo):")
for comp_idx in range(C_matrix.shape[0]):
    sim_data = np.array(mc_component_demand_2024[comp_idx])
    mean_val = np.mean(sim_data)
    ci_lower = np.percentile(sim_data, 2.5)
    ci_upper = np.percentile(sim_data, 97.5)
    print(f"  Componente {comp_idx+1}: Media={mean_val:.0f}, IC del 95%=[{ci_lower:.0f}, {ci_upper:.0f}]")

print("\nProyección de Demanda de Componentes 2025 (Monte Carlo):")
for comp_idx in range(C_matrix.shape[0]):
    sim_data = np.array(mc_component_demand_2025[comp_idx])
    mean_val = np.mean(sim_data)
    ci_lower = np.percentile(sim_data, 2.5)
    ci_upper = np.percentile(sim_data, 97.5)
    print(f"  Componente {comp_idx+1}: Media={mean_val:.0f}, IC del 95%=[{ci_lower:.0f}, {ci_upper:.0f}]")

# Optional: Plot histogram for a key result (e.g., Total Sales in 2025)
if 2025 in mc_total_sales_projections and len(mc_total_sales_projections[2025]) > 0:
    plt.figure(figsize=(10, 6))
    plt.hist(mc_total_sales_projections[2025], bins=50, density=True, alpha=0.75, color='purple', edgecolor='black')
    plt.title("Monte Carlo: Distribución de la proyección de ventas totales para 2025 (Hiperplano)")
    plt.xlabel("Ventas Totales Simuladas para 2025")
    plt.ylabel("Densidad de Probabilidad")
    # Añadir líneas de media y percentiles
    mean_val = np.mean(mc_total_sales_projections[2025])
    ci_lower = np.percentile(mc_total_sales_projections[2025], 2.5)
    ci_upper = np.percentile(mc_total_sales_projections[2025], 97.5)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Media: {mean_val:.0f}')
    plt.axvline(ci_lower, color='orange', linestyle='dotted', linewidth=2, label=f'2.5th Pctl: {ci_lower:.0f}')
    plt.axvline(ci_upper, color='orange', linestyle='dotted', linewidth=2, label=f'97.5th Pctl: {ci_upper:.0f}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()