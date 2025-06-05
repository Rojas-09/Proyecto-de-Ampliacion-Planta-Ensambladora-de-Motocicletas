# Proyecto de Ampliación: Planta Ensambladora de Motocicletas MotoTec

## 📋 Descripción del Proyecto

Proyecto de análisis predictivo para MotoTec que utiliza **métodos de mínimos cuadrados** y **álgebra lineal** para pronosticar la demanda de motocicletas y optimizar la planificación de inventario. El objetivo es reducir el inventario en un 25% sin afectar la producción.

## 🎯 Objetivos

- **Pronosticar ventas** de 4 tipos de motocicletas (2023-2027)
- **Calcular necesidades** de 10 componentes clave por año
- **Cuantificar incertidumbre** mediante modelos estadísticos avanzados
- **Optimizar inventario** reduciendo dependencia de componentes importados

## 📊 Datos Históricos

- **Período**: 2013-2022 (10 años)
- **Tipos de motos**: 4 categorías (Urbanas, Turismo, Off-road, Eléctricas)
- **Componentes**: 10 componentes críticos por motocicleta

## 🔧 Tecnologías Utilizadas

- **Python 3.x**
- **NumPy** - Cálculos matriciales y álgebra lineal
- **Matplotlib** - Visualización de datos
- **SciPy** - Análisis estadístico
- **Scikit-learn** - Modelos de regresión
- **LaTeX** - Documentación técnica

## 📁 Estructura del Proyecto

```
├── regresion_svd.py              # Implementación SVD para regresión
├── import numpy as np II.py      # Análisis completo principal
├── Lineal.tex                    # Documentación técnica completa
├── *.png                         # Gráficos y visualizaciones
├── *.pdf                         # Reportes y documentación
└── .vscode/                      # Configuración del entorno
```

## 🚀 Ejecución

### Análisis Principal
```bash
python "import numpy as np II.py"
```

### Análisis con SVD
```bash
python regresion_svd.py
```

## 📈 Resultados Clave

### Modelos de Regresión Individual
- **Tipo 1**: R² = 0.99, Modelo: `S₁(t) = 129.33 + 26.54t`
- **Tipo 2**: R² = 0.97, Modelo: `S₂(t) = 79.07 + 22.24t`
- **Tipo 3**: R² = 0.99, Modelo: `S₃(t) = -10.40 + 30.69t`
- **Tipo 4**: R² = 0.71, Modelo: `S₄(t) = -18.33 + 18.62t`

### Modelo Global (Hiperplano)
- **Ecuación**: `S_total(t) = 179.67 + 98.10t`
- **R²**: 0.9809 (98.09% de variabilidad explicada)
- **Error estándar**: ±39.3 unidades

### Proyecciones 2024-2025
- **2024**: 1,357 unidades (IC 95%: [1,242 - 1,472])
- **2025**: 1,455 unidades (IC 95%: [1,334 - 1,576])

## 🎲 Análisis de Monte Carlo

- **10,000 simulaciones** para cuantificar incertidumbre
- **Error relativo**: ±5.4% - 5.7%
- **Intervalos de confianza** del 95% para todos los componentes

## 📊 Visualizaciones Incluidas

- Ventas históricas y pronosticadas por tipo
- Ajuste global por año (Hiperplano)
- Demanda estimada de componentes
- Distribución Monte Carlo de proyecciones
- Comparación de métodos de proyección

## 🔍 Metodología

1. **Construcción de matriz de diseño** A = [1|t]
2. **Mínimos cuadrados**: β = (AᵀA)⁻¹AᵀS
3. **Descomposición SVD** para estabilidad numérica
4. **Análisis de incertidumbre** con intervalos de predicción
5. **Simulación estocástica** para validación

## 📋 Componentes Analizados

Matriz C (10×4) define requerimientos por tipo de moto:
- Promedio: **10.3 componentes por motocicleta**
- Componentes críticos: 6, 7, 8 (mayor demanda)

## 🎓 Contexto Académico

- **Curso**: Álgebra Lineal - 2º Semestre
- **Universidad**: Universidad Tecnológica
- **Autores**: Juan Andrés Rojas - Esteban Ibarra
- **Fecha**: Mayo 2025

## 📄 Documentación

El análisis completo está documentado en [`Lineal.tex`](Lineal.tex) con:
- Marco teórico detallado
- Metodología paso a paso
- Resultados y métricas completas
- Conclusiones y recomendaciones

## 🔗 Referencias

- Métodos de mínimos cuadrados
- Descomposición en valores singulares (SVD)
- Análisis de regresión multivariada
- Simulación de Monte Carlo
