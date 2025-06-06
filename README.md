# Proyecto de Ampliación: Planta Ensambladora de Motocicletas MotoTec

## 📋 Descripción del Proyecto

Proyecto de análisis predictivo para MotoTec que utiliza **métodos de mínimos cuadrados** y **álgebra lineal** para pronosticar la demanda de motocicletas y optimizar la planificación de inventario. El objetivo es reducir el inventario en un 25% sin afectar la producción.

## 🎥 Video Explicativo del Proyecto

[![Video Explicativo - Proyecto MotoTec](https://img.shields.io/badge/▶️%20Ver%20Video-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/6DL5IPnuLo0?si=0nF_Q2V1JAUpZO3w)

**🎬 Presentación completa del proyecto** donde explicamos la metodología, resultados y aplicación práctica del análisis de regresión lineal para el pronóstico de ventas de motocicletas.

---

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
├── Scripts/
│   ├── analisis_completo.py         # Análisis principal completo
│   ├── regresion_svd.py             # Implementación SVD para regresión
│   └── Lineal.tex                   # Documentación técnica completa
├── Image/
│   ├── Ventas historicas y pronosticadas por tipo moto.png
│   ├── Ajuste global por año (Hiperplano por minimos).png
│   ├── Demanda estimada de componentes para 2024 y 2025.png
│   ├── Monte carlo Distribucion de la proyeccion de ventas totales para 2025.png
│   ├── Proyeccion de ventas totales con modelo hiperplano e incertidumbre.png
│   └── Comparacion de metodos de proyeccion.png
├── Docs/
│   ├── Proyecto Ampliacion MotoTec.pdf
│   └── Proyecto_final_ýlgebra_Lineal (1).pdf
├── .vscode/
│   └── settings.json                # Configuración del entorno
└── README.md                        # Este archivo
```

## 🚀 Ejecución

### Análisis Principal Completo
```bash
python Scripts/analisis_completo.py
```

### Análisis con SVD
```bash
python Scripts/regresion_svd.py
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

- [Ventas históricas y pronosticadas por tipo](Image/Ventas%20historicas%20y%20pronosticadas%20por%20tipo%20moto.png)
- [Ajuste global por año (Hiperplano)](Image/Ajuste%20global%20por%20año%20(Hiperplano%20por%20minimos).png)
- [Demanda estimada de componentes](Image/Demanda%20estimada%20de%20componentes%20para%202024%20y%202025.png)
- [Distribución Monte Carlo de proyecciones](Image/Monte%20carlo%20Distribucion%20de%20la%20proyeccion%20de%20ventas%20totales%20para%202025.png)
- [Proyección con incertidumbre](Image/Proyeccion%20de%20ventas%20totales%20con%20modelo%20hiperplano%20e%20incertidumbre.png)
- [Comparación de métodos de proyección](Image/Comparacion%20de%20metodos%20de%20proyeccion.png)

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

El análisis completo está documentado en:
- [**Documentación técnica LaTeX**](Scripts/Lineal.tex) - Marco teórico, metodología y resultados detallados
- [**Reporte del proyecto MotoTec**](Docs/Proyecto%20Ampliacion%20MotoTec.pdf) - Documento empresarial
- [**Proyecto final de Álgebra Lineal**](Docs/Proyecto_final_ýlgebra_Lineal%20(1).pdf) - Entrega académica

## 💻 Scripts Principales

- [`analisis_completo.py`](Scripts/analisis_completo.py) - Análisis principal con regresión, Monte Carlo y visualizaciones
- [`regresion_svd.py`](Scripts/regresion_svd.py) - Implementación alternativa usando descomposición SVD
- [`Lineal.tex`](Scripts/Lineal.tex) - Documentación técnica completa en LaTeX

## 🎬 Recursos Multimedia

- **📺 [Video Explicativo Completo](https://youtu.be/6DL5IPnuLo0?si=0nF_Q2V1JAUpZO3w)** - Presentación detallada del proyecto, metodología y resultados

## 🔗 Referencias

- Métodos de mínimos cuadrados
- Descomposición en valores singulares (SVD)
- Análisis de regresión multivariada
- Simulación de Monte Carlo

---

**Repositorio GitHub**: [https://github.com/Rojas-09/Trabajo-Lineal.git](https://github.com/Rojas-09/Trabajo-Lineal.git)

**🎥 Video del Proyecto**: [Ver en YouTube](https://youtu.be/6DL5IPnuLo0?si=0nF_Q2V1JAUpZO3w)
