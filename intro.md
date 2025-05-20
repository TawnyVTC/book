

# **1. Introducción**

## 1.1. Retiros en Fondos de Inversión Colectiva como Fenómeno Económico

Los Fondos de Inversión Colectiva (FIC) y los Fondos de Capital Privado (FCP) son mecanismos que permiten a varios inversionistas reunir recursos para participar en portafolios diversificados. El comportamiento de los inversionistas, particularmente en cuanto a los **retiros o redenciones**, constituye una variable crítica para la estabilidad del fondo. Estos retiros reflejan decisiones motivadas por factores financieros, psicológicos y contextuales que afectan la sostenibilidad del portafolio.

De acuerdo con la Superintendencia Financiera de Colombia, el valor de un fondo al cierre del día \$t\$ se calcula con base en los recursos aportados, los rendimientos obtenidos y los gastos incurridos \cite{superfinanciera\_valoracion}. A pesar de la disponibilidad de esta información, **la predicción de los retiros diarios sigue siendo un desafío analítico** de alto valor práctico, especialmente para prevenir situaciones de iliquidez.

## 1.2. Objetivo del estudio

El presente estudio tiene como objetivo **predecir la cantidad de retiros o redenciones diarias en fondos de inversión**, utilizando un conjunto de variables relacionadas con el comportamiento del fondo y de los inversionistas. La propuesta introduce un enfoque novedoso mediante un **modelo ensamblado basado en regresión cuantil como meta-modelo**, con el fin de capturar patrones no lineales, distribuciones asimétricas y valores extremos.

## 1.3. Hipótesis técnica

> *Una arquitectura de stacking que combina modelos base heterogéneos con un meta-modelo de Regresión Cuantil permite mejorar la precisión y robustez en la predicción de retiros financieros, superando a modelos individuales en escenarios de alta varianza y colas pesadas.*

## 1.4. Enfoque metodológico

Se implementó un flujo de trabajo estructurado en dos etapas:

### 1.4.1. Entrenamiento, optimización y evaluación de modelos base

Se entrenaron y evaluaron seis modelos de regresión:

* Regresión Lineal
* Ridge
* Lasso
* KNN
* Random Forest
* XGBoost

Para mejorar el rendimiento y la precisión, se incorporaron técnicas de **optimización algorítmica** específicas para cada tipo de modelo:

* En los modelos basados en regresión lineal (Lineal, Ridge, Lasso), se utilizó el solver `'saga'` para mayor eficiencia en presencia de regularización.
* Para KNN, se compararon técnicas de búsqueda optimizada como **Ball Trees** y **KD-Trees**, además de una implementación acelerada con **FAISS**, que permitió reducir el tiempo de inferencia en más de un 98%.
* En el caso de XGBoost, se empleó el método de construcción de árboles **`hist`**, que acelera la búsqueda de divisiones, y se incorporó **early stopping** para evitar sobreajuste mediante detención temprana del entrenamiento.
* Random Forest fue optimizado mediante ajuste del número de árboles, la profundidad máxima y la selección de características por división.

Cada modelo fue ajustado utilizando una **búsqueda aleatoria de hiperparámetros (`RandomizedSearchCV`)** con validación cruzada (\$k = 3\$), priorizando la métrica \$R^2\$ como criterio de evaluación.

```python
# Fragmento clave del entrenamiento
if nombre_modelo in modelos_con_escalado:
    pasos.append(('scaler', StandardScaler()))
pasos.append(('model', modelo))

pipeline = Pipeline(pasos)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=params,
    n_iter=n_iter,
    cv=cv,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
```

### 1.4.2. Modelo ensamblado propuesto: **SQRE-WithdrawPredictor**

Posterior a la evaluación individual, se seleccionaron los dos modelos con mejor desempeño (Random Forest y XGBoost), los cuales fueron integrados mediante un **modelo de stacking** cuyo meta-modelo es una **Regresión Cuantil** enfocada en la mediana (cuantil \$q = 0.5\$).

Este modelo, denominado **SQRE-WithdrawPredictor**, busca no sólo minimizar errores promedio, sino también capturar la estructura de colas y la variabilidad condicional, común en los retiros financieros masivos.

## 1.5. Evaluación de resultados

Los modelos fueron evaluados según las siguientes métricas:

* \$\textbf{RMSE}\$: Raíz del error cuadrático medio
* \$\textbf{MAE}\$: Error absoluto medio
* \$\textbf{MAPE}\$: Error porcentual absoluto medio
* \$\textbf{R^2}\$: Coeficiente de determinación
* \$\textbf{Jarque-Bera}\$: Normalidad de los residuos
* \$\textbf{Jlung-Box}\$: Autocorrelación de residuos



Los resultados obtenidos con el modelo original mostraron un desempeño consistente tanto en el conjunto de entrenamiento como en el de prueba. En entrenamiento, el modelo alcanzó un \$R^2\$ de 0.79. En prueba, el rendimiento se mantuvo estable, con un \$R^2\$ de 0.77, indicando buena capacidad predictiva y generalización para la predicción de retiros financieros diarios.



## 1.6. Variables para el modelo

### Pregunta central:

> \textit{"¿Cómo influyen los aportes recibidos, el valor del fondo, el número de inversionistas y el tipo de entidad en la cantidad de retiros/redenciones realizadas?"}

### Variable respuesta:

**\$\textbf{RETIROS\_REDENCIONES}\$**

### Variables predictoras:

| Nombre de Variable       | Descripción                             |
| ------------------------ | --------------------------------------- |
| APORTES\_RECIBIDOS       | Cantidad de dinero ingresado            |
| PRECIERRE\_FONDO\_DIA\_T | Precio de cierre de la unidad del fondo |
| NUMERO\_INVERSIONISTAS   | Número de inversionistas del fondo      |
| TIPO\_ENTIDAD            | Tipo de entidad financiera              |

Estas variables se consideraron relevantes para modelar el comportamiento colectivo de los inversionistas y predecir dinámicamente los flujos de salida.

