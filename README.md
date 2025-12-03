# Predicci-n_de-Siniestros_de_Seguro-_Claims_Prediction
#Proyecto de Clasificación: Predicción de Siniestros de Seguro

#Resumen y Desafío

Este proyecto abordó la predicción de siniestros de seguro (Clase 1), un problema con severo **desbalance de clases**. El objetivo fue construir una solución robusta y de bajo riesgo de **fuga de datos (data leakage)**, superando el límite de rendimiento de los algoritmos.

La convergencia de tres modelos (**LogReg**, **Random Forest**, **XGBoost**) en métricas similares (acuracy 0.85) demostró que el **límite de la señal predictiva residía en las características**.

#Metodología y Pipeline Riguroso

Se implementó un **Pipeline de Scikit-learn** para garantizar la integridad y la reproducibilidad:

- **Imputación Avanzada:** Uso de **`IterativeImputer`** para la gestión predictiva de valores faltantes.
- **Transformación Consolidada:** **`ColumnTransformer`** aplicó **`StandardScaler`** (numéricas) y **`OneHotEncoder`** (categóricas) simultáneamente.
- **Selección de Variables:** **`GridSearchCV`** confirmó la **regularización L1** como la mejor opción, realizando selección automática de variables.

#Modelo Final: Priorizando el Riesgo (Recall)

La estrategia final se centró en ajustar la función de costo del modelo más interpretable para maximizar la detección de la Clase 1.

- **Estrategia Clave:** Implementación de **`class_weight='balanced'`** en la Regresión Logística optimizada.

| Métrica | LogReg Inicial | LogReg Final |
| :--- | :--- | :--- |
| **Recall (Clase 1)** | $0.72$ | $\mathbf{0.85}$ ($\uparrow 13\%$) |
| **F1-Score (Clase 1)** | $0.76$ | $\mathbf{0.77}$ |
| **AUC-ROC** | $0.9021$ | $\mathbf{0.9021}$ |

El modelo final, una **Regresión Logística con ajuste de pesos**, es la solución más eficiente. Es altamente interpretable (gracias a L1) y, crucialmente, elevó el **Recall al 85%**, proporcionando un valor real a la aseguradora al minimizar los Falsos Negativos.

---
**Tecnologías:** Python, Scikit-learn, XGBoost.
