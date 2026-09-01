# Detección de Fraude con Tarjetas de Crédito

Proyecto de portafolio de ciencia de datos: un pipeline de clasificación binaria para detectar transacciones fraudulentas con tarjeta de crédito, sobre un dataset extremadamente desbalanceado (~0.17% de fraude).

**Autor:** Sebastian Tapia

*(Este README también está disponible en inglés: [README.md](README.md))*

---

> **Nota:** mi experiencia laboral ha sido en bancos y aseguradoras, entornos donde se maneja información altamente confidencial que no puede compartirse ni usarse fuera de esos contextos. Por eso este proyecto de portafolio está construido íntegramente sobre un dataset público (Kaggle), sin ningún dato ni información proveniente de mi actividad profesional.

---

## 🗣️ Resumen ejecutivo (en términos simples)

Imagina que trabajas en un banco y cada día pasan por el sistema miles de compras con tarjeta de crédito. La gran mayoría son compras normales, pero de vez en cuando una es un fraude — alguien usó los datos de una tarjeta robada. El problema es que el fraude es rarísimo: de cada 1,000 compras, menos de 2 son fraude, así que detectarlo es como buscar una aguja en un pajar.

Este proyecto construye un sistema que, al mirar los datos de una transacción, puede decir "esto parece sospechoso, revísalo" o "esto se ve normal, apruébalo". El proceso siguió estos pasos:

1. **Mirar los datos primero** — comparar cómo se ven las compras normales vs. las fraudulentas (montos, horarios) antes de construir nada.
2. **Enseñarle al sistema a reconocer algo raro** — como casi no hay ejemplos de fraude para aprender, se usaron técnicas para que el sistema le preste más atención a esos pocos casos y no los pase por alto.
3. **Comparar varios métodos y quedarse con el que mejor funciona**, con evidencia y no por preferencia.
4. **Afinar los detalles** del método ganador para exprimirle un poco más de rendimiento.
5. **Decidir el punto de corte según el costo real** — no es lo mismo el costo de revisar una compra normal por error que el de dejar pasar un fraude real, así que el punto en el que el sistema decide "revisar" se elige minimizando ese costo para el negocio.
6. **Explicar las decisiones** — el sistema no solo dice "fraude", también muestra qué factores influyeron más en cada predicción, para que sea confiable y no una caja negra.

El resultado es un sistema que detecta **9 de cada 10 fraudes reales**, a cambio de que menos del 0.31% de las compras normales se marquen para revisión manual innecesaria — un balance razonable, dado que el costo de dejar pasar un fraude real es mucho mayor que el de revisar una compra normal de más.

---

## 📌 Resumen del proyecto (técnico)

El objetivo es construir un modelo que identifique transacciones fraudulentas con la mayor precisión posible, priorizando el **recall** (detectar el fraude real) sin generar un volumen inmanejable de falsos positivos (transacciones legítimas marcadas para revisión manual).

El proyecto cubre el ciclo completo: exploración de datos, manejo de desbalance de clases, comparación rigurosa de modelos vía validación cruzada, selección de umbral basada en costo de negocio, interpretabilidad con SHAP, y un ejemplo de scoring en producción.

## 📊 Dataset

- **Fuente:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle)
- **Filas:** 284,807 transacciones
- **Variables:** 30 features numéricas (`V1`–`V28` son componentes de un PCA ya aplicado por los autores originales por confidencialidad, más `Time` y `Amount`)
- **Target:** `Class` (1 = fraude, 0 = legítima)
- **Desbalance:** solo ~0.17% de las transacciones son fraude — el reto central del proyecto

## 🛠️ Metodología

1. **Análisis exploratorio**: distribución de clases, estadísticas descriptivas, distribución de `Amount` por clase (escala log) y distribución de transacciones por hora del día según clase.
2. **Preprocesamiento**: split estratificado 80/20 (train/test) para preservar la proporción de fraude en ambos conjuntos.
3. **Manejo del desbalance**: comparación sistemática de estrategias — `SMOTE` (sobremuestreo sintético), `class_weight`/`scale_pos_weight` (ponderación en la función de pérdida), y combinaciones de ambas.
4. **Selección de modelo por validación cruzada (5-fold, estratificada)**: se compararon Logistic Regression, Random Forest y XGBoost, cada uno bajo distintas estrategias de balanceo, usando **PR-AUC** (más informativo que ROC-AUC en clases muy desbalanceadas) como métrica de selección. El modelo final es el de mejor PR-AUC promedio en CV, sin favorecer ningún algoritmo de antemano.
5. **Tuning de hiperparámetros**: `RandomizedSearchCV` (20 iteraciones, misma CV de 5 folds) sobre la configuración ganadora del ablation, afinando `n_estimators`, `max_depth` y demás parámetros específicos del algoritmo elegido.
6. **Diagnóstico de overfitting**: comparación de métricas train vs. test para detectar brechas de generalización.
7. **Selección de umbral por costo de negocio**: en vez de usar el umbral por defecto (0.5), se optimiza el punto de corte minimizando un costo estimado (costo de revisar un falso positivo vs. costo de un fraude no detectado).
8. **Interpretabilidad**: importancia de variables vía `feature_importances_` y valores SHAP para explicar las predicciones del modelo.
9. **Simulación de scoring**: función `score_transaction()` que aplica el pipeline completo a una transacción nueva y devuelve una decisión (`APROBAR` / `REVISAR`), también disponible como script independiente (`predict.py`).

## 📈 Resultados

| Métrica | Valor |
|---|---|
| Modelo final | XGBoost (con SMOTE) |
| PR-AUC (CV, antes de tuning) | 0.8520 |
| PR-AUC (CV, después de tuning) | 0.8566 |
| Mejores hiperparámetros | `n_estimators=400`, `max_depth=5`, `learning_rate=0.2`, `subsample=0.6`, `colsample_bytree=0.8` |
| ROC-AUC (test) | 0.9834 |
| PR-AUC (test) | 0.8721 |
| Umbral óptimo elegido | 0.05 (según costo_fp=5, costo_fn=100) |
| Precision / Recall en umbral óptimo (clase fraude) | 0.50 / 0.90 |

**Variables más influyentes** (según SHAP): `V14`, `V4`, `V1`, `V3`, `V8` y `Time` concentran el mayor impacto en las predicciones del modelo.

**Interpretación de negocio**: con el umbral óptimo (0.05), de las 56,962 transacciones de test se marcan **175 para revisión manual** (~0.31% del total), de las cuales 88 son fraudes reales y 87 son falsas alarmas. Esto significa que se detecta **~90% del fraude real** (88 de 98 casos), dejando pasar solo 10 fraudes, a cambio de revisar manualmente un volumen muy pequeño de transacciones legítimas.

> **Nota sobre overfitting**: el modelo final alcanza PR-AUC = 1.0000 en train (ajuste perfecto) frente a 0.8721 en test — una brecha que indica que el modelo está memorizando parte del ruido de entrenamiento. El resultado en test sigue siendo sólido, pero una extensión natural del proyecto sería reducir esta brecha (menor `max_depth`, mayor regularización) y verificar si el PR-AUC de test mejora.

## 🧰 Tecnologías utilizadas

- **Python** — pandas, NumPy
- **Visualización** — Matplotlib, Seaborn
- **Modelado** — scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Interpretabilidad** — SHAP
- **Persistencia** — joblib

## 📁 Estructura del proyecto

```
├── Deteccion_fraude.ipynb    # Notebook principal: EDA, comparación de modelos, tuning, interpretabilidad
├── predict.py               # Script de scoring reutilizable, independiente del notebook (CLI + funciones importables)
├── creditcard.csv           # Dataset (no incluido en el repo por tamaño — descargar de Kaggle)
├── modelo_fraude_rf.pkl     # Modelo final entrenado y serializado (se genera al correr el notebook)
├── .gitignore                # Excluye dataset, modelo serializado, checkpoints y entornos virtuales
├── README.md                 # Versión en inglés
└── README.es.md              # Este archivo (español)
```

## ▶️ Cómo ejecutarlo

1. Descarga `creditcard.csv` desde [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) y colócalo en la raíz del proyecto.
2. Instala las dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap joblib
   ```
3. Abre y corre `Deteccion_fraude.ipynb` de principio a fin. Esto genera `modelo_fraude_rf.pkl`.
4. Para hacer scoring de nuevas transacciones sin abrir el notebook:
   ```bash
   python predict.py transacciones_nuevas.csv
   ```
   Esto genera `resultados_scoring.csv` con la probabilidad de fraude y la decisión (`APROBAR`/`REVISAR`) para cada fila. `predict.py` también puede importarse como módulo (`from predict import score_transaction, cargar_modelo`) para integrarse en otro sistema, por ejemplo una API.

## 🔍 Decisiones de diseño y limitaciones

- Se usó **PR-AUC** en lugar de accuracy como métrica principal, porque con ~0.17% de fraude un modelo que siempre predice "no fraude" tendría 99.8% de accuracy y sería inútil.
- El umbral de decisión se ajusta según un costo de negocio ilustrativo (`costo_fp` / `costo_fn` en el notebook); en un caso real estos valores deben calibrarse con datos reales de la operación (costo de revisión manual, pérdida promedio por fraude no detectado, etc.).
- El dataset ya viene con `V1`–`V28` transformadas por PCA, lo que limita el análisis de negocio directo sobre esas variables (no sabemos qué representa "V14" en términos reales) — solo `Time` y `Amount` son interpretables directamente.
- El tuning de hiperparámetros usa `RandomizedSearchCV` (20 iteraciones) en vez de una búsqueda exhaustiva por costo computacional — una búsqueda más fina (Optuna, Bayesian Optimization) es una extensión natural.
- El modelo final muestra un ajuste perfecto en train (PR-AUC = 1.0000) frente a 0.8721 en test — señal de overfitting que, aunque no invalida el resultado en test, deja margen de mejora vía mayor regularización (`max_depth` más bajo, `reg_lambda`/`reg_alpha` más altos, o `min_child_weight` mayor).

## 🚀 Posibles extensiones

- Búsqueda de hiperparámetros más fina con Optuna (optimización bayesiana en vez de muestreo aleatorio).
- Despliegue de `predict.py` como API (FastAPI/Flask) para scoring en tiempo real.
- Monitoreo de drift de datos si el modelo se usara en producción.
