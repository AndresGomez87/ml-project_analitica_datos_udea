# Laboratorio 5 — Clasificacion con Datos Desbalanceados
## Guia de Presentacion (10 minutos)

**Dataset:** BRFSS 2015 — CDC (Centers for Disease Control and Prevention)  
**Registros:** 253,680 adultos estadounidenses  
**Variable objetivo:** `Diabetes_binary` (0 = Sin diabetes, 1 = Prediabetes/Diabetes)  
**Desbalanceo:** 86.1% clase 0 vs 13.9% clase 1 — ratio **6.2:1**

---

## Minuto 1-2: El Problema

El dataset tiene 253,680 pacientes del CDC. Solo el 13.9% son diabeticos, una razon de desbalanceo de 6.2 a 1.

El problema no es el modelo: si no aplicamos ninguna tecnica, el modelo aprende a predecir "no tiene diabetes" en casi todos los casos y aun asi obtiene **86% de Accuracy**. Eso es clinicamente peligroso.

### Por que Accuracy engana aqui

| Clase | Registros | Porcentaje |
|---|---|---|
| Sin diabetes (0) | 218,334 | 86.1% |
| Diabetes / Prediabetes (1) | 35,346 | 13.9% |

Un modelo que prediga **siempre clase 0** obtiene 86% de Accuracy sin aprender nada util.

### La metrica que importa: Recall (Sensibilidad)

Un diabetico no detectado (**Falso Negativo**) puede sufrir:
- Complicaciones renales (nefropatia diabetica)
- Enfermedades cardiovasculares
- Perdida de vision (retinopatia)

**Preferimos equivocarnos diciendo "puede tener diabetes" (falsa alarma) antes que decir "esta sano" cuando no lo esta.**

> **Orden de prioridad de metricas:**
> `Recall > F1-Score > ROC-AUC > Precision > Accuracy`

---

## Minuto 3-4: Modelo Base y Por Que Falla

Se entreno una **Regresion Logistica** estandar sin ninguna tecnica de balanceo como referencia.

| Metrica | Resultado Baseline |
|---|---|
| Accuracy | 86% (engana) |
| Recall (diabeticos) | **~15%** |
| F1-Score | ~0.23 |

> De cada 100 diabeticos, el modelo solo detectaba 15.

Esto confirma que necesitamos tecnicas especiales para corregir el desbalanceo.

---

## Minuto 5-7: Las Tres Estrategias Implementadas

### Estrategia 1 — XGBoost con `scale_pos_weight`

**Que es XGBoost:**  
Algoritmo de *boosting por gradiente*: construye arboles de decision secuencialmente donde cada arbol nuevo corrige los errores del anterior. Es uno de los algoritmos mas potentes en competencias de ML y datos tabulares.

**Como maneja el desbalanceo:**  
El parametro `scale_pos_weight = n_negativos / n_positivos = 6.2` le dice al modelo que cada muestra de diabetico vale 6.2 veces mas que una muestra sana en la funcion de perdida. Esto penaliza mas los Falsos Negativos durante el entrenamiento.

**Hiperparametros aplicados directamente (sin GridSearch):**

| Parametro | Valor | Razon |
|---|---|---|
| `n_estimators` | 300 | Bosque grande para convergencia robusta |
| `learning_rate` | 0.05 | Convergencia lenta, evita sobreajuste |
| `max_depth` | 6 | Captura interacciones sin memorizar ruido |
| `subsample` | 0.8 | Regularizacion estocastica (similar a dropout) |
| `colsample_bytree` | 0.8 | Usa el 80% de features por arbol |
| `min_child_weight` | 5 | Evita particiones con muy pocas muestras minoritarias |
| `scale_pos_weight` | 6.2 | Ratio exacto clases mayoritaria/minoritaria |

**Ventaja clinica:** trabaja exclusivamente con datos reales, sin generar muestras sinteticas. Preferido cuando la calidad del dato medico es critica.

---

### Estrategia 2 — SMOTE + XGBoost

**Que es SMOTE:**  
*Synthetic Minority Over-sampling Technique*. Genera muestras sinteticas de la clase minoritaria interpolando entre los `k=5` vecinos mas cercanos de cada muestra diabetica en el espacio de features.

```
Muestra sintetica = muestra_A + λ × (muestra_B - muestra_A)
donde λ ∈ [0, 1] y muestra_B es un vecino aleatorio de muestra_A
```

**Flujo del pipeline:**
1. Preprocesamiento (imputacion de nulos)
2. SMOTE balancea el dataset a **1:1** (clase 0 = clase 1)
3. XGBoost entrena sobre el dataset ya balanceado (sin `scale_pos_weight`)

**Por que SMOTE y no ADASYN:**  
El dataset tiene 18 variables binarias u ordinales. ADASYN concentra la generacion sintetica en zonas de alta densidad con muestras dificiles, pero con features discretas puede generar ruido artificial. SMOTE es mas estable en espacios de features discretos.

---

### Estrategia 3 — Balanced Random Forest

**Que es un Random Forest:**  
Ensamble de arboles de decision donde cada arbol se entrena sobre un subconjunto aleatorio de datos (bootstrap) y de features. La prediccion final es el promedio de todos los arboles.

**Como lo modifica Balanced Random Forest:**  
En vez de bootstrap del dataset completo, para cada arbol toma:

| Clase | Random Forest normal | Balanced Random Forest |
|---|---|---|
| Sin diabetes (0) | ~178,000 muestras (train) | ~25,000 muestras |
| Diabetes (1) | ~25,000 muestras (train) | ~25,000 muestras |

Cada arbol ve un dataset **1:1** sin generar datos sinteticos.

**Hiperparametros aplicados:**

| Parametro | Valor | Razon |
|---|---|---|
| `n_estimators` | 300 | Estabiliza las importancias de variables |
| `max_depth` | 20 | Captura interacciones complejas entre factores de riesgo |
| `min_samples_leaf` | 3 | Evita hojas con muy pocas muestras (sobreajuste) |
| `replacement` | True | Muestreo con reemplazo por arbol |

**Ventaja adicional para contexto clinico:** produce `feature_importances_` muy interpretables. Un medico puede entender facilmente cuales son los factores de riesgo con mayor peso predictivo.

---

## Minuto 8: Validacion Cruzada y Ajuste de Umbral

### Validacion Cruzada — Stratified K-Fold (k=5)

Se evaluan los 4 modelos con **Stratified K-Fold de 5 pliegues** sobre los datos de entrenamiento (70% del dataset).

**Por que Stratified?** Garantiza que cada pliegue tenga exactamente la misma proporcion de diabeticos (13.9%) que el dataset original. Sin estratificacion, algunos pliegues podrian tener muy pocos diabeticos y sesgar la evaluacion.

### Ajuste de Umbral de Decision

El umbral por defecto es **0.5**: si la probabilidad predicha supera 0.5, el modelo clasifica como diabetico.

Podemos ajustarlo usando la **curva Precision-Recall**:

| Umbral | Efecto | Cuando usarlo |
|---|---|---|
| Alto (ej. 0.7) | Mas Precision, menos Recall | Cuando las falsas alarmas son muy costosas |
| 0.5 (default) | Balance estandar | Uso general |
| Bajo (ej. 0.3) | Mas Recall, menos Precision | Screening masivo, minimizar diabeticos no detectados |

> El umbral optimo automatico se calcula maximizando el F1-Score sobre la curva Precision-Recall.  
> Pero en la practica, esta decision es **clinica, no estadistica**.

---

## Minuto 9: Resultados Esperados

### Comparacion de Modelos (Validacion Cruzada k=5)

| Modelo | Recall | F1-Score | ROC-AUC | Precision |
|---|---|---|---|---|
| Baseline (LR sin balanceo) | ~0.15 | ~0.23 | ~0.82 | ~0.53 |
| XGBoost + scale_pos_weight | ~0.55-0.65 | ~0.40-0.48 | ~0.83 | ~0.35-0.42 |
| SMOTE + XGBoost | ~0.55-0.65 | ~0.42-0.50 | ~0.83 | ~0.35-0.42 |
| Balanced Random Forest | ~0.65-0.75 | ~0.42-0.50 | ~0.82 | ~0.30-0.38 |

### Criterios de Seleccion del Modelo Final

| Criterio | Consideracion |
|---|---|
| **Recall** | Prioridad maxima: minimizar diabeticos no detectados |
| **F1-Score** | Metrica de ranking: equilibra Recall y Precision |
| **ROC-AUC** | Capacidad discriminativa global, independiente del umbral |
| **Precision** | Reducir falsas alarmas (sanos clasificados como diabeticos) |
| **Accuracy** | NO es criterio valido en datos desbalanceados |

---

## Minuto 10: Conclusion

### Tres Puntos Clave

1. **Accuracy miente** en datos desbalanceados. Siempre reportar Recall, F1 y AUC-ROC en problemas medicos.

2. **No hay una tecnica universal**: cada estrategia tiene su lugar segun el contexto clinico.
   - `scale_pos_weight` → conservador, sin datos sinteticos
   - SMOTE → mas agresivo en recall, dataset sintetico balanceado
   - Balanced RF → mas interpretable para equipos clinicos

3. **El ajuste de umbral** es tan importante como el modelo. Un mismo modelo puede comportarse muy diferente segun el umbral, y esa es una decision clinica, no un hiperparametro tecnico.

---

## Resumen de Tecnicas

| Tecnica | Tipo | Mecanismo | Genera datos sinteticos |
|---|---|---|---|
| `scale_pos_weight` | Ponderacion interna | Ajusta la funcion de perdida del boosting | No |
| SMOTE | Sobremuestreo | Interpola vecinos de la clase minoritaria | Si |
| Balanced Random Forest | Ensamble balanceado | Submuestreo automatico por arbol | No |

---

## Frases Clave para la Presentacion

> *"Priorizamos Recall sobre Precision porque el costo de un Falso Negativo supera al de un Falso Positivo en screening medico."*

> *"Usamos StratifiedKFold para garantizar que el desbalanceo original se preserve en cada pliegue de validacion."*

> *"El ajuste de umbral via curva Precision-Recall nos permite adaptar el modelo a distintos contextos: un hospital de alto riesgo preferiria un umbral bajo para maximizar deteccion."*

> *"Elegimos SMOTE sobre ADASYN porque la mayoria de nuestras features son binarias u ordinales — SMOTE es mas estable en espacios discretos."*

> *"No realizamos GridSearch masivo: con 253K registros y 5-folds, implicaria horas de computo. Los hiperparametros elegidos son configuraciones robustas de la literatura de ML clinico."*

---

## Glosario Rapido

| Termino | Definicion breve |
|---|---|
| **Recall / Sensibilidad** | De todos los diabeticos reales, cuantos detecto el modelo |
| **Precision** | De todos los que clasifique como diabeticos, cuantos realmente lo son |
| **F1-Score** | Media armonica entre Recall y Precision |
| **ROC-AUC** | Area bajo la curva Recall vs (1-Especificidad). Mide discriminacion global |
| **Falso Negativo** | Diabetico real clasificado como sano — el error mas costoso |
| **Falso Positivo** | Sano clasificado como diabetico — genera ansiedad, mas pruebas |
| **Boosting** | Ensamble secuencial: cada modelo corrige errores del anterior |
| **SMOTE** | Generacion de muestras sinteticas interpolando vecinos cercanos |
| **Umbral de decision** | Probabilidad minima para clasificar como clase positiva (default 0.5) |

---

*Laboratorio 5 — Analitica de Datos | Dataset: BRFSS 2015 CDC*
