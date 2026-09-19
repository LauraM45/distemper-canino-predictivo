# Sistema Predictivo de Distemper Canino (Canine Distemper Virus - CDV) 

Sistema de aprendizaje automático clínico para la detección temprana y diagnóstico predictivo de **Moquillo Canino (CDV)**, implementado bajo el estándar de ingeniería y aseguramiento de calidad **CRISP-ML(Q)** (*Cross-Industry Standard Process for Machine Learning with Quality Assurance*).

---

##  1. Descripción del Proyecto

El virus del distemper canino (CDV) es un patógeno viral altamente contagioso, multisistémico y potencialmente letal que afecta a la población canina. En la práctica clínica veterinaria y en albergues o centros de rescate, el diagnóstico temprano resulta indispensable para aislar oportunamente al paciente e iniciar soporte terapéutico agresivo antes de la aparición de secuelas neurológicas irreversibles.

Este sistema implementa un pipeline de clasificación clínica multi-algoritmo diseñado para **maximizar la sensibilidad clínica (Recall $\ge 80\%$)** en la detección de casos confirmados, previniendo falsos negativos catastróficos y asegurando **resiliencia total ante arranque en frío (*Coldstart*)** para razas y centros veterinarios no observados previamente.

---

##  2. Esquema Clínico de Datos (12 Variables)

El sistema procesa y valida estrictamente las siguientes 12 variables (11 predictoras y 1 objetivo):

| # | Columna | Tipo | Rango / Valores Admisibles | Rol Clínico |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `edad_meses` | Numérico (`float`) | $[0.5, 240.0]$ | Edad del canino en meses (2 semanas a 20 años). |
| 2 | `sexo` | Categórico | `Macho`, `Hembra` | Dimorfismo sexual. |
| 3 | `raza` | Categórico | Alta cardinalidad (`Border Collie`, `Mestizo`, etc.) | Predisposición genética o fenotípica. |
| 4 | `talla` | Categórico | `Pequeño`, `Mediano`, `Grande`, `Gigante` | Clasificación somática del paciente. |
| 5 | `ubicacion_procedencia` | Categórico | Centros veterinarios, localidades o albergues | Trazabilidad epidemiológica. |
| 6 | `estado_vacunal` | Categórico | `Completo`, `Incompleto`, `No vacunado`, `Desconocido` | Inmunidad previa contra CDV. |
| 7 | `fiebre_hipertermia` | Binario | $\{0, 1\}$ | Presencia de temperatura corporal elevada. |
| 8 | `signos_respiratorios_oculonasales` | Binario | $\{0, 1\}$ | Secreción mucopurulenta, tos o disnea. |
| 9 | `signos_digestivos` | Binario | $\{0, 1\}$ | Vómito o diarrea. |
| 10 | `signos_neurologicos` | Binario | $\{0, 1\}$ | Mioclonías, ataxia, paresia o convulsiones. |
| 11 | `signos_dermatologicos` | Binario | $\{0, 1\}$ | Hiperqueratosis ("hard pad") o dermatitis pustulosa. |
| **12** | **`diagnostico_cdv_confirmado`** | **Binario** | **$\{0, 1\}$** | **Variable Objetivo (0 = Sano, 1 = Positivo CDV).** |

### Registro de Referencia Validado:
```csv
4.0,Macho,Border Collie,Mediano,Bogota D.C. (UNAL),Incompleto,1,0,0,1,1,1
```

---

##  3. Prevención de Fuga de Datos y Manejo de Coldstart

1. **Prevención de Data Leakage:**
   - Todas las transformaciones estadísticas (imputación por mediana, estandarización Z-score y codificación One-Hot) están estrictamente confinadas dentro de cada pliegue de validación cruzada (`Pipeline` + `ColumnTransformer`).
2. **Resiliencia Anti-Coldstart:**
   - La rama categórica utiliza `OneHotEncoder(min_frequency=2, handle_unknown="infrequent_if_exist")`.
   - Cuando un paciente presenta una raza o procedencia no vista en entrenamiento (ej. *Shiba Inu*), el sistema lo redirige de forma determinista hacia la categoría de infrecuentes, evitando excepciones en tiempo de ejecución o vectores de ceros descalibrados.

---

##  4. Arquitectura de Modelos Implementados

El sistema implementa, optimiza y compara tres arquitecturas complementarias:
1. **Árbol de Decisión (`DecisionTreeClassifier`):** Modelo clínico interpretable con reglas explícitas de decisión visualizables en diagramas de flujo.
2. **Random Forest (`RandomForestClassifier`):** Ensamble robusto no lineal con análisis de importancia relativa de variables clínicas (Gini Importance).
3. **Regresión Logística (`LogisticRegression`):** Modelo paramétrico calibrado que permite la extracción directa de coeficientes Log-Odds y razones de momios (*Odds Ratios*).

---

##  5. Instrucciones de Uso y Ejecución

### Requisitos Previos:
```bash
pip install -r requirements.txt
```

### 1. Entrenar y Evaluar el Sistema Completo:
```bash
python main.py
```
*Genera los pipelines serializados en `src/data/artifacts/` y las matrices de confusión, diagramas y reporte JSON en `demo_reports/`.*

### 2. Inferencia Clínica con el Registro de Referencia:
```bash
python predict.py
```

### 3. Prueba de Arranque en Frío (Coldstart):
```bash
python predict.py --coldstart-test
```

### 4. Diagnóstico por Lotes a partir de un CSV:
```bash
python predict.py --input-csv ruta/a/pacientes.csv --output-csv ruta/a/diagnosticos.csv
```

---

##  6. Metadatos de Gobernanza (`ModelMetadata`)

- **Nombre del Modelo:** `CDV-Predictive-Detector`
- **Versión:** `1.0.0`
- **Patología:** `Canine Distemper Virus (CDV)`
- **Sensibilidad Mínima Exigida:** `80.0%`
- **Estilo de Código y Documentación:** `NumPy Style` / `PEP 8`