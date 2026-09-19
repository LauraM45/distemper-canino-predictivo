# Informe de Arquitectura Técnica e Ingeniería Inversa: Sistema Predictivo de Distemper Canino (CDV)

**Rol:** Arquitecto de Software Principal & Especialista en Ingeniería Inversa de Código  
**Proyecto:** *Sistema de Detección Temprana y Diagnóstico Predictivo de Moquillo Canino (Canine Distemper Virus - CDV)*  
**Versión Auditada:** 1.0.0 | **Estándar:** CRISP-ML(Q)

---

## 1. Visión General del Sistema y Stack Tecnológico

### Propósito Inferido del Sistema
El sistema es una solución analítica de soporte a la decisión clínica veterinaria (CDSS - Clinical Decision Support System) diseñada para el diagnóstico presuntivo temprano y la estratificación de riesgo de **Distemper Canino (CDV / Moquillo)** a partir de 11 variables clínicas, demográficas y epidemiológicas del paciente.

El requerimiento no funcional primario de diseño es **garantizar una sensibilidad clínica (Recall) >= 80%** para la detección de casos positivos, minimizando falsos negativos que serían letales en medicina veterinaria o en entornos de albergues y centros de rescate. Adicionalmente, cuenta con diseño específico para resolver el problema de **arranque en frío (coldstart)**, permitiendo diagnosticar pacientes pertenecientes a razas o centros de procedencia nunca antes observados en entrenamiento sin interrupciones en tiempo de ejecución.

### Stack Tecnológico y Entorno de Ejecución
* **Lenguaje de Programación:** Python 3.10+ (aprovechando anotaciones tipadas modernas, unión de tipos str | Path, y dataclasses congeladas).
* **Bibliotecas Críticas (requirements.txt):**
  * scikit-learn >= 1.3: Motor principal de aprendizaje automático; orquesta preprocesamiento (ColumnTransformer, StandardScaler, OneHotEncoder, SimpleImputer), clasificadores (DecisionTreeClassifier, RandomForestClassifier, LogisticRegression), optimización por grilla (GridSearchCV), particionamiento estratificado (StratifiedKFold) y métricas de evaluación.
  * pandas >= 2.0: Ingesta tabular, normalización de esquemas y transformaciones de datos.
  * joblib >= 1.3: Serialización y deserialización binaria de los pipelines ajustados (.pkl).
  * matplotlib >= 3.7: Renderizado programático de diagramas de árboles, curvas de importancia y matrices de confusión (backend Agg).
  * numpy: Operaciones vectorizadas y cálculo de probabilidades.
* **Entorno de Ejecución:** Scripting CLI por lotes o unitario en entornos locales/contenedores (sin servidor HTTP ni procesos daemon permanentes).

### Estilo Arquitectónico Predominante
**Arquitectura de Pipelines de Machine Learning (ML Data & Model Pipeline / Monolito Modular)**.

#### Justificación
La base de código no implementa un servicio cliente-servidor distribuido ni microservicios, sino una arquitectura monolítica modular desacoplada en dos fases operativas bien delimitadas:
1. **Fase de Ciclo de Vida Offline / Entrenamiento:** Orquestada por main.py, donde los datos brutos se validan, limpian, dividen en pliegues estratificados, entrenan múltiples clasificadores y se persisten artefactos versionados.
2. **Fase de Inferencia Online / Batch:** Orquestada por predict.py mediante la clase CDVInferenceEngine, que no depende de la lógica de entrenamiento ni de los clasificadores raw, sino únicamente del contrato de datos y del pipeline serializado (.pkl).

---

## 2. Anatomía del Proyecto y Responsabilidades por Módulo

### Desglose Estructural de Directorios

`
Modelo-Predictivo-Distemper-Canino-main/
│
├── data/
│   ├── raw/
│   │   └── dataset_moquillo_real_v4.csv     # Dataset clínico de entrenamiento (40 registros)
│   ├── processed/                           # Directorio reservado para datos normalizados
│   └── data_analysis.md                     # Espacio reservado para EDA
│
├── demo_reports/                            # Entregables de auditoría CRISP-ML(Q)
│   ├── arbol_decision_cdv.png               # Topología gráfica del árbol de decisión
│   ├── coeficientes_regresion_logistica.png # Gráfico de coeficientes Log-Odds (Odds Ratios)
│   ├── importancia_variables_random_forest.png # Gini Feature Importance
│   ├── matriz_confusion_decisiontree.png    # Matriz OOF Árbol de Decisión
│   ├── matriz_confusion_logisticregression.png # Matriz OOF Regresión Logística
│   ├── matriz_confusion_randomforest.png    # Matriz OOF Random Forest
│   └── reporte_metricas_cdv.json            # Auditoría JSON consolidada con métricas por pliegue
│
├── evaluation/
│   └── model_comparation.ipynb             # Notebook exploratorio (actualmente 0 bytes)
│
├── src/
│   ├── data/
│   │   ├── artifacts/                       # Pipelines serializados de producción (.pkl)
│   │   ├── dataValidation.py                # Contrato formal de datos, rangos y reporte de calidad
│   │   └── featureEngineer.py               # Construcción de ColumnTransformer y Pipeline scikit-learn
│   └── models/
│       ├── DecisionTreeTraining.py          # Trainer, tuning y evaluador para Árbol de Decisión
│       ├── RandomForest.py                  # Trainer, tuning y evaluador para Random Forest
│       ├── LogisticalRegresion.py           # Trainer, tuning y evaluador para Regresión Logística
│       └── LogisticRegression.py            # Módulo alias para enmascarar error tipográfico
│
├── main.py                                  # Entry point: Orquestador CRISP-ML(Q) de entrenamiento
├── predict.py                               # Entry point: Motor de inferencia unitaria/batch
├── evaluate_unseen_farm.py                  # Script de validación de generalización / Coldstart
├── generateDecisionTree.py                  # CLI específico para ajuste del árbol de decisión
└── requirements.txt                         # Especificación de dependencias de producción
`

### Delimitación de Capas (Separación de Conceptos - SoC)

| Capa | Componentes Clave | Responsabilidad Primaria |
| :--- | :--- | :--- |
| **Presentación / Transporte (CLI)** | main.py, predict.py, evaluate_unseen_farm.py, generateDecisionTree.py | Análisis de argumentos por línea de comandos (argparse), lectura/escritura de archivos (CSV/JSON/PNG) y formato de salida en consola. |
| **Dominio y Reglas Clínicas** | ModelMetadata, CDVDataValidator | Definición contractual de las 11 variables, límites fisiológicos admisibles (ej. edad [0.5, 240] meses), categorías cerradas, normalización ortográfica y estratificación de riesgo clínico (>= 0.70 Alto, >= 0.40 Moderado, < 0.40 Bajo). |
| **Transformación / Feature Engineering** | CDVFeaturePipelineBuilder | Ensamblado del ColumnTransformer scikit-learn con aislamiento estricto de pliegues (prevención de Data Leakage) y bifurcación anti-coldstart (min_frequency=2, handle_unknown="infrequent_if_exist"). |
| **Modelado y Optimización** | DecisionTreeModelTrainer, RandomForestModelTrainer, LogisticRegressionModelTrainer | Búsqueda de hiperparámetros con GridSearchCV orientada a Recall, validación cruzada estratificada fuera de pliegue (Out-Of-Fold / OOF) y generación de gráficos explicativos. |
| **Persistencia** | joblib, pathlib.Path | Serialización binaria de pipelines (.pkl), almacenamiento de reportes auditables (reporte_metricas_cdv.json) e imágenes PNG. |

### Reglas de Dependencia Observadas
* **Unidireccionalidad Estricta:** Las capas superiores dependen de las inferiores.
  * main.py -> src.models.*, src.data.dataValidation.
  * src.models.* -> src.data.featureEngineer, src.data.dataValidation.
  * src.data.featureEngineer -> src.data.dataValidation.
  * src.data.dataValidation **no depende de ningún módulo interno del proyecto**. Encapsula el núcleo de dominio con librerías estándar y matemáticas (pandas, numpy, dataclasses).
* **Desacoplamiento de Inferencia:** predict.py **no importa** ningún módulo de src.models.*. Carga el pipeline completo empaquetado mediante joblib.load(), interactuando sólo con la API polimórfica de scikit-learn (predict y predict_proba).

---

## 3. Flujo de Control y Ciclo de Vida de una Petición / Proceso

### Puntos de Entrada
1. main.py [args]: Ejecución del pipeline integral de ingeniería, tuning multi-modelo, evaluación OOF y exportación de reportes.
2. predict.py [args]: Motor de inferencia en tiempo de ejecución. Permite:
   * Diagnóstico del registro de referencia por defecto.
   * Diagnóstico con categorías desconocidas (--coldstart-test).
   * Diagnóstico batch desde CSV (--input-csv y --output-csv).
3. evaluate_unseen_farm.py: Test unitario de estrés ante pacientes y clínicas veterinarias no vistas.
4. generateDecisionTree.py: Wrapper CLI focalizado en el reentrenamiento del modelo de árbol de decisión.

### Rastreo Paso a Paso: Flujo de Inferencia Clínica (predict.py)

`
[Entrada: Paciente JSON / CSV]
           │
           ▼
[1. CDVInferenceEngine.predict_patient()]
           │
           ▼
[2. CDVDataValidator.validate_and_clean(require_target=False)]
    ├── _validate_schema(): Comprueba 11 variables obligatorias
    ├── _to_numeric(): Coerción numérica tolerante a formatos comas/puntos
    ├── Normalización de texto: Limpieza .strip() y corrección ("Pequeo" -> "Pequeño")
    ├── _validate_ranges_and_binary(): Edad en [0.5, 240.0], signos clínicos en {0, 1}
    └── _validate_controlled_categories(): Sexo, Talla y Estado Vacunal controlados
           │
           ▼
[3. Extracción de Matriz de Características X (11 columnas ordenadas)]
           │
           ▼
[4. Ejecución del Scikit-Learn Pipeline (.pkl)]
    ├── ColumnTransformer:
    │   ├── Rama Numérica: SimpleImputer(median) -> StandardScaler()
    │   ├── Rama Categórica: SimpleImputer("Desconocido") -> OneHotEncoder(min_freq=2, coldstart)
    │   └── Rama Binaria: SimpleImputer(fill_value=0)
    └── Estimador (DecisionTree / RandomForest / LogisticRegression):
        ├── pipeline.predict(X)       --> Clase binaria {0, 1}
        └── pipeline.predict_proba(X) --> [P(Sano), P(CDV)]
           │
           ▼
[5. Lógica de Negocio y Estratificación de Riesgo]
    ├── P(CDV) >= 0.70  --> "Alto Riesgo" (Aislamiento preventivo + PCR urgente)
    ├── P(CDV) >= 0.40  --> "Riesgo Moderado" (Monitoreo + biometría hemática)
    └── P(CDV) < 0.40   --> "Bajo Riesgo" (Protocolo preventivo regular)
           │
           ▼
[Salida: Diagnóstico estructurado JSON o CSV enriquecido]
`

---

## 4. Capa de Datos y Estado

### Mecanismos de Persistencia
* **Almacenamiento Tabular:** No existen bases de datos relacionales (RDBMS) ni ORMs (como SQLAlchemy o Django ORM). La persistencia de datos es 100% en archivos planos CSV (data/raw/dataset_moquillo_real_v4.csv).
* **Serialización de Modelos:** Almacenamiento binario mediante joblib.dump() de objetos Pipeline scikit-learn en src/data/artifacts/.
* **Auditoría e Informes:** Almacenamiento en formato JSON de texto plano (demo_reports/reporte_metricas_cdv.json).

### Consistencia y Transacciones
* Todas las transformaciones se ejecutan en memoria volátil dentro de instancias de pandas.DataFrame.
* Al carecer de motor de base de datos transaccional con garantías ACID, la consistencia de los datos no se apoya en restricciones de base de datos (claves foráneas o tipos SQL), sino en la **capa de validación contractual previa** implementada en CDVDataValidator.validate_and_clean().

### Fuentes de Datos Identificadas
* Archivo único de datos clínicos reales: dataset_moquillo_real_v4.csv, compuesto por 40 registros clínicos observados principalmente en Bogotá D.C. (Universidad Nacional de Colombia - UNAL, Zonas Norte, Centro, Occidente, Sur y Suroriente) y Chiquinquirá.
* Balance del dataset: 29 casos positivos confirmados (72.5%) y 11 casos sanos (27.5%).

---

## 5. Aspectos Transversales (Cross-Cutting Concerns)

### Autenticación y Autorización
* **Inexistente a nivel de software:** El sistema opera como CLI local de laboratorio/estación de trabajo. La seguridad de acceso depende exclusivamente de los privilegios del sistema operativo del anfitrión.

### Manejo Centralizado de Excepciones
* **Excepción de Dominio:** Se define CDVValidationError (subclase de ValueError), con alias retrocompatible DataValidationError.
* **Estrategia Fail-Fast:** Ante cualquier inconsistencia contractual (columnas faltantes, caracteres no numéricos en variables continuas o signos clínicos fuera del conjunto {0, 1}), el sistema interrumpe la ejecución inmediatamente con un mensaje descriptivo sin permitir que datos corruptos alcancen los pipelines.
* **Comprobación de Estado:** Métodos como evaluate_cv(), persist_model() o save_tree_diagram() lanzan RuntimeError si se invocan antes de ejecutar fit().

### Observabilidad, Logging y Configuración
* **Logging:** No se utiliza el módulo estándar logging de Python ni bibliotecas como loguru o structlog. La telemetría operativa se emite mediante sentencias print() directamente a la consola (stdout).
* **Auditoría Estructurada:** Como estándar de calidad CRISP-ML(Q), el orquestador main.py vuelca un snapshot auditable en reporte_metricas_cdv.json, guardando métricas pliegue a pliegue, matrices de confusión completas y reporte de calidad de datos (DataQualityReport).
* **Configuración:** Parámetros configurables gestionados mediante argumentos por línea de comandos con valores por defecto basados en rutas dinámicas relativas PROJECT_ROOT = Path(__file__).resolve().parent.

---

## 6. Integraciones y Comunicación Externa

* **APIs Expuestas:** Ninguna. No existen endpoints REST (FastAPI/Flask), GraphQL ni gRPC configurados.
* **Servicios Externos Consumidos:** Ninguno. La solución es totalmente autocontenida y hermética (offline), sin conexiones a nubes públicas (AWS S3, GCS), gestores de experimentos de ML (MLflow, Weights & Biases) ni intermediarios de mensajería (Kafka, RabbitMQ).

---

## 7. Diagrama de Arquitectura

`mermaid
graph TD
    subgraph Capa_CLI ["1. Capa de Presentación / CLI Entry Points"]
        MainCLI["main.py (Orquestador CRISP-ML(Q))"]
        PredictCLI["predict.py (Inferencia Clínica)"]
        EvalUnseen["evaluate_unseen_farm.py (Prueba Coldstart)"]
        GenTree["generateDecisionTree.py (Entrenamiento Árbol)"]
    end

    subgraph Capa_Validacion ["2. Capa de Dominio y Validación Contractual"]
        Validator["CDVDataValidator"]
        Metadata["ModelMetadata"]
        QualityRep["DataQualityReport"]
        ValError["CDVValidationError (Fail-Fast)"]
    end

    subgraph Capa_Ingenieria ["3. Capa de Ingeniería de Características (Feature Engineering)"]
        Builder["CDVFeaturePipelineBuilder"]
        ColTrans["ColumnTransformer"]
        subgraph Ramas_Preprocesamiento ["Ramas de Transformación"]
            NumBranch["Numérica: SimpleImputer(median) + StandardScaler"]
            CatBranch["Categórica: SimpleImputer('Desconocido') + OneHotEncoder(min_freq=2, coldstart)"]
            BinBranch["Binaria: SimpleImputer(0)"]
        end
    end

    subgraph Capa_Modelado ["4. Capa de Modelado y Optimización (CRISP-ML(Q))"]
        DTTrainer["DecisionTreeModelTrainer"]
        RFTrainer["RandomForestModelTrainer"]
        LRTrainer["LogisticRegressionModelTrainer"]
        GSearch["GridSearchCV (scoring='recall')"]
        SKFold["StratifiedKFold (5 pliegues OOF)"]
    end

    subgraph Capa_Inferencia ["5. Capa de Inferencia en Producción"]
        InfEngine["CDVInferenceEngine"]
        RiskStrat["Estratificación de Riesgo Clínico (Alto / Moderado / Bajo)"]
    end

    subgraph Capa_Persistencia ["6. Capa de Persistencia y Artefactos"]
        RawCSV[("data/raw/dataset_moquillo_real_v4.csv")]
        ModelPKL[("src/data/artifacts/*.pkl")]
        MetricsJSON[("demo_reports/reporte_metricas_cdv.json")]
        PlotsPNG[("demo_reports/*.png")]
    end

    RawCSV -->|Lee dataset| MainCLI
    RawCSV -->|Lee dataset| EvalUnseen
    MainCLI --> Validator
    PredictCLI --> InfEngine
    EvalUnseen --> Validator

    Validator -->|Contrato Validado| Builder
    Metadata -.-> Validator

    Builder --> ColTrans
    ColTrans --> NumBranch
    ColTrans --> CatBranch
    ColTrans --> BinBranch

    MainCLI --> DTTrainer
    MainCLI --> RFTrainer
    MainCLI --> LRTrainer

    DTTrainer & RFTrainer & LRTrainer --> GSearch
    GSearch --> SKFold
    ColTrans --> GSearch

    DTTrainer -->|Persiste modelo| ModelPKL
    RFTrainer -->|Persiste modelo| ModelPKL
    LRTrainer -->|Persiste modelo| ModelPKL

    DTTrainer & RFTrainer & LRTrainer --> MetricsJSON
    DTTrainer & RFTrainer & LRTrainer --> PlotsPNG

    ModelPKL -->|joblib.load()| InfEngine
    InfEngine --> Validator
    InfEngine --> RiskStrat
`

---

## 8. Deuda Técnica y Observaciones Clave

### Hallazgos Críticos

#### 1. Colapso del Modelo de Regresión Logística por Optimización Ciega a Recall
En el informe de auditoría reporte_metricas_cdv.json, se evidencia que la Regresión Logística optimizada con scoring="recall" convergió a un modelo trivial que **clasifica el 100% de los pacientes como Positivos (CDV = 1)**:
* Matriz de Confusión: [[0, 11], [0, 29]].
* Sensibilidad (Recall): 1.0 (100%), pero Especificidad y Precisión para la clase sana = 0.0%.
* **Causa:** Al fijar scoring="recall" sin regularización que penalice los falsos positivos o sin calibrar el umbral de decisión, el optimizador GridSearchCV seleccionó C=0.01 y pesos donde predecir siempre positivo maximiza el recall a costa de destruir la capacidad discriminativa del modelo.
* **Impacto Clínico:** En un escenario real, este modelo provocaría el aislamiento innecesario y costoso del 100% de los perros sanos.

#### 2. Tamaño Muestral Crítico y Desbalance
El dataset cuenta con únicamente **40 registros** (N=40, 29 positivos y 11 sanos). En una validación cruzada de 5 pliegues, cada pliegue de prueba contiene sólo 8 observaciones (aproximadamente 2 sanos y 6 enfermos). Cualquier métrica calculada a este nivel presenta intervalos de confianza muy amplios y alta varianza estadística.

#### 3. Inconsistencia Ortográfica en Nombres de Módulos
Existe el archivo src/models/LogisticalRegresion.py con doble error tipográfico ("Logistical" en vez de "Logistic", y "Regresion" con una sola 's'). Se creó un módulo parche src/models/LogisticRegression.py para reexportar sus clases, pero en main.py (línea 20) se continúa importando la ruta con el error de escritura (from src.models.LogisticalRegresion import LogisticRegressionModelTrainer).

#### 4. Archivos Huérfanos y Vacíos (0 Bytes)
Existen archivos vacíos en el árbol del proyecto que confunden a nuevos desarrolladores:
* data/data_analysis.md (0 bytes).
* data/processed/processed_data.csv (0 bytes).
* evaluation/model_comparation.ipynb (0 bytes).

#### 5. Carencia de API Web y Protocolos de Telemetría
El sistema carece de interfaces de integración modernas (REST/gRPC) para comunicarse con sistemas de historia clínica electrónica veterinaria (EHR/EMR), limitando su adopción a scripts ejecutados por terminal. Los logs se generan mediante print(), imposibilitando el filtrado por severidad (INFO, WARNING, ERROR).

---

### Fortalezas de Ingeniería Observadas

1. **Prevención Robusta de Data Leakage:** El uso coordinado de ColumnTransformer dentro de pipelines de scikit-learn garantiza que las imputaciones por mediana y el escalado Z-score no sufran fuga de datos entre pliegues de entrenamiento y validación.
2. **Excelente Manejo de Arranque en Frío (Coldstart):** La parametrización de OneHotEncoder(min_frequency=2, handle_unknown="infrequent_if_exist") en featureEngineer.py es una práctica de ingeniería destacable. Permite que categorías nunca vistas (como nuevas razas o centros de salud) no rompan la inferencia y sean absorbidas por el bucket de infrecuentes.
3. **Validación Contractual de Dominio (Fail-Fast):** El módulo dataValidation.py implementa reglas rigurosas de plausibilidad biológica (edades admisibles, signos binarios) antes de cualquier cómputo.

---

### Recomendaciones Técnicas Inmediatas

1. **Corregir Función Objetivo en GridSearchCV:** Cambiar la métrica de scoring de scoring="recall" a una métrica balanceada como scoring="balanced_accuracy" o scoring="f1", o definir un scorer compuesto personalizado (recall >= 0.80 condicionado a specificity >= 0.60) para evitar clasificadores degenerados en Regresión Logística.
2. **Refactorización de Archivos y Nombres:** Renombrar definitivamente src/models/LogisticalRegresion.py a src/models/logistic_regression.py siguiendo PEP 8 y actualizar las referencias en main.py y __init__.py.
3. **Eliminar Archivos Vacíos:** Borrar los ficheros de 0 bytes o completarlos con la documentación correspondiente.
4. **Exponer API de Inferencia (FastAPI):** Encapsular CDVInferenceEngine en un microservicio con **FastAPI** y esquemas **Pydantic** para validar peticiones HTTP (POST /api/v1/predict), facilitando la integración con aplicaciones web o móviles para clínicas veterinarias.
5. **Migrar Salida de Consola a Logging Estructurado:** Sustituir los print() por el módulo estándar logging con formato JSON para facilitar su recolección y monitoreo en entornos de producción.
