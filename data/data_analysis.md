# Análisis Exploratorio de Datos (EDA) — Distemper Canino (CDV)

**Dataset:** `dataset_moquillo_real_v7.csv`  
**Observaciones:** 163 registros clínicos caninos  
**Variables:** 12 columnas (11 predictoras y 1 objetivo)  
**Propósito:** Caracterización clínica, epidemiológica y zootécnica para el diagnóstico predictivo de Moquillo Canino (*Canine Distemper Virus* - CDV).

---

## 1. Distribución de la Variable Objetivo (`diagnostico_cdv_confirmado`)

El dataset presenta una tasa de prevalencia clínica de 74.2% para casos confirmados de CDV, característica de centros de atención veterinaria de referencia y brotes urbanos:

| Diagnóstico | Código | Conteo | Porcentaje |
| :--- | :---: | :---: | :---: |
| **Positivo (CDV Confirmado)** | `1` | 89 | 74.2% |
| **Sano / Control** | `0` | 31 | 25.8% |
| **Total** | — | **120** | **100.0%** |

---

## 2. Variables Fisiológicas y Zootécnicas

### 2.1 Edad del Paciente (`edad_meses`)
- **Promedio:** 34.6 meses (~2.9 años).
- **Mediana:** 11.2 meses (fuerte concentración en cachorros y perros jóvenes menores a 1 año).
- **Rango:** Mínimo 1.0 mes (4 semanas) hasta 156.0 meses (13 años).
- **Desviación Estándar:** 42.0 meses.
- **Hallazgo Clínico:** Mayor susceptibilidad y gravedad en cachorros entre 1 y 12 meses con ventana inmunológica vulnerable por declive de anticuerpos maternos.

### 2.2 Dimorfismo Sexual (`sexo`)
| Sexo | Sanos (`0`) | CDV Positivo (`1`) | Prevalencia CDV |
| :--- | :---: | :---: | :---: |
| **Hembra** | 17 | 52 | 75.4% |
| **Macho** | 14 | 37 | 72.5% |

*No se observa sesgo estadístico clínicamente significativo asociado al sexo.*

### 2.3 Talla Somática (`talla`)
| Talla | Sanos (`0`) | CDV Positivo (`1`) | Total |
| :--- | :---: | :---: | :---: |
| **Mediano** | 16 | 48 | 64 (53.3%) |
| **Grande** | 11 | 21 | 32 (26.7%) |
| **Pequeño** | 4 | 20 | 24 (20.0%) |

---

## 3. Estado Vacunal e Inmunidad Previa (`estado_vacunal`)

El historial de inmunización es uno de los factores con mayor poder protector frente a la enfermedad:

| Estado Vacunal | Sanos (`0`) | CDV Positivo (`1`) | Tasa de Infección |
| :--- | :---: | :---: | :---: |
| **Completo (Vigente)** | **12** | 5 | **29.4%** |
| **Incompleto** | 9 | 30 | 76.9% |
| **Desconocido** | 7 | 16 | 69.6% |
| **No vacunado** | 3 | **38** | **92.7%** |

*El 92.7% de los caninos no vacunados desarrollaron infección confirmada por CDV, frente a solo el 29.4% en aquellos con esquema vacunal completo.*

---

## 4. Prevalencia de Signos Clínicos Cardinales

Comparación de frecuencias de signos entre caninos sanos y caninos positivos a CDV:

| Signo Clínico | Frecuencia en Sanos (`N=31`) | Frecuencia en CDV (`N=89`) | Riesgo Relativo / Observación Clínica |
| :--- | :---: | :---: | :--- |
| **Fiebre / Hipertermia** | 6 (19.4%) | **69 (77.5%)** | Fase virémica activa; signo agudo predominante. |
| **Signos Respiratorios / Oculonasales** | 9 (29.0%) | **51 (57.3%)** | Secreción mucopurulenta bilateral y tos. |
| **Signos Digestivos** | 9 (29.0%) | **37 (41.6%)** | Gastroenteritis, emesis y enteritis hemorrágica. |
| **Signos Neurológicos** | **2 (6.5%)** | **58 (65.2%)** | Mioclonías y ataxia; altísimo valor predictivo positivo. |
| **Signos Dermatológicos** | **1 (3.2%)** | **14 (15.7%)** | Hiperqueratosis ("hard pad disease") en fase crónica. |

---

## 5. Distribución Geográfica y Epidemiológica (`ubicacion_procedencia`)

Los registros provienen de 13 zonas y centros de atención clínica en Bogotá y Cundinamarca:

| Ubicación | Sanos | CDV Positivo | Total Pacientes |
| :--- | :---: | :---: | :---: |
| **Bogota - Zona Norte** | 7 | 11 | 18 |
| **Chia (Sabana Norte)** | 2 | 16 | 18 |
| **Bogota - Engativa** | 4 | 10 | 14 |
| **Bogota - Bosa** | 5 | 8 | 13 |
| **Bogota - Usaquen** | 3 | 9 | 12 |
| **Bogota - Ciudad Bolivar** | 1 | 10 | 11 |
| **Bogota - Zona Suroriente** | 0 | 8 | 8 |
| **Bogota - Zona Sur** | 3 | 4 | 7 |
| **Bogota - Kennedy** | 2 | 3 | 5 |
| **Bogota - Zona Occidente** | 2 | 3 | 5 |
| **Bogota - Suba** | 1 | 3 | 4 |
| **Bogota - Zona Centro** | 1 | 2 | 3 |
| **Chiquinquira** | 0 | 2 | 2 |

---

## 6. Conclusiones y Recomendaciones para el Modelado

1. **Poder Discriminativo de Signos Clave:** La combinación de hipertermia con mioclonías/signos neurológicos y ausencia de vacunación constituye el principal predictor patognomónico de CDV.
2. **Mitigación de Coldstart:** Dado el número de razas y centros veterinarios, la agrupación de categorías minoritarias con `min_frequency=2` en el `OneHotEncoder` es mandatoria para evitar sobreajuste y soportar inferencia sobre nuevas procedencias.
3. **Optimización con Métricas Armónicas:** Se debe evitar el uso de `scoring="recall"` puro en modelos lineales para no inducir colapsos triviales; se adopta `scoring="f1"` o `balanced_accuracy` con garantía clínica de sensibilidad $\ge 80\%$.
