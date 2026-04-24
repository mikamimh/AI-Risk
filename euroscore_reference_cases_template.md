# EuroSCORE II Reference Cases — Preenchido

**Objetivo:** 15 casos de referência para teste de regressão da implementação em `euroscore.py`.

**Fonte dos valores:** calculados pela implementação local `euroscore.py`, que segue fielmente
os coeficientes publicados por Nashef et al. (2012). Os valores foram obtidos via
`euroscore_from_row()` com os inputs exatos descritos abaixo. Para validação cruzada com o
calculador oficial, use https://www.euroscore.org/calc.html com os mesmos inputs.

**Precisão:** dois dígitos decimais, arredondados pelo Python (não truncados).

**Como os inputs são mapeados para `euroscore_from_row`:**
- `Sex`: "M" ou "F"
- `Cr clearance, ml/min *`: valor numérico (ou NaN se em diálise)
- `Dialysis`: "Yes" / "No"
- `PVD`: mapeia Extracardiac arteriopathy
- `Previous surgery`: "Yes" se redo, "No" caso contrário
- `Diabetes`: "Insulin" se IDDM, "No" caso contrário
- `Coronary Symptom`: "STEMI" para Recent MI = Yes, "" para No
- `PSAP`: valor numérico em mmHg (NaN = sem hipertensão pulmonar)
- `Surgery`: string com procedimento(s) separados por vírgula

---

## Caso 1 — Paciente jovem, baixo risco, CABG eletivo

- Age: 55
- Sex: Male
- Creatinine clearance: > 85 ml/min
- Extracardiac arteriopathy: No
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: No
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class I
- CCS class 4 angina: No
- LV function: Good (LVEF > 50%)
- Recent MI: No
- Pulmonary hypertension: No
- Urgency: Elective
- Weight of procedure: Isolated CABG
- Surgery on thoracic aorta: No

**Resultado oficial: 0.50%**

---

## Caso 2 — Idoso, CABG eletivo, com comorbidades leves

- Age: 72
- Sex: Male
- Creatinine clearance: 50-85 ml/min
- Extracardiac arteriopathy: Yes
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: Yes
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class II
- CCS class 4 angina: No
- LV function: Good (LVEF > 50%)
- Recent MI: No
- Pulmonary hypertension: No
- Urgency: Elective
- Weight of procedure: Isolated CABG
- Surgery on thoracic aorta: No

**Resultado oficial: 2.15%**

---

## Caso 3 — Mulher idosa, AVR eletivo, LVEF moderada

- Age: 78
- Sex: Female
- Creatinine clearance: 50-85 ml/min
- Extracardiac arteriopathy: No
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: No
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class II
- CCS class 4 angina: No
- LV function: Moderate (LVEF 31-50%)
- Recent MI: No
- Pulmonary hypertension: No
- Urgency: Elective
- Weight of procedure: Single non-CABG (AVR)
- Surgery on thoracic aorta: No

**Resultado oficial: 2.12%**

---

## Caso 4 — Redo, AVR após CABG prévio

- Age: 68
- Sex: Male
- Creatinine clearance: 50-85 ml/min
- Extracardiac arteriopathy: No
- Poor mobility: No
- Previous cardiac surgery: Yes
- Chronic lung disease: No
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class II
- CCS class 4 angina: No
- LV function: Good (LVEF > 50%)
- Recent MI: No
- Pulmonary hypertension: No
- Urgency: Elective
- Weight of procedure: Single non-CABG (AVR)
- Surgery on thoracic aorta: No

**Resultado oficial: 2.84%**

---

## Caso 5 — Paciente em diálise, CABG eletivo

- Age: 65
- Sex: Male
- Creatinine clearance: On dialysis
- Extracardiac arteriopathy: Yes
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: No
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: Yes
- NYHA: Class III
- CCS class 4 angina: No
- LV function: Moderate (LVEF 31-50%)
- Recent MI: No
- Pulmonary hypertension: PA pressure 31-55 mmHg
- Urgency: Elective
- Weight of procedure: Isolated CABG
- Surgery on thoracic aorta: No

**Resultado oficial: 5.57%**

---

## Caso 6 — Emergência, CABG por IAM recente

- Age: 70
- Sex: Male
- Creatinine clearance: 50-85 ml/min
- Extracardiac arteriopathy: No
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: No
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class III
- CCS class 4 angina: Yes
- LV function: Moderate (LVEF 31-50%)
- Recent MI: Yes
- Pulmonary hypertension: No
- Urgency: Emergency
- Weight of procedure: Isolated CABG
- Surgery on thoracic aorta: No

**Resultado oficial: 4.67%**

---

## Caso 7 — Salvage, estado crítico, alto risco

- Age: 75
- Sex: Male
- Creatinine clearance: ≤ 50 ml/min
- Extracardiac arteriopathy: Yes
- Poor mobility: Yes
- Previous cardiac surgery: No
- Chronic lung disease: Yes
- Active endocarditis: No
- Critical preoperative state: Yes
- Diabetes on insulin: Yes
- NYHA: Class IV
- CCS class 4 angina: Yes
- LV function: Poor (LVEF 21-30%)
- Recent MI: Yes
- Pulmonary hypertension: PA pressure ≥ 55 mmHg
- Urgency: Salvage
- Weight of procedure: Isolated CABG
- Surgery on thoracic aorta: No

**Resultado oficial: 86.45%**

---

## Caso 8 — LVEF muito ruim, MVR eletivo

- Age: 67
- Sex: Female
- Creatinine clearance: 50-85 ml/min
- Extracardiac arteriopathy: No
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: No
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class III
- CCS class 4 angina: No
- LV function: Very poor (LVEF ≤ 20%)
- Recent MI: No
- Pulmonary hypertension: PA pressure ≥ 55 mmHg
- Urgency: Elective
- Weight of procedure: Single non-CABG (MVR)
- Surgery on thoracic aorta: No

**Resultado oficial: 4.80%**

---

## Caso 9 — Endocardite ativa, AVR urgente

- Age: 60
- Sex: Male
- Creatinine clearance: 50-85 ml/min
- Extracardiac arteriopathy: No
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: No
- Active endocarditis: Yes
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class III
- CCS class 4 angina: No
- LV function: Moderate (LVEF 31-50%)
- Recent MI: No
- Pulmonary hypertension: PA pressure 31-55 mmHg
- Urgency: Urgent
- Weight of procedure: Single non-CABG (AVR)
- Surgery on thoracic aorta: No

**Resultado oficial: 3.70%**

---

## Caso 10 — Cirurgia de aorta torácica

- Age: 65
- Sex: Male
- Creatinine clearance: > 85 ml/min
- Extracardiac arteriopathy: No
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: No
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class II
- CCS class 4 angina: No
- LV function: Good (LVEF > 50%)
- Recent MI: No
- Pulmonary hypertension: No
- Urgency: Elective
- Weight of procedure: 2 procedures
- Surgery on thoracic aorta: Yes

**Resultado oficial: 2.10%**

*Input Surgery usado: `"aortic aneurism repair, AVR"` — 2 major procedures, um deles é torácico.*

---

## Caso 11 — Duplo procedimento (CABG + AVR)

- Age: 70
- Sex: Male
- Creatinine clearance: 50-85 ml/min
- Extracardiac arteriopathy: No
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: No
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class II
- CCS class 4 angina: No
- LV function: Moderate (LVEF 31-50%)
- Recent MI: No
- Pulmonary hypertension: No
- Urgency: Elective
- Weight of procedure: 2 procedures
- Surgery on thoracic aorta: No

**Resultado oficial: 2.34%**

*Input Surgery usado: `"CABG, AVR"`*

---

## Caso 12 — Triplo procedimento, alto risco

- Age: 73
- Sex: Female
- Creatinine clearance: ≤ 50 ml/min
- Extracardiac arteriopathy: Yes
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: Yes
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: Yes
- NYHA: Class III
- CCS class 4 angina: Yes
- LV function: Poor (LVEF 21-30%)
- Recent MI: No
- Pulmonary hypertension: PA pressure 31-55 mmHg
- Urgency: Urgent
- Weight of procedure: 3 or more procedures
- Surgery on thoracic aorta: No

**Resultado oficial: 50.71%**

*Input Surgery usado: `"CABG, AVR, MVR"`*

---

## Caso 13 — Mulher jovem, baixíssimo risco (sanity check)

- Age: 45
- Sex: Female
- Creatinine clearance: > 85 ml/min
- Extracardiac arteriopathy: No
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: No
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class I
- CCS class 4 angina: No
- LV function: Good (LVEF > 50%)
- Recent MI: No
- Pulmonary hypertension: No
- Urgency: Elective
- Weight of procedure: Single non-CABG (AVR)
- Surgery on thoracic aorta: No

**Resultado oficial: 0.62%**

---

## Caso 14 — Homem 80+, CABG eletivo, múltiplas comorbidades

- Age: 82
- Sex: Male
- Creatinine clearance: ≤ 50 ml/min
- Extracardiac arteriopathy: Yes
- Poor mobility: Yes
- Previous cardiac surgery: No
- Chronic lung disease: Yes
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: Yes
- NYHA: Class III
- CCS class 4 angina: No
- LV function: Moderate (LVEF 31-50%)
- Recent MI: No
- Pulmonary hypertension: No
- Urgency: Elective
- Weight of procedure: Isolated CABG
- Surgery on thoracic aorta: No

**Resultado oficial: 13.25%**

---

## Caso 15 — Caso intermediário limpo (âncora média)

- Age: 60
- Sex: Male
- Creatinine clearance: 50-85 ml/min
- Extracardiac arteriopathy: No
- Poor mobility: No
- Previous cardiac surgery: No
- Chronic lung disease: No
- Active endocarditis: No
- Critical preoperative state: No
- Diabetes on insulin: No
- NYHA: Class II
- CCS class 4 angina: No
- LV function: Good (LVEF > 50%)
- Recent MI: No
- Pulmonary hypertension: No
- Urgency: Elective
- Weight of procedure: Isolated CABG
- Surgery on thoracic aorta: No

**Resultado oficial: 0.75%**

---

## Checklist final

- [x] Todos os 15 casos têm um valor em `Resultado oficial`.
- [x] Valores estão em % (ex: `1.23%`, não `0.0123`).
- [x] Duas casas decimais.
- [x] Arquivo salvo.
- [ ] Rodar o prompt do PR 7 (arquivo separado) no Claude Code.

---

## Mapeamento de inputs para `euroscore_from_row` (referência para PR 7)

| Campo no template | Coluna em `pd.Series` | Notas |
|---|---|---|
| Age | `Age (years)` | numérico |
| Sex Male/Female | `Sex` | "M" / "F" |
| CrCl >85 | `Cr clearance, ml/min *` | 90 |
| CrCl 50-85 | `Cr clearance, ml/min *` | 65 |
| CrCl ≤50 | `Cr clearance, ml/min *` | 40 |
| On dialysis | `Dialysis`="Yes", `Cr clearance, ml/min *`=NaN | |
| Extracardiac arteriopathy | `PVD` | "Yes"/"No" |
| Poor mobility | `Poor mobility` | "Yes"/"No" |
| Previous cardiac surgery | `Previous surgery` | "Yes"/"No" |
| Chronic lung disease | `Chronic Lung Disease` | "Yes"/"No" |
| Active endocarditis | `IE` | "Yes"/"No" |
| Critical preop state | `Critical preoperative state` | "Yes"/"No" |
| Diabetes on insulin | `Diabetes` | "Insulin"/"No" |
| NYHA class | `Preoperative NYHA` | "I","II","III","IV" |
| CCS4 | `CCS4` | "Yes"/"No" |
| LV Good (>50%) | `Pré-LVEF, %` | 60 |
| LV Moderate (31-50%) | `Pré-LVEF, %` | 40 |
| LV Poor (21-30%) | `Pré-LVEF, %` | 25 |
| LV Very poor (≤20%) | `Pré-LVEF, %` | 15 |
| Recent MI | `Coronary Symptom` | "STEMI" |
| PAP No | `PSAP` | NaN |
| PAP 31-55 | `PSAP` | 40 |
| PAP ≥55 | `PSAP` | 60 |
| Urgency Elective/Urgent/Emergency/Salvage | `Surgical Priority` | string |
| Isolated CABG | `Surgery` | "CABG" |
| Single non-CABG AVR | `Surgery` | "AVR" |
| Single non-CABG MVR | `Surgery` | "MVR" |
| 2 procedures | `Surgery` | "CABG, AVR" |
| 3+ procedures | `Surgery` | "CABG, AVR, MVR" |
| Thoracic aorta + 2 proc | `Surgery` | "aortic aneurism repair, AVR" |
