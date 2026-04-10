# Data Format Requirements

## Input Excel Structure

AI Risk expects a **single Excel file** with the following sheets:

### Required Sheets

#### 1. **Preoperative**
Main clinical data before surgery.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| Name | String | Patient identifier | "John Doe" |
| Procedure Date | Date | Surgery date | 2026-03-01 |
| Age (years) | Numeric | Patient age | 65 |
| Sex | String | M/F/Male/Female | "M" |
| Surgery | String | Procedures (comma-separated) | "CABG, mitral" |
| Surgical Priority | String | Elective/Urgent/Emergency/Salvage | "Elective" |
| Hypertension | String | yes/no/sim/não | "yes" |
| Diabetes | String | yes/no/insulin | "no" |
| CVA | String | History of stroke | "no" |
| PVD | String | Peripheral vascular disease | "no" |
| Chronic Lung Disease | String | COPD/pulmonary disease | "no" |
| Creatinine (mg/dL) | Numeric | Preoperative creatinine | 1.2 |
| Cr clearance, ml/min | Numeric | Creatinine clearance | 65 |
| Dialysis | String | On dialysis | "no" |
| Critical preoperative state | String | Critical status | "no" |
| Poor mobility | String | Reduced physical mobility | "no" |
| Preoperative NYHA | String | 1/2/3/4 | "2" |
| HF | String | Heart failure | "no" |
| CCS4 | String | Canadian Angina Score 4 | "no" |

#### 2. **Pre-Echocardiogram**
Echocardiographic measurements.

| Column | Type | Description |
|--------|------|-------------|
| Patient | String | Patient identifier (for matching) |
| Exam date | Date | Echo date |
| LVEF, % | Numeric | Left ventricular ejection fraction (0-100) |
| PSAP | Numeric | Pulmonary artery systolic pressure (mmHg) |
| Aortic Stenosis | String | yes/no/mild/moderate/severe |
| Mitral Stenosis | String | yes/no/mild/moderate/severe |
| Aortic Regurgitation | String | yes/no/trace/mild/moderate/severe |
| Mitral Regurgitation | String | yes/no/trace/mild/moderate/severe |
| Tricuspid Regurgitation | String | yes/no/mild/moderate/severe |

#### 3. **Postoperative**
In-hospital outcomes.

| Column | Type | Description |
|--------|------|-------------|
| Patient | String | Patient identifier (for matching) |
| Procedure Date | Date | Surgery date (for matching) |
| Death | String/Numeric | yes/no/1/0 |

### Optional Sheets

#### **EuroSCORE II**
Manually calculated EuroSCORE II scores (0-100):

| Column | Type |
|--------|------|
| Patient | String |
| EuroSCORE II | Numeric |

#### **EuroSCORE II Automático**
Automatically calculated EuroSCORE II.

| Column | Type |
|--------|------|
| Patient | String |
| EuroSCORE II Automático | Numeric |

#### **STS Score**
Society of Thoracic Surgeons operative mortality (0-100):

| Column | Type |
|--------|------|
| Patient | String |
| Operative Mortality | Numeric |

---

## Alternate Data Formats

### CSV Format
- First column must be patient identifier
- Header row required
- Matching column "Death" or "morte_30d" required

### SQLite Database
- Tables must match required sheet names
- All data types as in Excel (text, date, numeric)

### Parquet Format
- Columnar format for large datasets
- Same schema as CSV/Excel

---

## Data Validation Rules

### Eligibility Criteria
✅ Included:
- Records with non-null Surgery field
- Records with non-null Procedure Date

❌ Excluded:
- Missing Surgery or Procedure Date
- No Preoperative-Postoperative match
- Age < 18 (optional)

### Matching Logic
- Patient matched using: **Name** + **Procedure Date**
- Name matching is case-insensitive, whitespace-tolerant
- Must have records in all three required sheets

### Data Cleaning
- Numeric values can use comma or period as decimal: `1.2` or `1,2`
- Missing values: empty cell or any of: `-`, `nan`, `none`, `unknown`, `not applicable`
- String values (yes/no): case-insensitive, accepts "sim"/"não", "true"/"false", "1"/"0"

---

## Example Data File

**Excel File: `sample_data.xlsx`**

```
Preoperative:
Name              | Age | ... | Procedure Date | Surgery        | Death
John Doe          | 65  | ... | 2026-01-15     | CABG           | (see Postoperative)
Jane Smith        | 72  | ... | 2026-01-16     | Aortic valve   | (see Postoperative)

Pre-Echocardiogram:
Name              | LVEF | PSAP
John Doe          | 45   | 32
Jane Smith        | 28   | 55

Postoperative:
Name              | Procedure Date | Death
John Doe          | 2026-01-15     | No
Jane Smith        | 2026-01-16     | Yes
```

---

## Troubleshooting

### "missing_sheets: Postoperative | found_sheets: ..."
- Check sheet names exactly match (case-sensitive)
- Ensure all 3 required sheets exist

### "excluded_no_pre_post_match: 10"
- Patient names don't match between sheets
- Procedure dates don't align
- Fix: Ensure name format is consistent

### "No eligible rows were found"
- All records missing Surgery or Procedure Date
- Check for typos in column headers

### "Feature has too many missing values"
- > 40% of feature column is empty
- Consider dropping feature or adding more data

---

## Data Privacy Considerations

⚠️ **Important:**
- Names and dates used **only for internal matching**
- Not stored in model cache or exported results
- Recommend: Use anonymized identifiers if possible
- Models trained on clinical features only, not patient IDs
