"""UI column-config helpers for st.dataframe calls.

Both functions accept a ``tr`` callable (the app's bilingual translation helper)
so they can be used independently of app.py's module-level ``tr`` closure.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st


def stats_table_column_config(kind: str, tr: Callable[[str, str], str]) -> dict:
    common = {
        "Score": st.column_config.TextColumn(
            tr("Score", "Escore"),
            help=tr(
                "Risk score or prediction model being evaluated.",
                "Escore de risco ou modelo preditivo em avaliação.",
            ),
        ),
        "n": st.column_config.NumberColumn(
            "n",
            help=tr(
                "Number of observations included in that analysis.",
                "Número de observações incluídas naquela análise.",
            ),
            format="%d",
        ),
        "AUC": st.column_config.NumberColumn(
            "AUC",
            help=tr(
                "Area under the ROC curve. Higher values indicate better overall discrimination.",
                "Área sob a curva ROC. Valores maiores indicam melhor discriminação global.",
            ),
            format="%.3f",
        ),
        "AUPRC": st.column_config.NumberColumn(
            "AUPRC",
            help=tr(
                "Area under the precision-recall curve. Especially useful when the event is relatively uncommon.",
                "Área sob a curva precisão-revocação. Especialmente útil quando o evento é relativamente incomum.",
            ),
            format="%.3f",
        ),
        "Brier": st.column_config.NumberColumn(
            "Brier",
            help=tr(
                "Brier score measures probabilistic accuracy. Lower values are better.",
                "O Brier score mede a acurácia probabilística. Valores menores são melhores.",
            ),
            format="%.4f",
        ),
        "Sensitivity": st.column_config.NumberColumn(
            tr("Sensitivity", "Sensibilidade"),
            help=tr(
                "Proportion of patients with the event correctly classified as positive at the selected threshold.",
                "Proporção de pacientes com evento corretamente classificados como positivos no limiar selecionado.",
            ),
            format="%.3f",
        ),
        "Specificity": st.column_config.NumberColumn(
            tr("Specificity", "Especificidade"),
            help=tr(
                "Proportion of patients without the event correctly classified as negative at the selected threshold.",
                "Proporção de pacientes sem evento corretamente classificados como negativos no limiar selecionado.",
            ),
            format="%.3f",
        ),
        "PPV": st.column_config.TextColumn(
            "PPV",
            help=tr(
                "Positive predictive value: probability that a patient classified as positive truly has the event. '—' means no positive predictions at this threshold.",
                "Valor preditivo positivo: probabilidade de um paciente classificado como positivo realmente apresentar o evento. '—' indica que não houve predições positivas neste limiar.",
            ),
        ),
        "NPV": st.column_config.TextColumn(
            "NPV",
            help=tr(
                "Negative predictive value: probability that a patient classified as negative truly does not have the event. '—' means no negative predictions at this threshold.",
                "Valor preditivo negativo: probabilidade de um paciente classificado como negativo realmente não apresentar o evento. '—' indica que não houve predições negativas neste limiar.",
            ),
        ),
        "AUC_IC95_inf": st.column_config.NumberColumn(tr("AUC CI low", "AUC IC95% inf"), format="%.3f"),
        "AUC_IC95_sup": st.column_config.NumberColumn(tr("AUC CI high", "AUC IC95% sup"), format="%.3f"),
        "AUPRC_IC95_inf": st.column_config.NumberColumn(tr("AUPRC CI low", "AUPRC IC95% inf"), format="%.3f"),
        "AUPRC_IC95_sup": st.column_config.NumberColumn(tr("AUPRC CI high", "AUPRC IC95% sup"), format="%.3f"),
        "Brier_IC95_inf": st.column_config.NumberColumn(tr("Brier CI low", "Brier IC95% inf"), format="%.4f"),
        "Brier_IC95_sup": st.column_config.NumberColumn(tr("Brier CI high", "Brier IC95% sup"), format="%.4f"),
    }

    if kind == "comparison":
        common.update(
            {
                tr("Comparison", "Comparação"): st.column_config.TextColumn(
                    tr("Comparison", "Comparação"),
                    help=tr(
                        "Pair of models or scores being compared.",
                        "Par de modelos ou escores sendo comparados.",
                    ),
                ),
                "Delta AUC (A-B)": st.column_config.NumberColumn(
                    "Delta AUC (A-B)",
                    help=tr(
                        "Difference in AUC between model A and model B. Positive values favor model A.",
                        "Diferença de AUC entre o modelo A e o modelo B. Valores positivos favorecem o modelo A.",
                    ),
                    format="%.3f",
                ),
                tr("95% CI low", "IC95% inf"): st.column_config.NumberColumn(tr("95% CI low", "IC95% inf"), format="%.3f"),
                tr("95% CI high", "IC95% sup"): st.column_config.NumberColumn(tr("95% CI high", "IC95% sup"), format="%.3f"),
                "p (bootstrap)": st.column_config.NumberColumn(
                    "p (bootstrap)",
                    help=tr(
                        "Approximate p-value from bootstrap comparison.",
                        "Valor de p aproximado obtido por comparação via bootstrap.",
                    ),
                    format="%.4f",
                ),
                "p (DeLong)": st.column_config.NumberColumn(
                    "p (DeLong)",
                    help=tr(
                        "P-value from DeLong test for correlated ROC curves.",
                        "Valor de p do teste de DeLong para curvas ROC correlacionadas.",
                    ),
                    format="%.4f",
                ),
            }
        )
    elif kind == "calibration":
        common.update(
            {
                "Calibration intercept": st.column_config.NumberColumn(
                    tr("Calibration-in-the-large", "Calibration-in-the-large"),
                    help=tr(
                        "Calibration intercept (calibration-in-the-large). Values closer to 0 indicate better average agreement between predicted and observed risk.",
                        "Intercepto de calibração (calibration-in-the-large). Valores mais próximos de 0 indicam melhor concordância média entre risco previsto e observado.",
                    ),
                    format="%.4f",
                ),
                "Calibration slope": st.column_config.NumberColumn(
                    tr("Calibration slope", "Slope de calibração"),
                    help=tr(
                        "Values closer to 1 indicate better calibration. Values below 1 may suggest overfitting.",
                        "Valores mais próximos de 1 indicam melhor calibração. Valores abaixo de 1 podem sugerir sobreajuste.",
                    ),
                    format="%.4f",
                ),
                "HL chi-square": st.column_config.NumberColumn(
                    tr("HL chi-square", "Qui-quadrado HL"),
                    help=tr(
                        "Hosmer-Lemeshow statistic, interpreted as complementary to visual calibration and Brier score.",
                        "Estatística de Hosmer-Lemeshow, interpretada como complementar à calibração visual e ao Brier score.",
                    ),
                    format="%.4f",
                ),
                "HL dof": st.column_config.NumberColumn(tr("HL dof", "GL HL"), format="%d"),
                "HL p-value": st.column_config.NumberColumn(
                    tr("HL p-value", "p do HL"),
                    help=tr(
                        "P-value of the Hosmer-Lemeshow test. It should not be used alone to define model adequacy.",
                        "Valor de p do teste de Hosmer-Lemeshow. Não deve ser usado isoladamente para definir a adequação do modelo.",
                    ),
                    format="%.4f",
                ),
            }
        )
    elif kind == "dca":
        common.update(
            {
                "Threshold": st.column_config.NumberColumn(
                    tr("Threshold", "Limiar"),
                    help=tr(
                        "Risk threshold at which a patient would be considered positive/high risk for decision-making.",
                        "Limiar de risco a partir do qual um paciente seria considerado positivo/alto risco para tomada de decisão.",
                    ),
                    format="%.2f",
                ),
                "Strategy": st.column_config.TextColumn(
                    tr("Strategy", "Estratégia"),
                    help=tr(
                        "Model or reference strategy (treat all / treat none) shown in decision curve analysis.",
                        "Modelo ou estratégia de referência (tratar todos / tratar ninguém) mostrada na decision curve analysis.",
                    ),
                ),
                "Net Benefit": st.column_config.NumberColumn(
                    tr("Net Benefit", "Benefício líquido"),
                    help=tr(
                        "Clinical utility measure in decision curve analysis. Higher values indicate greater usefulness at that threshold.",
                        "Medida de utilidade clínica na decision curve analysis. Valores mais altos indicam maior utilidade naquele limiar.",
                    ),
                    format="%.4f",
                ),
            }
        )
    elif kind == "reclass":
        common.update(
            {
                tr("Comparison", "Comparação"): st.column_config.TextColumn(tr("Comparison", "Comparação")),
                "NRI events": st.column_config.NumberColumn(
                    tr("NRI events", "NRI eventos"),
                    help=tr(
                        "Net reclassification improvement among patients with the event.",
                        "Net reclassification improvement entre pacientes com evento.",
                    ),
                    format="%.4f",
                ),
                "NRI non-events": st.column_config.NumberColumn(
                    tr("NRI non-events", "NRI não-eventos"),
                    help=tr(
                        "Net reclassification improvement among patients without the event.",
                        "Net reclassification improvement entre pacientes sem evento.",
                    ),
                    format="%.4f",
                ),
                "NRI total": st.column_config.NumberColumn(
                    tr("NRI total", "NRI total"),
                    help=tr(
                        "Overall net reclassification improvement. Positive values suggest better reclassification by the new model.",
                        "Melhora líquida global de reclassificação. Valores positivos sugerem melhor reclassificação pelo novo modelo.",
                    ),
                    format="%.4f",
                ),
                "IDI": st.column_config.NumberColumn(
                    "IDI",
                    help=tr(
                        "Integrated discrimination improvement. Positive values suggest improved average separation between events and non-events.",
                        "Integrated discrimination improvement. Valores positivos sugerem melhor separação média entre eventos e não eventos.",
                    ),
                    format="%.4f",
                ),
            }
        )
    elif kind == "subgroup":
        common.update(
            {
                "Subgroup": st.column_config.TextColumn(
                    tr("Subgroup", "Subgrupo"),
                    help=tr(
                        "Subgroup definition being evaluated.",
                        "Definição do subgrupo em avaliação.",
                    ),
                ),
                "Group": st.column_config.TextColumn(
                    tr("Group", "Grupo"),
                    help=tr(
                        "Specific category within the selected subgroup panel.",
                        "Categoria específica dentro do painel de subgrupos selecionado.",
                    ),
                ),
                "Deaths": st.column_config.NumberColumn(
                    tr("Deaths (primary outcome)", "Óbitos (desfecho primário)"),
                    help=tr(
                        "Number of 30-day deaths within that subgroup.",
                        "Número de óbitos em 30 dias dentro daquele subgrupo.",
                    ),
                    format="%d",
                ),
            }
        )
    return common


def general_table_column_config(kind: str, tr: Callable[[str, str], str]) -> dict:
    if kind == "leaderboard":
        return {
            "Modelo": st.column_config.TextColumn(
                tr("Model", "Modelo"),
                help=tr("Machine-learning algorithm evaluated in cross-validation.", "Algoritmo de aprendizado de máquina avaliado na validação cruzada."),
            ),
            "AUC": st.column_config.NumberColumn("AUC", help=tr("Overall discrimination. Higher is better.", "Discriminação global. Quanto maior, melhor."), format="%.3f"),
            "AUPRC": st.column_config.NumberColumn("AUPRC", help=tr("Precision-recall performance. Useful when the event is uncommon.", "Desempenho precisão-revocação. Útil quando o evento é incomum."), format="%.3f"),
            "Brier": st.column_config.NumberColumn("Brier", help=tr("Probabilistic accuracy. Lower is better.", "Acurácia probabilística. Quanto menor, melhor."), format="%.4f"),
            "Sensibilidade": st.column_config.NumberColumn(tr("Sensitivity", "Sensibilidade"), help=tr("Out-of-fold sensitivity at the optimal threshold (Youden's J).", "Sensibilidade out-of-fold no limiar ótimo (Youden's J)."), format="%.3f"),
            "Especificidade": st.column_config.NumberColumn(tr("Specificity", "Especificidade"), help=tr("Out-of-fold specificity at the optimal threshold (Youden's J).", "Especificidade out-of-fold no limiar ótimo (Youden's J)."), format="%.3f"),
            "Limiar_Youden": st.column_config.NumberColumn(tr("Youden threshold", "Limiar de Youden"), help=tr("Optimal out-of-fold threshold maximizing Youden's J.", "Limiar ótimo out-of-fold que maximiza o índice J de Youden."), format="%.3f"),
        }
    if kind == "eligibility":
        return {
            tr("Step", "Etapa"): st.column_config.TextColumn(
                tr("Step", "Etapa"),
                help=tr("Processing step in the eligibility flow from raw data to the final analytic dataset.", "Etapa do fluxo de elegibilidade desde os dados brutos até a base analítica final."),
            ),
            tr("Count", "Quantidade"): st.column_config.NumberColumn(
                tr("Count", "Quantidade"),
                help=tr("Number of records remaining or excluded at that step.", "Número de registros remanescentes ou excluídos naquela etapa."),
                format="%d",
            ),
        }
    if kind == "surgery_profile":
        return {
            tr("Surgery group", "Grupo cirúrgico"): st.column_config.TextColumn(
                tr("Surgery group", "Grupo cirúrgico"),
                help=tr("Grouped surgery category (descriptive, not model-oriented).", "Categoria cirúrgica agrupada (descritiva, não orientada ao modelo)."),
            ),
            tr("N", "N"): st.column_config.NumberColumn(
                tr("N", "N"),
                help=tr("Number of patients in the group.", "Número de pacientes no grupo."),
                format="%d",
            ),
            tr("Deaths", "Óbitos"): st.column_config.NumberColumn(
                tr("Deaths", "Óbitos"),
                help=tr("Number of deaths (primary outcome: morte_30d).", "Número de óbitos (desfecho primário: morte_30d)."),
                format="%d",
            ),
            tr("Mortality rate (%)", "Mortalidade (%)"): st.column_config.NumberColumn(
                tr("Mortality rate (%)", "Mortalidade (%)"),
                help=tr("Observed mortality rate in the group.", "Taxa de mortalidade observada no grupo."),
                format="%.1f",
            ),
        }
    if kind == "surgery_profile_raw":
        return {
            tr("Surgery (raw)", "Cirurgia (bruta)"): st.column_config.TextColumn(
                tr("Surgery (raw)", "Cirurgia (bruta)"),
                help=tr("Raw free-text surgery description from the source dataset.", "Descrição livre da cirurgia, conforme o arquivo fonte."),
            ),
            tr("N", "N"): st.column_config.NumberColumn(
                tr("N", "N"),
                help=tr("Number of patients with this exact surgery string.", "Número de pacientes com esta descrição exata."),
                format="%d",
            ),
            tr("Deaths", "Óbitos"): st.column_config.NumberColumn(
                tr("Deaths", "Óbitos"),
                help=tr("Number of deaths (primary outcome: morte_30d).", "Número de óbitos (desfecho primário: morte_30d)."),
                format="%d",
            ),
            tr("Mortality rate (%)", "Mortalidade (%)"): st.column_config.NumberColumn(
                tr("Mortality rate (%)", "Mortalidade (%)"),
                help=tr("Observed mortality rate for this raw surgery string.", "Taxa de mortalidade observada para esta descrição bruta."),
                format="%.1f",
            ),
        }
    if kind == "available_scores":
        return {
            tr("Score", "Escore"): st.column_config.TextColumn(
                tr("Score", "Escore"),
                help=tr("Score or model listed in the app summary.", "Escore ou modelo listado no resumo do aplicativo."),
            ),
            tr("Patients with value", "Pacientes com valor"): st.column_config.NumberColumn(
                tr("Patients with value", "Pacientes com valor"),
                help=tr("Number of patients with an available value for that score/model.", "Número de pacientes com valor disponível para aquele escore/modelo."),
                format="%d",
            ),
        }
    if kind == "patient_scores":
        return {
            tr("Score", "Escore"): st.column_config.TextColumn(
                tr("Score", "Escore"),
                help=tr("Model or score displayed for the current patient.", "Modelo ou escore exibido para o paciente atual."),
            ),
            tr("Probability", "Probabilidade"): st.column_config.TextColumn(
                tr("Probability", "Probabilidade"),
                help=tr("Predicted risk expressed as percentage.", "Risco predito expresso em porcentagem."),
            ),
            tr("Class", "Classe"): st.column_config.TextColumn(
                tr("Class", "Classe"),
                help=tr("Risk category derived from the predicted probability.", "Categoria de risco derivada da probabilidade predita."),
            ),
            tr("Model", "Modelo"): st.column_config.TextColumn(
                tr("Model", "Modelo"),
                help=tr("Machine-learning model used for the current patient prediction.", "Modelo de aprendizado de máquina usado para a predição do paciente atual."),
            ),
        }
    if kind == "export":
        return {
            "Name": st.column_config.TextColumn(
                tr("Patient name", "Nome do paciente"),
                help=tr(
                    "Patient identifier from the source spreadsheet.",
                    "Identificador do paciente vindo da planilha de origem.",
                ),
            ),
            "Surgery": st.column_config.TextColumn(
                tr("Surgery", "Cirurgia"),
                help=tr(
                    "Planned or recorded surgery description used in the analytic dataset.",
                    "Descrição da cirurgia planejada/registrada usada na base analítica.",
                ),
            ),
            "morte_30d": st.column_config.NumberColumn(
                tr("30-day death", "Óbito 30d"),
                help=tr(
                    "Observed 30-day outcome in the dataset (1 = death, 0 = no death).",
                    "Desfecho observado em 30 dias na base (1 = óbito, 0 = sem óbito).",
                ),
                format="%d",
            ),
            "ia_risk_oof": st.column_config.NumberColumn(
                "AI Risk",
                help=tr(
                    "AI Risk probability from out-of-fold validation predictions.",
                    "Probabilidade do AI Risk derivada das predições out-of-fold da validação.",
                ),
                format="%.4f",
            ),
            "euroscore_calc": st.column_config.NumberColumn(
                "EuroSCORE II",
                help=tr(
                    "EuroSCORE II probability calculated by the app from the published logistic equation (Nashef et al., 2012). Not read from the input file.",
                    "Probabilidade do EuroSCORE II calculada pelo app pela equação logística publicada (Nashef et al., 2012). Não lida do arquivo de entrada.",
                ),
                format="%.4f",
            ),
            "sts_score": st.column_config.NumberColumn(
                "STS",
                help=tr(
                    "STS Score Operative Mortality calculated by the app via automated query to the STS Score web calculator. Not read from the input file.",
                    "Mortalidade Operatória do STS Score calculada pelo app via consulta automatizada à calculadora web do STS Score. Não lida do arquivo de entrada.",
                ),
                format="%.4f",
            ),
            "classe_ia": st.column_config.TextColumn(tr("IA class", "Classe IA")),
            "classe_euro": st.column_config.TextColumn(tr("EuroSCORE class", "Classe EuroSCORE")),
            "classe_sts": st.column_config.TextColumn(tr("STS Score class", "Classe STS Score")),
        }
    return {}
