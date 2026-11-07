"""Ablação e testes estatísticos do experimento ENIAC 2026 (não altera o original).

Uso: python ablacao.py asthma_disease_data.csv ablacao_resultados.csv

Reproduz exatamente a base e as variáveis sintéticas do eniac_experiment.py e compara
cenários retirando as variáveis derivadas do próprio rótulo (high_risk / symptom_score).
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

CSV = sys.argv[1]

# ── Base e variáveis sintéticas: cópia fiel do eniac_experiment.py ────────────
df = pd.read_csv(CSV).drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
SYMPTOMS = ['Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing', 'NighttimeSymptoms']
df['symptom_score'] = df[SYMPTOMS].sum(axis=1)
df['high_risk'] = (df['symptom_score'] >= 3).astype(int)

np.random.seed(42)
pollution_norm = df['PollutionExposure'] / df['PollutionExposure'].max()
lung_deficit = 1 - (df['LungFunctionFEV1'] / df['LungFunctionFEV1'].max())
symptom_norm = df['symptom_score'] / 5.0
pollen_norm = df['PollenExposure'] / df['PollenExposure'].max()
risk_raw = 0.35 * symptom_norm + 0.30 * pollution_norm + 0.20 * lung_deficit + 0.15 * pollen_norm
df['riskScore'] = ((risk_raw + np.random.normal(0, 0.05, len(df))).clip(0, 1) * 100).round(1)
prob_crisis = (0.6 * symptom_norm + 0.4 * (df['riskScore'] / 100)).clip(0, 1) * 0.35
df['hadCrisisToday'] = np.random.binomial(1, prob_crisis).astype(int)
base_inhaler = 0.78 + 0.08 * df['HistoryOfAllergies'] - 0.06 * df['Smoking']
df['hasRescueInhaler'] = np.random.binomial(1, base_inhaler.clip(0.4, 0.97)).astype(int)
prob_emerg = (0.05 + 0.25 * df['high_risk'] + 0.15 * df['hadCrisisToday']).clip(0, 0.5)
had_emerg = np.random.binomial(1, prob_emerg)
count_emerg = np.random.choice([1, 2, 3, 4, 5], size=len(df), p=[0.50, 0.25, 0.13, 0.08, 0.04])
df['emergencyCount'] = (had_emerg * count_emerg).astype(int)

ORIGINAIS = ['Age', 'Gender', 'Ethnicity', 'Smoking', 'BMI',
             'LungFunctionFEV1', 'LungFunctionFVC',
             'PollutionExposure', 'PollenExposure', 'DustExposure', 'ExerciseInduced']
CENARIOS = {
    'S0 Original (15)': ORIGINAIS + ['riskScore', 'hadCrisisToday', 'hasRescueInhaler', 'emergencyCount'],
    'S1 sem emergencyCount (14)': ORIGINAIS + ['riskScore', 'hadCrisisToday', 'hasRescueInhaler'],
    'S2 sem derivadas do rótulo (12)': ORIGINAIS + ['hasRescueInhaler'],
    'S3 só colunas originais (11)': ORIGINAIS,
}

MODELOS = {
    'Regressão Logística': (LogisticRegression(max_iter=2000, random_state=42),
                            {'C': [0.01, 0.1, 1.0, 10.0]}),
    'Random Forest': (RandomForestClassifier(random_state=42),
                      {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42),
                          {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7],
                           'learning_rate': [0.01, 0.05, 0.1, 0.2]}),
    'XGBoost': (XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
                {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7],
                 'learning_rate': [0.01, 0.05, 0.1, 0.2]}),
}

y = df['high_risk']
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)  # 30 dobras pareadas

linhas, folds = [], {}
for cen, feats in CENARIOS.items():
    X = df[feats]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    sc = MinMaxScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
    for nome, (modelo, grade) in MODELOS.items():
        gs = GridSearchCV(modelo, grade, scoring='roc_auc', cv=cv5, n_jobs=-1).fit(X_tr_s, y_tr)
        best = gs.best_estimator_
        prob = best.predict_proba(X_te_s)[:, 1]
        pred = (prob >= 0.5).astype(int)
        thrs = np.linspace(0.01, 0.99, 300)
        f1s = [f1_score(y_te, (prob >= t).astype(int), zero_division=0) for t in thrs]
        t_opt = thrs[int(np.argmax(f1s))]
        linhas.append({
            'Cenário': cen, 'Modelo': nome,
            'CV AUC': gs.best_score_, 'AUC teste': roc_auc_score(y_te, prob),
            'Recall 0.5': recall_score(y_te, pred), 'F1 0.5': f1_score(y_te, pred),
            'Limiar ótimo': t_opt,
            'Recall ótimo': recall_score(y_te, (prob >= t_opt).astype(int)),
        })
        # 30 dobras com os hiperparâmetros escolhidos, mesmo particionamento para todos
        pipe = make_pipeline(MinMaxScaler(), best)
        folds[(cen, nome)] = cross_val_score(pipe, X, y, cv=rcv, scoring='roc_auc', n_jobs=-1)
        print(f'{cen:34s} {nome:20s} AUC teste={linhas[-1]["AUC teste"]:.3f}', flush=True)

res = pd.DataFrame(linhas)
pd.set_option('display.width', 200)
print('\n=== TABELA DE ABLAÇÃO (conjunto de teste, 20%) ===')
print(res.round(3).to_string(index=False))

print('\n=== AUC em 30 dobras (10-fold x 3): média ± desvio ===')
for (cen, nome), v in folds.items():
    print(f'{cen:34s} {nome:20s} {v.mean():.3f} ± {v.std():.3f}')

print('\n=== Wilcoxon pareado (30 dobras): XGBoost vs demais, por cenário ===')
for cen in CENARIOS:
    a = folds[(cen, 'XGBoost')]
    for nome in ['Regressão Logística', 'Random Forest', 'Gradient Boosting']:
        b = folds[(cen, nome)]
        stat, p = wilcoxon(a, b)
        print(f'{cen:34s} XGB vs {nome:20s} Δ={a.mean() - b.mean():+.3f}  p={p:.4f}')

print('\n=== Wilcoxon pareado: XGBoost S0 vs cada cenário ===')
a = folds[('S0 Original (15)', 'XGBoost')]
for cen in list(CENARIOS)[1:]:
    b = folds[(cen, 'XGBoost')]
    stat, p = wilcoxon(a, b)
    print(f'S0 vs {cen:34s} Δ={a.mean() - b.mean():+.3f}  p={p:.2e}')

res.to_csv(sys.argv[2], index=False)
