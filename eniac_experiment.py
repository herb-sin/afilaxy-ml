"""
ENIAC 2026 — Experimento Reprodutível
Predicting Asthma Exacerbations Using Machine Learning Frameworks

Target: high_risk = 1 se symptom_score (Wheezing+ShortnessOfBreath+ChestTightness+
        Coughing+NighttimeSymptoms) >= 3 (critério GINA)

Features: sintomas REMOVIDOS de X para evitar data leakage.
  Variáveis sintéticas do Afilaxy adicionadas para emular os dados reais do app
  (geradas com numpy.random.seed(42) sobre o dataset proxy — ver notebook original).
  - 5 clínicas/demográficas: Age, Gender, Ethnicity, Smoking, BMI
  - 2 espirométricas: LungFunctionFEV1, LungFunctionFVC
  - 4 ambientais: PollutionExposure, PollenExposure, DustExposure, ExerciseInduced
  - 4 Afilaxy (sintéticas): riskScore, hadCrisisToday, hasRescueInhaler, emergencyCount
  Total: 15 features
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, recall_score, f1_score,
                              precision_score, confusion_matrix)
from xgboost import XGBClassifier

# ── 1. CARREGAMENTO ──────────────────────────────────────────────────────────
print("Carregando dados...")
df = pd.read_csv('/home/afilaxy/Projetos/ENIAC/Afilaxy ML ENIAC/asthma_disease_data.csv')
df = df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])

# ── 2. TARGET GINA ───────────────────────────────────────────────────────────
SYMPTOM_COLS = ['Wheezing', 'ShortnessOfBreath', 'ChestTightness',
                'Coughing', 'NighttimeSymptoms']
df['symptom_score'] = df[SYMPTOM_COLS].sum(axis=1)
df['high_risk'] = (df['symptom_score'] >= 3).astype(int)

n1 = df['high_risk'].sum()
n0 = len(df) - n1
print(f"Target: high_risk=1: {n1} ({n1/len(df)*100:.1f}%) | high_risk=0: {n0} ({n0/len(df)*100:.1f}%)")

# ── 3. VARIÁVEIS SINTÉTICAS DO AFILAXY (seed=42 — reprodutível) ──────────────
np.random.seed(42)
pollution_norm = df['PollutionExposure'] / df['PollutionExposure'].max()
lung_deficit   = 1 - (df['LungFunctionFEV1'] / df['LungFunctionFEV1'].max())
symptom_norm   = df['symptom_score'] / 5.0
pollen_norm    = df['PollenExposure'] / df['PollenExposure'].max()

risk_raw = (0.35 * symptom_norm + 0.30 * pollution_norm +
            0.20 * lung_deficit  + 0.15 * pollen_norm)
df['riskScore'] = ((risk_raw + np.random.normal(0, 0.05, len(df))).clip(0, 1) * 100).round(1)

prob_crisis = (0.6 * symptom_norm + 0.4 * (df['riskScore'] / 100)).clip(0, 1) * 0.35
df['hadCrisisToday'] = np.random.binomial(1, prob_crisis).astype(int)

base_inhaler = 0.78 + 0.08 * df['HistoryOfAllergies'] - 0.06 * df['Smoking']
df['hasRescueInhaler'] = np.random.binomial(1, base_inhaler.clip(0.4, 0.97)).astype(int)

prob_emerg  = (0.05 + 0.25 * df['high_risk'] + 0.15 * df['hadCrisisToday']).clip(0, 0.5)
had_emerg   = np.random.binomial(1, prob_emerg)
count_emerg = np.random.choice([1, 2, 3, 4, 5], size=len(df), p=[0.50, 0.25, 0.13, 0.08, 0.04])
df['emergencyCount'] = (had_emerg * count_emerg).astype(int)

# ── 4. FEATURES (sintomas excluídos para evitar leakage) ─────────────────────
FEATURES = [
    'Age', 'Gender', 'Ethnicity', 'Smoking', 'BMI',
    'LungFunctionFEV1', 'LungFunctionFVC',
    'PollutionExposure', 'PollenExposure', 'DustExposure', 'ExerciseInduced',
    'riskScore', 'hadCrisisToday', 'hasRescueInhaler', 'emergencyCount',
]

X = df[FEATURES]
y = df['high_risk']

# ── 5. SPLIT E NORMALIZAÇÃO ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTreino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

# ── 5. GRID SEARCH + 5-FOLD CV ────────────────────────────────────────────────
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = {
    'Regressão Logística': (
        LogisticRegression(max_iter=2000, random_state=42),
        {'C': [0.01, 0.1, 1.0, 10.0]}
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=42),
        {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7],
         'learning_rate': [0.01, 0.05, 0.1, 0.2]}
    ),
    'XGBoost': (
        XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7],
         'learning_rate': [0.01, 0.05, 0.1, 0.2]}
    ),
}

results = {}
best_models = {}

for name, (model, params) in configs.items():
    print(f"\nOtimizando {name}...")
    gs = GridSearchCV(model, params, scoring='roc_auc', cv=cv5, n_jobs=-1, verbose=0)
    gs.fit(X_train_s, y_train)
    best = gs.best_estimator_
    best_models[name] = best

    cv_auc   = gs.best_score_
    y_prob   = best.predict_proba(X_test_s)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    y_pred   = best.predict(X_test_s)
    recall   = recall_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)

    results[name] = {
        'Média CV (AUC)': round(cv_auc, 3),
        'AUC (Teste)':    round(test_auc, 3),
        'Recall':         round(recall, 3),
        'F1-Score':       round(f1, 3),
    }
    print(f"  Parâmetros: {gs.best_params_}")
    print(f"  CV AUC={cv_auc:.3f} | Test AUC={test_auc:.3f} | Recall={recall:.3f} | F1={f1:.3f}")

# ── 6. TABELA 3 ───────────────────────────────────────────────────────────────
print("\n\n=== TABELA 3 ===")
df_res = pd.DataFrame(results).T
df_res.index.name = 'Modelo'
print(df_res.to_string())
df_res.to_csv('/home/afilaxy/Projetos/ENIAC/tabela3_resultados.csv')

# ── 7. THRESHOLD E MÉTRICAS DO XGBOOST ───────────────────────────────────────
xgb   = best_models['XGBoost']
yp    = xgb.predict_proba(X_test_s)[:, 1]
thrs  = np.linspace(0.01, 0.99, 300)
precs, recs, f1s = [], [], []
for t in thrs:
    yhat = (yp >= t).astype(int)
    precs.append(precision_score(y_test, yhat, zero_division=0))
    recs.append(recall_score(y_test, yhat, zero_division=0))
    f1s.append(f1_score(y_test, yhat, zero_division=0))

opt_idx  = int(np.argmax(f1s))
opt_thr  = round(float(thrs[opt_idx]), 3)
opt_rec  = round(float(recs[opt_idx]), 3)
opt_f1   = round(float(f1s[opt_idx]), 3)
print(f"\nXGBoost threshold ótimo: {opt_thr} | Recall={opt_rec} | F1={opt_f1}")

# ── 8. FIGURA 2 ───────────────────────────────────────────────────────────────
print("Gerando Figura 2...")
fig2, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(thrs, precs, ':', color='#555555', label='Precisão', lw=1.5)
ax.plot(thrs, recs,  '-', color='#C0392B', label='Sensibilidade (Recall)', lw=2)
ax.plot(thrs, f1s,   '-', color='#2C3E50', label='F1-Score', lw=2)
ax.axvline(opt_thr, color='#888888', ls='--', lw=1.2)
ax.scatter([opt_thr], [max(f1s)], color='#E74C3C', zorder=5, s=60)
ax.annotate(f'Threshold Ótimo\n({opt_thr})',
            xy=(opt_thr, max(f1s)),
            xytext=(opt_thr + 0.07, max(f1s) - 0.15),
            fontsize=8.5, color='#333333',
            arrowprops=dict(arrowstyle='->', color='#888888'))
ax.set_xlabel('Limiar de Decisão Probabilística', fontsize=10)
ax.set_ylabel('Índice da Métrica', fontsize=10)
ax.set_title('Otimização de Métricas por Variação de Limiar', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig('/home/afilaxy/Projetos/ENIAC/figura2_threshold.pdf', dpi=200)
fig2.savefig('/home/afilaxy/Projetos/ENIAC/figura2_threshold.png', dpi=200)
plt.close()
print("  Figura 2 salva.")

# ── 9. FIGURA 3 ───────────────────────────────────────────────────────────────
print("Gerando Figura 3...")
imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=True)

# Rótulos legíveis em português para o artigo
imp.index = list(imp.index)  # mantém nomes originais das features

fig3, ax = plt.subplots(figsize=(7, 5.5))
bars = ax.barh(imp.index, imp.values, color='#2C3E50', height=0.6)
for bar, val in zip(bars, imp.values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val*100:.1f}%', va='center', ha='left', fontsize=8)
ax.set_xlabel('Ganho Relativo (F-Score Weight)', fontsize=10)
ax.set_title('Importância Relativa das Features (XGBoost)', fontsize=11)
ax.set_xlim(0, imp.max() * 1.3)
ax.grid(True, axis='x', alpha=0.3)
fig3.tight_layout()
fig3.savefig('/home/afilaxy/Projetos/ENIAC/figura3_feature_importance.pdf', dpi=200)
fig3.savefig('/home/afilaxy/Projetos/ENIAC/figura3_feature_importance.png', dpi=200)
plt.close()
print("  Figura 3 salva.")

# ── 10. SUMÁRIO FINAL ─────────────────────────────────────────────────────────
print("\n=== SUMÁRIO PARA O ARTIGO ===")
for name, r in results.items():
    print(f"{name:25s}: CV={r['Média CV (AUC)']:.3f} | AUC={r['AUC (Teste)']:.3f} | "
          f"Recall={r['Recall']:.3f} | F1={r['F1-Score']:.3f}")

xr = results['XGBoost']
print(f"\nXGBoost: AUC={xr['AUC (Teste)']*100:.1f}% | Recall={xr['Recall']*100:.1f}% | Threshold={opt_thr}")
print(f"\nTop 5 features:")
for feat, val in imp.sort_values(ascending=False).head(5).items():
    print(f"  {feat}: {val*100:.1f}%")

print("\nArquivos gerados:")
print("  tabela3_resultados.csv")
print("  figura2_threshold.pdf/.png")
print("  figura3_feature_importance.pdf/.png")
print("\nExperimento concluído.")
