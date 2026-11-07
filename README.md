# afilaxy-ml
Reproducible ML experiment for asthma exacerbation risk screening (ENIAC 2026). XGBoost, Random Forest, Gradient Boosting, Logistic Regression on the Asthma Disease Dataset (Kaggle).

## ⚠️ Adendo: estudo de ablação (nov. 2026)

> **EN:** A post-publication ablation study reproduced Table 3 exactly, but once the three synthetic features derived from the label (`riskScore`, `hadCrisisToday`, `emergencyCount`) are removed, every model drops to AUC ≈ 0.50 (chance level). The reported performance stems from label leakage by construction, not predictive ability; the results should be read as a demonstration of the pipeline only. See [ERRATA.md](ERRATA.md#adendo-novembro-de-2026-estudo-de-ablação) and [`ablacao.py`](ablacao.py).

Os números da Tabela 3 se reproduzem, mas **não demonstram capacidade preditiva**: sem as três variáveis sintéticas derivadas do rótulo, a AUC de todos os modelos cai para cerca de 0,50, equivalente ao acaso. Detalhes, tabela e testes estatísticos no [adendo da errata](ERRATA.md#adendo-novembro-de-2026-estudo-de-ablação).

## ⚠️ Errata (5 nov. 2026)

> **EN:** The published ENIAC 2026 paper cites two references that do not exist (Alotaibi et al., 2020; Patel et al., 2021) and misdescribes or misreports others. These reference corrections do not change the data, code or published numbers (but see the ablation addendum above).

O artigo publicado nos anais do ENIAC 2026 contém erros nas referências:

- **Duas referências não existem:** Alotaibi et al. (2020) e Patel et al. (2021). As afirmações que se apoiavam nelas (Seções 1, 2.1, 3 e 4.2.1 e Tabela 2) foram retiradas ou reescritas.
- **Duas referências estavam descritas de forma errada:** Merchant et al. (2016) e Bousquet et al. (2017).
- **Seis referências tinham dados bibliográficos errados**, como ano, revista, volume ou páginas.

**As correções de referências não alteram os dados, o código nem os números publicados (Tabela 3 e Figuras 2 e 3).** A interpretação desses números foi revista no adendo acima.

- Lista completa de correções: [ERRATA.md](ERRATA.md)
- Versão corrigida do artigo: [ENIAC2026_artigo_corrigido.pdf](ENIAC2026_artigo_corrigido.pdf)
- Versão publicada, que contém os erros: [ENIAC__Afilaxy_Predicting_Asthma_Exacerbations_in_Digital_Health_Ecosystems_Using_Machine_Learning_Frameworks.pdf](ENIAC__Afilaxy_Predicting_Asthma_Exacerbations_in_Digital_Health_Ecosystems_Using_Machine_Learning_Frameworks.pdf)

A responsabilidade pelos erros é dos autores. Pedimos desculpas a quem leu ou citou o trabalho.
