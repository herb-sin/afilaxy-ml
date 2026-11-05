# Errata

**Artigo:** Sin, H. J. e Silva, M. D. (2026). *Predicting Asthma Exacerbations in Digital Health Ecosystems Using Machine Learning Frameworks*. Anais do Encontro Nacional de Inteligência Artificial e Computacional (ENIAC 2026). SBC.

**Data desta errata:** 5 de novembro de 2026

**Versão corrigida:** [`ENIAC2026_artigo_corrigido.pdf`](ENIAC2026_artigo_corrigido.pdf)

---

## Resumo

Depois da publicação, conferimos as 13 referências do artigo nos sites das editoras e em doi.org. Encontramos três tipos de problema:

1. **Duas referências não existem:** Alotaibi et al. (2020) e Patel et al. (2021). Não há publicação com esses títulos, autores e dados. As afirmações que se apoiavam nelas ficaram sem fonte.
2. **Duas referências estavam descritas de forma errada:** Merchant et al. (2016) e Bousquet et al. (2017) foram citadas com afirmações que não correspondem ao conteúdo desses trabalhos.
3. **Seis referências tinham dados bibliográficos errados**, como ano, revista, volume ou páginas.

**Os dados, o código, os experimentos e os resultados não mudam.** A Tabela 3, as Figuras 2 e 3 e todas as métricas (AUC de 91,3%, Sensibilidade de 89,2% no limiar de 0,466 e os demais valores) continuam valendo como publicados. As correções afetam apenas a fundamentação teórica (Seções 1, 2 e 3), a Tabela 2 e a lista de referências.

O artigo publicado nos anais não pode ser alterado. Esta errata e a versão corrigida neste repositório substituem os trechos indicados abaixo. A responsabilidade pelos erros é dos autores, e pedimos desculpas a quem leu ou citou o trabalho.

---

## 1. Referências inexistentes

### Alotaibi et al. (2020)

> Alotaibi, S. et al. (2020). Machine learning applications in asthma monitoring and management: A systematic review. *IEEE Access*, 8:109508–109521.

O IEEE Xplore não tem nenhum artigo com esse título. A referência foi **removida**.

| Onde aparecia | Afirmação | Correção |
|---|---|---|
| Seção 3, 2º parágrafo | Uma revisão sistemática "evidenciou que modelos baseados em comitês de árvores, como Random Forest e Gradient Boosting, superam abordagens lineares tradicionais" | Afirmação **retirada**. Em seu lugar, cita-se a revisão de Tsang et al. (2022), que aponta o predomínio de técnicas tradicionais de ML em estudos de mHealth para asma e alerta que a maioria dos algoritmos foi desenvolvida em bases pequenas e populações restritas. |
| Tabela 2 | Linha "Alotaibi (2020)" | Substituída por "Tsang (2022)". |

Observação: a afirmação retirada também não era sustentada pelos nossos próprios resultados. Na Tabela 3, a Regressão Logística teve a maior AUC no conjunto de teste (0,933).

### Patel et al. (2021)

> Patel, S. et al. (2021). Bridging the gap: patient-generated health data and digital health ecosystems. *npj Digital Medicine*, 4(1):112.

Nenhum dos artigos com autor Patel publicados na *npj Digital Medicine* em 2021 trata desse tema. A referência foi **removida**.

| Onde aparecia | Afirmação | Correção |
|---|---|---|
| Seção 1 (Introdução) | O "vazio assistencial" entre consultas | Citação retirada. O trecho passa a ser argumento dos próprios autores. |
| Seção 2.1 | PGHD capta o cotidiano do paciente de forma longitudinal | Citação retirada. O trecho permanece apoiado apenas em Demiris et al. (2019). |
| Seção 3, 2º parágrafo | A maior barreira para a IA em saúde móvel é a necessidade de alta sensibilidade com baixa complexidade estrutural, viabilizando execução local | A afirmação deixa de ser atribuída a terceiros e passa a ser **posição dos autores**. Para o tema de integração de PGHD, cita-se Genes et al. (2018), relato de caso sobre a integração de dados de um aplicativo de asma ao prontuário eletrônico. |
| Seção 4.2.1 | Base teórica da coleta de PGHD ("Patel et al. (2021) e Demiris et al. (2019)") | Passa a citar apenas Demiris et al. (2019). |
| Tabela 2 | Linha "Patel (2021)" | Substituída por "Genes (2018)". |

---

## 2. Referências descritas de forma errada

### Merchant et al. (2016)

O trabalho correto é um ensaio clínico randomizado com a plataforma Propeller Health, publicado no *Journal of Allergy and Clinical Immunology: In Practice*. Ele não trata de índice de qualidade do ar (AQI).

**Seção 3, texto publicado:**
> […] integraram sensores acoplados a inaladores de resgate para capturar passivamente o uso da medicação junto a dados de geolocalização, correlacionando com sucesso as flutuações do índice de qualidade do ar (AQI) como preditores eficientes de exacerbações urbanas.

**Texto corrigido:**
> […] acoplaram sensores a inaladores de resgate para monitorar em tempo real o uso de medicação de alívio (SABA), observando redução desse uso em um ensaio clínico randomizado.

**Seção 2.1:** a frase "Essa fusão de dados provê o arcabouço informacional necessário para treinar algoritmos preditivos supervisionados" citava Merchant et al. (2016). Passa a citar Tsang et al. (2022).

**Tabela 2:** "Correlação Espacial / Sensores + IoT / Resgate e AQI / Mapeamento de gatilhos" passa a ser "Ensaio Clínico Randomizado / Sensores em Inaladores / Uso de SABA / Monitoramento de medicação".

O papel do AQI e da poluição como gatilhos da asma continua sustentado por Guarnieri e Balmes (2014).

### Bousquet et al. (2017)

O trabalho correto é o estudo piloto MASK-rhinitis, publicado na *Allergy*, e não "MASK-asthma" no *Journal of Allergy and Clinical Immunology*.

**Seção 3, texto publicado:**
> Bousquet et al. 2017 estruturou a iniciativa europeia MASK-asthma, demonstrando que o uso de aplicativos móveis para o autorrelato diário de sintomas eleva a adesão do paciente e fornece um panorama longitudinal confiável da doença […]

**Texto corrigido:**
> Bousquet et al. (2017) demonstraram, no estudo piloto MASK, a viabilidade do autorrelato diário de sintomas via aplicativo móvel, embora limitado a heurísticas estáticas.

**Tabela 2:** o foco principal "Adesão ao tratamento" passa a ser "Viabilidade do autorrelato".

---

## 3. Dados bibliográficos corrigidos

| Referência | Como foi publicado | Correto |
|---|---|---|
| Altman e Bland | 2000, *BMJ* 321(7258):430 | **1994**, *BMJ* **309(6947):102**. A citação no texto (Seções 2.3 e 4.2.4) também muda de 2000 para 1994. |
| Bousquet et al. | 2017, *J Allergy Clin Immunol* 140(5):1245–1249, "MASK-asthma" | 2017, ***Allergy* 72(6):857–865**, estudo MASK-rhinitis |
| Carvalho-Pinto et al. | **R. P.** Carvalho-Pinto, **2022**, *J Bras Pneumol* **48(1)**:e20210273 | **R. M.** Carvalho-Pinto, **2021**, *J Bras Pneumol* **47(6)**:e20210273 |
| Demiris et al. | 2019, *Methods Inf Med* 58(S 01):e1–e2 | 2019, ***Nursing Outlook* 67(4):311–330** |
| Guarnieri e Balmes | 2014, *The Lancet* 383(**9918**):**724–734** | 2014, *The Lancet* 383(**9928**):**1581–1592** |
| Merchant et al. | 2016, *JMIR mHealth uHealth* 4(2):e40 | 2016, ***J Allergy Clin Immunol Pract* 4(3):455–463** |

---

## 4. Referências novas

Estas três referências entram no lugar de Alotaibi et al. (2020) e Patel et al. (2021). As três foram conferidas pelo DOI.

- **Tsang et al. (2022)**: revisão clínica sobre ML aplicado a dados de mHealth no manejo da asma. Substitui Alotaibi et al. (2020) na Seção 3 e Merchant et al. (2016) na Seção 2.1.
- **Genes et al. (2018)**: relato de caso sobre a integração de PGHD de um aplicativo de asma ao prontuário eletrônico. Substitui Patel et al. (2021) na Seção 3.
- **Demiris et al. (2019)**: já estava no artigo e passa a ser a única fonte para PGHD nas Seções 2.1 e 4.2.1.

---

## 5. Lista de referências corrigida

Afilaxy (2025). Afilaxy: A rede social de quem tem asma. https://afilaxy.com/. Acesso em: 23 ago. 2026.

Altman, D. G. and Bland, J. M. (1994). Diagnostic tests 2: Predictive values. *BMJ*, 309(6947):102. doi:10.1136/bmj.309.6947.102

Bousquet, J. et al. (2017). Pilot study of mobile phone technology in allergic rhinitis in European countries: the MASK-rhinitis study. *Allergy*, 72(6):857–865. doi:10.1111/all.13125

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1):5–32. doi:10.1023/A:1010933404324

Carvalho-Pinto, R. M. et al. (2021). Recomendações para o manejo da asma grave da Sociedade Brasileira de Pneumologia e Tisiologia – 2021. *Jornal Brasileiro de Pneumologia*, 47(6):e20210273. doi:10.36416/1806-3756/e20210273

Demiris, G., Iribarren, S. J., Sward, K., Lee, S., and Yang, R. (2019). Patient generated health data use in clinical practice: A systematic review. *Nursing Outlook*, 67(4):311–330. doi:10.1016/j.outlook.2019.04.005

Genes, N., Violante, S., Cetrangol, C., Rogers, L., Schadt, E. E., and Chan, Y.-F. Y. (2018). From smartphone to EHR: a case report on integrating patient-generated health data. *npj Digital Medicine*, 1:23. doi:10.1038/s41746-018-0030-8

Global Initiative for Asthma (2024). *Global Strategy for Asthma Management and Prevention*. GINA. Disponível em: https://www.ginasthma.org. Acesso em: 17 jun. 2026.

Guarnieri, M. and Balmes, J. R. (2014). Outdoor air pollution and asthma. *The Lancet*, 383(9928):1581–1592. doi:10.1016/S0140-6736(14)60617-6

Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer, New York, 2nd edition.

James, G., Witten, D., Hastie, T., and Tibshirani, R. (2021). *An Introduction to Statistical Learning: with Applications in R*. Springer, New York, 2nd edition.

Merchant, R. K., Inamdar, R., and Quade, R. C. (2016). Effectiveness of population health management using the Propeller Health asthma platform: a randomized clinical trial. *The Journal of Allergy and Clinical Immunology: In Practice*, 4(3):455–463. doi:10.1016/j.jaip.2015.11.022

Tsang, K. C. H., Pinnock, H., Wilson, A. M., and Shah, S. A. (2022). Application of machine learning algorithms for asthma management with mHealth: A clinical review. *Journal of Asthma and Allergy*, 15:855–873. doi:10.2147/JAA.S285742
