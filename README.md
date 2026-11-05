# Simulador Monte Carlo do Campeonato Brasileiro 2026

Simulador probabilístico das rodadas finais do Campeonato Brasileiro Série A 2026 baseado no método de Monte Carlo, inspirado no modelo estatístico do **Prof. Rogemar Riffel (IF-UFRGS)** e consumindo dados reais do **Globo Esporte (GE)** conforme a arquitetura do projeto [`campeonato-brasileiro-api`](https://github.com/ezefranca/campeonato-brasileiro-api).

---

## 🔬 Metodologia Científica (Modelo Prof. Rogemar Riffel - UFRGS)

O simulador replica os princípios descritos no projeto da UFRGS:

1. **Peso Composto (50/50)**:
   - **50% Desempenho Histórico**: Aproveitamento geral de pontos de cada clube na temporada até a 34ª rodada.
   - **50% Desempenho Recente**: Aproveitamento nas últimas 5 partidas disputadas (`ultimos_jogos`), capturando a fase e o momento anímico de cada equipe.
2. **Vantagem de Mando e Probabilidade de Empate**:
   - Incorpora o fator casa histórico do futebol brasileiro.
   - A probabilidade base de empate (~26%) decai exponencialmente conforme o desequilíbrio entre as forças relativas das equipes.
3. **Fator de "Zebra" de 5%**:
   - Modela o elemento estocástico imprevisível do futebol através de 5% de inversão estocástica de probabilidades nos confrontos.
4. **Monte Carlo em Larga Escala (1.000.000 de Simulações)**:
   - Implementação vetorizada com **NumPy** permitindo simular 1 milhão de edições das 4 rodadas finais em cerca de 5 segundos.
5. **Faixa de Incerteza ($\mu \pm 1\sigma$)**:
   - Cada projeção de pontos exibe a média acompanhada de $\pm 1$ desvio-padrão da distribuição simulada.
6. **Critérios Oficiais de Desempate CBF**:
   - 1º Pontos Ganhos $\to$ 2º Vitórias $\to$ 3º Saldo de Gols $\to$ 4º Gols Pró $\to$ 5º Sorteio.

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.9+ com `numpy` instalado no ambiente.
- Não requer instalação global de nenhum pacote npm nem bibliotecas adicionais.

### Execução Padrão (1.000.000 de Simulações)
```bash
python3 main.py
```

### Visualizar Matriz Completa de Posições (1º ao 20º)
```bash
python3 main.py --detalhado
```

### Ajustar Quantidade de Simulações ou Fator de Zebra
```bash
# 500.000 simulações com fator zebra de 8%
python3 main.py --simulacoes 500000 --zebra 0.08
```

### Forçar Atualização dos Dados (Ignorar Cache)
```bash
python3 main.py --refresh
```

### Incluir Jogos Pendentes da Rodada 34
```bash
python3 main.py --incluir-pendentes-r34
```

---

## 🧪 Testes Automatizados

Para executar os testes unitários de regras, probabilidades e desempate:
```bash
python3 -m unittest discover tests
```

---

## 📁 Estrutura do Projeto

```
brasileirao2026/
├── .cache/                # Cache local dos dados do GE (ignorado pelo git)
├── .gitignore
├── src/
│   ├── ge_fetcher.py      # Extração de dados da Série A 2026 e API de rodadas
│   ├── model_riffel.py    # Modelo 50/50 + Vantagem de Mando + Fator Zebra 5%
│   ├── simulator.py       # Motor Monte Carlo vetorizado em NumPy (1M iterações)
│   ├── stats.py           # Agregação de μ ± 1σ, Título, G4, G6, Sula e Z4
│   └── terminal_view.py   # Renderização formatada e limpa no terminal
├── tests/
│   ├── test_model.py      # Testes do modelo estatístico
│   └── test_simulator.py  # Testes de convergência e desempate CBF
├── main.py                # Ponto de entrada CLI
└── README.md
```


## Exemplo de resultado detalhado

Resultado do comando `python3 main.py --detalhado` executado em 05/11/2026.

```
================================================================================================================
      SIMULAÇÃO MONTE CARLO DO CAMPEONATO BRASILEIRO SÉRIE A — 2026
   Metodologia Estatística: Prof. Rogemar Riffel (IF-UFRGS) | Dados: Globo Esporte
================================================================================================================
• Modelo: Peso Composto 50/50 (Histórico Temporada + Forma Recente nas últimas 5 rodadas)
• Amostragem: 1,000,000 simulações de Monte Carlo | Fator de Zebra: 5%
• Cenário: 40 jogos restantes (Rodadas 35 a 38) | Incerteza indicada: ±1 desvio-padrão (1σ)
================================================================================================================
Pos  Time             Pts   V   SG | Projeção (±1σ)     Faixa    |  Título     G-4     G-6    Sula     Z-4
----------------------------------------------------------------------------------------------------------------
1º   Flamengo          76  23  +42 |   83.2 ± 2.6     [76 - 88]  |   96.4%  100.0%  100.0%    0.0%    0.0%
2º   Palmeiras         70  20  +34 |   76.6 ± 2.6     [70 - 82]  |    3.6%  100.0%  100.0%    0.0%    0.0%
3º   Cruzeiro          57  17   +6 |   63.5 ± 2.7     [57 - 69]  |    0.0%   91.2%   99.7%    0.3%    0.0%
4º   Fluminense        54  15  +10 |   59.3 ± 2.7     [54 - 66]  |    0.0%   39.9%   86.9%   13.1%    0.0% [Fim G-4]
5º   Athletico-PR      54  15   +8 |   59.2 ± 2.6     [54 - 66]  |    0.0%   33.6%   85.2%   14.8%    0.0%
6º   Bahia             53  14   +6 |   59.0 ± 2.7     [53 - 65]  |    0.0%   27.8%   76.6%   23.4%    0.0% [Fim G-6]
7º   Atlético-MG       50  14   +4 |   55.7 ± 2.7     [50 - 62]  |    0.0%    6.4%   34.9%   65.1%    0.0%
8º   Bragantino        47  13   +5 |   53.1 ± 2.7     [47 - 59]  |    0.0%    0.9%   11.5%   88.0%    0.0%
9º   Santos            47  12   +1 |   52.5 ± 2.6     [47 - 59]  |    0.0%    0.2%    4.3%   93.2%    0.0%
10º  São Paulo         44  12   +0 |   49.9 ± 2.7     [44 - 56]  |    0.0%   <0.1%    0.9%   89.4%   <0.1%
11º  Botafogo          42  11   -7 |   47.5 ± 2.6     [42 - 54]  |    0.0%    0.0%   <0.1%   60.3%    2.1%
12º  Coritiba          42  11  -11 |   46.5 ± 2.6     [42 - 54]  |    0.0%    0.0%   <0.1%   40.3%    4.5% [Fim G-12 / Sula]
13º  Vasco             41  11   -7 |   46.6 ± 2.7     [41 - 53]  |    0.0%    0.0%   <0.1%   51.8%    4.8%
14º  Mirassol          40  10   -9 |   45.5 ± 2.7     [40 - 52]  |    0.0%    0.0%    0.0%   28.0%   13.6%
15º  Vitória           38  10  -13 |   43.1 ± 2.6     [38 - 50]  |    0.0%    0.0%    0.0%   10.2%   36.1%
16º  Internacional     38   9   -4 |   43.3 ± 2.7     [38 - 50]  |    0.0%    0.0%    0.0%    9.8%   38.0% [Início Z-4 / Rebaixamento]
17º  Corinthians       37   9   -6 |   42.2 ± 2.6     [37 - 49]  |    0.0%    0.0%    0.0%    6.1%   50.9%
18º  Grêmio            37   9   -9 |   42.5 ± 2.6     [37 - 49]  |    0.0%    0.0%    0.0%    6.3%   50.2%
19º  Remo              29   6  -15 |   34.7 ± 2.7     [29 - 41]  |    0.0%    0.0%    0.0%    0.0%   99.8%
20º  Chapecoense       21   4  -35 |   24.6 ± 2.5     [21 - 33]  |    0.0%    0.0%    0.0%    0.0%  100.0%
================================================================================================================

📌 DESTAQUES DA SIMULAÇÃO:

🏆 DISPUTA PELO TÍTULO:
   • Flamengo          96.4% de chance (Projeção: 83.2 pts)
   • Palmeiras          3.6% de chance (Projeção: 76.6 pts)

⭐ BRIGA PELA LIBERTADORES (G-4):
   • Cruzeiro          91.2% de chance de G-4 (G-6:  99.7%)
   • Fluminense        39.9% de chance de G-4 (G-6:  86.9%)
   • Athletico-PR      33.6% de chance de G-4 (G-6:  85.2%)
   • Bahia             27.8% de chance de G-4 (G-6:  76.6%)
   • Atlético-MG        6.4% de chance de G-4 (G-6:  34.9%)

⚠️ RISCO DE REBAIXAMENTO (Z-4):
   • Chapecoense      100.0% de risco de Série B (Projeção: 24.6 pts)
   • Remo              99.8% de risco de Série B (Projeção: 34.7 pts)
   • Corinthians       50.9% de risco de Série B (Projeção: 42.2 pts)
   • Grêmio            50.2% de risco de Série B (Projeção: 42.5 pts)
   • Internacional     38.0% de risco de Série B (Projeção: 43.3 pts)
   • Vitória           36.1% de risco de Série B (Projeção: 43.1 pts)
   • Mirassol          13.6% de risco de Série B (Projeção: 45.5 pts)
   • Vasco              4.8% de risco de Série B (Projeção: 46.6 pts)
   • Coritiba           4.5% de risco de Série B (Projeção: 46.5 pts)
   • Botafogo           2.1% de risco de Série B (Projeção: 47.5 pts)


================================================================================================================
      DISTRIBUIÇÃO DETALHADA DE PROBABILIDADES POR POSIÇÃO FINAL (1º AO 20º)
================================================================================================================
Time             |  1º  2º  3º  4º  5º  6º  7º  8º  9º 10º 11º 12º 13º 14º 15º 16º 17º 18º 19º 20º
----------------------------------------------------------------------------------------------------------------
Flamengo         | 96%  4%                                                                        
Palmeiras        |  4% 96%                                                                        
Cruzeiro         |         76% 16%  6%  2%  ·   ·                                                 
Fluminense       |         11% 29% 27% 20%  9%  3%  1%  ·                                         
Athletico-PR     |          7% 27% 27% 24% 11%  3%  1%  ·                                         
Bahia            |          6% 22% 25% 24% 15%  6%  2%  ·                                         
Atlético-MG      |          1%  6% 11% 18% 31% 21% 10%  3%  1%  ·                                 
Bragantino       |          ·   1%  3%  7% 19% 29% 23% 11%  4%  2%  ·   ·                         
Santos           |              ·   1%  3% 11% 23% 29% 18%  8%  4%  2%  1%  ·                     
São Paulo        |                  ·   1%  3%  9% 19% 31% 17% 10%  6%  3%  1%  ·   ·             
Botafogo         |                      ·   ·   2%  7% 13% 19% 18% 15% 11%  7%  4%  2%  ·         
Coritiba         |                          ·   1%  3%  7% 13% 16% 18% 17% 12%  9%  4%  1%        
Vasco            |                          ·   1%  4%  9% 19% 19% 16% 13%  9%  6%  4%  1%        
Mirassol         |                          ·   ·   1%  4% 10% 13% 16% 16% 15% 12%  9%  5%  ·     
Vitória          |                              ·   ·   1%  3%  6%  8% 12% 16% 18% 20% 16%  ·     
Internacional    |                              ·   ·   1%  3%  6%  8% 12% 15% 17% 21% 17%  1%    
Corinthians      |                                  ·   1%  2%  4%  6%  8% 12% 17% 21% 27%  3%    
Grêmio           |                                  ·   1%  2%  4%  6%  9% 13% 16% 20% 28%  2%    
Remo             |                                                          ·   ·   1%  5% 93%  1%
Chapecoense      |                                                                          1% 99%
================================================================================================================
Legenda: ' · ' indica probabilidade menor que 0.5% mas matematicamente possível.
```

```
```
