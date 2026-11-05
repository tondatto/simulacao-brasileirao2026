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
