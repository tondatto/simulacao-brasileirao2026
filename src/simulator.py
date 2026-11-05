"""
Motor de Simulação Monte Carlo vetorizado em NumPy para o Campeonato Brasileiro.
Permite simular 1.000.000 de edições completas das rodadas finais em poucos segundos,
aplicando os critérios oficiais de desempate da CBF.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class MonteCarloSimulator:
    def __init__(
        self,
        standings: List[Dict[str, Any]],
        matches: List[Dict[str, Any]],
        num_simulations: int = 1_000_000,
        seed: int = 42
    ):
        self.num_simulations = num_simulations
        self.seed = seed
        self.standings = standings
        self.matches = matches

        # Mapeamento de times para índices 0..19
        self.team_names = [t["nome"] for t in standings]
        self.num_teams = len(self.team_names)
        self.team_to_idx = {name: i for i, name in enumerate(self.team_names)}

        # Vetores de estado inicial
        self.initial_points = np.array([t["pontos"] for t in standings], dtype=np.int32)
        self.initial_wins = np.array([t["vitorias"] for t in standings], dtype=np.int32)
        self.initial_gd = np.array([t["saldo_gols"] for t in standings], dtype=np.int32)
        self.initial_gp = np.array([t["gols_pro"] for t in standings], dtype=np.int32)

    def run(self) -> Dict[str, Any]:
        """
        Executa a simulação Monte Carlo vetorizada.
        Retorna matriz de posições (20x20), médias e desvios de pontos.
        """
        rng = np.random.default_rng(self.seed)
        N = self.num_simulations
        num_matches = len(self.matches)

        # Matrizes de pontuação acumulada para cada time em cada simulação
        pts = np.tile(self.initial_points, (N, 1)).astype(np.int64)
        wins = np.tile(self.initial_wins, (N, 1)).astype(np.int64)
        gd = np.tile(self.initial_gd, (N, 1)).astype(np.int64)
        gp = np.tile(self.initial_gp, (N, 1)).astype(np.int64)

        # Geração de números aleatórios para todos os jogos em lote
        rand_vals = rng.random((N, num_matches), dtype=np.float32)
        # Sorteio de margem de gols para vitórias (1, 2 ou 3 gols)
        rand_margin = rng.choice([1, 2, 3], size=(N, num_matches), p=[0.55, 0.30, 0.15])

        for k, match in enumerate(self.matches):
            h_idx = self.team_to_idx[match["mandante"]]
            a_idx = self.team_to_idx[match["visitante"]]

            pm = match["p_mandante"]
            pe = match["p_empate"]
            t1 = pm
            t2 = pm + pe

            r = rand_vals[:, k]
            margin = rand_margin[:, k]

            home_win = r < t1
            draw = (r >= t1) & (r < t2)
            away_win = r >= t2

            # Atualização de Pontos
            pts[:, h_idx] += home_win * 3 + draw
            pts[:, a_idx] += away_win * 3 + draw

            # Atualização de Vitórias
            wins[:, h_idx] += home_win
            wins[:, a_idx] += away_win

            # Atualização de Saldo de Gols e Gols Pró
            h_margin = home_win * margin
            a_margin = away_win * margin

            gd[:, h_idx] += h_margin - a_margin
            gd[:, a_idx] += a_margin - h_margin

            gp[:, h_idx] += h_margin + draw  # Estimativa de gols marcados
            gp[:, a_idx] += a_margin + draw

        # Ruído aleatório mínimo para desempate final (sorteio uniforme)
        noise = rng.integers(0, 99, size=(N, self.num_teams), dtype=np.int64)

        # Critérios oficiais de desempate CBF em chave composta:
        # 1. Pontos Ganhos (peso 10^10)
        # 2. Número de Vitórias (peso 10^7)
        # 3. Saldo de Gols (peso 10^4)
        # 4. Gols Pró (peso 10^2)
        # 5. Sorteio uniforme (ruído 0..99)
        score = (
            pts * 10_000_000_000 +
            wins * 10_000_000 +
            (gd + 500) * 10_000 +
            (gp + 500) * 100 +
            noise
        )

        # Ordenação decrescente: ranks[i, 0] é o campeão da simulação i
        ranks = np.argsort(-score, axis=1)

        # Contagem da matriz de frequências (Times x Posições de 0 a 19)
        position_counts = np.zeros((self.num_teams, self.num_teams), dtype=np.int64)
        for p in range(self.num_teams):
            position_counts[:, p] = np.bincount(ranks[:, p], minlength=self.num_teams)

        # Matriz de probabilidades (percentual de 0.0 a 100.0)
        prob_matrix = (position_counts / N) * 100.0

        # Estatísticas de pontos esperados por time
        mean_points = np.mean(pts, axis=0)
        std_points = np.std(pts, axis=0)
        min_points = np.min(pts, axis=0)
        max_points = np.max(pts, axis=0)

        results = {
            "num_simulations": N,
            "team_names": self.team_names,
            "prob_matrix": prob_matrix,
            "mean_points": mean_points,
            "std_points": std_points,
            "min_points": min_points,
            "max_points": max_points
        }

        return results
