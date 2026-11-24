import unittest
import numpy as np
from src.simulator import MonteCarloSimulator


class TestSimulator(unittest.TestCase):
    def setUp(self):
        # 4 times simples
        self.standings = [
            {"nome": "Time 1", "posicao": 1, "pontos": 70, "vitorias": 21, "saldo_gols": 30, "gols_pro": 60},
            {"nome": "Time 2", "posicao": 2, "pontos": 65, "vitorias": 19, "saldo_gols": 20, "gols_pro": 50},
            {"nome": "Time 3", "posicao": 3, "pontos": 40, "vitorias": 10, "saldo_gols": 0, "gols_pro": 35},
            {"nome": "Time 4", "posicao": 4, "pontos": 25, "vitorias": 6, "saldo_gols": -20, "gols_pro": 20}
        ]
        self.matches = [
            {"mandante": "Time 1", "visitante": "Time 2", "p_mandante": 0.45, "p_empate": 0.25, "p_visitante": 0.30},
            {"mandante": "Time 3", "visitante": "Time 4", "p_mandante": 0.60, "p_empate": 0.20, "p_visitante": 0.20}
        ]

    def test_probability_conservation(self):
        sim = MonteCarloSimulator(self.standings, self.matches, num_simulations=10_000, seed=123)
        res = sim.run()
        prob_matrix = res["prob_matrix"]

        # 1. A soma das probabilidades de campeão (posição 0) deve ser 100%
        self.assertAlmostEqual(float(np.sum(prob_matrix[:, 0])), 100.0, places=3)

        # 2. Para cada posição (coluna), a soma entre todos os times deve ser 100%
        for p in range(len(self.standings)):
            self.assertAlmostEqual(float(np.sum(prob_matrix[:, p])), 100.0, places=3)

        # 3. Para cada time (linha), a soma de suas chances em todas as posições deve ser 100%
        for t in range(len(self.standings)):
            self.assertAlmostEqual(float(np.sum(prob_matrix[t, :])), 100.0, places=3)

    def test_reproducibility_with_seed(self):
        sim1 = MonteCarloSimulator(self.standings, self.matches, num_simulations=5_000, seed=999)
        res1 = sim1.run()

        sim2 = MonteCarloSimulator(self.standings, self.matches, num_simulations=5_000, seed=999)
        res2 = sim2.run()

        np.testing.assert_array_almost_equal(res1["prob_matrix"], res2["prob_matrix"])
        np.testing.assert_array_almost_equal(res1["mean_points"], res2["mean_points"])

    def test_cbf_tie_breaker_wins(self):
        # Dois times que terminam empatados em pontos, mas Time A tem mais vitórias
        tied_standings = [
            {"nome": "Time A", "posicao": 1, "pontos": 50, "vitorias": 15, "saldo_gols": 10, "gols_pro": 40},
            {"nome": "Time B", "posicao": 2, "pontos": 50, "vitorias": 13, "saldo_gols": 10, "gols_pro": 40}
        ]
        # Sem jogos restantes
        sim = MonteCarloSimulator(tied_standings, [], num_simulations=100, seed=42)
        res = sim.run()
        # Time A deve ter 100% no 1º lugar porque tem mais vitórias (15 x 13)
        self.assertAlmostEqual(res["prob_matrix"][0, 0], 100.0, places=3)
        self.assertAlmostEqual(res["prob_matrix"][1, 1], 100.0, places=3)

    def test_score_matrix_simulation_updates_goals_and_tiebreakers(self):
        # Dois times empatados antes do confronto direto; Time B vence por 3x1 (probabilidade 1.0)
        standings = [
            {"nome": "Time A", "posicao": 1, "pontos": 50, "vitorias": 15, "saldo_gols": 10, "gols_pro": 40},
            {"nome": "Time B", "posicao": 2, "pontos": 50, "vitorias": 15, "saldo_gols": 10, "gols_pro": 40},
        ]
        mat = np.zeros((9, 9), dtype=np.float64)
        mat[1, 3] = 1.0  # Mandante (Time A) 1 x 3 Visitante (Time B)

        matches = [
            {
                "mandante": "Time A",
                "visitante": "Time B",
                "p_mandante": 0.0,
                "p_empate": 0.0,
                "p_visitante": 1.0,
                "score_matrix": mat,
            }
        ]
        sim = MonteCarloSimulator(standings, matches, num_simulations=50, seed=42)
        res = sim.run()
        self.assertAlmostEqual(res["mean_points"][0], 50.0, places=3)
        self.assertAlmostEqual(res["mean_points"][1], 53.0, places=3)
        self.assertAlmostEqual(res["prob_matrix"][1, 0], 100.0, places=3)
        self.assertAlmostEqual(res["prob_matrix"][0, 1], 100.0, places=3)

    def test_score_matrix_tiebreaker_by_exact_goal_difference(self):
        # Time A tem +2 SG via 2x0; Time B tem +1 SG via 1x0 contra adversários já zerados
        standings = [
            {"nome": "Time A", "posicao": 1, "pontos": 50, "vitorias": 15, "saldo_gols": 0, "gols_pro": 20},
            {"nome": "Time B", "posicao": 2, "pontos": 50, "vitorias": 15, "saldo_gols": 0, "gols_pro": 20},
            {"nome": "Time C", "posicao": 3, "pontos": 10, "vitorias": 2, "saldo_gols": -20, "gols_pro": 10},
            {"nome": "Time D", "posicao": 4, "pontos": 10, "vitorias": 2, "saldo_gols": -20, "gols_pro": 10},
        ]
        mat_2x0 = np.zeros((9, 9), dtype=np.float64)
        mat_2x0[2, 0] = 1.0
        mat_1x0 = np.zeros((9, 9), dtype=np.float64)
        mat_1x0[1, 0] = 1.0

        matches = [
            {"mandante": "Time A", "visitante": "Time C", "p_mandante": 1.0, "p_empate": 0.0, "p_visitante": 0.0, "score_matrix": mat_2x0},
            {"mandante": "Time B", "visitante": "Time D", "p_mandante": 1.0, "p_empate": 0.0, "p_visitante": 0.0, "score_matrix": mat_1x0},
        ]
        sim1 = MonteCarloSimulator(standings, matches, num_simulations=20, seed=7)
        sim2 = MonteCarloSimulator(standings, matches, num_simulations=20, seed=7)
        res1 = sim1.run()
        res2 = sim2.run()

        self.assertAlmostEqual(res1["prob_matrix"][0, 0], 100.0, places=3)
        self.assertAlmostEqual(res1["prob_matrix"][1, 1], 100.0, places=3)
        np.testing.assert_array_almost_equal(res1["prob_matrix"], res2["prob_matrix"])


if __name__ == "__main__":
    unittest.main()
