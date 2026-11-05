import unittest
from src.model_riffel import (
    calculate_team_strengths,
    calculate_match_probabilities,
    prepare_matches_with_probabilities
)


class TestModelRiffel(unittest.TestCase):
    def setUp(self):
        self.sample_standings = [
            {
                "nome": "Time A",
                "posicao": 1,
                "pontos": 60,
                "jogos": 30,
                "vitorias": 18,
                "saldo_gols": 25,
                "gols_pro": 50,
                "ultimos_jogos": ["v", "v", "v", "v", "v"]  # 15 pts em 5 jogos -> 100%
            },
            {
                "nome": "Time B",
                "posicao": 2,
                "pontos": 30,
                "jogos": 30,
                "vitorias": 8,
                "saldo_gols": -5,
                "gols_pro": 25,
                "ultimos_jogos": ["d", "d", "d", "d", "d"]  # 0 pts em 5 jogos -> 0%
            }
        ]

    def test_strengths_50_50(self):
        strengths = calculate_team_strengths(self.sample_standings)

        # Time A: hist = 60 / (3 * 30) = 66.67%, rec = 15 / 15 = 100%
        # Composta = 0.5 * 0.6667 + 0.5 * 1.0 = 0.8333
        st_a = strengths["Time A"]
        self.assertAlmostEqual(st_a["aprov_hist"], 60 / 90, places=4)
        self.assertAlmostEqual(st_a["aprov_rec"], 1.0, places=4)
        self.assertAlmostEqual(st_a["forca_composta"], 0.5 * (60 / 90) + 0.5 * 1.0, places=4)

        # Time B: hist = 30 / 90 = 33.33%, rec = 0 / 15 = 0%
        st_b = strengths["Time B"]
        self.assertAlmostEqual(st_b["aprov_hist"], 30 / 90, places=4)
        self.assertAlmostEqual(st_b["aprov_rec"], 0.0, places=4)
        self.assertAlmostEqual(st_b["forca_composta"], 0.5 * (30 / 90), places=4)

    def test_probabilities_sum_to_one(self):
        strengths = calculate_team_strengths(self.sample_standings)
        pm, pe, pv = calculate_match_probabilities("Time A", "Time B", strengths)
        self.assertAlmostEqual(pm + pe + pv, 1.0, places=6)
        self.assertGreater(pm, 0)
        self.assertGreater(pe, 0)
        self.assertGreater(pv, 0)

    def test_home_advantage(self):
        # Dois times idênticos
        equal_standings = [
            {"nome": "T1", "posicao": 1, "pontos": 45, "jogos": 30, "vitorias": 12, "saldo_gols": 0, "gols_pro": 30, "ultimos_jogos": ["v", "e", "d", "v", "e"]},
            {"nome": "T2", "posicao": 2, "pontos": 45, "jogos": 30, "vitorias": 12, "saldo_gols": 0, "gols_pro": 30, "ultimos_jogos": ["v", "e", "d", "v", "e"]}
        ]
        strengths = calculate_team_strengths(equal_standings)
        pm, pe, pv = calculate_match_probabilities("T1", "T2", strengths, fator_zebra=0.0)
        # Mandante (T1) deve ter maior probabilidade que o visitante (T2)
        self.assertGreater(pm, pv)

    def test_fator_zebra(self):
        strengths = calculate_team_strengths(self.sample_standings)
        # Sem zebra
        pm_no_zebra, _, pv_no_zebra = calculate_match_probabilities("Time A", "Time B", strengths, fator_zebra=0.0)
        # Com zebra de 5%
        pm_zebra, _, pv_zebra = calculate_match_probabilities("Time A", "Time B", strengths, fator_zebra=0.05)

        # O azarão (Time B) ganha chance extra com a zebra
        self.assertGreater(pv_zebra, pv_no_zebra)
        self.assertLess(pm_zebra, pm_no_zebra)


if __name__ == "__main__":
    unittest.main()
