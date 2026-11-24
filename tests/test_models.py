import unittest
import numpy as np

from src.models import MODEL_REGISTRY, get_model
from src.models.base import build_score_matrix, probs_1x2_from_score_matrix
from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloModel
from src.models.ensemble import EnsembleModel
from src.models.poisson import PoissonModel


def _build_synthetic_league():
    teams = ["Forte", "MedioA", "MedioB", "Fraco"]
    standings = [
        {"nome": "Forte", "posicao": 1, "pontos": 24, "jogos": 10, "vitorias": 8, "empates": 0, "derrotas": 2, "gols_pro": 24, "gols_contra": 6, "saldo_gols": 18, "ultimos_jogos": ["v", "v", "v", "v", "v"]},
        {"nome": "MedioA", "posicao": 2, "pontos": 15, "jogos": 10, "vitorias": 4, "empates": 3, "derrotas": 3, "gols_pro": 13, "gols_contra": 12, "saldo_gols": 1, "ultimos_jogos": ["v", "e", "d", "v", "e"]},
        {"nome": "MedioB", "posicao": 3, "pontos": 14, "jogos": 10, "vitorias": 4, "empates": 2, "derrotas": 4, "gols_pro": 12, "gols_contra": 13, "saldo_gols": -1, "ultimos_jogos": ["e", "v", "d", "e", "v"]},
        {"nome": "Fraco", "posicao": 4, "pontos": 4, "jogos": 10, "vitorias": 1, "empates": 1, "derrotas": 8, "gols_pro": 6, "gols_contra": 24, "saldo_gols": -18, "ultimos_jogos": ["d", "d", "e", "d", "d"]},
    ]
    # Liga sintética com múltiplos turnos para convergência do Dixon-Coles
    history = []
    match_id = 1
    results_grid = {
        ("Forte", "Fraco"): (3, 0),
        ("Forte", "MedioA"): (2, 0),
        ("Forte", "MedioB"): (2, 1),
        ("MedioA", "Forte"): (1, 2),
        ("MedioB", "Forte"): (0, 1),
        ("Fraco", "Forte"): (0, 3),
        ("MedioA", "MedioB"): (1, 1),
        ("MedioB", "MedioA"): (1, 1),
        ("MedioA", "Fraco"): (2, 0),
        ("MedioB", "Fraco"): (2, 1),
        ("Fraco", "MedioA"): (0, 1),
        ("Fraco", "MedioB"): (1, 1),
    }
    for cycle in range(3):
        for r_idx, ((h, a), (gh, ga)) in enumerate(results_grid.items(), start=1):
            history.append({
                "id_jogo": match_id,
                "rodada": cycle * 4 + ((r_idx - 1) // 3) + 1,
                "data_realizacao": f"2026-06-{min(28, cycle * 9 + r_idx):02d}T16:00",
                "mandante": h,
                "visitante": a,
                "placar_mandante": gh,
                "placar_visitante": ga,
                "finalizado": True,
            })
            match_id += 1
    upcoming = [
        {"id_jogo": 9001, "rodada": 13, "mandante": "Forte", "visitante": "Fraco", "finalizado": False},
        {"id_jogo": 9002, "rodada": 13, "mandante": "MedioA", "visitante": "MedioB", "finalizado": False},
    ]
    return teams, standings, history, upcoming


class TestModels(unittest.TestCase):
    def test_registry_lookup(self):
        for name in ["riffel", "poisson", "elo", "dixon-coles", "ensemble"]:
            self.assertIn(name, MODEL_REGISTRY)
            self.assertEqual(get_model(name).name, name)
        with self.assertRaises(ValueError):
            get_model("inexistente")

    def test_poisson_model_score_matrix_and_home_advantage(self):
        _, standings, _, upcoming = _build_synthetic_league()
        model = PoissonModel()
        enriched = model.prepare_matches(standings, upcoming, fator_zebra=0.0)
        for m in enriched:
            mat = m["score_matrix"]
            self.assertEqual(mat.shape, (9, 9))
            self.assertAlmostEqual(float(mat.sum()), 1.0, places=6)
            self.assertTrue(np.all(mat >= 0.0))
            self.assertAlmostEqual(m["p_mandante"] + m["p_empate"] + m["p_visitante"], 1.0, places=6)

        # Forte x Fraco tem probabilidade de mandante muito maior
        self.assertGreater(enriched[0]["p_mandante"], enriched[0]["p_visitante"])

        # Times iguais devem favorecer o mandante pela vantagem de casa
        equal_standings = [
            {"nome": "T1", "jogos": 10, "gols_pro": 12, "gols_contra": 12},
            {"nome": "T2", "jogos": 10, "gols_pro": 12, "gols_contra": 12},
        ]
        eq_res = model.prepare_matches(equal_standings, [{"mandante": "T1", "visitante": "T2"}], fator_zebra=0.0)[0]
        self.assertGreater(eq_res["p_mandante"], eq_res["p_visitante"])

        # Fator zebra reduz assimetria em favor do azarão
        enriched_z = model.prepare_matches(standings, upcoming[:1], fator_zebra=0.10)[0]
        self.assertGreater(enriched_z["p_visitante"], enriched[0]["p_visitante"])

    def test_elo_model_conservation_margin_and_normalization(self):
        teams, standings, history, upcoming = _build_synthetic_league()
        model = EloModel()
        ratings, draw_rate = model.compute_ratings(standings, history)
        self.assertAlmostEqual(sum(ratings.values()), len(teams) * model.initial_rating, places=6)
        self.assertGreater(ratings["Forte"], ratings["Fraco"])
        self.assertGreaterEqual(draw_rate, 0.15)
        self.assertLessEqual(draw_rate, 0.38)

        # Margem maior de gols gera maior ganho de rating
        m1 = [{"mandante": "MedioA", "visitante": "MedioB", "placar_mandante": 1, "placar_visitante": 0}]
        m3 = [{"mandante": "MedioA", "visitante": "MedioB", "placar_mandante": 3, "placar_visitante": 0}]
        r1, _ = model.compute_ratings(standings, m1)
        r3, _ = model.compute_ratings(standings, m3)
        self.assertGreater(r3["MedioA"], r1["MedioA"])

        enriched = model.prepare_matches(standings, upcoming, history=history, fator_zebra=0.05)
        for m in enriched:
            self.assertAlmostEqual(m["p_mandante"] + m["p_empate"] + m["p_visitante"], 1.0, places=6)

    def test_dixon_coles_converges_and_produces_valid_matrices(self):
        _, standings, history, upcoming = _build_synthetic_league()
        model = DixonColesModel(xi=0.003)
        fitted = model.fit(standings, history)
        self.assertTrue(fitted["success"])
        self.assertGreater(fitted["attack"]["Forte"], fitted["attack"]["Fraco"])
        self.assertLess(fitted["defense"]["Forte"], fitted["defense"]["Fraco"])

        # Matriz com rho != 0 continua normalizada e não-negativa
        mat = build_score_matrix(1.6, 1.1, rho=-0.1)
        self.assertAlmostEqual(float(mat.sum()), 1.0, places=6)
        self.assertTrue(np.all(mat >= 0.0))
        pm, pe, pv = probs_1x2_from_score_matrix(mat)
        self.assertAlmostEqual(pm + pe + pv, 1.0, places=6)

        # Sensibilidade ao decaimento temporal xi
        w_no_decay = model._compute_time_weights(history, xi=0.0)
        w_decay = model._compute_time_weights(history, xi=0.05)
        self.assertTrue(np.allclose(w_no_decay, 1.0))
        self.assertLess(w_decay[0], w_decay[-1])

        enriched = model.prepare_matches(standings, upcoming, history=history, fator_zebra=0.05)
        self.assertGreater(enriched[0]["p_mandante"], enriched[0]["p_visitante"])
        self.assertAlmostEqual(float(enriched[0]["score_matrix"].sum()), 1.0, places=6)

    def test_ensemble_model_within_constituent_bounds(self):
        _, standings, history, upcoming = _build_synthetic_league()
        ens = EnsembleModel()
        ens_res = ens.prepare_matches(standings, upcoming, history=history, fator_zebra=0.05)
        indiv_res = [
            m.prepare_matches(standings, upcoming, history=history, fator_zebra=0.05)
            for m in ens.models
        ]
        for idx, em in enumerate(ens_res):
            self.assertAlmostEqual(em["p_mandante"] + em["p_empate"] + em["p_visitante"], 1.0, places=6)
            pms = [r[idx]["p_mandante"] for r in indiv_res]
            self.assertGreaterEqual(em["p_mandante"], min(pms) - 1e-9)
            self.assertLessEqual(em["p_mandante"], max(pms) + 1e-9)


if __name__ == "__main__":
    unittest.main()
