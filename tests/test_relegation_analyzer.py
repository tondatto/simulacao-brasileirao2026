import unittest

from src.models.poisson import PoissonModel
from src.relegation_analyzer import analyze_team_relegation, resolve_team_name


def _make_standing(pos, nome, sigla, pontos, vitorias, sg, gp=30, gc=30, jogos=36):
    return {
        "nome": nome,
        "sigla": sigla,
        "posicao": pos,
        "pontos": pontos,
        "jogos": jogos,
        "vitorias": vitorias,
        "empates": max(0, pontos - 3 * vitorias),
        "derrotas": max(0, jogos - vitorias - max(0, pontos - 3 * vitorias)),
        "gols_pro": gp,
        "gols_contra": gc,
        "saldo_gols": sg,
        "ultimos_jogos": ["e", "e", "e", "e", "e"],
    }


def _make_league_standings():
    # Liga enxuta com 6 times (Z-4 simulado com relegation_slots=2 => safe_max_pos=4)
    return [
        _make_standing(1, "Líder", "LID", 75, 23, 30, 60, 30),
        _make_standing(2, "Vice", "VIC", 65, 20, 20, 50, 30),
        _make_standing(3, "São Paulo", "SAO", 45, 13, 2, 35, 33),
        _make_standing(4, "Grêmio", "GRE", 41, 11, -4, 30, 34),
        _make_standing(5, "Internacional", "INT", 40, 10, -5, 29, 34),
        _make_standing(6, "Lanterna", "LAN", 25, 6, -25, 20, 45),
    ]


def _m(rodada, home, away, pm=0.4, pe=0.3, pv=0.3):
    return {
        "rodada": rodada,
        "mandante": home,
        "visitante": away,
        "p_mandante": pm,
        "p_empate": pe,
        "p_visitante": pv,
    }


class TestRelegationAnalyzer(unittest.TestCase):
    def test_resolve_team_name_variants_and_errors(self):
        standings = [
            _make_standing(1, "Grêmio", "GRE", 41, 11, -4),
            _make_standing(2, "São Paulo", "SAO", 45, 13, 2),
            _make_standing(3, "Internacional", "INT", 40, 10, -5),
            _make_standing(4, "Atlético-MG", "CAM", 50, 14, 4),
            _make_standing(5, "Athletico-PR", "CAP", 57, 16, 9),
            _make_standing(6, "Chapecoense", "CHA", 21, 4, -40),
        ]
        self.assertEqual(resolve_team_name("gremio", standings), "Grêmio")
        self.assertEqual(resolve_team_name("SAO PAULO", standings), "São Paulo")
        self.assertEqual(resolve_team_name("int", standings), "Internacional")
        self.assertEqual(resolve_team_name("chape", standings), "Chapecoense")

        # Substring ambígua ("at" bate em Atlético-MG e Athletico-PR? "at" só está em Atlético se sem h; "a" bate em vários)
        with self.assertRaises(ValueError):
            resolve_team_name("a", standings)
        with self.assertRaises(ValueError):
            resolve_team_name("TimeInexistente", standings)

    def test_mathematically_relegated_team(self):
        standings = _make_league_standings()
        matches = [
            _m(37, "Lanterna", "Líder", 0.2, 0.2, 0.6),
            _m(38, "Vice", "Lanterna", 0.7, 0.2, 0.1),
        ]
        res = analyze_team_relegation(
            standings, matches, "Lanterna", num_simulations=2000, seed=1, relegation_slots=2
        )
        self.assertEqual(res["status"], "ja_rebaixado")
        self.assertAlmostEqual(res["best_survival_pct"], 0.0, places=3)
        self.assertTrue(all(c["prob_escapar_pct"] == 0.0 for c in res["combinations"]))

    def test_mathematically_safe_team(self):
        standings = _make_league_standings()
        matches = [
            _m(37, "Líder", "São Paulo", 0.7, 0.2, 0.1),
            _m(38, "Líder", "Grêmio", 0.7, 0.2, 0.1),
        ]
        res = analyze_team_relegation(
            standings, matches, "Líder", num_simulations=2000, seed=1, relegation_slots=2
        )
        self.assertEqual(res["status"], "ja_salvo")
        self.assertAlmostEqual(res["worst_survival_pct"], 100.0, places=3)
        self.assertEqual(res["key_rival_matches"], [])

    def test_combination_probabilities_and_total_probability_consistency(self):
        standings = _make_league_standings()
        # Internacional (40 pts) enfrenta Líder e Vice; Grêmio (41 pts) enfrenta São Paulo e Vice
        # Se Internacional vencer os dois (46 pts) e São Paulo (45) / Grêmio (41) jogarem contra outros:
        # Vamos colocar São Paulo=45 e Grêmio=41; para depender só de si com 46 pts?
        # Em 6 times com relegation_slots=2, 5º e 6º caem. Líder=75, Vice=65, São Paulo=45, Grêmio=41.
        # Se Internacional (40) chega a 46 pts, passa Grêmio (se Grêmio não chegar a 47 — Grêmio tem 1 jogo apenas!).
        matches = [
            _m(37, "Internacional", "Líder", 0.3, 0.3, 0.4),
            _m(38, "Internacional", "Vice", 0.35, 0.3, 0.35),
            _m(38, "Grêmio", "São Paulo", 0.4, 0.3, 0.3),  # Grêmio max 44 pts!
        ]
        res = analyze_team_relegation(
            standings, matches, "Internacional", num_simulations=5000, seed=42, relegation_slots=2
        )
        self.assertEqual(len(res["combinations"]), 9)
        total_combo_prob = sum(c["prob_cenario_pct"] for c in res["combinations"])
        self.assertAlmostEqual(total_combo_prob, 100.0, places=4)
        self.assertEqual(res["status"], "depende_so_de_si")

        expected_base = sum(
            (c["prob_cenario_pct"] / 100.0) * c["prob_escapar_pct"]
            for c in res["combinations"]
        )
        self.assertAlmostEqual(res["base_survival_pct"], expected_base, places=4)

    def test_six_pointer_opponent_update(self):
        # Grêmio (41 pts, 11V) vs Internacional (40 pts, 10V) em jogo único restante
        standings = _make_league_standings()
        matches = [_m(38, "Grêmio", "Internacional", 0.4, 0.3, 0.3)]
        res = analyze_team_relegation(
            standings, matches, "Internacional", num_simulations=1000, seed=7, relegation_slots=2
        )
        by_combo = {c["combo"]: c for c in res["combinations"]}
        # Se Inter vence (43 pts), Grêmio estaciona em 41 -> 100% de permanência
        self.assertAlmostEqual(by_combo[("V",)]["prob_escapar_pct"], 100.0, places=3)
        # Se Inter empata (41 pts, 10V) ou perde (40 pts), Grêmio fica à frente (42/44) -> 0%
        self.assertAlmostEqual(by_combo[("E",)]["prob_escapar_pct"], 0.0, places=3)
        self.assertAlmostEqual(by_combo[("D",)]["prob_escapar_pct"], 0.0, places=3)

    def test_score_matrix_compatibility(self):
        standings = _make_league_standings()
        raw_matches = [
            {"rodada": 37, "mandante": "Internacional", "visitante": "Lanterna"},
            {"rodada": 38, "mandante": "Grêmio", "visitante": "Líder"},
        ]
        enriched = PoissonModel().prepare_matches(standings, raw_matches, fator_zebra=0.05)
        res = analyze_team_relegation(
            standings, enriched, "Internacional", num_simulations=3000, seed=11, relegation_slots=2
        )
        self.assertEqual(len(res["combinations"]), 3)
        self.assertAlmostEqual(sum(c["prob_cenario_pct"] for c in res["combinations"]), 100.0, places=4)

    def test_key_rival_matches_identifies_decisive_fixture(self):
        # Inter tem 40 pts e joga contra Líder (40->43 se vencer). Grêmio tem 41 pts e joga contra Lanterna.
        # Vitória do Lanterna deixa Grêmio com 41, enquanto derrota do Lanterna põe Grêmio com 44 (intocável)!
        standings = _make_league_standings()
        matches = [
            _m(38, "Internacional", "Líder", 0.5, 0.25, 0.25),
            _m(38, "Grêmio", "Lanterna", 0.5, 0.25, 0.25),
        ]
        res = analyze_team_relegation(
            standings, matches, "Internacional", num_simulations=6000, seed=99, relegation_slots=2
        )
        self.assertTrue(len(res["key_rival_matches"]) >= 1)
        top_rival = res["key_rival_matches"][0]
        self.assertEqual(top_rival["mandante"], "Grêmio")
        self.assertEqual(top_rival["visitante"], "Lanterna")
        self.assertEqual(top_rival["melhor_resultado"], "Vitória do Lanterna")
        self.assertGreater(top_rival["impacto_pp"], 20.0)

    def test_zero_remaining_matches_for_team(self):
        standings = _make_league_standings()
        # Inter já encerrou com 40 pts; Grêmio tem 41 pts -> Inter já está atrás de 4 times (1..4) e não tem jogos
        matches = [_m(38, "Líder", "Vice", 0.5, 0.25, 0.25)]
        res = analyze_team_relegation(
            standings, matches, "Internacional", num_simulations=500, seed=42, relegation_slots=2
        )
        self.assertEqual(res["team"]["jogos_restantes"], 0)
        self.assertEqual(len(res["combinations"]), 1)
        self.assertEqual(res["status"], "ja_rebaixado")


if __name__ == "__main__":
    unittest.main()
