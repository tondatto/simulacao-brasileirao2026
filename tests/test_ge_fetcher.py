import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.ge_fetcher import fetch_ge_2026_data, fetch_season_history, select_pending_matches


def _make_match(rodada: int, home: str, away: str, finalizado: bool, match_id: int) -> dict:
    return {
        "id_jogo": match_id,
        "rodada": rodada,
        "data_realizacao": "2026-11-23T19:30",
        "mandante": home,
        "mandante_sigla": home[:3].upper(),
        "visitante": away,
        "visitante_sigla": away[:3].upper(),
        "placar_mandante": 1 if finalizado else None,
        "placar_visitante": 0 if finalizado else None,
        "finalizado": finalizado,
    }


def _make_round(rodada: int, finished_count: int, total: int = 10, id_offset: int = 0) -> list:
    matches = []
    for i in range(total):
        matches.append(
            _make_match(
                rodada=rodada,
                home=f"Time{i * 2 + 1}",
                away=f"Time{i * 2 + 2}",
                finalizado=(i < finished_count),
                match_id=id_offset + i + 1,
            )
        )
    return matches


class TestGeFetcher(unittest.TestCase):
    def test_select_pending_matches_filters_finished_and_formats_partial_round(self):
        data = {
            "rounds": {
                "35": _make_round(35, finished_count=10, id_offset=3500),
                "36": _make_round(36, finished_count=6, id_offset=3600),
                "37": _make_round(37, finished_count=0, id_offset=3700),
                "38": _make_round(38, finished_count=0, id_offset=3800),
            }
        }
        matches, label = select_pending_matches(data)
        self.assertEqual(len(matches), 24)
        self.assertTrue(all(not m["finalizado"] for m in matches))
        self.assertEqual(label, "Rodada 36 (4 pendentes) + Rodadas 37 a 38")

    def test_select_pending_matches_all_rounds_pending(self):
        data = {
            "rounds": {
                "35": _make_round(35, finished_count=0, id_offset=3500),
                "36": _make_round(36, finished_count=0, id_offset=3600),
                "37": _make_round(37, finished_count=0, id_offset=3700),
                "38": _make_round(38, finished_count=0, id_offset=3800),
            }
        }
        matches, label = select_pending_matches(data)
        self.assertEqual(len(matches), 40)
        self.assertEqual(label, "Rodadas 35 a 38")

    def test_select_pending_matches_with_postponed_earlier_round(self):
        data = {
            "rounds": {
                "32": _make_round(32, finished_count=9, id_offset=3200),
                "36": _make_round(36, finished_count=6, id_offset=3600),
                "37": _make_round(37, finished_count=0, id_offset=3700),
                "38": _make_round(38, finished_count=0, id_offset=3800),
            }
        }
        matches, label = select_pending_matches(data)
        self.assertEqual(len(matches), 25)
        self.assertEqual(matches[0]["rodada"], 32)
        self.assertEqual(label, "Rodada 32 (1 pendente) + Rodada 36 (4 pendentes) + Rodadas 37 a 38")

    def test_select_pending_matches_legacy_pending_r34_deduplication(self):
        r34_pending_dup = _make_match(34, "Time1", "Time2", finalizado=False, match_id=3401)
        r34_pending_only = _make_match(34, "Time3", "Time4", finalizado=False, match_id=3402)
        r34_finished = _make_match(34, "Time5", "Time6", finalizado=True, match_id=3403)
        data = {
            "rounds": {
                "34": [r34_pending_dup],
                "37": _make_round(37, finished_count=0, id_offset=3700),
            },
            "pending_r34": [r34_pending_dup, r34_pending_only, r34_finished],
        }
        matches, _ = select_pending_matches(data)
        ids = [m["id_jogo"] for m in matches]
        self.assertEqual(ids.count(3401), 1)
        self.assertIn(3402, ids)
        self.assertNotIn(3403, ids)
        self.assertEqual(len(matches), 12)

    def test_select_pending_matches_championship_finished(self):
        data = {
            "rounds": {
                "38": _make_round(38, finished_count=10, id_offset=3800),
            }
        }
        matches, label = select_pending_matches(data)
        self.assertEqual(matches, [])
        self.assertEqual(label, "Campeonato encerrado")

    def test_fetch_ge_2026_data_starts_at_rodada_atual_without_extra_calls(self):
        standings_raw = [
            {
                "equipe_id": i + 1,
                "nome_popular": f"Time{i + 1}",
                "sigla": f"T{i + 1:02d}",
                "ordem": i + 1,
                "pontos": 50 - i,
                "jogos": 35 if i < 8 else 36,
                "vitorias": 15,
                "empates": 5,
                "derrotas": 15,
                "gols_pro": 40,
                "gols_contra": 40,
                "saldo_gols": 0,
                "aproveitamento": 50.0,
                "ultimos_jogos": ["v", "e", "d", "v", "e"],
            }
            for i in range(20)
        ]
        # Rodada 36 tem os primeiros 4 jogos pendentes (Time1..Time8 com 35 jogos + 1 pendente na R36 + 2 em R37/R38 = 38)
        rounds_map = {
            36: [
                _make_match(36, f"Time{i * 2 + 1}", f"Time{i * 2 + 2}", finalizado=(i >= 4), match_id=3600 + i)
                for i in range(10)
            ],
            37: _make_round(37, finished_count=0, id_offset=3700),
            38: _make_round(38, finished_count=0, id_offset=3800),
        }
        html_payload = {
            "fase": {"slug": "fase-unica-2026"},
            "rodada": {"atual": 36, "ultima": 38},
            "edicao": {"nome": "Campeonato Brasileiro 2026"},
            "classificacao": standings_raw,
        }
        fake_html = f'<div data-bs-resource-id="res-123"></div><script>const classificacao = {json.dumps(html_payload)};</script>'

        called_rounds = []

        def fake_fetch_round(resource_id, slug_fase, rodada):
            called_rounds.append(rodada)
            return rounds_map[rodada]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.ge_fetcher._request_url", return_value=fake_html), \
                 patch("src.ge_fetcher._fetch_round_matches", side_effect=fake_fetch_round):
                result = fetch_ge_2026_data(force_refresh=True, cache_dir=tmpdir)

            self.assertEqual(called_rounds, [36, 37, 38])
            self.assertEqual(list(result["rounds"].keys()), ["36", "37", "38"])
            self.assertNotIn("pending_r34", result)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "ge_2026_data.json")))

    def test_fetch_ge_2026_data_backfills_postponed_earlier_round(self):
        # Todos em R36 com 36 jogos, exceto Time1 e Time2 com 35 jogos por jogo adiado na R34
        standings_raw = [
            {
                "equipe_id": i + 1,
                "nome_popular": f"Time{i + 1}",
                "sigla": f"T{i + 1:02d}",
                "ordem": i + 1,
                "pontos": 50 - i,
                "jogos": 35 if i < 2 else 36,
                "vitorias": 15,
                "empates": 5,
                "derrotas": 15,
                "gols_pro": 40,
                "gols_contra": 40,
                "saldo_gols": 0,
                "aproveitamento": 50.0,
                "ultimos_jogos": ["v", "e", "d", "v", "e"],
            }
            for i in range(20)
        ]
        rounds_map = {
            34: [
                _make_match(34, f"Time{i * 2 + 1}", f"Time{i * 2 + 2}", finalizado=(i != 0), match_id=3400 + i)
                for i in range(10)
            ],
            35: _make_round(35, finished_count=10, id_offset=3500),
            36: _make_round(36, finished_count=10, id_offset=3600),
            37: _make_round(37, finished_count=0, id_offset=3700),
            38: _make_round(38, finished_count=0, id_offset=3800),
        }
        html_payload = {
            "fase": {"slug": "fase-unica-2026"},
            "rodada": {"atual": 36, "ultima": 38},
            "edicao": {"nome": "Campeonato Brasileiro 2026"},
            "classificacao": standings_raw,
        }
        fake_html = f'<div data-bs-resource-id="res-123"></div><script>const classificacao = {json.dumps(html_payload)};</script>'

        called_rounds = []

        def fake_fetch_round(resource_id, slug_fase, rodada):
            called_rounds.append(rodada)
            return rounds_map[rodada]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.ge_fetcher._request_url", return_value=fake_html), \
                 patch("src.ge_fetcher._fetch_round_matches", side_effect=fake_fetch_round):
                result = fetch_ge_2026_data(force_refresh=True, cache_dir=tmpdir)

            self.assertEqual(called_rounds, [36, 37, 38, 35, 34])
            self.assertEqual(list(result["rounds"].keys()), ["34", "36", "37", "38"])

    def test_fetch_season_history_incremental_cache_and_ordering(self):
        data = {
            "campeonato": "Campeonato Brasileiro 2026",
            "resource_id": "res-123",
            "slug_fase": "fase-unica-2026",
            "rodada_atual": 3,
            "rounds": {
                "3": [
                    _make_match(3, "Time1", "Time2", finalizado=True, match_id=301),
                    _make_match(3, "Time3", "Time4", finalizado=False, match_id=302),
                ]
            },
        }
        rounds_api = {
            1: [_make_match(1, "Time1", "Time3", finalizado=True, match_id=101)],
            2: [_make_match(2, "Time2", "Time4", finalizado=True, match_id=201)],
        }
        calls = []

        def fake_fetch_round(resource_id, slug_fase, rodada):
            calls.append(rodada)
            return rounds_api[rodada]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.ge_fetcher._fetch_round_matches", side_effect=fake_fetch_round):
                hist1 = fetch_season_history(data, force_refresh=False, cache_dir=tmpdir)
                self.assertEqual(calls, [1, 2])
                self.assertEqual([m["id_jogo"] for m in hist1], [101, 201, 301])

                # Segunda chamada sem force_refresh deve reutilizar rodadas 1 e 2 já finalizadas
                calls.clear()
                hist2 = fetch_season_history(data, force_refresh=False, cache_dir=tmpdir)
                self.assertEqual(calls, [])
                self.assertEqual([m["id_jogo"] for m in hist2], [101, 201, 301])


if __name__ == "__main__":
    unittest.main()
