"""
Modelo Rating Elo Dinâmico com multiplicador de margem de gols (World Football Elo)
e conversão para probabilidades 1X2 via extensão de Davidson calibrada na liga.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.models.base import MatchModel, apply_zebra_1x2


class EloModel(MatchModel):
    name = "elo"
    description = "Rating Elo Dinâmico (Histórico Cronológico + Margem de Gols)"
    requires_history = True
    uses_score_simulation = False

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 25.0,
        home_advantage: float = 70.0,
    ):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage

    def compute_ratings(
        self,
        standings: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, float], float]:
        ratings: Dict[str, float] = {t["nome"]: self.initial_rating for t in standings}
        if not history:
            return ratings, 0.27

        draws = 0
        valid_matches = 0

        for m in history:
            gh = m.get("placar_mandante")
            ga = m.get("placar_visitante")
            if gh is None or ga is None:
                continue

            home = m["mandante"]
            away = m["visitante"]
            r_h = ratings.get(home, self.initial_rating)
            r_a = ratings.get(away, self.initial_rating)

            we = 1.0 / (1.0 + 10.0 ** (-(r_h + self.home_advantage - r_a) / 400.0))

            if gh > ga:
                w = 1.0
            elif gh == ga:
                w = 0.5
                draws += 1
            else:
                w = 0.0

            diff = abs(int(gh) - int(ga))
            if diff <= 1:
                g_mult = 1.0
            elif diff == 2:
                g_mult = 1.5
            else:
                g_mult = (11.0 + diff) / 8.0

            delta = self.k_factor * g_mult * (w - we)
            ratings[home] = r_h + delta
            ratings[away] = r_a - delta
            valid_matches += 1

        draw_rate = float(np.clip(draws / valid_matches, 0.15, 0.38)) if valid_matches > 0 else 0.27
        return ratings, draw_rate

    def match_probabilities(
        self,
        r_home: float,
        r_away: float,
        draw_rate: float,
        fator_zebra: float,
    ) -> Tuple[float, float, float]:
        # Calibração de Davidson: quando pi_h == pi_a, P(E) = nu / (2 + nu) = draw_rate
        nu = (2.0 * draw_rate) / max(1e-6, 1.0 - draw_rate)
        pi_h = 10.0 ** ((r_home + self.home_advantage) / 400.0)
        pi_a = 10.0 ** (r_away / 400.0)
        draw_term = nu * math.sqrt(pi_h * pi_a)
        denom = pi_h + pi_a + draw_term

        pm = pi_h / denom
        pe = draw_term / denom
        pv = pi_a / denom
        return apply_zebra_1x2(pm, pe, pv, fator_zebra)

    def prepare_matches(
        self,
        standings: List[Dict[str, Any]],
        matches: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None,
        fator_zebra: float = 0.05,
    ) -> List[Dict[str, Any]]:
        ratings, draw_rate = self.compute_ratings(standings, history)
        enriched: List[Dict[str, Any]] = []
        for m in matches:
            r_h = ratings.get(m["mandante"], self.initial_rating)
            r_a = ratings.get(m["visitante"], self.initial_rating)
            pm, pe, pv = self.match_probabilities(r_h, r_a, draw_rate, fator_zebra)
            enriched.append({
                **m,
                "p_mandante": pm,
                "p_empate": pe,
                "p_visitante": pv,
            })
        return enriched
