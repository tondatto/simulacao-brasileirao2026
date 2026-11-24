"""
Modelo Poisson de Ataque e Defesa com correção de placares baixos (Dixon-Coles leve).
Usa gols pró e gols contra da tabela atual para simular placares reais.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.models.base import (
    MatchModel,
    apply_zebra_score_matrix,
    build_score_matrix,
    probs_1x2_from_score_matrix,
)


class PoissonModel(MatchModel):
    name = "poisson"
    description = "Poisson Ataque/Defesa (Gols Pró/Contra + Correção Placares Baixos)"
    requires_history = False
    uses_score_simulation = True

    def __init__(self, prior_games: float = 3.0, home_advantage: float = 1.18, rho: float = -0.05):
        self.prior_games = prior_games
        self.home_advantage = home_advantage
        self.rho = rho

    def estimate_ratings(self, standings: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Dict[str, float]]]:
        total_games = sum(int(t.get("jogos", 0)) for t in standings)
        total_goals = sum(int(t.get("gols_pro", 0)) for t in standings)
        mu = (total_goals / total_games) if total_games > 0 and total_goals > 0 else 1.25

        ratings: Dict[str, Dict[str, float]] = {}
        for t in standings:
            jogos = max(0, int(t.get("jogos", 0)))
            gp = max(0, int(t.get("gols_pro", 0)))
            gc = max(0, int(t.get("gols_contra", 0)))

            att_rate = (gp + self.prior_games * mu) / (jogos + self.prior_games)
            def_rate = (gc + self.prior_games * mu) / (jogos + self.prior_games)

            ratings[t["nome"]] = {
                "attack": att_rate / mu,
                "defense": def_rate / mu,
            }
        return mu, ratings

    def prepare_matches(
        self,
        standings: List[Dict[str, Any]],
        matches: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None,
        fator_zebra: float = 0.05,
    ) -> List[Dict[str, Any]]:
        mu, ratings = self.estimate_ratings(standings)
        home_factor = math.sqrt(self.home_advantage)
        away_factor = 1.0 / home_factor

        enriched: List[Dict[str, Any]] = []
        for m in matches:
            r_h = ratings.get(m["mandante"], {"attack": 1.0, "defense": 1.0})
            r_a = ratings.get(m["visitante"], {"attack": 1.0, "defense": 1.0})

            lam_h = float(np.clip(mu * home_factor * r_h["attack"] * r_a["defense"], 0.15, 4.5))
            lam_a = float(np.clip(mu * away_factor * r_a["attack"] * r_h["defense"], 0.15, 4.5))

            raw_mat = build_score_matrix(lam_h, lam_a, rho=self.rho)
            mat = apply_zebra_score_matrix(raw_mat, fator_zebra)
            pm, pe, pv = probs_1x2_from_score_matrix(mat)

            enriched.append({
                **m,
                "p_mandante": pm,
                "p_empate": pe,
                "p_visitante": pv,
                "lambda_mandante": lam_h,
                "lambda_visitante": lam_a,
                "score_matrix": mat,
            })
        return enriched
