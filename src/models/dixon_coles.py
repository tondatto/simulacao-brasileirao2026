"""
Modelo Dixon-Coles (1997) completo ajustado por Máxima Verossimilhança (SciPy L-BFGS-B)
com decaimento temporal exponencial e simulação de placares reais.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize

from src.models.base import (
    MatchModel,
    apply_zebra_score_matrix,
    build_score_matrix,
    probs_1x2_from_score_matrix,
)
from src.models.poisson import PoissonModel


class DixonColesModel(MatchModel):
    name = "dixon-coles"
    description = "Dixon-Coles MLE (SciPy L-BFGS-B + Decaimento Temporal)"
    requires_history = True
    uses_score_simulation = True

    def __init__(self, xi: float = 0.003, l2_reg: float = 0.01):
        self.xi = xi
        self.l2_reg = l2_reg

    @staticmethod
    def _compute_time_weights(matches: List[Dict[str, Any]], xi: float) -> np.ndarray:
        if xi <= 0.0:
            return np.ones(len(matches), dtype=np.float64)

        parsed_dates: List[Optional[datetime]] = []
        for m in matches:
            raw_dt = m.get("data_realizacao")
            dt: Optional[datetime] = None
            if isinstance(raw_dt, str) and raw_dt:
                try:
                    dt = datetime.fromisoformat(raw_dt)
                except ValueError:
                    dt = None
            parsed_dates.append(dt)

        if all(dt is not None for dt in parsed_dates):
            max_dt = max(dt for dt in parsed_dates if dt is not None)
            deltas = np.array(
                [(max_dt - dt).total_seconds() / 86400.0 for dt in parsed_dates if dt is not None],
                dtype=np.float64,
            )
        else:
            rounds = [int(m.get("rodada") or 0) for m in matches]
            max_round = max(rounds) if rounds else 0
            deltas = np.array([(max_round - r) * 7.0 for r in rounds], dtype=np.float64)

        return np.exp(-xi * np.maximum(0.0, deltas))

    def fit(
        self,
        standings: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        valid_history = [
            m for m in history
            if m.get("placar_mandante") is not None and m.get("placar_visitante") is not None
        ]
        team_names = sorted({t["nome"] for t in standings} | {m["mandante"] for m in valid_history} | {m["visitante"] for m in valid_history})
        team_to_idx = {name: i for i, name in enumerate(team_names)}
        n_teams = len(team_names)

        h_idx = np.array([team_to_idx[m["mandante"]] for m in valid_history], dtype=np.int64)
        a_idx = np.array([team_to_idx[m["visitante"]] for m in valid_history], dtype=np.int64)
        y_h = np.array([int(m["placar_mandante"]) for m in valid_history], dtype=np.float64)
        y_a = np.array([int(m["placar_visitante"]) for m in valid_history], dtype=np.float64)
        weights = self._compute_time_weights(valid_history, self.xi)

        m00 = (y_h == 0) & (y_a == 0)
        m01 = (y_h == 0) & (y_a == 1)
        m10 = (y_h == 1) & (y_a == 0)
        m11 = (y_h == 1) & (y_a == 1)
        n_matches = len(valid_history)

        def neg_log_likelihood(params: np.ndarray) -> float:
            attack = params[:n_teams]
            defense = params[n_teams : 2 * n_teams]
            home_adv = params[2 * n_teams]
            rho = params[2 * n_teams + 1]

            eta_h = np.clip(attack[h_idx] + defense[a_idx] + home_adv, -4.0, 2.5)
            eta_a = np.clip(attack[a_idx] + defense[h_idx], -4.0, 2.5)
            lam_h = np.exp(eta_h)
            lam_a = np.exp(eta_a)

            tau = np.ones(n_matches, dtype=np.float64)
            if np.any(m00):
                tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
            if np.any(m01):
                tau[m01] = 1.0 + lam_h[m01] * rho
            if np.any(m10):
                tau[m10] = 1.0 + lam_a[m10] * rho
            if np.any(m11):
                tau[m11] = 1.0 - rho

            if np.any(tau <= 1e-9):
                return 1e12

            nll = float(np.sum(weights * (lam_h - y_h * eta_h + lam_a - y_a * eta_a - np.log(tau))))
            reg = self.l2_reg * float(np.sum(attack ** 2) + np.sum(defense ** 2))
            ident = 10.0 * float(np.mean(attack) ** 2)
            return nll + reg + ident

        x0 = np.zeros(2 * n_teams + 2, dtype=np.float64)
        x0[2 * n_teams] = 0.2
        x0[2 * n_teams + 1] = -0.05

        bounds = [(-2.0, 2.0)] * (2 * n_teams) + [(0.0, 0.8), (-0.25, 0.25)]
        res = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)

        params = res.x
        attack = params[:n_teams]
        defense = params[n_teams : 2 * n_teams]
        return {
            "attack": {name: float(attack[i]) for i, name in enumerate(team_names)},
            "defense": {name: float(defense[i]) for i, name in enumerate(team_names)},
            "home_adv": float(params[2 * n_teams]),
            "rho": float(params[2 * n_teams + 1]),
            "success": bool(res.success),
        }

    def prepare_matches(
        self,
        standings: List[Dict[str, Any]],
        matches: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None,
        fator_zebra: float = 0.05,
    ) -> List[Dict[str, Any]]:
        if not history:
            return PoissonModel().prepare_matches(standings, matches, history=history, fator_zebra=fator_zebra)

        fitted = self.fit(standings, history)
        attack = fitted["attack"]
        defense = fitted["defense"]
        home_adv = fitted["home_adv"]
        rho = fitted["rho"]

        enriched: List[Dict[str, Any]] = []
        for m in matches:
            home = m["mandante"]
            away = m["visitante"]
            lam_h = float(np.exp(np.clip(attack.get(home, 0.0) + defense.get(away, 0.0) + home_adv, -4.0, 2.5)))
            lam_a = float(np.exp(np.clip(attack.get(away, 0.0) + defense.get(home, 0.0), -4.0, 2.5)))

            raw_mat = build_score_matrix(lam_h, lam_a, rho=rho)
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
