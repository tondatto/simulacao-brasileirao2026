"""
Agregação e cálculo de estatísticas das simulações:
- Faixa de pontos projetados: média ± 1 desvio-padrão (μ ± 1σ)
- Probabilidades de Título (1º), G-4 (1º-4º), G-6 (1º-6º), Sul-Americana (7º-12º) e Rebaixamento Z-4 (17º-20º)
- Posição mais provável e distribuição completa
"""

from typing import Dict, Any, List
import numpy as np


def compute_team_statistics(
    standings: List[Dict[str, Any]],
    sim_results: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Consolida as probabilidades e métricas estatísticas para cada clube.
    Retorna lista ordenada pela classificação atual (ou pela pontuação esperada).
    """
    prob_matrix = sim_results["prob_matrix"]
    mean_points = sim_results["mean_points"]
    std_points = sim_results["std_points"]
    min_points = sim_results["min_points"]
    max_points = sim_results["max_points"]
    team_names = sim_results["team_names"]

    name_to_idx = {name: i for i, name in enumerate(team_names)}

    stats_list = []
    for t in standings:
        nome = t["nome"]
        idx = name_to_idx[nome]
        probs = prob_matrix[idx]  # Array de 20 valores (%)

        # Probabilidades por faixa de objetivo do Brasileirão
        titulo = float(probs[0])
        g4 = float(np.sum(probs[0:4]))
        g6 = float(np.sum(probs[0:6]))
        sula = float(np.sum(probs[6:12]))
        z4 = float(np.sum(probs[16:20]))

        pos_mais_provavel = int(np.argmax(probs) + 1)
        prob_max_pos = float(np.max(probs))

        stats_list.append({
            "pos_atual": t["posicao"],
            "nome": nome,
            "sigla": t.get("sigla", ""),
            "pontos_atuais": t["pontos"],
            "vitorias_atuais": t["vitorias"],
            "saldo_atual": t["saldo_gols"],
            "jogos_atuais": t["jogos"],
            "proj_media": float(mean_points[idx]),
            "proj_std": float(std_points[idx]),
            "proj_min": int(min_points[idx]),
            "proj_max": int(max_points[idx]),
            "titulo_pct": titulo,
            "g4_pct": g4,
            "g6_pct": g6,
            "sula_pct": sula,
            "z4_pct": z4,
            "pos_mais_provavel": pos_mais_provavel,
            "prob_max_pos": prob_max_pos,
            "posicoes_prob": [float(p) for p in probs]
        })

    return stats_list
