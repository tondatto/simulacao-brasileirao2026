#!/usr/bin/env python3
"""
Ponto de entrada principal do Simulador Monte Carlo do Campeonato Brasileiro 2026.
Inspirado na metodologia do Prof. Rogemar Riffel (IF-UFRGS) e dados do Globo Esporte.
"""

import argparse
import sys
import time
from typing import List, Dict, Any

from src.ge_fetcher import fetch_ge_2026_data
from src.model_riffel import calculate_team_strengths, prepare_matches_with_probabilities
from src.simulator import MonteCarloSimulator
from src.stats import compute_team_statistics
from src.terminal_view import (
    print_simulation_header,
    print_standings_table,
    print_highlights,
    print_detailed_matrix
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulador Monte Carlo do Brasileirão 2026 (Metodologia Prof. Rogemar Riffel - UFRGS)"
    )
    parser.add_argument(
        "--simulacoes", "-s",
        type=int,
        default=1_000_000,
        help="Número de simulações Monte Carlo a executar (padrão: 1.000.000)"
    )
    parser.add_argument(
        "--zebra", "-z",
        type=float,
        default=0.05,
        help="Fator de zebra estocástica (padrão: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade estatística (padrão: 42)"
    )
    parser.add_argument(
        "--detalhado", "-d",
        action="store_true",
        help="Exibir matriz detalhada com probabilidades de 1º a 20º para cada time"
    )
    parser.add_argument(
        "--refresh", "-r",
        action="store_true",
        help="Forçar atualização dos dados diretamente do Globo Esporte ignorando o cache local"
    )
    parser.add_argument(
        "--incluir-pendentes-r34",
        action="store_true",
        help="Incluir também os jogos ainda não finalizados da rodada 34 na simulação"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Obtenção de dados
    t0 = time.time()
    try:
        data = fetch_ge_2026_data(force_refresh=args.refresh)
    except Exception as e:
        print(f"Erro ao obter dados do Brasileirão: {e}", file=sys.stderr)
        sys.exit(1)

    standings = data["standings"]

    # 2. Seleção dos confrontos restantes a simular
    matches_to_simulate: List[Dict[str, Any]] = []

    # Se solicitado, inclui os jogos pendentes da rodada 34
    if args.incluir_pendentes_r34 and data.get("pending_r34"):
        matches_to_simulate.extend(data["pending_r34"])
        rodadas_str = f"Rodada 34 pendentes + Rodadas 35 a {data['rodada_final']}"
    else:
        rodadas_str = f"Rodadas 35 a {data['rodada_final']}"

    # Adiciona todos os jogos das rodadas 35 até o final
    for r in range(35, data["rodada_final"] + 1):
        round_matches = data.get("rounds", {}).get(str(r), [])
        matches_to_simulate.extend(round_matches)

    num_matches = len(matches_to_simulate)

    # 3. Cálculo das forças (50% histórico + 50% últimos 5 jogos) e probabilidades com fator de zebra
    strengths = calculate_team_strengths(standings)
    enriched_matches = prepare_matches_with_probabilities(
        matches_to_simulate, strengths, fator_zebra=args.zebra
    )

    # 4. Execução da Simulação Monte Carlo
    simulator = MonteCarloSimulator(
        standings=standings,
        matches=enriched_matches,
        num_simulations=args.simulacoes,
        seed=args.seed
    )

    t_sim_start = time.time()
    sim_results = simulator.run()
    t_sim_elapsed = time.time() - t_sim_start

    # 5. Agregação estatística
    stats_list = compute_team_statistics(standings, sim_results)

    # 6. Exibição no Terminal
    print_simulation_header(
        num_simulations=args.simulacoes,
        num_matches=num_matches,
        rodadas_str=rodadas_str,
        fator_zebra=args.zebra
    )
    print_standings_table(stats_list)
    print_highlights(stats_list)

    if args.detalhado:
        print_detailed_matrix(stats_list)

    print(f"⏱️ Tempo de processamento da simulação: {t_sim_elapsed:.2f}s (Total: {time.time()-t0:.2f}s)")


if __name__ == "__main__":
    main()
