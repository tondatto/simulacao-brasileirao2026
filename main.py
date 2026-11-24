#!/usr/bin/env python3
"""
Ponto de entrada principal do Simulador Monte Carlo do Campeonato Brasileiro 2026.
Inspirado na metodologia do Prof. Rogemar Riffel (IF-UFRGS) e dados do Globo Esporte.
"""

import argparse
import sys
import time
from typing import List, Dict, Any

from src.ge_fetcher import fetch_ge_2026_data, fetch_season_history, select_pending_matches
from src.models import MODEL_REGISTRY, get_model
from src.relegation_analyzer import analyze_team_relegation, resolve_team_name
from src.simulator import MonteCarloSimulator
from src.stats import compute_team_statistics
from src.terminal_view import (
    print_simulation_header,
    print_standings_table,
    print_highlights,
    print_detailed_matrix,
    print_relegation_analysis,
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
        "--modelo", "-m",
        choices=list(MODEL_REGISTRY.keys()),
        default="riffel",
        help="Metodologia estatística para projeção das partidas (padrão: riffel)"
    )
    parser.add_argument(
        "--analisar-time", "-t",
        type=str,
        default=None,
        help="Analisa um único time na briga contra o rebaixamento (nome ou sigla, ex.: Internacional, GRE)"
    )
    parser.add_argument(
        "--incluir-pendentes-r34",
        action="store_true",
        help=argparse.SUPPRESS
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

    target_team: str = ""
    if args.analisar_time is not None:
        try:
            target_team = resolve_team_name(args.analisar_time, standings)
        except ValueError as e:
            print(f"Erro ao selecionar time: {e}", file=sys.stderr)
            sys.exit(1)

    # 2. Seleção dos confrontos pendentes a simular
    matches_to_simulate, rodadas_str = select_pending_matches(data)
    num_matches = len(matches_to_simulate)

    # 3. Carga do modelo e cálculo de probabilidades / matrizes de placares
    model = get_model(args.modelo)
    history = fetch_season_history(data, force_refresh=args.refresh) if model.requires_history else None
    enriched_matches = model.prepare_matches(
        standings,
        matches_to_simulate,
        history=history,
        fator_zebra=args.zebra,
    )
    modo_simulacao = "Placares reais (gols)" if model.uses_score_simulation else "Resultado 1X2"

    # Modo específico: análise de permanência (Z-4) para um único clube
    if target_team:
        t_sim_start = time.time()
        analysis = analyze_team_relegation(
            standings=standings,
            enriched_matches=enriched_matches,
            team_name=target_team,
            num_simulations=args.simulacoes,
            seed=args.seed,
        )
        t_sim_elapsed = time.time() - t_sim_start
        print_relegation_analysis(
            analysis=analysis,
            modelo_desc=model.description,
            modo_simulacao=modo_simulacao,
            fator_zebra=args.zebra,
            rodadas_str=rodadas_str,
        )
        print(f"⏱️ Tempo de processamento da análise: {t_sim_elapsed:.2f}s (Total: {time.time()-t0:.2f}s)")
        return

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
        fator_zebra=args.zebra,
        modelo_desc=model.description,
        modo_simulacao=modo_simulacao,
    )
    print_standings_table(stats_list)
    print_highlights(stats_list)

    if args.detalhado:
        print_detailed_matrix(stats_list)

    print(f"⏱️ Tempo de processamento da simulação: {t_sim_elapsed:.2f}s (Total: {time.time()-t0:.2f}s)")


if __name__ == "__main__":
    main()
