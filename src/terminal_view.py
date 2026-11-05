"""
Visualizador de relatórios formatados para o terminal.
Apresenta as projeções probabilísticas, faixas de incerteza (μ ± 1σ) e
chances de Título, G-4, G-6, Sul-Americana e Rebaixamento Z-4.
"""

from typing import Dict, Any, List


def format_pct(val: float) -> str:
    if val >= 99.95:
        return "100.0%"
    elif val < 0.05 and val > 0:
        return "<0.1%"
    elif val == 0:
        return " 0.0%"
    else:
        return f"{val:5.1f}%"


def print_simulation_header(
    num_simulations: int,
    num_matches: int,
    rodadas_str: str,
    fator_zebra: float
):
    print("=" * 112)
    print("      SIMULAÇÃO MONTE CARLO DO CAMPEONATO BRASILEIRO SÉRIE A — 2026")
    print("   Metodologia Estatística: Prof. Rogemar Riffel (IF-UFRGS) | Dados: Globo Esporte")
    print("=" * 112)
    print(
        f"• Modelo: Peso Composto 50/50 (Histórico Temporada + Forma Recente nas últimas 5 rodadas)\n"
        f"• Amostragem: {num_simulations:,} simulações de Monte Carlo | Fator de Zebra: {fator_zebra*100:.0f}%\n"
        f"• Cenário: {num_matches} jogos restantes ({rodadas_str}) | Incerteza indicada: ±1 desvio-padrão (1σ)"
    )
    print("=" * 112)


def print_standings_table(stats_list: List[Dict[str, Any]]):
    print(
        f"{'Pos':<4} {'Time':<16} {'Pts':>3} {'V':>3} {'SG':>4} | "
        f"{'Projeção (±1σ)':^15} {'Faixa':^11} | "
        f"{'Título':>7} {'G-4':>7} {'G-6':>7} {'Sula':>7} {'Z-4':>7}"
    )
    print("-" * 112)

    for row in stats_list:
        pos = f"{row['pos_atual']}º"
        time_nome = row["nome"]
        pts = row["pontos_atuais"]
        v = row["vitorias_atuais"]
        sg = f"{row['saldo_atual']:+d}"

        proj = f"{row['proj_media']:4.1f} ± {row['proj_std']:3.1f}"
        faixa = f"[{row['proj_min']:2d} - {row['proj_max']:2d}]"

        p_tit = format_pct(row["titulo_pct"])
        p_g4 = format_pct(row["g4_pct"])
        p_g6 = format_pct(row["g6_pct"])
        p_sula = format_pct(row["sula_pct"])
        p_z4 = format_pct(row["z4_pct"])

        # Destacar visualmente zonas da tabela com marcadores sutis
        p_num = row["pos_atual"]
        if p_num == 4:
            divisor = " [Fim G-4]"
        elif p_num == 6:
            divisor = " [Fim G-6]"
        elif p_num == 12:
            divisor = " [Fim G-12 / Sula]"
        elif p_num == 16:
            divisor = " [Início Z-4 / Rebaixamento]"
        else:
            divisor = ""

        print(
            f"{pos:<4} {time_nome:<16} {pts:>3} {v:>3} {sg:>4} | "
            f"{proj:^15} {faixa:^11} | "
            f"{p_tit:>7} {p_g4:>7} {p_g6:>7} {p_sula:>7} {p_z4:>7}{divisor}"
        )

    print("=" * 112)


def print_highlights(stats_list: List[Dict[str, Any]]):
    # Candidatos ao Título
    candidatos_titulo = [t for t in stats_list if t["titulo_pct"] > 0.05]
    candidatos_titulo.sort(key=lambda x: -x["titulo_pct"])

    # Disputa do G-4
    disputa_g4 = [t for t in stats_list if 1.0 < t["g4_pct"] < 99.0]
    disputa_g4.sort(key=lambda x: -x["g4_pct"])

    # Risco de Rebaixamento (Z-4)
    risco_z4 = [t for t in stats_list if t["z4_pct"] > 0.05]
    risco_z4.sort(key=lambda x: -x["z4_pct"])

    print("\n📌 DESTAQUES DA SIMULAÇÃO:")

    print("\n🏆 DISPUTA PELO TÍTULO:")
    for t in candidatos_titulo:
        print(f"   • {t['nome']:<15} {format_pct(t['titulo_pct']):>7} de chance (Projeção: {t['proj_media']:.1f} pts)")

    if disputa_g4:
        print("\n⭐ BRIGA PELA LIBERTADORES (G-4):")
        for t in disputa_g4:
            print(f"   • {t['nome']:<15} {format_pct(t['g4_pct']):>7} de chance de G-4 (G-6: {format_pct(t['g6_pct'])})")

    if risco_z4:
        print("\n⚠️ RISCO DE REBAIXAMENTO (Z-4):")
        for t in risco_z4:
            print(f"   • {t['nome']:<15} {format_pct(t['z4_pct']):>7} de risco de Série B (Projeção: {t['proj_media']:.1f} pts)")

    print()


def print_detailed_matrix(stats_list: List[Dict[str, Any]]):
    """
    Exibe a matriz completa de probabilidades de 1º ao 20º para cada clube.
    """
    print("\n" + "=" * 112)
    print("      DISTRIBUIÇÃO DETALHADA DE PROBABILIDADES POR POSIÇÃO FINAL (1º AO 20º)")
    print("=" * 112)

    # Cabeçalho das 20 posições
    header = f"{'Time':<16} |"
    for p in range(1, 21):
        header += f" {p:2d}º"
    print(header)
    print("-" * 112)

    for row in stats_list:
        line = f"{row['nome']:<16} |"
        for p_val in row["posicoes_prob"]:
            if p_val >= 99.5:
                line += " 99%"
            elif p_val >= 0.5:
                line += f" {int(round(p_val)):2d}%"
            elif p_val > 0.01:
                line += "  · "
            else:
                line += "    "
        print(line)

    print("=" * 112)
    print("Legenda: ' · ' indica probabilidade menor que 0.5% mas matematicamente possível.\n")
