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
    fator_zebra: float,
    modelo_desc: str = "Peso Composto 50/50 (Histórico Temporada + Forma Recente nas últimas 5 rodadas)",
    modo_simulacao: str = "Resultado 1X2",
):
    print("=" * 112)
    print("      SIMULAÇÃO MONTE CARLO DO CAMPEONATO BRASILEIRO SÉRIE A — 2026")
    print("   Metodologia Estatística: Prof. Rogemar Riffel (IF-UFRGS) / Modelos Plugáveis | Dados: Globo Esporte")
    print("=" * 112)
    print(
        f"• Modelo: {modelo_desc} | Modo: {modo_simulacao}\n"
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


def _status_label(status: str) -> str:
    labels = {
        "ja_salvo": "✅ Já garantido na Série A (0% de risco de rebaixamento)",
        "depende_so_de_si": "💪 Depende só de si para escapar do Z-4",
        "depende_de_rivais": "🙏 Não depende só de si (precisa de combinação + tropeços de rivais)",
        "ja_rebaixado": "❌ Rebaixamento matemático (0% de chance mesmo vencendo todos os jogos)",
    }
    return labels.get(status, status)


def _eval_combo_label(prob_escapar_pct: float) -> str:
    if prob_escapar_pct >= 99.95:
        return "Salvo (100%)"
    if prob_escapar_pct >= 75.0:
        return "Alta chance"
    if prob_escapar_pct >= 50.0:
        return "Favorável"
    if prob_escapar_pct >= 15.0:
        return "Arriscado"
    if prob_escapar_pct > 0.05:
        return "Remoto"
    return "Rebaixado"


def print_relegation_analysis(
    analysis: Dict[str, Any],
    modelo_desc: str,
    modo_simulacao: str,
    fator_zebra: float,
    rodadas_str: str,
):
    team = analysis["team"]
    nome = team["nome"]
    sigla = f" ({team['sigla']})" if team.get("sigla") else ""

    print("=" * 112)
    print(f"      ANÁLISE DE PERMANÊNCIA (BRIGA CONTRA O Z-4) — {nome.upper()}{sigla}")
    print(f"   Metodologia: {modelo_desc} | Modo: {modo_simulacao}")
    print("=" * 112)
    print(
        f"• Situação Atual: {team['pos_atual']}º lugar | {team['pontos_atuais']} pts | "
        f"{team['vitorias_atuais']}V | SG {team['saldo_atual']:+d} | "
        f"{team['jogos_atuais']}J ({team['jogos_restantes']} jogos restantes; máx. {team['max_pontos_possiveis']} pts)\n"
        f"• Cenário: {rodadas_str} | Amostragem: {analysis['num_simulations']:,} simulações/cenário | Zebra: {fator_zebra*100:.0f}%\n"
        f"• Probabilidade Base: Escapar {format_pct(analysis['base_survival_pct']).strip()} | "
        f"Risco Z-4 {format_pct(analysis['base_relegation_pct']).strip()}\n"
        f"• Diagnóstico: {_status_label(analysis['status'])}"
    )
    print("=" * 112)

    # Jogos restantes do time
    team_matches = analysis["team_matches"]
    print(f"\n📅 JOGOS RESTANTES DO {nome.upper()}:")
    if not team_matches:
        print("   • Nenhum jogo restante na competição.")
    else:
        print(f"   {'Rod':<4} {'Mando':<6} {'Adversário':<26} | {'Vitória':>9} {'Empate':>9} {'Derrota':>9}")
        print("   " + "-" * 78)
        for m in team_matches:
            tag = " [Confronto direto]" if m["confronto_direto"] else ""
            adv_str = f"{m['adversario']} ({m['adversario_pos']}º){tag}"
            print(
                f"   R{m['rodada']:<3} {m['mando']:<6} {adv_str:<26} | "
                f"{format_pct(m['p_vitoria']):>9} {format_pct(m['p_empate']):>9} {format_pct(m['p_derrota']):>9}"
            )

    # Combinações enumeradas
    combos = analysis["combinations"]
    if analysis["enumerated"] and combos:
        max_show = 27
        shown_combos = combos[:max_show]
        omitted = len(combos) - len(shown_combos)

        print(f"\n🎯 COMBINAÇÕES DE RESULTADOS ({len(combos)} cenários possíveis):")
        print(f"   {'Cenário':<12} {'Resultados':<34} {'Pts':>4} | {'P(Cenário)':>10} {'P(Escapar)':>10}  {'Avaliação'}")
        print("   " + "-" * 96)
        for c in shown_combos:
            print(
                f"   {c['resumo']:<12} {c['resultados_str']:<34} {c['pontos_finais']:>4} | "
                f"{format_pct(c['prob_cenario_pct']):>10} {format_pct(c['prob_escapar_pct']):>10}  "
                f"{_eval_combo_label(c['prob_escapar_pct'])}"
            )
        if omitted > 0:
            zero_omitted = sum(1 for c in combos[max_show:] if c["prob_escapar_pct"] <= 0.05)
            print(f"   ... ({omitted} combinações menos favoráveis omitidas; {zero_omitted} com ~0.0% de chance)")

    # Meta por pontuação final
    pts_summary = analysis["points_summary"]
    if pts_summary:
        print("\n📊 META POR PONTUAÇÃO FINAL:")
        print(f"   {'Pts Finais':>10} {'(+Pts)':>7} | {'P(Atingir)':>10} {'P(Escapar)':>11} | {'Combinações'}")
        print("   " + "-" * 78)
        for p in pts_summary:
            gain_str = f"+{p['pontos_ganhos']}"
            print(
                f"   {p['pontos_finais']:>10} {gain_str:>7} | "
                f"{format_pct(p['prob_atingir_pct']):>10} {format_pct(p['prob_escapar_pct']):>11} | "
                f"{p['exemplos']}"
            )

    # O que o time deve buscar
    rec = analysis["recommended_target"]
    print(f"\n🧭 O QUE O {nome.upper()} DEVE BUSCAR:")
    if analysis["status"] == "ja_salvo":
        print("   • O clube já atingiu pontuação segura e não corre risco de rebaixamento.")
    elif analysis["status"] == "ja_rebaixado":
        print(
            f"   • Mesmo atingindo o máximo de {team['max_pontos_possiveis']} pts, "
            f"não alcança o 16º colocado considerando os confrontos restantes."
        )
    else:
        if rec["safe_points"] is not None:
            print(f"   • Meta segura (≥99.9%): chegar a {rec['safe_points']} pts.")
        if rec["easiest_safe_combo"] is not None:
            c = rec["easiest_safe_combo"]
            print(f"   • Combinação mais acessível para garantir 100%: {c['resumo']} -> {c['resultados_str']} ({c['pontos_finais']} pts).")
        elif rec["easiest_likely_combo"] is not None:
            c = rec["easiest_likely_combo"]
            print(
                f"   • Meta realista (≥50% de chance): {rec['likely_points']} pts — ex.: "
                f"{c['resumo']} ({c['resultados_str']}, {format_pct(c['prob_escapar_pct']).strip()} de escapar)."
            )
        elif rec["best_combo"] is not None:
            c = rec["best_combo"]
            print(
                f"   • Melhor cenário possível: {c['resumo']} ({c['resultados_str']}, {c['pontos_finais']} pts) -> "
                f"{format_pct(c['prob_escapar_pct']).strip()} de chance de escapar (depende de tropeços de rivais)."
            )

    # Jogos-chave de rivais ("para secar")
    rivals = analysis["key_rival_matches"]
    if rivals:
        print(f"\n👀 JOGOS DE RIVAIS MAIS DECISIVOS (condicionado a: {analysis['rival_conditioned_on']}):")
        print(f"   {'Rod':<4} {'Confronto':<36} {'Resultado Ideal':<26} {'Impacto':>8} {'Escapa c/ Ideal'}")
        print("   " + "-" * 96)
        for rv in rivals:
            confronto = f"{rv['mandante']} ({rv['mandante_pos']}º) x {rv['visitante']} ({rv['visitante_pos']}º)"
            impacto_str = f"+{rv['impacto_pp']:.1f} pp"
            print(
                f"   R{rv['rodada']:<3} {confronto:<36} {rv['melhor_resultado']:<26} "
                f"{impacto_str:>8} {format_pct(rv['prob_melhor_pct']):>14}"
            )

    print("=" * 112)
