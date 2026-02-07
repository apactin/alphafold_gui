from __future__ import annotations
import pandas as pd
from pathlib import Path


def _runs_metrics_text(self, metrics_csv: Path) -> str:
        """
        Schema-aware metrics renderer for your metrics_summary.csv.

        Expected columns (example):
        job, af3_model, relaxed_model, relax_scorefile,
        ligand_best_model, ligand_scorefile,
        protein_RMSD, ligand_RMSD,
        relax_total_score_min, relax_fa_atr, ...
        ligand_total_score_min, ligand_fa_atr, ...

        Output:
        - File + shape
        - Paths (key files)
        - RMSDs
        - Relax scores (grouped)
        - Ligand scores (grouped)
        - If multiple rows: compact leaderboard table
        """
        try:


            df = pd.read_csv(
                metrics_csv,
                sep=None,           # auto-detect (tab or comma)
                engine="python",
                on_bad_lines="skip",
                comment="#",
            )

            if df is None or df.empty:
                return (
                    f"\n\n===== METRICS =====\n"
                    f"File: {metrics_csv}\n\n"
                    f"(CSV parsed but no rows)\n"
                )

            df.columns = [str(c).strip() for c in df.columns]

            def fmt(v) -> str:
                if v is None:
                    return "—"
                try:
                    if isinstance(v, float):
                        if math.isnan(v):
                            return "—"
                        return f"{v:.3f}".rstrip("0").rstrip(".")
                except Exception:
                    pass
                s = str(v)
                return s if s.strip() else "—"

            def render_kv(title: str, items: list[tuple[str, str]]) -> str:
                if not items:
                    return ""
                w = max(len(k) for k, _ in items)
                lines = [title]
                for k, v in items:
                    lines.append(f"  {k:<{w}}  {v}")
                return "\n".join(lines) + "\n"

            def take_if_exists(row, cols: list[str]) -> list[tuple[str, str]]:
                out = []
                for c in cols:
                    if c in df.columns:
                        out.append((c, fmt(row.get(c))))
                return out

            # pick score column for sorting/summary
            score_candidates = [
                "relax_total_score_min",
                "ligand_total_score_min",
                "total_score",
                "relax_score",
                "score",
            ]
            score_col = next((c for c in score_candidates if c in df.columns), None)

            # choose "best" row if multiple
            df_show = df
            if score_col:
                try:
                    df_show = df.sort_values(score_col, ascending=True)
                except Exception:
                    df_show = df

            best = df_show.iloc[0]

            header = (
                f"\n\n===== METRICS =====\n"
                f"File: {metrics_csv}\n"
                f"Rows: {df.shape[0]}   Cols: {df.shape[1]}\n"
            )

            # --- Paths / key artifacts ---
            path_cols = [
                "job",
                "af3_model",
                "relaxed_model",
                "relax_scorefile",
                "ligand_best_model",
                "ligand_scorefile",
            ]
            paths_block = render_kv("Artifacts", take_if_exists(best, path_cols))

            # --- RMSDs ---
            rmsd_block = render_kv(
                "RMSDs",
                take_if_exists(best, ["protein_RMSD", "ligand_RMSD"]),
            )

            # --- Relax score terms ---
            # Always show total first, then common terms if present, then the rest (capped)
            relax_cols = [c for c in df.columns if c.startswith("relax_")]
            relax_primary = [
                "relax_total_score_min",
                "relax_fa_atr",
                "relax_fa_rep",
                "relax_fa_sol",
                "relax_fa_elec",
                "relax_hbond_sc",
                "relax_hbond_bb_sc",
                "relax_dslf_fa13",
            ]
            relax_items = []
            for c in relax_primary:
                if c in relax_cols:
                    relax_items.append((c, fmt(best.get(c))))
            # add remaining relax_ cols (stable order), but keep the block readable
            for c in sorted(set(relax_cols) - set(relax_primary)):
                if len(relax_items) >= 18:
                    break
                relax_items.append((c, fmt(best.get(c))))
            relax_block = render_kv("Rosetta relax (best row)", relax_items)

            # --- Ligand score terms ---
            ligand_cols = [c for c in df.columns if c.startswith("ligand_")]
            ligand_primary = [
                "ligand_total_score_min",
                "ligand_fa_atr",
                "ligand_fa_rep",
                "ligand_fa_sol",
                "ligand_fa_elec",
                "ligand_hbond_sc",
                "ligand_hbond_bb_sc",
                "ligand_dslf_fa13",
            ]
            ligand_items = []
            for c in ligand_primary:
                if c in ligand_cols:
                    ligand_items.append((c, fmt(best.get(c))))
            for c in sorted(set(ligand_cols) - set(ligand_primary)):
                if len(ligand_items) >= 18:
                    break
                ligand_items.append((c, fmt(best.get(c))))
            ligand_block = render_kv("Rosetta ligand (best row)", ligand_items)

            # --- If multiple rows, add a compact leaderboard table ---
            table_block = ""
            if df.shape[0] > 1:
                # compact columns for leaderboard
                table_cols = [c for c in ["job", "seed", "sample", score_col, "protein_RMSD", "ligand_RMSD"] if c and c in df.columns]
                d = df_show[table_cols].head(12).copy()

                # format cells as strings
                for c in table_cols:
                    d[c] = d[c].map(fmt)

                # column widths
                widths = {c: min(max(len(c), d[c].astype(str).map(len).max()), 24) for c in table_cols}
                hdr = "  ".join(f"{c:<{widths[c]}}" for c in table_cols)
                sep = "  ".join("-" * widths[c] for c in table_cols)
                rows = []
                for _, r in d.iterrows():
                    rows.append("  ".join(f"{str(r[c]):<{widths[c]}}" for c in table_cols))

                table_block = "Top rows\n" + hdr + "\n" + sep + "\n" + "\n".join(rows) + "\n"

            # Assemble output
            out = header + "\n"
            out += paths_block + "\n" if paths_block else ""
            out += rmsd_block + "\n" if rmsd_block else ""
            out += relax_block + "\n" if relax_block else ""
            out += ligand_block + "\n" if ligand_block else ""
            out += table_block

            return out

        except Exception as e:
            return f"\n\n===== METRICS =====\nFailed to read {metrics_csv}:\n{e}\n"
        
def _runs_metrics_html(self, metrics_csv: Path) -> str:
    def fmt(v) -> str:
        if v is None:
            return "—"
        try:
            if isinstance(v, float):
                if math.isnan(v):
                    return "—"
                return f"{v:.3f}".rstrip("0").rstrip(".")
        except Exception:
            pass
        s = str(v)
        return s if s.strip() else "—"

    def esc(s: str) -> str:
        # Local import to avoid making this module depend on Qt
        import html
        return html.escape(str(s))

    def kv_table(title: str, items: list[tuple[str, str]]) -> str:
        if not items:
            return ""
        rows = "\n".join(
            "<tr>"
            f"<td style='padding:3px 10px 3px 0; color:#666; white-space:nowrap;'><b>{esc(k)}</b></td>"
            f"<td style='padding:3px 0; word-break:break-all;'>{esc(v)}</td>"
            "</tr>"
            for k, v in items
        )
        return f"""
        <div style="margin-top:10px;"><b>{esc(title)}</b></div>
        <table style="border-collapse:collapse; width:100%;">
            {rows}
        </table>
        """.strip()

    def terms_table(title: str, items: list[tuple[str, str]]) -> str:
        if not items:
            return ""
        rows = "\n".join(
            "<tr>"
            f"<td style='padding:2px 10px 2px 0; color:#444; white-space:nowrap;'>{esc(k)}</td>"
            f"<td style='padding:2px 0; text-align:right; font-variant-numeric:tabular-nums;'>{esc(v)}</td>"
            "</tr>"
            for k, v in items
        )
        return f"""
        <div style="margin-top:10px;"><b>{esc(title)}</b></div>
        <table style="border-collapse:collapse;">
            {rows}
        </table>
        """.strip()

    def leaderboard_table(title: str, df_small: pd.DataFrame) -> str:
        if df_small is None or df_small.empty:
            return ""
        cols = list(df_small.columns)
        th = "".join(f"<th style='text-align:left; padding:4px 8px; border-bottom:1px solid rgba(0,0,0,0.12);'>{esc(c)}</th>" for c in cols)
        trs = []
        for _, r in df_small.iterrows():
            tds = []
            for c in cols:
                val = str(r[c])
                # right-align numeric-ish columns
                align = "right" if c.lower().endswith(("score", "rmsd")) or c.lower() in {"score", "total"} else "left"
                tds.append(f"<td style='padding:4px 8px; border-bottom:1px solid rgba(0,0,0,0.06); text-align:{align}; font-variant-numeric:tabular-nums;'>{esc(val)}</td>")
            trs.append("<tr>" + "".join(tds) + "</tr>")
        body = "\n".join(trs)
        return f"""
        <div style="margin-top:10px;"><b>{esc(title)}</b></div>
        <table style="border-collapse:collapse; width:100%;">
            <thead><tr>{th}</tr></thead>
            <tbody>{body}</tbody>
        </table>
        """.strip()

    try:
        df = pd.read_csv(
            metrics_csv,
            sep=None,
            engine="python",
            on_bad_lines="skip",
            comment="#",
        )

        if df is None or df.empty:
            return f"""
            <div style="margin-top:8px;">
              <b>Metrics</b><br>
              <span style="color:#666;">File:</span> {esc(metrics_csv)}<br>
              <i>(CSV parsed but no rows)</i>
            </div>
            """.strip()

        df.columns = [str(c).strip() for c in df.columns]

        # pick score column for sorting
        score_candidates = [
            "relax_total_score_min",
            "ligand_total_score_min",
            "total_score",
            "relax_score",
            "score",
        ]
        score_col = next((c for c in score_candidates if c in df.columns), None)

        df_show = df
        if score_col:
            try:
                df_show = df.sort_values(score_col, ascending=True)
            except Exception:
                df_show = df

        best = df_show.iloc[0]

        # Header
        header = f"""
        <div style="margin-top:6px; color:#666;">
        <span>File:</span> {esc(metrics_csv)}<br>
        <span>Rows:</span> {df.shape[0]} &nbsp;&nbsp; <span>Cols:</span> {df.shape[1]}
        </div>
        """.strip()

        # Artifacts (paths)
        path_cols = ["job", "af3_model", "relaxed_model", "relax_scorefile", "ligand_best_model", "ligand_scorefile"]
        artifacts = [(c, fmt(best.get(c))) for c in path_cols if c in df.columns]

        # RMSDs
        rmsd_cols = ["protein_RMSD", "ligand_RMSD"]
        rmsds = [(c, fmt(best.get(c))) for c in rmsd_cols if c in df.columns]

        # Score terms
        relax_cols = [c for c in df.columns if c.startswith("relax_") and c not in {"relax_scorefile"}]
        ligand_cols = [c for c in df.columns if c.startswith("ligand_") and c not in {"ligand_best_model", "ligand_scorefile", "ligand_RMSD"}]

        relax_primary = [
            "relax_total_score_min", "relax_fa_atr", "relax_fa_rep", "relax_fa_sol",
            "relax_fa_elec", "relax_hbond_sc", "relax_hbond_bb_sc", "relax_dslf_fa13",
        ]
        ligand_primary = [
            "ligand_total_score_min", "ligand_fa_atr", "ligand_fa_rep", "ligand_fa_sol",
            "ligand_fa_elec", "ligand_hbond_sc", "ligand_hbond_bb_sc", "ligand_dslf_fa13",
        ]

        relax_items: list[tuple[str, str]] = []
        for c in relax_primary:
            if c in relax_cols:
                relax_items.append((c, fmt(best.get(c))))
        for c in sorted(set(relax_cols) - set(relax_primary)):
            if len(relax_items) >= 18:
                break
            relax_items.append((c, fmt(best.get(c))))

        ligand_items: list[tuple[str, str]] = []
        for c in ligand_primary:
            if c in ligand_cols:
                ligand_items.append((c, fmt(best.get(c))))
        for c in sorted(set(ligand_cols) - set(ligand_primary)):
            if len(ligand_items) >= 18:
                break
            ligand_items.append((c, fmt(best.get(c))))

        # Leaderboard if multiple rows
        leaderboard = ""
        if df.shape[0] > 1:
            cols = [c for c in ["job", "seed", "sample", score_col, "protein_RMSD", "ligand_RMSD"] if c and c in df.columns]
            d = df_show[cols].head(12).copy()
            for c in cols:
                d[c] = d[c].map(fmt)
            leaderboard = leaderboard_table("Top rows", d)

        # Assemble
        parts = [
            header,
            kv_table("Artifacts", artifacts),
            terms_table("RMSDs", rmsds),
            terms_table("Rosetta relax (best row)", relax_items),
            terms_table("Rosetta ligand (best row)", ligand_items),
            leaderboard,
        ]
        return "\n".join([p for p in parts if p])

    except Exception as e:
        return f"""
        <div style="margin-top:8px;">
          <b>Metrics</b><br>
          <span style="color:#666;">File:</span> {esc(metrics_csv)}<br>
          <span style="color:#b00;"><b>Failed to read:</b></span> {esc(e)}
        </div>
        """.strip()