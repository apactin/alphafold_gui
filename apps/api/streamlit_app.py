#!/usr/bin/env python3
"""
streamlit_app.py — AF3 Web GUI (drop-in)
=======================================

This version is a clean drop-in replacement for your current Streamlit app, updated to match
the corrected FastAPI server.py I gave you (single endpoints, runs selection, metrics, downloads,
ligand preview + pdb download, and no request timeouts on long jobs).

Key fixes vs your current app:
- ✅ Uses the *new* server endpoints:
  - GET  /sequences          -> returns {"protein":[...], "dna":[...], "rna":[...]}
  - GET  /sequences/{kind}/{name}
  - GET  /ligands            -> returns list[{"name","smiles","hash"}]
  - GET  /ligands/{name}/preview, GET /ligands/{name}/pdb
  - GET  /runs               -> returns list[{"id","jobname","finished_at","record"}]
  - GET  /runs/{id}/metrics_html
  - GET  /runs/{id}/download?kind=job_folder|af_model|rosetta_model
  - POST /runs/{id}/run_rosetta?multi_seed=...
- ✅ Removes the old broken assumption that /sequences returns {"sequences": {...}}
- ✅ Autopopulates protein name/template/sequence and DNA/RNA sequence by fetching /sequences/{kind}/{name}
- ✅ Fixes remove_entry bug (was popping wrong key)
- ✅ Fixes timeout: /run is called with a very large timeout; status polling uses smaller timeout.
- ✅ Runs tab:
  - shows only jobname + finished_at
  - lets you select a run
  - shows metrics (HTML)
  - provides "Run Rosetta" button + multi_seed checkbox
  - downloads: job folder zip / AF model / Rosetta model
- ✅ Ligands tab:
  - displays the LIG.svg from the ligand hash cache
  - provides download link for LIG.pdb

Assumption:
- You updated server.py to the "drop-in" version I gave you in the previous message.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import pandas as pd

# ----------------------------
# API helpers
# ----------------------------
API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")

# Default request timeouts
TIMEOUT_FAST = 30
TIMEOUT_MED = 60
# /run is synchronous on the backend, so this must be long (or you must later convert backend to async/background)
TIMEOUT_RUN = 60 * 60 * 12  # 12 hours


def api_get(path: str, *, timeout: int = TIMEOUT_MED, **params) -> Any:
    r = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(path: str, *, json_body=None, timeout: int = TIMEOUT_MED, **params) -> Any:
    r = requests.post(f"{API_BASE}{path}", params=params, json=json_body, timeout=timeout)
    r.raise_for_status()
    # could be empty body; but your API returns json
    return r.json()


def api_delete(path: str, *, timeout: int = TIMEOUT_MED, **params) -> Any:
    r = requests.delete(f"{API_BASE}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_file_url(path: str) -> str:
    return f"{API_BASE}{path}"


# ----------------------------
# UI constants
# ----------------------------
st.set_page_config(page_title="AF3 Web GUI", layout="wide")

PTM_CHOICES = {
    "None": None,
    "Phosphoserine (pSer)": "SEP",
    "Phosphothreonine (pThr)": "TPO",
    "Phosphotyrosine (pTyr)": "PTR",
    "Phosphohistidine": "HIP",
    "Phosphocysteine": "CSP",
    "Monomethyllysine": "MLY",
    "Dimethyllysine": "M2L",
    "Trimethyllysine": "M3L",
    "Monomethylarginine": "MMA",
    "Dimethylarginine": "DMA",
    "N6-Acetyllysine": "ALY",
    "N-terminal acetylation": "ACE",
    "O-Acetylserine": "ASE",
    "Hydroxyproline": "HYP",
    "Hydroxylysine": "HYL",
    "Hydroxycysteine": "CSO",
    "Methionine sulfoxide": "MSO",
    "Cysteine sulfinic acid": "SFA",
    "Cysteine sulfonic acid": "CSA",
    "Sulfotyrosine": "TYS",
    "Pyroglutamate": "PCA",
    "Carboxymethyllysine": "CML",
    "Dehydroalanine": "DHA",
    "Selenomethionine": "MSE",
    "N-Formylmethionine": "FME",
    "N-Acetylglucosamine": "NAG",
    "Mannose": "MAN",
    "Galactose": "GAL",
    "Fucose": "FUC",
    "Sialic acid": "SIA",
    "Myristoylation": "MYS",
    "Palmitoylation": "PAM",
    "Farnesylation": "FAR",
    "Geranylgeranylation": "GGG",
}

ATOM_MAP = {
    "Cysteine (SG)": "SG",
    "Lysine (NZ)": "NZ",
    "Histidine (ND1)": "ND1",
    "Histidine (NE2)": "NE2",
    "Tyrosine (OH)": "OH",
    "Custom…": "__CUSTOM__",
}

ION_CHOICES = ["MG", "CA", "ZN", "NA", "K", "CL", "MN", "FE", "CO", "CU"]
COFACTOR_CHOICES = ["ATP", "ADP", "AMP", "NAD", "NADP", "FAD", "CoA", "SAM", "GTP", "GDP"]


# ----------------------------
# Session state
# ----------------------------
def init_state():
    st.session_state.setdefault("proteins", [])
    st.session_state.setdefault("dna", [])
    st.session_state.setdefault("rna", [])
    st.session_state.setdefault("task_id", "")
    st.session_state.setdefault("selected_run_id", None)  # int
    st.session_state.setdefault("runs_refresh_nonce", 0)


def add_entry(kind: str):
    if kind == "protein":
        st.session_state["proteins"].append({"name": "", "sequence": "", "template": "", "ptms": []})
    elif kind == "dna":
        st.session_state["dna"].append({"name": "", "sequence": "", "ptms": []})
    elif kind == "rna":
        st.session_state["rna"].append({"name": "", "sequence": "", "ptms": []})


def remove_entry(kind: str, idx: int):
    # kind must be one of: proteins/dna/rna (note: proteins plural in session_state)
    if kind in st.session_state and 0 <= idx < len(st.session_state[kind]):
        st.session_state[kind].pop(idx)


# ----------------------------
# PTM editor
# ----------------------------
def ptm_editor(ptms: list[dict], key_prefix: str) -> list[dict]:
    """
    ptms = [{"ccd":"SEP","pos":"12","label":"Phosphoserine (pSer)"}...]
    Returns cleaned list: [{"label":..., "ccd":..., "pos":...}, ...]
    """
    st.caption("PTMs")
    if st.button("➕ Add PTM", key=f"{key_prefix}_add_ptm"):
        ptms.append({"label": "None", "ccd": None, "pos": ""})

    to_delete = None
    keys = list(PTM_CHOICES.keys())

    for i, p in enumerate(ptms):
        cols = st.columns([3, 2, 1])
        with cols[0]:
            cur = p.get("label", "None")
            idx = keys.index(cur) if cur in PTM_CHOICES else 0
            label = st.selectbox(
                "Type",
                keys,
                index=max(0, idx),
                key=f"{key_prefix}_ptm_type_{i}",
                label_visibility="collapsed",
            )
        with cols[1]:
            pos = st.text_input(
                "Pos",
                value=str(p.get("pos", "")),
                key=f"{key_prefix}_ptm_pos_{i}",
                label_visibility="collapsed",
            )
        with cols[2]:
            if st.button("🗑", key=f"{key_prefix}_ptm_del_{i}"):
                to_delete = i

        ccd = PTM_CHOICES.get(label)
        p["label"] = label
        p["ccd"] = ccd
        p["pos"] = pos

    if to_delete is not None and 0 <= to_delete < len(ptms):
        ptms.pop(to_delete)

    cleaned = []
    for p in ptms:
        ccd = p.get("ccd")
        pos = str(p.get("pos", "")).strip()
        if ccd and pos:
            cleaned.append({"label": p.get("label", ""), "ccd": str(ccd), "pos": pos})
    return cleaned


# ----------------------------
# JobSpec builder
# ----------------------------
def build_job_spec(
    *,
    jobname: str,
    ligand_smiles: str,
    covalent: bool,
    chain: str,
    residue: str,
    prot_atom_label: str,
    custom_atom: str,
    lig_atom: str,
    ions: list[str],
    cofactors: list[str],
    seeds: str,
    skip_rosetta: bool,
    multi_seed: bool,
) -> dict:
    mapped = ATOM_MAP.get(prot_atom_label, "")
    prot_atom = custom_atom.strip() if mapped == "__CUSTOM__" else mapped

    seed_list = None
    if seeds.strip():
        seed_list = [int(x.strip()) for x in seeds.split(",") if x.strip().isdigit()]

    ligand = {
        "smiles": (ligand_smiles or "").strip(),
        "covalent": bool(covalent),
        "chain": chain.strip(),
        "residue": residue.strip(),
        "prot_atom": prot_atom.strip(),
        "ligand_atom": lig_atom.strip(),
        "ions": ",".join([x.strip() for x in ions]) if isinstance(ions, list) else str(ions),
        "cofactors": ",".join([x.strip() for x in cofactors]) if isinstance(cofactors, list) else str(cofactors),
    }
    if seed_list:
        ligand["modelSeeds"] = seed_list

    proteins = []
    for p in st.session_state["proteins"]:
        seq = (p.get("sequence") or "").strip()
        if not seq:
            continue
        ptms = p.get("ptms_cleaned", [])
        proteins.append(
            {
                "name": (p.get("name") or "").strip(),
                "sequence": seq,
                "template": (p.get("template") or "").strip(),
                "ptms": ptms,
                # legacy compatibility fields:
                "modification": (ptms[0]["ccd"] if ptms else "None"),
                "mod_position": (ptms[0]["pos"] if ptms else ""),
            }
        )

    dna_list = []
    for d in st.session_state["dna"]:
        seq = (d.get("sequence") or "").strip()
        if not seq:
            continue
        ptms = d.get("ptms_cleaned", [])
        dna_list.append(
            {
                "sequence": seq,
                "ptms": ptms,
                "modification": (ptms[0]["ccd"] if ptms else "None"),
                "pos": (ptms[0]["pos"] if ptms else ""),
            }
        )

    rna_list = []
    for r in st.session_state["rna"]:
        seq = (r.get("sequence") or "").strip()
        if not seq:
            continue
        ptms = r.get("ptms_cleaned", [])
        rna_list.append(
            {
                "sequence": seq,
                "ptms": ptms,
                "modification": (ptms[0]["ccd"] if ptms else "None"),
                "pos": (ptms[0]["pos"] if ptms else ""),
            }
        )

    return {
        "jobname": jobname.strip().replace("+", "_"),
        "proteins": proteins,
        "dna": dna_list,
        "rna": rna_list,
        "ligand": ligand,
        "skip_rosetta": bool(skip_rosetta),
        "multi_seed": bool(multi_seed),
    }


# ----------------------------
# Page: Alphafold job form
# ----------------------------
def job_spec_form():
    init_state()

    # NEW server behavior:
    # GET /sequences -> {"protein":[...], "dna":[...], "rna":[...]}
    # GET /ligands   -> [{"name","smiles","hash"}, ...]
    seq_groups = api_get("/sequences", timeout=TIMEOUT_FAST)
    lig_rows = api_get("/ligands", timeout=TIMEOUT_FAST)

    protein_saved = [""] + (seq_groups.get("protein") or [])
    dna_saved = [""] + (seq_groups.get("dna") or [])
    rna_saved = [""] + (seq_groups.get("rna") or [])

    lig_names = [""] + [r.get("name", "") for r in (lig_rows or []) if isinstance(r, dict)]
    lig_by_name = {r.get("name", ""): r for r in (lig_rows or []) if isinstance(r, dict)}

    st.subheader("Alphafold job")

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        jobname = st.text_input("Job name", key="job_jobname")
        skip_rosetta = st.checkbox("Skip Rosetta", value=False, key="job_skip_rosetta")
        multi_seed = st.checkbox("Multi seed", value=False, key="job_multi_seed")
        seeds = st.text_input("Model seeds (comma-separated)", key="job_seeds")

    with col2:
        lig_pick = st.selectbox("Ligand", lig_names, key="job_lig_pick")
        ligand_smiles = lig_by_name.get(lig_pick, {}).get("smiles", "") if lig_pick else ""
        st.text_input("Ligand SMILES (auto from saved)", value=ligand_smiles, disabled=True)

    with col3:
        covalent = st.checkbox("Covalent", value=False, key="job_covalent")
        chain = st.text_input("Chain", value="", key="job_chain")
        residue = st.text_input("Residue", value="", key="job_residue")
        prot_atom_label = st.selectbox("Protein atom", list(ATOM_MAP.keys()), index=0, key="job_prot_atom")
        custom_atom = ""
        if ATOM_MAP.get(prot_atom_label) == "__CUSTOM__":
            custom_atom = st.text_input("Custom protein atom (e.g. OG1)", key="job_custom_atom")
        lig_atom = st.text_input("Ligand atom (e.g. C7)", value="", key="job_lig_atom")
        ions = st.multiselect("Ions", ION_CHOICES, default=[], key="job_ions")
        cofactors = st.multiselect("Cofactors", COFACTOR_CHOICES, default=[], key="job_cofactors")

    st.divider()

    # ----------------------------
    # Proteins
    # ----------------------------
    st.markdown("### Proteins")
    if st.button("➕ Add protein", key="add_protein"):
        add_entry("protein")
        st.rerun()

    for i, p in enumerate(st.session_state["proteins"]):
        with st.expander(f"Protein {i+1}", expanded=True):
            row = st.columns([2, 2, 1])

            with row[0]:
                pick = st.selectbox("Load saved", protein_saved, key=f"prot_pick_{i}")
                if pick:
                    # ✅ fetch authoritative entry from server
                    ent = api_get(f"/sequences/protein/{pick}", timeout=TIMEOUT_FAST)
                    p["name"] = ent.get("name", pick)
                    p["sequence"] = ent.get("sequence", "")
                    p["template"] = ent.get("template", "")

            with row[1]:
                p["name"] = st.text_input("Name", value=p.get("name", ""), key=f"prot_name_{i}")
                p["template"] = st.text_input("Template", value=p.get("template", ""), key=f"prot_tmpl_{i}")

            with row[2]:
                if st.button("Remove", key=f"prot_rm_{i}"):
                    remove_entry("proteins", i)
                    st.rerun()

            p["sequence"] = st.text_area("Sequence", value=p.get("sequence", ""), height=140, key=f"prot_seq_{i}")

            p.setdefault("ptms", [])
            p["ptms_cleaned"] = ptm_editor(p["ptms"], key_prefix=f"prot_{i}")

    # ----------------------------
    # DNA
    # ----------------------------
    st.markdown("### DNA")
    if st.button("➕ Add DNA", key="add_dna"):
        add_entry("dna")
        st.rerun()

    for i, d in enumerate(st.session_state["dna"]):
        with st.expander(f"DNA {i+1}", expanded=False):
            pick = st.selectbox("Load saved", dna_saved, key=f"dna_pick_{i}")
            if pick:
                ent = api_get(f"/sequences/dna/{pick}", timeout=TIMEOUT_FAST)
                d["name"] = ent.get("name", pick)
                d["sequence"] = ent.get("sequence", "")

            d["sequence"] = st.text_area("Sequence", value=d.get("sequence", ""), height=100, key=f"dna_seq_{i}")
            d.setdefault("ptms", [])
            d["ptms_cleaned"] = ptm_editor(d["ptms"], key_prefix=f"dna_{i}")

            if st.button("Remove DNA", key=f"dna_rm_{i}"):
                remove_entry("dna", i)
                st.rerun()

    # ----------------------------
    # RNA
    # ----------------------------
    st.markdown("### RNA")
    if st.button("➕ Add RNA", key="add_rna"):
        add_entry("rna")
        st.rerun()

    for i, r in enumerate(st.session_state["rna"]):
        with st.expander(f"RNA {i+1}", expanded=False):
            pick = st.selectbox("Load saved", rna_saved, key=f"rna_pick_{i}")
            if pick:
                ent = api_get(f"/sequences/rna/{pick}", timeout=TIMEOUT_FAST)
                r["name"] = ent.get("name", pick)
                r["sequence"] = ent.get("sequence", "")

            r["sequence"] = st.text_area("Sequence", value=r.get("sequence", ""), height=100, key=f"rna_seq_{i}")
            r.setdefault("ptms", [])
            r["ptms_cleaned"] = ptm_editor(r["ptms"], key_prefix=f"rna_{i}")

            if st.button("Remove RNA", key=f"rna_rm_{i}"):
                remove_entry("rna", i)
                st.rerun()

    st.divider()

    # ----------------------------
    # RUN BUTTON + TASK VIEW
    # ----------------------------
    run_col1, run_col2 = st.columns([2, 3])

    with run_col1:
        if st.button("🚀 Run now", key="run_now"):
            if not jobname.strip():
                st.error("Job name is required.")
                return

            spec = build_job_spec(
                jobname=jobname,
                ligand_smiles=ligand_smiles,
                covalent=covalent,
                chain=chain,
                residue=residue,
                prot_atom_label=prot_atom_label,
                custom_atom=custom_atom,
                lig_atom=lig_atom,
                ions=ions,
                cofactors=cofactors,
                seeds=seeds,
                skip_rosetta=skip_rosetta,
                multi_seed=multi_seed,
            )

            # /run is synchronous on server.py — set huge timeout to avoid Streamlit request timeout.
            out = api_post("/run", json_body=spec, timeout=TIMEOUT_RUN)
            st.session_state["task_id"] = out.get("task_id", "")
            st.success(f"Started: {st.session_state['task_id']}")

            # Kick a runs refresh
            st.session_state["runs_refresh_nonce"] += 1

    with run_col2:
        task_id = st.session_state.get("task_id", "")
        if task_id:
            # poll quickly (short timeout) — server returns fast
            t = api_get(f"/tasks/{task_id}", timeout=TIMEOUT_FAST)
            st.write(f"Status: **{t.get('status','')}**")
            st.code("\n".join(t.get("logs", [])[-200:]), language="text")
            if t.get("status") == "error":
                st.error(t.get("error", ""))


# ----------------------------
# Sidebar: Profiles
# ----------------------------
init_state()

prof = api_get("/profiles", timeout=TIMEOUT_FAST)
profiles = prof.get("profiles", [])
current = prof.get("current", "")

st.title("AF3 Web GUI")

pcol1, pcol2, pcol3 = st.columns([2, 2, 3])
with pcol1:
    sel = st.selectbox(
        "Profile",
        [""] + profiles,
        index=(profiles.index(current) + 1 if current in profiles else 0),
        key="profile_select",
    )
with pcol2:
    if st.button("Activate", key="profile_activate") and sel:
        api_post(f"/profiles/{sel}/activate", timeout=TIMEOUT_MED)
        st.success(f"Active profile: {sel}")
        # refresh page state after activation
        st.session_state["runs_refresh_nonce"] += 1
        st.rerun()
with pcol3:
    new_name = st.text_input("Create new profile", placeholder="e.g. Oliver", key="profile_new_name")
    if st.button("Create", key="profile_create") and new_name.strip():
        api_post("/profiles", json_body={"name": new_name.strip()}, timeout=TIMEOUT_MED)
        st.rerun()

st.divider()

tabs = st.tabs(["Sequences", "Ligands", "Alphafold", "Runs", "Config"])

# ----------------------------
# Tab: Sequences
# ----------------------------
with tabs[0]:
    st.subheader("Saved sequences")

    seq_groups = api_get("/sequences", timeout=TIMEOUT_FAST)
    prot_names = seq_groups.get("protein") or []
    dna_names = seq_groups.get("dna") or []
    rna_names = seq_groups.get("rna") or []

    left, right = st.columns([2, 3])

    with left:
        kind = st.selectbox("Type", ["protein", "dna", "rna"], key="seq_kind")
        names = {"protein": prot_names, "dna": dna_names, "rna": rna_names}.get(kind, [])
        pick = st.selectbox("Select", [""] + names, key="seq_pick")

        if pick:
            ent = api_get(f"/sequences/{kind}/{pick}", timeout=TIMEOUT_FAST)
            st.write(f"Type: {ent.get('type','')}")
            st.text_area("Sequence", ent.get("sequence", ""), height=180, disabled=True, key="seq_view")
            st.text_input("Template", ent.get("template", ""), disabled=True, key="tmpl_view")

            if st.button("Delete selected", key="seq_delete"):
                api_delete(f"/sequences/{pick}", timeout=TIMEOUT_MED)  # server uses name-only delete
                st.rerun()

    with right:
        st.subheader("Add / update")
        nm = st.text_input("Name", key="seq_new_name")
        typ = st.selectbox("Type", ["protein", "dna", "rna"], key="seq_new_type")
        seq = st.text_area("Sequence", height=220, key="seq_new_seq")
        tmpl = st.text_input("Template (optional)", key="seq_new_tmpl")

        if st.button("Save sequence", key="seq_save"):
            api_post(
                "/sequences",
                json_body={"name": nm, "sequence": seq, "type": typ, "template": tmpl},
                timeout=TIMEOUT_MED,
            )
            st.success("Saved")
            st.rerun()

# ----------------------------
# Tab: Ligands
# ----------------------------
with tabs[1]:
    st.subheader("Saved ligands")

    lig_rows = api_get("/ligands", timeout=TIMEOUT_FAST) or []
    lig_names = [r.get("name", "") for r in lig_rows if isinstance(r, dict)]
    lig_by_name = {r.get("name", ""): r for r in lig_rows if isinstance(r, dict)}

    left, right = st.columns([2, 3])

    with left:
        pick = st.selectbox("Select ligand", [""] + sorted(lig_names, key=str.lower), key="lig_pick")
        if pick:
            ent = lig_by_name.get(pick, {})
            st.code(ent.get("smiles", ""), language="text")
            st.caption(f"Hash: {ent.get('hash', '')}")

            # ✅ show LIG.svg preview from ligand hash cache
            try:
                st.image(api_file_url(f"/ligands/{pick}/preview"), caption="LIG.svg (cache)", use_container_width=True)
            except Exception:
                st.info("No preview image available.")

            # ✅ download LIG.pdb
            st.markdown(f"- **Download PDB:** {api_file_url(f'/ligands/{pick}/pdb')}")

            if st.button("Delete ligand", key="lig_delete"):
                api_delete(f"/ligands/{pick}", timeout=TIMEOUT_MED)
                st.rerun()

    with right:
        st.subheader("Create ligand")
        nm = st.text_input("Ligand name", key="lig_new_name")
        smiles = st.text_input("SMILES", key="lig_new_smiles")
        if st.button("Generate + save ligand", key="lig_create"):
            out = api_post("/ligands", json_body={"name": nm, "smiles": smiles}, timeout=TIMEOUT_RUN)
            st.success(f"Saved: {out.get('name')}")
            st.rerun()

# ----------------------------
# Tab: Alphafold
# ----------------------------
with tabs[2]:
    job_spec_form()

# ----------------------------
# Tab: Runs
# ----------------------------
with tabs[3]:
    st.subheader("Runs history")

    # --------- NORMALIZE /runs RESPONSE (FIXES YOUR ERROR) ---------
    runs_resp = api_get("/runs")

    if isinstance(runs_resp, dict) and isinstance(runs_resp.get("runs"), list):
        # legacy shape: {"runs": [...]}
        runs = runs_resp["runs"]
    elif isinstance(runs_resp, list):
        # new shape: plain list
        runs = runs_resp
    else:
        runs = []
    # ---------------------------------------------------------------

    if not runs:
        st.info("No runs yet.")
        st.stop()

    # Build rows with a stable run_id (use server id if present, else index)
    rows = []
    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            continue
        run_id = r.get("id", i)
        rows.append(
            {
                "Select": False,
                "run_id": int(run_id) if str(run_id).isdigit() else i,
                "jobname": (r.get("jobname") or "").strip(),
                "finished_at": (r.get("finished_at") or r.get("created_at") or "").strip(),
            }
        )

    df = pd.DataFrame(rows)

    # Persist checkbox state across reruns
    st.session_state.setdefault("runs_editor", df)

    st.caption("Check one or more runs to view metrics / actions.")
    edited = st.data_editor(
        st.session_state["runs_editor"],
        use_container_width=True,
        hide_index=True,
        disabled=["run_id", "jobname", "finished_at"],
        column_config={
            "Select": st.column_config.CheckboxColumn(required=False),
            "run_id": st.column_config.NumberColumn("ID"),
            "jobname": st.column_config.TextColumn("Job"),
            "finished_at": st.column_config.TextColumn("Finished"),
        },
        key="runs_editor_widget",
    )

    st.session_state["runs_editor"] = edited

    selected = edited[edited["Select"] == True]
    selected_ids = selected["run_id"].tolist() if not selected.empty else []

    if not selected_ids:
        st.info("Select a run to see metrics.")
        st.stop()

    active_id = int(selected_ids[0])

    st.divider()
    st.subheader(f"Selected run: {active_id}")

    # ---- Metrics ----
    try:
        m = api_get(f"/runs/{active_id}/metrics_html")
        html = m.get("html", "")
        if html:
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No metrics returned.")
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")

    st.divider()

    # ---- Actions ----
    a1, a2 = st.columns([2, 3])

    with a1:
        st.subheader("Run Rosetta")
        ms = st.checkbox("multi_seed", value=False, key="runs_ms")
        if st.button("Run Rosetta on selected", key="runs_run_rosetta"):
            try:
                out = api_post(
                    f"/runs/{active_id}/run_rosetta",
                    params={"multi_seed": bool(ms)},
                    timeout=60 * 60,
                )
                st.success("Rosetta finished.")
                if isinstance(out, dict) and out.get("stdout"):
                    st.code(out["stdout"], language="text")
            except Exception as e:
                st.error(f"Rosetta failed: {e}")

    with a2:
        st.subheader("Downloads")
        st.caption("Downloads open in a new tab.")

        dl_job = st.checkbox("Entire job folder (.zip)", value=True, key="dl_job")
        dl_af = st.checkbox("Alphafold model (*.cif)", value=False, key="dl_af")
        dl_ros = st.checkbox(
            "Rosetta model (model_relaxed_restored_0001.pdb)", value=False, key="dl_ros"
        )

        def dl_url(kind: str) -> str:
            return f"{API_BASE}/runs/{active_id}/download?kind={kind}"

        if dl_job:
            st.link_button("Download job folder", dl_url("job_folder"))
        if dl_af:
            st.link_button("Download AF model", dl_url("af_model"))
        if dl_ros:
            st.link_button("Download Rosetta model", dl_url("rosetta_model"))

# ----------------------------
# Tab: Config
# ----------------------------
with tabs[4]:
    st.subheader("Config")
    st.info("Next step: expose config.yaml read/write endpoints (same as your Config tab).")
