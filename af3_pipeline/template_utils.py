#!/usr/bin/env python3
"""
template_utils.py
=================

Fetches a PDB structure in mmCIF format, parses chain sequences and resolved residues,
selects the best matching chain to a given query sequence, builds a query-template
index mapping suitable for AlphaFold 3, and caches CIFs centrally.

✅ Refactored to use cache_utils
✅ Supports mapping-level caching (PDB + chain + query hash)
✅ Config-aware (ligand/template cache paths now read from config.yaml)
"""

import requests
import json
from typing import Dict, List
from Bio import pairwise2
from pathlib import Path
# --- Configuration ---
from .config import cfg

# ✅ FIXED: use relative import for internal module
from .cache_utils import (
    compute_hash,
    get_cache_file,
    exists_in_cache,
    save_to_cache,
    load_from_cache
)

# ==============================
# 🌐 RCSB Download
# ==============================
RCSB_URL = "https://files.rcsb.org/download/{}.cif"

# ==============================
# 📥 CIF Download (PDB-level)
# ==============================
def load_or_download_cif(pdb_id: str) -> str:
    """
    Load CIF from cache if exists, otherwise download and cache.
    """
    pdb_id_upper = pdb_id.upper()
    cif_filename = f"{pdb_id_upper}.cif"

    if exists_in_cache("templates", pdb_id_upper, cif_filename):
        print(f"📦 Using cached CIF for {pdb_id_upper}")
        return load_from_cache("templates", pdb_id_upper, cif_filename)

    print(f"🌐 Downloading CIF for {pdb_id_upper} from RCSB...")
    r = requests.get(RCSB_URL.format(pdb_id_upper))
    if r.status_code != 200:
        raise RuntimeError(f"❌ Failed to download CIF for {pdb_id_upper} ({r.status_code})")

    cif_text = r.text
    save_to_cache("templates", pdb_id_upper, cif_filename, cif_text)
    return cif_text


# ==============================
# 🧬 Parse polymer sequences
# ==============================
def parse_poly_seq(cif_text: str) -> Dict[str, List[str]]:
    seq_blocks = {}
    reading = False
    headers = []
    chain_idx = None
    monid_idx = None

    for line in cif_text.splitlines():
        if line.startswith("loop_"):
            reading = False
        elif line.startswith("_pdbx_poly_seq_scheme."):
            if not reading:
                reading = True
                headers = []
            headers.append(line.strip())
        elif reading and not line.startswith("_"):
            if chain_idx is None:
                chain_idx = next(i for i, h in enumerate(headers) if h.endswith("asym_id"))
                monid_idx = next(i for i, h in enumerate(headers) if h.endswith("mon_id"))
            parts = line.split()
            if len(parts) <= monid_idx:
                continue
            chain = parts[chain_idx]
            resn = parts[monid_idx]
            seq_blocks.setdefault(chain, []).append(resn)

    return seq_blocks


# ==============================
# 🧪 Parse resolved residues
# ==============================
def parse_atom_site(cif_text: str) -> Dict[str, List[int]]:
    resolved = {}
    reading = False
    headers = []
    chain_idx = None
    seq_idx = None

    for line in cif_text.splitlines():
        if line.startswith("loop_"):
            reading = False
        elif line.startswith("_atom_site."):
            if not reading:
                reading = True
                headers = []
            headers.append(line.strip())
        elif reading and not line.startswith("_"):
            if chain_idx is None:
                chain_idx = next(i for i, h in enumerate(headers) if h.endswith("label_asym_id"))
                seq_idx = next(i for i, h in enumerate(headers) if h.endswith("label_seq_id"))
            parts = line.split()
            if len(parts) <= seq_idx:
                continue
            chain = parts[chain_idx]
            try:
                resnum = int(parts[seq_idx])
            except ValueError:
                continue
            resolved.setdefault(chain, set()).add(resnum - 1)  # 0-based indexing
    return {ch: sorted(list(vals)) for ch, vals in resolved.items()}


# ==============================
# 🧭 Chain matching
# ==============================
AA_MAP = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

def convert_one_to_three(seq: str) -> List[str]:
    return [AA_MAP.get(aa, "UNK") for aa in seq]

def sequence_identity(seq1: List[str], seq2: List[str]) -> float:
    length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:length], seq2[:length]))
    return matches / max(len(seq1), len(seq2))

def pick_best_chain(chain_seqs: Dict[str, List[str]], query_seq: str) -> str:
    query_three_letter = convert_one_to_three(query_seq)
    best_chain = None
    best_score = -1
    for chain_id, seq in chain_seqs.items():
        score = sequence_identity(seq, query_three_letter)
        if score > best_score:
            best_score = score
            best_chain = chain_id
    if best_chain is None:
        raise RuntimeError("❌ No suitable chain found in CIF.")
    print(f"✅ Selected chain {best_chain} (identity: {best_score:.2f})")
    return best_chain


# ==============================
# 🧮 Index mapping
# ==============================
def build_index_mapping(query_seq: str, template_seq: List[str], resolved_indices: List[int]):
    """
    Align query (1-letter) to template (3-letter) sequence using global alignment,
    then return index pairs (queryIndices, templateIndices) following AF3 spec.

    - Skips template residues that are unresolved
    - Uses 0-based indices
    """
    rev_map = {v: k for k, v in AA_MAP.items()}
    template_one = "".join(rev_map.get(aa, "X") for aa in template_seq)
    alignments = pairwise2.align.globalms(query_seq, template_one, 2, -1, -2, -2, one_alignment_only=True)
    aln_q, aln_t, score, start, end = alignments[0]

    q_idx, t_idx = 0, 0
    query_indices, template_indices = [], []

    for q_res, t_res in zip(aln_q, aln_t):
        if q_res != "-" and t_res != "-":
            if t_idx in resolved_indices:
                query_indices.append(q_idx)
                template_indices.append(t_idx)
        if q_res != "-":
            q_idx += 1
        if t_res != "-":
            t_idx += 1

    print(f"✅ Alignment complete: {len(query_indices)} mapped residues (score={score:.1f})")
    return query_indices, template_indices


def extract_single_chain_cif(full_cif_text: str, chain_id: str) -> str:
    """
    Return a minimal CIF text containing only the specified chain.
    """
    lines = full_cif_text.splitlines()
    output_lines = []
    atom_site_headers = []
    chain_col_index = None

    for idx, line in enumerate(lines):
        if line.startswith("_atom_site."):
            atom_site_headers.append(line)
            if "label_asym_id" in line:
                chain_col_index = len(atom_site_headers) - 1
            elif chain_col_index is None and "label_asym_id" in line:
                chain_col_index = len(atom_site_headers) - 1

    atom_site_mode = False
    header_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("loop_"):
            output_lines.append(line)
            atom_site_mode = False
            header_count = 0
            continue

        if stripped.startswith("_atom_site."):
            output_lines.append(line)
            header_count += 1
            if header_count == len(atom_site_headers):
                atom_site_mode = True
            continue

        if atom_site_mode and stripped and not stripped.startswith("_"):
            if chain_col_index is None:
                output_lines.append(line)
                continue
            cols = stripped.split()
            if len(cols) > chain_col_index and cols[chain_col_index] == chain_id:
                output_lines.append(line)
            continue

        if not atom_site_mode:
            output_lines.append(line)

    return "\n".join(output_lines)


# ==============================
# 🧠 Main template processing
# ==============================
def get_template_mapping(query_seq: str, pdb_id: str) -> dict:
    pdb_id_upper = pdb_id.upper()
    cif_text = load_or_download_cif(pdb_id_upper)
    chain_seqs = parse_poly_seq(cif_text)
    resolved = parse_atom_site(cif_text)

    if len(chain_seqs) == 0:
        raise RuntimeError(f"❌ No chain sequences found in {pdb_id_upper} CIF")

    best_chain = pick_best_chain(chain_seqs, query_seq)
    if best_chain not in resolved:
        raise RuntimeError(f"❌ No resolved residues found for chain {best_chain}")

    reduced_cif_text = extract_single_chain_cif(cif_text, best_chain)

    query_hash = compute_hash(query_seq)
    mapping_key = f"{pdb_id_upper}_{best_chain}_{query_hash}"
    mapping_filename = f"{mapping_key}.json"

    if exists_in_cache("templates", mapping_key, mapping_filename):
        print(f"📦 Using cached mapping for {pdb_id_upper}, chain {best_chain}")
        mapping_data = json.loads(load_from_cache("templates", mapping_key, mapping_filename))
        mapping_data["mmcif"] = reduced_cif_text
        return mapping_data

    query_idx, template_idx = build_index_mapping(
        query_seq,
        chain_seqs[best_chain],
        resolved[best_chain]
    )

    mapping_data = {
        "mmcif": reduced_cif_text,
        "queryIndices": query_idx,
        "templateIndices": template_idx,
        "chain": best_chain,
        "alignmentScore": len(query_idx),
        "cifPath": str(get_cache_file("templates", pdb_id_upper, f"{pdb_id_upper}.cif")),
        "mappingKey": mapping_key
    }

    save_to_cache("templates", mapping_key, mapping_filename, json.dumps(mapping_data))
    return mapping_data


# ==============================
# 🧪 CLI Test
# ==============================
if __name__ == "__main__":
    pdb_id = input("Enter PDB ID (e.g., 8OV6): ").strip()
    query_seq = input("Enter query sequence (1-letter AA): ").strip()
    result = get_template_mapping(query_seq, pdb_id)
    print("\n✅ Template processing complete")
    print("Chain:", result["chain"])
    print("Mapping key:", result["mappingKey"])
    print("CIF path:", result["cifPath"])
    print("Query indices:", result["queryIndices"][:10], "...")
    print("Template indices:", result["templateIndices"][:10], "...")
