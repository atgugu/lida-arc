from __future__ import annotations

from pathlib import Path
import networkx as nx

from ..memory.pam import PerceptualAssociativeMemory
from ..memory.workspace import SituationalModel


def write_pam_dot(pam: PerceptualAssociativeMemory, out_path: Path) -> Path:
    """Write the PAM graph to a Graphviz DOT file with basic labels."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("digraph PAM {\n")
        # Nodes
        for nid, data in pam.g.nodes(data=True):
            label = data.get("label", nid)
            kind = data.get("kind", "")
            act = float(data.get("activation", 0.0))
            f.write(f'  "{nid}" [label="{label}\n({kind}) a={act:.2f}"];\n')
        # Edges
        for src, dst, ed in pam.g.edges(data=True):
            rel = ed.get("rel", "")
            w = float(ed.get("weight", 0.0))
            f.write(f'  "{src}" -> "{dst}" [label="{rel}:{w:.2f}"];\n')
        f.write("}\n")
    return out_path


def write_workspace_dot(ws: SituationalModel, out_path: Path) -> Path:
    """Export workspace objects and relations to DOT."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("digraph Workspace {\n")
        for oid, obj in ws.objects.items():
            strength = max(obj.features.values()) if obj.features else 0.0
            f.write(f'  "{oid}" [label="{oid} a={strength:.2f}"];\n')
        for r in ws.relations:
            f.write(f'  "{r.subj}" -> "{r.obj}" [label="{r.rel} ({r.confidence:.2f})"];\n')
        f.write("}\n")
    return out_path
