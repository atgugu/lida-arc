from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx

from ..memory.pam import PerceptualAssociativeMemory, PAMNode
from ..action.procedural import Scheme
from ..memory.episodic import Episode, TransientEpisodicMemory, EpisodicMemory


SCHEMA = {
    "pam_nodes": "CREATE TABLE IF NOT EXISTS pam_nodes (node_id TEXT PRIMARY KEY, kind TEXT, label TEXT, base_strength REAL, activation REAL)",
    "pam_edges": "CREATE TABLE IF NOT EXISTS pam_edges (src TEXT, dst TEXT, rel TEXT, weight REAL, PRIMARY KEY(src,dst,rel))",
    "schemes": "CREATE TABLE IF NOT EXISTS schemes (scheme_id TEXT PRIMARY KEY, context_cues TEXT, action_name TEXT, expected_effects TEXT, utility REAL)",
    "tem": "CREATE TABLE IF NOT EXISTS tem (cycle_index INTEGER, payload TEXT, tags TEXT, strength REAL)",
    "em": "CREATE TABLE IF NOT EXISTS em (cycle_index INTEGER, payload TEXT, tags TEXT, strength REAL)",
}


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        for sql in SCHEMA.values():
            cur.execute(sql)
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # PAM
    def save_pam(self, pam: PerceptualAssociativeMemory) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM pam_nodes")
        cur.execute("DELETE FROM pam_edges")
        for nid, data in pam.g.nodes(data=True):
            cur.execute(
                "INSERT INTO pam_nodes(node_id, kind, label, base_strength, activation) VALUES (?,?,?,?,?)",
                (nid, data.get("kind"), data.get("label"), float(data.get("base_strength", 1.0)), float(data.get("activation", 0.0))),
            )
        for src, dst, ed in pam.g.edges(data=True):
            cur.execute(
                "INSERT INTO pam_edges(src, dst, rel, weight) VALUES (?,?,?,?)",
                (src, dst, ed.get("rel"), float(ed.get("weight", 0.0))),
            )
        self.conn.commit()

    def load_pam(self, pam: PerceptualAssociativeMemory) -> None:
        pam.g = nx.DiGraph()
        cur = self.conn.cursor()
        for row in cur.execute("SELECT node_id, kind, label, base_strength, activation FROM pam_nodes"):
            node = PAMNode(row[0], row[1], row[2], float(row[3]), float(row[4]))
            pam.ensure_node(node)
            pam.set_activation(node.node_id, node.activation)
        for row in cur.execute("SELECT src, dst, rel, weight FROM pam_edges"):
            pam.add_link(row[0], row[1], row[2], float(row[3]))

    # Schemes
    def save_schemes(self, schemes: Iterable[Scheme]) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM schemes")
        for sc in schemes:
            cur.execute(
                "INSERT INTO schemes(scheme_id, context_cues, action_name, expected_effects, utility) VALUES (?,?,?,?,?)",
                (
                    sc.scheme_id,
                    json.dumps(sc.context_cues, separators=(",", ":")),
                    sc.action_name,
                    json.dumps(sc.expected_effects, separators=(",", ":")),
                    float(sc.utility),
                ),
            )
        self.conn.commit()

    def load_schemes(self) -> List[Scheme]:
        out: List[Scheme] = []
        cur = self.conn.cursor()
        for row in cur.execute("SELECT scheme_id, context_cues, action_name, expected_effects, utility FROM schemes"):
            out.append(
                Scheme(
                    scheme_id=row[0],
                    context_cues=json.loads(row[1]) if row[1] else {},
                    action_name=row[2],
                    expected_effects=json.loads(row[3]) if row[3] else {},
                    utility=float(row[4]),
                )
            )
        return out

    # Episodic
    def save_tem(self, tem: TransientEpisodicMemory) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM tem")
        for ep in tem._episodes:
            cur.execute(
                "INSERT INTO tem(cycle_index, payload, tags, strength) VALUES (?,?,?,?)",
                (int(ep.cycle_index), json.dumps(ep.payload, separators=(",", ":")), json.dumps(list(ep.tags)), float(ep.strength)),
            )
        self.conn.commit()

    def save_em(self, em: EpisodicMemory) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM em")
        for ep in em._episodes:
            cur.execute(
                "INSERT INTO em(cycle_index, payload, tags, strength) VALUES (?,?,?,?)",
                (int(ep.cycle_index), json.dumps(ep.payload, separators=(",", ":")), json.dumps(list(ep.tags)), float(ep.strength)),
            )
        self.conn.commit()

    def load_tem(self, tem: TransientEpisodicMemory) -> None:
        tem._episodes.clear()
        cur = self.conn.cursor()
        for row in cur.execute("SELECT cycle_index, payload, tags, strength FROM tem"):
            tem._episodes.append(
                Episode(cycle_index=int(row[0]), payload=json.loads(row[1]), tags=tuple(json.loads(row[2])), strength=float(row[3]))
            )

    def load_em(self, em: EpisodicMemory) -> None:
        em._episodes.clear()
        cur = self.conn.cursor()
        for row in cur.execute("SELECT cycle_index, payload, tags, strength FROM em"):
            em._episodes.append(
                Episode(cycle_index=int(row[0]), payload=json.loads(row[1]), tags=tuple(json.loads(row[2])), strength=float(row[3]))
            )

    # Convenience
    def save_all(self, pam: PerceptualAssociativeMemory, schemes: Iterable[Scheme], tem: TransientEpisodicMemory, em: EpisodicMemory) -> None:
        self.save_pam(pam)
        self.save_schemes(schemes)
        self.save_tem(tem)
        self.save_em(em)

    def load_all(self, pam: PerceptualAssociativeMemory, proc_schemes_out: List[Scheme], tem: TransientEpisodicMemory, em: EpisodicMemory) -> None:
        self.load_pam(pam)
        proc_schemes_out.clear()
        proc_schemes_out.extend(self.load_schemes())
        self.load_tem(tem)
        self.load_em(em)
