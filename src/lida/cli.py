from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from .core.cycle import CognitiveCycleEngine, CycleConfig
from .core.scheduler import CodeletScheduler
from .memory.pam import PerceptualAssociativeMemory, PAMNode
from .memory.workspace import SituationalModel
from .memory.episodic import TransientEpisodicMemory, EpisodicMemory
from .attention.global_workspace import GlobalWorkspace
from .attention.novelty import NoveltyTracker
from .attention.learning import AttentionalLearner
from .action.procedural import ProceduralMemory, Scheme
from .action.behavior_net import BehaviorNet, BNConfig
from .action.learning import TDLearner
from .core.events import Coalition
from .affect.drives import Drives
from .perception.codelets import make_spread_codelet
from .perception.relations_builder import make_relations_codelet
from .perception.category_promotion import make_category_promotion_codelet
from .perception.attn_codelets import make_attention_seed_codelets
from .perception.category_induction import CategoryInductionTracker, make_category_induction_codelet
from .env.symbolic_foraging import SymbolicForagingEnv
from .env.gridworld import Gridworld
from .perception.structure_building import ingest_observation
from .perception.neighborhood import ingest_grid_neighborhood
from .core.codelet import Codelet
from .persistence.sqlite_store import SQLiteStore
from .tools.report import load_traces, compute_metrics, write_markdown
from .tools.graph_export import write_pam_dot, write_workspace_dot


def _print(msg: str) -> None:
    print(msg, flush=True)


async def _demo_run(hz: float, max_cycles: int, trace_dir: Optional[str], env_name: str) -> None:
    pass


def run_cmd(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="lida run")
    parser.add_argument("--hz", type=float, default=8.0)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--env", type=str, choices=["symbolic", "grid"], default="symbolic")
    parser.add_argument("--load-snapshot", type=str, default=None)
    parser.add_argument("--save-snapshot", type=str, default=None)
    args = parser.parse_args(argv)

    trace_path = Path(args.trace_dir) if args.trace_dir else None
    engine = CognitiveCycleEngine(CycleConfig(hz=args.hz, understanding_budget_ms=40, attention_budget_ms=30, action_budget_ms=30, log_fn=_print, trace_dir=trace_path))

    pam = PerceptualAssociativeMemory()
    workspace = SituationalModel()
    gw = GlobalWorkspace()
    novelty = NoveltyTracker()
    attn_learner = AttentionalLearner()
    tem = TransientEpisodicMemory(decay=0.02)
    em = EpisodicMemory(consolidate_threshold=0.6)
    proc = ProceduralMemory()
    bn = BehaviorNet(BNConfig(precondition_weight=1.0, utility_weight=1.0, cooldown_steps=2, cooldown_penalty=0.2))
    td = TDLearner()
    drives = Drives()

    # Load snapshot if provided
    if args.load_snapshot:
        store = SQLiteStore(Path(args.load_snapshot))
        schemes_loaded = store.load_schemes()
        for sc in schemes_loaded:
            proc.upsert_scheme(sc)
        store.load_pam(pam)
        store.load_tem(tem)
        store.load_em(em)
        store.close()

    # Seed knowledge and default schemes if empty
    if pam.g.number_of_nodes() == 0:
        pam.ensure_node(PAMNode("feat:food", kind="feature", label="food"))
        pam.ensure_node(PAMNode("feat:shiny", kind="feature", label="shiny"))
        pam.ensure_node(PAMNode("feat:hazard", kind="feature", label="hazard"))
        pam.add_link("feat:shiny", "feat:food", rel="assoc", weight=0.1)
    if not list(proc.iter_schemes()):
        proc.upsert_scheme(Scheme("act:forage", {"feat:food": 1.0}, "forage", {"energy": +0.05}, utility=0.0))
        proc.upsert_scheme(Scheme("act:investigate", {"feat:shiny": 1.0}, "investigate", {}, utility=0.0))
        for a in ("up", "down", "left", "right"):
            proc.upsert_scheme(Scheme(f"act:move:{a}", {"bias:any": 0.1}, a, {}, utility=0.0))

    grid = Gridworld()
    symb = SymbolicForagingEnv(["FOOD", "SHINY", "FOOD", "SHINY"])  # default

    ingest_shared = {"seeds": {}}

    def _ingest_action():
        # Set feelings as PAM activations from drives each cycle
        pam.feelings_from_map({
            "stress": max(0.0, 1.0 - drives.safety.level),
            "hunger": max(0.0, 1.0 - drives.energy.level),
        })
        if args.env == "grid":
            seeds = ingest_grid_neighborhood(pam, workspace, grid._tile_token, grid.x, grid.y)
        else:
            tok = symb.observe()
            seeds = ingest_observation(pam, workspace, [(tok, 1.0), ("bias:any", 0.5)])
        ingest_shared["seeds"] = seeds

    ingest_c = Codelet(name="ingest_env", urgency=2.0, action=_ingest_action, kind="perceptual", metadata={"shared": ingest_shared})
    shared = ingest_c.metadata["shared"]
    spread_c = make_spread_codelet(pam, shared)
    relate_c = make_relations_codelet(workspace)
    tracker = CategoryInductionTracker()
    induce_c = make_category_induction_codelet(tracker, pam, workspace)
    promote_c = make_category_promotion_codelet(pam, workspace)
    attn_codelets = make_attention_seed_codelets(workspace)

    conscious_payload: dict[str, float] | None = None
    last_winner_id: str | None = None
    last_features: tuple[tuple[str, float], ...] | None = None

    async def understanding() -> None:
        ingest_c.run()
        spread_c.run()
        relate_c.run()
        induce_c.run()
        promote_c.run()
        feats = {nid: act for nid, act in pam.top_k_active(6)}
        relations_snapshot = list(workspace.relations)
        workspace.clear()
        for nid, act in feats.items():
            workspace.upsert_object(nid, features={nid: act})
        workspace.relations.extend(relations_snapshot)

    async def attention() -> None:
        nonlocal conscious_payload, last_winner_id, last_features
        hunger = drives.hunger_boost()
        sched = CodeletScheduler()
        for c in attn_codelets:
            sched.push(c)
        max_runs = 2 if len(workspace.objects) <= 4 else 1
        for c in sched.pop(max_runs=max_runs):
            c.run()
        proposals = attn_codelets[0].metadata.get("coalitions", [])
        relevance = {}
        for coal in proposals:
            cue = {k: float(v) for k, v in coal.features if ":" in k and not k.startswith("rel:")}
            sim = tem.retrieve_similarity(cue, top_n=3) + em.retrieve_similarity(cue, top_n=3)
            base_rel = 0.0
            if coal.id == "feat:food" or any(k == "subj:feat:food" or k == "obj:feat:food" for k, _ in coal.features):
                base_rel = max(0.0, hunger - 1.0)
            relevance[coal.id] = base_rel + sim
        winner, _ = gw.compete(proposals, relevance=relevance)
        cc = gw.broadcast(engine._cycle_index, winner)
        conscious_payload = cc.payload if cc else None
        last_winner_id = winner.id if winner else None
        last_features = tuple(winner.features) if winner else None
        engine.trace({"type": "attention", "cycle": engine._cycle_index, "coalitions": [{"id": c.id, "salience": c.salience, "relevance": relevance.get(c.id, 0.0)} for c in proposals], "winner": last_winner_id})

    async def action() -> None:
        bn.tick()
        # Encode with feeling tags
        if conscious_payload is not None:
            from lida.core.events import ConsciousContent
            tags = {"conscious"}
            if (1.0 - drives.safety.level) > 0.3:
                tags.add("feeling:stress")
            if (1.0 - drives.energy.level) > 0.3:
                tags.add("feeling:hunger")
            tem.encode(ConsciousContent(cycle_index=engine._cycle_index, tags=frozenset(tags), payload=conscious_payload))
        tem.step_decay()
        em.consolidate_from(tem)

        reward = 0.0
        tile = None
        if not conscious_payload:
            drives.step_decay()
            # small decay even if no action
            for sc in proc.iter_schemes():
                td.decay(sc, factor=0.995)
            return None
        cands = proc.instantiate(conscious_payload)
        winner = bn.select(cands)
        if winner:
            action_name = winner.scheme.action_name
            if action_name == "forage":
                drives.apply_effect("energy", winner.scheme.expected_effects.get("energy", 0.0))
            if args.env == "grid":
                reward = grid.step(action_name)
                tile = grid.observe()
                if tile == "HAZARD":
                    drives.apply_effect("safety", -0.05)
                else:
                    drives.apply_effect("safety", +0.005)
            else:
                reward = symb.evaluate(action_name)
            # Affect-modulated learning rate: higher under stress/hunger
            affect_alpha = 0.1 + 0.2 * max(1.0 - drives.safety.level, 1.0 - drives.energy.level)
            td.update(winner.scheme, reward, alpha=affect_alpha)
            bn.on_selected(winner.scheme.scheme_id)
            engine.trace({"type": "action", "cycle": engine._cycle_index, "action": action_name, "reward": reward, "utility": winner.scheme.utility, "tile": tile})
        drives.step_decay()
        # utility decay across all schemes
        for sc in proc.iter_schemes():
            td.decay(sc, factor=0.995)
        return None

    engine.on_understanding(understanding)
    engine.on_attention(attention)
    engine.on_action(action)

    asyncio.run(engine.run(max_cycles=args.max_cycles))

    if args.save_snapshot:
        store = SQLiteStore(Path(args.save_snapshot))
        store.save_all(pam, list(proc.iter_schemes()), tem, em)
        store.close()


def snapshot_cmd(argv: Optional[list[str]] = None) -> None:
    # unchanged
    parser = argparse.ArgumentParser(prog="lida snapshot")
    sub = parser.add_subparsers(dest="op", required=True)
    save_p = sub.add_parser("save", help="Save snapshot to sqlite db")
    save_p.add_argument("db", type=str)
    load_p = sub.add_parser("load", help="Load snapshot from sqlite db")
    load_p.add_argument("db", type=str)
    args = parser.parse_args(argv)

    db = Path(args.db)
    store = SQLiteStore(db)

    if args.op == "save":
        pam = PerceptualAssociativeMemory()
        tem = TransientEpisodicMemory()
        em = EpisodicMemory()
        schemes: list[Scheme] = []
        store.save_all(pam, schemes, tem, em)
        print(f"Saved snapshot to {db}")
    elif args.op == "load":
        pam = PerceptualAssociativeMemory()
        tem = TransientEpisodicMemory()
        em = EpisodicMemory()
        schemes: list[Scheme] = []
        store.load_all(pam, schemes, tem, em)
        print(f"Loaded snapshot from {db}: pam_nodes={pam.g.number_of_nodes()} schemes={len(schemes)} tem={len(tem._episodes)} em={len(em._episodes)}")


def demo_cmd(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="lida demo")
    parser.add_argument("--env", type=str, choices=["symbolic", "grid"], default="symbolic")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--hz", type=float, default=8.0)
    parser.add_argument("--trace-dir", type=str, default="./lida/runs")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--snapshot-prefix", type=str, default=None)
    args = parser.parse_args(argv)

    trace_path = Path(args.trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)

    pam = PerceptualAssociativeMemory()
    workspace = SituationalModel()
    gw = GlobalWorkspace()
    novelty = NoveltyTracker()
    attn_learner = AttentionalLearner()
    tem = TransientEpisodicMemory(decay=0.02)
    em = EpisodicMemory(consolidate_threshold=0.6)
    proc = ProceduralMemory()
    bn = BehaviorNet(BNConfig(precondition_weight=1.0, utility_weight=1.0, cooldown_steps=2, cooldown_penalty=0.2))
    td = TDLearner()
    drives = Drives()

    pam.ensure_node(PAMNode("feat:food", kind="feature", label="food"))
    pam.ensure_node(PAMNode("feat:shiny", kind="feature", label="shiny"))
    pam.ensure_node(PAMNode("feat:hazard", kind="feature", label="hazard"))
    pam.add_link("feat:shiny", "feat:food", rel="assoc", weight=0.1)
    proc.upsert_scheme(Scheme("act:forage", {"feat:food": 1.0}, "forage", {"energy": +0.05}, utility=0.0))
    proc.upsert_scheme(Scheme("act:investigate", {"feat:shiny": 1.0}, "investigate", {}, utility=0.0))
    for a in ("up", "down", "left", "right"):
        proc.upsert_scheme(Scheme(f"act:move:{a}", {"bias:any": 0.1}, a, {}, utility=0.0))

    grid = Gridworld()
    symb = SymbolicForagingEnv(["FOOD", "SHINY", "FOOD", "SHINY"])  # default

    if args.snapshot_prefix:
        store = SQLiteStore(Path(f"{args.snapshot_prefix}_pre.db"))
        store.save_all(pam, list(proc.iter_schemes()), tem, em)
        store.close()

    async def run_episode(ep_idx: int, cycles: int) -> None:
        nonlocal pam, workspace, gw, novelty, attn_learner, tem, em, proc, bn, td, drives
        workspace.clear()
        drives.energy.level = 0.6
        drives.safety.level = 0.9
        if args.env == "symbolic":
            symb.reset()
        else:
            grid.reset()

        engine = CognitiveCycleEngine(CycleConfig(hz=args.hz, understanding_budget_ms=40, attention_budget_ms=30, action_budget_ms=30, log_fn=_print, trace_dir=trace_path))
        engine.trace({"type": "episode", "event": "start", "idx": ep_idx, "cycle": engine._cycle_index})
        ingest_shared = {"seeds": {}}

        def _ingest_action():
            # Feelings activation
            pam.feelings_from_map({
                "stress": max(0.0, 1.0 - drives.safety.level),
                "hunger": max(0.0, 1.0 - drives.energy.level),
            })
            if args.env == "grid":
                seeds = ingest_grid_neighborhood(pam, workspace, grid._tile_token, grid.x, grid.y)
            else:
                tok = symb.observe()
                seeds = ingest_observation(pam, workspace, [(tok, 1.0), ("bias:any", 0.5)])
            ingest_shared["seeds"] = seeds

        ingest_c = Codelet(name="ingest_env", urgency=2.0, action=_ingest_action, kind="perceptual", metadata={"shared": ingest_shared})
        shared = ingest_c.metadata["shared"]
        spread_c = make_spread_codelet(pam, shared)
        relate_c = make_relations_codelet(workspace)
        tracker = CategoryInductionTracker()
        induce_c = make_category_induction_codelet(tracker, pam, workspace)
        promote_c = make_category_promotion_codelet(pam, workspace)
        attn_codelets = make_attention_seed_codelets(workspace)

        conscious_payload: dict[str, float] | None = None
        last_features: tuple[tuple[str, float], ...] | None = None

        async def understanding() -> None:
            ingest_c.run()
            spread_c.run()
            relate_c.run()
            induce_c.run()
            promote_c.run()
            feats = {nid: act for nid, act in pam.top_k_active(6)}
            relations_snapshot = list(workspace.relations)
            workspace.clear()
            for nid, act in feats.items():
                workspace.upsert_object(nid, features={nid: act})
            workspace.relations.extend(relations_snapshot)

        async def attention() -> None:
            nonlocal conscious_payload, last_features
            hunger = drives.hunger_boost()
            safety_bias = drives.safety_boost()
            sched = CodeletScheduler()
            for c in attn_codelets:
                sched.push(c)
            max_runs = 2 if len(workspace.objects) <= 4 else 1
            for c in sched.pop(max_runs=max_runs):
                c.run()
            proposals = attn_codelets[0].metadata.get("coalitions", [])
            relevance = {}
            for coal in proposals:
                cue = {k: float(v) for k, v in coal.features if ":" in k and not k.startswith("rel:")}
                sim = tem.retrieve_similarity(cue, top_n=3) + em.retrieve_similarity(cue, top_n=3)
                base_rel = 0.0
                if coal.id == "feat:food" or any(k == "subj:feat:food" or k == "obj:feat:food" for k, _ in coal.features):
                    base_rel = max(0.0, hunger - 1.0)
                if coal.id == "feat:hazard" or any("feat:hazard" in k for k, _ in coal.features):
                    base_rel -= max(0.0, safety_bias - 1.0)
                boost = attn_learner.compute_boost(coal.features)
                relevance[coal.id] = base_rel + sim + boost
            winner, _ = gw.compete(proposals, relevance=relevance)
            cc = gw.broadcast(engine._cycle_index, winner)
            conscious_payload = cc.payload if cc else None
            last_features = tuple(winner.features) if winner else None
            utils = {"forage": 0.0, "investigate": 0.0, "move": 0.0}
            counts = {"forage": 0, "investigate": 0, "move": 0}
            for sc in proc.iter_schemes():
                if sc.action_name in ("forage", "investigate"):
                    utils[sc.action_name] += sc.utility
                    counts[sc.action_name] += 1
                elif sc.action_name in ("up", "down", "left", "right"):
                    utils["move"] += sc.utility
                    counts["move"] += 1
            avg_utils = {k: (utils[k] / counts[k] if counts[k] else 0.0) for k in utils}
            engine.trace({"type": "attention", "cycle": engine._cycle_index, "coalitions": [{"id": c.id, "salience": c.salience, "relevance": relevance.get(c.id, 0.0)} for c in proposals], "winner": winner.id if winner else None, "avg_utils": avg_utils})

        async def action() -> None:
            bn.tick()
            if conscious_payload is not None:
                from lida.core.events import ConsciousContent
                tags = {"conscious"}
                if (1.0 - drives.safety.level) > 0.3:
                    tags.add("feeling:stress")
                if (1.0 - drives.energy.level) > 0.3:
                    tags.add("feeling:hunger")
                tem.encode(ConsciousContent(cycle_index=engine._cycle_index, tags=frozenset(tags), payload=conscious_payload))
            tem.step_decay()
            em.consolidate_from(tem)

            reward = 0.0
            tile = None
            if not conscious_payload:
                drives.step_decay()
                for sc in proc.iter_schemes():
                    td.decay(sc, factor=0.995)
                return None
            cands = proc.instantiate(conscious_payload)
            winner = bn.select(cands)
            if winner:
                action_name = winner.scheme.action_name
                if action_name == "forage":
                    drives.apply_effect("energy", winner.scheme.expected_effects.get("energy", 0.0))
                if args.env == "grid":
                    reward = grid.step(action_name)
                    tile = grid.observe()
                    if tile == "HAZARD":
                        drives.apply_effect("safety", -0.05)
                    else:
                        drives.apply_effect("safety", +0.005)
                else:
                    reward = symb.evaluate(action_name)
                affect_alpha = 0.1 + 0.2 * max(1.0 - drives.safety.level, 1.0 - drives.energy.level)
                td.update(winner.scheme, reward, alpha=affect_alpha)
                attn_learner.update(last_features or tuple(), reward)
                bn.on_selected(winner.scheme.scheme_id)
                engine.trace({"type": "action", "cycle": engine._cycle_index, "action": action_name, "reward": reward, "utility": winner.scheme.utility, "tile": tile})
            drives.step_decay()
            for sc in proc.iter_schemes():
                td.decay(sc, factor=0.995)
            return None

        engine.on_understanding(understanding)
        engine.on_attention(attention)
        engine.on_action(action)
        await engine.run(max_cycles=cycles)
        engine.trace({"type": "episode", "event": "end", "idx": ep_idx, "cycle": engine._cycle_index})

    for e in range(args.episodes):
        asyncio.run(run_episode(e, cycles=50))

    if args.snapshot_prefix:
        store = SQLiteStore(Path(f"{args.snapshot_prefix}_post.db"))
        store.save_all(pam, list(proc.iter_schemes()), tem, em)
        store.close()

    pam_dot = write_pam_dot(pam, Path(args.trace_dir) / "report" / "pam.dot")
    ws_dot = write_workspace_dot(workspace, Path(args.trace_dir) / "report" / "workspace.dot")

    if args.report:
        traces = load_traces(Path(args.trace_dir))
        metrics = compute_metrics(traces)
        report_path = write_markdown(Path(args.trace_dir) / "report", metrics, pam_dot_path=pam_dot)
        print(f"Report written to {report_path}; workspace DOT at {ws_dot}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="lida")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="Run the LIDA cycle engine")
    sub.add_parser("snapshot", help="Save or load snapshots")
    sub.add_parser("demo", help="Run multi-episode demo and optionally write report")

    args, rest = parser.parse_known_args(argv)
    if args.cmd == "run":
        run_cmd(rest)
    elif args.cmd == "snapshot":
        snapshot_cmd(rest)
    elif args.cmd == "demo":
        demo_cmd(rest)


if __name__ == "__main__":
    main()
