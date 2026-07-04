"""MISC: MIE-guided Inference-time Self-Correction (idea.md section 2).

Implements the full family so the whole experiment matrix comes from one class:
  - method="misc"        : structured (typed) routing + accept/rollback loop
  - method="best_of_n"   : scalar baseline (same MIE, uses only total score)
  - method="one_shot"    : single generation (B=1)
  - method="caption_upsample" : Best-of-N over FLUX.2 native prompt upsampling

Compute is aligned by the generation budget B (number of generator calls).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import config


@dataclass
class Trace:
    task_id: str
    method: str
    routing: str
    action_mode: str
    seed_mode: str
    budget: int
    accepted_steps: int = 0
    rejected_steps: int = 0
    gen_calls: int = 0
    init_total: float = 0.0
    final_total: float = 0.0
    final_dims: dict = field(default_factory=dict)
    final_prompt: str = ""
    collateral_events: int = 0        # steps where a non-target dim dropped
    step_log: list = field(default_factory=list)

    @property
    def collateral_damage_rate(self) -> float:
        steps = self.accepted_steps + self.rejected_steps
        return (self.collateral_events / steps) if steps else 0.0


class MISCPipeline:
    def __init__(self, critic, generator, *,
                 method="misc",
                 routing=config.ROUTING_DEFAULT,
                 action_mode=config.ACTION_MODE_DEFAULT,
                 seed_mode=config.SEED_MODE_DEFAULT,
                 n_init=config.N_INIT,
                 k_steps=config.K_STEPS,
                 budget: Optional[int] = None):
        self.critic = critic
        self.gen = generator
        self.method = method
        self.routing = routing
        self.action_mode = action_mode
        self.seed_mode = seed_mode
        self.n_init = n_init
        self.k_steps = k_steps
        # Aligned compute budget (generator calls). Default: N + K.
        self.budget = budget if budget is not None else (n_init + k_steps)
        # Fixability estimate (prior; optionally updated online across tasks).
        self.fixability = dict(config.FIXABILITY)

    # ------------------------------------------------------------------
    def _score(self, image, prompt, refs, names):
        return self.critic.diagnose(image, prompt, refs, names)

    def _route(self, diag, rng) -> str:
        if self.routing == "random":
            return rng.choice(config.DIMS)
        if self.routing == "static":
            return config.PRIORITY[0]
        if self.routing == "diagnostic":
            # raw argmin over absolute scores. Because E>A>I in practice, this
            # degenerates to almost-always Interaction (the hardest dim). Kept as
            # the negative ablation that motivates calibrated routing.
            return min(config.DIMS, key=lambda k: (diag[k], config.PRIORITY.index(k)))
        # calibrated (default): shortfall vs each dim's OWN baseline x fixability.
        priority = {
            k: max(0.0, config.MIE_REF[k] - diag[k]) * self.fixability[k]
            for k in config.DIMS
        }
        if max(priority.values()) <= 0.0:
            # nothing is anomalously low -> fall back to weakest-by-shortfall
            return max(config.DIMS, key=lambda k: (config.MIE_REF[k] - diag[k],
                                                    -config.PRIORITY.index(k)))
        return max(config.DIMS, key=lambda k: (priority[k], -config.PRIORITY.index(k)))

    def _update_fixability(self, dim, gain):
        if not config.ONLINE_GAIN:
            return
        lr = config.ONLINE_GAIN_LR
        # normalize gain into ~[0,1] range (gains are small score deltas)
        g = max(0.0, min(1.0, gain * 5.0))
        self.fixability[dim] = (1 - lr) * self.fixability[dim] + lr * g

    def _seed_for(self, base, t, stuck=0):
        # resampled mode always jitters; otherwise jitter only once knobs kick in
        if self.seed_mode == "resampled":
            return base + 1000 * (t + 1)
        if config.KNOB_ESCALATION and stuck >= config.STUCK_TO_KNOB:
            return base + 777 * stuck
        return base

    def _guidance_for(self, stuck=0):
        """Escalate FLUX.2 guidance when prompt edits on a dim keep stalling."""
        if not config.KNOB_ESCALATION or stuck < config.STUCK_TO_KNOB:
            return None  # use generator default
        bump = config.GUIDANCE_STEP * (stuck - config.STUCK_TO_KNOB + 1)
        return min(config.GUIDANCE_MAX, config.GUIDANCE_SCALE + bump)

    # ------------------------------------------------------------------
    def run(self, task) -> tuple:
        """Return (best_image, Trace). `task` is data.Task."""
        rng = random.Random(hash(task.task_id) & 0xFFFFFFFF)
        refs = task.load_refs()
        refs_by_subject = task.refs_by_subject()
        names = task.subject_names
        p0 = task.prompt

        tr = Trace(task_id=task.task_id, method=self.method, routing=self.routing,
                   action_mode=self.action_mode, seed_mode=self.seed_mode,
                   budget=self.budget)

        if self.method == "one_shot":
            img = self.gen.generate(p0, refs, seed=config.BASE_SEED); tr.gen_calls += 1
            diag = self._score(img, p0, refs, names)
            return self._finish(img, p0, diag, tr, best_init=diag["total"])

        if self.method == "caption_upsample":
            return self._best_of_n(task, refs, refs_by_subject, names, p0, tr,
                                   upsample=True)

        if self.method == "best_of_n":
            return self._best_of_n(task, refs, refs_by_subject, names, p0, tr,
                                   upsample=False)

        return self._misc(task, refs, refs_by_subject, names, p0, tr, rng)

    # ------------------------------------------------------------------
    def _best_of_n(self, task, refs, refs_by_subject, names, p0, tr, upsample):
        best_img = best_diag = None
        for i in range(self.budget):
            temp = config.CAPTION_UPSAMPLE_TEMPERATURE if upsample else None
            img = self.gen.generate(p0, refs, seed=config.BASE_SEED + i,
                                    caption_upsample_temperature=temp)
            tr.gen_calls += 1
            diag = self._score(img, p0, refs, names)
            if best_diag is None or diag["total"] > best_diag["total"]:
                best_img, best_diag = img, diag
            if i == 0:
                tr.init_total = diag["total"]
        return self._finish(best_img, p0, best_diag, tr, best_init=tr.init_total)

    # ------------------------------------------------------------------
    def _misc(self, task, refs, refs_by_subject, names, p0, tr, rng):
        import actions

        # --- init: Best-of-N over N seeds, pick top total ---
        state = {"prompt": p0, "ref_images": refs}
        best_img = best_diag = None
        for i in range(self.n_init):
            img = self.gen.generate(p0, refs, seed=config.BASE_SEED + i)
            tr.gen_calls += 1
            diag = self._score(img, p0, refs, names)
            if best_diag is None or diag["total"] > best_diag["total"]:
                best_img, best_diag, state["ref_images"] = img, diag, refs
        tr.init_total = best_diag["total"]

        cur_img, cur_diag = best_img, best_diag
        traj_best = (cur_img, cur_diag, dict(state))  # global-best over trajectory
        escalation = {k: 0 for k in config.DIMS}      # per-dim unaccepted streak

        # --- correction loop ---
        remaining = self.budget - self.n_init
        for t in range(min(self.k_steps, max(0, remaining))):
            # Stop when all E/A/I quality scores clear the gate. NOTE: we do NOT
            # threshold `total` -- the MIE preference score is unbounded and only
            # meaningful for comparison, so an absolute stop must use the [0,1]
            # category scores.
            if min(cur_diag[k] for k in config.DIMS) >= config.THETA_GATE:
                break
            weak = self._route(cur_diag, rng)
            stuck = escalation[weak]
            # graded intensity from the score DEFICIT (+escalation) -- uses the
            # magnitude of the weakness, not just which dim is weakest.
            level = actions.level_from_score(cur_diag[weak], stuck)
            cand_state = actions.apply_action(
                state, weak, level, self.action_mode, rng,
                original_prompt=p0, refs_by_subject=refs_by_subject,
                subject_names=names, step=t, diag=cur_diag,
            )
            # when a dim keeps stalling, escalate FLUX.2-native knobs (guidance + seed)
            cand_img = self.gen.generate(
                cand_state["prompt"], cand_state.get("ref_images", refs),
                seed=self._seed_for(config.BASE_SEED, t, stuck),
                guidance=self._guidance_for(stuck),
            )
            tr.gen_calls += 1
            cand_diag = self._score(cand_img, cand_state["prompt"],
                                    cand_state.get("ref_images", refs), names)

            self._update_fixability(weak, cand_diag["total"] - cur_diag["total"])
            improved = cand_diag["total"] > cur_diag["total"] + config.EPS_ACCEPT
            no_regress = all(
                cand_diag[k] >= cur_diag[k] - config.DELTA_FLOOR
                for k in config.DIMS if k != weak
            )
            # collateral damage: any non-target dim dropped meaningfully
            if any(cand_diag[k] < cur_diag[k] - 1e-6 for k in config.DIMS if k != weak):
                tr.collateral_events += 1

            tr.step_log.append({
                "t": t, "weak": weak, "level": level,
                "prompt": cand_state["prompt"],
                "before": cur_diag["total"], "after": cand_diag["total"],
                "accepted": bool(improved and no_regress),
            })

            if improved and no_regress:
                cur_img, cur_diag, state = cand_img, cand_diag, cand_state
                tr.accepted_steps += 1
                escalation[weak] = 0  # progress -> reset intensity escalation
                if cur_diag["total"] > traj_best[1]["total"]:
                    traj_best = (cur_img, cur_diag, dict(state))
            else:
                tr.rejected_steps += 1               # rollback: keep cur state
                escalation[weak] += 1                # no progress -> escalate next time

        # --- constrained-optimal output over the whole trajectory ---
        img, diag, st = traj_best
        return self._finish(img, st["prompt"], diag, tr, best_init=tr.init_total)

    # ------------------------------------------------------------------
    def _finish(self, img, prompt, diag, tr, best_init):
        tr.init_total = tr.init_total or best_init
        tr.final_total = diag["total"]
        tr.final_dims = {k: diag[k] for k in config.DIMS}
        tr.final_prompt = prompt
        return img, tr
