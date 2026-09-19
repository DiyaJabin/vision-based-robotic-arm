"""Dependency-independent bounded observe/select/execute orchestration.

Adapters provide scene/perception and robot operations. Outcomes are returned
as Python data, never written as experiment logs.
"""
from dataclasses import dataclass
from core.contracts import RunOutcome


@dataclass(frozen=True)
class LoopConfig:
    max_observations: int = 12
    max_pick_attempts: int = 6

    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in vars(self).values()):
            raise ValueError("Loop limits must be positive integers.")


def run_loop(adapter, config: LoopConfig | None = None) -> RunOutcome:
    """Adapter: prepare_observation, observe_and_rank, execute, remaining_ids, hold.

    A physical execution failure stops safely; perception rejection may select
    another candidate in the same observation. Empty candidates are not success
    while unhandled objects remain. No unbounded retries or hidden fallback.
    """
    cfg=config or LoopConfig()
    picks=[]
    rejections=[]
    observations=0
    status,reason="stopped","observation_limit"
    try:
        for _ in range(cfg.max_observations):
            adapter.prepare_observation()
            ranked,rejected=adapter.observe_and_rank()
            observations+=1
            rejections.extend(rejected)
            remaining=tuple(adapter.remaining_ids())
            if not remaining:
                status,reason="completed","all_objects_placed"
                break
            if not ranked:
                status,reason="stopped","no_valid_targets"
                break
            if len(picks)>=cfg.max_pick_attempts:
                status,reason="stopped","pick_attempt_limit"
                break
            outcome=adapter.execute(ranked[0])
            picks.append(outcome)
            if outcome.status!="placed":
                status,reason="failed",outcome.reason
                break
    except Exception as error:
        status,reason="failed",f"{type(error).__name__}: {error}"
    finally:
        try:
            adapter.hold()
        except Exception as error:
            status,reason="failed",f"hold_failed:{type(error).__name__}: {error}"
    return RunOutcome(status,reason,tuple(picks),observations,
                      tuple(adapter.remaining_ids()),tuple(rejections))
