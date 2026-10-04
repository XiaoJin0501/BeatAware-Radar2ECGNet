from scripts import run_dual_head_interaction as core


def test_harmonic_checkpoint_rule_rejects_immature_event_head():
    old=core.CHECKPOINT_RULE
    try:
        core.CHECKPOINT_RULE="shape_event_hmean"
        immature={"shape":{"pcc":.94,"rmse":.08},"independent":{"macro_f1_100ms":0.0}}
        mature={"shape":{"pcc":.88,"rmse":.10},"independent":{"macro_f1_100ms":.75}}
        assert core.rank(mature,"A1") > core.rank(immature,"A1")
    finally:
        core.CHECKPOINT_RULE=old
