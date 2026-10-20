"""Freeze new sources and transitive baseline/metric dependencies before training."""
import hashlib
import json
import subprocess
import bridge


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_state():
    if subprocess.check_output(['git','status','--porcelain'], cwd=bridge.ROOT, text=True).strip():
        raise RuntimeError('Clean committed worktree required')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=bridge.ROOT,text=True).strip()


def sources():
    return {str(p):sha(p) for p in sorted(bridge.HERE.iterdir()) if p.suffix in ('.py','.json')}


def create():
    code_state()
    if (bridge.OUT/'SOURCE_FREEZE.json').exists():
        return verify()
    bridge.atomic_json(bridge.OUT/'SOURCE_FREEZE.json',dict(commit=code_state(), source=sources(),
        old_metric_freeze=sha(bridge.old_environment.OUT/'METRIC_FREEZE.json'),
        metric_policy='append-only regional extension; historical metrics unchanged',
        parameters_frozen_before_training=True))
    return verify()


def verify():
    bridge.old_freeze.verify()
    frozen=json.loads((bridge.OUT/'SOURCE_FREEZE.json').read_text())
    assert frozen['source']==sources(), 'Sources changed after pre-registration'
    assert frozen['old_metric_freeze']==sha(bridge.old_environment.OUT/'METRIC_FREEZE.json')
    assert frozen['commit']==code_state()
    return frozen
