from __future__ import annotations

import pytest


PRO_REPOS = [
    "NodeBB/NodeBB",
    "ansible/ansible",
    "element-hq/element-web",
    "flipt-io/flipt",
    "future-architect/vuls",
    "gravitational/teleport",
    "internetarchive/openlibrary",
    "navidrome/navidrome",
    "protonmail/webclients",
    "qutebrowser/qutebrowser",
    "tutao/tutanota",
]


def _selected_tests(repo: str) -> str:
    if repo in {
        "flipt-io/flipt",
        "future-architect/vuls",
        "gravitational/teleport",
        "navidrome/navidrome",
    }:
        return '["TestParser", "TestHandler"]'
    if repo in {"ansible/ansible", "internetarchive/openlibrary", "qutebrowser/qutebrowser"}:
        return '["tests/test_example.py"]'
    return '["test/example.test.ts"]'


def _make_pro_instance(repo: str) -> dict:
    return {
        "instance_id": f"instance_{repo.replace('/', '__')}-dummy",
        "repo": repo,
        "base_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "patch": "",
        "test_patch": "",
        "problem_statement": "Dummy task",
        "fail_to_pass": "['TestParser']",
        "pass_to_pass": "[]",
        "selected_test_files_to_run": _selected_tests(repo),
    }


@pytest.mark.parametrize("repo", PRO_REPOS)
def test_swebench_pro_repo_specs_do_not_raise(repo: str) -> None:
    pytest.importorskip("swebench")
    from swebench.harness.test_spec.test_spec import make_test_spec

    spec = make_test_spec(_make_pro_instance(repo), namespace="none")
    assert spec.repo == repo
    assert spec.version == "default"
