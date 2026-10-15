from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from src.docker_agents import NVM_NODE_INSTALL


@pytest.mark.parametrize("prefix_state", ["set", "empty", "unset"])
def test_node_install_unsets_prefix_until_install_finishes_and_restores_it(tmp_path, prefix_state):
    """Exercise the generated shell without downloading or installing anything."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("bash", "mkdir", "chmod"):
        executable = shutil.which(name)
        assert executable is not None
        (binaries / name).symlink_to(executable)
    nvm_dir = tmp_path / "nvm"
    nvm_dir.mkdir()
    observed = tmp_path / "events"

    for name, body in {"apt-get": "exit 0", "curl": "printf ':\\n'"}.items():
        executable = binaries / name
        executable.write_text("#!/bin/sh\n" + body + "\n")
        executable.chmod(0o755)
    (nvm_dir / "nvm.sh").write_text(r'''
printf 'source:%s\n' "${NPM_CONFIG_PREFIX+x}" >> "$NVM_TEST_EVENTS"
nvm() {
    printf 'install:%s\n' "${NPM_CONFIG_PREFIX+x}" >> "$NVM_TEST_EVENTS"
    [ "$1" = install ] && [ "$2" = 22 ] || return 12
    [ "${NPM_CONFIG_PREFIX+x}" != x ] || return 11
    printf '#!/bin/sh\necho v22.22.2\n' > "$NVM_TEST_BIN/node"
    chmod +x "$NVM_TEST_BIN/node"
}
''')
    env = dict(os.environ)
    env.update(NVM_DIR=str(nvm_dir), NVM_TEST_EVENTS=str(observed), NVM_TEST_BIN=str(binaries),
               PATH=str(binaries))
    env.pop("NPM_CONFIG_PREFIX", None)
    if prefix_state != "unset":
        env["NPM_CONFIG_PREFIX"] = str(tmp_path / "npm-global") if prefix_state == "set" else ""
    script = NVM_NODE_INSTALL + r'''
printf 'after:%s:%s\n' "${NPM_CONFIG_PREFIX+x}" "${NPM_CONFIG_PREFIX-}" >> "$NVM_TEST_EVENTS"
test "${NPM_CONFIG_PREFIX_SAVED+x}" != x
test "${NPM_CONFIG_PREFIX_WAS_SET+x}" != x
'''
    result = subprocess.run(["/bin/bash", "-c", script], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    restored = "after::" if prefix_state == "unset" else f"after:x:{env['NPM_CONFIG_PREFIX']}"
    assert observed.read_text().splitlines() == ["source:", "source:", "install:", restored]
