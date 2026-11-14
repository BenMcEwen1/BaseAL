"""Tests for the package metadata and the API server entry point."""
import re
from pathlib import Path

import pytest

import baseal

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_version_is_exposed_at_package_top_level():
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", baseal.__version__)


def test_installed_metadata_matches_package_version():
    # pyproject.toml reads the version dynamically from baseal.__version__.
    from importlib.metadata import version
    assert version("BaseAL") == baseal.__version__


def test_web_app_reads_the_same_version_string():
    # app/vite.config.js parses __version__ out of baseal/__init__.py with this regex.
    init_src = (REPO_ROOT / "baseal" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_src, re.MULTILINE)
    assert match and match.group(1) == baseal.__version__


def test_api_reports_package_version_and_project_root(tmp_path):
    pytest.importorskip("fastapi", reason="needs the 'api' extra")
    from baseal.api import main as api

    assert api.info()["version"] == baseal.__version__

    original = api.BASE_DIR
    try:
        api.set_project_root(tmp_path)
        assert api.BASE_DIR == tmp_path.resolve()
        assert api.EMBEDDINGS_BASE_PATH == tmp_path.resolve() / "ESC10_BASEAL" / "embeddings"
    finally:
        api.set_project_root(original)


def test_sampling_import_does_not_load_torch():
    # Lets projects that only need the sampling strategies use them without
    # paying for PyTorch and the UMAP warm-up at import time.
    import subprocess
    import sys
    code = ("import sys; from baseal.utils.sampling import SamplingStrategy, ValidationStrategy; "
            "print('torch' in sys.modules, 'umap' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"
