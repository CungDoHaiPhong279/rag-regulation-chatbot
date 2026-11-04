import math
from pathlib import Path
import subprocess
import sys

import pytest

from tests.fakes import FailingEmbedder, FakeEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_fake_embedder_deterministic_and_normalized():
    emb = FakeEmbedder()
    a1, a2 = emb.embed_dense(["Học phí"]), emb.embed_dense(["Học phí"])
    assert a1 == a2 and len(a1[0]) == 8
    assert abs(math.sqrt(sum(x * x for x in a1[0])) - 1.0) < 1e-6
    assert emb.embed_dense(["Tín chỉ"])[0] != a1[0]
    assert emb.calls == 3


def test_failing_embedder_raises():
    with pytest.raises(RuntimeError, match="embed failed"):
        FailingEmbedder().embed_dense(["x"])


def test_importing_embedder_does_not_load_torch():
    code = "import src.ingestion.embedder, sys; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


@pytest.mark.slow
def test_bge_m3_real_dimension():
    from src.ingestion.embedder import BgeM3Embedder

    emb = BgeM3Embedder()
    vecs = emb.embed_dense(["Học phí"])
    assert len(vecs) == 1 and len(vecs[0]) == emb.dense_dim == 1024
    assert abs(math.sqrt(sum(x * x for x in vecs[0])) - 1.0) < 1e-3
