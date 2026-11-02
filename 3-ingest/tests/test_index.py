import numpy as np
import scipy.sparse as sp

from ingest import dense, sparse


def shard(tmp_path, name, ids, rows, n_cols=50):
    """rows: list of {col: weight}; written as a CSR shard like `rag embed` does."""
    m = sp.lil_matrix((len(rows), n_cols), dtype=np.float32)
    for i, r in enumerate(rows):
        for c, w in r.items():
            m[i, c] = w
    return sparse.save_shard(str(tmp_path / name), ids, m.tocsr())


def test_sparse_build_and_search(tmp_path):
    a = shard(tmp_path, "a", ["c1", "c2"], [{1: 0.5, 2: 0.2}, {3: 0.9}])
    b = shard(tmp_path, "b", ["c3"], [{1: 0.1, 3: 0.1, 4: 0.7}])
    sparse.build([a, b], str(tmp_path / "idx"), n_cols=50)
    idx = sparse.load(str(tmp_path / "idx"))
    ids, scores = idx.search([{1: 1.0}, {3: 1.0, 4: 1.0}, {9: 1.0}], k=2)
    assert list(ids[0]) == ["c1", "c3"] and np.allclose(scores[0], [0.5, 0.1])
    assert list(ids[1]) == ["c2", "c3"] and np.allclose(scores[1], [0.9, 0.8])
    assert np.all(scores[2] == 0)
    assert idx.params["n_docs"] == 3


def test_sparse_query_weight_scales_score(tmp_path):
    a = shard(tmp_path, "a", ["c1"], [{1: 0.5}])
    sparse.build([a], str(tmp_path / "idx"), n_cols=50)
    _, scores = sparse.load(str(tmp_path / "idx")).search([{1: 2.0}], k=1)
    assert np.isclose(scores[0][0], 1.0)


def test_dense_search_across_shards(tmp_path):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(7, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    p1 = dense.save_shard(str(tmp_path / "p1"), ["a", "b", "c", "d"], vecs[:4])
    p2 = dense.save_shard(str(tmp_path / "p2"), ["e", "f", "g"], vecs[4:])
    ids, scores = dense.search(vecs[[5, 1]], [p1, p2], k=3)
    assert ids[0][0] == "f" and ids[1][0] == "b"
    assert np.isclose(scores[0][0], 1.0, atol=1e-2) and scores[0][0] >= scores[0][1] >= scores[0][2]


def test_dense_search_blocks_and_k_larger_than_shard(tmp_path):
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(9, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    shards = [dense.save_shard(str(tmp_path / f"p{i}"), [f"x{j}" for j in range(i * 3, i * 3 + 3)], vecs[i * 3:i * 3 + 3]) for i in range(3)]
    ids, scores = dense.search(vecs, shards, k=5, block=2)  # k > shard size, several query blocks
    assert [i[0] for i in ids] == [f"x{j}" for j in range(9)]
    assert scores.shape == (9, 5) and np.all(np.diff(scores, axis=1) <= 1e-6)


def test_sparse_ignores_out_of_vocab_token(tmp_path):
    a = shard(tmp_path, "a", ["c1"], [{1: 0.5}])
    sparse.build([a], str(tmp_path / "idx"), n_cols=50)
    _, scores = sparse.load(str(tmp_path / "idx")).search([{1: 1.0, 999: 3.0}], k=1)
    assert np.isclose(scores[0][0], 0.5)
