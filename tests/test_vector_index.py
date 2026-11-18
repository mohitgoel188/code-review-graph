"""The numpy similarity scan must rank exactly as the pure-Python scan does."""

import random
import sqlite3
import sys
from unittest.mock import patch

import pytest

from code_review_graph import vector_index
from code_review_graph.embeddings import (
    EmbeddingStore,
    QueryEmbedding,
    _encode_vector,
)
from code_review_graph.graph import GraphStore
from code_review_graph.search import hybrid_search

np = pytest.importorskip("numpy")

PROVIDER = "test:provider"
SCORE_TOLERANCE = 1e-12


class Provider:
    """Embeds every query as one fixed vector and counts the calls."""

    name = PROVIDER

    def __init__(self, query_vec):
        self.query_vec = list(query_vec)
        self.queries = []

    def embed(self, texts):
        return [list(self.query_vec) for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return list(self.query_vec)

    @property
    def dimension(self):
        return len(self.query_vec)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.delenv("CRG_NUMPY_SEARCH", raising=False)
    monkeypatch.delenv("CRG_VECTOR_CACHE_MB", raising=False)
    vector_index.clear_cache()
    yield
    vector_index.clear_cache()


def _open(db, query_vec):
    with patch("code_review_graph.embeddings.get_provider", return_value=Provider(query_vec)):
        return EmbeddingStore(db)


def _insert(store, vectors, provider=PROVIDER, prefix="node"):
    start = store.count()
    for offset, vec in enumerate(vectors):
        store._conn.execute(
            "INSERT OR REPLACE INTO embeddings (qualified_name, vector, text_hash, provider) "
            "VALUES (?, ?, ?, ?)",
            (f"{prefix}{start + offset}", _encode_vector(vec), "hash", provider),
        )
    store._conn.commit()


def _random_vectors(rng, rows, dimension):
    return [[rng.uniform(-1.0, 1.0) for _ in range(dimension)] for _ in range(rows)]


def _both(store, limit):
    """Rank once through numpy and once through the reference scan."""
    fast = store.search("query", limit=limit)
    reference = store._search_python(store.provider.query_vec, PROVIDER, limit)
    return fast, reference


def _assert_same_ranking(fast, reference):
    assert [name for name, _ in fast] == [name for name, _ in reference]
    for (_, fast_score), (_, reference_score) in zip(fast, reference):
        assert fast_score == pytest.approx(reference_score, abs=SCORE_TOLERANCE)


def _used_numpy(store):
    return vector_index.load_matrix(store._conn, store.db_path, PROVIDER) is not None


class TestRankingMatchesPurePython:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("limit", [1, 7, 60, 500])
    def test_random_stores(self, tmp_path, seed, limit):
        rng = random.Random(seed)
        dimension = rng.choice([3, 16, 64, 384])
        store = _open(tmp_path / "e.db", _random_vectors(rng, 1, dimension)[0])
        _insert(store, _random_vectors(rng, rng.choice([1, 5, 200, 700]), dimension))

        assert _used_numpy(store)
        _assert_same_ranking(*_both(store, limit))
        store.close()

    def test_duplicate_vectors_keep_table_order(self, tmp_path):
        rng = random.Random(1)
        base = _random_vectors(rng, 40, 32)
        # Every vector is stored three times, interleaved with the others.
        store = _open(tmp_path / "e.db", _random_vectors(rng, 1, 32)[0])
        _insert(store, base + base[::-1] + base)

        fast, reference = _both(store, 25)
        assert fast == reference  # tied rows are scored by the reference cosine itself
        store.close()

    def test_scaled_copies_and_zero_vectors(self, tmp_path):
        rng = random.Random(2)
        base = _random_vectors(rng, 30, 48)
        scaled = [[component * factor for component in vec] for vec in base
                  for factor in (0.5, 2.0, 3.0)]
        zeros = [[0.0] * 48 for _ in range(5)]
        store = _open(tmp_path / "e.db", _random_vectors(rng, 1, 48)[0])
        _insert(store, zeros + base + scaled + zeros)

        for limit in (4, 33, 1000):
            fast, reference = _both(store, limit)
            _assert_same_ranking(fast, reference)
        store.close()

    def test_all_rows_tied(self, tmp_path):
        store = _open(tmp_path / "e.db", [1.0, 0.0, 0.0])
        _insert(store, [[0.0, 1.0, 0.0]] * 12)

        fast, reference = _both(store, 5)
        assert fast == reference
        assert [name for name, _ in fast] == [f"node{i}" for i in range(5)]
        store.close()

    def test_vectors_one_float_step_apart(self, tmp_path):
        rng = random.Random(3)
        base = _random_vectors(rng, 1, 64)[0]
        nudged = []
        for index in range(60):
            vec = np.array(base, dtype=np.float32)
            vec[index] = np.nextafter(vec[index], np.float32(2.0))
            nudged.append(vec.tolist())
        store = _open(tmp_path / "e.db", _random_vectors(rng, 1, 64)[0])
        _insert(store, nudged + _random_vectors(rng, 100, 64))

        fast, reference = _both(store, 20)
        assert [name for name, _ in fast] == [name for name, _ in reference]
        store.close()

    def test_empty_store(self, tmp_path):
        store = _open(tmp_path / "e.db", [1.0, 2.0])
        assert store.search("query") == []
        assert store._search_python([1.0, 2.0], PROVIDER, 20) == []
        store.close()

    def test_dimension_mismatch_scores_zero_in_table_order(self, tmp_path):
        rng = random.Random(4)
        store = _open(tmp_path / "e.db", [1.0, 2.0, 3.0])
        _insert(store, _random_vectors(rng, 9, 8))

        fast, reference = _both(store, 4)
        assert fast == reference
        assert fast == [(f"node{i}", 0.0) for i in range(4)]
        store.close()

    def test_zero_query_vector(self, tmp_path):
        rng = random.Random(5)
        store = _open(tmp_path / "e.db", [0.0] * 8)
        _insert(store, _random_vectors(rng, 9, 8))

        fast, reference = _both(store, 4)
        assert fast == reference
        store.close()

    def test_only_the_current_provider_is_scanned(self, tmp_path):
        rng = random.Random(6)
        store = _open(tmp_path / "e.db", _random_vectors(rng, 1, 8)[0])
        _insert(store, _random_vectors(rng, 20, 8), provider="other:provider", prefix="other")
        _insert(store, _random_vectors(rng, 20, 8))

        fast, reference = _both(store, 50)
        _assert_same_ranking(fast, reference)
        assert len(fast) == 20
        assert all(name.startswith("node") for name, _ in fast)
        store.close()


class TestFallbackToPurePython:
    def _store(self, tmp_path):
        rng = random.Random(7)
        store = _open(tmp_path / "e.db", _random_vectors(rng, 1, 16)[0])
        _insert(store, _random_vectors(rng, 50, 16))
        return store

    def test_numpy_not_installed(self, tmp_path, monkeypatch):
        store = self._store(tmp_path)
        expected = store.search("query", limit=10)

        monkeypatch.setitem(sys.modules, "numpy", None)  # makes ``import numpy`` fail
        vector_index.clear_cache()
        assert vector_index.load_numpy() is None
        assert not _used_numpy(store)
        fallback = store.search("query", limit=10)

        assert fallback == store._search_python(store.provider.query_vec, PROVIDER, 10)
        _assert_same_ranking(expected, fallback)
        assert vector_index.cache_stats()["entries"] == 0
        store.close()

    def test_disabled_by_environment(self, tmp_path, monkeypatch):
        store = self._store(tmp_path)
        monkeypatch.setenv("CRG_NUMPY_SEARCH", "0")

        assert not _used_numpy(store)
        assert store.search("query", limit=10) == store._search_python(
            store.provider.query_vec, PROVIDER, 10,
        )
        store.close()

    def test_rows_of_differing_sizes(self, tmp_path):
        store = self._store(tmp_path)
        _insert(store, [[1.0, 2.0, 3.0]])

        assert not _used_numpy(store)
        assert store.search("query", limit=10) == store._search_python(
            store.provider.query_vec, PROVIDER, 10,
        )
        store.close()

    def test_non_finite_stored_values(self, tmp_path):
        store = self._store(tmp_path)
        _insert(store, [[float("inf")] + [0.0] * 15])

        assert not _used_numpy(store)
        fast = store.search("query", limit=10)
        reference = store._search_python(store.provider.query_vec, PROVIDER, 10)
        assert [name for name, _ in fast] == [name for name, _ in reference]
        store.close()

    def test_extreme_query_vector(self, tmp_path):
        store = self._store(tmp_path)
        store.provider.query_vec = [1e300] * 16

        matrix = vector_index.load_matrix(store._conn, store.db_path, PROVIDER)
        assert vector_index.top_matches(matrix, store.provider.query_vec, 10, None) is None
        assert store.search("query", limit=10) == store._search_python(
            store.provider.query_vec, PROVIDER, 10,
        )
        store.close()


class TestMatrixCache:
    def _store(self, tmp_path, name="e.db", rows=30):
        rng = random.Random(8)
        store = _open(tmp_path / name, _random_vectors(rng, 1, 16)[0])
        _insert(store, _random_vectors(rng, rows, 16))
        return store

    def test_matrix_is_float32_and_reused(self, tmp_path):
        store = self._store(tmp_path)
        first = vector_index.load_matrix(store._conn, store.db_path, PROVIDER)
        second = vector_index.load_matrix(store._conn, store.db_path, PROVIDER)

        assert first is second
        assert first.vectors.dtype == np.float32
        assert first.vectors.shape == (30, 16)
        assert not first.vectors.flags.writeable
        assert vector_index.cache_stats() == {"entries": 1, "bytes": first.nbytes}
        store.close()

    def test_reused_by_a_new_connection_to_the_same_file(self, tmp_path):
        store = self._store(tmp_path)
        first = vector_index.load_matrix(store._conn, store.db_path, PROVIDER)
        store.close()

        reopened = _open(tmp_path / "e.db", [1.0] * 16)
        assert vector_index.load_matrix(reopened._conn, reopened.db_path, PROVIDER) is first
        reopened.close()

    def test_reused_across_opens_of_a_graph_database(self, tmp_path):
        # A graph database is in WAL mode: every open creates a write-ahead
        # log and the last close removes it, which must not look like a write.
        db = tmp_path / "graph.db"
        GraphStore(str(db)).close()
        rng = random.Random(11)
        seeded = _open(db, [1.0] * 16)
        _insert(seeded, _random_vectors(rng, 20, 16))
        seeded.close()

        matrices = []
        for _ in range(3):
            graph = GraphStore(str(db))
            store = _open(db, [1.0] * 16)
            matrices.append(vector_index.load_matrix(store._conn, store.db_path, PROVIDER))
            store.close()
            graph.close()
        assert matrices[0] is matrices[1] is matrices[2]

    def test_invalidated_by_an_added_row(self, tmp_path):
        store = self._store(tmp_path)
        before = store.search("query", limit=100)
        best = [component * 5.0 for component in store.provider.query_vec]
        _insert(store, [best], prefix="added")

        after = store.search("query", limit=100)
        assert len(after) == len(before) + 1
        assert after[0][0] == "added30"
        _assert_same_ranking(after, store._search_python(store.provider.query_vec, PROVIDER, 100))
        store.close()

    def test_invalidated_by_a_removed_row(self, tmp_path):
        store = self._store(tmp_path)
        top = store.search("query", limit=1)[0][0]
        store.remove_node(top)

        after = store.search("query", limit=100)
        assert top not in [name for name, _ in after]
        _assert_same_ranking(after, store._search_python(store.provider.query_vec, PROVIDER, 100))
        store.close()

    def test_invalidated_by_a_row_rewritten_from_another_connection(self, tmp_path):
        store = self._store(tmp_path)
        last = store.search("query", limit=100)[-1][0]

        other = sqlite3.connect(str(tmp_path / "e.db"))
        best = [component * 2.0 for component in store.provider.query_vec]
        other.execute(
            "UPDATE embeddings SET vector = ? WHERE qualified_name = ?",
            (_encode_vector(best), last),
        )
        other.commit()
        other.close()

        after = store.search("query", limit=100)
        assert after[0][0] == last
        _assert_same_ranking(after, store._search_python(store.provider.query_vec, PROVIDER, 100))
        store.close()

    def test_budget_of_zero_keeps_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRG_VECTOR_CACHE_MB", "0")
        store = self._store(tmp_path)

        _assert_same_ranking(*_both(store, 10))
        assert vector_index.cache_stats() == {"entries": 0, "bytes": 0}
        store.close()

    def test_least_recently_used_store_is_evicted(self, tmp_path, monkeypatch):
        stores = [self._store(tmp_path, name=f"e{i}.db") for i in range(3)]
        size = vector_index.load_matrix(stores[0]._conn, stores[0].db_path, PROVIDER).nbytes
        monkeypatch.setattr(vector_index, "_cache_budget_bytes", lambda: 2 * size)

        matrices = [
            vector_index.load_matrix(store._conn, store.db_path, PROVIDER) for store in stores
        ]
        assert vector_index.cache_stats() == {"entries": 2, "bytes": 2 * size}
        # The first store was pushed out; the other two are still held.
        assert vector_index.load_matrix(
            stores[2]._conn, stores[2].db_path, PROVIDER,
        ) is matrices[2]
        assert vector_index.load_matrix(
            stores[1]._conn, stores[1].db_path, PROVIDER,
        ) is matrices[1]
        assert vector_index.load_matrix(
            stores[0]._conn, stores[0].db_path, PROVIDER,
        ) is not matrices[0]
        for store in stores:
            store.close()

    def test_bad_budget_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("CRG_VECTOR_CACHE_MB", "plenty")
        assert vector_index._cache_budget_bytes() == 1024 * 1024 * 1024


class TestQueryEmbeddedOnce:
    def test_shared_across_stores(self, tmp_path):
        rng = random.Random(9)
        provider = Provider(_random_vectors(rng, 1, 8)[0])
        shared = QueryEmbedding("query")
        with patch("code_review_graph.embeddings.get_provider", return_value=provider):
            stores = [EmbeddingStore(tmp_path / f"e{i}.db") for i in range(4)]
        for store in stores:
            _insert(store, _random_vectors(rng, 12, 8))

        with_shared = [store.search("query", limit=5, query_embedding=shared) for store in stores]
        assert provider.queries == ["query"]

        without = [store.search("query", limit=5) for store in stores]
        assert with_shared == without
        assert len(provider.queries) == 5
        for store in stores:
            store.close()

    def test_ignored_for_a_different_query(self, tmp_path):
        provider = Provider([1.0, 0.0])
        with patch("code_review_graph.embeddings.get_provider", return_value=provider):
            store = EmbeddingStore(tmp_path / "e.db")
        _insert(store, [[1.0, 0.0]])

        store.search("another question", query_embedding=QueryEmbedding("query"))
        assert provider.queries == ["another question"]
        store.close()

    def test_failed_embedding_is_retried(self):
        class Flaky(Provider):
            def embed_query(self, text):
                self.queries.append(text)
                if len(self.queries) == 1:
                    raise RuntimeError("model busy")
                return list(self.query_vec)

        provider = Flaky([1.0, 0.0])
        shared = QueryEmbedding("query")
        with pytest.raises(RuntimeError, match="model busy"):
            shared.vector_for(provider)
        assert shared.vector_for(provider) == [1.0, 0.0]
        assert shared.vector_for(provider) == [1.0, 0.0]
        assert len(provider.queries) == 2

    def test_cross_repo_search_embeds_the_query_once(self, tmp_path):
        from code_review_graph.tools import cross_repo_search_func

        rng = random.Random(10)
        provider = Provider(_random_vectors(rng, 1, 8)[0])
        repos = []
        for index in range(3):
            repo = tmp_path / f"repo{index}"
            (repo / ".code-review-graph").mkdir(parents=True)
            db = repo / ".code-review-graph" / "graph.db"
            GraphStore(str(db)).close()
            with patch("code_review_graph.embeddings.get_provider", return_value=provider):
                store = EmbeddingStore(db)
            _insert(store, _random_vectors(rng, 6, 8))
            store.close()
            repos.append({"path": str(repo), "alias": f"repo{index}"})

        with (
            patch("code_review_graph.registry.Registry") as registry_cls,
            patch(
                "code_review_graph.tools.registry_tools.get_db_path",
                side_effect=lambda repo: repo / ".code-review-graph" / "graph.db",
            ),
            patch("code_review_graph.embeddings.get_provider", return_value=provider),
        ):
            registry_cls.return_value.list_repos.return_value = repos
            result = cross_repo_search_func(query="where is the splash screen")

        assert result["status"] == "ok"
        assert result["repos_searched"] == ["repo0", "repo1", "repo2"]
        assert provider.queries == ["where is the splash screen"]

    def test_hybrid_search_still_embeds_per_call_without_a_holder(self, tmp_path):
        provider = Provider([1.0, 0.0])
        db = tmp_path / "graph.db"
        graph = GraphStore(str(db))
        with patch("code_review_graph.embeddings.get_provider", return_value=provider):
            store = EmbeddingStore(db)
            _insert(store, [[1.0, 0.0]])
            store.close()
            hybrid_search(graph, "query")
            hybrid_search(graph, "query")
        graph.close()

        assert provider.queries == ["query", "query"]
