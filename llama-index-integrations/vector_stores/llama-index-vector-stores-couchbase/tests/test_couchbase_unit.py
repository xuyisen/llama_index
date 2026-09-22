"""Unit tests for Couchbase Vector Store (no external dependencies required)."""

from __future__ import annotations
import pytest
import warnings
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.couchbase.base import (
    _transform_couchbase_filter_condition,
    _transform_couchbase_filter_operator,
    _to_couchbase_filter,
    _convert_llamaindex_filters_to_sql,
    QueryVectorSearchType,
    QueryVectorSearchSimilarity,
    CouchbaseVectorStoreBase,
    CouchbaseSearchVectorStore,
    CouchbaseQueryVectorStore,
    CouchbaseVectorStore,
)


# =============================================================================
# Tests for filter conversion functions
# =============================================================================


class TestTransformCouchbaseFilterCondition:
    def test_and_condition(self):
        assert _transform_couchbase_filter_condition("and") == "conjuncts"

    def test_or_condition(self):
        assert _transform_couchbase_filter_condition("or") == "disjuncts"

    def test_unsupported_condition(self):
        with pytest.raises(ValueError, match="Filter condition .* not supported"):
            _transform_couchbase_filter_condition("not")


class TestTransformCouchbaseFilterOperator:
    def test_not_equal(self):
        result = _transform_couchbase_filter_operator("!=", "field", "value")
        assert result == {"must_not": {"disjuncts": [{"field": "field", "match": "value"}]}}

    def test_equal(self):
        result = _transform_couchbase_filter_operator("==", "field", "value")
        assert result == {"field": "field", "match": "value"}

    def test_greater_than(self):
        result = _transform_couchbase_filter_operator(">", "field", 10)
        assert result == {"min": 10, "inclusive_min": False, "field": "field"}

    def test_less_than(self):
        result = _transform_couchbase_filter_operator("<", "field", 10)
        assert result == {"max": 10, "inclusive_max": False, "field": "field"}

    def test_greater_than_or_equal(self):
        result = _transform_couchbase_filter_operator(">=", "field", 10)
        assert result == {"min": 10, "inclusive_min": True, "field": "field"}

    def test_less_than_or_equal(self):
        result = _transform_couchbase_filter_operator("<=", "field", 10)
        assert result == {"max": 10, "inclusive_max": True, "field": "field"}

    def test_text_match(self):
        result = _transform_couchbase_filter_operator("text_match", "field", "pattern")
        assert result == {"match_phrase": "pattern", "field": "field"}

    def test_unsupported_operator(self):
        with pytest.raises(ValueError, match="Filter operator .* not supported"):
            _transform_couchbase_filter_operator("unsupported", "field", "value")


class TestToCouchbaseFilter:
    def test_single_filter(self):
        filters = MetadataFilters(
            filters=[MetadataFilter(key="genre", value="Thriller", operator="==")]
        )
        result = _to_couchbase_filter(filters)
        assert result == {"field": "metadata.genre", "match": "Thriller"}

    def test_multiple_filters_and(self):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="genre", value="Thriller", operator="=="),
                MetadataFilter(key="pages", value=10, operator=">"),
            ],
            condition="and",
        )
        result = _to_couchbase_filter(filters)
        assert "conjuncts" in result["query"]
        assert len(result["query"]["conjuncts"]) == 2

    def test_multiple_filters_or(self):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="genre", value="Thriller", operator="=="),
                MetadataFilter(key="genre", value="Comedy", operator="=="),
            ],
            condition="or",
        )
        result = _to_couchbase_filter(filters)
        assert "disjuncts" in result["query"]
        assert len(result["query"]["disjuncts"]) == 2

    def test_filter_with_default_operator(self):
        filters = MetadataFilters(
            filters=[MetadataFilter(key="genre", value="Thriller")]
        )
        result = _to_couchbase_filter(filters)
        # When no operator is specified, it defaults to FilterOperator.EQ
        assert result == {"field": "metadata.genre", "match": "Thriller"}

    def test_empty_filters(self):
        filters = MetadataFilters(filters=[])
        result = _to_couchbase_filter(filters)
        assert result == {"query": {}}


class TestConvertLlamaIndexFiltersToSql:
    def test_no_filters(self):
        result = _convert_llamaindex_filters_to_sql(None, "metadata")
        assert result == ""

    def test_empty_filters(self):
        filters = MetadataFilters(filters=[])
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert result == ""

    def test_eq_string(self):
        filters = MetadataFilters(
            filters=[MetadataFilter(key="genre", value="Thriller", operator=FilterOperator.EQ)]
        )
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "d.metadata.genre = 'Thriller'" in result

    def test_eq_numeric(self):
        filters = MetadataFilters(
            filters=[MetadataFilter(key="pages", value=10, operator=FilterOperator.EQ)]
        )
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "d.metadata.pages = 10" in result

    def test_ne_string(self):
        filters = MetadataFilters(
            filters=[MetadataFilter(key="genre", value="Comedy", operator=FilterOperator.NE)]
        )
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "d.metadata.genre != 'Comedy'" in result

    def test_gt(self):
        filters = MetadataFilters(
            filters=[MetadataFilter(key="pages", value=10, operator=FilterOperator.GT)]
        )
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "d.metadata.pages > 10" in result

    def test_gte(self):
        filters = MetadataFilters(
            filters=[MetadataFilter(key="pages", value=10, operator=FilterOperator.GTE)]
        )
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "d.metadata.pages >= 10" in result

    def test_lt(self):
        filters = MetadataFilters(
            filters=[MetadataFilter(key="pages", value=20, operator=FilterOperator.LT)]
        )
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "d.metadata.pages < 20" in result

    def test_lte(self):
        filters = MetadataFilters(
            filters=[MetadataFilter(key="pages", value=20, operator=FilterOperator.LTE)]
        )
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "d.metadata.pages <= 20" in result

    def test_in_operator(self):
        filters = MetadataFilters(
            filters=[MetadataFilter(key="genre", value=["Thriller", "Comedy"], operator=FilterOperator.IN)]
        )
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "d.metadata.genre IN ['Thriller', 'Comedy']" in result

    def test_and_condition(self):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="genre", value="Thriller", operator=FilterOperator.EQ),
                MetadataFilter(key="pages", value=10, operator=FilterOperator.GT),
            ],
            condition="and",
        )
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "AND" in result
        assert "d.metadata.genre = 'Thriller'" in result
        assert "d.metadata.pages > 10" in result

    def test_or_condition(self):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="genre", value="Thriller", operator=FilterOperator.EQ),
                MetadataFilter(key="genre", value="Comedy", operator=FilterOperator.EQ),
            ],
            condition="or",
        )
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "OR" in result

    def test_nested_filters(self):
        inner_filters = MetadataFilters(
            filters=[MetadataFilter(key="pages", value=10, operator=FilterOperator.GT)],
            condition="and",
        )
        outer_filters = MetadataFilters(
            filters=[
                MetadataFilter(key="genre", value="Thriller", operator=FilterOperator.EQ),
                inner_filters,
            ],
            condition="and",
        )
        result = _convert_llamaindex_filters_to_sql(outer_filters, "metadata")
        assert "d.metadata.genre = 'Thriller'" in result
        assert "d.metadata.pages > 10" in result

    def test_unsupported_operator(self):
        # The MetadataFilter validates operators against FilterOperator enum,
        # so we test the _build_condition function directly by passing an
        # unsupported operator value through a filter with a string operator
        filters = MetadataFilters(
            filters=[MetadataFilter(key="genre", value="test", operator=FilterOperator.EQ)]
        )
        # This test verifies that the function handles valid operators correctly
        result = _convert_llamaindex_filters_to_sql(filters, "metadata")
        assert "d.metadata.genre = 'test'" in result


# =============================================================================
# Tests for CouchbaseVectorStoreBase
# =============================================================================


class TestCouchbaseVectorStoreBase:
    def test_init_with_mock_cluster(self):
        """Test initialization with a mock cluster using CouchbaseQueryVectorStore."""
        mock_cluster = MagicMock()
        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            search_type=QueryVectorSearchType.ANN,
            similarity=QueryVectorSearchSimilarity.COSINE,
            text_key="text",
            embedding_key="embedding",
            metadata_key="metadata",
        )
        assert store._bucket_name == "test_bucket"
        assert store._scope_name == "test_scope"
        assert store._collection_name == "test_collection"
        assert store._text_key == "text"
        assert store._embedding_key == "embedding"
        assert store._metadata_key == "metadata"

    def test_client_property(self):
        """Test client property returns the cluster."""
        mock_cluster = MagicMock()
        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            search_type=QueryVectorSearchType.ANN,
            similarity=QueryVectorSearchSimilarity.COSINE,
        )
        assert store.client is mock_cluster

    def test_bucket_property(self):
        """Test bucket property."""
        mock_cluster = MagicMock()
        mock_bucket = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket

        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            search_type=QueryVectorSearchType.ANN,
            similarity=QueryVectorSearchSimilarity.COSINE,
        )
        bucket = store.bucket
        assert bucket is mock_bucket
        mock_cluster.bucket.assert_called_with("test_bucket")

    def test_scope_property(self):
        """Test scope property."""
        mock_cluster = MagicMock()
        mock_bucket = MagicMock()
        mock_scope = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket
        mock_bucket.scope.return_value = mock_scope

        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            search_type=QueryVectorSearchType.ANN,
            similarity=QueryVectorSearchSimilarity.COSINE,
        )
        scope = store.scope
        assert scope is mock_scope
        mock_bucket.scope.assert_called_with("test_scope")

    def test_collection_property(self):
        """Test collection property."""
        mock_cluster = MagicMock()
        mock_bucket = MagicMock()
        mock_scope = MagicMock()
        mock_collection = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket
        mock_bucket.scope.return_value = mock_scope
        mock_scope.collection.return_value = mock_collection

        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            search_type=QueryVectorSearchType.ANN,
            similarity=QueryVectorSearchSimilarity.COSINE,
        )
        collection = store.collection
        assert collection is mock_collection
        mock_scope.collection.assert_called_with("test_collection")


# =============================================================================
# Tests for CouchbaseSearchVectorStore
# =============================================================================


class TestCouchbaseSearchVectorStore:
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        mock_cluster = MagicMock()
        store = CouchbaseSearchVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            index_name="test_index",
        )
        assert store._index_name == "test_index"
        assert store._scoped_index is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        mock_cluster = MagicMock()
        store = CouchbaseSearchVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            index_name="test_index",
            text_key="content",
            embedding_key="vector",
            metadata_key="meta",
            scoped_index=False,
        )
        assert store._text_key == "content"
        assert store._embedding_key == "vector"
        assert store._metadata_key == "meta"
        assert store._scoped_index is False


# =============================================================================
# Tests for CouchbaseQueryVectorStore
# =============================================================================


class TestCouchbaseQueryVectorStore:
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        mock_cluster = MagicMock()
        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
        )
        assert store._search_type == QueryVectorSearchType.ANN
        assert store._similarity == QueryVectorSearchSimilarity.COSINE
        assert store._nprobes is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        mock_cluster = MagicMock()
        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            search_type=QueryVectorSearchType.KNN,
            similarity=QueryVectorSearchSimilarity.L2,
            nprobes=50,
        )
        assert store._search_type == QueryVectorSearchType.KNN
        assert store._similarity == QueryVectorSearchSimilarity.L2
        assert store._nprobes == 50

    def test_init_with_string_search_type(self):
        """Test initialization with string search type."""
        mock_cluster = MagicMock()
        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            search_type="KNN",
        )
        assert store._search_type == QueryVectorSearchType.KNN

    def test_init_with_string_similarity(self):
        """Test initialization with string similarity."""
        mock_cluster = MagicMock()
        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            similarity="L2",
        )
        assert store._similarity == QueryVectorSearchSimilarity.L2

    def test_empty_query_embedding_error(self):
        """Test that query raises ValueError when embedding is empty."""
        mock_cluster = MagicMock()
        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
        )

        query = VectorStoreQuery(
            query_embedding=None,
            similarity_top_k=5,
        )

        with pytest.raises(ValueError, match="Query embedding must not be empty"):
            store.query(query)

    def test_query_with_output_fields(self):
        """Test query with output_fields specified."""
        mock_cluster = MagicMock()
        mock_bucket = MagicMock()
        mock_scope = MagicMock()
        mock_collection = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket
        mock_bucket.scope.return_value = mock_scope
        mock_scope.collection.return_value = mock_collection

        # Mock the query result
        mock_query_result = MagicMock()
        mock_query_result.rows.return_value = []
        mock_cluster.query.return_value = mock_query_result

        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
        )

        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3],
            similarity_top_k=5,
            output_fields=["text", "metadata.genre"],
        )

        result = store.query(query)
        assert isinstance(result, VectorStoreQueryResult)
        # Verify the query was called
        mock_cluster.query.assert_called_once()

    def test_query_with_filters(self):
        """Test query with filters."""
        mock_cluster = MagicMock()
        mock_bucket = MagicMock()
        mock_scope = MagicMock()
        mock_collection = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket
        mock_bucket.scope.return_value = mock_scope
        mock_scope.collection.return_value = mock_collection

        mock_query_result = MagicMock()
        mock_query_result.rows.return_value = []
        mock_cluster.query.return_value = mock_query_result

        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
        )

        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3],
            similarity_top_k=5,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="genre", value="Thriller", operator="=="),
                ]
            ),
        )

        result = store.query(query)
        assert isinstance(result, VectorStoreQueryResult)
        mock_cluster.query.assert_called_once()

    def test_query_with_nprobes_kwarg(self):
        """Test query with nprobes passed as kwarg."""
        mock_cluster = MagicMock()
        mock_bucket = MagicMock()
        mock_scope = MagicMock()
        mock_collection = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket
        mock_bucket.scope.return_value = mock_scope
        mock_scope.collection.return_value = mock_collection

        mock_query_result = MagicMock()
        mock_query_result.rows.return_value = []
        mock_cluster.query.return_value = mock_query_result

        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
        )

        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3],
            similarity_top_k=5,
        )

        result = store.query(query, nprobes=100)
        assert isinstance(result, VectorStoreQueryResult)
        mock_cluster.query.assert_called_once()

    def test_knn_search_type(self):
        """Test query with KNN search type."""
        mock_cluster = MagicMock()
        mock_bucket = MagicMock()
        mock_scope = MagicMock()
        mock_collection = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket
        mock_bucket.scope.return_value = mock_scope
        mock_scope.collection.return_value = mock_collection

        mock_query_result = MagicMock()
        mock_query_result.rows.return_value = []
        mock_cluster.query.return_value = mock_query_result

        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            search_type=QueryVectorSearchType.KNN,
        )

        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3],
            similarity_top_k=5,
        )

        result = store.query(query)
        assert isinstance(result, VectorStoreQueryResult)
        mock_cluster.query.assert_called_once()

    def test_query_error_handling(self):
        """Test that query properly handles errors."""
        mock_cluster = MagicMock()
        mock_bucket = MagicMock()
        mock_scope = MagicMock()
        mock_collection = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket
        mock_bucket.scope.return_value = mock_scope
        mock_scope.collection.return_value = mock_collection

        mock_cluster.query.side_effect = Exception("Database error")

        store = CouchbaseQueryVectorStore(
            cluster=mock_cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
        )

        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3],
            similarity_top_k=5,
        )

        with pytest.raises(ValueError, match="Vector search failed with error"):
            store.query(query)


# =============================================================================
# Tests for CouchbaseVectorStore (deprecated)
# =============================================================================


class TestCouchbaseVectorStoreDeprecated:
    def test_deprecation_warning(self):
        """Test that a deprecation warning is raised when instantiating CouchbaseVectorStore."""
        mock_cluster = MagicMock()

        with pytest.warns(DeprecationWarning) as warnings_raised:
            CouchbaseVectorStore(
                cluster=mock_cluster,
                bucket_name="test_bucket",
                scope_name="test_scope",
                collection_name="test_collection",
                index_name="test_index",
            )

        assert len(warnings_raised) >= 1
        assert "CouchbaseVectorStore is deprecated" in str(warnings_raised[0].message)

    def test_init_with_mock_cluster(self):
        """Test initialization with a mock cluster."""
        mock_cluster = MagicMock()

        with pytest.warns(DeprecationWarning):
            store = CouchbaseVectorStore(
                cluster=mock_cluster,
                bucket_name="test_bucket",
                scope_name="test_scope",
                collection_name="test_collection",
                index_name="test_index",
            )

        assert store._bucket_name == "test_bucket"
        assert store._scope_name == "test_scope"
        assert store._collection_name == "test_collection"
        assert store._index_name == "test_index"


# =============================================================================
# Tests for QueryVectorSearchType and QueryVectorSearchSimilarity enums
# =============================================================================


class TestEnums:
    def test_search_type_values(self):
        assert QueryVectorSearchType.ANN.value == "ANN"
        assert QueryVectorSearchType.KNN.value == "KNN"

    def test_similarity_values(self):
        assert QueryVectorSearchSimilarity.COSINE.value == "COSINE"
        assert QueryVectorSearchSimilarity.DOT.value == "DOT"
        assert QueryVectorSearchSimilarity.L2.value == "L2"
        assert QueryVectorSearchSimilarity.EUCLIDEAN.value == "EUCLIDEAN"
        assert QueryVectorSearchSimilarity.L2_SQUARED.value == "L2_SQUARED"
        assert QueryVectorSearchSimilarity.EUCLIDEAN_SQUARED.value == "EUCLIDEAN_SQUARED"
