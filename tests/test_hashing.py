"""Tests for canonical hashing determinism."""

import hashlib
import json

from rankigi import _canonical_json, _sha256


class TestCanonicalJson:
    """Canonical JSON must be deterministic regardless of key insertion order."""

    def test_sorted_keys(self):
        a = _canonical_json({"z": 1, "a": 2, "m": 3})
        b = _canonical_json({"a": 2, "m": 3, "z": 1})
        assert a == b
        assert a == '{"a":2,"m":3,"z":1}'

    def test_nested_sorted_keys(self):
        a = _canonical_json({"outer": {"z": 1, "a": 2}})
        b = _canonical_json({"outer": {"a": 2, "z": 1}})
        assert a == b

    def test_compact_separators(self):
        result = _canonical_json({"key": "value"})
        assert " " not in result
        assert result == '{"key":"value"}'

    def test_list_order_preserved(self):
        result = _canonical_json({"items": [3, 1, 2]})
        assert result == '{"items":[3,1,2]}'

    def test_empty_dict(self):
        assert _canonical_json({}) == "{}"

    def test_empty_string(self):
        assert _canonical_json("") == '""'

    def test_none_value(self):
        assert _canonical_json(None) == "null"

    def test_nested_deeply(self):
        obj = {"a": {"b": {"c": {"d": 1}}}}
        result = _canonical_json(obj)
        assert result == '{"a":{"b":{"c":{"d":1}}}}'


class TestSha256:
    """SHA-256 hashing must be deterministic and match stdlib."""

    def test_string_input(self):
        result = _sha256("hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert result == expected

    def test_dict_input_deterministic(self):
        a = _sha256({"z": 1, "a": 2})
        b = _sha256({"a": 2, "z": 1})
        assert a == b

    def test_dict_matches_canonical_json_hash(self):
        obj = {"key": "value"}
        canonical = _canonical_json(obj)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert _sha256(obj) == expected

    def test_different_inputs_different_hashes(self):
        assert _sha256("a") != _sha256("b")

    def test_empty_string(self):
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256("") == expected

    def test_unicode_input(self):
        result = _sha256("hello world")
        expected = hashlib.sha256("hello world".encode("utf-8")).hexdigest()
        assert result == expected

    def test_large_input(self):
        big = "x" * 100_000
        result = _sha256(big)
        expected = hashlib.sha256(big.encode("utf-8")).hexdigest()
        assert result == expected

    def test_numeric_types(self):
        # int and float produce different canonical JSON
        assert _sha256(1) != _sha256(1.0)  # "1" vs "1.0"

    def test_list_input(self):
        a = _sha256([1, 2, 3])
        b = _sha256([1, 2, 3])
        assert a == b
        # different order = different hash
        assert _sha256([1, 2, 3]) != _sha256([3, 2, 1])
