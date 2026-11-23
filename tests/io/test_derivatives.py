import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import dirconn
from dirconn import Connectivity, Provenance
from dirconn.io import derivatives
from dirconn.io.derivatives import (
    derivative_root,
    filename,
    read_edges,
    result_task_key,
    unit_dir,
    write_dataset_description,
    write_result,
)

NODES = ("a", "b", "c")
ENT = {"sub": "02", "ses": "01", "task": "rest", "run": "1"}


KINDS = ["cortical", "cortical", "direct"]


def _conn(seed=0):
    rng = np.random.default_rng(seed)
    prov = Provenance(method="granger", parameters={"order": 3})
    return Connectivity.from_array(
        rng.random((3, 3, 5)), source=NODES, freqs=np.linspace(1, 40, 5), provenance=prov
    )


def _band(seed):
    rng = np.random.default_rng(seed)
    prov = Provenance(method="granger", parameters={"order": 3})
    return Connectivity.from_array(rng.random((3, 3)), source=NODES, provenance=prov)


def _test(band):
    p = np.full((3, 3), 0.01)
    np.fill_diagonal(p, np.nan)
    sig = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(sig, False)
    return SimpleNamespace(
        band=band,
        pvalues=p,
        adjusted=p * 2,
        significant=sig,
        test="surrogate",
        n_surrogates=99,
        seed=np.int64(7),
        generator={"name": "block_shuffle", "params": {"block": 50}},
        observed_history=({"op": "block_shuffle_blocks", "n": np.int64(4)},),
    )


def _write(tmp_path, tests, bands, long_table=True, conn=None, node_kinds=KINDS, sfreq=250.0):
    bids = tmp_path / "bids"
    src = bids / "derivatives" / "dirconn-roi" / "sub-02" / "x.tsv"
    return write_result(
        derivative_root(tmp_path / "out"),
        ENT,
        "eeg",
        desc="gc",
        conn=_conn() if conn is None else conn,
        bands=bands,
        tests=tests,
        sources=[src, Path("/elsewhere/y.tsv")],
        bids_dir=bids,
        task_key="abc123",
        long_table=long_table,
        node_kinds=node_kinds,
        sfreq=sfreq,
    )


def test_filename_orders_entities():
    shuffled = {"run": "1", "task": "rest", "sub": "02", "ses": "01"}
    assert (
        filename(shuffled, desc="gc", suffix="conn", ext=".nc")
        == "sub-02_ses-01_task-rest_run-1_desc-gc_conn.nc"
    )


def test_session_appears_in_path_only_when_present(tmp_path):
    root = derivative_root(tmp_path)
    assert root == tmp_path / "dirconn"
    assert unit_dir(root, ENT, "eeg") == root / "sub-02" / "ses-01" / "eeg"
    assert unit_dir(root, {"sub": "02", "task": "rest"}, "meg") == root / "sub-02" / "meg"


def test_write_result_round_trips_the_connectivity_and_sidecar_fields(tmp_path):
    bands = {"alpha": _band(1), "beta": _band(2)}
    res = _write(tmp_path, [_test("alpha"), _test("beta")], bands)
    back = Connectivity.from_netcdf(res.nc)
    np.testing.assert_allclose(back.values, _conn().values)
    assert res.nc.name == "sub-02_ses-01_task-rest_run-1_desc-gc_conn.nc"
    assert res.json.name.endswith("_conn.json")
    side = json.loads(res.json.read_text(encoding="utf-8"))
    assert side["Measure"] == "granger"
    assert side["Directed"] is True
    assert side["Estimator"] == "gc"
    assert side["Parameters"] == {"order": 3}
    assert side["NodeLabels"] == list(NODES)
    assert side["NodeKinds"] == ["cortical", "cortical", "direct"]
    assert list(side["Bands"]) == ["alpha", "beta"]
    assert side["SamplingFrequency"] == 250.0
    assert side["TaskKey"] == "abc123"
    assert side["DirconnVersion"] == dirconn.__version__
    assert side["DerivativesVersion"] == "1"
    assert side["Sources"] == ["derivatives/dirconn-roi/sub-02/x.tsv", "/elsewhere/y.tsv"]
    assert [s["Band"] for s in side["Statistics"]] == ["alpha", "beta"]
    assert side["Statistics"][0]["Seed"] == 7
    assert "Description" in side


def test_edges_table_has_one_row_per_pair_and_band(tmp_path):
    bands = {"alpha": _band(1), "beta": _band(2)}
    res = _write(tmp_path, [_test("alpha"), _test("beta")], bands)
    df = read_edges(res.edges)
    assert list(df.columns) == ["source", "target", "band", "value", "p", "p_adj", "significant"]
    assert len(df) == 3 * 2 * 2
    assert (df["source"] != df["target"]).all()
    row = df[(df.source == "a") & (df.target == "b") & (df.band == "beta")].iloc[0]
    assert np.isclose(row["value"], bands["beta"].values[0, 1, 0, 0])
    assert np.isclose(row["p"], 0.01) and np.isclose(row["p_adj"], 0.02)
    assert bool(row["significant"]) is True


def test_edges_without_tests_use_na(tmp_path):
    res = _write(tmp_path, [], {}, conn=_band(3))
    df = read_edges(res.edges)
    assert len(df) == 6
    assert set(df["band"]) == {"broadband"}
    assert df["p"].isna().all() and df["p_adj"].isna().all() and df["significant"].isna().all()
    assert "n/a" in res.edges.read_text(encoding="utf-8")
    assert _write(tmp_path, [], {}, long_table=False, conn=_band(3)).edges is None


def test_dataset_description_has_generated_by(tmp_path):
    root = tmp_path / "new" / "dirconn"
    write_dataset_description(root, version="1.2.3", container={"Type": "docker", "Tag": None})
    d = json.loads((root / "dataset_description.json").read_text(encoding="utf-8"))
    assert d["DatasetType"] == "derivative" and d["BIDSVersion"] == "1.9.0"
    gb = d["GeneratedBy"][0]
    assert gb["Name"] == "dirconn" and gb["Version"] == "1.2.3"
    assert gb["CodeURL"] == "https://github.com/HanweiLiu-Viola/dirconn"
    assert gb["Container"] == {"Type": "docker", "Tag": None}
    assert d["Dirconn"] == {"DerivativesVersion": "1"}
    write_dataset_description(root, version="1.2.4", container=None)
    d = json.loads((root / "dataset_description.json").read_text(encoding="utf-8"))
    assert "Container" not in d["GeneratedBy"][0]


def test_result_task_key_reads_back(tmp_path):
    res = _write(tmp_path, [], {})
    assert result_task_key(res.json) == "abc123"
    assert result_task_key(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert result_task_key(bad) is None
    bad.write_text("{}", encoding="utf-8")
    assert result_task_key(bad) is None
    assert isinstance(read_edges(res.edges), pd.DataFrame)


def test_sidecar_omits_node_kinds_and_sfreq_when_unknown(tmp_path):
    res = _write(tmp_path, [], {}, node_kinds=None, sfreq=None)
    side = json.loads(res.json.read_text(encoding="utf-8"))
    assert "NodeKinds" not in side
    assert "SamplingFrequency" not in side


def test_node_kinds_must_match_the_node_labels(tmp_path):
    with pytest.raises(ValueError, match="node_kinds"):
        _write(tmp_path, [], {}, node_kinds=["cortical"])


def _time_resolved(seed):
    rng = np.random.default_rng(seed)
    prov = Provenance(method="adtf", parameters={})
    return Connectivity.from_array(
        rng.random((3, 3, 1, 4)), source=NODES, times=np.arange(4.0), provenance=prov
    )


def test_time_resolved_bands_give_one_row_per_time_point(tmp_path):
    p = np.full((3, 3, 1, 4), 0.01)
    p[0, 1, 0, 2] = 0.5
    significant = p < 0.05
    test = SimpleNamespace(
        **{**vars(_test("alpha")), "pvalues": p, "adjusted": p, "significant": significant}
    )
    band = _time_resolved(1)
    res = _write(tmp_path, [test], {"alpha": band})
    df = read_edges(res.edges)
    assert list(df.columns) == list(derivatives.TIME_EDGE_COLUMNS)
    assert len(df) == 6 * 4  # ordered pairs x time points
    cell = df[(df.source == NODES[0]) & (df.target == NODES[1]) & (df.time == 2.0)].iloc[0]
    assert cell.value == pytest.approx(band.values[0, 1, 0, 2])
    assert (cell.p, cell.p_adj, bool(cell.significant)) == (0.5, 0.5, False)
    assert df["significant"].sum() == 23


def test_failed_edges_write_leaves_no_sidecar(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(derivatives, "_edge_rows", fail)
    with pytest.raises(OSError, match="disk full"):
        _write(tmp_path, [], {}, conn=_band(3))
    directory = unit_dir(derivative_root(tmp_path / "out"), ENT, "eeg")
    assert not list(directory.glob("*_conn.json"))
    assert not list(directory.glob(".*"))


def test_sidecar_is_written_last_and_atomically(tmp_path, monkeypatch):
    order = []
    real = derivatives._write_json_atomic

    def spy(path, payload):
        order.append(("json", [q.name for q in path.parent.iterdir()]))
        real(path, payload)

    monkeypatch.setattr(derivatives, "_write_json_atomic", spy)
    res = _write(tmp_path, [], {}, conn=_band(3))
    ((kind, present),) = order
    assert kind == "json"
    assert res.nc.name in present and res.edges.name in present
    assert res.json.name not in present
    assert sorted(q.name for q in res.json.parent.iterdir()) == sorted(
        [res.nc.name, res.json.name, res.edges.name]
    )
