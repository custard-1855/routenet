import numpy as np
import pytest
import tensorflow as tf

from routenet.routenet import (
    HParams,
    _parse_hparams,
    RouteNet,
    DelayRouteNet,
    DropRouteNet,
    PearsonCorrelation,
    _collate_graphs,
    scale_fn,
)


# ── フィクスチャ ──────────────────────────────────────────────

@pytest.fixture
def hparams():
    return HParams(link_state_dim=4, path_state_dim=2, T=2,
                   readout_units=8, readout_layers=1, batch_size=2)


@pytest.fixture
def minimal_features():
    """最小グラフ: 3 リンク, 2 パス, 4 要素"""
    return {
        "links":      tf.constant([0, 1, 1, 2], dtype=tf.int64),
        "paths":      tf.constant([0, 0, 1, 1], dtype=tf.int64),
        "sequences":  tf.constant([0, 1, 0, 1], dtype=tf.int64),
        "traffic":    tf.constant([0.1, 0.2], dtype=tf.float32),
        "capacities": tf.constant([10., 10., 10.], dtype=tf.float32),
        "packets":    tf.constant([100., 200.], dtype=tf.float32),
        "n_links":    tf.constant(3, dtype=tf.int64),
        "n_paths":    tf.constant(2, dtype=tf.int64),
        "n_total":    tf.constant(4, dtype=tf.int64),
    }


# ── HParams ───────────────────────────────────────────────────

def test_hparams_defaults():
    hp = HParams()
    assert hp.T == 3
    assert hp.batch_size == 32


def test_hparams_parse():
    hp = _parse_hparams(HParams(), "T=5,learning_rate=0.01")
    assert hp.T == 5
    assert hp.learning_rate == pytest.approx(0.01)


# ── RouteNet モデル ───────────────────────────────────────────

def test_routenet_build(hparams):
    model = RouteNet(hparams, output_units=1)
    model.build()
    assert model.built


def test_routenet_forward_delay(hparams, minimal_features):
    model = RouteNet(hparams, output_units=2)
    model.build()
    out = model(minimal_features, training=False)
    assert out.shape == (2, 2)


def test_routenet_forward_drop(hparams, minimal_features):
    model = RouteNet(hparams, output_units=1)
    model.build()
    out = model(minimal_features, training=False)
    assert out.shape == (2, 1)


def test_routenet_variables_exist(hparams):
    model = RouteNet(hparams, output_units=1)
    model.build()
    paths = [v.path for v in model.trainable_variables]
    assert any("edge_update" in p for p in paths)
    assert any("path_rnn" in p for p in paths)
    assert any("readout" in p for p in paths)


# ── scale_fn ─────────────────────────────────────────────────

def test_scale_fn_traffic():
    x = tf.constant([0.18], dtype=tf.float32)
    assert abs(float(scale_fn("traffic", x)[0])) < 1e-5


def test_scale_fn_capacities():
    x = tf.constant([10.0], dtype=tf.float32)
    assert float(scale_fn("capacities", x)[0]) == pytest.approx(1.0)


def test_scale_fn_passthrough():
    x = tf.constant([42.0])
    assert float(scale_fn("other", x)[0]) == pytest.approx(42.0)


# ── _collate_graphs ───────────────────────────────────────────

def _make_ragged_batch():
    def r(lists, dtype): return tf.ragged.constant(lists, dtype=dtype)
    return {
        "links":      r([[0, 1], [0, 1, 2]], tf.int64),
        "paths":      r([[0, 0], [0, 0, 0]], tf.int64),
        "sequences":  r([[0, 1], [0, 1, 2]], tf.int64),
        "traffic":    r([[0.1, 0.2], [0.3]], tf.float32),
        "capacities": r([[10., 10.], [10., 10., 10.]], tf.float32),
        "packets":    r([[100., 200.], [300.]], tf.float32),
        "delay":      r([[1., 2.], [3.]], tf.float32),
        "logdelay":   r([[0.0, 0.7], [1.1]], tf.float32),
        "drops":      r([[0., 0.], [0.]], tf.float32),
        "jitter":     r([[0.01, 0.02], [0.03]], tf.float32),
        "n_links":    tf.constant([2, 3], dtype=tf.int64),
        "n_paths":    tf.constant([2, 1], dtype=tf.int64),
        "n_total":    tf.constant([2, 3], dtype=tf.int64),
    }


def test_collate_graphs_link_offset():
    features, _ = _collate_graphs(_make_ragged_batch())
    links = features["links"].numpy()
    assert links[2] == 0 + 2  # 2 番目グラフ先頭: offset = n_links[0] = 2
    assert links[3] == 1 + 2


def test_collate_graphs_path_offset():
    features, _ = _collate_graphs(_make_ragged_batch())
    paths = features["paths"].numpy()
    assert all(p >= 2 for p in paths[2:])  # offset = n_paths[0] = 2


def test_collate_graphs_totals():
    features, _ = _collate_graphs(_make_ragged_batch())
    assert int(features["n_links"]) == 5   # 2 + 3
    assert int(features["n_paths"]) == 3   # 2 + 1


# ── PearsonCorrelation ────────────────────────────────────────

def test_pearson_perfect_positive():
    m = PearsonCorrelation()
    y = tf.constant([1., 2., 3., 4.])
    m.update_state(y, y)
    assert m.result().numpy() == pytest.approx(1.0, abs=1e-5)


def test_pearson_perfect_negative():
    m = PearsonCorrelation()
    m.update_state(tf.constant([1., 2., 3., 4.]), tf.constant([4., 3., 2., 1.]))
    assert m.result().numpy() == pytest.approx(-1.0, abs=1e-5)


def test_pearson_reset():
    m = PearsonCorrelation()
    m.update_state(tf.constant([1., 2., 3.]), tf.constant([1., 2., 3.]))
    m.reset_state()
    assert m.result().numpy() == pytest.approx(0.0, abs=1e-5)


# ── DelayRouteNet / DropRouteNet 統合 ─────────────────────────

def test_delay_routenet_train_step(hparams, minimal_features):
    model = DelayRouteNet(hparams, output_units=2)
    model.build()
    lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.9)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr))
    labels = {
        "delay": tf.ones([2]), "jitter": tf.ones([2]) * 0.01,
        "logdelay": tf.zeros([2]), "drops": tf.zeros([2]),
    }
    result = model.train_step((minimal_features, labels))
    assert "loss" in result


def test_drop_routenet_train_step(hparams, minimal_features):
    model = DropRouteNet(hparams, output_units=1)
    model.build()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    labels = {
        "drops": tf.constant([5., 10.]), "delay": tf.zeros([2]),
        "jitter": tf.zeros([2]), "logdelay": tf.zeros([2]),
    }
    result = model.train_step((minimal_features, labels))
    assert "loss" in result
