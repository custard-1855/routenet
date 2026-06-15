# Plan: RouteNet TensorFlow 2.x Migration

## Context
`routenet/routenet.py` は TF1 の Estimator API・`tf.contrib`・`tf.nn.dynamic_rnn` など多数の非推奨 API を使用している。TF2 の Eager Execution・Keras API・`tf.data` モダンパイプラインへ移行することでコードの保守性と将来互換性を確保する。`routenet/upcdataset.py` は軽微な変更のみ必要。

---

## 変更ファイル
- `routenet/routenet.py` — メイン変更対象（全面的リファクタ）
- `routenet/upcdataset.py` — 1 行変更のみ

---

## 変更内容（routenet/routenet.py）

### 1. HParams → dataclass（行 7-20）
```python
# Before
hparams = tf.contrib.training.HParams(node_count=14, ...)

# After
from dataclasses import dataclass, field

@dataclass
class HParams:
    node_count: int = 14
    link_state_dim: int = 4
    path_state_dim: int = 2
    T: int = 3
    readout_units: int = 8
    learning_rate: float = 0.001
    batch_size: int = 32
    dropout_rate: float = 0.5
    l2: float = 0.1
    l2_2: float = 0.01
    learn_embedding: bool = True
    readout_layers: int = 2

hparams = HParams()
```
`hparams.parse(args.hparams)` は `_parse_hparams(hparams, csv_str)` ヘルパーで代替。

### 2. l2_regularizer（build() 行 48, 57）
```python
# Before
tf.contrib.layers.l2_regularizer(self.hparams.l2)

# After
tf.keras.regularizers.L2(self.hparams.l2)
```

### 3. dynamic_rnn → tf.keras.layers.RNN（call() 行 104-110）
`build()` 内で `path_update` を RNN ラッパーに変更：
```python
# build() 内
self.path_rnn = tf.keras.layers.RNN(
    tf.keras.layers.GRUCell(self.hparams.path_state_dim),
    return_sequences=True,
    return_state=True,
    name="path_rnn",
)
self.path_rnn.build(tf.TensorShape([None, None, self.hparams.link_state_dim]))
```

`call()` 内で置き換え：
```python
# Before
outputs, path_state = tf.nn.dynamic_rnn(
    self.path_update, link_inputs, sequence_length=lens,
    initial_state=path_state, dtype=tf.float32,
)

# After
mask = tf.sequence_mask(lens, maxlen=max_len)
outputs, path_state = self.path_rnn(
    link_inputs, mask=mask, initial_state=[path_state]
)
```

### 4. tf.unsorted_segment_sum（行 112）
```python
# Before
m = tf.unsorted_segment_sum(m, links, f_["n_links"])

# After
m = tf.math.unsorted_segment_sum(m, links, f_["n_links"])
```

### 5. delay_model_fn / drop_model_fn → Keras subclass（行 128-294）
Estimator の `model_fn` を廃止し、2 つのサブクラスで代替：

```python
class DelayRouteNet(RouteNet):
    def train_step(self, data):
        features, labels = data
        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = _heteroscedastic_loss(features, labels, predictions)
            loss += tf.add_n(self.losses)  # L2 regularization
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # update metrics
        ...

    def test_step(self, data):
        features, labels = data
        predictions = self(features, training=False)
        loss = _heteroscedastic_loss(features, labels, predictions)
        # update metrics
        ...
```

同様に `DropRouteNet` で binomial loss を実装。

### 6. カスタム損失関数（新関数）
```python
def _heteroscedastic_loss(features, labels, predictions):
    loc = predictions[..., 0]
    scale = tf.math.softplus(C + predictions[..., 1]) + 1e-9
    n = features["packets"] - labels["drops"]
    _2sigma = 2.0 * scale ** 2
    nll = n * labels["jitter"] / _2sigma + ...
    return tf.reduce_sum(nll) / 1e6

def _binomial_loss(features, labels, logits):
    loss_ratio = labels["drops"] / features["packets"]
    return tf.reduce_sum(
        features["packets"] * tf.nn.sigmoid_cross_entropy_with_logits(...)
    ) / 1e5
```

### 7. Pearson 相関カスタムメトリクス（新クラス）
`tf.contrib.metrics.streaming_pearson_correlation` の代替：
```python
class PearsonCorrelation(tf.keras.metrics.Metric):
    # sum_x, sum_y, sum_xy, sum_x2, sum_y2, count を蓄積
    def update_state(self, y_true, y_pred, sample_weight=None): ...
    def result(self): ...  # 相関係数を返す
```

### 8. tf.metrics.* → tf.keras.metrics.*
```python
tf.metrics.mean(x)              → tf.keras.metrics.Mean()
tf.metrics.mean_absolute_error  → tf.keras.metrics.MeanAbsoluteError()
```

### 9. データパイプライン（行 401-424）
```python
# Before
ds.apply(tf.data.experimental.parallel_interleave(...))
ds.apply(tf.data.experimental.shuffle_and_repeat(...))
ds = ds.filter(lambda x: tf.random_uniform(...))
it = ds.make_one_shot_iterator()
sample = transformation_func(it, ...)

# After
ds = files.interleave(tf.data.TFRecordDataset, cycle_length=4,
                      num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.shuffle(shuffle_buf).repeat()  # or just filter
ds = ds.filter(lambda x: tf.random.uniform(shape=()) < 0.1)
ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.prefetch(tf.data.AUTOTUNE)

# Ragged バッチで graph concatenation
ds = ds.ragged_batch(hparams.batch_size, drop_remainder=True)
ds = ds.map(_collate_graphs)
```

`transformation_func` と `cummax` を廃止し、`_collate_graphs` に統合：
```python
def _collate_graphs(batch):
    n_links = batch["n_links"]  # [B]
    n_paths = batch["n_paths"]  # [B]
    links_offsets = tf.cast(
        tf.concat([[0], tf.cumsum(n_links)[:-1]], axis=0), tf.int64
    )
    paths_offsets = tf.cast(
        tf.concat([[0], tf.cumsum(n_paths)[:-1]], axis=0), tf.int64
    )
    links = batch["links"]  # RaggedTensor [B, None]
    paths = batch["paths"]
    # row-wise offset broadcast
    links_flat = links.flat_values + tf.repeat(links_offsets, links.row_lengths())
    paths_flat = paths.flat_values + tf.repeat(paths_offsets, paths.row_lengths())
    features = {
        "traffic": batch["traffic"].flat_values,
        "capacities": batch["capacities"].flat_values,
        "sequences": batch["sequences"].flat_values,
        "packets": batch["packets"].flat_values,
        "links": links_flat,
        "paths": paths_flat,
        "n_links": tf.reduce_sum(n_links),
        "n_paths": tf.reduce_sum(n_paths),
        "n_total": tf.reduce_sum(batch["n_total"]),
    }
    labels = {
        "delay": batch["delay"].flat_values,
        "logdelay": batch["logdelay"].flat_values,
        "drops": batch["drops"].flat_values,
        "jitter": batch["jitter"].flat_values,
    }
    return features, labels
```

### 10. tf.io.VarLenFeature / FixedLenFeature（行 322-334）
```python
tf.VarLenFeature   → tf.io.VarLenFeature
tf.FixedLenFeature → tf.io.FixedLenFeature
```

### 11. serving_input_receiver_fn → @tf.function（行 427-446）
```python
# Before: tf.placeholder ベース
# After: @tf.function(input_signature=[...]) でモデルに直接 serving signature を付与
@tf.function(input_signature=[{
    "capacities": tf.TensorSpec([None], tf.float32),
    ...
}])
def serve(features):
    normalized = {k: scale_fn(k, v) for k, v in features.items()}
    return self(normalized, training=False)

tf.saved_model.save(model, args.model_dir, signatures={"serving_default": serve})
```

### 12. train() 関数（行 449-491）
```python
# Before: Estimator / train_and_evaluate
# After: Keras compile + fit
def train(args):
    tf.get_logger().setLevel("INFO")
    ...
    model = DelayRouteNet(hp, output_units=2) if args.target == "delay" \
        else DropRouteNet(hp, output_units=1)
    model.build()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        hp.learning_rate, decay_steps=50000, decay_rate=0.9, staircase=True
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule))

    train_ds = tfrecord_input_fn(args.train, hp, shuffle_buf=args.shuffle_buf)
    eval_ds  = tfrecord_input_fn(args.evaluation, hp, shuffle_buf=None)

    model.fit(
        train_ds,
        epochs=args.train_steps // steps_per_epoch,
        validation_data=eval_ds,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(args.model_dir, save_best_only=True),
            tf.keras.callbacks.TensorBoard(args.model_dir),
        ],
    )
```

---

## 変更内容（routenet/upcdataset.py）

行 173:
```python
# Before
writer = tf.python_io.TFRecordWriter(tfrecords_name)

# After
writer = tf.io.TFRecordWriter(tfrecords_name)
```

---

## 変更しないもの（mpnn/ ディレクトリ）
`mpnn/graph_nn.py`, `graph_nn2.py`, `eval.py` は Session/Graph ベースの別実装。今回のスコープ外。

---

## 検証方法

### セットアップ
```bash
uv add --dev pytest tensorflow
uv run pytest tests/ -v
```

### テストファイル: `tests/test_routenet.py`

実装すべきテストケース：

```python
import numpy as np
import pytest
import tensorflow as tf

from routenet.routenet import (
    HParams, _parse_hparams,
    RouteNet, DelayRouteNet, DropRouteNet,
    PearsonCorrelation, _collate_graphs, scale_fn,
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
    names = [v.name for v in model.trainable_variables]
    assert any("edge_update" in n for n in names)
    assert any("path_rnn"    in n for n in names)
    assert any("readout"     in n for n in names)


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
```

### 実行コマンド
```bash
uv run pytest tests/ -v                  # 全テスト
uv run pytest tests/ -v -k "collate"    # graph batching のみ
uv run pytest tests/ -v -k "pearson"    # メトリクスのみ
uv run pytest tests/ -v -k "routenet"   # モデル forward のみ
```
