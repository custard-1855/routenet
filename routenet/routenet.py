import argparse
import dataclasses
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras


# Step 1: HParams → dataclass
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


_C_CONSTANT = tf.constant(np.log(np.expm1(0.098)), dtype=tf.float32)


def _parse_hparams(hparams, csv_str):
    for pair in csv_str.split(","):
        k, v = pair.strip().split("=")
        k, v = k.strip(), v.strip()
        current = getattr(hparams, k)
        if isinstance(current, bool):
            setattr(hparams, k, v.lower() in ("true", "1"))
        elif isinstance(current, int):
            setattr(hparams, k, int(v))
        elif isinstance(current, float):
            setattr(hparams, k, float(v))
        else:
            setattr(hparams, k, v)
    return hparams


class RouteNet(tf.keras.Model):
    def __init__(self, hparams, output_units=1, final_activation=None, **kwargs):
        super(RouteNet, self).__init__(**kwargs)

        self.hparams = hparams
        self.output_units = output_units
        self.final_activation = final_activation

    def build(self, input_shape=None):
        del input_shape

        self.edge_update = tf.keras.layers.GRUCell(
            self.hparams.link_state_dim, name="edge_update"
        )

        # Step 3: dynamic_rnn → tf.keras.layers.RNN
        self.path_rnn = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.hparams.path_state_dim, name="path_update"),
            return_sequences=True,
            return_state=True,
            name="path_rnn",
        )

        self.readout = tf.keras.models.Sequential(name="readout")

        for _ in range(self.hparams.readout_layers):
            self.readout.add(
                tf.keras.layers.Dense(
                    self.hparams.readout_units,
                    activation=tf.nn.selu,
                    # Step 2: l2_regularizer → tf.keras.regularizers.L2
                    kernel_regularizer=tf.keras.regularizers.L2(self.hparams.l2),
                )
            )
            self.readout.add(tf.keras.layers.Dropout(rate=self.hparams.dropout_rate))

        self.final = keras.layers.Dense(
            self.output_units,
            kernel_regularizer=tf.keras.regularizers.L2(self.hparams.l2_2),
            activation=self.final_activation,
        )

        self.edge_update.build(tf.TensorShape([None, self.hparams.path_state_dim]))
        self.path_rnn.build(tf.TensorShape([None, None, self.hparams.link_state_dim]))
        self.readout.build(input_shape=[None, self.hparams.path_state_dim])
        self.final.build(
            input_shape=[None, self.hparams.path_state_dim + self.hparams.readout_units]
        )

        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "hparams": dataclasses.asdict(self.hparams),
            "output_units": self.output_units,
            "final_activation": self.final_activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        if isinstance(config["hparams"], dict):
            config["hparams"] = HParams(**config["hparams"])
        return cls(**config)

    def call(self, inputs, training=False):
        f_ = inputs
        shape = tf.stack([f_["n_links"], self.hparams.link_state_dim - 1], axis=0)
        link_state = tf.concat(
            [tf.expand_dims(f_["capacities"], axis=1), tf.zeros(shape)], axis=1
        )

        shape = tf.stack([f_["n_paths"], self.hparams.path_state_dim - 1], axis=0)
        path_state = tf.concat(
            [tf.expand_dims(f_["traffic"][0 : f_["n_paths"]], axis=1), tf.zeros(shape)],
            axis=1,
        )

        links = f_["links"]
        paths = f_["paths"]
        seqs = f_["sequences"]

        for _ in range(self.hparams.T):
            h_ = tf.gather(link_state, links)

            ids = tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs) + 1
            shape = tf.stack([f_["n_paths"], max_len, self.hparams.link_state_dim])
            lens = tf.math.segment_sum(data=tf.ones_like(paths), segment_ids=paths)

            link_inputs = tf.scatter_nd(ids, h_, shape)

            # Step 3: tf.nn.dynamic_rnn → tf.keras.layers.RNN
            mask = tf.sequence_mask(lens, maxlen=max_len)
            outputs, path_state = self.path_rnn(
                link_inputs, mask=mask, initial_state=[path_state]
            )
            m = tf.gather_nd(outputs, ids)
            # Step 4: tf.unsorted_segment_sum → tf.math.unsorted_segment_sum
            m = tf.math.unsorted_segment_sum(m, links, f_["n_links"])

            link_state, _ = self.edge_update(m, [link_state])

        if self.hparams.learn_embedding:
            r = self.readout(path_state, training=training)
            o = self.final(tf.concat([r, path_state], axis=1))
        else:
            r = self.readout(tf.stop_gradient(path_state), training=training)
            o = self.final(tf.concat([r, tf.stop_gradient(path_state)], axis=1))

        return o


# Step 6: Custom loss functions
def _heteroscedastic_loss(features, labels, predictions):
    loc = predictions[..., 0]
    scale = tf.math.softplus(_C_CONSTANT + predictions[..., 1]) + np.float32(1e-9)

    n = features["packets"] - labels["drops"]
    _2sigma = np.float32(2.0) * scale**2
    nll = (
        n * labels["jitter"] / _2sigma
        + n * tf.math.squared_difference(labels["delay"], loc) / _2sigma
        + n * tf.math.log(scale)
    )
    return tf.reduce_sum(nll) / np.float32(1e6)


def _binomial_loss(features, labels, logits):
    loss_ratio = labels["drops"] / features["packets"]
    return tf.reduce_sum(
        features["packets"]
        * tf.nn.sigmoid_cross_entropy_with_logits(labels=loss_ratio, logits=logits)
    ) / np.float32(1e5)


# Step 7: Custom metrics
class MeanRelativeError(tf.keras.metrics.Metric):
    def __init__(self, name="mean_relative_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        relative = tf.abs(tf.math.divide_no_nan(y_true - y_pred, y_true))
        self.total.assign_add(tf.reduce_sum(relative))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


class PearsonCorrelation(tf.keras.metrics.Metric):
    def __init__(self, name="pearson_correlation", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_x = self.add_weight(name="sum_x", initializer="zeros")
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_xy = self.add_weight(name="sum_xy", initializer="zeros")
        self.sum_x2 = self.add_weight(name="sum_x2", initializer="zeros")
        self.sum_y2 = self.add_weight(name="sum_y2", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        self.sum_x.assign_add(tf.reduce_sum(y_true))
        self.sum_y.assign_add(tf.reduce_sum(y_pred))
        self.sum_xy.assign_add(tf.reduce_sum(y_true * y_pred))
        self.sum_x2.assign_add(tf.reduce_sum(y_true**2))
        self.sum_y2.assign_add(tf.reduce_sum(y_pred**2))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        n = self.count
        num = n * self.sum_xy - self.sum_x * self.sum_y
        den = tf.sqrt(
            tf.maximum(
                (n * self.sum_x2 - self.sum_x**2) * (n * self.sum_y2 - self.sum_y**2),
                0.0,
            )
        )
        return tf.math.divide_no_nan(num, den)

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


# Step 5: delay_model_fn → Keras subclass
class DelayRouteNet(RouteNet):
    def train_step(self, data):
        features, labels = data
        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = _heteroscedastic_loss(features, labels, predictions)
            loss += tf.add_n(self.losses) if self.losses else 0.0

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(labels["delay"], predictions[..., 0])

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        features, labels = data
        predictions = self(features, training=False)
        loss = _heteroscedastic_loss(features, labels, predictions)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(labels["delay"], predictions[..., 0])

        return {m.name: m.result() for m in self.metrics}


# Step 5: drop_model_fn → Keras subclass
class DropRouteNet(RouteNet):
    def train_step(self, data):
        features, labels = data
        with tf.GradientTape() as tape:
            logits = self(features, training=True)
            logits = tf.squeeze(logits, axis=-1)
            loss = _binomial_loss(features, labels, logits)
            loss += tf.add_n(self.losses) if self.losses else 0.0

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        features, labels = data
        logits = self(features, training=False)
        logits = tf.squeeze(logits, axis=-1)
        loss = _binomial_loss(features, labels, logits)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)

        return {m.name: m.result() for m in self.metrics}


def scale_fn(k, val):
    if k == "traffic":
        return (val - 0.18) / 0.15
    if k == "capacities":
        return val / 10.0
    return val


# Step 10: tf.VarLenFeature / FixedLenFeature → tf.io.*
def parse(serialized, normalize=True):
    with tf.device("/cpu:0"):
        with tf.name_scope("parse"):
            features = tf.io.parse_single_example(
                serialized,
                features={
                    "traffic": tf.io.VarLenFeature(tf.float32),
                    "delay": tf.io.VarLenFeature(tf.float32),
                    "logdelay": tf.io.VarLenFeature(tf.float32),
                    "jitter": tf.io.VarLenFeature(tf.float32),
                    "drops": tf.io.VarLenFeature(tf.float32),
                    "packets": tf.io.VarLenFeature(tf.float32),
                    "capacities": tf.io.VarLenFeature(tf.float32),
                    "links": tf.io.VarLenFeature(tf.int64),
                    "paths": tf.io.VarLenFeature(tf.int64),
                    "sequences": tf.io.VarLenFeature(tf.int64),
                    "n_links": tf.io.FixedLenFeature([], tf.int64),
                    "n_paths": tf.io.FixedLenFeature([], tf.int64),
                    "n_total": tf.io.FixedLenFeature([], tf.int64),
                },
            )
            for k in [
                "traffic",
                "delay",
                "logdelay",
                "jitter",
                "drops",
                "packets",
                "capacities",
                "links",
                "paths",
                "sequences",
            ]:
                features[k] = tf.sparse.to_dense(features[k])
                if normalize:
                    features[k] = scale_fn(k, features[k])

    return features


# Step 9: _collate_graphs replaces transformation_func + cummax
def _collate_graphs(batch):
    n_links = batch["n_links"]
    n_paths = batch["n_paths"]
    links_offsets = tf.cast(tf.concat([[0], tf.cumsum(n_links)[:-1]], axis=0), tf.int64)
    paths_offsets = tf.cast(tf.concat([[0], tf.cumsum(n_paths)[:-1]], axis=0), tf.int64)
    links = batch["links"]
    paths = batch["paths"]
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


# Step 9: data pipeline — interleave / shuffle / ragged_batch
def tfrecord_input_fn(filenames, hparams, shuffle_buf=1000):
    files = tf.data.Dataset.from_tensor_slices(filenames)
    files = files.shuffle(len(filenames))

    ds = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle_buf:
        ds = ds.shuffle(shuffle_buf).repeat()
    else:
        ds = ds.filter(lambda x: tf.random.uniform(shape=()) < 0.1)

    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ragged_batch(hparams.batch_size, drop_remainder=True)
    ds = ds.map(_collate_graphs)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# Step 12: train() — Keras compile + fit
def train(args):
    print(args)
    tf.get_logger().setLevel("INFO")

    hp = HParams()
    if args.hparams:
        _parse_hparams(hp, args.hparams)

    model = (
        DelayRouteNet(hp, output_units=2)
        if args.target == "delay"
        else DropRouteNet(hp, output_units=1)
    )
    model.build()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        hp.learning_rate, decay_steps=50000, decay_rate=0.9, staircase=True
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule), metrics=[PearsonCorrelation()]
    )

    train_ds = tfrecord_input_fn(args.train, hp, shuffle_buf=args.shuffle_buf)
    eval_ds = tfrecord_input_fn(args.evaluation, hp, shuffle_buf=None)

    steps_per_epoch = 1000

    # Step 11: serving_input_receiver_fn → @tf.function
    @tf.function(
        input_signature=[
            {
                "capacities": tf.TensorSpec([None], tf.float32),
                "traffic": tf.TensorSpec([None], tf.float32),
                "links": tf.TensorSpec([None], tf.int64),
                "paths": tf.TensorSpec([None], tf.int64),
                "sequences": tf.TensorSpec([None], tf.int64),
                "packets": tf.TensorSpec([None], tf.float32),
                "n_links": tf.TensorSpec([], tf.int64),
                "n_paths": tf.TensorSpec([], tf.int64),
                "n_total": tf.TensorSpec([], tf.int64),
            }
        ]
    )
    def serve(features):
        normalized = {k: scale_fn(k, v) for k, v in features.items()}
        return model(normalized, training=False)

    model.fit(
        train_ds,
        epochs=args.train_steps // steps_per_epoch,
        validation_data=eval_ds,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(args.model_dir, save_best_only=True),
            tf.keras.callbacks.TensorBoard(args.model_dir),
        ],
    )

    tf.saved_model.save(model, args.model_dir, signatures={"serving_default": serve})


def main():
    parser = argparse.ArgumentParser(description="RouteNet script")

    subparsers = parser.add_subparsers(help="sub-command help")

    parser_train = subparsers.add_parser("train", help="Train options")
    parser_train.add_argument(
        "--hparams", type=str, help='Comma separated list of "name=value" pairs.'
    )
    parser_train.add_argument(
        "--train", help="Train Tfrecords files", type=str, nargs="+"
    )
    parser_train.add_argument(
        "--evaluation", help="Evaluation Tfrecords files", type=str, nargs="+"
    )
    parser_train.add_argument("--model_dir", help="Model directory", type=str)
    parser_train.add_argument(
        "--train_steps", help="Training steps", type=int, default=100
    )
    parser_train.add_argument(
        "--eval_steps",
        help="Evaluation steps, default None= all",
        type=int,
        default=None,
    )
    parser_train.add_argument(
        "--shuffle_buf",
        help="Buffer size for samples shuffling",
        type=int,
        default=10000,
    )
    parser_train.add_argument(
        "--target", help="Predicted variable", type=str, default="delay"
    )
    parser_train.add_argument("--warm", help="Warm start from", type=str, default=None)
    parser_train.set_defaults(func=train)
    args = parser.parse_args()

    return args.func(args)


if __name__ == "__main__":
    main()
