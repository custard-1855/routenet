"""
訓練パイプライン — ゼロからモデルができるまでの3ステップ。

全ステップを順番に実行:
    uv run pytest tasks/ -s -v

個別ステップ:
    uv run pytest tasks/ -s -v -k download   # ダウンロードのみ
    uv run pytest tasks/ -s -v -k extract    # 展開のみ
    uv run pytest tasks/ -s -v -k train      # 訓練のみ

各ステップは冪等（すでに完了していればスキップ）。
"""

import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf

from routenet.routenet import (
    DelayRouteNet,
    HParams,
    MeanRelativeError,
    PearsonCorrelation,
    tfrecord_input_fn,
)

# ── 設定 ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATASET_URL = "https://knowledgedefinednetworking.org/data/datasets_v1/nsfnetbw.tar.gz"
DATASET_ARCHIVE = DATA_DIR / "nsfnetbw.tar.gz"
MODEL_DIR = Path("models") / "delay"

HPARAMS = HParams(batch_size=32, T=3, readout_layers=2)
STEPS_PER_EPOCH = 200
EPOCHS = 20
VALIDATION_STEPS = 50

# ── 論文レベル設定 ────────────────────────────────────────────────────
HPARAMS_FULL = HParams(
    link_state_dim=16,
    path_state_dim=32,
    readout_units=256,
    T=8,
    dropout_rate=0.5,
    l2=0.1,
    batch_size=128,
    readout_layers=2,
)
STEPS_PER_EPOCH_FULL = 250  # 1000 steps × batch32 相当（128×250 = 32×1000）
EPOCHS_FULL = 300
VALIDATION_STEPS_FULL = 100
SHUFFLE_BUF_FULL = 30000
MODEL_DIR_FULL = Path("models") / "delay_full"


# ── ステップ 1: ダウンロード ──────────────────────────────────────────


def test_01_download():
    """NSFNet データセットを KDN からダウンロード（~900 MB）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_ARCHIVE.exists():
        mb = DATASET_ARCHIVE.stat().st_size / 1e6
        print(f"\nskip — already exists: {DATASET_ARCHIVE} ({mb:.0f} MB)")
        return

    print(f"\ndownloading {DATASET_URL} ...")

    def _progress(count, block, total):
        if total > 0:
            print(f"\r  {count * block / total * 100:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(DATASET_URL, DATASET_ARCHIVE, reporthook=_progress)
    mb = DATASET_ARCHIVE.stat().st_size / 1e6
    print(f"\nsaved → {DATASET_ARCHIVE} ({mb:.0f} MB)")


# ── ステップ 2: 展開 ─────────────────────────────────────────────────


def test_02_extract():
    """アーカイブから TFRecords を展開する。"""
    assert DATASET_ARCHIVE.exists(), (
        f"{DATASET_ARCHIVE} が見つかりません。先に download を実行してください:\n"
        f"  uv run pytest tasks/ -s -k download"
    )

    existing = list(DATA_DIR.rglob("*.tfrecords"))
    if existing:
        print(
            f"\nskip — {len(existing)} 件の .tfrecords が既に {DATA_DIR}/ 以下に存在します"
        )
        return

    with tarfile.open(DATASET_ARCHIVE, "r:gz") as tar:
        members = [m for m in tar.getmembers() if ".tfrecords" in m.name]
        print(f"\n{len(members)} 件の tfrecords を展開中 ...")
        tar.extractall(DATA_DIR, members=members, filter="data")

    train_n = len([p for p in DATA_DIR.rglob("*.tfrecords") if "train" in str(p)])
    eval_n = len([p for p in DATA_DIR.rglob("*.tfrecords") if "evaluat" in str(p)])
    print(f"train: {train_n} 件  eval: {eval_n} 件")


# ── ステップ 3: 訓練 ─────────────────────────────────────────────────


def test_03_train():
    """DelayRouteNet を NSFNet TFRecords で訓練する。"""
    train_files = sorted(
        str(p) for p in DATA_DIR.rglob("*.tfrecords") if "train" in str(p)
    )
    eval_files = sorted(
        str(p) for p in DATA_DIR.rglob("*.tfrecords") if "evaluat" in str(p)
    )
    assert train_files, (
        f"train 用 tfrecords が {DATA_DIR} 以下に見つかりません。"
        f"先に extract を実行してください:\n  uv run pytest tasks/ -s -k extract"
    )
    assert eval_files, (
        f"eval 用 tfrecords が {DATA_DIR} 以下に見つかりません。"
        f"先に extract を実行してください:\n  uv run pytest tasks/ -s -k extract"
    )
    print(f"\ntrain: {len(train_files)} 件  eval: {len(eval_files)} 件")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = DelayRouteNet(HPARAMS, output_units=2)
    model.build()

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        HPARAMS.learning_rate, decay_steps=50000, decay_rate=0.9, staircase=True
    )
    # run_eagerly=True: Metal GPU の while_loop グラフコンパイルバグを回避
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), run_eagerly=True)

    train_ds = tfrecord_input_fn(train_files, HPARAMS, shuffle_buf=10000)
    eval_ds = tfrecord_input_fn(eval_files, HPARAMS, shuffle_buf=None)

    model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=eval_ds,
        validation_steps=VALIDATION_STEPS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(MODEL_DIR / "ckpt.keras"),
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(str(MODEL_DIR / "logs")),
            tf.keras.callbacks.CSVLogger(str(MODEL_DIR / "training.csv")),
        ],
    )
    print(f"\nモデル保存先 → {MODEL_DIR}/ckpt.keras")


# ── ステップ 4: 評価 ─────────────────────────────────────────────────


def test_04_evaluate():
    """訓練済みモデルを eval データで評価する（MAE・Pearson ρ）。"""
    ckpt = MODEL_DIR / "ckpt.keras"
    assert ckpt.exists(), (
        f"{ckpt} が見つかりません。先に train を実行してください:\n"
        f"  uv run pytest tasks/ -s -k train"
    )

    model = tf.keras.models.load_model(
        str(ckpt),
        custom_objects={"DelayRouteNet": DelayRouteNet, "HParams": HParams},
    )

    eval_files = sorted(
        str(p) for p in DATA_DIR.rglob("*.tfrecords") if "evaluat" in str(p)
    )
    assert eval_files, (
        f"eval 用 tfrecords が {DATA_DIR} 以下に見つかりません。"
        f"先に extract を実行してください:\n  uv run pytest tasks/ -s -k extract"
    )
    print(f"\neval: {len(eval_files)} 件")

    eval_ds = tfrecord_input_fn(eval_files, HPARAMS, shuffle_buf=None)

    mae = tf.keras.metrics.MeanAbsoluteError()
    mre = MeanRelativeError()
    rho = PearsonCorrelation()
    n_samples = 0

    for features, labels in eval_ds.take(VALIDATION_STEPS):
        preds = model(features, training=False)
        delay_pred = preds[..., 0]
        mae.update_state(labels["delay"], delay_pred)
        mre.update_state(labels["delay"], delay_pred)
        rho.update_state(labels["delay"], delay_pred)
        n_samples += int(labels["delay"].shape[0])

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_files": len(eval_files),
        "validation_steps": VALIDATION_STEPS,
        "n_samples": n_samples,
        "mae": float(mae.result().numpy()),
        "mre": float(mre.result().numpy()),
        "pearson_rho": float(rho.result().numpy()),
    }

    out = MODEL_DIR / "eval_results.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"サンプル数: {results['n_samples']}")
    print(f"MAE:       {results['mae']:.4f}")
    print(f"MRE:       {results['mre']:.4f}")
    print(f"Pearson ρ: {results['pearson_rho']:.4f}")
    print(f"保存先 → {out}")


# ── ステップ 5: 相対誤差の CDF ───────────────────────────────────────


def test_05_cdf():
    """相対誤差の累積分布関数をプロットして保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ckpt = MODEL_DIR / "ckpt.keras"
    assert ckpt.exists(), (
        f"{ckpt} が見つかりません。先に train を実行してください:\n"
        f"  uv run pytest tasks/ -s -k train"
    )

    model = tf.keras.models.load_model(
        str(ckpt),
        custom_objects={"DelayRouteNet": DelayRouteNet, "HParams": HParams},
    )

    eval_files = sorted(
        str(p) for p in DATA_DIR.rglob("*.tfrecords") if "evaluat" in str(p)
    )
    assert eval_files, (
        f"eval 用 tfrecords が {DATA_DIR} 以下に見つかりません。"
        f"先に extract を実行してください:\n  uv run pytest tasks/ -s -k extract"
    )

    eval_ds = tfrecord_input_fn(eval_files, HPARAMS, shuffle_buf=None)

    rel_errors = []
    for features, labels in eval_ds.take(VALIDATION_STEPS):
        preds = model(features, training=False)
        delay_pred = preds[..., 0].numpy()
        delay_true = labels["delay"].numpy()
        rel = (delay_true - delay_pred) / delay_true
        rel_errors.append(rel)

    rel_errors = np.concatenate(rel_errors)
    clipped = np.clip(rel_errors, -1.0, 1.0)
    clipped_sorted = np.sort(clipped)
    cdf = np.arange(1, len(clipped_sorted) + 1) / len(clipped_sorted)

    fig, ax = plt.subplots()
    ax.plot(clipped_sorted, cdf)
    ax.set_xlabel(r"Relative Error  $(y - \hat{y})\,/\,y$")
    ax.set_ylabel("CDF")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("CDF of Relative Error (delay)")
    ax.grid(True)

    out = MODEL_DIR / "cdf_relative_error.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n保存先 → {out}")
    print(f"サンプル数: {len(rel_errors)}")
    print(f"[-1,1] 外の割合: {np.mean(np.abs(rel_errors) > 1):.3f}")


# ── ステップ 6: 論文レベル訓練 ────────────────────────────────────────


def test_06_train_full():
    """論文レベルのハイパーパラメータで DelayRouteNet を訓練する。

    スモークテスト（test_03_train）との違い:
      - link_state_dim=16, path_state_dim=32, readout_units=256, T=8
      - 300 epoch × 1000 steps = 300,000 steps（元実装 ~293,000 に相当）
      - shuffle_buf=30,000
      - 保存先: models/delay_full/（スモーク用を上書きしない）
    """
    train_files = sorted(
        str(p) for p in DATA_DIR.rglob("*.tfrecords") if "train" in str(p)
    )
    eval_files = sorted(
        str(p) for p in DATA_DIR.rglob("*.tfrecords") if "evaluat" in str(p)
    )
    assert train_files, (
        f"train 用 tfrecords が {DATA_DIR} 以下に見つかりません。"
        f"先に extract を実行してください:\n  uv run pytest tasks/ -s -k extract"
    )
    assert eval_files, (
        f"eval 用 tfrecords が {DATA_DIR} 以下に見つかりません。"
        f"先に extract を実行してください:\n  uv run pytest tasks/ -s -k extract"
    )
    print(f"\ntrain: {len(train_files)} 件  eval: {len(eval_files)} 件")
    MODEL_DIR_FULL.mkdir(parents=True, exist_ok=True)

    model = DelayRouteNet(HPARAMS_FULL, output_units=2)
    model.build()

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        HPARAMS_FULL.learning_rate, decay_steps=50000, decay_rate=0.9, staircase=True
    )
    # run_eagerly=True: Metal GPU の while_loop グラフコンパイルバグを回避
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), run_eagerly=True)

    train_ds = tfrecord_input_fn(
        train_files, HPARAMS_FULL, shuffle_buf=SHUFFLE_BUF_FULL
    )
    eval_ds = tfrecord_input_fn(eval_files, HPARAMS_FULL, shuffle_buf=None)

    model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH_FULL,
        epochs=EPOCHS_FULL,
        validation_data=eval_ds,
        validation_steps=VALIDATION_STEPS_FULL,
        validation_freq=10,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(MODEL_DIR_FULL / "ckpt.keras"),
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=False,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(str(MODEL_DIR_FULL / "logs")),
            tf.keras.callbacks.CSVLogger(str(MODEL_DIR_FULL / "training.csv")),
        ],
    )
    print(f"\nモデル保存先 → {MODEL_DIR_FULL}/ckpt.keras")


# ── ステップ 7: 論文レベルモデルの評価 ───────────────────────────────


def test_07_evaluate_full():
    """test_06_train_full で訓練したモデルを評価する（MAE・MRE・Pearson ρ）。"""
    ckpt = MODEL_DIR_FULL / "ckpt.keras"
    assert ckpt.exists(), (
        f"{ckpt} が見つかりません。先に train_full を実行してください:\n"
        f"  uv run pytest tasks/ -s -k train_full"
    )

    model = tf.keras.models.load_model(
        str(ckpt),
        custom_objects={"DelayRouteNet": DelayRouteNet, "HParams": HParams},
    )

    eval_files = sorted(
        str(p) for p in DATA_DIR.rglob("*.tfrecords") if "evaluat" in str(p)
    )
    assert eval_files, (
        f"eval 用 tfrecords が {DATA_DIR} 以下に見つかりません。"
        f"先に extract を実行してください:\n  uv run pytest tasks/ -s -k extract"
    )
    print(f"\neval: {len(eval_files)} 件")

    eval_ds = tfrecord_input_fn(eval_files, HPARAMS_FULL, shuffle_buf=None)

    mae = tf.keras.metrics.MeanAbsoluteError()
    mre = MeanRelativeError()
    rho = PearsonCorrelation()
    n_samples = 0

    for features, labels in eval_ds.take(VALIDATION_STEPS_FULL):
        preds = model(features, training=False)
        delay_pred = preds[..., 0]
        mae.update_state(labels["delay"], delay_pred)
        mre.update_state(labels["delay"], delay_pred)
        rho.update_state(labels["delay"], delay_pred)
        n_samples += int(labels["delay"].shape[0])

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_files": len(eval_files),
        "validation_steps": VALIDATION_STEPS_FULL,
        "n_samples": n_samples,
        "mae": float(mae.result().numpy()),
        "mre": float(mre.result().numpy()),
        "pearson_rho": float(rho.result().numpy()),
    }

    out = MODEL_DIR_FULL / "eval_results.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"サンプル数: {results['n_samples']}")
    print(f"MAE:       {results['mae']:.4f}")
    print(f"MRE:       {results['mre']:.4f}")
    print(f"Pearson ρ: {results['pearson_rho']:.4f}")
    print(f"保存先 → {out}")


# ── ステップ 9: 論文レベルモデルの相対誤差 CDF ───────────────────────


def test_09_cdf_full():
    """test_06_train_full で訓練したモデルの相対誤差 CDF をプロットして保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ckpt = MODEL_DIR_FULL / "ckpt.keras"
    assert ckpt.exists(), (
        f"{ckpt} が見つかりません。先に train_full を実行してください:\n"
        f"  uv run pytest tasks/ -s -k train_full"
    )

    model = tf.keras.models.load_model(
        str(ckpt),
        custom_objects={"DelayRouteNet": DelayRouteNet, "HParams": HParams},
    )

    eval_files = sorted(
        str(p) for p in DATA_DIR.rglob("*.tfrecords") if "evaluat" in str(p)
    )
    assert eval_files, (
        f"eval 用 tfrecords が {DATA_DIR} 以下に見つかりません。"
        f"先に extract を実行してください:\n  uv run pytest tasks/ -s -k extract"
    )

    eval_ds = tfrecord_input_fn(eval_files, HPARAMS_FULL, shuffle_buf=None)

    rel_errors = []
    for features, labels in eval_ds.take(VALIDATION_STEPS_FULL):
        preds = model(features, training=False)
        delay_pred = preds[..., 0].numpy()
        delay_true = labels["delay"].numpy()
        rel = (delay_true - delay_pred) / delay_true
        rel_errors.append(rel)

    rel_errors = np.concatenate(rel_errors)
    clipped = np.clip(rel_errors, -1.0, 1.0)
    clipped_sorted = np.sort(clipped)
    cdf = np.arange(1, len(clipped_sorted) + 1) / len(clipped_sorted)

    fig, ax = plt.subplots()
    ax.plot(clipped_sorted, cdf)
    ax.set_xlabel(r"Relative Error  $(y - \hat{y})\,/\,y$")
    ax.set_ylabel("CDF")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("CDF of Relative Error (delay, full model)")
    ax.grid(True)

    out = MODEL_DIR_FULL / "cdf_relative_error.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n保存先 → {out}")
    print(f"サンプル数: {len(rel_errors)}")
    print(f"[-1,1] 外の割合: {np.mean(np.abs(rel_errors) > 1):.3f}")


# ── GBN データ設定 ────────────────────────────────────────────────────
GBN_URL = "https://knowledgedefinednetworking.org/data/datasets_v1/gbnbw.tar.gz"
GBN_ARCHIVE = DATA_DIR / "gbn.tar.gz"
GBN_DIR = DATA_DIR / "gbnbw"


# ── ステップ 10: GBN ダウンロード ──────────────────────────────────────


def test_10_download_gbn():
    """GBN (17 ノード) データセットを KDN からダウンロード。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if GBN_ARCHIVE.exists():
        mb = GBN_ARCHIVE.stat().st_size / 1e6
        print(f"\nskip — already exists: {GBN_ARCHIVE} ({mb:.0f} MB)")
        return

    print(f"\ndownloading {GBN_URL} ...")

    def _progress(count, block, total):
        if total > 0:
            print(f"\r  {count * block / total * 100:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(GBN_URL, GBN_ARCHIVE, reporthook=_progress)
    mb = GBN_ARCHIVE.stat().st_size / 1e6
    print(f"\nsaved → {GBN_ARCHIVE} ({mb:.0f} MB)")


# ── ステップ 11: GBN 展開 ─────────────────────────────────────────────


def test_11_extract_gbn():
    """GBN アーカイブから TFRecords を展開する。"""
    assert GBN_ARCHIVE.exists(), (
        f"{GBN_ARCHIVE} が見つかりません。先に download_gbn を実行してください:\n"
        f"  uv run pytest tasks/ -s -k download_gbn"
    )

    existing = list(GBN_DIR.rglob("*.tfrecords")) if GBN_DIR.exists() else []
    if existing:
        print(
            f"\nskip — {len(existing)} 件の .tfrecords が既に {GBN_DIR}/ 以下に存在します"
        )
        return

    with tarfile.open(GBN_ARCHIVE, "r:gz") as tar:
        members = [m for m in tar.getmembers() if ".tfrecords" in m.name]
        print(f"\n{len(members)} 件の tfrecords を展開中 ...")
        tar.extractall(DATA_DIR, members=members, filter="data")

    train_n = len([p for p in GBN_DIR.rglob("*.tfrecords") if "train" in str(p)])
    eval_n = len([p for p in GBN_DIR.rglob("*.tfrecords") if "evaluat" in str(p)])
    print(f"train: {train_n} 件  eval: {eval_n} 件")


# ── ステップ 12: GBN 評価（スモークモデル） ───────────────────────────


def test_12_evaluate_gbn():
    """NSFNet で訓練したスモークモデルを GBN (17 ノード) で評価する。

    NSFNet (14 ノード) → GBN (17 ノード) のクロストポロジー汎化性能を確認する。
    """
    ckpt = MODEL_DIR / "ckpt.keras"
    assert ckpt.exists(), (
        f"{ckpt} が見つかりません。先に train を実行してください:\n"
        f"  uv run pytest tasks/ -s -k train"
    )

    assert GBN_DIR.exists(), (
        f"{GBN_DIR} が見つかりません。先に extract_gbn を実行してください:\n"
        f"  uv run pytest tasks/ -s -k extract_gbn"
    )

    gbn_eval_files = sorted(
        str(p) for p in GBN_DIR.rglob("*.tfrecords") if "evaluat" in str(p)
    )
    if not gbn_eval_files:
        gbn_eval_files = sorted(str(p) for p in GBN_DIR.rglob("*.tfrecords"))
    assert gbn_eval_files, (
        f"GBN tfrecords が {GBN_DIR} 以下に見つかりません。"
        f"先に extract_gbn を実行してください:\n  uv run pytest tasks/ -s -k extract_gbn"
    )
    print(f"\ngbn eval: {len(gbn_eval_files)} 件")

    model = tf.keras.models.load_model(
        str(ckpt),
        custom_objects={"DelayRouteNet": DelayRouteNet, "HParams": HParams},
    )

    eval_ds = tfrecord_input_fn(gbn_eval_files, HPARAMS, shuffle_buf=None)

    mae = tf.keras.metrics.MeanAbsoluteError()
    mre = MeanRelativeError()
    rho = PearsonCorrelation()
    n_samples = 0

    for features, labels in eval_ds.take(VALIDATION_STEPS):
        preds = model(features, training=False)
        delay_pred = preds[..., 0]
        mae.update_state(labels["delay"], delay_pred)
        mre.update_state(labels["delay"], delay_pred)
        rho.update_state(labels["delay"], delay_pred)
        n_samples += int(labels["delay"].shape[0])

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "topology": "GBN",
        "n_nodes": 17,
        "trained_on": "NSFNet",
        "model": "smoke",
        "eval_files": len(gbn_eval_files),
        "validation_steps": VALIDATION_STEPS,
        "n_samples": n_samples,
        "mae": float(mae.result().numpy()),
        "mre": float(mre.result().numpy()),
        "pearson_rho": float(rho.result().numpy()),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "eval_results_gbn.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"サンプル数: {results['n_samples']}")
    print(f"MAE:       {results['mae']:.4f}")
    print(f"MRE:       {results['mre']:.4f}")
    print(f"Pearson ρ: {results['pearson_rho']:.4f}")
    print(f"保存先 → {out}")


# ── ステップ 13: GBN 評価（論文レベルモデル） ─────────────────────────


def test_13_evaluate_gbn_full():
    """NSFNet で訓練した論文レベルモデルを GBN (17 ノード) で評価する。

    NSFNet (14 ノード) → GBN (17 ノード) のクロストポロジー汎化性能を確認する。
    """
    ckpt = MODEL_DIR_FULL / "ckpt.keras"
    assert ckpt.exists(), (
        f"{ckpt} が見つかりません。先に train_full を実行してください:\n"
        f"  uv run pytest tasks/ -s -k train_full"
    )

    assert GBN_DIR.exists(), (
        f"{GBN_DIR} が見つかりません。先に extract_gbn を実行してください:\n"
        f"  uv run pytest tasks/ -s -k extract_gbn"
    )

    gbn_eval_files = sorted(
        str(p) for p in GBN_DIR.rglob("*.tfrecords") if "evaluat" in str(p)
    )
    if not gbn_eval_files:
        gbn_eval_files = sorted(str(p) for p in GBN_DIR.rglob("*.tfrecords"))
    assert gbn_eval_files, (
        f"GBN tfrecords が {GBN_DIR} 以下に見つかりません。"
        f"先に extract_gbn を実行してください:\n  uv run pytest tasks/ -s -k extract_gbn"
    )
    print(f"\ngbn eval: {len(gbn_eval_files)} 件")

    model = tf.keras.models.load_model(
        str(ckpt),
        custom_objects={"DelayRouteNet": DelayRouteNet, "HParams": HParams},
    )

    eval_ds = tfrecord_input_fn(gbn_eval_files, HPARAMS_FULL, shuffle_buf=None)

    mae = tf.keras.metrics.MeanAbsoluteError()
    mre = MeanRelativeError()
    rho = PearsonCorrelation()
    n_samples = 0

    for features, labels in eval_ds.take(VALIDATION_STEPS_FULL):
        preds = model(features, training=False)
        delay_pred = preds[..., 0]
        mae.update_state(labels["delay"], delay_pred)
        mre.update_state(labels["delay"], delay_pred)
        rho.update_state(labels["delay"], delay_pred)
        n_samples += int(labels["delay"].shape[0])

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "topology": "GBN",
        "n_nodes": 17,
        "trained_on": "NSFNet",
        "model": "full",
        "eval_files": len(gbn_eval_files),
        "validation_steps": VALIDATION_STEPS_FULL,
        "n_samples": n_samples,
        "mae": float(mae.result().numpy()),
        "mre": float(mre.result().numpy()),
        "pearson_rho": float(rho.result().numpy()),
    }

    MODEL_DIR_FULL.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR_FULL / "eval_results_gbn.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"サンプル数: {results['n_samples']}")
    print(f"MAE:       {results['mae']:.4f}")
    print(f"MRE:       {results['mre']:.4f}")
    print(f"Pearson ρ: {results['pearson_rho']:.4f}")
    print(f"保存先 → {out}")


# ── ステップ 8: バッチサイズ比較スモーク ─────────────────────────────


def test_08_smoke_batch_size():
    """batch_size=32 vs 128 の epoch 時間を比較する（各 5 epoch）。

    OOM の有無と時間短縮効果を確認するためのスモークテスト。
    モデルは保存しない。
    """
    import time

    train_files = sorted(
        str(p) for p in DATA_DIR.rglob("*.tfrecords") if "train" in str(p)
    )
    assert train_files, (
        f"train 用 tfrecords が {DATA_DIR} 以下に見つかりません。"
        f"先に extract を実行してください:\n  uv run pytest tasks/ -s -k extract"
    )

    results = {}
    for batch_size in [32, 128]:
        hp = HParams(
            link_state_dim=16,
            path_state_dim=32,
            readout_units=256,
            T=8,
            dropout_rate=0.5,
            l2=0.1,
            batch_size=batch_size,
            readout_layers=2,
        )
        # バッチサイズに比例してステップ数を調整（見るサンプル数を揃える）
        steps = 200 * 32 // batch_size

        model = DelayRouteNet(hp, output_units=2)
        model.build()
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            hp.learning_rate, decay_steps=50000, decay_rate=0.9, staircase=True
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), run_eagerly=True)

        train_ds = tfrecord_input_fn(train_files, hp, shuffle_buf=SHUFFLE_BUF_FULL)

        print(f"\nbatch_size={batch_size}, steps_per_epoch={steps}")
        t0 = time.perf_counter()
        model.fit(train_ds, steps_per_epoch=steps, epochs=5, verbose=0)
        elapsed = time.perf_counter() - t0

        sec_per_epoch = elapsed / 5
        results[batch_size] = sec_per_epoch
        print(f"  elapsed: {elapsed:.1f}s  ({sec_per_epoch:.1f}s/epoch)")

    bs32, bs128 = results[32], results[128]
    speedup = bs32 / bs128
    print(
        f"\nbatch_size=128 は batch_size=32 の {speedup:.2f}x {'速い' if speedup > 1 else '遅い'}"
    )
