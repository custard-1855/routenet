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

import tarfile
import urllib.request
from pathlib import Path

import tensorflow as tf

from routenet.routenet import HParams, DelayRouteNet, tfrecord_input_fn

# ── 設定 ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATASET_URL = "https://knowledgedefinednetworking.org/data/datasets_v1/nsfnetbw.tar.gz"
DATASET_ARCHIVE = DATA_DIR / "nsfnetbw.tar.gz"
MODEL_DIR = Path("models") / "delay"

HPARAMS = HParams(batch_size=32, T=3, readout_layers=2)
STEPS_PER_EPOCH = 200
EPOCHS = 20
VALIDATION_STEPS = 50


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
        print(f"\nskip — {len(existing)} 件の .tfrecords が既に {DATA_DIR}/ 以下に存在します")
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
    model.compile(optimizer=tf.keras.optimizers.Adam(lr))

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
                str(MODEL_DIR / "ckpt"),
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(str(MODEL_DIR / "logs")),
            tf.keras.callbacks.CSVLogger(str(MODEL_DIR / "training.csv")),
        ],
    )
    print(f"\nモデル保存先 → {MODEL_DIR}/ckpt")
