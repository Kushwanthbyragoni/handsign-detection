import os
import glob
import json
import time
from collections import Counter, OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ----------------- CONFIG -----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))         # src/
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))      # project root
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 160           # keep matched with app.py; change to 128 for faster runs if needed
BATCH_SIZE = 32          # reduce if you still get OOM (try 16)
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 8
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------- Helper: collect filepaths + labels -----------------
def collect_image_paths_and_labels(roots):
    """
    Walk the given roots and return:
      paths: list of file paths
      labels: list of integer labels
      label_mapping: {class_name: idx}
    This only collects file paths (no image reading) -> memory light.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    class_names = []
    # detect class dirs
    for root in roots:
        if not os.path.exists(root):
            continue
        for cur, dirs, files in os.walk(root):
            # if this dir contains any image files, treat basename as class
            found = False
            for e in exts:
                if glob.glob(os.path.join(cur, e)):
                    found = True
                    break
            if found:
                class_names.append(os.path.basename(cur))
    # unique preserve order
    class_names = list(OrderedDict.fromkeys([c for c in class_names if c and c != os.path.basename(root)]))
    if not class_names:
        # fallback: include leaf folders under roots
        class_names = []
        for root in roots:
            if not os.path.exists(root):
                continue
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if os.path.isdir(p):
                    class_names.append(name)
        class_names = sorted(list(set(class_names)))
    class_names = sorted(list(set(class_names)))
    label_mapping = {name: idx for idx, name in enumerate(class_names)}

    paths = []
    labels = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for cur, dirs, files in os.walk(root):
            # gather image files in cur
            img_files = []
            for e in exts:
                img_files.extend(glob.glob(os.path.join(cur, e)))
            if not img_files:
                continue
            cls = os.path.basename(cur)
            if cls not in label_mapping:
                # skip unexpected folders
                continue
            idx = label_mapping[cls]
            for p in img_files:
                paths.append(p)
                labels.append(idx)

    if not paths:
        raise RuntimeError("No images found in dataset roots: " + ", ".join(roots))
    return paths, labels, label_mapping

# ----------------- tf.data pipeline -----------------
def make_dataset_from_files(paths, labels, shuffle=True, batch_size=BATCH_SIZE):
    """
    Create a tf.data.Dataset that reads images from disk, decodes, resizes, normalizes.
    Returns dataset ready to feed model.fit()
    """
    paths = np.array(paths, dtype=str)
    labels = np.array(labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(paths)), seed=SEED)

    def _load_and_preprocess(path, label):
        # read file
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return img, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# ----------------- Model builder -----------------
def create_model(num_classes):
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet", alpha=0.75)
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model, base

# ----------------- Training -----------------
def main():
    start = time.time()
    print("\n" + "=" * 60)
    print("Memory-safe Hand Sign Training (tf.data streaming)")
    print("=" * 60 + "\n")

    roots = [
        os.path.join(DATA_DIR, "Gesture Image Data"),
        os.path.join(DATA_DIR, "Gesture Image Pre-Processed Data"),
    ]
    roots = [r for r in roots if os.path.exists(r)]
    if not roots:
        raise RuntimeError("No dataset roots found. Check data/Gesture Image Data/ ...")

    print("Dataset roots used:")
    for r in roots:
        print("  -", r)
    print()

    # collect file paths and labels (no image read yet)
    paths, labels, label_mapping = collect_image_paths_and_labels(roots)
    print(f"Total image files found: {len(paths)}")
    counts = Counter(labels)
    print("Samples per class (label idx -> count):")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")
    print()

    # save label mapping
    label_map_path = os.path.join(MODEL_DIR, "label_mapping.json")
    with open(label_map_path, "w") as f:
        json.dump(label_mapping, f, indent=4)
    print("Saved label mapping to", label_map_path)

    # split -> train/val/test (by indices)
    # convert to numpy arrays for indexing
    paths = np.array(paths)
    labels = np.array(labels)
    # deterministic shuffle before splitting
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(paths))
    paths = paths[order]; labels = labels[order]

    n = len(paths)
    test_split = int(n * 0.15)
    val_split = int((n - test_split) * 0.15)

    test_paths = paths[:test_split]; test_labels = labels[:test_split]
    rest_paths = paths[test_split:]; rest_labels = labels[test_split:]
    val_paths = rest_paths[:val_split]; val_labels = rest_labels[:val_split]
    train_paths = rest_paths[val_split:]; train_labels = rest_labels[val_split:]

    print(f"Dataset sizes -> Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}\n")

    # Build datasets using tf.data (streamed)
    train_ds = make_dataset_from_files(train_paths, train_labels, shuffle=True, batch_size=BATCH_SIZE)
    val_ds = make_dataset_from_files(val_paths, val_labels, shuffle=False, batch_size=BATCH_SIZE)
    test_ds = make_dataset_from_files(test_paths, test_labels, shuffle=False, batch_size=BATCH_SIZE)

    # Create model
    model, base = create_model(len(label_mapping))
    model.summary()

    # Phase 1: train head
    model.compile(optimizer=optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    ckpt1 = ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"), save_best_only=True, monitor="val_accuracy", mode="max", verbose=1)
    reduce1 = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    es1 = EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1)

    history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1, callbacks=[ckpt1, reduce1, es1], verbose=1)

    # Phase 2: fine-tune last layers
    base.trainable = True
    for layer in base.layers[:-60]:
        layer.trainable = False

    model.compile(optimizer=optimizers.Adam(8e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    ckpt2 = ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"), save_best_only=True, monitor="val_accuracy", mode="max", verbose=1)
    reduce2 = ReduceLROnPlateau(monitor="val_loss", factor=0.6, patience=2, verbose=1, min_lr=1e-7)
    es2 = EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1)

    history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE2, callbacks=[ckpt2, reduce2, es2], verbose=1)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print("\n" + "=" * 60)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")
    print("=" * 60)

    # Save final model (native keras + h5 for compatibility)
    model.save(os.path.join(MODEL_DIR, "asl_model.keras"))
    model.save(os.path.join(MODEL_DIR, "asl_model.h5"))
    print("Saved models to", MODEL_DIR)

    total_mins = (time.time() - start) / 60.0
    print(f"Total training time: {total_mins:.2f} minutes")

if __name__ == "__main__":
    main()
