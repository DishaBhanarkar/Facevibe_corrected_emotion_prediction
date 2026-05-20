import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

print("TF version:", tf.__version__)

IMG_SIZE   = 48
BATCH_SIZE = 8
TRAIN_PATH = "backend/archive/train"
TEST_PATH  = "backend/archive/test"

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names
print("Classes:", class_names)

class_counts = []
for cls in class_names:
    path = os.path.join(TRAIN_PATH, cls)
    class_counts.append(len(os.listdir(path)))

total = sum(class_counts)
print("\nSamples per emotion:")
for name, count in zip(class_names, class_counts):
    print(f"  {name}: {count}")

all_labels = []
for i, count in enumerate(class_counts):
    all_labels.extend([i] * count)

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
class_weight_dict = dict(enumerate(class_weights_array))
print("\nClass weights:", {class_names[i]: round(w, 2) for i, w in class_weight_dict.items()})

norm = layers.Rescaling(1./255)

def prep(ds):
    return ds.map(
        lambda x, y: (norm(x), y),
        num_parallel_calls=1
    ).prefetch(1)

train_data = prep(train_data)
val_data   = prep(val_data)
test_data  = prep(test_data)


reg = tf.keras.regularizers.l2(1e-4)

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Block 1
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    # Head
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=reg),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(7, activation='softmax')
])

model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "emotion_model.keras",
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=3,
        factor=0.5,
        min_lr=1e-6,
        verbose=1
    )
]


print("\n===== Training started =====\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    callbacks=callbacks,
    class_weight=class_weight_dict     
)


print("\n===== Test set evaluation =====")
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.4f}")


print("\n===== Sanity check: predictions on test set =====")
preds_list = []
for images, labels in test_data.take(20):
    preds = model.predict(images, verbose=0)
    preds_list.extend(np.argmax(preds, axis=1))

unique, counts = np.unique(preds_list, return_counts=True)
print("Predicted classes and how often:")
for u, c in zip(unique, counts):
    print(f"  {class_names[u]}: {c} times")

print("\n✅ Done! Model saved as emotion_model.keras")
print("Class order:", class_names)