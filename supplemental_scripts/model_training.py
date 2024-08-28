import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.data import AUTOTUNE

dataset_path = './data_collection/training_data/'

train_dataset = image_dataset_from_directory(
    dataset_path,
    image_size=(50, 50),
    batch_size=32,
    label_mode='categorical',
    color_mode='grayscale',
)

val_size = 0.2
train_size = int((1 - val_size) * len(train_dataset))
train_dataset = train_dataset.take(train_size).repeat().prefetch(buffer_size=AUTOTUNE)
val_dataset = train_dataset.skip(train_size)#.repeat().prefetch(buffer_size=AUTOTUNE)

num_classes = 26

model=Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.summary()
print(val_dataset)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    steps_per_epoch=800,
    verbose=1,
    validation_steps=200
)
print("Training Complete")

val_loss, val_accuracy = model.evaluate(val_dataset, steps=100)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

model.save('scrabble_cnn_model.keras')
