import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers, callbacks
import os
base_dir = '/content/drive/MyDrive/PROJECT C/cattle_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
# STEP 4: Image settings
img_size = (224, 224)
batch_size = 32
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)
train_data = train_gen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical'
)
test_data = test_gen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', shuffle=False
)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze VGG16 layers
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')
])
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model_ckpt = callbacks.ModelCheckpoint(
    '/content/drive/MyDrive/PROJECT C/vgg16_cattle_model.h5',
    save_best_only=True
)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop, model_ckpt]
)
test_loss, test_acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {test_acc:.4f}")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255
test_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/PROJECT C/cattle_dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)
cm = confusion_matrix(y_true, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
display.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.title('Model Training History')
plt.legend()
plt.grid(True)
plt.show()

