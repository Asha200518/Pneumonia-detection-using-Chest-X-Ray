import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load the model
model = load_model('pneumonia_model.h5')

# 2. Setup the test data generator
# This automatically finds all images in your 'test' subfolders
test_dir = r'Datasets\chest_xray\chest_xray\test'
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Crucial: keeps images in order to match labels
)

# 3. Run predictions on EVERYTHING in the test folder
print("Evaluating... this may take a minute.")
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = test_generator.classes

# 4. Generate the Statistics
print("\n--- Final Statistics ---")
print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

# 5. Plot the Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NORMAL', 'PNEUMONIA'], 
            yticklabels=['NORMAL', 'PNEUMONIA'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
