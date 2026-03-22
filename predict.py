import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# 1. Load your trained model
model = load_model('pneumonia_model.h5')

# 2. Select your image (Make sure this filename exists!)
img_path = r'Datasets\chest_xray\chest_xray\test\NORMAL\IM-0016-0001.jpeg'

# 3. Process the image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 4. Predict
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    result = "PNEUMONIA DETECTED"
else:
    result = "NORMAL"

# 5. Show result
plt.imshow(img)
plt.title(f"Result: {result}")
plt.show()
