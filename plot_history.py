import matplotlib.pyplot as plt

# Data taken from your training logs in the first screenshot
epochs = range(1, 11)
train_acc = [0.86, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.93, 0.93, 0.93]
val_acc = [0.75, 0.78, 0.82, 0.85, 0.84, 0.86, 0.87, 0.88, 0.86, 0.88]

plt.figure(figsize=(10, 5))

# Plot Accuracy
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'go-', label='Validation Accuracy')
plt.title('Model Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.show()
