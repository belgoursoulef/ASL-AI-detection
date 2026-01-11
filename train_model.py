import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load your collected data
X = np.load('landmarks.npy')
y = np.load('labels.npy')

print("Data shape:", X.shape)
print("Labels shape:", y.shape)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)), 
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.fit(
    X,
    y_categorical,
    epochs=30,
    batch_size=8,
    validation_split=0.1
)


model.save('sign_model.h5')

print("Training complete. Model saved as sign_model.h5")
