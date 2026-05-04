import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0


model = keras.Sequential([
   
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),          
    layers.Dense(64, activation="relu"),   
    layers.Dropout(0.3),       
    layers.Dense(10, activation="softmax") 
], name="mnist_cnn")

model.summary()

# Compilação do modelo 
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Treinamento do modelo
print("\n Iniciando treinamento...\n")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,  
    verbose=1
)


print("\n Avaliando no conjunto de teste...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n Acurácia final no teste: {test_accuracy * 100:.2f}%")
print(f"   Loss final no teste:     {test_loss:.4f}")


model.save("model.h5")
print("\n Modelo salvo em: modelo.h5!")