import tensorflow as tf
import os

print("Carregando modelo treinado (model.h5)...")
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado com sucesso.")

# Exibe o tamanho original do modelo .h5
original_size = os.path.getsize("model.h5")
print(f"\n Tamanho original (.h5): {original_size / 1024:.1f} KB")

print("\n niciando conversão para TensorFlow Lite...")
print("    Técnica: Dynamic Range Quantization (float32 → int8 nos pesos)")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Executando a conversão para tflite
tflite_model = converter.convert()

# Salva modelo otimizado tflite
output_path = "model.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

# Gera relatório de otimização
optimized_size = os.path.getsize(output_path)
reduction = (1 - optimized_size / original_size) * 100

print(f"\n Modelo TFLite salvo em: {output_path}")
print(f"\n Relatório de Otimização:")
print(f"   Tamanho original  (.h5):     {original_size / 1024:.1f} KB")
print(f"   Tamanho otimizado (.tflite): {optimized_size / 1024:.1f} KB")
print(f"   Redução de tamanho:          {reduction:.1f}%")
print(f"\n Modelo pronto para deployment em dispositivos Edge AI.")
