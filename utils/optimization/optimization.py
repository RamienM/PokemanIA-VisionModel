import onnx
import onnxruntime as ort
import torch
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse
import os

# Argumentos de entrada
file = "Stelle_Seg_prefinal.onnx"

# Verifica que el archivo existe
if not os.path.isfile(file):
    raise FileNotFoundError(f"No se encontró el archivo:{file}")

# Cargar y verificar el modelo ONNX
print(f"Cargando modelo: {file}")
onnx_model = onnx.load(file)
onnx.checker.check_model(onnx_model)
print("✅ Modelo ONNX válido.")

# Crear sesión de inferencia ONNX Runtime
session = ort.InferenceSession(file, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Inferencia de prueba con entrada aleatoria
print("⏳ Ejecutando inferencia de prueba...")
input_shape = session.get_inputs()[0].shape
input_shape = [s if isinstance(s, int) else 1 for s in input_shape]  # reemplaza 'batch_size' por 1
input_tensor = torch.randn(*input_shape)
result = session.run([output_name], {input_name: input_tensor.numpy()})
print("✅ Inferencia original completada.")

# Cuantizar dinámicamente el modelo
quantized_model_path = file.replace(".onnx", "_quant.onnx")
print("⚙️ Cuantizando modelo (INT8)...")
quantize_dynamic(
    model_input=file,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8
)
print(f"✅ Modelo cuantizado guardado como: {quantized_model_path}")

# Verificación con modelo cuantizado
quant_session = ort.InferenceSession(quantized_model_path, providers=["CPUExecutionProvider"])
quant_result = quant_session.run([output_name], {input_name: input_tensor.numpy()})
print("✅ Inferencia cuantizada completada.")
