# Torch_tflite


To convert pytorch models to TFlite (.tflite) using ai-edge-torch first download ai-edge-torch (pip install ai-edge-torch)

Preparing the PyTorch Model:

Ensure that the PyTorch model is in an exportable format  (.export()) with any custom operations or layers translated to compatible ones. 
Then

- Define your model artitechture and set to eval. 
- load weights (make sure strict=True)
- create wrapper if needed 
<img width="533" alt="image" src="https://github.com/user-attachments/assets/e5c29f79-580c-474e-b742-4ec1dacff318">

- use ai_edge_torch.convert() to convert wrapped model to edge model
- Use edge_model.export() to convert to tflite
- debug any errors using the documentation https://github.com/google-ai-edge/ai-edge-torch/blob/c9973d2e7423e86f420576c0e5cac1181f79ac0e/docs/pytorch_converter/README.md

Example notebook included in repository. 
