### Adaptive layer normalization

Maths of adaLN:
```python
normalized_input = (input - mean(input)) / std(input)
shifted_input = normalized_input + shift
scaled_input = shifted_input * scale
```

**Adaptive Layer Normalization (AdaLN)** is a technique that introduces conditional information into Layer Normalization (LN).
This allows the normalization process to adapt to different input distributions, enhancing the model's ability to handle diverse data.

python/pytorch implementation:
```python
layer_norm = nn.LayerNorm()

def adaLN_modulate(self, x, shift_value, scale_value):
    x = layer_norm(x)
    x = x * (1 + scale_value.unsqueeze(1))
    x = x + shift_value.unsqueeze(1)

    return x
```
### Classifier-free guidance
**Classifier-Free Guidance (CFG)** is a technique used in generative models, particularly diffusion models, to improve the quality and controllability of generated samples. It works by introducing a noise schedule that gradually reduces the amount of noise added to the data during training. This noise schedule is controlled by a guidance scale, which determines how much the model should focus on the original data versus the noise.

**components**
- Noise Schedule: This is a predefined schedule that determines how much noise is added to the data at each step of the training process.
- Guidance Scale: This parameter controls the balance between the original data and the noise. A higher guidance scale means the model will focus more on the original data, while a lower guidance scale means the model will focus more on the noise.

**Mechanism**

1. **Training:**
   - The model is trained on noisy versions of the data.
   - The noise schedule is used to gradually reduce the amount of noise added to the data over time.
   - The guidance scale is used to control how much the model should focus on the original data versus the noise.

2. **Sampling:**
   - To generate a new sample, the model starts with a random noise vector.
   - The noise schedule is used to gradually reduce the amount of noise in the vector.
   - At each step, the model is asked to predict the original data based on the current noisy vector.
   - The guidance scale is used to control how much the model's prediction should be influenced by the original data.

By adjusting the guidance scale, we can control the trade-off between fidelity (how closely the generated samples resemble the original data) and diversity (how different the generated samples are from each other). A higher guidance scale will produce more faithful but less diverse samples, while a lower guidance scale will produce more diverse but less faithful samples.

Advantages:
- Improved sample quality:
- Increased controllability: control the trade-off between fidelity and diversity.
- Simpler implementation and does not require a separate classifier

CFG has become a popular technique in generative modeling and has been used in various applications, such as image generation, text generation, and audio synthesis.
