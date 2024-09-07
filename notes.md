Maths of adaLN:
```python
normalized_input = (input - mean(input)) / std(input)
shifted_input = normalized_input + shift
scaled_input = shifted_input * scale
```

Adaptive Layer Normalization (AdaLN) is a technique that introduces conditional information into Layer Normalization (LN).
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

