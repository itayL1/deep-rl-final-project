def f(output_size, ksize, stride):
    return (output_size - 1) * stride + ksize


ordered_layers = [
    dict(ksize=5, strides=2),
    dict(ksize=4, strides=2),
    dict(ksize=3, strides=2),
    dict(ksize=2, strides=2),

    dict(ksize=4, strides=1),
    dict(ksize=4, strides=1),
]
last_layer_output_size = 1

curr_layer_receptive_field_size = last_layer_output_size
for layer in reversed(ordered_layers):
    curr_layer_receptive_field_size = f(curr_layer_receptive_field_size, layer['ksize'], layer['strides'])
print(f"curr_layer_receptive_field_size: {curr_layer_receptive_field_size}")

