# Open-AI GPT 2 Implementation in Tensorflow

This Repository provides ```Tensorflow.keras``` implementation for [Open-AI GPT-2](https://github.com/openai/gpt-2).
The implementation is modular using ```tf.keras``` subclassing method. It is compatible with both _Graph Mode_ and _Eager Mode_
execution.

This repository is built under __Tensorflow 1.13.1__, but is compatible with __Tensorflow 2.0__.

### Example

The config file for model is same as original model.

```python
import json

with open("117M/hparams.json") as f:
    config = json.load(f)
```
For creating model, it is enough to pass config as a dictionary

```python
from gpt2 import GPT2

model = GPT2(config=config, name="gpt2")
```

Using the model is simple just as any other ```tf.keras.Model```.

```python
import tensorflow as tf

x = tf.placeholder(dtype=tf.int32, shape=[None, None])
y = model(x)
```

You can use more options in calling the model in both _Graph_ and _Eager_ mode.

In order to load pre-trained weights from original checkpoint, use builder.

