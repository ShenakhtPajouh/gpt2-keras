# Builder for Open-AI GPT 2

```builder.py``` provides the ability of build a pre-trained model of __GPT2__ in 
our implementation format using an original checkpoint.

The best way of using this abilty is run the code to make and __.h5__ file of model weights and load this weights your code.
An example of using ```builder.py``` is presented below:

```bash
#  you should pass checkpoint, config and target .h5 paths
#  as arguments.
$   python builder.py --config=/tmp/model/117M/hparams.json \
    --checkpoint=/tmp/model/117M/model.ckpt \
    --target=/tmp/model/117M/weights.h5
```

Then in your code you should load weights:

```python
# ...
model.load_weights("/tmp/model/117M/weights.h5")
# ...
```

You can also use ```builder.build``` method in your code directly.

