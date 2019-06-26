import tensorflow as tf
import numpy as np
import gpt2
import original_gpt2
import argparse
import json

class ReArrange(object):
    """
    Map original model weights to our tf.keras model weights.
    """

    @classmethod
    def Embedding(cls, weights):
        weights = [weights[1], weights[0]]
        new_weights = []
        for w in weights:
            name = w.name.split(":")[0]
            name = name + "/rearrange"
            new_w = tf.identity(w, name=name)
            new_weights.append(new_w)
        return new_weights

    @classmethod
    def LayerNorm(cls, weights):
        weights = [weights[1], weights[0]]
        new_weights = []
        for w in weights:
            name = w.name.split(":")[0]
            name = name + "/rearrange"
            new_w = tf.identity(w, name=name)
            new_weights.append(new_w)
        return new_weights

    @classmethod
    def SelfAttention(cls, weights):
        kernels = []
        biases = []
        new_names = ["query", "key", "value"]
        kernel = weights[0]
        name = kernel.name.split(":")[0]
        shape = kernel.shape.as_list()
        new_shape = [shape[1], 3, shape[2] // 3]
        kernel = tf.reshape(kernel, new_shape)
        _kernels = tf.unstack(kernel, axis=1)
        for kernel, nm in zip(_kernels, new_names):
            _kernel = tf.identity(kernel, name=name + "/rearrange/" + nm)
            kernels.append(_kernel)
        bias = weights[1]
        name = bias.name.split(":")[0]
        shape = bias.shape.as_list()
        new_shape = [3, shape[0] // 3]
        bias = tf.reshape(bias, new_shape)
        _biases = tf.unstack(bias, axis=0)
        for bias, nm in zip(_biases, new_names):
            _bias = tf.identity(bias, name=name + "/rearrange/" + nm)
            biases.append(_bias)
        weights = []
        for kernel, bias in zip(kernels, biases):
            weights.append(kernel)
            weights.append(bias)
        return weights

    @classmethod
    def Dense(cls, weights):
        kernel = weights[0]
        name = kernel.name.split(":")[0]
        kernel = tf.squeeze(kernel, 0, name=name + "/rearrange")
        bias = weights[1]
        name = bias.name.split(":")[0]
        bias = tf.identity(bias, name=name + "/rearrange")
        return [kernel, bias]

    @classmethod
    def Attention(cls, weights):
        new_weights = []
        new_weights = new_weights + cls.LayerNorm(weights[0:2])
        new_weights = new_weights + cls.SelfAttention(weights[2:4])
        new_weights = new_weights + cls.Dense(weights[4:6])
        return new_weights

    @classmethod
    def MLP(cls, weights):
        new_weights = []
        new_weights = new_weights + cls.LayerNorm(weights[0:2])
        new_weights = new_weights + cls.Dense(weights[2:4])
        new_weights = new_weights + cls.Dense(weights[4:6])
        return new_weights

    @classmethod
    def Block(cls, weights):
        return cls.Attention(weights[0:6]) + cls.MLP(weights[6:12])

    @classmethod
    def Transformer(cls, weights):
        blocks_num = (len(weights) - 2) // 12
        blocks_weights = weights[0:-2]
        new_weights = []
        for block in range(blocks_num):
            new_weights = new_weights + cls.Block(blocks_weights[block * 12:(block + 1) * 12])
        new_weights = new_weights + cls.LayerNorm(weights[-2:])
        return new_weights

    @classmethod
    def GPT2(cls, weights):
        return cls.Embedding(weights[0:2]) + cls.Transformer(weights[2:])


def build(config, checkpoint_path, session=None, name=None):
    """

    Build a GPT2 model (in tf.keras format) from pre-trained original checkpoint.

    Args:
        config: A dictionary, for model hyper parameters.
        checkpoint_path: path of original checkpoint.
        session: since it's needed to load weights on tf.Session in Graph Execution, a session should be
                 passed to this method, or else the method uses default session if it exist. in Eager Execution
                 there is no need for this argument.
        name: name of model.

    Returns:
        a GPT2 model which the pre-trained weights are loaded.

    """
    if name is None:
        name = "gpt2"
    conf = tf.ConfigProto(device_count={'GPU': 0})
    with open(config) as f:
        config = json.load(f)
    graph = tf.Graph()
    with graph.as_default():
        x = tf.ones(shape=(1, 1), dtype=tf.int32)
        hparams = original_gpt2.default_hparams()
        hparams.override_from_dict(config)
        _ = original_gpt2.model(hparams, x)
        original_weights = tf.global_variables()
        original_weights = ReArrange.GPT2(original_weights)
        saver = tf.train.Saver()
        sess = tf.Session(config=conf, graph=graph)
        saver.restore(sess=sess, save_path=checkpoint_path)
    original_weights = sess.run(original_weights)
    sess.close()
    eager = session is None and tf.executing_eagerly()
    def _build():
        x = tf.ones(shape=(1, 1), dtype=tf.int32)
        model = gpt2.GPT2(config=config, name=name)
        y = model(x)
        weights = model.weights
        assigns = [u.assign(v) for u, v in zip(weights, original_weights)]
        return model, assigns
    if eager:
        model, _ = _build()
    else:
        if session is None:
            try:
                session = tf.get_default_session()
            except:
                raise Exception("No session is given and there is no default session")
        with session.graph.as_default():
            model, assigns = _build()
        session.run(assigns)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file path", required=True)
    parser.add_argument("--checkpoint", help="checkpoint file path", required=True)
    parser.add_argument("--target", help="target h5 file path", required=True)
    args = parser.parse_args()
    conf = tf.ConfigProto(device_count={'GPU': 0})
    tf.enable_eager_execution(config=conf)
    model = build(args.config, args.checkpoint, name="bert")
    model.save_weights(args.target, save_format="h5")
