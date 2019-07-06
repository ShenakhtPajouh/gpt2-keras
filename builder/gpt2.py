import tensorflow as tf
import numpy as np


def get_tensor_shape(x):
    x = tf.convert_to_tensor(x)
    static_shape = x.shape.as_list()
    if tf.executing_eagerly():
        return static_shape
    dynamic_shape = tf.shape(x)
    if static_shape is None:
        return dynamic_shape
    dynamic_shape = tf.unstack(dynamic_shape)
    shape = []
    for st, dyn in zip(static_shape, dynamic_shape):
        if st is None:
            shape.append(dyn)
        else:
            shape.append(st)
    return shape


def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def dropout_fn(x, dropout):
    if dropout is None or dropout == 0.0:
        return x
    else:
        return tf.nn.dropout(x, rate=dropout)


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, trainable=True, name=None):
        super().__init__(name=name, trainable=trainable)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        self.beta = self.add_weight(name="beta", shape=input_shape[-1:], initializer=tf.zeros_initializer())
        self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:], initializer=tf.ones_initializer())
        super().build(input_shape)

    def call(self, inputs, axis=-1, epsilon=1e-5):
        mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
        rdev = tf.math.rsqrt(variance + epsilon)
        x = (inputs - mean) * rdev
        output = x * self.gamma + self.beta
        return output

    def __call__(self, inputs, axis=-1, epsilon=1e-5):
        return super().__call__(inputs=inputs,
                                axis=axis, epsilon=epsilon)


class SelfAttention(tf.keras.layers.Layer):

    def __init__(self, num_attention_heads=1, size_per_head=512,
                 one_sided=True,
                 query_act=None,
                 initializer_range=0.02,
                 value_act=None,
                 key_act=None,
                 trainable=True,
                 name=None):
        super().__init__(name=name, trainable=trainable)
        # `query_layer` = [B*F, N*H]
        self.attention_size = num_attention_heads * size_per_head
        self.query_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=query_act,
            name="query",
            kernel_initializer=tf.random_normal_initializer(stddev=initializer_range)
        )
        # `key_layer` = [B*T, N*H]
        self.key_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=tf.random_normal_initializer(stddev=initializer_range)
        )
        # `value_layer` = [B*T, N*H]
        self.value_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
            kernel_initializer=tf.random_normal_initializer(stddev=initializer_range)
        )
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.one_sided = one_sided

    def reshape(self, x, use_2d=False, shape=None):
        if use_2d:
            batch_size, seq_length = shape[0], shape[1]
        else:
            _shape = get_tensor_shape(x)
            batch_size, seq_length = _shape[0], _shape[1]
        x = tf.reshape(x, [batch_size, seq_length, self.num_attention_heads, self.size_per_head])
        x = tf.transpose(x, [0, 2, 1, 3])
        return x

    def final_shape(self, x, use_2d=False):
        shape = get_tensor_shape(x)
        batch_size, seq_length = shape[0], shape[2]
        x = tf.transpose(x, [0, 2, 1, 3])
        if use_2d:
            x = tf.reshape(x, [batch_size * seq_length, self.num_attention_heads * self.size_per_head])
        else:
            x = tf.reshape(x, [batch_size, seq_length, self.num_attention_heads * self.size_per_head])
        return x

    def get_mask(self, inputs_shape, cache_length=None, mask=None):
        batch_size, seq_length = inputs_shape[0], inputs_shape[2]
        if self.one_sided:
            rng = tf.range(seq_length)
            one_sided_mask = tf.less_equal(rng, tf.expand_dims(rng, 1))
            if cache_length is not None:
                prev_mask = tf.ones([seq_length, cache_length], tf.bool)
                one_sided_mask = tf.concat([prev_mask, one_sided_mask], 1)
        if mask is not None:
            if cache_length is not None:
                prev_mask = tf.ones([batch_size, cache_length], tf.bool)
                mask = tf.concat([prev_mask, mask], 1)
            if cache_length is None:
                cache_length = 0
            mask = tf.reshape(mask, [batch_size, 1, 1, seq_length + cache_length])
        if self.one_sided:
            if mask is not None:
                one_sided_mask = tf.logical_and(mask, one_sided_mask)
            return one_sided_mask
        else:
            return mask

    def attend(self, query, key, value, mask=None, dropout=None):
        _sqrt = tf.math.sqrt(self.size_per_head)
        _sqrt = tf.cast(_sqrt, query.dtype)
        coefficients = tf.matmul(query, key, transpose_b=True) / _sqrt
        if mask is not None:
            mask = tf.cast(mask, coefficients.dtype)
            coefficients = coefficients * mask - (1 - mask) * 1e5
        coefficients = tf.math.softmax(coefficients, -1)
        coefficients = dropout_fn(coefficients, dropout)
        results = tf.matmul(coefficients, value)
        return results

    def call(self, inputs, cache=None, mask=None,
             attention_dropout=None, return_cache=False,
             use_2d=False, shape=None):
        """
        inputs: a tensor of shape [batch_size, seq_length, dim] if use_2d is false,
                else a tensor of shape [batch_size * seq_length, dim]
        cache: A dictionary consist of key and value from previous calls.
        mask: a boolean tensor of shape [batch_size, seq_length]
        attention_probs_dropout_prob: dropout use for attention mechanism
        return_cache: if True, it returns key and values as besides layer output
        use_2d: if it is True, the model uses 2D matrices as inputs and outputs
        shape: if use_2d is True, then the shape is [batch_size, seq_length]
        """
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        if use_2d and shape is None:
            raise ValueError("if use_2d is True, then the shape must be specified")
        query = self.reshape(query, use_2d, shape)
        key = self.reshape(key, use_2d, shape)
        value = self.reshape(value, use_2d, shape)
        cache_length = None
        if cache is not None:
            key = tf.concat([cache["key"], key], 2)
            value = tf.concat([cache["value"], value], 2)
            cache_length = get_tensor_shape(cache["key"])[2]
        mask = self.get_mask(inputs, cache_length, mask)
        result = self.attend(query, key, value, mask, attention_dropout)
        result = self.final_shape(result, use_2d)
        if return_cache:
            cache = {"key": key, "value": value}
            return result, cache
        else:
            return result

    def __call__(self, inputs, cache=None, mask=None,
             attention_dropout=None, return_cache=False,
             use_2d=False, shape=None):
        """
        inputs: a tensor of shape [batch_size, seq_length, dim] if use_2d is false,
                else a tensor of shape [batch_size * seq_length, dim]
        cache: A dictionary consist of key and value from previous calls.
        mask: a boolean tensor of shape [batch_size, seq_length]
        attention_probs_dropout_prob: dropout use for attention mechanism
        return_cache: if True, it returns key and values as besides layer output
        use_2d: if it is True, the model uses 2D matrices as inputs and outputs
        shape: if use_2d is True, then the shape is [batch_size, seq_length]
        """
        return super().__call__(
            inputs=inputs,
            cache=cache,
            mask=mask,
            attention_dropout=attention_dropout,
            return_cache=return_cache,
            use_2d=use_2d,
            shape=shape
        )

class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, config, name=None, trainable=True, initializer_range=0.02):
        super().__init__(name=name, trainable=trainable)
        self.layer_norm = LayerNormalization(name="layer_norm")
        self.self_attention = SelfAttention(num_attention_heads=config["n_head"],
                                            size_per_head=config["n_embd"] // config["n_head"],
                                            initializer_range=initializer_range,
                                            name="self"
                                            )
        self.projection = tf.keras.layers.Dense(units=config["n_embd"],
                                                kernel_initializer=tf.random_normal_initializer(stddev=initializer_range),
                                                name="projection")


    def call(self, inputs, cache=None, dropout=None, attention_dropout=None,
             return_cache=False, use_2d=False, shape=None):
        """

        inputs: a tensor of shape [batch_size, seq_length, dim] if use_2d is False, else [batch_size * seq_length, dim]
        cache: (Optional): a dictionary of tensors key and value from previous calls.
        return_cache: if True, returns a dictionary of key and value tensors besides layer output.
        use_2d: if is True then the inputs and outputs are 2D tensors instead of 3D (for tpu performance)
        shape: if use_2d then it's [batch_size, seq_length]
        """
        x = self.layer_norm(inputs)
        x = self.self_attention(x, attention_dropout=attention_dropout,
                                cache=cache,
                                return_cache=return_cache,
                                use_2d=use_2d,
                                shape=shape)
        if return_cache:
            x, cache = x
        x = self.projection(x)
        x = dropout_fn(x, dropout)
        if return_cache:
            return x, cache
        else:
            return x

    def __call__(self, inputs, cache=None, dropout=None, attention_dropout=None,
                 return_cache=False, use_2d=False, shape=None):
        """

        inputs: a tensor of shape [batch_size, seq_length, dim] if use_2d is False, else [batch_size * seq_length, dim]
        cache: (Optional): a dictionary of tensors key and value from previous calls.
        return_cache: if True, returns a dictionary of key and value tensors besides layer output.
        use_2d: if is True then the inputs and outputs are 2D tensors instead of 3D (for tpu performance)
        shape: if use_2d then it's [batch_size, seq_length]
        """
        return super().__call__(
            inputs=inputs,
            cache=cache,
            dropout=dropout,
            attention_dropout=attention_dropout,
            return_cache=return_cache,
            use_2d=use_2d,
            shape=shape
        )



class MultiLayerPerceptron(tf.keras.layers.Layer):

    def __init__(self, activation_fn=None, embedding_size=768,
                 perceptron_size=3072, trainable=True,
                 initializer_range=0.02, name=None):
        super().__init__(name=name, trainable=trainable)
        self.layer_norm = LayerNormalization(name="layer_norm")
        self.perceptron = tf.keras.layers.Dense(units=perceptron_size,
                                                activation=activation_fn,
                                                kernel_initializer=tf.random_normal_initializer(stddev=initializer_range),
                                                name="perceptron")
        self.projection = tf.keras.layers.Dense(units=embedding_size,
                                                kernel_initializer=tf.random_normal_initializer(stddev=initializer_range),
                                                name="projection")

    def call(self, inputs, dropout=None):
        """

        inputs: tensor of [batch_size, seq_length, dim]

        """
        x = self.layer_norm(inputs)
        x = self.perceptron(x)
        x = self.projection(x)
        x = dropout_fn(x, dropout)
        return x

    def __call__(self, inputs, dropout=None):
        return super().__call__(inputs=inputs,
                                dropout=dropout)


class Block(tf.keras.layers.Layer):

    def __init__(self, config, trainable=True, initializer_range=0.02, name=None):
        super().__init__(name=name, trainable=trainable)
        self.attention = AttentionLayer(config=config,
                                        initializer_range=initializer_range,
                                        name="attention")
        self.mlp = MultiLayerPerceptron(activation_fn=gelu,
                                        embedding_size=config["n_embd"],
                                        perceptron_size=4 * config["n_embd"],
                                        initializer_range=initializer_range,
                                        name="mlp")

    def call(self, inputs, cache=None, dropout=None, attention_dropout=None,
            return_cache=False, use_2d=False, shape=None):
        x = inputs
        a = self.attention(inputs=x,
                           cache=cache,
                           dropout=dropout,
                           attention_dropout=attention_dropout,
                           return_cache=return_cache,
                           use_2d=use_2d,
                           shape=shape)
        if return_cache:
            a, cache = a
        x = x + a
        m = self.mlp(inputs=x,
                     dropout=dropout)
        x = x + m
        if return_cache:
            return x, cache
        else:
            return x

    def __call__(self, inputs, cache=None, dropout=None, attention_dropout=None,
                 return_cache=False, use_2d=False, shape=None):
        return super().__call__(inputs=inputs,
                                cache=cache,
                                dropout=dropout,
                                attention_dropout=attention_dropout,
                                return_cache=return_cache,
                                use_2d=use_2d,
                                shape=shape)


class Transformer(tf.keras.Model):

    def __init__(self, config, trainable=True, name=None):
        super().__init__(name=name)
        self.trainable = trainable
        self.blocks = []
        self.blocks_num = config["n_layer"]
        for ids in range(self.blocks_num):
            block = Block(config=config,
                          name="block_%d" % ids)
            self.blocks.append(block)
        self.layer_norm = LayerNormalization(name="layer_norm")

    def call(self, inputs, cache=None, dropout=None, attention_dropout=None,
             return_cache=False, blocks=None, use_2d=False, shape=None):
        """

        inputs: a tensor of shape [batch_size, seq_length, dim], if use_2d is False, else [batch_size * seq_length, dim]
        cache: a list of dictionaries. key and values from previous calls.
        blocks: a list. if it is specified, the output will be a dictionary {layer_num: layer_output}
        return_cache: if it is true, it will returns cache for blocks
        use_2d: if it is True, then the operations will define base on 2D tensors. (for tpu performance)
        shape: if use_2d is True, then it is [batch_size, seq_length]

        """
        if blocks is None:
            max_block = self.blocks_num - 1
        elif len(blocks) == 0:
            max_block = self.blocks_num - 1
            blocks = None
        else:
            _blocks = []
            for i in blocks:
                if i >= 0:
                    k = i
                else:
                    k = self.blocks_num - i
                if k >= self.blocks_num or k < 0:
                    raise ValueError("output blocks should be in range [" + str(0) + ", " +
                                     str(self.blocks_num - 1) + "]")
                _blocks.append(k)
            _blocks = list(sorted(_blocks))
            blocks = _blocks
            max_block = blocks[-1]
        if blocks is not None:
            outputs = {}
        if return_cache:
            new_cache = []
        output = inputs
        for ids in range(max_block + 1):
            if cache is None:
                _cache = None
            else:
                _cache = cache[ids]
            output = self.blocks[ids](inputs=output,
                                      cache=_cache,
                                      dropout=dropout,
                                      attention_dropout=attention_dropout,
                                      return_cache=return_cache,
                                      use_2d=use_2d,
                                      shape=shape)
            if return_cache:
                output, _cache = output
                new_cache.append(_cache)
            if blocks is not None:
                if ids in blocks:
                    outputs[ids] = output
        if blocks is None:
            output = self.layer_norm(output)
            result = output
        else:
            result = outputs
        if return_cache:
            return result, new_cache
        else:
            return result

    def __call__(self, inputs, cache=None, dropout=None, attention_dropout=None,
                 return_cache=False, blocks=None, use_2d=False, shape=None):
        """

        inputs: a tensor of shape [batch_size, seq_length, dim], if use_2d is False, else [batch_size * seq_length, dim]
        cache: a list of dictionaries. key and values from previous calls.
        blocks: a list. if it is specified, the output will be a dictionary {layer_num: layer_output}
        return_cache: if it is true, it will returns cache for blocks
        use_2d: if it is True, then the operations will define base on 2D tensors. (for tpu performance)
        shape: if use_2d is True, then it is [batch_size, seq_length]

        """
        return super().__call__(
            inputs=inputs,
            cache=cache,
            dropout=dropout,
            attention_dropout=attention_dropout,
            return_cache=return_cache,
            blocks=blocks,
            use_2d=use_2d,
            shape=shape
        )


class Embedding(tf.keras.layers.Layer):

    def __init__(self, embedding_size, vocab_size, max_position_length,
                 trainable=True, name=None, initializer_range=0.02,
                 dtype=None):
        if dtype is None:
            dtype = tf.float32
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self.word_embedding = None
        self.position_embedding = None
        self.initializer_range = initializer_range
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.max_position_length = max_position_length

    def build(self, input_shape):
        self.word_embedding = self.add_weight(
            name="word_embedding",
            shape=(self.vocab_size, self.embedding_size),
            initializer=tf.random_normal_initializer(stddev=self.initializer_range),
        )
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.max_position_length, self.embedding_size),
            initializer=tf.random_normal_initializer(stddev=self.initializer_range),
        )

    def call(self, inputs, start=None):
        """

        inputs: integer tensor of [batch_size, seq_length]
        start: start of positional embedding

        """
        shape = get_tensor_shape(inputs)
        x = tf.gather(self.word_embedding, inputs)
        if start is None:
            start = 0
        end = start + shape[1]
        pe = self.position_embedding[start:end]
        x = x + pe
        return x

    def __call__(self, inputs, start=None):
        """

        if use_one_hot_keys is True, then inputs are one_hot tensors of shape [batch_size, seq_length, vocab_size],
        else it is an integer tensor of [batch_size, seq_length] of token ids.
        start: start of positional embedding

        """
        return super().__call__(inputs=inputs, start=start)


class GPT2(tf.keras.Model):

    def __init__(self, config, name=None, trainable=True, dtype=None):
        super().__init__(name=name)
        self.trainable = trainable
        self.embedding = Embedding(
            embedding_size=config['n_embd'],
            vocab_size=config['n_vocab'],
            max_position_length=config['n_ctx'],
            name="embedding",
            dtype=dtype
        )
        self.transformer = Transformer(config, name="transformer")

    def call(self, inputs, cache=None,
             dropout=None, attention_dropout=None,
             return_cache=False, return_logits=True, use_2d=False):
        """

        inputs: an integer tensor of shape [batch_size, seq_length] if not use_2d is False
                else a one_hot tensor of shape [batch_size, seq_length, vocab_size]
        cache: a list of dictionaries {"key": key, "value": value} of previous keys and values. it uses for generation
        use_one_hot_keys: if True it uses one hot tensors for embedding layer.
        return_cache: if True returns new keys and values alongside output. it uses for generation.
        return_logits: if True, return logits, else return last layer embedding.
        use_2d: for tpu performances: use 2D tensors for operations and return the output in 2D shape: [batch_size * seq_length, -1]

        """
        if cache is not None:
            _cache = cache[0]["key"]
            start = get_tensor_shape(_cache)[2]
        else:
            start = None
        x = self.embedding(inputs, start)
        if use_2d:
            shape = get_tensor_shape(x)
            x = tf.reshape(x, [shape[0] * shape[1], shape[2]])
            shape = shape[0:2]
        else:
            shape = None
        x = self.transformer(
            inputs=x,
            cache=cache,
            dropout=dropout,
            attention_dropout=attention_dropout,
            return_cache=return_cache,
            use_2d=use_2d,
            shape=shape
        )
        if return_cache:
            x, cache = x
        if return_logits:
            shape = get_tensor_shape(x)
            if not use_2d:
                x = tf.reshape(x, [shape[0] * shape[1], shape[2]])
            logits = tf.matmul(x, self.embedding.word_embedding, transpose_b=True)
            if not use_2d:
                logits = tf.reshape(logits, [shape[0], shape[1], self.embedding.vocab_size])
            result = logits
        else:
            result = x
        if return_cache:
            return result, cache
        else:
            return result

    def __call__(self, inputs, cache=None,
                 dropout=None, attention_dropout=None,
                 return_cache=False, return_logits=True,
                 use_2d=False):
        """

        inputs: an integer tensor of shape [batch_size, seq_length]
        cache: a list of dictionaries {"key": key, "value": value} of previous keys and values. it uses for generation
        use_one_hot_keys: if True it uses one hot tensors for embedding layer.
        return_cache: if True returns new keys and values alongside output. it uses for generation.
        return_logits: if True, return logits, else return last layer embedding.
        use_2d: for tpu performances: use 2D tensors for operations and return the output in 2D shape: [batch_size * seq_length, -1]

        """
        return super().__call__(
            inputs=inputs,
            cache=cache,
            dropout=dropout,
            attention_dropout=attention_dropout,
            return_cache=return_cache,
            return_logits=return_logits,
            use_2d=use_2d
        )

