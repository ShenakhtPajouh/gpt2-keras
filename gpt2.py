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


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, trainable=True, name=None):
        super().__init__(name=name, trainable=trainable)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        self.beta = self.add_weight(name="beta", shape=input_shape[-1:], initializer=tf.zeros_initializer(),
                                    dtype=tf.float32)
        self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:], initializer=tf.ones_initializer(),
                                     dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, axis=-1, epsilon=10e-5):
        mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
        rdev = tf.rsqrt(variance + epsilon)
        x = (inputs - mean) * rdev
        output = x * self.gamma + self.beta
        return output

    def __call__(self, inputs, axis=-1, epsilon=10e-12):
        return super().__call__(inputs=inputs,
                                axis=axis, epsilon=epsilon)



def attention(query, key, value, multi_heads, mask=None,
              attention_dropout=None, cached_dropout=None):
    """
    query: A tensor of shape [batch_size, query_num, dim]
    key: A tensor of shape [batch_size, key_num, dim]
    value: A tensor of shape [batch_size, key_num, value_dim]
    query_mask: a boolean tensor of shape [batch_size, query_num]
    mask: a boolean tensor for coefficient mask. could be a tensor of [batch_size, multi_heads, query_num, key_num] or a compatible shape
    one_sided: a boolean which determines if the attention mechanism is one sided or not
    attention_dropout: dropout for attention coefficients
    """
    batch_size, query_num, dim = get_tensor_shape(query)
    _, key_num, value_dim = get_tensor_shape(value)
    new_dim = dim // multi_heads
    query = tf.reshape(query, [batch_size, query_num, multi_heads, new_dim])
    key = tf.reshape(key, [batch_size, key_num, multi_heads, new_dim])
    value = tf.reshape(value, [batch_size, key_num, multi_heads, value_dim // multi_heads])
    query = tf.transpose(query, [0, 2, 1, 3])
    key = tf.transpose(key, [0, 2, 3, 1])
    value = tf.transpose(value, [0, 2, 1, 3])
    coefficients = tf.matmul(query, key) / tf.sqrt(float(new_dim))
    if mask is not None:

        mask = tf.cast(mask, coefficients.dtype)
        coefficients = coefficients * mask - (1-mask) * 10e10
    coefficients = tf.nn.softmax(coefficients, -1)
    coefficients, dropout = dropout_fn(coefficients, attention_dropout, cached_dropout)
    result = tf.matmul(coefficients, value)
    result = tf.transpose(result, [0, 2, 1, 3])
    result = tf.reshape(result, [batch_size, query_num, value_dim])
    return result, dropout


def dropout_fn(input_tensor, dropout_prob=None, cached_dropout=None):
    """Perform dropout.
    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).
    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0 and (cached_dropout is None):
        return input_tensor, None
    if cached_dropout is not None:
        dropout = cached_dropout
    else:
        shape = get_tensor_shape(input_tensor)
        dropout = tf.random.uniform(shape=shape) > dropout_prob
    output = input_tensor * tf.cast(dropout, input_tensor.dtype)
    return output, dropout


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
        self.trainable = trainable
        self.dropout_cache = None
        self.one_sided = one_sided

    def get_mask(self, inputs, cache=None, mask=None):
        shape = get_tensor_shape(inputs)
        if cache is not None:
            cache_num = get_tensor_shape(cache["key"])[1]
        else:
            cache_num = 0
        if self.one_sided:
            _range = tf.range(shape[1])
            one_sided_mask = tf.expand_dims(_range, 1) >= _range
            if cache is not None:
                cache_mask = tf.ones((shape[1], cache_num), dtype=tf.bool)
                one_sided_mask = tf.concat([cache_mask, one_sided_mask], 1)
            one_sided_mask = tf.reshape(one_sided_mask, [1, 1, shape[1], shape[1] + cache_num])
        if mask is not None:
            if cache is not None:
                cache_mask = tf.ones((shape[0], cache_num), dtype=tf.bool)
                mask = tf.concat([cache_mask, mask], 1)
            mask = tf.reshape(mask, [shape[0], 1, 1, shape[1] + cache_num])
            if self.one_sided:
                mask = tf.logical_and(mask, one_sided_mask)
        if mask is None and self.one_sided:
            return one_sided_mask
        elif mask is None:
            return None
        else:
            return mask

    def call(self, inputs, cache=None, mask=None,
             dropout_reuse=False, attention_dropout=None,
             return_cache=False):
        """
        inputs: a tensor of shape [batch_size, seq_length, dim]
        mask: a boolean tensor of shape [batch_size, seq_length]
        attention_probs_dropout_prob: dropout use for attention mechanism
        """
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        if dropout_reuse:
            dropout_cache = self.dropout_cache
        else:
            dropout_cache = None
        if cache is not None:
            key = tf.concat([cache["key"], key], 1)
            value = tf.concat([cache["value"], value], 1)
        mask = self.get_mask(inputs, cache, mask)
        result, self.dropout_cache = attention(query, key, value, self.num_attention_heads,
                                               mask=mask, attention_dropout=attention_dropout,
                                               cached_dropout=dropout_cache)
        if return_cache:
            cache = {"key": key, "value": value}
            return result, cache
        else:
            return result

    def __call__(self, inputs, cache=None, mask=None, attention_dropout=None,
                 dropout_reuse=False, return_cache=False):
        return super().__call__(
            inputs=inputs,
            mask=mask,
            attention_dropout=attention_dropout,
            cache=cache,
            dropout_reuse=dropout_reuse,
            return_cache=return_cache,
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
        self.dropout_cache = None

    def call(self, inputs, cache=None, dropout=None, attention_dropout=None, dropout_reuse=False,
             return_cache=False):
        x = self.layer_norm(inputs)
        x = self.self_attention(x, attention_dropout=attention_dropout,
                                cache=cache, dropout_reuse=dropout_reuse,
                                return_cache=return_cache)
        if return_cache:
            x, cache = x
        x = self.projection(x)
        if dropout_reuse:
            x, _ = dropout_fn(x, cached_dropout=self.dropout_cache)
        else:
            x, self.dropout_cache = dropout_fn(x, dropout)
        if return_cache:
            return x, cache
        else:
            return x

    def __call__(self, inputs, cache=None, dropout=None, attention_dropout=None,
                 dropout_reuse=False, return_cache=False):
        return super().__call__(inputs=inputs,
                                cache=cache, dropout=dropout,
                                attention_dropout=attention_dropout,
                                dropout_reuse=dropout_reuse,
                                return_cache=return_cache)

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
        self.dropout_cache = None

    def call(self, inputs, dropout=None, dropout_reuse=False):
        x = self.layer_norm(inputs)
        x = self.perceptron(x)
        x = self.projection(x)
        if dropout_reuse:
            x, _ = dropout_fn(x, cached_dropout=self.dropout_cache)
        else:
            x, self.dropout_cache = dropout_fn(x, dropout)
        return x

    def __call__(self, inputs, dropout=None, dropout_reuse=False):
        return super().__call__(inputs=inputs,
                                dropout=dropout,
                                dropout_reuse=dropout_reuse)


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
             dropout_reuse=False, return_cache=False):
        x = inputs
        a = self.attention(inputs=x,
                           cache=cache,
                           dropout=dropout,
                           attention_dropout=attention_dropout,
                           dropout_reuse=dropout_reuse,
                           return_cache=return_cache)
        if return_cache:
            a, cache = a
        x = x + a
        m = self.mlp(inputs=x,
                     dropout=dropout,
                     dropout_reuse=dropout_reuse)
        x = x + m
        if return_cache:
            return x, cache
        else:
            return x

    def __call__(self, inputs, cache=None, dropout=None, attention_dropout=None,
                 dropout_reuse=False, return_cache=False):
        return super().__call__(inputs=inputs,
                                cache=cache,
                                dropout=dropout,
                                attention_dropout=attention_dropout,
                                dropout_reuse=dropout_reuse,
                                return_cache=return_cache)

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
             dropout_reuse=False, return_cache=False, blocks=None):
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
                                      dropout_reuse=dropout_reuse,
                                      return_cache=return_cache)
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
                 dropout_reuse=False, return_cache=False, blocks=None):
        return super().__call__(
            inputs=inputs,
            cache=cache,
            dropout=dropout,
            attention_dropout=attention_dropout,
            dropout_reuse=dropout_reuse,
            return_cache=return_cache,
            blocks=blocks
        )


class Embedding(tf.keras.layers.Layer):

    def __init__(self, embedding_size, vocab_size, max_position_length,
                 trainable=True, name=None, initializer_range=0.02):
        super().__init__(name=name, trainable=trainable)
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
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=self.initializer_range),
        )
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.max_position_length, self.embedding_size),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=self.initializer_range),
        )

    def call(self, inputs, use_one_hot_keys=False, start=None):
        shape = get_tensor_shape(inputs)
        if use_one_hot_keys:
            x = tf.reshape(inputs, [shape[0] * shape[1]])
            x = tf.one_hot(x, self.vocab_size)
            x = tf.matmul(x, self.word_embedding)
            x = tf.reshape(x, [shape[0], shape[1], self.embedding_size])
        else:
            x = tf.gather(self.word_embedding, inputs)
        if start is None:
            start = 0
        end = start + shape[1]
        pe = self.position_embedding[start:end]
        x = x + pe
        return x

    def __call__(self, inputs, use_one_hot_keys=False, start=None):
        return super().__call__(inputs=inputs,
                                use_one_hot_keys=use_one_hot_keys,
                                start=start)


class GPT2(tf.keras.Model):

    def __init__(self, config, name=None, trainable=True):
        super().__init__(name=name)
        self.trainable = trainable
        self.embedding = Embedding(
            embedding_size=config['n_embd'],
            vocab_size=config['n_vocab'],
            max_position_length=config['n_ctx'],
            name="embedding"
        )
        self.transformer = Transformer(config, name="transformer")

    def call(self, inputs, cache=None, use_one_hot_keys=False,
             dropout=None, attention_dropout=None,
             dropout_reuse=False, return_cache=False, return_logits=True):
        if cache is not None:
            _cache = cache[0]
            start = get_tensor_shape(_cache)[1]
        else:
            start = None
        x = self.embedding(inputs, use_one_hot_keys, start)
        x = self.transformer(
            inputs=x,
            cache=cache,
            dropout=dropout,
            attention_dropout=attention_dropout,
            dropout_reuse=dropout_reuse,
            return_cache=return_cache
        )
        if return_cache:
            x, cache = x
        if return_logits:
            shape = get_tensor_shape(x)
            x = tf.reshape(x, [shape[0] * shape[1], shape[2]])
            logits = tf.matmul(x, self.embedding.word_embedding, transpose_b=True)
            logits = tf.reshape(logits, [shape[0], shape[1], self.embedding.vocab_size])
            result = logits
        else:
            result = x
        if return_cache:
            return result, cache
        else:
            return result

    def __call__(self, inputs, cache=None, use_one_hot_keys=False,
                 dropout=None, attention_dropout=None,
                 dropout_reuse=False, return_cache=False, return_logits=True):
        return super().__call__(
            inputs=inputs,
            cache=cache,
            use_one_hot_keys=use_one_hot_keys,
            dropout=dropout,
            attention_dropout=attention_dropout,
            dropout_reuse=dropout_reuse,
            return_cache=return_cache,
            return_logits=return_logits
        )



















