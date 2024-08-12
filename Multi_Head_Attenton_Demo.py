import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
import numpy as np

#This program demonstrates the Multi-Head Attention Layer concept
#as described in the paper "Attention is all you need"

class ScaledDotProductAttention(Layer):
    def call(self, queries, keys, values, mask=None):
        dk = tf.cast(tf.shape(keys)[-1], tf.float32)
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(dk)

        if mask is not None:
            scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, values)
        return output, attention_weights

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, queries, keys, values, mask=None):
        batch_size = tf.shape(queries)[0]

        queries = self.wq(queries)
        keys = self.wk(keys)
        values = self.wv(values)

        queries = self.split_heads(queries, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)

        attention, weights = ScaledDotProductAttention()(queries, keys, values, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.num_heads * self.depth))

        output = self.dense(concat_attention)

        return output, weights

# Test the Multi-Head Attention Mechanism
def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    d_model = 512
    num_heads = 8
    batch_size = 64
    seq_len = 10

    mha = MultiHeadAttention(d_model, num_heads)

    queries = tf.random.uniform((batch_size, seq_len, d_model))
    keys = tf.random.uniform((batch_size, seq_len, d_model))
    values = tf.random.uniform((batch_size, seq_len, d_model))

    output, weights = mha(queries, keys, values)

    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

if __name__ == "__main__":
    main()