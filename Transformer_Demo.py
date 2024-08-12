import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np

# Define the multi-head attention layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]

        q = self.split_heads(self.wq(query), batch_size)
        k = self.split_heads(self.wk(key), batch_size)
        v = self.split_heads(self.wv(value), batch_size)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        return self.dense(output)

# Define the position-wise feed-forward network
class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(PositionwiseFeedForward, self).__init__()
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)

    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)

# Define the transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.att(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define the positional encoding layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_positional_encoding, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = tf.keras.layers.Embedding(input_dim=max_positional_encoding, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        position = tf.range(start=0, limit=seq_len, delta=1)
        position = tf.expand_dims(position, axis=0)
        positional_encoding = self.positional_encoding(position)
        return x + positional_encoding

# Define the full transformer model
def build_transformer_model(vocab_size, d_model, num_heads, dff, num_layers, max_positional_encoding, rate=0.1):
    inputs = Input(shape=(None,))
    embedding = Embedding(vocab_size, d_model)(inputs)
    
    # Add positional encoding
    positional_encoding_layer = PositionalEncoding(max_positional_encoding, d_model)
    embedding = positional_encoding_layer(embedding)

    x = Dropout(rate)(embedding)

    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, dff, rate)(x, training=True, mask=None)

    x = tf.reduce_mean(x, axis=1)
    x = Dense(vocab_size, activation='softmax')(x)

    return Model(inputs=inputs, outputs=x)

# Example usage
vocab_size = 10000
d_model = 512
num_heads = 8
dff = 2048
num_layers = 4
max_positional_encoding = 1000

model = build_transformer_model(vocab_size, d_model, num_heads, dff, num_layers, max_positional_encoding)
model.summary()

# Dummy input for demonstration
dummy_input = np.random.randint(0, vocab_size, (1, 20))  # Batch size 1, sequence length 20
dummy_output = model(dummy_input)
print("Dummy output shape:", dummy_output.shape)