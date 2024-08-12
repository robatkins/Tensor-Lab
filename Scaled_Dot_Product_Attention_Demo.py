import tensorflow as tf

#This program demonstrates the Scaled Dot-Product Mechanism as
#described in the "Attention is all you need" paper

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def call(self, queries, keys, values, mask=None):
        """
        Compute the scaled dot-product attention.
        
        Args:
            queries: Query tensor of shape (..., seq_len_q, depth)
            keys: Key tensor of shape (..., seq_len_k, depth)
            values: Value tensor of shape (..., seq_len_v, depth_v)
            mask: Float tensor of shape (..., seq_len_q, seq_len_k). Defaults to None.
        
        Returns:
            output: Output tensor of shape (..., seq_len_q, depth_v)
            attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
        """
        dk = tf.cast(tf.shape(keys)[-1], tf.float32)
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(dk)

        if mask is not None:
            scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, values)
        return output, attention_weights

# Test the Scaled Dot-Product Attention Mechanism
def main():
    tf.random.set_seed(42)

    batch_size = 64
    seq_len_q = 10
    seq_len_kv = 10
    depth = 512

    queries = tf.random.uniform((batch_size, seq_len_q, depth))
    keys = tf.random.uniform((batch_size, seq_len_kv, depth))
    values = tf.random.uniform((batch_size, seq_len_kv, depth))

    attention_layer = ScaledDotProductAttention()
    output, attention_weights = attention_layer(queries, keys, values)

    print(f"Queries shape: {queries.shape}")
    print(f"Keys shape: {keys.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

if __name__ == "__main__":
    main()