import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


class DKN(tf.keras.Model):
    def __init__(self, args):
        super(DKN, self).__init__()
        self.args = args
        self._build_embeddings(args)
        self._build_layers(args)

    def _build_embeddings(self, args):
        # Load embeddings
        word_embs = np.load('../data/news/word_embeddings_' + str(args.word_dim) + '.npy')
        entity_embs = np.load('../data/kg/entity_embeddings_' + args.KGE + '_' + str(args.entity_dim) + '.npy')

        # Create embedding layers
        self.word_embedding = tf.keras.layers.Embedding(
            word_embs.shape[0], word_embs.shape[1],
            embeddings_initializer=tf.keras.initializers.Constant(word_embs),
            name='word_embedding'
        )
        self.entity_embedding = tf.keras.layers.Embedding(
            entity_embs.shape[0], entity_embs.shape[1],
            embeddings_initializer=tf.keras.initializers.Constant(entity_embs),
            name='entity_embedding'
        )

        if args.use_context:
            context_embs = np.load(
                '../data/kg/context_embeddings_' + args.KGE + '_' + str(args.entity_dim) + '.npy')
            self.context_embedding = tf.keras.layers.Embedding(
                context_embs.shape[0], context_embs.shape[1],
                embeddings_initializer=tf.keras.initializers.Constant(context_embs),
                name='context_embedding'
            )

    def _build_layers(self, args):
        # Transform layers
        if args.transform:
            self.entity_transform = tf.keras.layers.Dense(
                args.entity_dim, activation='tanh',
                kernel_regularizer=tf.keras.regularizers.l2(args.l2_weight),
                name='entity_transform'
            )
            if args.use_context:
                self.context_transform = tf.keras.layers.Dense(
                    args.entity_dim, activation='tanh',
                    kernel_regularizer=tf.keras.regularizers.l2(args.l2_weight),
                    name='context_transform'
                )

        # CNN layers
        self.conv_layers = []
        self.pool_layers = []
        for filter_size in args.filter_sizes:
            self.conv_layers.append(tf.keras.layers.Conv2D(
                filters=args.n_filters,
                kernel_size=(filter_size, args.word_dim + args.entity_dim),
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(args.l2_weight),
                name=f'conv_{filter_size}'
            ))
            self.pool_layers.append(tf.keras.layers.MaxPool2D(
                pool_size=(args.max_title_length - filter_size + 1, 1),
                name=f'pool_{filter_size}'
            ))

    def _kcnn(self, words, entities):
        # Embedding lookups
        embedded_words = self.word_embedding(words)
        embedded_entities = self.entity_embedding(entities)

        if self.args.transform:
            embedded_entities = self.entity_transform(embedded_entities)

        if self.args.use_context:
            embedded_contexts = self.context_embedding(entities)
            if self.args.transform:
                embedded_contexts = self.context_transform(embedded_contexts)
            concat_input = tf.concat([embedded_words, embedded_entities, embedded_contexts], axis=-1)
        else:
            concat_input = tf.concat([embedded_words, embedded_entities], axis=-1)

        concat_input = tf.expand_dims(concat_input, -1)

        outputs = []
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            conv_out = conv(concat_input)
            pool_out = pool(conv_out)
            outputs.append(pool_out)

        output = tf.concat(outputs, axis=-1)
        output = tf.squeeze(output, [1, 2])
        return output

    def call(self, inputs, training=None):
        clicked_words = inputs['clicked_words']
        clicked_entities = inputs['clicked_entities']
        news_words = inputs['news_words']
        news_entities = inputs['news_entities']

        # Process clicked history
        batch_size = tf.shape(clicked_words)[0]
        clicked_words_flat = tf.reshape(clicked_words, [-1, self.args.max_title_length])
        clicked_entities_flat = tf.reshape(clicked_entities, [-1, self.args.max_title_length])

        clicked_embeddings = self._kcnn(clicked_words_flat, clicked_entities_flat)
        clicked_embeddings = tf.reshape(
            clicked_embeddings,
            [batch_size, self.args.max_click_history, -1]
        )

        # Process candidate news
        news_embeddings = self._kcnn(news_words, news_entities)
        news_embeddings_expanded = tf.expand_dims(news_embeddings, 1)

        # Attention mechanism
        attention_weights = tf.reduce_sum(clicked_embeddings * news_embeddings_expanded, axis=-1)
        attention_weights = tf.nn.softmax(attention_weights)
        attention_weights = tf.expand_dims(attention_weights, -1)

        user_embeddings = tf.reduce_sum(clicked_embeddings * attention_weights, axis=1)

        # Final prediction
        scores = tf.reduce_sum(user_embeddings * news_embeddings, axis=-1)
        return tf.sigmoid(scores)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(data, training=True)

            # Compute loss
            loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data['labels'], y_pred)
            )

            # Add regularization losses
            loss += sum(self.losses)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def test_step(self, data):
        # Forward pass
        y_pred = self(data, training=False)
        return y_pred

    def compile(self, args):
        super().compile()
        self.optimizer = tf.keras.optimizers.Adam(args.lr)
