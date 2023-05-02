from preprocess_val import get_data
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        #reduce layer sizes, add dropout, learning rate, fewer epochs?

        self.batch_size = 32
        self.num_classes = 1 # only predicting one value
        self.lr = .001
        self.epochs = 10
        # self.stride = (default is 1 so only need this if want something different?)
        self.padding = "SAME"
        self.embedding_size = 100  # 80? (from paper)
        self.vocab_size = 96272
        self.hidden_size = 40  # 256
        self.weight_decay = 1e-6
        self.momentum = 0.9

        # the default initializer here is "uniform" we can play around with it
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, self.embedding_size, embeddings_initializer="uniform")

        self.permute = tf.keras.layers.Permute((2, 1), input_shape=(529, 64))

        # did some research... top three seem to be HeNormal, Kaiming, and Xavier but dont know which is best
        # I think HeNormal(kaiming) is best, top  seem to be xavier, and he normal
        self.conv1d = tf.keras.layers.Conv1D(
            16, 2, strides=2, padding=self.padding, activation="relu", kernel_initializer="HeNormal")

        # flatten?
        self.permute2 = tf.keras.layers.Permute(
            (1, 2), input_shape=(529, 64))  # does this do anything??
        # LSTM
        self.LSTM = tf.keras.layers.LSTM(
            40)  # what size??
        self.drop = tf.keras.layers.Dropout(.5)
        self.flat = tf.keras.layers.Flatten()
        # Dropout
        self.seq = tf.keras.Sequential([tf.keras.layers.Dense(
            64, activation="tanh"), tf.keras.layers.Dropout(.5), tf.keras.layers.Dense(self.num_classes, activation="relu")])

        # dense
        # paper output is 64
        # dropout
        # self.dropout2 = tf.keras.layers.Dropout(.5)
        # dense

        self.optimizer = tf.keras.optimizers.experimental.SGD(self.lr, self.momentum, weight_decay=self.weight_decay) # SGD
        self.loss = tf.keras.losses.MeanSquaredError() # reduction ??

    def call(self, inputs):

        # 2: (32, 529, 100)
        # 2.5: (32, 100, 529)
        # 4: (32, 50, 16)
        # 4: (32, 50, 16)
        # 5: (32, 100)
        # 6: (32, 100)
        # 7: (32, 256)
        # 8: (32, 256)
        # 9: (32, 3)

        logits = self.embedding(inputs)
        # print("2:", logits.shape) 
        # logits = self.permute(logits)
        # print("2.5:", logits.shape)
        logits = self.conv1d(logits)
        # print(logits.shape) #
        logits = tf.nn.max_pool(logits, 2, strides=None, padding=self.padding)
        # print("4:", logits.shape) 
        # logits = self.permute2(logits)
        # print("4:", logits.shape) 
        logits = self.LSTM(logits)
        # logits = self.flat(logits)
        logits = self.drop(logits)
        # print("5:", logits.shape)
        logits = self.seq(logits)
        # print(logits)
        # logits = [val for song in logits for val in song] # def not the best way to do this 
        
        logits = tf.reshape(logits, [32])
        
        return logits

    def r2_score(self, logits, labels):

        metric = tfa.metrics.r_square.RSquare()
        metric.update_state(labels, logits)
        result = metric.result()

        return result.numpy() #uses R squared to return dif/acc (?)


def train(model, train_lyrics, train_labels):

    # train model -- maybe shuffle inputs -- look at hw2 and hw3

    avg_r2 = 0
    avg_loss = 0
    counter = 0

    # use train_captions or image_features or both?
    index_range = tf.random.shuffle(range(len(train_lyrics)))
    shuffled_lyrics = tf.gather(train_lyrics, index_range)
    # do these also need to be shuffled?
    shuffled_labels = tf.gather(train_labels, index_range)

    # train_lyrics.shape[0] + 1 if tensor
    for batch_num, b1 in enumerate(range(model.batch_size, len(train_lyrics) + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        batch_lyrics = shuffled_lyrics[b0:b1]
        batch_labels = shuffled_labels[b0:b1]

        with tf.GradientTape() as tape:
            logits = model(batch_lyrics)

            loss = model.loss(batch_labels, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(
                zip(grads, model.trainable_variables))
        r_squared = model.r2_score(logits, batch_labels)

        avg_r2 += r_squared
        avg_loss += loss
        counter += 1

        print(
            f"\r[Train {batch_num+1:4n}/{3764}]\t loss={loss:.3f}\t r_squared: {r_squared:.3f}", end='')
    print()
    return avg_loss/counter, avg_r2/counter


def test(model, test_lyrics, test_labels):

    # Tests the model on the test inputs and labels.

    avg_r2 = 0
    avg_loss = 0
    counter = 0
    # make test_lyrics.shape[0] + 1?
    for batch_num, b1 in enumerate(range(model.batch_size, len(test_lyrics) + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        batch_lyrics = test_lyrics[b0:b1]
        batch_labels = test_labels[b0:b1]

        logits = model(batch_lyrics)
        loss = model.loss(batch_labels, logits)

        r2 = model.r2_score(logits, batch_labels)
        avg_r2 += r2
        avg_loss += loss
        counter += 1
        print(
            f"\r[Valid {batch_num+1:4n}/{941}]\t loss={loss:.3f}\t r_squared: {r2:.3f}", end='')

    print()
    return avg_r2/counter, avg_loss/counter


def main():

    train_lyrics, test_lyrics, train_labels, test_labels = get_data(
        "data/labeled_lyrics_cleaned.csv")

    model = Model()

    for e in range(model.epochs):
        print("epoch", e+1)
        train(model, train_lyrics, train_labels)

    t = test(model, test_lyrics, test_labels)

    tf.print("Final R2 Score:", t[0])

    return


if __name__ == '__main__':
    main()
