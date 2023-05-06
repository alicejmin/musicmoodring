from json.encoder import INFINITY
from preprocess_mood import get_data
import tensorflow as tf
import numpy as np
# imports for plotting
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.batch_size = 32
        self.num_classes = 3
        self.lr = .001
        self.epochs = 30
        self.weight_decay = 1e-6
        self.momentum = 0.9
        # self.stride = (default is 1 so only need this if want something different?)
        self.padding = "SAME"
        self.embedding_size = 100  # 80? (from paper)
        self.vocab_size = 15245
        self.hidden_size = 40  # 256
        # for plot
        self.epoch_list = []
        self.plot_df = pd.DataFrame()
        self.plot_class_df = pd.DataFrame()
        self.plot_sadness = pd.DataFrame()
        self.plot_tenderness = pd.DataFrame()
        self.plot_tension = pd.DataFrame()

        # the default initializer here is "uniform" we can play around with it
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, self.embedding_size)

        self.permute = tf.keras.layers.Permute((2, 1), input_shape=(529, 64))
        self.conv1d = tf.keras.layers.Conv1D(
            16, 2, strides=2, activation='relu')

        self.maxpool = tf.keras.layers.MaxPool1D(2)

        # LSTM #try using GRU
        self.LSTM = tf.keras.layers.LSTM(
            40, activation='leaky_relu')  # activation?
        self.drop = tf.keras.layers.Dropout(.5)
        self.flat = tf.keras.layers.Flatten()
        # Dropout
        self.seq = tf.keras.Sequential([tf.keras.layers.Dense(
            64, kernel_regularizer=tf.keras.regularizers.L1(.01), activation='relu'), tf.keras.layers.Dropout(.5), tf.keras.layers.Dense(self.num_classes, activation='softmax')])

        # self.seq = tf.keras.Sequential([tf.keras.layers.Dense(
        #     64, kernel_regularizer=tf.keras.regularizers.L1(.01), activation='relu'), tf.keras.layers.Dropout(.5), tf.keras.layers.Dense(self.num_classes, activation='softmax')])

        self.optimizer = tf.keras.optimizers.SGD(self.lr, self.momentum)  # SGD
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        # self.accuracy = tf.keras.metrics.Accuracy()

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
        logits = self.conv1d(logits)
        logits = self.maxpool(logits)
        # logits = tf.nn.max_pool(logits, 3, strides=2, padding=self.padding)
        logits = self.LSTM(logits)
        logits = self.flat(logits)
        logits = self.drop(logits)
        logits = self.seq(logits)

        # logits = self.embedding(inputs)
        # logits = self.conv1d(logits)
        # logits = self.maxpool(logits)
        # logits = tf.nn.max_pool(logits, 3, strides=2, padding=self.padding)
        # logits = self.drop(logits)
        # logits = self.flat(logits)
        # logits = self.seq(logits)

        return logits

    def accuracy(self, logits, labels):
        num_correct_classes = 0
        for song in range(self.batch_size):
            if tf.argmax(logits[song]) == tf.argmax(labels[song]):
                num_correct_classes += 1
        accuracy = num_correct_classes/self.batch_size
        return accuracy

    def acc_per_class(self, logits, labels):

        # look at acc dist across different classes

        correct_tension = 0
        tot_tension = 0
        correct_sadness = 0
        tot_sad = 0
        correct_tenderness = 0
        tot_tender = 0
        for song in range(self.batch_size):
            if tf.argmax(logits[song], axis=-1) == tf.argmax(labels[song]):
                if tf.argmax(labels[song]).numpy() == 1:
                    correct_tension += 1
                    tot_tension += 1
                elif tf.argmax(labels[song]).numpy() == 0:
                    correct_sadness += 1
                    tot_sad += 1
                else:
                    tot_tender += 1
                    correct_tenderness += 1
            else:
                if tf.argmax(labels[song]).numpy() == 1:
                    tot_tension += 1
                elif tf.argmax(labels[song]).numpy() == 0:
                    tot_sad += 1
                else:
                    tot_tender += 1

        if tot_tension == 0:
            acc_tension = 0
        else:
            acc_tension = correct_tension/tot_tension
        if tot_sad == 0:
            acc_sadness = 0
        else:
            acc_sadness = correct_sadness/tot_sad
        if tot_tension == 0:
            acc_tension = 0
        else:
            acc_tenderness = correct_tenderness/tot_tender
        
        
        return acc_tension, acc_sadness, acc_tenderness

    def loss(self, labels, logits):
        # penialize wrong answers for sadness and praise correct answers for tension, tenderness
        cce = tf.keras.losses.CategoricalCrossentropy()
        print(logits)
        for i in range(logits.shape[0]):
            copy = logits
            if tf.argmax(logits[i]) != tf.argmax(labels[i]):
                # make copy
                index = tf.argmax(logits[i]).numpy()
                copy = copy.numpy()
                copy[i][index] = .0001
                copy = tf.convert_to_tensor(logits)
            else:
                index = tf.argmax(logits[i]).numpy()
                copy = copy.numpy()
                copy[i][index] = .99
                copy = tf.convert_to_tensor(logits)
        loss = cce(labels, copy)
        # print(loss)
        return loss


def train(model, train_lyrics, train_labels):

    # train model -- maybe shuffle inputs -- look at hw2 and hw3

    avg_acc = 0
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
            # print(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(
                zip(grads, model.trainable_variables))
        acc = model.accuracy(logits, batch_labels)

        tension, sadness, tenderness = model.acc_per_class(
            logits, batch_labels)  # for chart and helpful prints
        # tension, sadness, tenderness = model.acc_per_class(logits, batch_labels)

        avg_acc += acc
        avg_loss += loss
        counter += 1

        model.epoch_list.append((acc, loss))

        # print(f"\r[Train {batch_num+1}/{27}]\t tension: {tension:.3f}\t sadness: {sadness:.3f}\t tenderness: {tenderness:.3f}", end='')

        print(
            f"\r[Train {batch_num+1}/{27}]\t loss={loss:.3f}\t acc: {acc:.3f}", end='')
    print()
    # for charts
    model.plot_df = pd.DataFrame(
        model.epoch_list, columns=['accuracy', 'loss'])

    model.plot_class_df = pd.DataFrame({'classes': [
                                       'tension', 'sadness', 'tenderness'], 'accuracy': [tension, sadness, tenderness]})

    return avg_loss/counter, avg_acc/counter


def test(model, test_lyrics, test_labels):

    # Tests the model on the test inputs and labels.

    avg_acc = 0
    avg_loss = 0
    counter = 0
    # make test_lyrics.shape[0] + 1?
    for batch_num, b1 in enumerate(range(model.batch_size, len(test_lyrics) + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        batch_lyrics = test_lyrics[b0:b1]
        batch_labels = test_labels[b0:b1]

        logits = model(batch_lyrics)
        loss = model.loss(batch_labels, logits)

        acc = model.accuracy(logits, batch_labels)
        avg_acc += acc
        avg_loss += loss
        counter += 1
        print(
            f"\r[Valid {batch_num+1}/{6}]\t loss={loss:.3f}\t acc: {acc:.3f}", end='')

    print()
    return avg_acc/counter, avg_loss/counter


# loss vs. accuracy scatter plot
def plot_results(plot_df: pd.DataFrame) -> None:
    plot_df.plot.scatter(x='accuracy', y='loss',
                         title="training accuracy results table")


def plot_sad(plot_df: pd.DataFrame) -> None:
    plot_df.plot.line(x='epoch', y='accuracy',
                      title="sad")


def plot_tender(plot_df: pd.DataFrame) -> None:
    plot_df.plot.line(x='epoch', y='accuracy',
                      title="tender")


def plot_tension(plot_df: pd.DataFrame) -> None:
    plot_df.plot.line(x='epoch', y='accuracy',
                      title="tension")


# bar chart for each class
def plot_classes(plot_class_df: pd.DataFrame) -> None:
    plot_class_df.plot.bar(x='classes', y='accuracy',
                           title="accuracy per class")


def main():

    train_lyrics, test_lyrics, train_labels, test_labels = get_data(
        "data/singlelabel.csv")

    model = Model()

    sad = []
    tender = []
    tension_list = []
    # model.compile(model.optimizer, loss=model.loss, metrics=['accuracy'], steps_per_execution=10)

    # model.fit(
    #     train_lyrics, train_labels,
    #     epochs=model.epochs,
    #     batch_size=model.batch_size,
    #     validation_split=.1
    # )

    # model.evaluate(
    #     test_lyrics, test_labels,
    #     batch_size=model.batch_size
    # )

    #early stopping
    for e in range(model.epochs):
        print("epoch", e+1)
        train(model, train_lyrics, train_labels)

        # may need to change inputs for acc_per_class
        sad.append((e, model.acc_per_class(train_lyrics, train_labels)[1]))
        tender.append((e, model.acc_per_class(train_lyrics, train_labels)[2]))
        tension_list.append(
            (e, model.acc_per_class(train_lyrics, train_labels)[0]))

     # for chart

    model.plot_sadness = pd.DataFrame(
        sad, columns=['epoch', 'accuracy'])
    model.plot_tenderness = pd.DataFrame(
        tender, columns=['epoch', 'accuracy'])
    model.plot_tension = pd.DataFrame(
        tension_list, columns=['epoch', 'accuracy'])

    t = test(model, test_lyrics, test_labels)

    tf.print("Final Accuracy:", t[0])

    plot_results(model.plot_df)
    plt.show()

    plot_classes(model.plot_class_df)
    plt.show()

    plot_sad(model.plot_sadness)
    plt.show()

    plot_tension(model.plot_tension)
    plt.show()

    plot_tender(model.plot_tenderness)
    plt.show()

    return


if __name__ == '__main__':
    main()


# def early_stop(self, loss, epoch):
    # CHANGEEEEE
    # self.scheduler(loss, epoch)
    # self.learning_rate = self.optimizer.param_groups[0]['lr']
    # stop = self.learning_rate < self.stopping_rate

    # return stop
    # loss, _ = train(model, train_lyrics, train_labels)
    #     if model.early_stop(loss, e+1):
    #         break
