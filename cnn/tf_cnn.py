import tensorflow as tf
import numpy as np
import pickle

# define the hyper parameters
root = "/content/gdrive/My Drive/EColab/dataset/math/"

num_batches = 1000
batch_size = 256
learning_rate = 1e-4


class DataLoader():
    def __init__(self, num_batch=20):
        xs = []
        ys = []
        for num in range(num_batch):
            fn = root + 'data_batch_{}.pkl'.format(num + 1)
            x, y = self.__load_data_batch(fn)
            xs.append(x)
            ys.append(y)
        self.data = np.concatenate(xs)
        self.labels = np.concatenate(ys)
        del x, y
        self.Xtest, self.Ytest = self.__load_data_batch(root +
                                                        'test_batch_1.pkl')

    def __load_data_batch(self, filename):
        with open(filename, 'rb') as fn:
            data_dict = pickle.load(fn, encoding='iso-8859-1')
            X = data_dict['data'].astype("float")
            y = np.array(data_dict['labels'])
        return X, y

    def next_batch(self, batch_size=1024):
        mask = np.random.choice(
            self.data.shape[0], size=batch_size, replace=True)
        batch_data = self.data[mask]
        batch_labels = self.labels[mask]
        return batch_data, batch_labels


# create a cnn model
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # create layers of cnn
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(12 * 12 * 64, ))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=82)

    def call(self, inputs):
        # execute the model
        inputs = tf.reshape(inputs, [-1, 45, 45, 1])
        out = self.conv1(inputs)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

    def predict(self, inputs):
        # predict the label of data
        logits = self(inputs)
        return logits


model = CNN()
dataloader = DataLoader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

for iter in range(num_batches):
    X, y = dataloader.next_batch()
    with tf.GradientTape() as tape:
        y_logit_pred = model(tf.convert_to_tensor(X))
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=y, logits=y_logit_pred)
        if (iter + 1) % 100 == 0:
            print("Iteration: {0:>5}, Loss: {1}".format(iter, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
tf.losses.softmax_cross_entropy()
num_test = np.shape(dataloader.Xtest)[0]
y_pred = model.predict(dataloader.Xtest).numpy()
y_pred_cls = np.argmax(y_pred, axis=1)
y_true_cls = np.argmax(dataloader.Ytest, axis=1)
print("test accuracy: %f" % (sum(y_pred_cls == y_true_cls) / num_test))
