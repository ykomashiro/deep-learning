import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist_data_folder = r'\MNIST'
mnist = input_data.read_data_sets(mnist_data_folder, one_hot=True)
X = mnist.train.images[0:1000]
y = mnist.train.labels[0:1000]


class TSNE(object):
    def cal_similarity(self, x):
        """
        Calculate the distance between x[i] and x[j] (x[i]!=x[j]).
            :param x: numpy array of shape (N, C) containing input data.
        """
        sum_x = np.sum(np.square(x), 1)
        dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
        return dist

    def cal_perplexity(self, sub_dist, idx=0, beta=1):
        """
        Calculate the perplexity.
            :param sub_dist: i^th vector of dist matrix.
            :param idx: index of diagonal element in i^th vector of dist matrix.
            :param beta: relate to variance.(1/sigmma^2)
        """
        prob = np.exp(-sub_dist*beta)
        prob[int(idx)] = 0.0
        sum_prob = np.sum(prob)
        # calculate the perplexity.
        # Note: we only calculate the entropy for convenience.
        prep = np.log(sum_prob) + beta*np.sum(sub_dist*prob)/sum_prob
        prob /= sum_prob
        return prep, prob

    def search_prob(self, x, eps=1e-5, perplexity=32.0, beta=None, max_iter=200):
        N, _ = x.shape
        dist = self.cal_similarity(x)
        base_prep = np.log2(perplexity)
        prob = np.zeros((N, N))
        beta = np.ones((N, 1))
        for i in range(N):
            minbeta = None
            maxbeta = None
            current_prep, current_prob = self.cal_perplexity(
                dist[i], i, beta[i])

            diff = current_prep-base_prep
            iters = 0
            # search proper beta(sigmma)
            while np.abs(diff) > eps and iters < max_iter:
                if diff > 0:
                    minbeta = beta[i]
                    if maxbeta == None:
                        beta[i] *= 2
                    else:
                        beta[i] = (beta[i]+maxbeta)/2
                else:
                    maxbeta = beta[i]
                    if minbeta == None:
                        beta[i] /= 2
                    else:
                        beta[i] = (minbeta+beta[i])/2
                # upddate
                current_prep, current_prob = self.cal_perplexity(
                    dist[i], i, beta[i])
                diff = current_prep-base_prep
                iters += 1
            prob[i] = current_prob
        return prob

    def fit(self, x, output_dim=2, perplexity=32.0, max_iter=1000):
        assert len(x.shape) == 2, "x must a two-dimension matrix"
        N, _ = x.shape
        eta = 200
        momentum = 0.6
        y = np.random.normal(0, 1e-4, (N, output_dim))
        dy = np.zeros_like(y)
        prev_y = np.zeros_like(y)
        P = self.search_prob(x, perplexity=perplexity)
        P += P.T
        P /= 2*N
        P = np.maximum(P, 1e-9)
        # optimize the low dimension output.
        for itr in range(max_iter):
            if itr % 100 == 0:
                print(itr)
            sum_y = np.sum(np.square(y), 1)
            dist_y = 1 / np.add(1+np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y)
            dist_y[np.diag_indices_from(dist_y)] = 0
            Q = dist_y / np.sum(dist_y)
            Q = np.maximum(Q, 1e-9)
            PQ = P-Q
            # compute the grdient of output y.
            for i in range(N):
                dy[i] = 4*np.sum((PQ[i]*dist_y[i]).reshape(N, 1)
                                 * ((y[i]-y)), axis=0)
            prev_y = -momentum * prev_y - eta * dy
            y = y + prev_y
        return y


if __name__ == "__main__":
    rgb = ['k', 'r', 'y', 'g', 'c', 'b', 'm', 'peru', 'orange', 'lime']
    index = list()
    for i in range(10):
        index.append(y == i)
    tsne = TSNE()
    result = tsne.fit(X)
    for i in range(10):
        plt.scatter(result[index[i], 0], result[index[i], 1], c=rgb[i])
    plt.show()
