import numpy as np
import sys

epochs = 50
hidden_layers = 100
learning_rate = 0.001
num_of_classes = 10
num_of_pixels = 784


def mySigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return mySigmoid(x) * (1 - mySigmoid(x))


def softMax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# shuffle and keep both arrays aligned.
def shuffle_data(train_x, labels):
    p = np.random.permutation(train_x.shape[0])
    return train_x[p], labels[p]


def forwardPropagation(row, label, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    z1 = np.dot(w1, row) + b1
    h1 = mySigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = softMax(z2)
    ret = {'x': row, 'y': label, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for key in params:
        ret[key] = params[key]
    return ret


def backPropagation(forward_cache):
    x, y, z1, h1, z2, h2 = [forward_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2')]
    dz2 = h2
    dz2[int(y)] -= 1
    dw2 = np.dot(dz2, h1.T)
    db2 = dz2
    dz1 = np.dot(h2.T,
                 forward_cache["w2"]).T * sigmoid_derivative(z1)
    dw1 = np.dot(dz1, x.T)
    db1 = dz1
    return {'db1': db1, 'dw1': dw1, 'db2': db2, 'dw2': dw2}


def neuralNetwork(train_x, labels):
    # initializing weights and biases
    w1 = np.random.uniform(-0.05, 0.05, (hidden_layers, num_of_pixels))
    b1 = np.random.uniform(-0.5, 0.5, (hidden_layers, 1))
    w2 = np.random.uniform(-0.1, 0.1, (num_of_classes, hidden_layers))
    b2 = np.random.uniform(-0.5, 0.5, (num_of_classes, 1))
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    for j in range(epochs):
        # shuffle data so every epoch the alg can learn.
        train_x, labels = shuffle_data(train_x, labels)
        for k in range(len(train_x)):
            # reshaping the dimensions.
            cur = train_x[k].reshape(-1, 1)
            # send the data to forward prop.
            forward_res = forwardPropagation(cur, labels[k], params)
            # send the result to back prop.
            back_res = backPropagation(forward_res)
            db1, dw1, db2, dw2 = [back_res[key] for key in ('db1', 'dw1', 'db2', 'dw2')]
            # updating weights and biases
            params['b1'] = params['b1'] - (learning_rate * db1)
            params['w1'] = params['w1'] - (learning_rate * dw1)
            params['b2'] = params['b2'] - (learning_rate * db2)
            params['w2'] = params['w2'] - (learning_rate * dw2)

    return params


def prediction(test_set, b1, w1, b2, w2):
    f = open("test_y", "w+")
    for row in test_set:
        row = row.reshape(-1, 1)
        params = {'b1': b1, 'w1': w1, 'b2': b2, 'w2': w2}
        fcahce = forwardPropagation(row, 3, params)
        f.write(str(np.argmax(fcahce['h2'])) + "\n")
    f.close()


def main():
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    labels = np.loadtxt(train_y)
    train_set = np.loadtxt(train_x)
    test_set = np.loadtxt(test_x)
    normalized_train = train_set / 255
    normalized_test = test_set / 255
    cahce = neuralNetwork(normalized_train, labels)
    b1 = cahce['b1']
    w1 = cahce['w1']
    b2 = cahce['b2']
    w2 = cahce['w2']
    prediction(normalized_test, b1, w1, b2, w2)


if __name__ == "__main__":
    main()
