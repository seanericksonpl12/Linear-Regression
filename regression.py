import numpy as np
from matplotlib import pyplot as plt
import csv

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

# Update 10/12/2021: check rounding on compute betas and prediction

def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        rtrn = np.array([rows for rows in reader])
        rtrn = np.delete(np.delete(rtrn, 0, 0), 0, 1)
    
    return rtrn.astype(float)


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    print(len(dataset))
    average = ( np.sum(dataset[:, col], dtype="float64") / float(len(dataset)) )
    stdv = np.sqrt( (np.sum( np.power([abs(x - average) for x in dataset[:, col]], 2), dtype="float64")) / float(len(dataset)) )
    print(round(average, 2))
    print(round(stdv, 2))
    
def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    x = np.power(((dataset[:, cols] @ betas[1:]) + betas[0]) - dataset[:, 0], 2)
    return np.sum(x, dtype="float64") / len(dataset)

def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []
    x = ((dataset[:, cols] @ betas[1:]) + betas[0]) - dataset[:, 0]
    grads.append((2*np.sum(x, dtype="float64")) / len(dataset))
    for col in cols:
        grads.append((2*np.sum(x @ dataset[:,col], dtype="float64")) / len(dataset))
    return np.array(grads)


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    for i in range(1, T+1):
        
        grad = gradient_descent(dataset, cols, betas)
        b = [(betas[x] - (eta*grad[x])) for x in range(0, len(grad))]
        betas = b
        mse = regression(dataset, cols, betas)
        rtrn = [i, mse]
        rtrn.extend(b)
        for element in rtrn:
            if(isinstance(element, float)):
                element = round(element, 2)
            print(element, end=" ")
        print()


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    x = dataset[:, cols]
    x = np.flip(np.append(np.flip(x, 1), np.ones((len(x), 1)), axis=1), 1)
    betas = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(dataset[:, 0])

    mse = regression(dataset, cols, betas)
    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = np.array(compute_betas(dataset, cols)[1:])
    return np.sum(features @ betas[1:], dtype="float64") + betas[0]


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    y1 = (X * betas[1]) + betas[0] + np.random.normal(0, sigma, X.shape)
    y2 = (np.power(X, 2) * alphas[1]) + alphas[0] + np.random.normal(0, sigma, X.shape)
    return np.append(y1, X, axis=1), np.append(y2, X, axis=1)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = np.random.randint(-100, 101, size=1000).reshape((1000,1))
    sigma = 1e-4
    betas = np.array([2, 3])
    alphas = np.array([2, 3])
    xAxis = []
    yAxisL = []
    yAxisQ = []
    while(sigma <= 1e5):
        b, a = synthetic_datasets(betas, alphas, X, sigma)
        xAxis.append(sigma)
        yAxisL.append(compute_betas(b, [1])[0])
        yAxisQ.append(compute_betas(a, [1])[0])
        sigma = sigma * 10
        
    plt.plot(xAxis, yAxisL, '-o', label="Linear")
    plt.plot(xAxis, yAxisQ, '-o', label="Quadratic")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Sigma")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc='lower right')
    plt.savefig('mse.pdf')

if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
