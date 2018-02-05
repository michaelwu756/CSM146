#!/usr/bin/python
# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os
import math

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

######################################################################
# classes
######################################################################

class Data :

    def __init__(self, X=None, y=None) :
        """
        Data class.

        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y

    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.

        Parameters
        --------------------
            filename -- string, filename
        """

        # determine filename
        dir = os.path.dirname('__file__')
        f = os.path.join(dir, '..', 'data', filename)

        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]

    def plot(self, name, **kwargs) :
        """Plot data."""

        if 'color' not in kwargs :
            kwargs['color'] = 'b'

        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        if name==None:
            plt.show()
        else:
            plt.savefig(name)
            plt.clf()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, name=None, **kwargs) :
    data = Data(X, y)
    data.plot(name, **kwargs)


class PolynomialRegression() :

    def __init__(self, m=1, reg_param=0) :
        """
        Ordinary least squares regression.

        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param


    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].

        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features

        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """

        n,d = X.shape

        # part b: modify to create matrix for simple linear model
        # part g: modify to create matrix for polynomial model
        m = self.m_
        Phi = np.zeros((n,m+1))
        for i in range(0,n):
            val=[1]
            index=[(m+1)*i]
            for j in range(0,m):
                val.append(val[j]*X.flat[i])
                index.append(index[j]+1)
            np.put(Phi, index, val)

        return Phi


    def fit_GD(self, X, y, eta=None,
                eps=0, tmax=10000, verbose=False) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes

        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0 :
            raise Exception("GD with regularization not implemented")

        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()

        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration

        # GD loop
        for t in xrange(tmax) :
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None :
                eta = 1/float(1+t)
            else :
                eta = eta_input

            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta using vector math

            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            y_pred = np.zeros(n)
            for i in range(0,n):
                np.put(y_pred,i,np.dot(X[i],self.coef_))
            self.coef_=self.coef_-2*eta*np.dot(y_pred-y,X)
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)

            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break

            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec

        print 'number of iterations: %d' % (t+1)

        return self


    def fit(self, X, y, l2regularize = None ) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization

        Returns
        --------------------
            self    -- an instance of self
        """

        X = self.generate_polynomial_features(X) # map features

        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        self.coef_=np.dot(np.linalg.pinv(np.dot(X.T,X)),np.dot(X.T,y))


    def predict(self, X) :
        """
        Predict output for X.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features

        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")

        X = self.generate_polynomial_features(X) # map features

        # part c: predict y
        n,d = X.shape
        y = np.zeros(n)
        for i in range(0,n):
            np.put(y,i,np.dot(X[i],self.coef_))

        return y


    def cost(self, X, y) :
        """
        Calculates the objective function.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        # part d: compute J(theta)
        n,d = X.shape
        y_pred=self.predict(X)
        cost = 0
        for i in range(0,n):
            cost+=(y_pred.flat[i]-y.flat[i])**2

        return cost


    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            error   -- float, RMSE
        """
        # part h: compute RMSE
        n,d = X.shape
        error = math.sqrt(self.cost(X,y)/n)
        return error


    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'

        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


######################################################################
# main
######################################################################

def main() :
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')

    # part a: main code for visualizations
    print 'Visualizing data...'
    #plot_data(train_data.X, train_data.y, 'trainData.pdf')
    #plot_data(test_data.X, test_data.y, 'testData.pdf')

    # parts b-f: main code for linear regression
    print 'Investigating linear regression...'
    import time
    reg=PolynomialRegression(1)
    eta=0.0001
    print("eta="+str(eta))
    start=time.time()
    reg.fit_GD(train_data.X, train_data.y, eta)
    end=time.time()
    print("coefficients="+str(reg.coef_))
    print("cost="+str(reg.cost(train_data.X, train_data.y)))
    print("time="+str(end-start))
    print("")

    eta=0.001
    print("eta="+str(eta))
    start=time.time()
    reg.fit_GD(train_data.X, train_data.y, eta)
    end=time.time()
    print("coefficients="+str(reg.coef_))
    print("cost="+str(reg.cost(train_data.X, train_data.y)))
    print("time="+str(end-start))
    print("")

    eta=0.01
    print("eta="+str(eta))
    start=time.time()
    reg.fit_GD(train_data.X, train_data.y, eta)
    end=time.time()
    print("coefficients="+str(reg.coef_))
    print("cost="+str(reg.cost(train_data.X, train_data.y)))
    print("time="+str(end-start))
    print("")

    eta=0.0407
    print("eta="+str(eta))
    start=time.time()
    reg.fit_GD(train_data.X, train_data.y, eta)
    end=time.time()
    print("coefficients="+str(reg.coef_))
    print("cost="+str(reg.cost(train_data.X, train_data.y)))
    print("time="+str(end-start))
    print("")

    print("eta=1/(1+k)")
    start=time.time()
    reg.fit_GD(train_data.X, train_data.y)
    end=time.time()
    print("coefficients="+str(reg.coef_))
    print("cost="+str(reg.cost(train_data.X, train_data.y)))
    print("time="+str(end-start))
    print("")

    print("Closed form fit")
    start=time.time()
    reg.fit(train_data.X, train_data.y)
    end=time.time()
    print("coefficients="+str(reg.coef_))
    print("cost="+str(reg.cost(train_data.X, train_data.y)))
    print("time="+str(end-start))
    print("")

    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    print 'Investigating polynomial regression...'

    ### ========== TODO : END ========== ###


    print "Done!"

if __name__ == "__main__" :
    main()
