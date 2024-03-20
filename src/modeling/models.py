import numpy
from src.modeling.losses import loss_function

class LinearRegressionAS():
    def __init__(self)->None:
        pass

    def train(self, X_train:numpy.ndarray, y_train:numpy.ndarray)->None:
        self.X = X_train
        self.y = y_train

        self.weights = numpy.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)

    def predict(self, X:numpy.ndarray)->numpy.ndarray:
        return X.dot(self.weights)


class LinearRegressionASL2():
    def __init__(self, lambda2:float)->None:
        self.lambda2 = lambda2

    def train(self, X_train:numpy.ndarray, y_train:numpy.ndarray)->None:
        self.X = X_train
        self.y = y_train

        self.weights = numpy.linalg.inv(self.X.T.dot(self.X) + self.lambda2*numpy.eye(self.X.shape[1])).dot(self.X.T).dot(self.y)

    def predict(self, X:numpy.ndarray)->numpy.ndarray:
        return X.dot(self.weights)


class LinearRegressionGD():
    def __init__(self, lr:float, epochs:int, weights_init:str, loss:str)->None:
        self.lr = lr
        self.epochs = epochs
        self.weights_init = weights_init
        self.loss = loss

        self.losses = []

    def add_intercept(self, X:numpy.ndarray)->numpy.ndarray:
        intercept = numpy.ones((X.shape[0],1))
        return numpy.concatenate([intercept, X], axis=1)

    def weights_initialize(self, X_train:numpy.ndarray, type:str)->None:
        match type:
            case "zeros":
                self.weights = numpy.zeros(X_train.shape[1])
            
            case _:
                raise(NotImplementedError)

    def calculate_grad(self, X:numpy.ndarray, y:numpy.ndarray, predictions:numpy.ndarray)->float:
        return 2*(numpy.dot(X.T, (predictions - y))) / y.size

    def train(self, X_train:numpy.ndarray, y_train:numpy.ndarray)->None:
        X = self.add_intercept(X_train)
        self.weights_initialize(X, self.weights_init)

        for epoch in range(self.epochs):
            prediction = X.dot(self.weights)
            loss = loss_function(prediction, y_train, self.loss)
            self.losses.append(loss)

            gradient = self.calculate_grad(X, y_train, prediction)
            self.weights -= self.lr * gradient

            if epoch % 100 == 0:
                print(f"epoch's number: {epoch} | loss: {loss}")
        print("\n")

    def predict(self, X_test:numpy.ndarray)->numpy.ndarray:
        X_test = self.add_intercept(X_test)
        return X_test.dot(self.weights)


class LinearRegressionGDL1():
    def __init__(self, lr:float, epochs:int, weights_init:str, loss:str, lambda1:float)->None:
        self.lr = lr
        self.epochs = epochs
        self.weights_init = weights_init
        self.loss_type = loss
        self.lambda1 = lambda1

        self.losses = []

    def add_intercept(self, X:numpy.ndarray)->numpy.ndarray:
        intercept = numpy.ones((X.shape[0],1))
        return numpy.concatenate([intercept, X], axis=1)
    
    def weights_initialize(self, X_train:numpy.ndarray, type:str)->None:
        match type:
            case "zeros":
                self.weights = numpy.zeros(X_train.shape[1])
            
            case _:
                raise(NotImplementedError)
            
    def calculate_grad(self, X:numpy.ndarray, y:numpy.ndarray, predictions:numpy.ndarray)->float:
        return 2*(numpy.dot(X.T, (predictions - y))) / y.size + self.lambda1 * len(self.weights)
    
    def train(self, X_train:numpy.ndarray, y_train:numpy.ndarray)->None:
        X = self.add_intercept(X_train)
        self.weights_initialize(X, self.weights_init)

        for epoch in range(self.epochs):
            prediction = X.dot(self.weights)
            loss = loss_function(prediction,y_train, self.loss_type, self.weights, lambda1=self.lambda1)
            self.losses.append(loss)

            gradient = self.calculate_grad(X, y_train, prediction)
            self.weights -= self.lr * gradient

            if epoch % 100 == 0:
                print(f"epoch's number: {epoch} | loss: {loss}")
        print("\n")

    def predict(self, X_test:numpy.ndarray)->numpy.ndarray:
        X_test = self.add_intercept(X_test)
        return X_test.dot(self.weights)



class LinearRegressionGDL2():
    def __init__(self, lr:float, epochs:int, weights_init:str, loss:str, lambda2:float)->None:
        self.lr = lr
        self.epochs = epochs
        self.weights_init = weights_init
        self.loss_type = loss
        self.lambda2 = lambda2

        self.losses = []

    def add_intercept(self, X:numpy.ndarray)->numpy.ndarray:
        intercept = numpy.ones((X.shape[0],1))
        return numpy.concatenate([intercept, X], axis=1)
    
    def weights_initialize(self, X_train:numpy.ndarray, type:str)->None:
        match type:
            case "zeros":
                self.weights = numpy.zeros(X_train.shape[1])
            
            case _:
                raise(NotImplementedError)
            
    def calculate_grad(self, X:numpy.ndarray, y:numpy.ndarray, predictions:numpy.ndarray)->float:
        return 2*(numpy.dot(X.T, (predictions - y))) / y.size + self.lambda2 * 2 * self.weights.sum()
    
    def train(self, X_train:numpy.ndarray, y_train:numpy.ndarray)->None:
        X = self.add_intercept(X_train)
        self.weights_initialize(X, self.weights_init)

        for epoch in range(self.epochs):
            prediction = X.dot(self.weights)
            loss = loss_function(prediction,y_train, self.loss_type, self.weights, lambda2=self.lambda2)
            self.losses.append(loss)

            gradient = self.calculate_grad(X, y_train, prediction)
            self.weights -= self.lr * gradient

            if epoch % 100 == 0:
                print(f"epoch's number: {epoch} | loss: {loss}")
        print("\n")

    def predict(self, X_test:numpy.ndarray)->numpy.ndarray:
        X_test = self.add_intercept(X_test)
        return X_test.dot(self.weights)


class LinearRegressionGDElasticNet():
    def __init__(self, lr:float, epochs:int, weights_init:str, loss:str, lambda1:float, lambda2:float)->None:
        self.lr = lr
        self.epochs = epochs
        self.weights_init = weights_init
        self.loss_type = loss
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.losses = []

    def add_intercept(self, X:numpy.ndarray)->numpy.ndarray:
        intercept = numpy.ones((X.shape[0],1))
        return numpy.concatenate([intercept, X], axis=1)
    
    def weights_initialize(self, X_train:numpy.ndarray, type:str)->None:
        match type:
            case "zeros":
                self.weights = numpy.zeros(X_train.shape[1])
            
            case _:
                raise(NotImplementedError)
            
    def calculate_grad(self, X:numpy.ndarray, y:numpy.ndarray, predictions:numpy.ndarray)->float:
        return 2*(numpy.dot(X.T, (predictions - y))) / y.size + self.lambda1 * len(self.weights) + self.lambda2 * 2 * self.weights.sum()
    
    def train(self, X_train:numpy.ndarray, y_train:numpy.ndarray)->None:
        X = self.add_intercept(X_train)
        self.weights_initialize(X, self.weights_init)

        for epoch in range(self.epochs):
            prediction = X.dot(self.weights)
            loss = loss_function(prediction,y_train, self.loss_type, self.weights, lambda1=self.lambda1, lambda2=self.lambda2)
            self.losses.append(loss)

            gradient = self.calculate_grad(X, y_train, prediction)
            self.weights -= self.lr * gradient

            if epoch % 100 == 0:
                print(f"epoch's number: {epoch} | loss: {loss}")
        print("\n")

    def predict(self, X_test:numpy.ndarray)->numpy.ndarray:
        X_test = self.add_intercept(X_test)
        return X_test.dot(self.weights)
    

def get_model(model_options:dict):
    match model_options['type']:
        case "LinearRegressionAS":
            return LinearRegressionAS()
        
        case "LinearRegressionASL2":
            return LinearRegressionASL2(
                model_options['lambda'])
        
        case "LinearRegressionGD":
            return LinearRegressionGD(
                model_options['lr'],
                model_options['epochs'],
                model_options['weights_init'],
                model_options['loss'])
        
        case "LinearRegressionGDL1":
            return LinearRegressionGDL1(
                model_options['lr'],
                model_options['epochs'],
                model_options['weights_init'],
                model_options['loss'],
                model_options['lambda1'])
        
        case "LinearRegressionGDL2":
            return LinearRegressionGDL2(
                model_options['lr'],
                model_options['epochs'],
                model_options['weights_init'],
                model_options['loss'],
                model_options['lambda2'])
        
        case "LinearRegressionGDElasticNet":
            return LinearRegressionGDElasticNet(
                model_options['lr'],
                model_options['epochs'],
                model_options['weights_init'],
                model_options['loss'],
                model_options['lambda1'],
                model_options['lambda2'])

        case _:
            raise(NotImplementedError)