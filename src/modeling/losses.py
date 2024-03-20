import numpy

def mse(predictions:numpy.ndarray, y:numpy.ndarray)->float:
    return ((y-predictions)**2).mean()

def mse_l1(predictions:numpy.ndarray, y:numpy.ndarray, weights:numpy.ndarray, lambda1:float)->float:
    return ((y-predictions)**2).mean() + lambda1 * numpy.abs(weights).sum()

def mse_l2(predictions:numpy.ndarray, y:numpy.ndarray, weights:numpy.ndarray, lambda2:float)->float:
    return ((y-predictions)**2).mean() + lambda2 * (weights**2).sum()

def mse_elasticnet(predictions:numpy.ndarray, y:numpy.ndarray, weights:numpy.ndarray, lambda1:float, lambda2:float)->float:
    return ((y-predictions)**2).mean() + lambda1 * numpy.abs(weights).sum() + lambda2 * (weights**2).sum()

def loss_function(predictions:numpy.ndarray, y:numpy.ndarray, type:str, weights=None, lambda1=None, lambda2=None):
    match type:
        case "MSE":
            return mse(predictions, y)
        case "MSE_L1":
            return mse_l1(predictions, y, weights, lambda1)
        case "MSE_L2":
            return mse_l2(predictions, y, weights, lambda2)
        case "MSE_ELASTICNET":
            return mse_elasticnet(predictions, y, weights, lambda1, lambda2)
        
        case _:
            raise(NotImplementedError)