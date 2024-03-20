import numpy

def mae(predictions:numpy.ndarray, y:numpy.ndarray)->float:
    return numpy.abs(y - predictions).mean() # edit?

def mse(predictions:numpy.ndarray, y:numpy.ndarray)->float:
    return numpy.power(y-predictions,2).mean()

def rmse(predictions:numpy.ndarray, y:numpy.ndarray)->float:
    return numpy.sqrt(numpy.power(y-predictions,2).mean())

def evaluate(predictions:numpy.ndarray, y:numpy.ndarray, eval_name:str):
    match eval_name:
        case "MAE":
            return mae(predictions,y)
        case "MSE":
            return mse(predictions,y)
        case "RMSE":
            return rmse(predictions,y)
        
        case _:
            raise(NotImplementedError)