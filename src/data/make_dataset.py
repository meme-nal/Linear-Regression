import numpy
import pandas

# SHUFFLING
def shuffle(dataset:pandas.DataFrame)->pandas.DataFrame:
    dataset = dataset.sample(frac=1, random_state=19)
    return dataset

# NORMALIZATION
def z_norm(dataset:numpy.ndarray)->numpy.ndarray:
    for i in range(dataset.shape[1]):
        dataset[:,i] = (dataset[:,i] - dataset[:,i].min()) / (dataset[:,i].max() - dataset[:,i].min())
    return dataset


def train_test_split(dataset:numpy.ndarray, train_size:float)->tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    X = dataset[:,:-1]
    y = dataset[:,-1]

    X_train = X[:int(train_size*X.shape[0]),:]
    X_test = X[int(train_size*X.shape[0]):,:]
    y_train = y[:int(train_size*len(y))]
    y_test = y[int(train_size*len(y)):]

    return (X_train, X_test, y_train, y_test)


def get_dataset_data(dataset_options:dict)->tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    dataset = pandas.read_csv(dataset_options['data_path'])
    dataset = shuffle(dataset)
    dataset = dataset.to_numpy()
    dataset = z_norm(dataset)

    X_train, X_test, y_train, y_test = train_test_split(dataset, dataset_options['train_size'])
    return (X_train, X_test, y_train, y_test)
    



