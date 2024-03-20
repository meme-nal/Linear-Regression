from timeit import default_timer as timer
import yaml
from src.data.make_dataset import get_dataset_data
from src.modeling.models import get_model
from src.modeling.evaluations import evaluate
from src.visualization import visualize
from generate_results import generate_results



results = {}
train_times = {}

with open("./src/config/config.yml") as stream:
    try:
        exp00 = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# data
X_train, X_test, y_train, y_test = get_dataset_data(exp00['dataset'])


# FIRST MODEL | Linear Regression with analitical solution and without regularizations
model = get_model(exp00['model'])

start = timer()
model.train(X_train, y_train)
end = timer()

train_time = end - start

predictions = model.predict(X_test)
accuracy = evaluate(predictions, y_test, exp00['evaluation'])

results[exp00['model']['type']] = accuracy
train_times[exp00['model']['type']] = train_time




# SECOND MODEL | Linear Regression with analitical solution and regularizations
with open("./src/config/experiments/exp01_config.yml") as stream:
    try:
        exp01 = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model = get_model(exp01['model'])

start = timer()
model.train(X_train, y_train)
end = timer()

train_time = end - start

predictions = model.predict(X_test)
accuracy = evaluate(predictions, y_test, exp01['evaluation'])

results[exp01['model']['type']] = accuracy
train_times[exp01['model']['type']] = train_time




# THIRD MODEL | Linear Regression with gradient descent and without regularizations
with open("./src/config/experiments/exp02_config.yml") as stream:
    try:
        exp02 = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model = get_model(exp02['model'])

start = timer()
model.train(X_train, y_train)
end = timer()

train_time = end - start

predictions = model.predict(X_test)
accuracy = evaluate(predictions, y_test, exp02['evaluation'])

results[exp02['model']['type']] = accuracy
train_times[exp02['model']['type']] = train_time
visualize.plot_losses(exp02['model']['epochs'], model.losses, name="exp02")




# FOURTH MODEL | Linear Regression with gradient descent and L1-regularization
with open("./src/config/experiments/exp03_config.yml") as stream:
    try:
        exp03 = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model = get_model(exp03['model'])

start = timer()
model.train(X_train, y_train)
end = timer()

train_time = end - start

predictions = model.predict(X_test)
accuracy = evaluate(predictions, y_test, exp03['evaluation'])

results[exp03['model']['type']] = accuracy
train_times[exp03['model']['type']] = train_time
visualize.plot_losses(exp03['model']['epochs'], model.losses, name="exp03")




# FIFTH MODEL | Linear Regression with gradient descent and L2-regularization
with open("./src/config/experiments/exp04_config.yml") as stream:
    try:
        exp04 = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model = get_model(exp04['model'])

start = timer()
model.train(X_train, y_train)
end = timer()

train_time = end - start

predictions = model.predict(X_test)
accuracy = evaluate(predictions, y_test, exp04['evaluation'])

results[exp04['model']['type']] = accuracy
train_times[exp04['model']['type']] = train_time
visualize.plot_losses(exp04['model']['epochs'], model.losses, name="exp04")



# SIXTH MODEL | Linear Regression with gradient descent and ElasticNet regularization
with open("./src/config/experiments/exp05_config.yml") as stream:
    try:
        exp05 = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model = get_model(exp05['model'])

start = timer()
model.train(X_train, y_train)
end = timer()

train_time = end - start

predictions = model.predict(X_test)
accuracy = evaluate(predictions, y_test, exp05['evaluation'])

results[exp05['model']['type']] = accuracy
train_times[exp05['model']['type']] = train_time
visualize.plot_losses(exp05['model']['epochs'], model.losses, name="exp05")


generate_results(results, train_times)