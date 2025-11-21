import utils.util
from nn_models.MLP import MLP
from sklearn.metrics import accuracy_score
import numpy as np
X_train, X_test, y_train, y_test = utils.util.get_data_multiclass()

mlp = MLP(neurons_num=[3,4,6,10,23,22,23],learning_rate=  0.01,epochs = 5000,activation="tanh")
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
print(y_pred)
print(y_test)


acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")