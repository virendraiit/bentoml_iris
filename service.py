import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# Cerate bentoml service named here "iris_classifier". Other is model runner which specific to one model. Write one model after other
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

#Create api. Input and output as numpy array with n dimension
# input series- here 4 features- petal,sepal-length,width

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result