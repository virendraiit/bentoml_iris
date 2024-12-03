import bentoml

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner() # get latest model from the iris_clf model located in local bentoml folder
iris_clf_runner.init_local()
print(iris_clf_runner.predict.run([[5.9, 3.0, 5.1, 1.8]]))