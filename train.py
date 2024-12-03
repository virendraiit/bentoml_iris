import bentoml

from sklearn import svm
from sklearn import datasets

# Load training dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train the model
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# Save model to the BentoML local model store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"Model saved: {saved_model}")

# print result- (Keep this tag ready for inferencing) Model saved: Model(tag="iris_clf:zitbfzfrio3pj6fu")