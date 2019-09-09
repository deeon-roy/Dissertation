from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC 
from sklearn.externals import joblib

def kNN(train_set, train_classes, n_neighbours=1): 
	model_knn = KNeighborsClassifier(n_neighbors=n_neighbours)
	model_knn.fit(train_set, train_classes) 
	return model_knn

def SVM(train_set, train_classes, kernel='rbf', C=10.0, gamma=0.1): 
	svm = SVC(kernel=kernel, C=C, gamma=gamma)
	svm = svm.fit(train_set, train_classes)
	return svm 

def save_model(model, filename): 
	joblib.dump(model, filename+".joblib")

def load_model(filename): 
	model = joblib.load(filename+".joblib")
	return model

def validate(model, test_set, test_classes): 
	predicted_labels = model.predict(test_set)

	print("Recall: ", recall_score(test_classes, predicted_labels,average=None))
	print("Precision: ", precision_score(test_classes, predicted_labels,average=None))
	print("F1-Score: ", f1_score(test_classes, predicted_labels, average=None))
	print("Accuracy: %.2f  ," % accuracy_score(test_classes, predicted_labels,normalize=True), accuracy_score(test_classes, predicted_labels,normalize=False) )
	print("Number of samples:",test_classes.shape[0])

	print(classification_report(test_classes, predicted_labels))

	return predicted_labels 

def validate_dl(model, test_set, test_classes, labelencoder): 
	predicitions = model.predict_generator(test_set, test_classes, steps=len(test_set), verbose=1)
	predicitions_round = np.around(test_predictions).astype("int")
	predictions_int = np.argmax(predicted_round, axis=1)
	predictions_labels = labelencoder.inverse_transform(np.ravel(predictions_int))

	print("Recall: ", recall_score(test_classes, predictions_int,average=None))
	print("Precision: ", precision_score(test_classes, predictions_int,average=None))
	print("F1-Score: ", f1_score(test_classes, predicions_int, average=None))
	print("Accuracy: %.2f  ," % accuracy_score(test_classes, predictions_int,normalize=True), accuracy_score(test_classes, predictions_int,normalize=False) )
	print("Number of samples:",test_classes.shape[0])

	print(classification_report(test_classes, predictions_int))

	return predictions_int, predictions_labels

