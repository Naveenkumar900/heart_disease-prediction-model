import pickle
file="heart-disease-prediction-knn-model.pkl"
fileobj=open(file,'rb')
knn=pickle.load(fileobj)
print(knn)