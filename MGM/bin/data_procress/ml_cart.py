from sklearn import tree
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image
import pydotplus

def make_tree_graph(clf,feature_name):
    dot_data = tree.export_graphviz(clf, max_depth=3, feature_names = feature_name, out_file=None,
                  filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())

    # graph.write_pdf("MGM_tree.pdf")
    # Image(graph.create_png())

    # tree.export_graphviz(clf, out_file="./mgm_ml/iris.do")

def Class_Tree(X_train,y_train,X_test,y_test,feature_name):
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    make_tree_graph(clf,feature_name)



def load_data():
    X_train = np.loadtxt(open("./mgm_ml/train_feature.csv", "rb"), delimiter="\t")
    y_train = np.loadtxt(open("./mgm_ml/train_label.csv", "rb"), delimiter="\t")
    X_test = np.loadtxt(open("./mgm_ml/valid_feature.csv", "rb"), delimiter="\t")
    y_test = np.loadtxt(open("./mgm_ml/valid_label.csv", "rb"), delimiter="\t")
    feature_name = pd.read_csv("./mgm_ml/feature_name.csv").ix[:,0].values
    return X_train, y_train, X_test, y_test, feature_name

if __name__ == '__main__':
    X_train, y_train, X_test, y_test,feature_name = load_data()
    Class_Tree(X_train, y_train, X_test, y_test, feature_name)