from main import df
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from warnings import simplefilter
from sklearn.ensemble import RandomForestClassifier

simplefilter('ignore')

class TreeTest:

    def rand_forest(df):
        tree = RandomForestClassifier(random_state=52, n_estimators=100)
        x = df[[x for x in df.columns if x != 'DefectStatus']]
        y = df['DefectStatus']

        X_train, X_holdout, y_train, y_holdout = train_test_split(x, y,
                                                                  test_size=0.3,
                                                                  random_state=17)

        tree.fit(X_train, y_train)

        return classification_report(y_holdout, tree.predict(X_holdout))

print(TreeTest.rand_forest(df=df))

