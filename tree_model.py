from main import df
from abc import ABC, abstractmethod

from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
                              ExtraTreesClassifier, HistGradientBoostingClassifier)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline

from warnings import simplefilter
simplefilter('ignore')

class Model(ABC):

    @abstractmethod
    def CreateTest(df, name='DefectStatus'):
        """Creates test depended on all data except one column, also reduces imbalance of the classes"""

        x = df[[x for x in df.columns if x != name]]
        y = df['DefectStatus']
        r_state = 52
        pipe = Pipeline([('over', RandomOverSampler(random_state=r_state)), ('near', NearMiss())])

        x, y = pipe.fit_resample(x, y)
        X_train, X_holdout, y_train, y_holdout = train_test_split(x, y,
                                                                  test_size=0.3,
                                                                  random_state=17)
        return [X_train, X_holdout, y_train, y_holdout, r_state]

    @abstractmethod
    def learn(df):
         pass

    @abstractmethod
    def CreateEstimator():
        """Creates list of the ML estimators (RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, HistGRadientBoostingClassifier)"""

        r_state = 52
        estimator = []

        estimator.append(('RFC', RandomForestClassifier(random_state=r_state)))
        estimator.append(('GBC', GradientBoostingClassifier(random_state=r_state)))
        estimator.append(('ETC', ExtraTreesClassifier(random_state=r_state)))
        estimator.append(('HGBC', HistGradientBoostingClassifier(random_state=r_state)))

        return estimator

class Voting(Model):

    def learn(df, type='soft'):

        X_train, X_holdout, y_train, y_holdout, r_state = Model.CreateTest(df=df)

        vot = VotingClassifier(estimators=Model.CreateEstimator(), voting=type,
                               n_jobs=-1, weights=[4,3,2,1])
        vot.fit(X_train, y_train)

        return classification_report(y_holdout, vot.predict(X_holdout))

print(Voting.learn(df=df))
