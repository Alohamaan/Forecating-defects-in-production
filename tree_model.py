from main import df
from abc import ABC, abstractmethod

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              VotingClassifier, ExtraTreesClassifier, BaggingClassifier)

from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from warnings import simplefilter
simplefilter('ignore')

class Model(ABC):

    @abstractmethod
    def CreateTest(df, name='DefectStatus'):

        """Creates test, using every column in dataframe except one, if not mentioned, except DefectStatus,
        also reduces class imbalance"""

        x = df[[x for x in df.columns if x != name]]
        y = df[name]

        r_state = 52
        pipe = Pipeline([('over', RandomOverSampler(random_state=r_state)), ('nearmiss', NearMiss())])
        x, y = pipe.fit_resample(x, y)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)

        return [X_train, X_test, y_train, y_test, r_state]

    @abstractmethod
    def learn(df):
        pass

    @abstractmethod
    def CreateEstimator():

        """Creates list of estimators, consists of next ML algorithms: RandomForestClassifier, ExtraTreesClassifier,
        HistGradientBoostingClassifier, BaggingClassifier"""

        estimator = []

        estimator.append(('RFC', RandomForestClassifier(random_state=52)))
        estimator.append(('EXT', ExtraTreesClassifier(random_state=52)))
        estimator.append(('HGBC', HistGradientBoostingClassifier(random_state=52)))
        estimator.append(('BGC', BaggingClassifier(random_state=52)))

        return estimator

class Voting(Model):

    def learn(df, type='soft'):

        X_train, X_test, y_train, y_test, r_state = Model.CreateTest(df=df)

        vot = VotingClassifier(estimators=Model.CreateEstimator(), voting=type,
                               n_jobs=-1, weights=[4,3,2,1])

        vot.fit(X_train, y_train)
        return f1_score(y_test, vot.predict(X_test))

print(Voting.learn(df=df))
