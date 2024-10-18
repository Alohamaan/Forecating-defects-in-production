import re

from abc import ABC, abstractmethod
import numpy as np
from scripts.main import df
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              VotingClassifier, ExtraTreesClassifier, BaggingClassifier)

from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

import shap
from alibi.explainers import AnchorTabular
from alibi.utils import gen_category_map

from warnings import simplefilter
simplefilter('ignore')

class Model(ABC):

    @abstractmethod
    def CreateTest(df: pd.DataFrame, name='DefectStatus'):

        """Creates test, using every column in dataframe except one, if not mentioned, except DefectStatus,
        also reduces class imbalance"""

        x = df[[x for x in df.columns if x != name]]
        y = df[[name]]

        r_state = 52
        pipe = Pipeline([('over', RandomOverSampler(random_state=r_state)),
                         ('nearmiss', NearMiss())])
        x, y = pipe.fit_resample(x, y)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)#17

        return [X_train, X_test, y_train, y_test, r_state]

    @abstractmethod
    def learn(df: pd.DataFrame):
        pass

    @abstractmethod
    def CreateEstimator(self=None):

        """Creates list of estimators, consists of next ML algorithms: RandomForestClassifier, ExtraTreesClassifier,
        HistGradientBoostingClassifier, BaggingClassifier"""

        estimator = []

        estimator.append(('RFC', RandomForestClassifier(random_state=52)))
        estimator.append(('EXT', ExtraTreesClassifier(random_state=52)))
        estimator.append(('HGBC', HistGradientBoostingClassifier(random_state=52)))
        estimator.append(('BGC', BaggingClassifier(random_state=52)))

        return estimator

class Voting(Model):

    """Creates Voting Classifier made of 4 estimators, that were mentioned in Model.CreateEstimator"""

    def learn(df: pd.DataFrame, type = 'soft', weight=None):
        if weight is None:
            weight = [4, 3, 2, 1]

        X_train, X_test, y_train, y_test, r_state = Model.CreateTest(df=df)
        x = df[[x for x in df.columns if x != 'DefectStatus']]
        y = df['DefectStatus']

        vot = VotingClassifier(estimators=Model.CreateEstimator(),
                               n_jobs=-1, weights=weight)
        vot.fit(X_train, y_train)
        return vot


class XAI:

    def BuildExplanationPlot(model, X_train: pd.DataFrame, X_test: pd.DataFrame, slice=100):
        X_train = shap.sample(X_train, slice)
        X_test = shap.sample(X_test, slice)

        ex = shap.KernelExplainer(model.predict, X_train, keep_index=True)
        shap_values = ex.shap_values(X_test)
        expected_value = ex.expected_value
        shap_explained = shap.Explanation(shap_values, feature_names=X_train.columns)

        shap.summary_plot(shap_explained, X_test)

        shap.plots.bar(shap_values=shap_explained, max_display=len(X_train.columns))

        shap.plots.decision(base_value=expected_value, shap_values=shap_values,
                            feature_names=list(X_train.columns))
        return

    def BuildAnchorTabular(model, X_train: pd.DataFrame, X_test: pd.DataFrame, slice=100):
        features = list(X_train.columns)
        cat_map = gen_category_map(X_train)

        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()

        ex = AnchorTabular(model.predict, feature_names=features, categorical_names=cat_map)
        ex.fit(X_train)

        res = {x:0 for x in features}
        for i in range(1, len(X_test[:slice])):

            explanation = ex.explain(X=X_test[i-1:i]) # тк он возвращает список, то его можно разбить и после подсчитать встречу каждого из якорей
            temp = explanation.data['anchor'] # и построить столбчатый график того, как часто каждый из них встречался

            if len(temp) > 0:
                for x in temp:
                    res[re.search('[a-z]+', x, flags=re.IGNORECASE).group()] += 1

        anchorAppearences = sorted(res.items(), key=lambda x: x[1], reverse=True)

        data = pd.DataFrame(data=anchorAppearences, columns=['Feature', 'Score'])
        data.plot.bar(y='Score', x='Feature')
        plt.show()

        return

