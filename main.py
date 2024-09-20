import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import permutations as pm

df = pd.read_csv('manufacturing_defect_dataset.csv')    
pd.set_option('display.max_columns', 18)                
pd.set_option('display.max_rows', 10)                   

def AnswerDependedOnQuestion(df, question1, question2):

    """возвращает график датафрейма, сгруппированного по среднему значению первого вопроса"""

    return df.groupby([f'{question1}'])[f'{question2}'].agg([np.mean]).sort_values(by='mean').plot(kind = 'bar', grid = True, ylabel = f'mean {question2}', rot = 0)

class Question1:

    def DeliveryDelay(df):

        """показывает корреляцию между уровнем дефектов и временем задержки поставок"""

        return AnswerDependedOnQuestion(df, 'DeliveryDelay', 'DefectRate')
        
    def SupplierQuality(df):                                                        

        """показывает корреляцию между уровнем дефектов и качеством поставщика"""

        df = pd.DataFrame([df['SupplierQuality'].round(), df['DefectRate']]).T
        return AnswerDependedOnQuestion(df, 'SupplierQuality', 'DefectRate')


class Question2:
    
    def ProductionVolume(df):

        """показывает корреляцию между простоями в производстве и объемами производства"""

        df['ProductionVolume'] = df['ProductionVolume'].apply(lambda x: x // 100 * 100)
        return AnswerDependedOnQuestion(df, 'ProductionVolume', 'DowntimePercentage')
    
    def MaintenanceHours(df):

        """показывает корреляцию между простоями в производстве и временем на обслуживание"""

        return AnswerDependedOnQuestion(df, 'MaintenanceHours', 'DowntimePercentage')
    
class Question3:
    
    def DefectRate(df):

        """показывает корреляцию между продуктивностью работников и уровнем дефектов"""

        df = pd.DataFrame([df['WorkerProductivity'].round(), df['DefectRate']]).T
        return AnswerDependedOnQuestion(df, 'WorkerProductivity', 'DefectRate') #в среднем у более продуктивных работников уровень дефектов меньше, но не сильно коррелируют
    
    def QualityScore(df):

        """показывает корреляцию между продуктивностью работников и оценкой качества"""

        df = pd.DataFrame([df['WorkerProductivity'].round(), df['QualityScore']]).T
        return AnswerDependedOnQuestion(df, 'WorkerProductivity', 'QualityScore')
    
class Question4:
    
    def EnergyConsumption(df):

        """показывает разницу между потреблением энергии на производствах с высоким и низким уровнем дефектов"""
        
        with_high_rate = df.loc[df['DefectStatus'] == 1, 'EnergyConsumption'].mean()
        with_low_rate = df.loc[df['DefectStatus'] == 0, 'EnergyConsumption'].mean()
        
        df2 = pd.DataFrame([{'High Defect Rate' : with_high_rate, 'Low Defect Rate' : with_low_rate}])
        
        return df2.plot(kind = 'bar', grid = True, ylabel = 'EnergyConsumption')
    
class Question5:
    
    def GetCostsPermuntations(df):
        """возвращает все перестановки отдельных видов затрат попарно и список затрат"""

        costs = []
        for x in df.columns:
            if 'Cost' in x:
                costs.append(f'high {x}')
                costs.append(f'low {x}')
        df2 = list(pm(costs, r=2))

        res = []
        for x in df2:

            if x[0].lstrip('highlow') != x[1].lstrip('highlow'):
                res.append(x)
                
        return (res[:len(res) // 2], costs)
    
    def Interaction(df):
        """показывает то, как все виды затрат взаимодействуют между собой для достижения минимального уровня дефектов"""

        perms = Question5.GetCostsPermuntations(df)[0]
        for x, y in perms:
            if x[3] == ' ':
                if y[3] == ' ':
                    df[f'{x} and {y}'] = np.where((df[f'{x.lstrip("highlow ")}'] <= df[f'{x.lstrip("highlow ")}'].mean()) 
                                                & (df[f'{y.lstrip("highlow ")}'] <= df[f'{y.lstrip("highlow ")}'].mean()), True, False)
                    
                else:
                    df[f'{x} and {y}'] = np.where((df[f'{x.lstrip("highlow ")}'] <= df[f'{x.lstrip("highlow ")}'].mean()) 
                                                & (df[f'{y.lstrip("highlow ")}'] >= df[f'{y.lstrip("highlow ")}'].mean()), True, False)
                    
            else:
                if y[3] == ' ':
                    df[f'{x} and {y}'] = np.where((df[f'{x.lstrip("highlow ")}'] >= df[f'{x.lstrip("highlow ")}'].mean()) 
                                                & (df[f'{y.lstrip("highlow ")}'] <= df[f'{y.lstrip("highlow ")}'].mean()), True, False)
                    
                else:
                    df[f'{x} and {y}'] = np.where((df[f'{x.lstrip("highlow ")}'] >= df[f'{x.lstrip("highlow ")}'].mean()) 
                                                & (df[f'{y.lstrip("highlow ")}'] >= df[f'{y.lstrip("highlow ")}'].mean()), True, False)
                    

        
        df2 = pd.DataFrame()
        for x, y in perms:
            df2[f'{x} and {y}'] = np.where(df[f'{x} and {y}'] == True, df['DefectRate'], np.nan)
            df2[f'{x} and {y}'] = df2[f'{x} and {y}'].mean()
        df2 = df2.drop_duplicates()
        return df2.plot(kind = 'bar', grid = True, ylabel = 'Interaction')
    
    


'''
в целом, я знаю, что код далеко не лучшего качества
как минимум из-за большого количества дубляжа кода,
но я пока не знаю как его убрать 
'''

