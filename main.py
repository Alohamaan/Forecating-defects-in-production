import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display

df = pd.read_csv('manufacturing_defect_dataset.csv')    
pd.set_option('display.max_columns', 18)                
pd.set_option('display.max_rows', 10)                   

def AnswerDependedOnQuestion(df, question1, question2):
    return df.groupby([f'{question1}'])[f'{question2}'].agg([np.mean]).sort_values(by='mean').plot(kind = 'bar', rot = 0)

class Question1:

    def DeliveryDelay(df):
        return AnswerDependedOnQuestion(df, 'DeliveryDelay', 'DefectRate')
        
    def SupplierQuality(df):                                                                                                        
        df = pd.DataFrame([df['SupplierQuality'].round(), df['DefectRate']]).T
        return AnswerDependedOnQuestion(df, 'SupplierQuality', 'DefectRate')


class Question2:
    
    def ProductionVolume(df):
        df['ProductionVolume'] = df['ProductionVolume'].apply(lambda x: x // 100 * 100)
        return AnswerDependedOnQuestion(df, 'ProductionVolume', 'DowntimePercentage')
    
    def MaintenanceHours(df):
        return AnswerDependedOnQuestion(df, 'MaintenanceHours', 'DowntimePercentage')
    
class Question3:
    
    def DefectRate(df):
        df = pd.DataFrame([df['WorkerProductivity'].round(), df['DefectRate']]).T
        return AnswerDependedOnQuestion(df, 'WorkerProductivity', 'DefectRate') #в среднем у более продуктивных работников уровень дефектов меньше, но не сильно коррелируют
    
    def QualityScore(df):
        df = pd.DataFrame([df['WorkerProductivity'].round(), df['QualityScore']]).T
        return AnswerDependedOnQuestion(df, 'WorkerProductivity', 'QualityScore')
    
class Question4:
    
    def EnergyConsumption(df):
        
        with_high_rate = df.loc[df['DefectStatus'] == 1, 'EnergyConsumption'].sum()
        with_low_rate = df.loc[df['DefectStatus'] == 0, 'EnergyConsumption'].sum()
        
        df2 = pd.DataFrame([{'High Defect Rate' : with_high_rate, 'Low Defect Rate' : with_low_rate}])
        
        return df2.plot(kind = 'bar')
    
class Question5:
    
    def Interaction(df):
        df['HighAdditiveCostAndMatCost'] = np.where((df['AdditiveMaterialCost'] > 300) & (df['ProductionCost'] > 12500), True, False)
        df['HighMaterialCostAndLowAdditCost'] = np.where((df['ProductionCost'] > 12500) & (df['AdditiveMaterialCost'] < 300), True, False)
        df['LowAdditiveCostAndMatCost'] = np.where((df['AdditiveMaterialCost'] < 300) & (df['ProductionCost'] < 12500), True, False)
        df['LowMaterialCostAndHighAdditCost'] = np.where((df['ProductionCost'] < 12500) & (df['AdditiveMaterialCost'] > 300), True, False)
        
        df2 = pd.DataFrame([{'High Additive Cost and Low Production Cost' : df.loc[df['LowMaterialCostAndHighAdditCost'] == True, 'DefectRate'].mean(),
                             'High Additive Cost and Production Cost' : df.loc[df['HighAdditiveCostAndMatCost'] == True, 'DefectRate'].mean(),
                             'Low Additive Cost and High Production Cost' : df.loc[df['HighMaterialCostAndLowAdditCost'] == True, 'DefectRate'].mean(),
                             'Low Additive Cost and Production Cost' : df.loc[df['LowAdditiveCostAndMatCost'] == True, 'DefectRate'].mean()
                             }])
        return df2.plot(kind = 'bar')


'''
в целом, я знаю, что код далеко не лучшего качества
как минимум из-за большого количества дубляжа кода,
но я пока не знаю как его убрать 
'''

