import pandas as pd

class Parser():
    def __init__(self, DataFrame: pd.DataFrame):
        self.data = DataFrame
    
    def greetings(self):
        """Function returns dict, that contains dlg_id, """
        greet = ['добр', 'здрав']
        res = {}
        for id, group in self.data.groupby('dlg_id'):
            test = [x for x in group.loc[group['role'] == 'client', 'text']\
                for y in greet if y in x]
            res[id] = test
        
        return res
