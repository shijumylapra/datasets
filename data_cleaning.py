#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For Objective - 1

This code is for data cleaning and preparation such that it can be used by any other notebook or code.

The data cleaning/preparation includes:
    1) removing appeal cases
    2) removing extreme cases
    3) adjusting award amount using inflation
    4) combining frequency of similar keywords if asked
    
"""

import pandas as pd
import numpy as np
import re
from functools import reduce

class data_cleaning():
    def __init__(self,inflation = True):
        self.inflation = inflation
        self.xls_data = pd.read_excel(r'../Data/Painworth Data Clean.xlsx',converters={'Trial year': str})
        self.r_data = pd.read_csv('../Data/full_keycount_df.csv')
        self.combined = {
                'brain':['brain','concussion','pituitary gland'], 'neck':['neck'], 'shoulder':['shoulder'],
                'face':['ears', 'eyes', 'teeth','face','mouth/jaw','nose'],
                'head':['head'],
                'arms':['elbow', 'hand', 'wrist','finger','arm'],
                'hair':['hair'],
                'legs':['ankle','buttock', 'toe', 'foot', 'knee', 'leg','hip','limp','sacrum','tailbone/coccyx','pelvis','groin','standing','sitting','walking'],
                'spine':['spine','nervous system', 'whiplash','paraplegia','paralysis', 'polio','quadriplegia','seizures','ankylosing spondylitis'],
                'back':['mid back','upper back','lower back','back pain'],
                'esophagus':['esophagus','feeding difficulties','larynx','throat','speech'],
                'trunck': ['ribs','collar bone/clavicle','breast','chest','bone'],
                'skin':['scar','skin','bedsores'],
                'soft_tissue':['soft tissue injuries'],
                'psychological':['loss of balance','vertigo/dizziness','stress/post traumatic stress disorder','shock',
                               'sleep','addiction','deconditioning','depression','embarrassment',
                              'epilepsy','fatigue','insomnia','humiliation','sexual abuse/assault','behavioral difficulties','psychological symptoms'],
                'organ':['appendix','spleen', 'bladder','bowel','colon','gallbladder','heart',
                       'kidney','liver','lung','pancreas','abdomen', 'stomach'],
                'blood':['blood','diabetes','blood pressure'],
                'genitals':['menstruation', 'genitals', 'vagina','hernia','ovaries/tubes',
                          'perineum','sexual dysfunction','infertility','uterus', 'urinary tract'],
                'muscle':['dystonia','fibromyalgia'],
                'disease':['hepatitis c', 'herpes'],
                'drug':['drug dependency'],
                'surgery':['surgery','rehabilitation'],
                'pregnancy':['premature birth','pregnancy','labour and delivery'],
                'others':['cystic fibrosis','developmental delay', 'sinus',
                        'disability','life expectancy reduced',
                        'lymph nodes','malnutrition','independence','weight']
                }
        self.cpi_04_to_18 = np.asarray([2,2.33,2.19,3.39,-0.95,1.83,2.74,1.25,1.32,2.11,1.27,1.26,1.16,2.99])/100 +1
        self.pred_data_clean()
        self.keycombined_df()
        
            
    def pred_data_clean(self):
        # locate appeal cases
        case_name = pd.DataFrame(self.xls_data, columns= ['Case name']).values.tolist() 
        appeal = ['BCCA','ABCA','ONCA']
        appeal_ind = []
        for i in range(len(case_name)):
            for item in appeal:
                if item.lower() in case_name[i][0].lower():
                    appeal_ind.append(i)           
        
        self.r_data=self.r_data.set_index(pd.Series(np.asarray(self.r_data['CaseNum'])))
        
        if self.inflation:
            trial_year = [int(re.search(r'\d+',date).group()) for date in self.xls_data['Trial year']]
            # The paragraph info from #416 and #417 can't be extracted due to the inconsistency of label structure
            trial_year = trial_year[:415]+ trial_year[417:]    
            self.df = self.r_data.copy(deep = True)
            self.df['trial year'] = trial_year
            self.df = self.df[self.df.iloc[:,1:-2].T.any()] # remove 0 rows #253 274 578 709 1146
            self.df = self.df.loc[:, (self.df != 0).any(axis=0)] # remove 0 columns
            self.df = self.df.iloc[self.df.iloc[:,-2].to_numpy().nonzero()[0],:] # remove 0 awards # 7 164 197 1445
            del_appeal = [] 
            for i in self.df.index:
                del_appeal.append(i not in appeal_ind)
            self.df = self.df[del_appeal] # remove appeal cases
            self.df = self.df[self.df.iloc[:,-2]<400000] # remove the extreme cases (#542 #1453 #1454)
            
            true_award = []
            for i in self.df['CaseNum']:
                if int(self.df.loc[i]['trial year']) == 2018:
                    true_award.append(self.df.loc[i]['general damage'])
                else:
                    year_ind = int(self.df.loc[i]['trial year']-2004)
                    true_award.append(self.df.loc[i]['general damage']*np.prod(self.cpi_04_to_18[year_ind:]))                
            self.df['true gd award'] = true_award 
    
        else:
            self.df = self.r_data[self.r_data.iloc[:,1:-1].T.any()] # remove 0 rows #253 274 578 709 1146
            self.df = self.df.loc[:, (self.df != 0).any(axis=0)] # remove 0 columns
            self.df = self.df.iloc[self.df.iloc[:,-1].to_numpy().nonzero()[0],:] # remove 0 awards # 7 164 197 1445
            del_appeal = [] 
            for i in self.df.index:
                del_appeal.append(i not in appeal_ind)
            self.df = self.df[del_appeal] # remove appeal cases
            self.df = self.df[self.df.iloc[:,-1]<400000] # remove the extreme case (1450)
            
        return self.df   

    def keycombined_df(self):
        self.df_new = pd.DataFrame(columns = ['CaseNum']+list(self.combined.keys()))
        self.df_new['CaseNum'] = list(self.df['CaseNum'])
        self.df_new = self.df_new.set_index(pd.Series(np.asarray(self.df['CaseNum'])))
        self.df_new['general damage'] = list(self.df['general damage'])
        if 'trial year' in self.df.columns:
            self.df_new['trial year'] = list(self.df['trial year'])
            self.df_new['true gd award'] = list(self.df['true gd award'])
            
        for group in self.combined.keys():
            self.df_new[group] = self.df[self.combined[group]].sum(axis=1)
        
        return self.df_new     
    
    
dc = data_cleaning()
df = dc.pred_data_clean() # Inflation adjusted data
df_new = dc.keycombined_df() # Combined frequency



    



    