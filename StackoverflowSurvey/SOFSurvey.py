import pandas as pd
pd.set_option('display.max_columns',85)

naVals = ['None','Missing','NA']
surveyData = pd.read_csv('D:\\tensor\StackoverflowSurvey\developer_survey_2019\survey_results_public.csv', na_values=naVals)
surveySchema = pd.read_csv('D:\\tensor\StackoverflowSurvey\developer_survey_2019\survey_results_schema.csv', index_col= "Column")

# print(surveyData.columns)
# print(surveyData.dtypes)
# $$$$$ Access rows and col $$$$$
# print(surveyData.iloc[[0,1]])
# print(surveyData.iloc[[0,1],[0,1,2]])
# print(surveyData.iloc[0:1,0:2])
# print(surveyData.loc[[0,1],['Respondent','Hobbyist','MainBranch']])
# print(surveyData.loc[0:1,['Respondent','Hobbyist','MainBranch']])
# print(surveyData.loc[0:1,'Respondent':'Student'])

# print(type(surveyData))

# $$$$$ change index $$$$$
# print(surveyData.index)
surveyData.set_index('Respondent', inplace= True)
# print(surveyData.index)
# print(surveySchema.loc['MgrIdiot'])
# print(surveySchema.loc['MgrIdiot','QuestionText'])
# print(surveySchema.sort_index())

# $$$$$ freq count $$$$$
# print(surveyData['Country'].value_counts())

# $$$$$ filtering $$$$$
# countries = ['United States','Canada','United Kingdom','India']
# highSal = (surveyData['ConvertedComp'] > 100000) & (surveyData['Country'].isin(countries))
# print(surveyData.loc[highSal, ['Country','LanguageWorkedWith','ConvertedComp']])
# print(surveyData.loc[highSal,'Country'].value_counts())

# print(surveyData.loc[((surveyData['LanguageWorkedWith'].str.contains('Python', na = False))& highSal), ['Country','LanguageWorkedWith','ConvertedComp']])
# print(surveyData.loc[((surveyData['LanguageWorkedWith'].str.contains('Python', na = False)) & highSal), 'Country'].value_counts())

# $$$$$ renaming col names $$$$$
surveyData.rename(columns = {'ConvertedComp' : 'Salary'}, inplace= True)
# print(surveyData.columns)

# $$$$$ apply - for series only $$$$$
# print(surveyData['Country'].iloc[0:10])
# print(surveyData['Country'].iloc[0:10].apply(len))

# def lowerCase(x):
#     return x.lower()
# print(surveyData['Country'].iloc[0:10].apply(lowerCase))

# print(surveyData['Country'].iloc[0:10].apply(lambda x:x.lower()))

# print(surveyData[['Country','Salary']].iloc[0:10])
# print(surveyData[['Country','Salary']].iloc[0:10].apply(len))
# print(surveyData[['Country','Salary']].iloc[0:10].apply(len,axis = 'columns'))
# print(surveyData[['Country','Salary']].iloc[0:10].apply(pd.Series.min))
# print(surveyData[['Country','Salary']].iloc[0:10].apply(lambda x:x.min()))

# $$$$$ applymap - for the whole dataframe $$$$$
# print(surveyData[['Country','EdLevel']].iloc[0:10])
# print(surveyData[['Country','EdLevel']].iloc[0:10].applymap(len))
# print(surveyData[['Country','EdLevel']].iloc[0:10].applymap(str.lower))

# $$$$$ map - changing values of all rows in a col $$$$$
# print(surveyData['Hobbyist'])
# surveyData['Hobbyist'] = surveyData['Hobbyist'].map({'Yes':True, 'No':False})
# print(surveyData['Hobbyist'])
# print(surveyData['Hobbyist'].value_counts())

# $$$$$ replace - change only specific rows in a col $$$$$

# print(surveyData[['Country']].iloc[0:10])
# print(surveyData['Country'].iloc[0:10].replace({'United Kingdom':'UK','United States':'US'}))
# by using map all values are updated
# print(surveyData['Country'].iloc[0:10].map({'United Kingdom':'UK','United States':'US'}))

# $$$$$ add and remove col $$$$$
# print(surveyData[['Ethnicity', 'Gender']].iloc[0:20])
# tmp = surveyData[['Ethnicity', 'Gender']].iloc[0:20]
# tmp['Gender_Ethnicity'] = tmp['Gender'] + ' of ' + tmp['Ethnicity']
# print(tmp)

# tmp.drop(columns=['Gender_Ethnicity'],inplace=True)
# print(tmp)
# print(tmp['Ethnicity'])
# print(tmp['Ethnicity'].str.split(';',expand = True))

# $$$$$ add and remove rows $$$$$
# tmp = surveyData[['Ethnicity', 'Gender']].iloc[0:9]
# tmp = tmp.append({'Ethnicity':'African'},ignore_index=True)
# print(tmp)
# tmp = tmp.drop(index = 9)
# tmp = tmp.drop(index = tmp[tmp['Ethnicity'] == 'African'].index)
# print(tmp)

# $$$$$ sorting $$$$$
# tmp = surveyData[['Country', 'Salary']].iloc[0:20]
# print(tmp)

# print(tmp.sort_values(by = ['Country','Salary'], ascending=[True,False]))
# print(tmp.sort_index())

# print(tmp.nlargest(10,'Salary'))
# print(tmp.nsmallest(10,'Salary'))

# $$$$$ Grouping and Aggregating $$$$$

# print(surveyData['Salary'].median())
# print(surveyData['Salary'].count())
# print(surveyData.describe())
# print(surveyData['Hobbyist'].value_counts())
# print(surveyData['SocialMedia'].value_counts())
# print(surveyData['SocialMedia'].value_counts(normalize=True)*100)
# print(surveyData['Country'].value_counts())

# tmp = surveyData.groupby(['Country'])['SocialMedia'].value_counts()
# print(tmp)
# print(tmp['India'])

# tmp = surveyData.groupby(['Country'])['Salary'].median()
# print(tmp.sort_values(ascending=False).head(50))

# tmp = surveyData.groupby(['Country'])['Salary'].agg(['median','mean','count'])
# print(tmp.sort_values(by = 'count',ascending=False).head(50))

# tmp = surveyData.groupby(['Country'])['LanguageWorkedWith'].apply(lambda x:x.str.contains('Python', na = False).sum())
# print(tmp.sort_values(ascending=False).head(50))

# tmp = pd.concat([tmp,surveyData['Country'].value_counts()], axis='columns')
# tmp.rename(columns={'LanguageWorkedWith':'PythonCount', 'Country':'TotalCount'},inplace=True)
# tmp['percentPython'] = tmp['PythonCount']/tmp['TotalCount']*100
# print(tmp.sort_values(['TotalCount'],ascending=False).head(50))

# $$$$$ Cleaning Data - Casting Datatypes and Handling Missing Values $$$$$

# tmp = surveyData[['Country', 'Salary', 'CodeRevHrs']].iloc[0:20]
# print(tmp)
# print(tmp.dropna())
# print(tmp.isna())
# print(tmp.dropna(axis = 'columns', how='any'))
# print(tmp.dropna(axis = 'index', how='all'))
# print(tmp.dropna(axis = 'index', how='any', subset = ['Salary']))

# print(tmp.dropna(axis = 'index', how='any', subset = ['Salary','CodeRevHrs']))
# print(tmp.dropna(axis = 'index', how='all', subset = ['Salary','CodeRevHrs']))

# print(tmp.fillna(-1))
# print(tmp.dtypes)
# print(tmp.fillna(-1).dtypes)
# tmp['CodeRevHrs'] = tmp['CodeRevHrs'].fillna(-1).astype(int)
# print(tmp.dtypes)

# tmp = surveyData['YearsCode']
# print(tmp.dtypes)
# print(tmp.unique())
# tmp = tmp.replace({'Less than 1 year':0, 'More than 50 years':51})
# tmp = tmp.fillna(0).astype(int)
# print(tmp.dtypes)
# print(tmp.unique())
# print(tmp.mean())

# $$$$$ Dates and Time Series Data $$$$$

# tsData = pd.read_csv('D:\\tensor\StackoverflowSurvey\TS.txt')
# print(tsData.head(10))
# print(tsData.dtypes)
#
# tsData['Date'] = pd.to_datetime(tsData['Date'], format='%Y-%m-%d %I-%p')
# print(tsData.head(10))
# print(tsData.dtypes)

# dParser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %I-%p')
tsData = pd.read_csv('D:\\tensor\StackoverflowSurvey\TS.txt', parse_dates= ['Date'], date_parser=(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %I-%p')))
# print(tsData.head(10))

tsData['dayOfWeek'] = tsData['Date'].dt.day_name()
# print(tsData.head(10))
# print(tsData['dayOfWeek'].unique())
# print(tsData['Date'].min())
# print(tsData['Date'].max())
# print(tsData['Date'].max()-tsData['Date'].min())

# tmp = tsData[(tsData['Date']>='2019') & (tsData['Date']<'2020')]
# tmp = tsData[(tsData['Date']>=pd.to_datetime('2019-01-01')) & (tsData['Date']<pd.to_datetime('2019-03-01'))]
# print(tmp)
# print(tmp['Date'].min())
# print(tmp['Date'].max())

tsData.set_index('Date', inplace=True)
# tmp = tsData['2019-01':'2019-02']
# print(tmp)
# print(tmp.index.min())
# print(tmp.index.max())
# print(tmp['Close'].mean())

# tmp = tsData['High'].resample('D').max()
# print(tmp)

tmp = tsData.resample('W').agg({'Close':'mean', 'High':'max', 'Low':'min', 'Volume':'sum'})
print(tmp)










