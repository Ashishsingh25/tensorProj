import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
# import pandas_datareader.data as web
from numpy import nan as NA
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.parser import parse
import pytz

# numpy

# points = np.arange(-5,5,0.01)
# xs, ys = np.meshgrid(points, points)
# z = np.sqrt(xs ** 2 + ys ** 2)
# plt.imshow(z, cmap=plt.cm.gray)
# plt.colorbar()
# plt.show()

# xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
# yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
# cond = np.array([True, False, True, True, False])
# result = np.where(cond, xarr, yarr)
# print(result)
# arr = np.random.randn(4, 4)
# print(arr)
# result = np.where(arr > 0, arr, 0)
# print(result)

# arr = np.random.randn(4, 4)
# print(arr)
# print(arr.mean())
# print(np.mean(arr))
# print(np.sum(arr,axis=0))
# print(np.sum(arr,axis=1))
# print(np.cumsum(arr))
# print(arr.cumsum(axis=0))
# print(arr.cumsum(axis=1))

# arr = np.random.randn(100)
# print(sum(arr>0),(arr>0).sum())

# bools = np.array([False, False, True, False])
# print(bools.any())
# print(bools.all())

# arr = np.random.randn(6)
# print(arr)
# print(np.sort(arr))
# print(arr)
# arr.sort()
# print(arr)

# arr = np.random.randn(5, 3)
# print(arr)
# arr.sort(axis=0)
# print(arr)
# arr.sort(axis=1)
# print(arr)

# large_arr = np.random.randn(1000)
# large_arr.sort()
# print(large_arr[int(0.05 * len(large_arr))]) # 5% quantile

# names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# print(np.unique(names))
# print(sorted(set(names)))

# values = np.array([6, 0, 0, 3, 2, 5, 6])
# print(np.in1d(values, [2, 3, 6]))

# x = np.array([[1., 2., 3.], [4., 5., 6.]])
# y = np.array([[6., 23.], [-1, 7], [8, 9]])
# print(x.dot(y), np.dot(x, y))

# samples = np.random.normal(size=(4, 4))
# print(samples)

# nsteps = 1000
# draws = np.random.randint(0, 2, size=nsteps)
# print(draws)
# steps = np.where(draws > 0, 1, -1)
# walk = steps.cumsum()
# print(walk)
# plt.plot(walk)
# plt.show()
# print((np.abs(walk) >= 10).argmax())

# pandas

# obj = pd.Series([4, 7, -5, 3])
# print(obj.values)
# print(obj.index)
# obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
# print(obj)

# obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
# print(obj2.values)
# print(obj2.index)
# print(obj2[['c', 'a', 'd']])
# print(obj2[obj2 > 0])
# print(obj2 * 2)
# print(np.exp(obj2))
# print('b' in obj2)

# sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
# obj3 = pd.Series(sdata)
# print(obj3)
# states = ['California', 'Ohio', 'Oregon', 'Texas']
# obj4 = pd.Series(sdata, index=states)
# print(obj4)
# print(pd.isnull(obj4), obj4.isnull())
# print(obj3 + obj4)
# obj4.name = 'population'
# obj4.index.name = 'state'
# print(obj4)

# data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
#  'year': [2000, 2001, 2002, 2001, 2002, 2003],
#  'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
# frame = pd.DataFrame(data)
# print(frame)
# frame = pd.DataFrame(data, columns=['year', 'state', 'pop'])
# print(frame)

# frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
#                       index=['one', 'two', 'three', 'four','five', 'six'])
# print(frame2)
# print(frame2.loc['three'])
# frame2['debt'] = 16.5
# print(frame2)
# frame2['debt'] = np.arange(6.)
# print(frame2)
# val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
# frame2['debt'] = val
# print(frame2)
# frame2['eastern'] = frame2.state == 'Ohio'
# print(frame2)
# del frame2['eastern']
# print(frame2.columns)

# pop = {'Nevada': {2001: 2.4, 2002: 2.9},
#        'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
# frame3 = pd.DataFrame(pop)
# print(frame3)
# print(frame3.T)
# frame3.index.name = 'year'
# frame3.columns.name = 'state'
# print(frame3)
# print('Ohio' in frame3.columns)
# print(2003 in frame3.index)

# obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
# print(obj3)
# obj3 = obj3.reindex(range(6), method= 'ffill')
# print(obj3)

# frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
#                      index=['a', 'c', 'd'],
#                      columns=['Ohio', 'Texas', 'California'])
# print(frame)
# frame2 = frame.reindex(['a', 'b', 'c', 'd'])
# print(frame2)
# states = ['Texas', 'Utah', 'California']
# print(frame.reindex(columns=states))

# obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
# print(obj)
# print(obj.drop(['d', 'c']))
# data = pd.DataFrame(np.arange(16).reshape((4, 4)),
#                     index=['Ohio', 'Colorado', 'Utah', 'New York'],
#                     columns=['one', 'two', 'three', 'four'])
# print(data)
# print(data.drop(['Colorado', 'Ohio']))
# print(data.drop(['two','four'],axis=1))

# obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
# print(obj)
# print(obj[1:3]) # 3 is exclusive without index
# print(obj['b':'d']) # d is inclusive with index

# data = pd.DataFrame(np.arange(16).reshape((4, 4)),
#                     index=['Ohio', 'Colorado', 'Utah', 'New York'],
#                     columns=['one', 'two', 'three', 'four'])
# print(data)
# print(data[['three', 'one']])
# print(data[:2])
# print(data[data['three'] > 5])
# data[data < 5] = 0
# print(data)
# print(data.loc['Colorado', ['two', 'three']])
# print(data.iloc[2, [3, 0, 1]])
# print(data.loc[:'Utah', 'two'])
# print(data.iloc[:, :3][data.three > 5])

# ser = pd.Series(np.arange(3.))
# print(ser[-1]) # throws an error as indexes are numerical
# ser2 = pd.Series(np.arange(3.), index=['a', 'b', 'c'])
# print(ser2)
# print(ser2[-1]) # does not throw error

# s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
# s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],
#                index=['a', 'c', 'e', 'f', 'g'])
# print(s1, s2)
# print(s1 + s2)

# df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
#                    index=['Ohio', 'Texas', 'Colorado'])
# df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
#                    index=['Utah', 'Ohio', 'Texas', 'Oregon'])
# print(df1)
# print(df2)
# print(df1 + df2)

# df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),columns=list('abcd'))
# df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),columns=list('abcde'))
# df2.loc[1, 'b'] = np.nan
# print(df1)
# print(df2)
# print(df1 + df2)
# print(df1.add(df2, fill_value=0))
# print(df1.reindex(columns=df2.columns, fill_value=0))

# arr = np.arange(12.).reshape((3, 4))
# print(arr)
# print(arr[0])
# print(arr - arr[0]) # broadcasting (applies to each row)

# frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
#                      columns=list('bde'),
#                      index=['Utah', 'Ohio', 'Texas', 'Oregon'])
# series = frame.iloc[0]
# print(frame, series)
# print(frame - series) # broadcasting (applies to each row)
# series2 = pd.Series(range(3), index=['b', 'e', 'f'])
# print(series2)
# print(frame + series2)
# series3 = frame['d']
# print(frame.sub(series3, axis = 0))

# obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
# print(obj)
# print(obj.sort_index())
# frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
#                      index=['three', 'one'],
#                      columns=['d', 'a', 'b', 'c'])
# print(frame)
# print(frame.sort_index())
# print(frame.sort_index(1))
# print(frame.sort_index(axis=1, ascending=False))
# obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
# print(obj)
# print(obj.sort_values())
# frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
# print(frame)
# print(frame.sort_values(by='b'))
# print(frame.sort_values(by=['a', 'b'], ascending=[True,False]))

# obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
# print(obj)
# print(obj.rank())
# print(obj.rank(method='first'))

# obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
# print(obj)
# print(obj.index.is_unique)
# print(obj['a'])
# df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
# print(df)
# print(df.loc['b'])

# df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],[np.nan, np.nan], [0.75, -1.3]],
#                   index=['a', 'b', 'c', 'd'],
#                   columns=['one', 'two'])
# print(df)
# print(df.sum())
# print(df.sum(axis = 1))
# print(df.idxmax())
# print(df.describe())
# obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
# print(obj)
# print(obj.describe())

# all_data = {ticker: web.get_data_yahoo(ticker)
#             for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']}
#
# price = pd.DataFrame({ticker: data['Adj Close']
#                       for ticker, data in all_data.items()})
# volume = pd.DataFrame({ticker: data['Volume']
#                        for ticker, data in all_data.items()})
# print(price.head())
# returns = price.pct_change()
# print(returns)
# print(returns['MSFT'].corr(returns['IBM']))
# print(returns['MSFT'].cov(returns['IBM']))
# print(returns.corr())
# print(returns.cov())
# print(returns.corrwith(returns.IBM))
# print(returns.corrwith(volume))

# obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
# print(obj.unique())
# print(obj.value_counts())
# print(pd.value_counts(obj.values, sort=False))
# to_match = pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])
# unique_vals = pd.Series(['c', 'b', 'a'])
# print(pd.Index(unique_vals).get_indexer(to_match))

# read write

# tables = pd.read_html('https://www.fdic.gov/bank/individual/failed/banklist.html')
# failures = tables[0]
# print(failures.iloc[1])
# print(pd.to_datetime(failures['Closing Date']).dt.year.value_counts())

# data cleaning

# data = pd.Series([1, NA, 3.5, NA, 7])
# print(data)
# print(data.dropna())
# print(data[data.notnull()])
# data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],[NA, NA, NA], [NA, 6.5, 3.]])
# print(data)
# print(data.dropna())
# print(data.dropna(how='all'))
# data[4] = NA
# print(data)
# print(data.dropna(how='all',axis=1))

# df = pd.DataFrame(np.random.randn(7, 3))
# df.iloc[:4, 1] = NA
# df.iloc[:2, 2] = NA
# print(df)
# print(df.dropna(thresh=2))
# print(df.fillna(0))
# print(df.fillna({1: 0.5, 2: 0}))

# df = pd.DataFrame(np.random.randn(6, 3))
# df.iloc[2:, 1] = NA
# df.iloc[4:, 2] = NA
# print(df)
# print(df.fillna(method = 'ffill'))
# print(df.fillna(method='ffill', limit=2))

# data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
#                      'k2': [1, 1, 2, 3, 3, 4, 4]})
# print(data)
# print(data.drop_duplicates())
# data['v1'] = range(7)
# print(data)
# print(data.drop_duplicates(['k1']))

# data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon','Pastrami', 'corned beef', 'Bacon','pastrami', 'honey ham', 'nova lox'],
#                      'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
# print(data)
# meat_to_animal = {
#  'bacon': 'pig',
#  'pulled pork': 'pig',
#  'pastrami': 'cow',
#  'corned beef': 'cow',
#  'honey ham': 'pig',
#  'nova lox': 'salmon'
# }
# # print(data['food'].str.lower().map(meat_to_animal))
# data['animal'] = data['food'].str.lower().map(meat_to_animal)
# print(data)

# data = pd.Series([1., -999., 2., -999., -1000., 3.])
# print(data)
# print(data.replace(-999,0))
# print(data.replace([-999,-1000],0))
# print(data.replace({-999: np.nan, -1000: 0}))

# data = pd.DataFrame(np.arange(12).reshape((3, 4)),
#                     index=['Ohio', 'Colorado', 'New York'],
#                     columns=['one', 'two', 'three', 'four'])
# print(data)
# data.index = data.index.str.upper()
# print(data)
# data.index = data.index.map(lambda x : x[:4].upper())
# print(data)
# print(data.rename(index=str.title, columns=str.upper))
# print(data.rename(index={'OHIO': 'INDIANA'},columns={'three': 'peekaboo'}))

# ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# bins = [18, 25, 35, 60, 100]
# print(ages)
# cats = pd.cut(ages, bins)
# print(cats)
# print(cats.codes)
# print(cats.categories)
# print(cats.value_counts())
# group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
# print(pd.cut(ages, bins, labels=group_names))

# data = np.random.rand(20)
# print(data)
# print(pd.cut(data,4,precision=2))
# data = np.random.randn(1000)
# cats = pd.qcut(data, 6)
# print(cats.value_counts())
# cats = pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
# print(cats.value_counts())

# data = pd.DataFrame(np.random.randn(1000, 4))
# print(data.describe())
# print(data[(np.abs(data) > 3).any(1)])
# data[np.abs(data) > 3] = np.sign(data) * 3
# print(data.describe())

# df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
#                    'data1': range(6)})
# print(df)
# print(pd.get_dummies(df['key']))

# val = 'a,b, guido'
# print(val.split(','))
# pieces = [x.strip() for x in val.split(',')]
# print(pieces)
# print('::'.join(pieces))
# print(val.replace(',', '::'))

# text = "foo bar\t baz \tqux"
# print(text)
# print(re.split('\s+', text))
# text = """Dave dave@google.com
# Steve steve@gmail.com
# Rob rob@gmail.com
# Ryan ryan@yahoo.com
# """
# pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
# print(text)
# regex = re.compile(pattern, flags=re.IGNORECASE)
# print(regex.findall(text))
# print(regex.sub('REDACTED', text))
# pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
# regex = re.compile(pattern, flags=re.IGNORECASE)
# m = regex.match('wesm@bright.net')
# print(m.groups())
# print(regex.findall(text))
# print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))

# pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
# data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com','Rob': 'rob@gmail.com', 'Wes': np.nan}
# data = pd.Series(data)
# print(data)
# print(data.str.contains('gmail'))
# print(data.str.findall(pattern, flags=re.IGNORECASE))

# Wrangling

# data = pd.Series(np.random.randn(9),
#                  index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
#                         [1, 2, 3, 1, 3, 1, 2, 2, 3]])
# print(data)
# print(data['b':'c'])
# print(data.loc[['b', 'd']])
# print(data.loc[:, 2])
# print(data.unstack())
# print(data.unstack().stack())

# frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
#                      index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
#                      columns=[['Ohio', 'Ohio', 'Colorado'],
#                               ['Green', 'Red', 'Green']])
# frame.index.names = ['key1', 'key2']
# frame.columns.names = ['state', 'color']
# print(frame)
# print(frame.swaplevel('key1', 'key2'))
# print(frame.sort_index(level=1))
# print(frame.swaplevel(0, 1).sort_index(level=0))
# print(frame.sum(level='key2'))
# print(frame.sum(axis = 1,level='color'))
#
# frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
#                       'c': ['one', 'one', 'one', 'two', 'two','two', 'two'],
#                       'd': [0, 1, 2, 0, 1, 2, 3]})
# print(frame)
# frame2 = frame.set_index(['c', 'd'])
# print(frame2)
# print(frame.set_index(['c','d'], drop=False))
# print(frame2.reset_index())

# df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
#                     'data1': range(7)})
# df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
#                     'data2': range(3)})
# print(df1)
# print(df2)
# print(pd.merge(df1,df2, on= 'key'))
# print(pd.merge(df1,df2, on= 'key', how='outer'))
# df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
#                     'data1': range(7)})
# df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'],
#                     'data2': range(3)})
# print(df3)
# print(df4)
# print(pd.merge(df3,df4, left_on='lkey', right_on= 'rkey'))
# df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
#                     'data1': range(6)})
# df2 = pd.DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
#                     'data2': range(5)})
# print(df1)
# print(df2)
# print(pd.merge(df1,df2, on='key', how='inner')) # Cartesian product of the rows
# left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'],
#                      'key2': ['one', 'two', 'one'],
#                      'lval': [1, 2, 3]})
# right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
#                       'key2': ['one', 'one', 'one', 'two'],
#                       'rval': [4, 5, 6, 7]})
# print(left)
# print(right)
# print(pd.merge(left,right, on=['key1','key2'], how='outer'))
# print(pd.merge(left, right, on='key1', suffixes=('_left', '_right')))

# left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
#                       'value': range(6)})
# right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
# print(left1)
# print(right1)
# print(pd.merge(left1, right1, left_on='key', right_index=True))

# lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio','Nevada', 'Nevada'],
#                       'key2': [2000, 2001, 2002, 2001, 2002],
#                       'data': np.arange(5.)})
# righth = pd.DataFrame(np.arange(12).reshape((6, 2)),
#                       index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
#                              [2001, 2000, 2000, 2000, 2001, 2002]],
#                       columns=['event1', 'event2'])
# print(lefth)
# print(righth)
# print(pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer'))

# left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
#                      index=['a', 'c', 'e'],
#                      columns=['Ohio', 'Nevada'])
# right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
#                       index=['b', 'c', 'd', 'e'],
#                       columns=['Missouri', 'Alabama'])
# print(left2)
# print(right2)
# print(pd.merge(left2,right2, left_index=True, right_index=True))
# print(left2.join(right2, how='inner'))
# another = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
#                        index=['a', 'c', 'e', 'f'],
#                        columns=['New York', 'Oregon'])
# print(left2.join([right2, another], how='inner'))

# arr = np.arange(12).reshape((3, 4))
# print(arr)
# print(np.concatenate([arr, arr], axis=1))

# s1 = pd.Series([0, 1], index=['a', 'b'])
# s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
# s3 = pd.Series([5, 6], index=['f', 'g'])
# print(s1)
# print(s2)
# print(s3)
# print(pd.concat([s1, s2, s3]))
# print(pd.concat([s1, s2, s3], axis= 1))
# s4 = pd.concat([s1, s3])
# print(s4)
# print(pd.concat([s1, s4], axis=1))
# result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
# print(result)

# df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
#                    columns=['one', 'two'])
# df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
#                    columns=['three', 'four'])
# print(df1)
# print(df2)
# print(pd.concat([df1, df2], axis=1, keys=['level1', 'level2']))

# df1 = pd.DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
# df2 = pd.DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
# print(df1)
# print(df2)
# print(pd.concat([df1,df2]))

# a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
#               index=['f', 'e', 'd', 'c', 'b', 'a'])
# b = pd.Series(np.arange(len(a), dtype=np.float64),
#               index=['f', 'e', 'd', 'c', 'b', 'a'])
# b[-1] = np.nan
# print(a)
# print(b)
# print(np.where(pd.isnull(a), b, a))

# df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan],
#                     'b': [np.nan, 2., np.nan, 6.],
#                     'c': range(2, 18, 4)})
# df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.],
#                     'b': [np.nan, 3., 4., 6., 8.]})
# print(df1)
# print(df2)
# print(df1.combine_first(df2))

# data = pd.DataFrame(np.arange(6).reshape((2, 3)),
#                     index=pd.Index(['Ohio', 'Colorado'], name='state'),
#                     columns=pd.Index(['one', 'two', 'three'],name='number'))
# print(data)
# result = data.stack()
# print(data.stack())
# print(data.stack().unstack('state'))
# df = pd.DataFrame({'left': result, 'right': result + 5},
#                   columns=pd.Index(['left', 'right'], name='side'))
# print(df)
# print(df.unstack('state'))

# plots

# data = np.arange(10)
# print(data)
# plt.plot(data)
# plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# plt.plot(np.random.randn(50).cumsum(), 'k--')
# ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
# ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
# plt.show()

# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
# for i in range(2):
#     for j in range(2):
#         axes[i, j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()

# plt.plot(np.random.randn(30).cumsum(), 'ko--')
# plt.show()

# data = np.random.randn(30).cumsum()
# plt.plot(data, 'k--', label='Default')
# plt.plot(data, 'k-', drawstyle='steps-post', label='steps-post')
# plt.legend(loc='best')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(np.random.randn(1000).cumsum(), 'k', label='one')
# ax.plot(np.random.randn(1000).cumsum(), 'k--', label='two')
# ax.plot(np.random.randn(1000).cumsum(), 'k.', label='three')
# ax.legend(loc='best')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
# circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
# pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
#                    color='g', alpha=0.5)
# ax.add_patch(rect)
# ax.add_patch(circ)
# ax.add_patch(pgon)
# plt.show()

# s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
# s.plot()
# plt.show()

# df = pd.DataFrame(np.random.randn(10, 4).cumsum(0),
#                   columns=['A', 'B', 'C', 'D'],
#                   index=np.arange(0, 100, 10))
# print(df)
# df.plot()
# plt.show()

# fig, axes = plt.subplots(2, 1)
# data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
# data.plot.bar(ax=axes[0], color='k', alpha=0.7)
# data.plot.barh(ax=axes[1], color='k', alpha=0.7)
# plt.show()

# df = pd.DataFrame(np.random.rand(6, 4),
#                   index=['one', 'two', 'three', 'four', 'five', 'six'],
#                   columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
# print(df)
# df.plot.bar()
# plt.show()
# df.plot.barh(stacked=True, alpha=0.5)
# plt.show()

# tips = pd.read_csv('D:\\tensor\ WesMcKinney\\tips.csv')
# tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
# sns.barplot(x='tip_pct', y='day', data=tips, orient='h')
# plt.show()
# sns.barplot(x='tip_pct', y='day', hue='time', data=tips, orient='h')
# sns.set(style="whitegrid")
# plt.show()
# tips['tip_pct'].plot.hist(bins=50)
# tips['tip_pct'].plot.density()
# plt.show()
# sns.catplot(x='day', y='tip_pct', hue='time', col='smoker',
#                kind='bar', data=tips[tips.tip_pct < 1])
# sns.catplot(x='day', y='tip_pct', row='time', col='smoker',
#                kind='bar', data=tips[tips.tip_pct < 1])
# plt.show()
# sns.catplot(x='tip_pct', y='day', kind='box',
#                data=tips[tips.tip_pct < 0.5])
# plt.show()

# comp1 = np.random.normal(0, 1, size=200)
# comp2 = np.random.normal(10, 2, size=200)
# values = pd.Series(np.concatenate([comp1, comp2]))
# print(values)
# sns.distplot(values, bins=100, color='k')
# plt.show()

# macro = pd.read_csv('D:\\tensor\ WesMcKinney\macrodata.csv')
# data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
# trans_data = np.log(data).diff().dropna()
# print(trans_data[-5:])
# sns.regplot('m1', 'unemp', data=trans_data)
# plt.title('Changes in log %s versus log %s' % ('m1', 'unemp'))
# plt.show()
# sns.pairplot(trans_data, diag_kind='kde', plot_kws={'alpha': 0.2})
# plt.show()

# aggregation

# df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
#                    'key2' : ['one', 'two', 'one', 'two', 'one'],
#                    'data1' : np.random.randn(5),
#                    'data2' : np.random.randn(5)})
# print(df)
# print(df['data1'].groupby(df['key1']).mean())
# print(df['data1'].groupby([df['key1'], df['key2']]).mean().unstack())
# print(df.groupby('key1').mean())
# print(df.groupby(['key1', 'key2']).mean())
# print(df.groupby(['key1', 'key2']).size())
# for name, group in df.groupby('key1'):
#     print(name)
#     print(group)
# for (k1,k2), group in df.groupby(['key1','key2']):
#     print((k1,k2))
#     print(group)
# print(df.groupby('key1')['data1'].quantile(.9))
# def peak_to_peak(arr):
#     return arr.max() - arr.min()
# print(df.groupby('key1')['data1'].agg(peak_to_peak))
# print(df.groupby('key1')['data1'].describe())

# people = pd.DataFrame(np.random.randn(5, 5),
#                       columns=['a', 'b', 'c', 'd', 'e'],
#                       index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
# people.iloc[2:3, [1, 2]] = np.nan # Add a few NA values
# print(people)
# mapping = {'a': 'red', 'b': 'red', 'c': 'blue','d': 'blue', 'e': 'red', 'f' : 'orange'}
# print(people.groupby(mapping,axis=1).sum())
# print(people.groupby(len).sum())
# key_list = ['one', 'one', 'one', 'two', 'two']
# print(people.groupby([len, key_list]).min())

# columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
#                                      [1, 3, 5, 1, 3]],
#                                     names=['cty', 'tenor'])
# hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
# print(hier_df)
# print(hier_df.groupby(level='cty', axis=1).count())

# tips = pd.read_csv('D:\\tensor\ WesMcKinney\\tips.csv')
# tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
# print(tips.head())
# grouped = tips.groupby(['day', 'smoker'])
# grouped_pct = grouped['tip_pct']
# print(grouped_pct.agg('mean'))
# print(grouped_pct.agg(['mean', 'std', peak_to_peak]))
# print(grouped_pct.agg([('foo', 'mean'), ('bar', np.std)]))
# functions = ['count', 'mean', 'max']
# result = grouped['tip_pct', 'total_bill'].agg(functions)
# print(result)
# print(grouped.agg({'tip_pct' : ['min', 'max', 'mean', 'std'],'size' : 'sum'}))
# print(tips.groupby(['day', 'smoker'], as_index=False).mean())

# def top(df, n=5, column='tip_pct'):
#     return df.sort_values(by=column)[-n:]
# print(top(tips, n=6))
# print(tips.groupby('smoker').apply(top))
# print(tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill'))
# print(tips.groupby('smoker', group_keys=False).apply(top))

# frame = pd.DataFrame({'data1': np.random.randn(1000),'data2': np.random.randn(1000)})
# print(frame.head())
# quartiles = pd.cut(frame.data1, 4)
# print(quartiles.head())
# def get_stats(group):
#     return {'min': group.min(), 'max': group.max(),
#             'count': group.count(), 'mean': group.mean()}
# grouped = frame.data2.groupby(quartiles)
# print(grouped.apply(get_stats).unstack())
# print(frame.data2.groupby(pd.qcut(frame.data1,10)).apply(get_stats).unstack())

# s = pd.Series(np.random.randn(6))
# s[::2] = np.nan
# print(s)
# print(s.fillna(s.mean()))

# states = ['Ohio', 'New York', 'Vermont', 'Florida','Oregon', 'Nevada', 'California', 'Idaho']
# group_key = ['East'] * 4 + ['West'] * 4
# data = pd.Series(np.random.randn(8), index=states)
# data[['Vermont', 'Nevada', 'Idaho']] = np.nan
# print(data)
# print(data.groupby(group_key).mean())
# fill_mean = lambda g: g.fillna(g.mean())
# print(data.groupby(group_key).apply(fill_mean))
# fill_values = {'East': 0.5, 'West': -1}
# fill_func = lambda g: g.fillna(fill_values[g.name])
# print(data.groupby(group_key).apply(fill_func))

# Hearts, Spades, Clubs, Diamonds
# suits = ['H', 'S', 'C', 'D']
# card_val = (list(range(1, 11)) + [10] * 3) * 4
# base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
# cards = []
# for suit in ['H', 'S', 'C', 'D']:
#     cards.extend(str(num) + suit for num in base_names)
# deck = pd.Series(card_val, index=cards)
# print(deck)
# def draw(deck, n=5):
#     return deck.sample(n)
# print(draw(deck))
# get_suit = lambda card: card[-1]
# print(deck.groupby(get_suit).apply(draw, n=2))

# df = pd.DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
#                    'data': np.random.randn(8),
#                    'weights': np.random.rand(8)})
# print(df)
# get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
# print(df.groupby('category').apply(get_wavg))

# tips = pd.read_csv('D:\\tensor\ WesMcKinney\\tips.csv')
# tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
# print(tips.head())
# print(tips.pivot_table(index=['day', 'smoker']))
# print(tips.pivot_table(['tip', 'total_bill'], index=['time', 'day'], aggfunc=sum,
#                        columns='smoker', margins=True, fill_value=0))
# print(pd.crosstab([tips.time, tips.day], tips.smoker, margins=True))

# time series

# now = datetime.now()
# print(now)
# print(now.year, now.day, now.month)
# delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
# print(delta)
# print(str(now))
# print(now.strftime('%Y-%m-%d'))

# datestrs = ['7/6/2011', '8/6/2011']
# print([datetime.strptime(x, '%m/%d/%Y') for x in datestrs])
# print(parse('2011-01-03'))
# print(parse('Jan 31, 1997 10:45 PM'))
# print(parse('6/20/2011', dayfirst=True))

# datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00',np.nan]
# print(pd.to_datetime(datestrs))

# dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
#          datetime(2011, 1, 7), datetime(2011, 1, 8),
#          datetime(2011, 1, 10), datetime(2011, 1, 12)]
# ts = pd.Series(np.random.randn(6), index=dates)
# print(ts)
# print(ts['1/10/2011'])
# print(ts['20110110'])
# print(ts[datetime(2011, 1, 7):])
# print(ts.truncate(after='1/9/2011'))

# longer_ts = pd.Series(np.random.randn(1000),
#                       index=pd.date_range('1/1/2000', periods=1000))
# print(longer_ts)
# print(longer_ts['2001'])
# print(longer_ts['2001-05'])

# dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
# long_df = pd.DataFrame(np.random.randn(100, 4),index=dates,
#                        columns=['Colorado', 'Texas', 'New York', 'Ohio'])
# print(dates)
# print(long_df)
# print(long_df.loc['5-2001'])

# print(pd.date_range('2000-01-01', '2000-12-01', freq='BM'))
# print(pd.date_range('2012-05-02 12:56:31', periods=5, normalize=True))
# print(pd.date_range('2000-01-01', '2000-01-03 23:59', freq='4h'))
# print(pd.date_range('2000-01-01', periods=10, freq='1h30min'))
# print(pd.date_range('2012-01-01', '2012-09-01', freq='WOM-3FRI'))

# ts = pd.Series(np.random.randn(4),
#                index=pd.date_range('1/1/2000', periods=4, freq='M'))
# print(ts)
# print(ts.shift(2))
# print(ts.shift(2, freq='M'))
# print(ts.shift(-2))
# print(ts.shift(3, freq='D'))
# print(ts.shift(1, freq='90T'))

# ts = pd.Series(np.random.randn(20),
#                index=pd.date_range('1/15/2000', periods=20, freq='4d'))
# print(ts)
# print(ts.groupby(pd.tseries.offsets.MonthEnd().rollforward).mean())
# print(ts.resample('M').mean())

# tz = pytz.timezone('America/New_York')
# print(tz)
# rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts)
# print(ts.index.tz)
# ts_utc = ts.tz_localize('UTC')
# print(ts_utc)
# print(ts_utc.index.tz)
# ts_eastern = ts.tz_localize('America/New_York')
# print(ts_eastern)
# print(ts_eastern.tz_convert('UTC'))

# rng = pd.date_range('3/7/2012 9:30', periods=10, freq='B')
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts)
# ts1 = ts[:7].tz_localize('Europe/London')
# print(ts1)
# ts2 = ts1[2:].tz_convert('Europe/Moscow')
# print(ts2)
# result = ts1 + ts2
# print(result.index)

# rng = pd.period_range('2006', '2009', freq='A-DEC')
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts)
# print(ts.asfreq('M', how='start'))
# print(ts.asfreq('B', how='end'))

# rng = pd.period_range('2011Q3', '2012Q4', freq='Q-JAN')
# ts = pd.Series(np.arange(len(rng)), index=rng)
# print(ts)
# new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
# ts.index = new_rng.to_timestamp()
# print(ts)

# rng = pd.date_range('2000-01-01', periods=3, freq='M')
# ts = pd.Series(np.random.randn(3), index=rng)
# print(ts)
# pts = ts.to_period()
# print(pts)
# rng = pd.date_range('1/29/2000', periods=6, freq='D')
# ts2 = pd.Series(np.random.randn(6), index=rng)
# print(ts2)
# print(ts2.to_period('M'))
# print(ts2.to_period('M').to_timestamp(how='end'))

# data = pd.read_csv('D:\\tensor\ WesMcKinney\macrodata.csv')
# print(data.head())
# index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
# data.index = index
# print(data.head())

# rng = pd.date_range('2000-01-01', periods=100, freq='D')
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts)
# print(ts.resample('M').mean())

# rng = pd.date_range('2000-01-01', periods=12, freq='T')
# ts = pd.Series(np.arange(12), index=rng)
# print(ts)
# print(ts.resample('5min').sum())
# print(ts.resample('5min', closed='right').sum())
# print(ts.resample('5min', closed='right', label='right').sum())
# print(ts.resample('5min', closed='right', label='right', loffset='-1s').sum())
# print(ts.resample('5min').ohlc())

# frame = pd.DataFrame(np.random.randn(2, 4), index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
#                      columns=['Colorado', 'Texas', 'New York', 'Ohio'])
# print(frame)
# df_daily = frame.resample('D').asfreq()
# print(df_daily)
# print(frame.resample('D').ffill())
# print(frame.resample('W-THU').ffill())

# frame = pd.DataFrame(np.random.randn(24, 4),
#                      index=pd.period_range('1-2000', '12-2001', freq='M'),
#                      columns=['Colorado', 'Texas', 'New York', 'Ohio'])
# print(frame)
# annual_frame = frame.resample('A-DEC').mean()
# print(annual_frame)
# print(annual_frame.resample('Q-DEC', convention='end').ffill())

# close_px_all = pd.read_csv('D:\\tensor\ WesMcKinney\stock_px_2.csv', parse_dates=True, index_col=0)
# print(close_px_all.head())
# close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
# close_px = close_px.resample('B').ffill()
# print(close_px.head())

# close_px.AAPL.plot()
# close_px_all.AAPL.plot()
# close_px.AAPL.rolling(250).mean().plot()
# plt.show()

# appl_std250 = close_px.AAPL.rolling(250, min_periods=10).std()
# print(appl_std250.head(15))
# appl_std250.plot()
# plt.show()

# close_px.rolling(60).mean().plot(logy=True)
# plt.show()

# close_px.rolling('20D').mean().plot()
# plt.show()

# aapl_px = close_px.AAPL['2006':'2007']
# ma60 = aapl_px.rolling(30, min_periods=20).mean()
# ewma60 = aapl_px.ewm(span=30).mean()
# aapl_px.plot()
# ma60.plot(style='k--', label='Simple MA')
# ewma60.plot(style='k-', label='EW MA')
# plt.legend()
# plt.show()

# spx_px = close_px_all['SPX']
# spx_rets = spx_px.pct_change()
# returns = close_px.pct_change()
# corr = returns.AAPL.rolling(125, min_periods=100).corr(spx_rets)
# corr.plot()
# plt.show()

# corr = returns.rolling(125, min_periods=100).corr(spx_rets)
# corr.plot()
# plt.show()

# from scipy.stats import percentileofscore
# score_at_2percent = lambda x: percentileofscore(x, 0.02)
# result = returns.AAPL.rolling(250).apply(score_at_2percent)
# result.plot()
# plt.show()

# df = pd.DataFrame({'key': ['a', 'b', 'c'] * 4,
#                    'value': np.arange(12.)})
# print(df)
# g = df.groupby('key').value
# print(g.mean())
# print(g.transform(lambda x: x.mean()))
# print(g.transform('mean'))

# N = 15
# times = pd.date_range('2017-05-20 00:00', freq='1min', periods=N)
# df2 = pd.DataFrame({'time': times.repeat(3),
#                     'key': np.tile(['a', 'b', 'c'], N),
#                     'value': np.arange(N * 3.)})
# print(df2)
# time_key = pd.Grouper(freq='5min')
# resampled = (df2.set_index('time')
#              .groupby(['key', time_key])
#              .sum())
# print(resampled)

# modeling libs

import patsy

# data = pd.DataFrame({'x0': [1, 2, 3, 4, 5],
#                      'x1': [0.01, -0.01, 0.25, -4.1, 0.],
#                      'y': [-1.5, 0., 3.6, 1.3, -2.]})
# print(data)
# y, X = patsy.dmatrices('y ~ x0 + x1', data)
# print(y)
# print(X)
# coef, resid, _, _ = np.linalg.lstsq(X, y, rcond=None)
# print(coef)
# print(resid)
# coef = pd.Series(coef.squeeze(), index=X.design_info.column_names)
# print(coef)

# y, X = patsy.dmatrices('y ~ x0 + np.log(np.abs(x1) + 1)', data)
# print(X)

# y, X = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data)
# print(X)
# new_data = pd.DataFrame({
#     'x0': [6, 7, 8, 9],
#     'x1': [3.1, -0.5, 0, 2.3],
#     'y': [1, 2, 3, 4]})
# new_X = patsy.build_design_matrices([X.design_info], new_data)
# print(new_X)

# y, X = patsy.dmatrices('y ~ I(x0 + x1)', data)
# print(X)

# data = pd.DataFrame({'key1': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b'],
#                      'key2': [0, 1, 0, 1, 0, 1, 0, 0],
#                      'v1': [1, 2, 3, 4, 5, 6, 7, 8],
#                      'v2': [-1, 0, 2.5, -0.5, 4.0, -1.2, 0.2, -1.7]})
# print(data)
# y, X = patsy.dmatrices('v2 ~ key1 + 0', data)
# print(X)
# y, X = patsy.dmatrices('v2 ~ C(key2) + 0', data)
# print(X)
# data['key2'] = data['key2'].map({0: 'zero', 1: 'one'})
# print(data)
# y, X = patsy.dmatrices('v2 ~ key1 + key2', data)
# print(X)
# y, X = patsy.dmatrices('v2 ~ key1 + key2 + key1:key2', data)
# print(X)

import statsmodels.api as sm
import statsmodels.formula.api as smf

# def dnorm(mean, variance, size=1):
#     if isinstance(size, int):
#         size = size,
#     return mean + np.sqrt(variance) * np.random.randn(*size)
# # For reproducibility
# np.random.seed(12345)
# N = 100
# X = np.c_[dnorm(0, 0.4, size=N),
#           dnorm(0, 0.6, size=N),
#           dnorm(0, 0.2, size=N)]
# eps = dnorm(0, 0.1, size=N)
# beta = [0.1, 0.3, 0.5]
# y = np.dot(X, beta) + eps
# print(y)
# print(X)
# X_model = sm.add_constant(X)
# model = sm.OLS(y, X)
# results = model.fit()
# print(results.summary())

# data = pd.DataFrame(X, columns=['col0', 'col1', 'col2'])
# data['y'] = y
# print(data)
# results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
# print(results.summary())
# print(results.predict(data[:5]))

# init_x = 4
# import random
# values = [init_x, init_x]
# N = 1000
# b0 = 0.8
# b1 = -0.4
# noise = dnorm(0, 0.1, N)
# for i in range(N):
#     new_x = values[-1] * b0 + values[-2] * b1 + noise[i]
#     values.append(new_x)
#
# MAXLAGS = 5
# model = sm.tsa.AutoReg(values, lags=MAXLAGS)
# results = model.fit()
# print(results.summary())

from sklearn.linear_model import LogisticRegression

train = pd.read_csv('D:\\pydataBookWes\\pydata-book-2nd-edition\\pydata-book-2nd-edition\\datasets\\titanic\\train.csv')
test = pd.read_csv('D:\\pydataBookWes\\pydata-book-2nd-edition\\pydata-book-2nd-edition\\datasets\\titanic\\test.csv')
# print(train.head())
# print(train.isnull().sum())
print(test.isnull().sum())
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(train['Age'].median())
train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)
predictors = ['Pclass', 'IsFemale', 'Age']
X_train = train[predictors].values
X_test = test[predictors].values
y_train = train['Survived'].values
print(X_train[:5])
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_predict = model.predict(X_test)
# print(y_predict[:10])

from sklearn.linear_model import LogisticRegressionCV
# model_cv = LogisticRegressionCV(10)
# model_cv.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
model = LogisticRegression(C=10)
scores = cross_val_score(model, X_train, y_train, cv=4)
print(scores)




