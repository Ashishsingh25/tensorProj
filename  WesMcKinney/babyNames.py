import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = 'D:\\pydataBookWes\\pydata-book-2nd-edition\\pydata-book-2nd-edition\\datasets\\babynames\\yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)
# Concatenate everything into a single DataFrame
names = pd.concat(pieces, ignore_index=True)
# print(names[:5])

total_births = names.pivot_table('births', index='year', columns='sex', aggfunc=sum)
# total_births.plot(title='Total births by sex and year')
# plt.show()

def add_prop(group):
    group['prop'] = group.births / group.births.sum()
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)
# print(names.groupby(['year', 'sex']).prop.sum())

## Method 1
# def get_top1000(group):
#     return group.sort_values(by='births', ascending=False)[:1000]
# top1000 = names.groupby(['year', 'sex']).apply(get_top1000)
# # Drop the group index, not needed
# top1000.reset_index(inplace=True, drop=True)

## Method 2
pieces = []
for year, group in names.groupby(['year', 'sex']):
    pieces.append(group.sort_values(by='births', ascending=False)[:1000])
top1000 = pd.concat(pieces, ignore_index=True)

boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births', index='year', columns='name', aggfunc=sum)
# print(total_births[:5])
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
# subset.plot(subplots=True, figsize=(12, 10), grid=False, title="Number of births per year")
# plt.show()
table = top1000.pivot_table('prop', index='year', columns='sex', aggfunc=sum)
# print(table.head())
# table.plot(title='Sum of table1000.prop by year and sex', yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
# plt.show()

# df = boys[boys.year == 2010]
# prop_cumsum = df.sort_values(by='prop', ascending=False)['prop'].cumsum()
# print(prop_cumsum.values.searchsorted(0.5) + 1)
# df = boys[boys.year == 1900]
# prop_cumsum = df.sort_values(by='prop', ascending=False)['prop'].cumsum()
# print(prop_cumsum.values.searchsorted(0.5) + 1)
def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1
diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
# print(diversity[:5])
# diversity.plot(title="Number of popular names in top 50%")
# plt.show()

# The “last letter” revolution
# extract last letter from name column
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
# print(names[:5])
# print(last_letters[:5])
table = names.pivot_table('births', index=last_letters, columns=['sex', 'year'], aggfunc=sum)
# print(table[:5])
# subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
# letter_prop = subtable / subtable.sum()
# print(letter_prop[:5])
# fig, axes = plt.subplots(2, 1, figsize=(10, 8))
# letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
# letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)
# plt.show()
letter_prop = table / table.sum()
dny_ts = letter_prop.loc[['d', 'n', 'y'], 'M'].T
# print(dny_ts.head())
# dny_ts.plot()
# plt.show()

# Boy names that became girl names (and vice versa)
all_names = pd.Series(top1000.name.unique())
lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
filtered = top1000[top1000.name.isin(lesley_like)]
# print(filtered.groupby('name').births.sum())
table = filtered.pivot_table('births', index='year', columns='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
table.plot(style={'M': 'k-', 'F': 'k--'})
plt.show()



























