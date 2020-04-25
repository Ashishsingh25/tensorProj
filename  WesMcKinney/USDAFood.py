import json
import pandas as pd
import matplotlib.pyplot as plt

db = json.load(open('D:\\pydataBookWes\\pydata-book-2nd-edition\\pydata-book-2nd-edition\\datasets\\usda_food\\database.json'))
# print(len(db))
# print(db[0].keys())
# print(db[0]['nutrients'][0])
# nutrients = pd.DataFrame(db[0]['nutrients'])
# print(nutrients[:5])
nutrients = []
for rec in db:
    fnuts = pd.DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)
nutrients = pd.concat(nutrients, ignore_index=True)
nutrients = nutrients.drop_duplicates()
col_mapping = {'description' : 'nutrient', 'group' : 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
# print(nutrients.info())

info_keys = ['description', 'group', 'id', 'manufacturer']
info = pd.DataFrame(db, columns=info_keys)
# print(info.info())
# print(pd.value_counts(info.group)[:10])
col_mapping = {'description' : 'food', 'group' : 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
# print(info.info())
ndata = pd.merge(nutrients, info, on='id', how='outer')
print(ndata.iloc[0])

# result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
# print(result[:5])
# result['Zinc, Zn'].sort_values().plot(kind='barh')
# plt.show()

by_nutrient = ndata.groupby(['nutgroup', 'nutrient'])
get_maximum = lambda x: x.loc[x.value.idxmax()]
get_minimum = lambda x: x.loc[x.value.idxmin()]
max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]
# make the food a little smaller
max_foods.food = max_foods.food.str[:50]
print(max_foods.loc['Amino Acids']['food'])




















