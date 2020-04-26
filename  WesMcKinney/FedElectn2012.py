import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fec = pd.read_csv('D:\\pydataBookWes\\pydata-book-2nd-edition\\pydata-book-2nd-edition\\datasets\\fec\\P00000001-ALL.csv', low_memory=False)
# print(fec.iloc[0])
# print(fec.info())
unique_cands = fec.cand_nm.unique()
# print(unique_cands)
parties = {'Bachmann, Michelle': 'Republican', 'Cain, Herman': 'Republican','Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican', 'Johnson, Gary Earl': 'Republican', 'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat', 'Paul, Ron': 'Republican', 'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican', "Roemer, Charles E. 'Buddy' III": 'Republican', 'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}
# Add it as a column
fec['party'] = fec.cand_nm.map(parties)
# print(fec['party'].value_counts())
fec = fec[fec.contb_receipt_amt > 0]
fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]
# print(fec.contbr_occupation.value_counts()[:10])

occ_mapping = {
 'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
 'INFORMATION REQUESTED' : 'NOT PROVIDED',
 'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
 'C.E.O.': 'CEO'
}
# If no mapping provided, return x
f = lambda x: occ_mapping.get(x, x)
fec.contbr_occupation = fec.contbr_occupation.map(f)

emp_mapping = {
 'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
 'INFORMATION REQUESTED' : 'NOT PROVIDED',
 'SELF' : 'SELF-EMPLOYED',
 'SELF EMPLOYED' : 'SELF-EMPLOYED',
}
# If no mapping provided, return x
f = lambda x: emp_mapping.get(x, x)
fec.contbr_employer = fec.contbr_employer.map(f)

# by_occupation = fec.pivot_table('contb_receipt_amt', index='contbr_occupation', columns='party', aggfunc='sum')
# print(by_occupation.head())
# over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
# print(over_2mm.head())
# over_2mm.plot(kind='barh')
# plt.show()

# def get_top_amounts(group, key, n=5):
#     totals = group.groupby(key)['contb_receipt_amt'].sum()
#     return totals.nlargest(n)
# grouped = fec_mrbo.groupby('cand_nm')
# print(grouped.apply(get_top_amounts, 'contbr_occupation', n=7))
# print(grouped.apply(get_top_amounts, 'contbr_employer', n=10))

# Bucketing Donation Amounts
bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
grouped = fec_mrbo.groupby(['cand_nm', labels])
# print(grouped.size().unstack(0))

bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
# print(bucket_sums)
normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
# print(normed_sums)
# normed_sums[:-2].plot(kind='barh')
# plt.show()

# Donation Statistics by State

grouped = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
# print(totals)
totals = totals[totals.sum(1) > 100000]
# print(totals[:10])
percent = totals.div(totals.sum(1), axis=0)
print(percent[:10])





