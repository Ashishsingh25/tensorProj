import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


path = 'D:\\pydataBookWes\\pydata-book-2nd-edition\\pydata-book-2nd-edition\\datasets\\bitly_usagov\\example.txt'
records = [json.loads(line) for line in open(path)]
# print(records[0])

# find the most often-occurring time zones

time_zones = [rec['tz'] for rec in records if 'tz' in rec]
# print(time_zones[:10])

## Method 1 using standard python

# def get_counts(sequence):
#     counts = {}
#     for x in sequence:
#         if x in counts:
#             counts[x] += 1
#         else:
#             counts[x] = 1
#     return counts
#
# from collections import defaultdict
# def get_counts2(sequence):
#     counts = defaultdict(int) # values will initialize to 0
#     for x in sequence:
#         counts[x] += 1
#     return counts
#
# counts = get_counts(time_zones)
# # print(counts)
#
# def top_counts(count_dict, n=10):
#     value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
#     # print(type(value_key_pairs))
#     value_key_pairs.sort()
#     return value_key_pairs[-n:]
#
# print(top_counts(counts))

## Method 2 using Counter

# from collections import Counter
# counts = Counter(time_zones)
# print(counts.most_common(10))

## Method 3 using pandas

frame = pd.DataFrame(records)
print(frame.head())
# print(frame['tz'].value_counts())
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
print(tz_counts[:10])

# sns.barplot(y=tz_counts[:10].index, x=tz_counts[:10].values)
# plt.show()

# Decompose the top time zones into Windows and nonWindows users
cframe = frame[frame.a.notnull()]
# print(cframe[:5])
cframe['os'] = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
# print(cframe['os'].head())
by_tz_os = cframe.groupby(['tz', 'os'])
agg_counts = by_tz_os.size().unstack().fillna(0)
indexer = agg_counts.sum(1).argsort()
count_subset = agg_counts.take(indexer[-10:])
count_subset = count_subset.stack()
count_subset.name = 'total'
count_subset = count_subset.reset_index()
print(count_subset)
# sns.barplot(x='total', y='tz', hue='os', data=count_subset)
# plt.show()
def norm_total(group):
    group['normed_total'] = group.total / group.total.sum()
    return group
results = count_subset.groupby('tz').apply(norm_total)
sns.barplot(x='normed_total', y='tz', hue='os', data=results)
plt.show()










