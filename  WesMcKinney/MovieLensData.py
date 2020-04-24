import pandas as pd

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('D:\\pydataBookWes\\pydata-book-2nd-edition\\pydata-book-2nd-edition\\datasets\\movielens\\users.dat', sep='::', header=None, names=unames, engine='python')

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('D:\\pydataBookWes\\pydata-book-2nd-edition\\pydata-book-2nd-edition\\datasets\\movielens\\ratings.dat', sep='::', header=None, names=rnames, engine='python')

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('D:\\pydataBookWes\\pydata-book-2nd-edition\\pydata-book-2nd-edition\\datasets\\movielens\\movies.dat', sep='::', header=None, names=mnames, engine='python')

# print(users[:5])
# print(ratings[:5])
# print(ratings.shape)
# print(movies[:5])

data = pd.merge(pd.merge(ratings, users), movies)
# print(data.loc[0])
mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
# print(mean_ratings[:5])
ratings_by_title = data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title >= 250]
mean_ratings = mean_ratings.loc[active_titles]
# print(mean_ratings[:5])
# print(mean_ratings.sort_values(by='F', ascending=False).head())
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
print(sorted_by_diff[:5])
# Reverse order of rows, take first 10 rows
print(sorted_by_diff[::-1][:10])

# Standard deviation of rating grouped by title
rating_std_by_title = data.groupby('title')['rating'].std()
# Filter down to active_titles
rating_std_by_title = rating_std_by_title.loc[active_titles]
# Order Series by value in descending order
print(rating_std_by_title.sort_values(ascending=False)[:10])



















