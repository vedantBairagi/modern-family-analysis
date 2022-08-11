import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime as dt
from wordcloud import WordCloud, STOPWORDS

st.title('''
        Modern Family Analysis
    ''')

st.markdown('![Show Picture](https://m.media-amazon.com/images/M/MV5BMTg3ODU2Mzg0NV5BMl5BanBnXkFtZTgwOTU5MDE5MjE@._V1_.jpg)')

data = pd.read_csv('modern_family_info (1).csv', parse_dates=['Airdate'])
data1 = pd.read_csv('updated.csv', parse_dates=['Airdate'], index_col=0)
st.write('## Basic Inspections')
st.write('#### Raw Data')
st.dataframe(data.head(50))

st.write('#### Data Types')
st.write(dict(data.dtypes))

st.write('#### Checking for any null values')
st.write(dict(data.isnull().sum()))

st.write('#### Adding some new Features')
data['season'] = data['Season-Episode'].apply(lambda x: int(x[1:x.find('-')]))
data['episode'] = data['Season-Episode'].apply(lambda x: int(x[x.find('E')+1:]))


# seasons = list(data['season'].unique())
# baseurl = "https://www.imdb.com/title/tt1442437/episodes?season="
# all_ep_tags = []
# for season in seasons:
#     req = requests.get(f"{baseurl}{season}")
#     soup = BeautifulSoup(req.content, 'lxml')
#     a_tags = soup.find_all('a', {'itemprop':'name'})
#     hrefs = [f"https://www.imdb.com{a['href']}" for a in a_tags]
#     all_ep_tags.extend(hrefs)
# data['imdb_id'] = all_ep_tags
st.dataframe(data1[['Title', 'season', 'episode', 'imdb_id']].head(30))

st.write('## Updated Data')
st.dataframe(data1.head(50))

st.write('#### Best Season & Worst Season According to Rating Average')

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(y=[f"Season{i}" for i in range(1, 12)],x=data1.groupby('season')['Rating'].mean().apply(lambda x: round(x,2)), color='#1faaee', ax=ax)
_  = plt.xticks(rotation=90)
for value in ax.containers:
    ax.bar_label(value,)

st.pyplot(fig, )

st.write("""We can see that **Season 1** is the Best Overall Rated season while **Season 10**
is the Worst Overall Rated season

Now Let's See the Best and Worst rated episodes""")

st.write('#### Best and Worst Rated Episodes')

st.write('##### Best Rated Episode')
best = data1.loc[data1['Rating'].idxmax()]
st.markdown(f"![Pic from the episode]({best['Image Link']})")
st.markdown(f"""
* Title: {best['Title']}
* Season: {best['season']}
* Release Date: {best['Airdate']}
* IMDB Page Link: {best['imdb_id']}
""")

st.write('##### Worst Rated Episode')
worst = data1.loc[data1['Rating'].idxmin()]
st.markdown(f"![Pic from the episode]({worst['Image Link']})")
st.markdown(f"""
* Title: {worst['Title']}
* Season: {worst['season']}
* Release Date: {worst['Airdate']}
* IMDB Page Link: {worst['imdb_id']}
""")

st.write('#### Rating Distribution')
fig, ax = plt.subplots(figsize=(12, 8))
sns.histplot(data.Rating, kde=True, ax=ax)


st.pyplot(fig, )
st.write(f"""All right!!! Based on the Histogram
I have made three categories for the episodes
1. **Poor** if Rating is below 7.5
2. **Average** if Rating is between 7.5 and 8.2
3. **Best** if Rating is above 8.2""")
def categorical_rating(rating: float):
    if rating < 7.5:
        return 'Poor'
    elif rating > 8.2:
        return 'Best'
    else:
        return 'Average'
data1['remark'] = data1.Rating.apply(categorical_rating)
st.write('**Count of Categories**')
st.write(data1.remark.value_counts())

year_wise = data1.groupby(data1.Airdate.dt.year)['Title'].count()
month_wise = data1.groupby(data1.Airdate.dt.month)['Title'].count()
month_wise = pd.DataFrame(month_wise)

month_dict = {1: 'Jan', 2: 'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov',
             12: 'Dec'}
month = lambda x: month_dict.get(x)
month_wise.index = list(map(month, list(month_wise.index)))
st.write('#### No of Episodes in Each Year')
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x=year_wise.index, y=year_wise, ax=ax)
plt.xlabel('Year')
plt.ylabel('No of Episodes Aired')
for value in ax.containers:
    ax.bar_label(value, )
st.pyplot(fig, )

st.write('#### No of Episodes in Each Month')
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x=month_wise.index, y=month_wise.Title, ax=ax)
plt.xlabel('Months')
plt.ylabel('No of Episodes Aired')

for value in ax.containers:
    ax.bar_label(value, )
st.pyplot(fig, )
st.write('''We can see from the Months plot that there are no episodes aired 
in June, July and August''')

st.write('#### WordCloud of Descriptions')
st.markdown("![Cloud](./cloud.png)")


def episode(season=1, episode=1):
    season = data1[data1.season == season]
    episode = season[season.episode == episode]

    st.markdown(f"![Pic from the episode]({episode.iloc[0, 6]})")
    st.markdown(f"""
    * Title: {episode.iloc[0, 0]}
    * Rating: {episode.iloc[0, 3]}
    * Description: {episode.iloc[0, 5]}
    * IMDB Page Link: {episode.iloc[0, 9]}
    * SE: {episode.iloc[0, 2]}
    """)
st.write('#### Check Episode details by Season')
seas = st.selectbox('Select Season', [num for num in range(1, 12)])
eps = st.selectbox('Select episode', list(data1[data1.season == seas].episode))
if st.button('Submit'):
    episode(seas, eps)








