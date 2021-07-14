import fasttext
import wordcloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier
hair_dryer = pd.read_csv('hair_dryer.tsv', sep='\t', header=0, index_col='review_id')
microwave = pd.read_csv('microwave.tsv', sep='\t', header=0, index_col='review_id')
pacifier = pd.read_csv('pacifier.tsv', sep='\t', header=0, index_col='review_id')
col_names = ["customer_id", "product_parent", "star_rating", "helpful_votes", "vine", "verified_purchase",
             "total_votes",
             "review_headline", "review_body", "review_date"]
data_names = ["hair_dryer", "microwave", "pacifier"]
data_total = [hair_dryer, microwave, pacifier]
for i, name in enumerate(data_names):
    data_total[i] = data_total[i][col_names]

for i, name in enumerate(data_names):
    print(name, "desc")
    print(f"The number of {name} data", len(name))
    print(data_total[i].describe())
    print("The number of star_rating",
          len(data_total[i]["star_rating"][(data_total[i]["star_rating"] <= 5) & (data_total[i]["star_rating"] >= 1)]))
    print("The useful number of total_votes", len(data_total[i]["total_votes"][data_total[i]["total_votes"] != 0]))
    print()


def vis_data(data, name=None):
    # 认为剔除br
    for i, row in enumerate(data["review_body"]):
        data["review_body"][i] = row.replace("br", "")
    sns.set(rc={'figure.facecolor': 'white'})
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    print("there is any null data or not:", data.isnull().any().any())
    print("adding a length column for analyzing the length of the reviews")
    data['review_body_length'] = data['review_body'].apply(len)
    # 这里会出错，所以没有考虑评论的标题
    #     data['review_headline_length'] = data['review_headline'].apply(len)
    #     data.groupby('length').describe().sample(10)
    print(data.groupby('star_rating').describe())
    ratings = data["star_rating"].value_counts()
    label_rating = ratings.index
    size_rating = ratings.values

    colors = ['pink', 'lightblue', 'aqua', 'gold', 'crimson']

    rating_piechart = go.Pie(labels=label_rating,
                             values=size_rating,
                             marker=dict(colors=colors),
                             name=f'{name}', hole=0.3)

    df = [rating_piechart]

    layout = go.Layout(title=f'Distribution of Ratings for {name}', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')

    fig = go.Figure(data=df,
                    layout=layout)

    py.iplot(fig)

    ####################################################条形图
    color = plt.cm.copper(np.linspace(0, 1, 20))
    data['product_parent'].value_counts()[:20].plot.bar(color=color, figsize=(15, 9), colormap="#ffffff")
    plt.title(f'Distribution of {name} in kinds(1-20st)', fontsize=20)
    plt.xlabel(f'{hair_dryer} kind')
    plt.ylabel('count')
    plt.show()

    #####################################################饼形
    verified_purchase = data['verified_purchase'].value_counts()

    label_verified_purchase = verified_purchase.index
    size_verified_purchase = verified_purchase.values

    colors = ['yellow', 'lightgreen']

    feedback_piechart = go.Pie(labels=label_verified_purchase,
                               values=size_verified_purchase,
                               marker=dict(colors=colors),
                               name=f'{name}', hole=0.3)

    df2 = [feedback_piechart]

    layout = go.Layout(
        title=f'Distribution of verified_purchase for {name}', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')

    fig = go.Figure(data=df2,
                    layout=layout)

    py.iplot(fig)

    # 统计图
    sns.set_style("whitegrid")
    data['review_body_length'].value_counts().plot.hist(color='skyblue', figsize=(15, 5), bins=50)
    plt.title('Distribution of Length in Reviews')
    plt.xlabel('lengths')
    plt.ylabel('count')
    plt.show()

    print("Let's Check some of the reviews according to thier lengths")
    print(data[data['review_body_length'] == 1]['review_body'].iloc[0])
    print(data[data['review_body_length'] == 21]['review_body'].iloc[0])
    print(data[data['review_body_length'] == 50]['review_body'].iloc[0])
    print(data[data['review_body_length'] == 150]['review_body'].iloc[0])

    ### 绘制一个奇怪的图性
    plt.rcParams['figure.figsize'] = (15, 9)
    plt.style.use('fivethirtyeight')

    namess = data['product_parent'].value_counts()[:20].index.tolist()
    p_data = data.loc[data["product_parent"].isin(namess)]
    sns.set(rc={'figure.facecolor': 'white'})
    sns.set_style("whitegrid")
    sns.boxenplot(p_data['product_parent'], p_data['star_rating'], palette='spring')
    plt.title("product_kind vs star_ratings")
    plt.xticks(rotation=90)
    plt.show()

    # 绘制散点图
    plt.rcParams['figure.figsize'] = (15, 9)
    plt.style.use('fivethirtyeight')
    sns.set_style("whitegrid")
    sns.swarmplot(p_data['product_parent'], data['review_body_length'], palette='deep')
    plt.title("product kind vs Length of Ratings")
    plt.xticks(rotation=90)
    plt.show()

    ## 绘制奇怪图二
    import warnings
    warnings.filterwarnings('ignore')

    plt.rcParams['figure.figsize'] = (12, 7)
    plt.style.use('fivethirtyeight')

    sns.set_style("whitegrid")
    sns.violinplot(data['verified_purchase'], data['star_rating'], palette='cool')
    plt.title("verified_purchase wise Mean Ratings")
    plt.show()

    ### 绘制箱线图
    warnings.filterwarnings('ignore')

    plt.rcParams['figure.figsize'] = (12, 7)
    plt.style.use('fivethirtyeight')
    sns.set_style("whitegrid")
    sns.boxplot(data['star_rating'], data['review_body_length'], palette='Blues')
    plt.title("review_body_length vs star_rating")
    plt.show()

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    # 进行词频的统计
    cv = CountVectorizer(stop_words='english')
    transformer = TfidfTransformer()
    words = transformer.fit_transform(cv.fit_transform(data.review_body))
    sum_words = words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

    plt.style.use('fivethirtyeight')
    color = plt.cm.ocean(np.linspace(0, 1, 20))
    sns.set_style("whitegrid")
    frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color=color)
    plt.title("Most Frequently Occuring Words - Top 20")
    plt.show()

    ## 词云图
    from wordcloud import WordCloud

    wordcloud = WordCloud(background_color='lightcyan', width=2000, height=2000).generate_from_frequencies(
        dict(words_freq))

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.title("Vocabulary from Reviews", fontsize=20)
    plt.show()

    namesss = data['product_parent'].value_counts()[:5].index.tolist()
    pp_data = data.loc[data["product_parent"].isin(namesss)]
    print(frequency.head(20))
    trace = go.Scatter3d(
        x=data['review_body_length'],
        y=data['star_rating'],
        z=pp_data['product_parent'],
        mode='markers',
        name=name,
        marker=dict(
            size=10,
            color=data['star_rating'],
            colorscale='Viridis', ))
    df = [trace]
    layout = go.Layout(
        title='Length vs Frequency word vs Ratings', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=df, layout=layout)
    iplot(fig)


if __name__ == '__main__':
    # vis_data(hair_dryer, "hair_dryer")
    model = fasttext.train_supervised('train.txt', label_prefix='__label__', thread=4, epoch=400)
    pass
    print(np.random.normal(size=5))