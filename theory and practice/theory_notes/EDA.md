# EDA

I will be writing this series to find a systematic way for exploratory data analysis. I will try to identify the most common practices and methods. 

## Steps

1. Look at the target distribution. For the pandas series it can be done in the following way:

    ```python
    df['target_column'].value_counts()
    ```
    You might need to do some preprocessing if the data is not distributed evently. Moreover, the target value distribution affects the validation method. 

2. Look at the distribution of several categorical variables. It can be done the same command as the one mentioned above.
    ```python
    train['column_name'].value_counts()
    ```

## Build a simple model
1. Define cross-validation strategy
    ```python
        n_folds = 5
        # The distribution of the response variable is equal accross folds in statified folds.
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=17)

    ```


## Visualizations

Currently there are a lot of great libraries for data visualization in Python and I usually use several of them:

- basic pandas plotting: this is a simplified version of using matplotlib, which can be used for some fast and simple plotting;
- matplotlib: you can do everything, though it can require a lot of code. Also making interactive plots is difficult or impossible;
- seaborn: it is good when you want to plot interactions between several features;
- plotly: I used it to make interactive plots and it is great for this, but recently I switched to altair;
- altair: this is a python wrapper for vega-lite. It is possible to make almost any plot there and interactivity is great. On the other hand it could be difficult to get used to it's syntax;

Example:
- Pandas
    ```python
    ax = train_df['game_mode'].value_counts().plot(kind='bar', title='Count of games in different modes')
    ax.set_xlabel('Game modes')
    ax.set_ylabel('Counts')
    ```
- Matplotlib
    ```python
    train_modes = train_df['game_mode'].value_counts().reset_index().rename(columns={'index': 'game_mode', 'game_mode': 'count'})
    plt.bar(range(len(train_modes['game_mode'])), train_modes['count']);
    plt.xticks(range(len(train_modes['game_mode'])), train_modes['game_mode']);
    plt.xlabel('Game mode');
    plt.ylabel('Counts');
    plt.title('Counts of games in different modes');
    ```
- Seaborn
    ```python
    sns.countplot(data=train_df, x='game_mode', order=train_df['game_mode'].value_counts().index);
    plt.title('Counts of games in different modes');
    ```
