import numpy as np
import matplotlib.pyplot as plt
import tqdm


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_popularity_score(shorts_df):
    # make popularity score
    normalize_viewcnt = (shorts_df['videoViewCount']+1)/shorts_df['subscriberCount']
    minmax_normalize_viewcnt = NormalizeData(normalize_viewcnt)
    return minmax_normalize_viewcnt

def visualize(df):
    fig, ax = plt.subplots()
    # 첫 번째 그래프 (수직선)
    line_idx = min(df[df['gradient'] < 1.0]['popularity_score'])
    # print(line_idx)
    ax.axvline(line_idx, 0, 1, color='red', linestyle='--', linewidth=1)

    df.plot.scatter(x='popularity_score', y='cdf', grid=True, ax=ax)

    # Calculate the slope of the tangent line at line_idx
    epsilon = 0.00027199999999999447 #1e-5  # Small positive value for ε
    idx_cdf = df[df['popularity_score'] == line_idx]['cdf'].values[0]

    ax.set_xlabel('Normalized Popularity Score')
    ax.set_ylabel('Cumulative Distribution Function')
    # ax.set_title(f'Combined Graph - Slope at x={line_idx}: {slope:.2f}')
    plt.show()

def measure_popularity_score(shorts_df):
    df = shorts_df[shorts_df['popularity_score']<0.2]
    df = df.sort_values('popularity_score')
    df['popularity_score'] = (df['popularity_score'] - df['popularity_score'].min()) / (df['popularity_score'].max() - df['popularity_score'].min())
    df['cdf'] = df['popularity_score'].rank(method = 'average', pct = True)
    # window 안에서 가장 큰 값에서 작은 값의 기울기
    gradients = []
    c = 0.0005 #0.00022 # 0.0005
    for i in tqdm.trange(len(df)):
        x_point = df.iloc[i]
        
        x_upper = x_point['popularity_score'] + c
        x_lower = x_point['popularity_score'] - c
        
        window = df[df['popularity_score'] < x_upper]
        window = window[window['popularity_score'] > x_lower]
        
        gradients.append((window.iloc[-1]['cdf'] - window.iloc[0]['cdf']) / 0.002)
    df['gradient'] = gradients
    
    visualize(df)

    popular_index = df[df['gradient'] < 1.0].index
    shorts_df = shorts_df[shorts_df['popularity_score']<0.2]
    shorts_df['popularity'] = 0
    shorts_df['popularity'][popular_index] = 1
    return shorts_df