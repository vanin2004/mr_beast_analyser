import pandas as pd
from normalize import normalize_column


def sample_df():
    return pd.DataFrame({
        'timestamp': [0, 3, 5,8, 9],
        'views': [10, None,None, 30, 0],
        'likes': [1, 2, None, None , 4]
    })


def run_examples():
    df = sample_df()
    print('Input:')
    print(df)
    res = normalize_column(df, x='timestamp', y=['views', 'likes'], x0=0, step=1, xmax=10)
    print('\nResult:')
    print(res)


if __name__ == '__main__':
    run_examples()

