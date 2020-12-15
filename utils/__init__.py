import pandas as pd

df = pd.read_csv('data/dow_30_2009_2020.csv')

rebalance_window = 63
validation_window = 63
unique_trade_date = df[(df.datadate > 20151001) & (
    df.datadate <= 20200707)].datadate.unique()

df_trade_date = pd.DataFrame({'datadate': unique_trade_date})


def get_firstrun_account_value():
    model_name = "ensemble"
    df_account_value = pd.DataFrame()
    for i in range(rebalance_window+validation_window, len(unique_trade_date)+1, rebalance_window):
        temp = pd.read_csv(
            'results/firstRun/account_value_trade_{}_{}.csv'.format(model_name, i))
        df_account_value = df_account_value.append(temp, ignore_index=True)
    df_account_value = pd.DataFrame({'account_value': df_account_value['0']})
    sharpe = (252**0.5)*df_account_value.account_value.pct_change(1).mean() / \
        df_account_value.account_value.pct_change(1).std()
    print(sharpe)
    df_account_value = df_account_value.join(
        df_trade_date[63:].reset_index(drop=True))
    return df_account_value


def get_account_value(model_name):
    # file_name = 'results/account_value_trade_{}_{}.csv'.format(model_name, 0)
    # df_account_value = pd.read_csv(file_name)
    df_account_value = pd.DataFrame()
    for i in range(rebalance_window+validation_window, len(unique_trade_date)+1, rebalance_window):
        temp = pd.read_csv(
            'results/account_value_trade_{}_{}.csv'.format(model_name, i))
        df_account_value = df_account_value.append(temp, ignore_index=True)
    df_account_value = pd.DataFrame({'account_value': df_account_value['0']})
    sharpe = (252**0.5)*df_account_value.account_value.pct_change(1).mean() / \
        df_account_value.account_value.pct_change(1).std()
    print(sharpe)
    df_account_value = df_account_value.join(
        df_trade_date[63:].reset_index(drop=True))
    return df_account_value


def get_daily_return(df):
    df['daily_return'] = df.account_value.pct_change(1)
    # df=df.dropna()
    print('Sharpe: ', (252**0.5) *
          df['daily_return'].mean() / df['daily_return'].std())
    return df


def backtest_strat(df):
    strategy_ret = df.copy()
    strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
    strategy_ret.set_index('Date', drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize('UTC')
    del strategy_ret['Date']
    ts = pd.Series(strategy_ret['daily_return'].values,
                   index=strategy_ret.index)
    return ts
