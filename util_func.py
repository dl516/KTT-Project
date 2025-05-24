import glob
import pandas as pd
import numpy as np
import itertools
import time
import statsmodels.api as sm
from functools import reduce
from scipy.stats import zscore


def collect_and_cache_data():
    """
    Load the data and save it in pickle format for later use.
    Feature date is saved as daily file to enhance reading speed.
    """

    # Read raw reference data and saved as one pickle file for easier access later
    ref1 = pd.read_csv('data/security_reference_data_w_ret1d_1.csv')
    ref2 = pd.read_csv('data/security_reference_data_w_ret1d_2.csv')
    ref = pd.concat([ref1, ref2], ignore_index=True)
    ref_df = format_data(ref)
    ref_df.to_pickle('data/ref_df.pkl.gz')

    print("\n=================================================="
          "\n finished collecting reference data"
          "\n==================================================")

    # Read risk factors and saved as one pickle file for easier access later
    risk1 = pd.read_csv('data/risk_factors_1.csv')
    risk2 = pd.read_csv('data/risk_factors_2.csv')
    risk = pd.concat([risk1, risk2], ignore_index=True)
    risk_df = format_data(risk)
    risk_df.to_pickle('data/risk_df.pkl.gz')

    print("\n=================================================="
          "\n finished collecting risk factors"
          "\n==================================================")

    # Read raw feature data and saved as one pickle file for easier access later
    file_paths = glob.glob('data/data_set/*.csv')
    ls = []

    for f in file_paths:
        df = format_data(pd.read_csv(f))
        # To make sure, for each given date, there is no duplicated feature value for each security
        res = df.groupby(['data_date', 'security_id'])[df.columns[2]].mean().reset_index()
        ls.append(res)

    merged_feature = reduce(lambda left, right: pd.merge(left, right, on=['data_date', 'security_id'], how='outer'), ls)
    merged_feature.to_pickle('data/feature_raw.pkl.gz')

    print("\n=================================================="
          "\n finished collecting raw feature data"
          "\n==================================================")

    # Save raw feature as daily files to enhance the speed of processing feature data due to personal laptop's speed limitation
    filtered_ls = ref_df[ref_df['in_trading_universe'] == 'Y'][['data_date', 'security_id', 'group_id']]
    date_ls = merged_feature['data_date'].drop_duplicates().to_list()

    for i in date_ls:
        file = merged_feature[(merged_feature['data_date'] == i)]
        file_name = 'data/raw_feature/' + str(i)[:10] + '.pkl.gz'
        file.to_pickle(file_name)

        #Save a filtered daily data (only keep the security in the trading universe on corresponding trading date
        filtered_index = filtered_ls[filtered_ls['data_date']==i][['security_id','group_id']]
        filtered_file = filtered_index.merge(file, on=['security_id'], how='inner')
        filtered_file['group_level_1'] = filtered_file['group_id'].str[:2]
        filtered_file_name = 'data/filtered_feature/' + str(i)[:10] + '.pkl.gz'
        filtered_file.to_pickle(filtered_file_name)

        print(f"feature of data date [{i.date()}] is cached")

    print("\n=================================================="
          "\n finished collecting and caching all data"
          "\n==================================================")


def format_data(df):
    """
    Format data entries into consistent format
    """
    df = df.sort_values(['data_date', 'security_id']).reset_index(drop=True)
    col_ls = ['data_date', 'security_id', 'group_id', 'in_trading_universe']

    for col in col_ls:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df['data_date'] = pd.to_datetime(df['data_date'], format='%Y%m%d')

    return df


def find_dates(orig_date, date_map, shift, n):
    """
    Given a start date, shift the start date, and look for n days in the past within a fixed date map
    Args:
        orig_date: start date
        date_map: a dataframe has the full data list
        shift: number of days to shift start date
        n: number of days to look for after shift
    Returns:
        if n=0: returns a date or None
        else: returns a list of dates or empty list
    """
    index_orig = date_map[date_map['data_date'] == orig_date].index[0]
    index_new = index_orig + shift

    if n > 0:  # when n > 0, always returns a list
        if index_new <= 0:
            return []

        else:
            return date_map.loc[max(0, index_new - n + 1):index_new, 'data_date'].tolist()

    elif index_new in date_map.index:
        return date_map.loc[index_new][0]  # one date

    return None


def group_z(group):
    """
    Calculate ranked zscore for each group
    """
    ranks_g = group.rank(method='average')
    z_scores_g = ranks_g.apply(zscore, nan_policy='omit')

    return z_scores_g


def build_basic_fea(trade_date, date_map, func, f_col_group):
    """
    Build basic (includes the industry-neutral) features based on most recent date's raw feature value
    Args:
        trade_date: feature is computed for its current trade date
        date_map: a dataframe has the full data list
        func: customized function to calculate ranked zscore by group
        f_col_group: specify the group of features to be computed
    Returns:
        A dataframe with computed features as columns
    """
    st = time.time()
    fea_date = find_dates(trade_date, date_map, -1, 0)

    if fea_date is None:
        res = pd.DataFrame()

    else:
        fea_file = 'data/filtered_feature/' + str(fea_date)[:10] + '.pkl.gz'
        fea_df = pd.read_pickle(fea_file)
        df = fea_df.set_index('security_id').copy()[f_col_group + ['group_level_1']]

        #compute ranked zcore universally
        ranks = df[f_col_group].rank(method='average')
        z_scores = ranks.apply(zscore, nan_policy='omit')
        basic_fea = z_scores / z_scores.abs().sum()
        basic_fea.columns = [f'{col}_basic' for col in basic_fea.columns]

        # compute ranked zcore by group level 1 (industry neutral)
        group_z_df = df[['group_level_1'] + f_col_group].groupby('group_level_1', group_keys=False).apply(func)
        basic_fea_g = group_z_df / group_z_df.abs().sum()
        res = pd.concat([basic_fea, basic_fea_g.add_suffix('_basic_g1')], axis=1)

    et = time.time()
    print(f"basic feature on trade date [{trade_date.date()}] takes [{(et - st):.1f}] seconds")

    return res


def read_feature_file(file_dates, cols, need_to_filter=False, sec_ls=None, tag=None):
    """
    Read feature files based on given dates, security universe and output needed feature columns
    Args:
        file_dates: feature file needed on this list of dates
        cols: specify the feature columns in output
        need_to_filter: specify whether to read from filtered feature data or not
        sec_ls: specify the security universe
        tag: used to categorize the data input for later calculation, e.g. Q_0 data, or Q_lag1 data
    Returns:
        A dataframe combines all needed feature data for given dates
    """
    res = pd.DataFrame()

    if need_to_filter:
        cols_idx = ['security_id', 'data_date']

    else:
        cols_idx = ['group_level_1', 'security_id', 'data_date']

    if len(file_dates) > 0:
        ls = []

        for i in file_dates:
            fea_file = 'data/filtered_feature/' + str(i)[:10] + '.pkl.gz'
            df = pd.read_pickle(fea_file)

            if need_to_filter:
                df_filtered = df[df['security_id'].isin(sec_ls)][cols_idx + cols].copy()

            else:
                df_filtered = df[cols_idx + cols].copy()

            ls.append(df_filtered)

        df_all = pd.concat(ls)
        res = df_all.sort_values(cols_idx).set_index(cols_idx)

        if tag is not None:
            res['tag'] = tag

    return res


def read_feature_file_q(trade_date, date_map, sec_ls, cols):
    """
    For given trade date, read current and past 4 quarter feature data
    Args:
        trade_date: the current trade date to decide the current quarter Q0
        date_map: list of all trading dates
        sec_ls: specify the security universe
        cols: specify the feature columns needed in output
    Returns:
        A dataframe combines all needed feature data for given dates:
        One security could have 5 values [Q0, Q_pre1, Q_pre2, Q_pre3, Q_pre4] if not missing
    """
    cur_date = find_dates(trade_date, date_map, -1, 0)
    res = pd.DataFrame()
    cols_idx = ['security_id', 'data_date']

    if cur_date is not None:
        cur_f = 'data/raw_feature/' + str(cur_date)[:10] + '.pkl.gz'
        cur_f = pd.read_pickle(cur_f)
        df_filtered = cur_f[cur_f['security_id'].isin(sec_ls)][cols_idx + cols].copy().set_index(cols_idx)
        df_q0 = df_filtered.dropna(how='all').reset_index().set_index('security_id')
        df_q0['tag'] = 'Q_0'  # get the data at current quarter, next to find previous 4Q
        sec_cvg = df_q0.index.tolist()
        df_q0 = df_q0.set_index('data_date', append=True)

        res_list = [df_q0]

        for i in range(1, 5):
            q_pre_dates = find_dates(trade_date, date_map, 15+(-60)*i-1, 35)

            if len(q_pre_dates) > 0:  # check if dates are available
                df_q = read_feature_file(file_dates=q_pre_dates, cols=cols, need_to_filter=True, sec_ls=sec_cvg, tag='Q_pre' + str(i))
                df_q = df_q.dropna(how='all', subset=cols).sort_index(level=cols_idx).reset_index(drop=False)
                df_q = df_q.set_index(cols_idx)
                res_list.append(df_q)

        res = pd.concat(res_list, axis=0).sort_index()

    return res


def build_dif_fea_type2(trade_date, date_map, func, f_col_group):
    """
    Build change (includes the industry-neutral) features on given arguments on data selection
    change features are computed as the difference for average values of two specified periods
    shift_n specify the period used to compute the change
    avg_n specify how many days used to calculate the average
        trade_date: feature is computer for its current trade date
        date_map: a dataframe has the full data list
        func: customized function to calculate ranked zscore by group
        f_col_group: specify the group of features to be computed (type 2 feature here)
    Returns:
        A dataframe with computed features as columns
    """
    st = time.time()
    res = pd.DataFrame()
    cur_pre_dates = {}
    file_dates = []

    for (shift_n, avg_n) in list(itertools.product([21, 63, 126], [5, 21])):
        cur_dates = find_dates(trade_date, date_map, -1, avg_n)
        pre_dates = find_dates(trade_date, date_map, (-1)*shift_n-1, avg_n)
        cur_pre_dates[(shift_n, avg_n)] = (cur_dates, pre_dates)
        file_dates += list(np.unique(cur_dates + pre_dates))

    file_dates = sorted(list(np.unique(file_dates)))
    fea_df = read_feature_file(file_dates, f_col_group)

    if len(fea_df) > 0:  # calculate the difference
        for (shift_n, avg_n) in list(itertools.product([21, 63, 126], [5, 21])):
            cur_dates, pre_dates = cur_pre_dates[(shift_n, avg_n)]
            cur_df = fea_df.loc[fea_df.index.get_level_values(level='data_date').isin(cur_dates)]
            pre_df = fea_df.loc[fea_df.index.get_level_values(level='data_date').isin(pre_dates)]
            avg_cur = cur_df[f_col_group].groupby([pd.Grouper(level='security_id')]).mean()
            avg_pre = pre_df.groupby([pd.Grouper(level='security_id')]).mean()
            diff_df = (avg_cur - avg_pre) / avg_pre.abs()
            diff_df = diff_df.rename(columns={col: col + '_pre'+str(shift_n) + '_Avg' + str(avg_n) for col in diff_df.columns})

            # compute ranked zcore universally
            ranks = diff_df.rank(method='average')
            z_scores = ranks.apply(zscore, nan_policy='omit')
            df = z_scores / z_scores.abs().sum()

            # compute ranked zcore by group level 1 (industry neutral)
            df_group_sec = fea_df.copy().reset_index()[['group_level_1', 'security_id']]
            df_group_sec_unique = df_group_sec.drop_duplicates(subset='security_id')
            sec_to_group1 = df_group_sec_unique.set_index('security_id')['group_level_1']
            diff_df['group_level_1'] = diff_df.index.map(sec_to_group1)
            diff_group_z = diff_df.groupby('group_level_1', group_keys=False).apply(func)
            diff_fea_g = diff_group_z / diff_group_z.abs().sum()

            df_all = pd.concat([df, diff_fea_g.add_suffix('_g1')], axis=1)
            res = pd.concat([res, df_all], axis=1, join='outer')

    et = time.time()
    print(f"change of type 2 feature on trade date [{trade_date.date()}] takes [{(et - st):.1f}] seconds")

    return res


def normalize_long_short(x):
    """
        Normalized for long/short
    """
    long = x.clip(lower=0) / x.clip(lower=0).sum() * 0.5
    short = x.clip(upper=0) / x.clip(upper=0).abs().sum() * 0.5

    return long + short


def build_dif_fea_type1(trade_date, date_map, sec_ls, f_col_group):
    """
    Build change features on event-driven features (i.e., type 1).
    For one security, event-driven feature is available on a quarterly basis
    change features are computed using values in two specified quarter
        trade_date: the current trade date to decide the current quarter Q0
        date_map: list of all trading dates
        sec_ls: specify the security universe
        f_col_group: specify the feature type to be computed on
    Returns:
        A dataframe with computed features as columns
    """
    st = time.time()
    res = pd.DataFrame()
    cols_idx = ['security_id', 'tag']

    input_df = read_feature_file_q(trade_date, date_map, sec_ls, f_col_group)

    if len(input_df) > 0:
        work_df = input_df.reset_index(drop=False).groupby(cols_idx)[f_col_group].mean()

        def percentage_chg(q0, qx):
            return (q0 - qx) / (qx.abs() + q0.abs())

        q0_df = work_df.loc[work_df.index.get_level_values(level='tag') == 'Q_0'].droplevel(level='tag')
        q1_df = work_df.loc[work_df.index.get_level_values(level='tag') == 'Q_pre1'].droplevel(level='tag')
        q4_df = work_df.loc[work_df.index.get_level_values(level='tag') == 'Q_pre4'].droplevel(level='tag')
        q1_4_df = work_df.loc[work_df.index.get_level_values(level='tag') != 'Q_0'].copy()
        q1_4_avg_df = q1_4_df.groupby(level=['security_id']).mean()

        delta_q1 = percentage_chg(q0=q0_df, qx=q1_df)
        delta_q4 = percentage_chg(q0=q0_df, qx=q4_df)
        delta_q1_4 = percentage_chg(q0=q0_df, qx=q1_4_avg_df)

        delta_q1.columns = [f"{col}_Q0_Q1" for col in delta_q1.columns]
        delta_q4.columns = [f"{col}_Q0_Q4" for col in delta_q4.columns]
        delta_q1_4.columns = [f"{col}_Q0_AvgQ1Q4" for col in delta_q1_4.columns]

        delta_q1 = normalize_long_short(delta_q1)
        delta_q4 = normalize_long_short(delta_q4)
        delta_q1_4 = normalize_long_short(delta_q1_4)

        res = pd.concat([delta_q1, delta_q4, delta_q1_4], axis=1)

    et = time.time()
    print(f"change of type 1 feature on trade date [{trade_date.date()}] takes [{(et - st):.1f}] seconds")

    return res


def build_peer_spread(trade_date, sec_ls, ref0, ret_df):
    """
    Build peer spread (peer - self) features using security historical return data
    Find 5 securities in the same industry (group level 1) with the highest
    correlations as the security's peer group on every trade date.
    Use the peer group's average return (minus its own return) in customized window as the new feature
        trade_date:
        sec_ls: define the security trading universe for given trade date
        ref0: all securities' return on the given trade date
        ret_df: historical return data for all securities
    Returns:
        A dataframe with computed features as columns
    """
    st = time.time()
    n_peer = 5
    ref0['grp'] = ref0['group_id'].str[:2]
    grps = ref0['grp'].drop_duplicates().sort_values().to_list()
    sec_all = ref0.groupby('grp')['security_id'].apply(lambda x: x.drop_duplicates().to_list())
    sec_tu = ref0.loc[ref0['in_trading_universe'] == 'Y'].groupby('grp')['security_id'].apply(lambda x: x.drop_duplicates().to_list())
    ret_df = ret_df.sort_index().reindex(columns=ref0['security_id'].drop_duplicates().to_list())
    # list of window choices to compute the average return
    retw_list = [5, 10, 21, 63]
    res = pd.DataFrame(index=sec_ls, columns=[f'peer_spread{x}' for x in retw_list])
    res.index.name = 'security_id'

    def mark_peer(row, n):
        out = pd.Series(index=row.index, data=0)
        out.loc[row.nlargest(n).index] = 1
        return out

    for grp in grps:
        sec_all_grp = sec_all[grp].copy()
        sec_tu_grp = sec_tu[grp].copy()
        ret_corr = ret_df[sec_all_grp].corr().loc[sec_tu_grp, sec_all_grp]
        ret_corr[ret_corr == 1] = np.nan
        peer_df = ret_corr.apply(mark_peer, n=n_peer, axis=1).fillna(0)

        for retw in retw_list:
            col = f'peer_spread{retw}'
            # calculate the most recent mean in the given window
            r = ret_df[sec_all_grp].tail(retw).mean()
            mmt = peer_df.dot(r.fillna(0.0))
            cnt = peer_df.dot(r.notnull())
            mmt = mmt / cnt
            res.loc[mmt.index, col] = mmt - r.loc[mmt.index] / (n_peer**0.5)

    res = res.astype(float)
    res = res.rank(method='average')
    res = res.apply(zscore, nan_policy='omit')
    res = res / res.abs().sum()

    et = time.time()
    print(f"peer spread feature on trade date [{trade_date.date()}] takes [{(et - st):.1f}] seconds")

    return res


def get_future_ret(ref_df, all_names_univ, all_ref_dates, n):
    """
    Obtain future n days' return on given security universe and date range
        ref_df: raw reference data
        all_names_univ: all security universes including non-tradable securities
        all_ref_dates: all trading dates
        n: specify how many days of return are needed
    Returns:
        A dataframe with return time series of n days for all securities
    """
    # Create a fixed scope dataframe to store needed info: future returns, processed feature value by security_id and date
    ret_multi_index = pd.MultiIndex.from_product([all_names_univ, all_ref_dates], names=['security_id', 'data_date'])
    ret_mul = pd.DataFrame(index=ret_multi_index)

    raw_sorted = ref_df.sort_values(by=['security_id', 'data_date']).set_index(['security_id', 'data_date']).copy()
    ret_merged = pd.merge(ret_mul, raw_sorted, on=['security_id', 'data_date'], how='left')

    for i in range(1, n+1):
        ret_merged[f'ret_post_{i}'] = ret_merged.groupby(level=0)['ret1d'].shift(-1*i)

    return ret_merged.reset_index().sort_values(by=['data_date', 'security_id']).set_index(['data_date', 'security_id'])


def backtest_ret(df, fea_cols, ret_cols):
    """
    Calculate features' returns up to 15 days into the future
        df: input dataframe contains all feature and return time series
        fea_cols: all feature columns
        ret_cols: all return columns
    Returns:
        ret_dly: features' next n day daily returns
        ret_sr_matrix: A dataframe with backtest results of given features
    """
    def calc_ret(x):
        return x[fea_cols].T.dot(x[ret_cols])

    res_train_fill0 = df.fillna(0)
    res_train_ = res_train_fill0.groupby('data_date').apply(calc_ret)
    res_train = res_train_.reset_index().copy().rename(columns={'level_1': 'feature'}).set_index('feature')

    # calculate annualized sharpe ratios for each of 15 days into the future
    annualizer = 252 ** 0.5
    ret_mean = res_train[ret_cols].groupby('feature').mean()
    ret_std = res_train[ret_cols].groupby('feature').std()
    sr_matrix = annualizer * ret_mean / ret_std

    return res_train, sr_matrix


def calc_turnover(feature_df, all_ref_dates, ref_date_map):
    """
    Calculate the turnover of given features
        feature_df: input dataframe contains all constructed feature
        all_ref_dates: all trading dates
        ref_date_map: a dataframe has the list of all trading dates
    Returns:
        A dataframe with turnover computed for given features
    """
    ls = {}

    for trade_date in all_ref_dates:
        pre_date = find_dates(trade_date, ref_date_map, -1, 0)
        if pre_date not in feature_df.index:
            continue
        else:
            df_cur = feature_df.loc[trade_date].copy()
            df_pre = feature_df.loc[pre_date].copy()
            union_index = df_cur.index.union(df_pre.index)

            pre_aligned = df_pre.reindex(union_index)
            cur_aligned = df_cur.reindex(union_index)
            pre_filled = pre_aligned.fillna(0)
            cur_filled = cur_aligned.fillna(0)
            turnover = (cur_filled - pre_filled).abs().sum() / pre_aligned.abs().sum(min_count=1)
            df_turnover = turnover.to_frame(name='turnover')
            ls[trade_date] = df_turnover

    df_all = pd.concat(ls)
    df_all.index.names = ['data_date', 'feature']
    res = df_all.reset_index().sort_values(['feature', 'data_date']).set_index(['feature', 'data_date'])

    return res


def build_smo_fea(trade_date, sec_ls, fea_df, smo_cfg):
    """
    Calculate smoothed features
        fea_df: input dataframe of underlying features to be smoothed
        smo_cfg: a dictionary of smoothing window parameters
    Returns:
        A dataframe with computed features as columns
    """
    st = time.time()
    cols_smo = []

    for col_fea, (lag_start, lag_end) in smo_cfg.items():
        col_smo = f'{col_fea}_smo_{lag_start}_{lag_end}'
        mask_null = ~fea_df.index.get_level_values(level='data_date').isin(range(lag_start-1, lag_end))
        fea_df.loc[mask_null, col_fea] = np.nan
        fea_df = fea_df.rename(columns={col_fea: col_smo})
        cols_smo.append(col_smo)

    res = fea_df[cols_smo].groupby(level='security_id').mean().reindex(sec_ls)
    res = res.apply(normalize_long_short, axis=0)

    et = time.time()
    print(f"smoothed feature on trade date [{trade_date.date()}] takes [{(et - st):.1f}] seconds")

    return res


def build_strategy(trade_date, sec_ls, fea_df, fea_w):
    """
    Calculate smoothed features
        fea_df: input dataframe of underlying features to be combined
        fea_w: a dictionary of feature weights
    Returns:
        A dataframe with combined strategy
    """
    st = time.time()

    fea_list = list(fea_w.keys())
    fea_df = fea_df[fea_list].copy()
    w_df = pd.DataFrame(index=fea_df.index, data=fea_w)
    w_df[fea_df.isnull()] = np.nan
    fea_x_w = (fea_df * w_df).sum(axis=1)
    w = w_df.sum(axis=1)
    res = fea_x_w / w
    res = res.reindex(sec_ls)
    res.name = 'strategy'
    mask_null = w < 0.5
    # set missing if total weight less than 50%
    res.loc[mask_null] = np.nan
    res = normalize_long_short(res)
    res = res.to_frame()

    et = time.time()
    print(f"strategy on trade date [{trade_date.date()}] takes [{(et - st):.1f}] seconds")

    return res


def build_strategy_hedged(trade_date, sec_ls, fea_ser, rk_df):
    """
    Calculate smoothed features
        fea_ser: input series of underlying strategy to be hedged
        rk_df: input dataframe of risk exposures
    Returns:
        A dataframe with hedged strategy
    """
    st = time.time()

    # fill missing target positions and exposures with 0.0

    y = fea_ser.reindex(sec_ls).fillna(0.0).copy()
    y.name = 'strategy'
    X = rk_df.reindex(sec_ls).fillna(0.0).copy()
    X['__dollar__'] = 1.0
    ols = sm.OLS(endog=y, exog=X)
    fit = ols.fit()
    res = pd.Series(fit.resid)
    res.name = 'strategy_hg'
    res = normalize_long_short(res)
    res = res.to_frame()

    if not y.index.equals(res.index):
        raise Exception(f"residual index not aligned with input strategy")

    et = time.time()
    print(f"hedged strategy on trade date [{trade_date.date()}] takes [{(et - st):.1f}] seconds")

    return res


def calc_drawdown(ret_ser):
    """
    Calculate the strategy drawdown
    """
    ret_ser = ret_ser.sort_index()
    dd_ser = pd.Series(index=ret_ser.index, data=np.nan)
    ret_cumu = 1.0
    running_max = 1.0
    dd_summary_list = []
    in_dd = False
    dd_start = None

    for idx_tm, ret in ret_ser.items():
        ret_cumu *= (1.0 + ret)
        running_max = max(running_max, ret_cumu)
        dd = (ret_cumu - running_max) / running_max
        dd_ser.loc[idx_tm] = dd

        if (dd < 0) and (not in_dd):
            in_dd = True
            dd_start = idx_tm

        elif (dd == 0) and in_dd:
            in_dd = False
            dd_end = idx_tm
            trough_size = min(dd_ser[dd_start:dd_end])
            trough_idx = dd_ser[dd_start:dd_end].idxmin()
            this_dd_info = {
                'start': dd_start,
                'end': dd_end,
                'trough_size': trough_size,
                'trough_time': trough_idx,
            }
            dd_summary_list.append(this_dd_info)

    dd_summary = pd.DataFrame(dd_summary_list)

    return dd_ser, dd_summary


def calc_perf(input_df):
    """
    Calculate portfolio performance matrix
        input_df: input dataframe of daily return and turnover
    Returns:
        A dataframe with performance parameters such as gross return, net return, transact cost etc.
    """
    input_df = input_df.rename(columns={'ret': 'gross_ret'})
    input_df['cost'] = input_df['to'] * 1e-4
    input_df['net_ret'] = input_df['gross_ret'] - input_df['cost']
    turnover = input_df['to'].mean()
    gross_ret = input_df['gross_ret'].mean() * 252
    net_ret = input_df['net_ret'].mean() * 252
    vol = input_df['net_ret'].std() * (252 ** 0.5)
    sr = net_ret / vol
    hit = (input_df['net_ret'] > 0).sum() / input_df['net_ret'].count()
    ret_per_trade = input_df['net_ret'].mean() / input_df['to'].mean() * 1e4

    res = pd.DataFrame(index=[0],
                       columns=['gross_ret', 'net_ret', 'vol', 'sr', 'turnover_dly', 'hit', 'ret_per_trade'],
                       data=[[gross_ret, net_ret, vol, sr, turnover, hit, ret_per_trade]])

    return res


