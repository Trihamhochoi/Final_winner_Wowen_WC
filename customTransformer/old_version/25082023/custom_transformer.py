# SET UP BASE ESTIMATOR
import sklearn.compose as sc
import sklearn.preprocessing as sp
from sklearn.base import TransformerMixin, BaseEstimator
from imblearn.pipeline import Pipeline
# Cleansing data
from data_cleansing import Wrangling_data
from database import SQLiteDBManager

# TRAIN TEST DATASET
from sklearn.model_selection import train_test_split
# HANDLE DATAFRAME
import pandas as pd
import numpy as np
import os
from datetime import date as dt


# CREATE CLASS FOR CUSTOM TRANSFORMER
class Streak_score_wavg(BaseEstimator, TransformerMixin):
    def __init__(self, df, feat_names: list, n: int = 10):
        self.df = df
        self.feat_names = feat_names
        self.n = n

    # Build function to count how many matches used in streaks
    def _count_streak_length(self,
                             df: pd.DataFrame,
                             team: str,
                             date=pd.to_datetime(dt.today()),
                             n: int = 10):
        filter_ = (df['Date'] < date) & ((df['Home'] == team) | (df['Away'] == team))
        sort_team = df[filter_].sort_values('Date', ascending=False).head(n)
        # count how many matched before a specific date
        count = sort_team.shape[0]
        return count

    # Function to calculate weighted average of Streak score
    def _weight_avg_streak_score(self,
                                 df: pd.DataFrame,
                                 team: str,
                                 date=pd.to_datetime(dt.today()),
                                 n: int = 10
                                 ):
        """
        Calculate the weighted average of team according to the streak of n matches
        team: team name
        date: check all matches have occurred before this date
        n: Number of matches that you get

        The weight is calculated by absolute rank differance
        result belongs in range [-3,-1,0,1,3]
        - -3: the highest compensation (punishment)
        - -1: the normal compensation
        - 0: all matches are  draw
        - 1: the normal accomplishment
        - 3: the highest accomplishment
        """

        # Build system to compensate the higher rank losing, otherwise, accomplish the lower rank team winning
        def streak_score(is_home: bool, diff_rank: float, result: str):  # team is home or away
            if is_home:  # if team is home
                dict_score = {'H': 1, 'A': -1, 'D': 0}
                match_score = dict_score[result]

            else:  # if team is away
                dict_score = {'H': -1, 'A': 1, 'D': 0}
                diff_rank = -diff_rank
                match_score = dict_score[result]

            # Calculate the streak score if strong team loses weak team => compensation. In other hand, if lose team wins  strong team => accomplishment
            if diff_rank * match_score < 0:
                if match_score == 1:
                    score = (match_score + 2) * abs(diff_rank)
                else:
                    score = (match_score - 2) * abs(diff_rank)
            else:
                score = match_score * abs(diff_rank)

            # Return score
            return round(score, 2)

        #### Get all matches occurring before a specific date and containing this team ###
        try:
            filter_ = (df['Date'] < date) & ((df['Home'] == team) | (df['Away'] == team))
            sort_team = df[filter_].sort_values('Date', ascending=False).head(n)

            # define 3 list: rank diiference list, result_list, is_home_list: whether this team is Home? if Home value is True, else False
            diff_ls = list(sort_team['rank_diff'].values)
            result_ls = list(sort_team['result'].values)
            is_home_ls = [True if team == h else False for h, a in
                          sort_team[['Home', 'Away']].itertuples(index=False, name=None)]

            # Calculate the streak score according the above values
            ls_streak_score = [streak_score(is_home=i, diff_rank=d, result=r) for i, d, r in
                               zip(is_home_ls, diff_ls, result_ls)]

            # Define the weight:
            diff_abs_ls = sum([abs(round(dr, 2)) for dr in diff_ls])

            # Return result
            if diff_abs_ls == 0 or len(ls_streak_score) == 0:
                return np.nan
            else:
                weight_avg = sum(ls_streak_score) / diff_abs_ls
                return weight_avg
        except ZeroDivisionError:
            return np.nan

    def fit(self, X, y=None):
        print('\n>>>>Streak_score_wavg.fit() called.\n')
        self.X = X
        return self

    def transform(self, X, y=None):
        print('\n>>>>Streak_score_wavg.transform() called.\n')
        self.X = X

        # Home Streak score
        self.X[self.feat_names[0]] = self.X.apply(lambda x: self._weight_avg_streak_score(df=self.df,
                                                                                          team=x['Home'],
                                                                                          date=x['Date'],
                                                                                          n=self.n), axis=1)

        # Away Streak score
        self.X[self.feat_names[1]] = self.X.apply(lambda x: self._weight_avg_streak_score(df=self.df,
                                                                                          team=x['Away'],
                                                                                          date=x['Date'], n=self.n),
                                                  axis=1)

        # Home Streak length
        self.X[self.feat_names[2]] = self.X.apply(lambda x: self._count_streak_length(df=self.df,
                                                                                      team=x['Home'],
                                                                                      date=x['Date'], n=self.n), axis=1)

        # Away Streak length
        self.X[self.feat_names[3]] = self.X.apply(lambda x: self._count_streak_length(df=self.df,
                                                                                      team=x['Away'],
                                                                                      date=x['Date'], n=self.n), axis=1)
        print('\n>>>>Finish Streak_score_wavg.transform().\n')
        return self.X


class GD_weight_avg(BaseEstimator, TransformerMixin):
    def __init__(self, df, feat_names: list, n: int = 10):
        self.df = df
        self.feat_names = feat_names
        self.weight = (self.df[['rank_diff']]).to_numpy()
        self.scaler = sp.MinMaxScaler(feature_range=(0, 10))
        self.scaler.fit(self.weight)
        self.n = n

    # Build function to count how many matches used in streaks
    def _GA_GF_score(self,
                     is_home: bool,
                     diff_rank: float,
                     GA: float,
                     GF: float):  # team is home or away

        ##### DETERMINE THE WEIGHT HIGH OF LOW ######
        goal_diff = GF - GA
        weight_rank = 0
        # 2 cases: if diff_rank > 0 and goal diff > 0: rank of Home is higher than Away and Honme win Away.
        # Or rank of Home < Away and Home lose Away => there is a small weight
        if (diff_rank > 0 and goal_diff >= 0) or (diff_rank < 0 and goal_diff <= 0):
            # use abs then add negative => get small weight
            weight_rank = float(self.scaler.transform(X=np.array([[-abs(diff_rank)]])).squeeze())

        # 2 cases: if diff_rank > 0 and goal diff < 0: rank of Home is higher than Away BUT Honme lose Away.
        # Or rank of Home < Away BUT Home win Away => there is a significant weight
        elif (diff_rank > 0 and goal_diff < 0) or (diff_rank < 0 and goal_diff > 0):
            weight_rank = float(self.scaler.transform(X=np.array([[abs(diff_rank)]])).squeeze())

        if not is_home:  # if team is Away
            goal_diff = - goal_diff

        dict_score = {'goal_diff': goal_diff,
                      'weight': weight_rank,
                      'weight_goal_diff': goal_diff * weight_rank
                      }
        return dict_score

    # Function to calculate weighted average of Streak score
    def _weight_avg_Goal_Diff(self,
                              df: pd.DataFrame,
                              team: str,
                              date=pd.to_datetime(dt.today()),
                              n: int = 10):
        """
       Calculate the weighted average of team
       Weighted average of Home team's goal difference from 5 to 10 recent matches (Applied the punish-accomplish system to adjust weight)
       The weight is calculated by the rank difference which is normalized. Params:
       - team: team name
       - date: check all matches have occurred before this date
       - n: Number of matches that you get

       There are 2 cases considering a significant weight:
       - If diff_rank > 0 and goal diff < 0: rank of Home is higher than Away BUT Home lose Away.
       - Or diff_rank < 0 and goal diff > 0: rank of Home < Away BUT Home win Away
        """

        try:
            # Get all matches occurring before a specific date and containing this team
            _filter = (df['Date'] < date) & ((df['Home'] == team) | (df['Away'] == team))
            sort_team = df[_filter].sort_values('Date', ascending=False).head(n)

            # define 4 list: GA list, GF list, rank_diff list, is_home_list: whether this team is Home? if Home value is True, else False
            rank_ls = list(sort_team['rank_diff'].values)
            gf_ls = list(sort_team['score_home'].values)
            ga_ls = list(sort_team['score_away'].values)
            is_home_ls = [True if team == h else False for h, a in
                          sort_team[['Home', 'Away']].itertuples(index=False, name=None)]

            # Calculate the total weight score regarding to the weight
            weight_diff_list = [self._GA_GF_score(is_home=i,
                                                  diff_rank=dr,
                                                  GF=gf,
                                                  GA=ga) for i, dr, gf, ga in zip(is_home_ls,
                                                                                  rank_ls,
                                                                                  gf_ls,
                                                                                  ga_ls)]

            # Return result
            if sum([i['weight'] for i in weight_diff_list]) == 0:
                return np.nan
            else:
                # Define the weight:
                sum_weight_diff_rank = sum([i['weight_goal_diff'] for i in weight_diff_list])
                sum_weight = sum([i['weight'] for i in weight_diff_list])
                result = sum_weight_diff_rank / sum_weight
                return result
        except ZeroDivisionError:
            return np.nan

    def fit(self, X, y=None):
        self.X_ = X
        print('\n>>>>GD_weight_avg.fit() called.\n')
        return self

    def transform(self, X, y=None):
        print('\n>>>>GD_weight_avg.transform() called.\n')
        self.X_ = X.copy()

        # Goal Difference of Home team
        self.X_[self.feat_names[0]] = self.X_.apply(lambda x: self._weight_avg_Goal_Diff(df=self.df,
                                                                                         team=x['Home'],
                                                                                         date=x['Date'],
                                                                                         n=self.n), axis=1)

        # Goal Difference of Away team
        self.X_[self.feat_names[1]] = self.X_.apply(lambda x: self._weight_avg_Goal_Diff(df=self.df,
                                                                                         team=x['Away'],
                                                                                         date=x['Date'],
                                                                                         n=self.n), axis=1)

        print('\n>>>>Finish GD_weight_avg.transform().\n')
        return self.X_


class Merge_rank_league_perform(BaseEstimator, TransformerMixin):
    def __init__(self):
        None

    def _merge_rank(self, matches, rank_tb):
        # add rank of Home
        comb = pd.merge(left=matches,
                        right=rank_tb,
                        left_on='Home',
                        right_on='team',
                        how='left').rename(columns={'RK': 'home_rank'}).drop(columns='team')

        # add rank of Away
        comb = pd.merge(left=comb,
                        right=rank_tb,
                        left_on='Away',
                        right_on='team',
                        how='left', suffixes=('_home', '_away')).rename(
            columns={'RK': 'away_rank', 'total_point_home': 'home_point', 'total_point_away': 'away_point'}).drop(
            columns=['team', 'team_short_home', 'team_short_away'])
        return comb

    def fit(self, X, y=None):
        print('\n>>>>Merge_rank_league_perform.fit() called.\n')
        return self

    def transform(self, X: pd.DataFrame, y=None):
        print('\n>>>>Merge_rank_league_perform.transform() called.\n')
        X_ = X.copy()

        # Compatible name of Home and Away
        X_[['Home', 'Away']] = X_[['Home',
                                   'Away']].replace({'Korea Republic': 'Korea Rep',
                                                     'Equatorial Guinea': 'Equ. Guinea',
                                                     'Republic of Ireland': 'Rep. of Ireland',
                                                     'Hong Kong, China': 'Hong Kong',
                                                     "Bosnia and Herzegovina": "Bosnia & Herz'na",
                                                     'Czechia': 'Czech Republic',
                                                     'Papua New Guinea': 'Papua NG',
                                                     'Antigua and Barbuda': 'Antigua',
                                                     'Cabo Verde': 'Cape Verde',
                                                     'Dominican Republic': 'Dominican Rep.',
                                                     'The Gambia': 'Gambia',
                                                     'North Macedonia': 'N. Macedonia',
                                                     'St Kitts and Nevis': 'St. Kitts & Nevis',
                                                     'St Lucia': 'St. Lucia',
                                                     'South Sudan': 'Sudan',
                                                     'Trinidad and Tobago': 'Trin & Tobago',
                                                     'United Arab Emirates': 'UAE'
                                                     })
        try:
            # Connect database
            self.db = SQLiteDBManager()

            # Merge wrangle_df and league table
            if 'Tournament_id' in X_.columns and 'round_id' in X_.columns:
                pass
            else:
                league_tb = self.db.export_table_to_dataframe(table_name="league_tb")
                X_ = pd.merge(left=X_,
                              right=league_tb,
                              on=['Tournament', 'Round'],
                              how='left').drop(columns=['Tournament',
                                                        'Round'])

            # Merge rank table
            rank_tb = self.db.export_table_to_dataframe(table_name="fifa_rank_tb")
            X_ = self._merge_rank(matches=X_, rank_tb=rank_tb)

            # with teams are no rank => fill it with rank 190 and point 350 (last rank and smallest point)
            X_ = X_.fillna(value={'home_rank': 190,
                                  'away_rank': 190,
                                  'home_point': 350,
                                  'away_point': 350})

            #  Merge with team performance table
            perf_team = self.db.export_table_to_dataframe(table_name="team_performance")
            X_ = pd.merge(left=X_,
                          right=perf_team[['Team', 'home_perf_wm']],
                          left_on='Home',
                          right_on='Team',
                          how='left').drop(columns='Team')
            X_ = pd.merge(left=X_,
                          right=perf_team[['Team', 'away_perf_wm']],
                          left_on='Away',
                          right_on='Team',
                          how='left').drop(columns='Team')

            # Calculate rank diff
            X_['rank_diff'] = X_['home_point'] - X_['away_point']

            # close db
            self.db.close_connection()
            print('\n>>>>Closed DataBase successfully.\n')

        except Exception as e:
            # close db
            self.db.close_connection()
            # Raise error
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            return message

        else:
            print('\n>>>>Finish Merge_rank_league_perform.transform().\n')

            return X_


class Features_chosen(BaseEstimator, TransformerMixin):
    def __init__(self, features: list):
        self.features = features
        # self.minimum_length = minimum_length

    def fit(self, X, y=None):
        print('\n>>>>Features_chosen.fit() called.\n')
        return self

    def transform(self, X: pd.DataFrame, y=None):
        self.X = X.copy()
        #filter_ = (self.X['Home_streak_length'] >= self.minimum_length) & (self.X['Away_streak_length'] >= se)
        self.X = self.X[self.features]
        print('\n>>>>Features_chosen.transform() finished.\n')
        return self.X


if __name__ == '__main__':
    path = 'C:/Users/user2/PycharmProjects/selenium_scraping_match/data'

    wrangler = Wrangling_data(data_path=path)
    clean_df = wrangler.df_wrangling(update_league_tb=True, update_team_perf=False,recal_ELO=True)

    data_list = [os.path.join(path, file) for file in os.listdir(path) if os.path.splitext(file)[-1] == '.csv']
    dfs = [pd.read_csv(filepath_or_buffer=f, parse_dates=['Date']) for f in data_list]
    full_df = pd.concat(dfs, ignore_index=True).sort_values(by='Date').reset_index(drop=True)

    # Build PipeLine
    streak_feat_list = ['Home_streak_score', 'Away_streak_score', 'Home_streak_length', 'Away_streak_length']
    GD_feat_list = ['Home_diff_score', 'Away_diff_score']
    features_col = ['Date', 'Tournament_id', 'round_id', 'Home', 'Away', 'home_perf_wm', 'away_perf_wm',
                   'rank_diff', 'Home_streak_score', 'Away_streak_score', 'Home_streak_length',

                   'Away_streak_length', 'Home_diff_score', 'Away_diff_score']

    feature_engineer = Pipeline(steps=[
        # ('imputer', SimpleImputer(strategy='mean', missing_values=np.nan)),

        ('merger', Merge_rank_league_perform()),
        ('streak_score', Streak_score_wavg(df=clean_df, feat_names=streak_feat_list, n=6)),
        ('goal_diff', GD_weight_avg(df=clean_df, feat_names=GD_feat_list, n=6)),
        ('get_feature', Features_chosen(features=features_col))
    ])
    feature_engineer.fit(full_df)
    full_df_pre = feature_engineer.transform(full_df)

    # Filter with streak match >= 5 matches
    # filter_length = (X['Home_streak_length'] >= 5) & (X['Away_streak_length'] >= 5)
    # X = X.loc[filter_length, features_col].reset_index(drop=True).drop(columns=['Home_streak_length',
    #                                                                             'Away_streak_length'])
    # new_df_pre = new_df_pre[features_col]
