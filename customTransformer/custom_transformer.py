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
import warnings


# CREATE CLASS FOR CUSTOM TRANSFORMER
class Data_Wrangling(BaseEstimator, TransformerMixin):
    def __init__(self,
                 db_name: str = "./database/soccer_database.db",
                 update_league_round_weight_tb: bool = False,
                 update_team_perf: bool = False):

        self.db_name = db_name
        self.update_league_round_weight_tb = update_league_round_weight_tb
        self.update_team_perf = update_team_perf

    def fit(self, X, y=None):
        print('\n>>>>Data_Wrangling.fit() called.')
        self.X = X
        return self

    def transform(self, X, y=None):
        print('>>>>Data_Wrangling.transform() called.')
        self.X = X
        db = SQLiteDBManager(db_name=self.db_name)
        try:
            self.X[['Home',
                    'Away']] = self.X[['Home',
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

            if 'year_league_id' not in self.X.columns and 'league_id' not in self.X.columns and 'round_id' not in self.X.columns:
                # Extract year of match
                self.X['Year'] = self.X.Date.dt.year

                # Extract year of tournament
                self.X['tour_start'] = self.X.groupby('Tournament')['Year'].transform('min')

                # Compatible with merged tables and update date columns
                self.X['Date'] = pd.to_datetime(self.X['Date'].dt.date)

                if self.update_league_round_weight_tb:
                    assert self.X.shape[0] > 1200
                    db.update_league_tb(df=self.X)
                    print('>>>>Finished the updated process in league round table\n')

                # Merge wrangle_df and league table
                self.X = self.X.rename(columns={'Round': 'round_name'})
                tb_name = "league_weight_tb"
                league_weight_tb = db.export_table_into_dataframe(table_name=tb_name)

                self.X = pd.merge(left=self.X,
                                  right=league_weight_tb,
                                  on=['tour_start', 'Tournament', 'round_name'],
                                  how='inner')

                if self.update_team_perf:
                    assert self.X.shape[0] > 1200
                    db.update_team_perf_tb_new(df=self.X)
                    print('>>>>Finished the updated process for Team Performance\n')
                db.close_connection()
            else:
                league_weight_tb = db.export_table_into_dataframe(table_name="league_weight_tb")
                league_tb = db.export_table_into_dataframe(table_name="league_tb")
                round_tb = db.export_table_into_dataframe(table_name="round_tb")
                year_start_tb = db.export_table_into_dataframe(table_name="year_start_tb")
                db.close_connection()

                # Check this game occurring in latest year or not
                if self.X['year_league_id'].max() > league_weight_tb['year_league_id'].max():
                    self.X['year_league_id'] = league_weight_tb['year_league_id'].max()

                X_copy = self.X.copy()
                self.X = pd.merge(left=self.X,
                                  right=league_weight_tb,
                                  on=['year_league_id', 'league_id', 'round_id'],
                                  how='inner')

                if self.X.shape[0] ==0:
                    self.X = pd.merge(left=X_copy,
                                      right=year_start_tb.drop(columns='updated_date'),
                                      on='year_league_id',
                                      how='left')

                    self.X = pd.merge(left=self.X,
                                      right=league_tb.drop(columns='updated_date'),
                                      on='league_id',
                                      how='left')

                    self.X = pd.merge(left=self.X,
                                      right=round_tb.drop(columns='updated_date'),
                                      on='round_id',
                                      how='left')
                    self.X['total_weight'] = self.X[['league_weight', 'year_weight', 'round_weight']].sum(axis=1)

        except Exception as e:
            # Raise error
            raise e
            # template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            # message = template.format(type(e).__name__, e.args)
            # print(message)
            # return message

        else:
            print('>>>>Finish Data_Wrangling.transform().\n')
            target_col = ['year_league_id', 'league_id', 'round_id', 'total_weight', 'Date',
                          'Home', 'Away']

            # With new data
            if 'result' in self.X.columns:
                target_col.extend(['score_home', 'score_away', 'result', 'pen_result'])
            return self.X[target_col]


class Calculate_ELO_point(BaseEstimator, TransformerMixin):
    def __init__(self,
                 recal_ELO: bool = False,
                 save_to_db: bool = False,
                 db_name: str = "./database/soccer_database.db"):
        self.team_dict = None
        self.recal_ELO = recal_ELO
        self.db_name = db_name
        self.save_to_db = save_to_db

    def _update_ELO_point(self, home, away, result: str, weight):
        """
        Set a base point as 1500 for all teams, then start calculating from 1st match to last match with rule:
        ELO point of Team A: Ra
        ELO point of Team B: Rb
        """
        # Get team point before the match
        if home in self.team_dict.keys() and away in self.team_dict.keys():
            home_point = self.team_dict[home]
            away_point = self.team_dict[away]
        else:
            home_point = 1500
            away_point = 1500

        # Check result to get corresponding point
        if result == 'H':
            Aa = 1
            Ab = 0
        elif result == 'A':
            Aa = 0
            Ab = 1
        else:
            Aa = 0.5
            Ab = 0.5

        # Calculate the expect point
        qa = 10 ** (home_point / 400)
        qb = 10 ** (away_point / 400)
        Ea = qa / (qa + qb)
        Eb = qb / (qa + qb)

        # new point of Home team and away team
        Ra_ = round(home_point + weight * (Aa - Ea), 2)
        Rb_ = round(away_point + weight * (Ab - Eb), 2)

        # Update team point
        if home in self.team_dict.keys() and away in self.team_dict.keys():
            self.team_dict[home] = Ra_
            self.team_dict[away] = Rb_
        return [Ra_, Rb_]

    def fit(self, X, y=None):
        self.X = X
        print('\n>>>>Calculate_ELO_point.fit() called.')
        return self

    def transform(self, X: pd.DataFrame, y=None):
        print('>>>>Calculate_ELO_point.transform() called.')
        self.X = X
        # Connect database
        db = SQLiteDBManager(db_name=self.db_name)
        try:
            # Get dict of national team
            elo_team_rank = db.export_table_into_dataframe(table_name='elo_rank_tb')

            # Merge with my ELO rank I built:
            if 'result' in self.X.columns:
                if self.recal_ELO:
                    nation_tb = db.export_table_into_dataframe(table_name='nation_tb')
                    self.team_dict = {i: 1500 for i in nation_tb['nation'].values}
                else:
                    self.team_dict = {i['Team']: i['ELO_point'] for i in
                                      elo_team_rank[['Team', 'ELO_point']].to_dict(orient='records')}

                self.X['point_list'] = (self.X[['Home', 'Away', 'result', 'total_weight']]
                                        .apply(lambda x: self._update_ELO_point(home=x['Home'],
                                                                                away=x['Away'],
                                                                                result=x['result'],
                                                                                weight=x['total_weight']),
                                               axis=1))

                self.X['home_point'] = self.X['point_list'].apply(lambda x: x[0])
                self.X['away_point'] = self.X['point_list'].apply(lambda x: x[1])
                self.X = self.X.drop(columns='point_list')

                # Update ELO point in the database
                if self.recal_ELO is True and self.recal_ELO != self.save_to_db:
                    self.save_to_db = True
                    print("\tWARNING: params 'save_to_db' was updated to True because you re-calculate ELO rank")

                if self.save_to_db:
                    my_elo_rank = (
                        pd.DataFrame(self.team_dict.items(), columns=['Team', 'ELO_point'])
                        .sort_values(by='ELO_point', ascending=False)
                        .reset_index(drop=True)
                    )
                    my_elo_rank['updated_date'] = pd.Timestamp.now(db.tz)
                    my_elo_rank = (my_elo_rank[['Team', 'ELO_point', 'updated_date']]
                                   .reset_index()
                                   .rename(columns={'index': 'team_id'}))
                    db.import_dataframe_in_db(df=my_elo_rank, table_name='elo_rank_tb')

            # If new data, just merge
            else:
                # Get elo rank for Home team
                self.X = pd.merge(left=self.X,
                                  right=elo_team_rank[['Team', 'ELO_point']],
                                  left_on='Home',
                                  right_on='Team',
                                  how='left').rename(columns={'ELO_point': 'home_point'})

                # Get elo rank for Away team
                self.X = pd.merge(left=self.X,
                                  right=elo_team_rank[['Team', 'ELO_point']],
                                  left_on='Away',
                                  right_on='Team',
                                  how='left').rename(columns={'ELO_point': 'away_point'})

            # Close Database
            db.close_connection()
            # Calculate rank diff
            self.X['rank_diff'] = self.X['home_point'] - self.X['away_point']

        except Exception as e:
            # close db
            db.close_connection()
            # Raise error
            raise e
            # template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            # message = template.format(type(e).__name__, e.args)
            # print(message)
            # return None

        else:
            print('>>>>Finish Calculate_ELO_point.transform().\n')
            return self.X


class Cal_Team_Performance(BaseEstimator, TransformerMixin):
    def __init__(self,
                 db_name: str = "./database/soccer_database.db"):
        self.db_name = db_name

    def fit(self, X, y=None):
        self.X = X.copy()
        print('\n>>>>Cal_Team_Performance.fit() called.')
        return self

    def _calculate_perfomance_team(self, perf_df, team, year_id):
        fil = (perf_df['Team'] == team) & (perf_df['year_league_id'].isin([year_id, year_id - 1, year_id - 2]))
        perf_one_team = perf_df[fil]
        shape = perf_one_team.shape[0]
        if shape > 0:
            performance = (perf_one_team['year_perfomance_wm'] * perf_one_team['total_matches']).sum() / (
                perf_one_team['total_matches']).sum()
        else:
            performance = 0
        return performance

    def transform(self, X: pd.DataFrame, y=None):
        print('>>>>Cal_Team_Performance.transform() called.')
        self.X = X.copy()
        self.db = SQLiteDBManager(db_name=self.db_name)
        try:
            perf_team = self.db.export_table_into_dataframe(table_name='team_performance')
            self.db.close_connection()

            # Calculate Home and Away Performance
            self.X['home_perf_wm'] = self.X[['Home', 'year_league_id']].apply(
                lambda x: self._calculate_perfomance_team(perf_df=perf_team,
                                                          team=x['Home'],
                                                          year_id=x['year_league_id']), axis=1)

            self.X['away_perf_wm'] = self.X[['Away', 'year_league_id']].apply(
                lambda x: self._calculate_perfomance_team(perf_df=perf_team,
                                                          team=x['Away'],
                                                          year_id=x['year_league_id']), axis=1)

        except Exception as e:
            # Raise error
            raise e

        else:
            print('>>>>Finish Cal_Team_Performance.transform().\n')
            return self.X


class Streak_score_wavg(BaseEstimator, TransformerMixin):
    def __init__(self,
                 feat_names: list,
                 n: int = 6,
                 db_name: str = "./database/soccer_database.db"):
        self.feat_names = feat_names
        self.n = n
        self.db_name = db_name

    # Build function to count how many matches used in streaks
    def _count_streak_length(self,
                             df: pd.DataFrame,
                             team: str,
                             date=pd.to_datetime(dt.today()),
                             n: int = 6):
        # Build dataframe
        # Filter
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
                                 n: int = 6
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

            # Calculate the streak score if strong team loses weak team => compensation. In other hand, if lose team
            # wins  strong team => accomplishment
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
            # filter date
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
        print('\n>>>>Streak_score_wavg.fit() called.')
        self.X = X.copy()
        return self

    def transform(self, X, y=None):
        print('>>>>Streak_score_wavg.transform() called.')
        self.X = X.copy()
        self.database = SQLiteDBManager(db_name=self.db_name)
        df = self.database.export_table_into_dataframe('clean_match_table')
        df['Date'] = pd.to_datetime(df['Date'])
        # Home Streak score
        self.X[self.feat_names[0]] = self.X.apply(lambda x: self._weight_avg_streak_score(df=df,
                                                                                          team=x['Home'],
                                                                                          date=x['Date'],
                                                                                          n=self.n), axis=1)
        # Away Streak score
        self.X[self.feat_names[1]] = self.X.apply(lambda x: self._weight_avg_streak_score(df=df,
                                                                                          team=x['Away'],
                                                                                          date=x['Date'],
                                                                                          n=self.n), axis=1)
        # Home Streak length
        self.X[self.feat_names[2]] = self.X.apply(lambda x: self._count_streak_length(df=df,
                                                                                      team=x['Home'],
                                                                                      date=x['Date'],
                                                                                      n=self.n), axis=1)

        # Away Streak length
        self.X[self.feat_names[3]] = self.X.apply(lambda x: self._count_streak_length(df=df,
                                                                                      team=x['Away'],
                                                                                      date=x['Date'],
                                                                                      n=self.n), axis=1)

        print('>>>>Finish Streak_score_wavg.transform().\n')
        return self.X


class GD_weight_avg(BaseEstimator, TransformerMixin):
    def __init__(self, feat_names: list,
                 n: int = 6,
                 db_name="./database/soccer_database.db"):
        self.db_name = db_name
        self.feat_names = feat_names
        self.n = n

        #  Extract Dataframe from DataBase
        db = SQLiteDBManager(db_name=self.db_name)
        self.df = db.export_table_into_dataframe(table_name='clean_match_table')
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Create Scaler
        weight = (self.df[['rank_diff']]).to_numpy()
        self.scaler = sp.MinMaxScaler(feature_range=(0, 10))
        # Fit with weight
        self.scaler.fit(weight)

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
        elif (diff_rank > 0 > goal_diff) or (diff_rank < 0 < goal_diff):
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
        self.X_ = X.copy()
        print('\n>>>>GD_weight_avg.fit() called.')
        return self

    def transform(self, X, y=None):
        print('>>>>GD_weight_avg.transform() called.')
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

        print('>>>>Finish GD_weight_avg.transform().\n')
        return self.X_


class Features_chosen(BaseEstimator, TransformerMixin):
    def __init__(self, minimum_length: int = 4):
        self.minimum_length = minimum_length

    def fit(self, X, y=None):
        self.X = X.copy()
        print('\n>>>>Features_chosen.fit() called.')
        return self

    def transform(self, X: pd.DataFrame, y=None):
        print('>>>>Features_chosen.transform() called.')
        self.X = X.copy()
        self.features = ['Date', 'year_league_id', 'league_id', 'round_id', 'Home', 'Away', 'rank_diff',
                         'home_perf_wm', 'away_perf_wm', 'Home_streak_score', 'Away_streak_score',
                         'Home_streak_length', 'Away_streak_length', 'Home_diff_score', 'Away_diff_score']
        if 'result' in self.X.columns:
            self.features.extend(['result', 'pen_result'])
            filter_ = (self.X['Home_streak_length'] >= self.minimum_length) & (
                    self.X['Away_streak_length'] >= self.minimum_length)
            self.X = (self.X.loc[filter_, self.features]
                      .reset_index(drop=True)
                      .drop(columns=['Home_streak_length', 'Away_streak_length'], errors='ignore')
                      )
        else:
            self.X = self.X[self.features].drop(columns=['Home_streak_length', 'Away_streak_length'], errors='ignore')
        print('\n>>>>Features_chosen.transform() finished.\n')
        return self.X


if __name__ == '__main__':
    path = 'C:/Users/user2/PycharmProjects/selenium_scraping_match/data'

    # wrangler = Wrangling_data(data_path=path)
    # clean_df = wrangler.df_wrangling(update_league_tb=True, update_team_perf=False,recal_ELO=True)

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

        ('merger', Calculate_ELO_point()),
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
