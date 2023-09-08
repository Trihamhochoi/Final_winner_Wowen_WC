from database import SQLiteDBManager
import pandas as pd
import os


class Wrangling_data:
    def __init__(self, data_path):
        self.team_dict = None
        self.data_list = [os.path.join(data_path, file) for file in os.listdir(data_path) if
                          os.path.splitext(file)[-1] == '.csv']
        dfs = [pd.read_csv(filepath_or_buffer=f, parse_dates=['Date']) for f in self.data_list]
        self.raw_df = pd.concat(dfs, ignore_index=True).sort_values(by='Date').reset_index(drop=True)
        self.wrangle_df = None

        self.db = SQLiteDBManager()

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

    def df_wrangling(self, update_league_tb: bool = False, update_team_perf: bool = False, recal_ELO: bool = False):
        # Define function to merge rank vs wrangle df

        # Create a copy
        self.wrangle_df = self.raw_df.copy()

        # Compatible name of Home and Away
        self.wrangle_df[['Home',
                         'Away']] = self.wrangle_df[['Home',
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

        # Extract year of match
        self.wrangle_df['Year'] = self.wrangle_df.Date.dt.year

        # Extract year of tournament
        self.wrangle_df['tour_start'] = self.wrangle_df.groupby('Tournament')['Year'].transform('min')

        # Compatible with merged tables and update date columns
        self.wrangle_df['Date'] = pd.to_datetime(self.wrangle_df['Date'].dt.date)

        # Update league table or not
        try:
            if update_league_tb:
                self.db.update_league_tb(df=self.wrangle_df)
                print('\n>>>>Finished the updated process\n')

            # Merge wrangle_df and league table
            self.wrangle_df = self.wrangle_df.rename(columns={'Round': 'round_name'})
            tb_name = "league_weight_tb"
            league_weight_tb = self.db.export_table_into_dataframe(table_name=tb_name)
            self.wrangle_df = pd.merge(left=self.wrangle_df,
                                       right=league_weight_tb,
                                       on=['tour_start', 'Tournament', 'round_name'],
                                       how='inner')
            target_col = ['year_league_id', 'league_id', 'round_id', 'total_weight', 'Date', 'Home', 'Away',
                          'score_home',
                          'score_away', 'result', 'pen_result']
            self.wrangle_df = self.wrangle_df[target_col]

            # Get dict of national team
            if recal_ELO:
                nation_tb = self.db.export_table_into_dataframe(table_name='nation_tb')
                self.team_dict = {i: 1500 for i in nation_tb['nation'].values}
            else:
                elo_team_rank = self.db.export_table_into_dataframe(table_name='elo_rank_tb')
                self.team_dict = {i['Team']: i['ELO_point'] for i in
                                  elo_team_rank[['Team', 'ELO_point']].to_dict(orient='records')}

            # Merge with my ELO rank I built:
            self.wrangle_df['point_list'] = self.wrangle_df[['Home',
                                                             'Away',
                                                             'result',
                                                             'total_weight']].apply(
                lambda x: self._update_ELO_point(home=x['Home'],
                                                 away=x['Away'],
                                                 result=x['result'],
                                                 weight=x['total_weight']),
                axis=1)
            self.wrangle_df['home_point'] = self.wrangle_df['point_list'].apply(lambda x: x[0])
            self.wrangle_df['away_point'] = self.wrangle_df['point_list'].apply(lambda x: x[1])
            self.wrangle_df = self.wrangle_df.drop(columns='point_list')

            my_elo_rank = (
                pd.DataFrame(self.team_dict.items(), columns=['Team', 'ELO_point'])
                .sort_values(by='ELO_point', ascending=False)
                .reset_index(drop=True)
            )
            my_elo_rank['updated_date'] = pd.Timestamp.now(self.db.tz)
            my_elo_rank = (my_elo_rank[['Team', 'ELO_point', 'updated_date']]
                           .reset_index()
                           .rename(columns={'index': 'team_id'}))

            # Save the ELO point in the database
            self.db.import_dataframe_in_db(df=my_elo_rank,table_name='elo_rank_tb')

            # Update team performance or not
            if update_team_perf:
                self.db.update_team_perf_tb(df=self.wrangle_df)
                print('\n>>>>Update Team Performance table successfully.\n')

            perf_name = "team_performance"
            perf_team = self.db.export_table_into_dataframe(table_name=perf_name)

            #  Merge with team performance table
            self.wrangle_df = pd.merge(left=self.wrangle_df, right=perf_team[['Team', 'home_perf_wm']], left_on='Home',
                                       right_on='Team',
                                       how='left').drop(columns='Team')
            self.wrangle_df = pd.merge(left=self.wrangle_df, right=perf_team[['Team', 'away_perf_wm']], left_on='Away',
                                       right_on='Team',
                                       how='left').drop(columns=['Team'])

            # Calculate rank diff and choose some specific variables
            # target_col = [col for col in self.wrangle_df.columns if not col.startswith('pen')]
            # self.wrangle_df = self.wrangle_df.drop(columns=['home_rank', 'away_rank'])
            self.wrangle_df['rank_diff'] = self.wrangle_df['home_point'] - self.wrangle_df['away_point']

            # close db
            self.db.close_connection()

        except Exception as e:
            # close db
            self.db.close_connection()
            # Raise error
            raise e
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            return message

        else:
            return self.wrangle_df


if __name__ == '__main__':
    path = 'C:/Users/user2/PycharmProjects/selenium_scraping_match/data'

    raw_df = Wrangling_data(data_path=path)
    clean_df = raw_df.df_wrangling(update_league_tb=True, update_team_perf=False, recal_ELO=False)
    print(clean_df.head())
