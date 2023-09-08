import sqlite3
import pandas as pd
import numpy as np
from pytz import timezone


class SQLiteDBManager:
    def __init__(self, db_name: str = "./database/soccer_database.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.tz = timezone('Asia/Ho_Chi_Minh')
        print(f'Connect database {self.db_name} successfully')

    def create_table_from_dataframe(self, df, table_name):
        df.to_sql(table_name, self.conn, index=False, if_exists='replace')

    def import_dataframe_into_db(self, df, table_name):
        df.to_sql(table_name, self.conn, index=False, if_exists='replace')
        print(f"DataFrame is import into '{table_name}' in '{self.db_name}' successfully.")

    # Export table into dataframe
    def export_table_to_dataframe(self, table_name):
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, self.conn)
        print(f" Table '{table_name}' is exported to Dataframe successfully.")
        return df

    # list all table in database
    def list_tables_in_database(self):

        # Query the "sqlite_master" table to get all table names
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        # Extract the table names from the result
        table_names = [table[0] for table in self.cursor.fetchall()]
        if table_names:
            print("Tables in the database:")
            return table_names
        else:
            return "No tables found in the database."

    # UPDATE LEAGUE TABLE
    def update_league_tb(self, df):
        try:
            # get league table
            tour_round_table = df.groupby(['tour_start', 'Tournament', 'Round'], as_index=False).agg(
                {'Home': 'count'}).iloc[:, [0, 1, 2]]
            tour_round_table['league_name'] = tour_round_table['Tournament'].str.extract(pat=r'(\d*)(.+)', expand=True)[
                1].str.strip()

            # define tour list
            league_df = tour_round_table['Tournament'].str.extract(pat=r'(\d*)(.+)', expand=True)[
                1].str.strip().drop_duplicates().sort_values().reset_index(drop=True).to_frame().rename(
                columns={1: 'league_name'})

            # define weight
            weight_league = {
                'Friendlies (W)': 20,
                'SheBelieves Cup': 20,
                'Algarve Cup': 20,
                "AFC Women's Asian Cup qualification": 25,
                'WCQ — CONCACAF (M)': 25,
                "AFC Women's Asian Cup": 30,
                'Africa Women Cup of Nations': 30,
                'CONCACAF W Championship': 35,
                'Copa América Femenina': 35,
                "UEFA Women's Euro Qualification": 30,
                'WCQ — UEFA (W)': 40,
                "UEFA Women's Euro": 45,
                "FIFA Women's World Cup": 50,
            }
            weight_df = pd.DataFrame(weight_league.items(), columns=['league_name', 'league_weight'])

            # combine weight with league weight
            league_df = (pd.merge(left=league_df, right=weight_df, on='league_name')
                         .sort_values(by='league_weight')
                         .reset_index(drop=True)
                         .reset_index()
                         .rename(columns={'index': 'league_id'}))
            league_df['updated_date'] = pd.Timestamp.now(self.tz)
            print('Tournament table is updated successfully')
            self.import_dataframe_into_db(df=league_df, table_name="league_tb")

            # Create Round DF
            round_list = [
                ['not_available', 'Group stage'],
                ['Qualifying stage', 'Preliminary round', 'First round'],
                ['Classification round', 'Play-offs', 'Round of 16', 'Repechage', 'Second round'],
                ['Quarter-finals', 'Fifth-place match', 'Third round'],
                ['Semi-finals'],
                ['Final'],
                ['Third-place match']
            ]
            round_map = {r: (i) for i, rounds in enumerate(round_list) for r in rounds}
            round_df = pd.DataFrame(round_map.items(), columns=['round_name', 'round_id']).reset_index().rename(
                columns={'index': 'id'})
            round_df['round_weight'] = round_df['round_id'] * 2
            round_df['round_weight'] = round_df['round_weight'].apply(lambda x: 2 if x < 2 else 9 if x > 10 else x)
            round_df['updated_date'] = pd.Timestamp.now(self.tz)
            print('Round table is updated successfully')
            self.import_dataframe_into_db(df=round_df, table_name="round_tb")

            # Create league_start_table
            league_start_table = (tour_round_table['tour_start']
                                  .drop_duplicates()
                                  .sort_values()
                                  .to_frame()
                                  .reset_index(drop=True)
                                  .reset_index()
                                  .rename(columns={'index': 'year_league_id'}))
            league_start_table['year_weight'] = (league_start_table['year_league_id']) * 4
            league_start_table['updated_date'] = pd.Timestamp.now(self.tz)
            print('League_start table is updated successfully')
            self.import_dataframe_into_db(df=league_start_table, table_name="year_start_tb")

            # Create League Weight table
            league_weight_table = pd.merge(left=tour_round_table, right=league_df, on='league_name', how='left')
            league_weight_table = pd.merge(left=league_weight_table, right=round_df, left_on='Round', right_on='round_name',
                                           how='left').drop(columns='Round')
            league_weight_table = pd.merge(left=league_weight_table, right=league_start_table, on='tour_start', how='left')
            target_col = ['year_league_id', 'league_id', 'round_id', 'tour_start', 'Tournament', 'league_name',
                          'round_name', 'year_weight', 'league_weight', 'round_weight']
            league_weight_table = league_weight_table[target_col].sort_values(
                by=['year_league_id', 'league_id', 'round_id']).reset_index(drop=True)
            league_weight_table['total_weight'] = league_weight_table[['league_weight', 'year_weight', 'round_weight']].sum(axis=1)
            league_weight_table['updated_date'] = pd.Timestamp.now(self.tz)

            # Import to database
            print('League_weight table is updated successfully')
            self.import_dataframe_into_db(df=league_weight_table, table_name="league_weight_tb")
        except Exception as e:
            raise e

    # UPDATE TEAM PERFORMANCE
    def update_team_perf_tb(self, df):

        # Build win match for Home and Away
        df['home_win'] = df['result'].apply(lambda x: 1 if x == 'H' else 0)
        df['away_win'] = df['result'].apply(lambda x: 1 if x == 'A' else 0)
        # HOME PERFORMANCE
        home_perf = df.groupby(['year_league_id',
                                'league_id',
                                'round_id',
                                'Home'], as_index=False).agg(played_matches=('result', 'count'),
                                                             win_matches=('home_win', 'sum'),
                                                             weight=('total_weight', 'min'))

        home_perf['win_percent'] = round(home_perf['win_matches'] / home_perf['played_matches'], 4)

        home_perf['weight'] = home_perf['Tournament_id'] + home_perf['round_id']

        # Define a lambda function to compute the weighted mean:
        hwm = lambda x: np.average(x, weights=home_perf.loc[x.index, "weight"])

        # Calculate Home performance according to weight defined before
        home_perf_squad = (home_perf
                           .groupby('Home', as_index=False)
                           .agg(home_perf_wm=("win_percent", hwm), total_matches=('played_matches', 'sum'))
                           .rename(columns={'Home': 'Team'}))

        # AWAY PERFORMANCE
        away_perf = df.groupby(['year_league_id',
                                'league_id',
                                'round_id',
                                'Away'], as_index=False).agg(played_matches=('result', 'count'),
                                                             win_matches=('away_win', 'sum'),
                                                             weight=('total_weight', 'min'))

        away_perf['win_percent'] = round(away_perf['win_matches'] / away_perf['played_matches'], 4)

        away_perf['weight'] = away_perf['Tournament_id'] + away_perf['round_id']

        awm = lambda x: np.average(x, weights=away_perf.loc[x.index, "weight"])

        away_perf_squad = (away_perf.groupby('Away', as_index=False)
                           .agg(away_perf_wm=("win_percent", awm), total_matches=('played_matches', 'sum'))
                           .rename(columns={'Away': 'Team'}))

        # MERGE 2 DF
        perf_team = pd.merge(left=home_perf_squad,
                             right=away_perf_squad,
                             on='Team',
                             how='outer',
                             suffixes=('_home', '_away')).fillna(-1)
        perf_team['updated_date'] = pd.Timestamp.now(self.tz)

        # IMPORT TO DATABASE
        table_name = 'team_performance'
        print('Team performance is updated successfully')
        self.import_dataframe_into_db(df=perf_team, table_name=table_name)
        return None

    def close_connection(self):
        self.conn.close()


if __name__ == '__main__':
    database = SQLiteDBManager()
    tb_name = "league_tb"
    lg_df = database.export_table_to_dataframe(table_name=tb_name)
    database.close_connection()
