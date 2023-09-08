# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from datetime import datetime

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


virtual_result_dfs = list()


def predict_winner(model,
                   virtual_match_data: list,
                   actual_df,
                   feat_eng_pipeline,
                   ):
    # Recalculate probability
    def probability_in_2_ouputs(pro_h, pro_a):
        result = pro_h / (pro_h + pro_a)
        return result

    if type(virtual_match_data) == list:
        # Prepare data in case virtual_match_data is dict
        virtual_match_df = pd.DataFrame(virtual_match_data)
        virtual_match_df['Group_id'] = np.where(virtual_match_df.index < 4, 1, 2).astype(np.int64)
        virtual_match_df['Match_id'] = virtual_match_df.groupby('Group_id', as_index=False).cumcount()
    else:
        virtual_match_df = virtual_match_data

    # Check date of match if match occurred, merge with final data, else get the current date
    round_id = virtual_match_df['round_id'].unique()[0]
    league_id = virtual_match_df['league_id'].unique()[0]
    year_league_id = virtual_match_df['year_league_id'].unique()[0]
    match_occurred = any((actual_df['league_id'] == league_id) & (actual_df['round_id'] == round_id))
    if match_occurred:
        fil = ((actual_df['year_league_id'] == year_league_id) & (actual_df['league_id'] == league_id) & (
                    actual_df['round_id'] == round_id))
        true_df = actual_df[fil]

        virtual_match_df = pd.merge(left=virtual_match_df,
                                    right=true_df[['year_league_id','league_id', 'round_id', 'Home', 'Away', 'Date']],
                                    on=['year_league_id','league_id', 'round_id', 'Home', 'Away'], how='left')

        # If this round is playing but match is not occuring => fillna with current date
        virtual_match_df['Date'] = virtual_match_df['Date'].fillna(pd.to_datetime(datetime.now().date()))
    else:
        # the round have been occurring in the reality
        virtual_match_df['Date'] = pd.to_datetime(datetime.now().date())

    # transform dataframe
    virtual_match_feat_df = feat_eng_pipeline.transform(virtual_match_df)

    # Get match id and group id
    virtual_match_feat_df = pd.merge(left=virtual_match_feat_df,
                                     right=virtual_match_df[['year_league_id','league_id', 'round_id', 'Home', 'Away','Group_id','Match_id']],
                                     on=['year_league_id','league_id', 'round_id', 'Home', 'Away']
                                     ,how='left')

    # Predict the match then combine to dataframe
    virtual_match_feat_df.loc[:, 'prediction'] = model.predict(virtual_match_feat_df)

    # Predict probability then combine to dataframe
    prob_name = ['prob_' + a for a in model.classes_]
    virtual_match_feat_df.loc[:, prob_name] = model.predict_proba(virtual_match_feat_df)

    if match_occurred:
        # Compare with actual match
        virtual_result_df = pd.merge(left=virtual_match_feat_df.drop(columns=['Date']),
                                     right=true_df[
                                         ['round_id', 'Home', 'Away', 'result', 'pen_result', 'Date']],
                                     on=['round_id', 'Home', 'Away'])

        # Eliminate the probability of Draw, only using the Home prob or Away prob because this is the knock out stage
        virtual_result_df['prob_H'] = virtual_result_df['prob_H'] / (
                    virtual_result_df['prob_H'] + virtual_result_df['prob_A'])
        virtual_result_df['prob_A'] = virtual_result_df['prob_A'] / (
                    virtual_result_df['prob_H'] + virtual_result_df['prob_A'])
        virtual_result_df['prediction'] = np.where(virtual_result_df['prob_H'] > virtual_result_df['prob_A'], 'H', 'A')
        virtual_result_df = virtual_result_df.drop(columns='prob_D')

        fil = (virtual_result_df['result'] == virtual_result_df['prediction']) | (
                    virtual_result_df['pen_result'] == virtual_result_df['prediction'])
        virtual_result_df['is_correct'] = np.where(fil, True, False)

        # Get the result
        print(f"the model predict correct {fil.sum()} on {fil.count()} matches")
        print(
            f" the wrong results are below: \n{virtual_result_df[~fil][['year_league_id','league_id', 'round_id', 'Home', 'Away', 'prob_H', 'prob_A', 'prediction', 'result', 'pen_result']]}")
        order_col = ['Date','year_league_id', 'league_id', 'round_id', 'Group_id', 'Match_id', 'Home', 'Away', 'prob_H', 'prob_A',
                     'prediction', 'result', 'pen_result', 'is_correct']
        return virtual_result_df[order_col]
    else:
        # If the match still does not occur, the match will be double
        system_calculate = any(virtual_match_feat_df.duplicated(subset=['Group_id', 'Match_id']))
        if system_calculate:
            # Recalcualte the probility of Home and Away because from round 16, there is 2 likelihood occuring: Home win or Away win
            virtual_match_feat_df['prob_H_'] = virtual_match_feat_df[['prob_H', 'prob_A']].apply(
                lambda x: probability_in_2_ouputs(pro_h=x['prob_H'], pro_a=x['prob_A']), axis=1)
            virtual_match_feat_df['prob_A_'] = 1 - virtual_match_feat_df['prob_H_']

            # Create data frame to get which team be able to win from 2 match I created
            ls_ = list()
            for i, h, a, p_h, p_a in virtual_match_feat_df[['Home', 'Away', 'prob_H_', 'prob_A_']].itertuples(
                    index=True, name=None):
                try:
                    match_dict = dict()
                    if h == virtual_match_feat_df.loc[i + 1, 'Away'] and a == virtual_match_feat_df.loc[i + 1, 'Home']:

                        match_dict['Home'] = h
                        match_dict['Away'] = a
                        pro_h_2 = (p_h + virtual_match_feat_df.loc[i + 1, 'prob_A_']) / 2
                        pro_a_2 = (p_a + virtual_match_feat_df.loc[i + 1, 'prob_H_']) / 2
                        match_dict['prob_H'] = pro_h_2
                        match_dict['prob_A'] = pro_a_2
                        if pro_h_2 > pro_a_2:
                            match_dict['prediction'] = 'H'
                        else:
                            match_dict['prediction'] = 'A'

                        ls_.append(match_dict)
                    else:
                        pass
                except Exception as e:
                    break

                    # Combine to 2 df to compatible with next fucntion
            df_prob = pd.DataFrame(ls_)
            virtual_match_feat_df = virtual_match_feat_df.drop_duplicates(subset=['Date', 'year_league_id','league_id', 'round_id', 'Group_id', 'Match_id'])[
                ['Date','year_league_id', 'league_id', 'round_id', 'Group_id', 'Match_id', 'Home', 'Away']]

            return pd.merge(left=virtual_match_feat_df,
                            right=df_prob,
                            on=['Home', 'Away'],
                            how='outer')

        else:
            virtual_match_feat_df['prob_H'] = virtual_match_feat_df[['prob_H', 'prob_A']].apply(
                lambda x: probability_in_2_ouputs(pro_h=x['prob_H'], pro_a=x['prob_A']), axis=1)
            virtual_match_feat_df['prob_A'] = 1 - virtual_match_feat_df['prob_H']
            virtual_match_feat_df['prediction'] = np.where(
                virtual_match_feat_df['prob_H'] > virtual_match_feat_df['prob_A'], 'H', 'A')

        order_col = ['Date','year_league_id', 'league_id', 'round_id', 'Group_id', 'Match_id', 'Home', 'Away', 'prob_H', 'prob_A',
                     'prediction']
        return virtual_match_feat_df[order_col]


def arrange_next_round(virtual_df):
    ls_winner = list()
    # check df columns contain reuslt and pen result or not?
    if 'pen_result' in virtual_df.columns and 'result' in virtual_df.columns:
        print('There is this round in historical data')
        cols = ['year_league_id', 'league_id', 'round_id', 'Group_id', 'Match_id', 'Home', 'Away', 'result', 'pen_result',
                'prediction']
        match_df = virtual_df[cols]

        # Get winner
        for y_id,t_id, r_id, g_id, m_id, h, a, r, pen_r, prd in match_df.itertuples(index=False, name=None):
            dict_match = dict()
            dict_match['year_league_id'] = y_id
            dict_match['league_id'] = t_id
            dict_match['round_id'] = r_id + 1
            dict_match['Group_id'] = g_id
            if m_id % 2 == 0:
                dict_match['Match_id'] = 0
            else:
                dict_match['Match_id'] = 2
                # Get team winner
            if r == 'A':
                dict_match['winner_team'] = a
            elif r == 'H':
                dict_match['winner_team'] = h
            else:
                if pen_r == 'H':
                    dict_match['winner_team'] = h
                else:
                    dict_match['winner_team'] = a

            # add dict to list winner
            ls_winner.append(dict_match)
    else:
        # In case: This game is virtual
        print('This round still not happens in the reality >> Start to calculate...')
        cols = ['year_league_id', 'league_id', 'round_id', 'Group_id', 'Match_id', 'Home', 'Away', 'prediction', 'prob_H', 'prob_A']
        match_df = virtual_df[cols]
        # Get winner
        for y_id, t_id, r_id, g_id, m_id, h, a, prd, pro_a, pro_h in match_df.itertuples(index=False, name=None):
            dict_match = dict()
            dict_match['year_league_id'] = y_id
            dict_match['league_id'] = t_id
            dict_match['round_id'] = r_id + 1
            dict_match['Group_id'] = g_id
            if m_id % 2 == 0:
                dict_match['Match_id'] = 0
            else:
                dict_match['Match_id'] = 2
                # Get team winner
            if prd == 'A':
                dict_match['winner_team'] = a
            elif prd == 'H':
                dict_match['winner_team'] = h
            else:
                if pro_a > pro_h:
                    dict_match['winner_team'] = a
                else:
                    dict_match['winner_team'] = h

            # add dict to list winner
            ls_winner.append(dict_match)

    # Create next round:
    winner_group = pd.DataFrame(ls_winner)

    # In final 2 team head to head => no more group
    if winner_group.shape[0] == 2:
        winner_group['Group_id'] = 0

    winner_group = winner_group.groupby(['year_league_id', 'league_id', 'round_id', 'Group_id', 'Match_id'], as_index=False).agg(
        {'winner_team': list})

    # create next round in 2 way
    next_round = pd.concat([winner_group.drop(columns='winner_team'),
                            pd.DataFrame(winner_group['winner_team'].tolist(), index=winner_group.index)],
                           axis=1).rename(columns={0: 'Home', 1: 'Away'})
    next_round_revese = pd.concat([winner_group.drop(columns='winner_team'),
                                   pd.DataFrame(winner_group['winner_team'].tolist(), index=winner_group.index)],
                                  axis=1).rename(columns={1: 'Home', 0: 'Away'})
    comb_next_round = pd.concat([next_round, next_round_revese], axis=0).sort_values(
        by=['year_league_id', 'league_id', 'round_id', 'Group_id', 'Match_id'])

    return comb_next_round
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
