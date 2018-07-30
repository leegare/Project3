import sqlite3
import pandas as pd
import numpy as np

def load_db_to_df(db_name):
    table_name_1  = 'Player_Attributes'
    table_name_2  = 'Player'
    table_name_3  = 'Match'
    table_name_4  = 'League'
    table_name_5  = 'Country'
    table_name_6  = 'Team'
    table_name_7  = 'Team_Attributes'
    table_name_8  = 'sqlite_sequence'
    q1 = 'SELECT * FROM {tn}'.format(tn=table_name_1)
    q2 = 'SELECT * FROM {tn}'.format(tn=table_name_2)
    q3 = 'SELECT * FROM {tn}'.format(tn=table_name_3)
    q4 = 'SELECT * FROM {tn}'.format(tn=table_name_4)
    q5 = 'SELECT * FROM {tn}'.format(tn=table_name_5)
    q6 = 'SELECT * FROM {tn}'.format(tn=table_name_6)
    q7 = 'SELECT * FROM {tn}'.format(tn=table_name_7)
    q8 = 'SELECT * FROM {tn}'.format(tn=table_name_8)
    conn = sqlite3.connect(db_name)
    player_attributes = pd.read_sql_query(q1, conn, index_col=['id'])
    player = pd.read_sql_query(q2, conn, index_col=['id'])
    match = pd.read_sql_query(q3, conn, index_col=['match_api_id'])
    league = pd.read_sql_query(q4, conn, index_col=['id'])
    country = pd.read_sql_query(q5, conn, index_col=['id'])
    team = pd.read_sql_query(q6, conn, index_col=['id'])
    team_attributes = pd.read_sql_query(q7, conn, index_col=['id'])
    sqlite_sequence = pd.read_sql_query(q8, conn)
    conn.close()
    return player_attributes, player, match, league, country, team, team_attributes, sqlite_sequence

def load_db_to_df_interim(db_name):
    table_name_1  = 'Player_Attributes'
    table_name_2  = 'Player'
    table_name_3  = 'Match'
    table_name_4  = 'League'
    table_name_6  = 'Team'
    table_name_7  = 'Team_Attributes'
    q1 = 'SELECT * FROM {tn}'.format(tn=table_name_1)
    q2 = 'SELECT * FROM {tn}'.format(tn=table_name_2)
    q3 = 'SELECT * FROM {tn}'.format(tn=table_name_3)
    q4 = 'SELECT * FROM {tn}'.format(tn=table_name_4)
    q6 = 'SELECT * FROM {tn}'.format(tn=table_name_6)
    q7 = 'SELECT * FROM {tn}'.format(tn=table_name_7)
    conn = sqlite3.connect(db_name)
    player_attributes = pd.read_sql_query(q1, conn, index_col=['id'])
    player = pd.read_sql_query(q2, conn, index_col=['player_api_id'])
    match = pd.read_sql_query(q3, conn, index_col=['match_api_id'])
    league = pd.read_sql_query(q4, conn, index_col=['country_id'])
    team = pd.read_sql_query(q6, conn, index_col=['team_api_id'])
    team_attributes = pd.read_sql_query(q7, conn, index_col=['id'])
    conn.close()
    return player_attributes, player, match, league, team, team_attributes

def show_db_tables(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    query_check  = 'SELECT name FROM sqlite_master WHERE type= "table";'
    c.execute(query_check)
    table_set = [item[0] for item in c.fetchall()]
    print(table_set)

def save_tables(db_name, player_attributes, player, match, league, team, team_attributes):
    conn = sqlite3.connect(db_name)
    player_attributes.to_sql("player_attributes", conn, if_exists="replace")
    player.to_sql("player", conn, if_exists="replace")
    match.to_sql("match", conn, if_exists="replace")
    league.to_sql("league", conn, if_exists="replace")
    team.to_sql("team", conn, if_exists="replace")
    team_attributes.to_sql("team_attributes", conn, if_exists="replace")
    conn.commit()
    conn.close()
    print('Tables Saved to',db_name)

def check_dupes(table):
    print(table.shape[0], table.drop_duplicates().shape[0])

def check_nan(table):
    print('Current count:',table.shape, 'Count without Nulls:',table.dropna().shape)
    print(table.isnull().sum())
