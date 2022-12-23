# coding: utf-8
# @Author: Kaiyuan Hou
import threading
from threading import Thread, Lock
import pandas as pd
import numpy as np
import time
import sqlite3

lock = Lock()
crime_data = pd.read_csv('./Dataset/NYPD_Arrest_Data__Year_to_Date_.csv')
interest_columns = ['ARREST_KEY', 'OFNS_DESC', 'AGE_GROUP', 'PERP_RACE', 'Latitude', 'Longitude']

def create_connection(db_file):
    con = None
    try:
        con = sqlite3.connect(db_file, check_same_thread=False)
    except:
        print("cannot connect to database.")
    return con


def create_table(connection, cursor):
    cursor.execute('DROP TABLE IF EXISTS crime;')
    cursor.execute('''
                  CREATE TABLE crime
                  (crime_id INTEGER PRIMARY KEY, 
                  offense_type TEXT,
                  age_group TEXT,
                  prep_race TEXT,
                  latitude REAL,
                  longitude REAL,
                  time_stamp REAL,
                  escape_time REAL
                  );
                  ''')
    connection.commit()


def post_to_database(connection, cursor, event):
    db_attributes = ['crime_id', 'offense_type', 'age_group', 'prep_race',
                    'latitude', 'longitude', 'time_stamp', 'escape_time']
    columns = ', '.join(db_attributes)

    placeholders = ', '.join('?' * len(event))
    sql = 'INSERT INTO crime ({}) VALUES ({}) RETURNING crime_id'.format(columns, placeholders)
    values = [x for x in event.values()]
    try:
        lock.acquire(True)
        cursor.execute(sql, values)
        # cursor.execute(
        #     "SELECT count(*) FROM crime "
        # )
        # cursor.execute(
        #     "Select * FROM crime"
        # )
        print(f'new crime happened, crime id: {cursor.fetchall()[0][0]}')
        # print(cursor.fetchall())
        # output = pd.DataFrame(cursor.fetchall())
        # print(output)
        # print(f'there are still {c.fetchall()[0][0]} escaping')
    finally:
        lock.release()
    connection.commit()


def new_crime_event(connection, cursor):
    while True:
        # print("new crime!")
        sample = crime_data.sample(ignore_index=True)
        new_crime = sample[interest_columns].to_dict('records')[0]
        new_crime['time_stamp'] = time.time()
        new_crime['escape_time'] = new_crime['time_stamp'] + np.random.normal(20, 5)
        # new_crime['escape_time'] = new_crime['time_stamp'] + 5
        post_to_database(connection, cursor, new_crime)
        time.sleep(int(np.random.normal(10, 3)))
    # return new_crime_event()


def criminal_arrested(connection, cursor):
    while True:
        now = time.time()
        # sql = 'SELECT crime_id, sum(time_stamp, escape_time) as escaping FROM crime WHERE escaping < :now'
        try:
            lock.acquire(True)
            cursor.execute(
                "DELETE FROM crime WHERE escape_time < ?  RETURNING crime_id", (now,)
            )
            ret = cursor.fetchall()
            if ret:
                for crimeID in ret[0]:
                    print(f'criminal arrested, crime ID: {crimeID}')

            # cursor.execute(
            #     "Select * FROM crime"
            # )
            # output = pd.DataFrame(cursor.fetchall())
            # if len(output):
            #     print('after arrestment!!!')
            #     print(output)
            # cursor.execute(
            #     "SELECT crime_id, time_stamp, escape_time as escaping FROM crime WHERE escape_time < ?", (now,)
            # )
            # output = pd.DataFrame(cursor.fetchall(), columns=['crime_id', 'time_stamp', 'escape_time'])
            # if len(output):
            #     for idx in output.index:
            #         cursor.execute(
            #             "DELETE FROM crime WHERE crime_id = ?", (output['crime_id'][idx],)
            #         )
            #         start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(output['time_stamp'][idx]))
            #         print(f'Arrest: crime ID: {output["crime_id"][idx]}, start time: {start_time}')

        finally:
            lock.release()
        connection.commit()
        time.sleep(1)


def stream_data():
    conn = create_connection('crime_events')

    c = conn.cursor()
    create_table(conn, c)

    new_crime_thread = Thread(target=new_crime_event, args=(conn, c))

    new_crime_thread.start()
    arrested_thread = Thread(target=criminal_arrested, args=(conn, c))
    arrested_thread.start()


def get_active_crime():
    try:
        lock.acquire(True)
        con = create_connection('crime_events')
        df = pd.read_sql_query("SELECT * FROM crime ", con)
        con.commit()
    finally:
        lock.release()
    return df


if __name__ == '__main__':
    stream_data()
    while True:
        print(get_active_crime())
        time.sleep(2)
