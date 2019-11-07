# -*- coding: utf-8 -*-
"""
Object-oriented wrappers for navite DB interfaces implementing common DB operations.

---
This file is part of Nifty python package. Copyright (c) by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
"""

import sqlite3
import psycopg2


#####################################################################################################################################################
#####
#####  DB - BASE CLASS
#####

class DB(object):
    
    db = None               # connection object of a native DB interface
    
    PLACEHOLDER = '?'       # placeholder string treated by native execute() as a place in a query where a parameter value should be pasted


    def __init__(self, *args, **kwargs):
        
        self.reconnect(*args, **kwargs)
        
    def reconnect(self, *args, **kwargs):
        
        self.db = self.connect(*args, **kwargs)
    
    def connect(self, *args, **kwargs):
        
        raise NotImplementedError
        
    def commit(self):
        
        self.db.commit()

    def _check_connected(self):
        
        if not self.db: raise Exception("Database not connected")


    def execute(self, query, args = [], commit = False):
        """For executing queries that don't return any result."""
        
        cur = self.db.cursor()
        cur.execute(query, args)
        cur.close()
        if commit: self.commit()


    def select(self, query, args = []):
        
        cur = self.db.cursor()
        cur.execute(query, args)
        for record in cur:
            yield record
        cur.close()


    def select_one(self, query, args = []):

        cur = self.db.cursor()
        cur.execute(query, args)
        rec = cur.fetchone()
        cur.close()
        return rec


    def count(self, table, where = None):

        query = "SELECT COUNT(*) FROM %s" % table
        if where:
            query += " WHERE %s" % where
            
        cur = self.db.cursor()
        cur.execute(query)
        
        count = None
        for row in cur: count = row[0]
        cur.close()
        
        return count
        

    def insert_row(self, table, row, attrs = None):
        """
        Insert a `row` (a tuple of values) to a `table`.
        """
        
        assert attrs is None or len(attrs) == len(row)
        
        row = tuple(row)        # convert a list to tuple if needed

        attrs_list = ""
        if attrs is not None:
            assert len(attrs) == len(row)
            attrs_list = "(%s)" % ','.join(attrs)

        placeholders = '(%s)' % ','.join([self.PLACEHOLDER] * len(row))

        query = "INSERT INTO %s %s VALUES (%s)" % (table, attrs_list, placeholders)

        self.execute(query, row)
        

    def insert_dict(self, table, record, attrs = None):
        """
        Insert a `record` (a dictionary of values) to a `table`.
        """
        
        if attrs is None:
            attrs = record.keys()
            # attrs = [atr for atr in record.keys() if not atr.startswith('__')]

        attrs_list   = "(%s)" % ','.join(attrs)
        placeholders = '(%s)' % ','.join([self.PLACEHOLDER] * len(values))

        values = tuple(record[atr] for atr in attrs)
        assert len(attrs) == len(values)

        query = "INSERT INTO %s %s VALUES %s" % (table, attrs_list, placeholders)

        self.execute(query, values)
        

    def insert_rows(self, table, rows, attrs = None):
        """
        Like insert_row(), but inserts a list of multiple rows (tuples) at once.
        """
        
        if not rows: return
        LEN = len(rows[0])
        
        assert all(len(row) == LEN for row in rows)
        values = sum(map(tuple, rows), tuple())

        attrs_list = ""
        if attrs is not None:
            assert len(attrs) == LEN
            attrs_list = "(%s)" % ','.join(attrs)

        row_placeholders = '(%s)' % ','.join([self.PLACEHOLDER] * LEN)
        all_placeholders = ','.join([row_placeholders] * len(rows))

        query = "INSERT INTO %s %s VALUES %s" % (table, attrs_list, all_placeholders)
        
        self.execute(query, values)


    def insert_dicts(self, table, records, attrs = None):
        """
        Like insert_dict(), but inserts a list of multiple records (dicts) at once.
        """
        
        if not records: return
        if attrs is None:
            attrs = records[0].keys()
        attrs_list = "(%s)" % ','.join(attrs)

        def extract(rec):
            return [rec[atr] for atr in attrs]

        values = sum(map(extract, records), [])

        row_placeholders = '(%s)' % ','.join([self.PLACEHOLDER] * len(attrs))
        all_placeholders = ','.join([row_placeholders] * len(records))

        query = "INSERT INTO %s %s VALUES %s" % (table, attrs_list, all_placeholders)
        
        self.execute(query, values)



#####################################################################################################################################################
#####
#####  SQLite3
#####

class SQLite3(DB):

    PLACEHOLDER = '?'

    def connect(self, *args, **kwargs):
        
        return sqlite3.connect(*args, **kwargs)
    
    
    def executescript(self, sql, commit = False):
        """SQLite's non-standard method to execute multiple SQL statements at once - execute() can't be used for this purpose."""
        
        self.db.executescript(sql)
        if commit: self.commit()


#####################################################################################################################################################
#####
#####  PostgreSQL
#####

class PostgreSQL(DB):
    
    PLACEHOLDER = '%s'
    
    def connect(self, *args, **kwargs):
        
        return psycopg2.connect(*args, **kwargs)
    
    
