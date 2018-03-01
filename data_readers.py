import pandas as pd
import data_utils as du
import numpy as np
import datetime

class DataReaders():
    def __init__(self, context, w_init, implicit, m_name):
        self.context = context
        self.w_init = w_init
        self.implicit = implicit
        self.c = None
        self.weights = [1., 1.]
        self.m_name = m_name

        if w_init == 'all-diff' or w_init == 'c-diff':
            self.weights.append(0.1)
        else:
            self.weights.append(1)

        self.cols = ['UserId', 'ItemId']

    def get_movieLens1M(self):
        cols = ['UserId', 'ItemId', 'Rating', 'Timestamp']
        train = pd.read_csv('../data/movielens/ml-1m/train75.dat',sep='::', names=cols)
        test = pd.read_csv('../data/movielens/ml-1m/test25.dat',sep='::', names=cols)

        itemIds_neg = np.random.randint(np.max(train.ItemId),size=len(train.ItemId))
        train_neg = pd.DataFrame.from_items([('UserId', train.UserId), ('ItemId', itemIds_neg)])
        
        # adding context
        if self.context is True:
            movies = pd.read_csv('../data/movielens/ml-1m/movies.dat',sep='::', names=['ItemId','Name','Genres'])
            movies['Genre'] = movies['Genres'].apply(lambda x: x.split('|')[0])
            train = pd.merge(train, movies[['ItemId','Genre']], on='ItemId')
            test = pd.merge(test, movies[['ItemId','Genre']], on='ItemId')
            self.cols.append('Genre')

        if self.m_name == 'wmf':
            self.c = train.Rating
            train.Rating = train.Rating.apply(lambda x: 1 if x >= 3 else 0)

        return train, test, train_neg

    def get_MSD_T50(self):
        train = pd.read_csv('../data/MSD/MSD_w_Genre_T50_train_listen_group.csv')
        test = pd.read_csv('../data/MSD/MSD_w_Genre_T50_test_listen_group.csv')

        itemIds_neg = np.random.choice(train.ItemId, len(train.ItemId))
        train_neg = pd.DataFrame.from_items([('UserId', train.UserId), ('ItemId', itemIds_neg)])

        if self.context is True:
            self.cols.append('Genre')

        if self.m_name == 'wmf':
            train = self.explicit_binary(train, train_neg)
            self.c = train.Listens.apply(lambda x: 100. if x > 100 else x)/100.
            
        return train, test, train_neg

    def get_MSD_T20(self):
        train = pd.read_csv('../data/MSD/MSD_cut20_train.csv')
        test = pd.read_csv('../data/MSD/MSD_cut20_test.csv')

        itemIds_neg = np.random.choice(train.ItemId, len(train.ItemId))
        train_neg = pd.DataFrame.from_items([('UserId', train.UserId), ('ItemId', itemIds_neg)])

        if self.context is True:
            self.cols.append('Genre')

        if self.m_name == 'wmf':
            train = self.explicit_binary(train, train_neg)
            self.c = train.Listens.apply(lambda x: 100. if x > 100 else x)/100.
            
        return train, test, train_neg

    def get_goodbooks(self):
        cols = ['UserId', 'ItemId', 'Rating']
        train = pd.read_csv('../data/goodbooks/ratings_train.csv',sep=',', skiprows=1, names=cols)
        test = pd.read_csv('../data/goodbooks/ratings_test.csv',sep=',', skiprows=1, names=cols)

        itemIds_neg = np.random.randint(np.max(train.ItemId),size=len(train.ItemId))
        train_neg = pd.DataFrame.from_items([('UserId', train.UserId), ('ItemId', itemIds_neg)])
    
        if self.context is True:
            b = pd.read_csv('../data/goodbooks/books.csv')
            bt = pd.read_csv('../data/goodbooks/book_tags.csv')
            tag_per_book = bt.groupby('goodreads_book_id').tag_id.apply(lambda x: list(x)[1])
            tag_per_book_df = pd.DataFrame({'goodreads_book_id':tag_per_book.index, 'tag_id':tag_per_book.values})
            tg = pd.merge(tag_per_book_df, b[['goodreads_book_id','book_id']], on='goodreads_book_id')

            train = pd.merge(train, tg[['book_id','tag_id']], left_on=['ItemId'], right_on=['book_id'])
            test = pd.merge(test, tg[['book_id','tag_id']], left_on=['ItemId'], right_on=['book_id'])

            self.cols.append('tag_id')

        return train, test, train_neg
    
    def get_frappe(self):
        cols = ['UserId','ItemId','cnt','daytime','weekday','isweekend','homework','cost','weather','country','city']
        train = pd.read_csv('../data/frappe/train75.csv',sep='\t', names=cols,skiprows=1)
        test = pd.read_csv('../data/frappe/test25.csv',sep='\t', names=cols,skiprows=1)
    
        itemIds_neg = np.random.randint(np.max(train.ItemId),size=len(train.ItemId))
        train_neg = pd.DataFrame.from_items([('UserId', train.UserId), ('ItemId', itemIds_neg)])
        
        if self.context is True:
            self.cols.append('isweekend')
            self.cols.append('homework')

            if self.w_init == 'all-diff' or self.w_init == 'c-diff':
                self.weights.append(0.1)
            else:
                self.weights.append(1)

        if self.m_name == 'wmf':
            train = self.explicit_binary(train, train_neg)
            self.c = train.cnt.apply(lambda x: 100. if x > 100 else x)/100.
    
        return train, test, train_neg

    def get_kassandr(self):
        cols = ['UserId', 'ItemId', 'CountryCode', 'Category', 'Merchant', 'Date', 'Rating']
        train = pd.read_csv('../data/Kasandr/de/train_de.csv',sep='\t', names=cols, skiprows=1)
        train = train[train.Rating == 1]
        test = pd.read_csv('../data/Kasandr/de/test_de.csv',sep='\t', names=cols, skiprows=1)
        test = test[test.Rating == 1]
    
        itemIds_neg = np.random.choice(train.ItemId, len(train.ItemId))
        train_neg = pd.DataFrame.from_items([('UserId', train.UserId), ('ItemId', itemIds_neg)])
    
        if self.context is True:
            train['IsWeekend'] = train.Date.apply(lambda x: 1 if datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").weekday() >= 5 else 0)
            test['IsWeekend'] = test.Date.apply(lambda x: 1 if datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").weekday() >= 5 else 0)
            self.cols.append('IsWeekend')

        return train, test, train_neg

    def get_xing(self):
        cols = ['UserId','ItemId','Level']
        train = pd.read_csv('../data/xing/20/all-pos-train.csv',sep=',', names=cols,skiprows=1)
        test = pd.read_csv('../data/xing/20/all-pos-test.csv',sep=',', names=cols,skiprows=1)
    
        itemIds_neg = np.random.randint(np.max(train.ItemId),size=len(train.ItemId))
        train_neg = pd.DataFrame.from_items([('UserId', train.UserId), ('ItemId', itemIds_neg)])
    
        return train, test, train_neg
  
    def explicit_binary(self, train, train_neg):
        train['Rating'] = 1
        train_neg['Rating'] = 0
        s = pd.concat([train, train_neg], ignore_index=True)
        s = s.sample(frac=1).reset_index(drop=True)
    
        return s.fillna(1)