import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as data

class StockData(data.Dataset):
    """
    Stock dataset class to load data and convert them to datasets
    """

    def __init__(self, seq_len, stage, symbol, start_date, end_date):
        """
        Args:
            seq_len ([type]): Sequence Lenght
            stage ([type]): Trainin, validation or test identifier
            symbol ([type]): NASDAQ stock symbol
            start_date ([type]): start date of data
            end_date ([type]): end date of data
        """
        self.seq_len = seq_len
        self.stage = stage
        self.stock = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.load_stock_data()

    def __len__(self):
        """
        Magic method for dataset to return lengh.
        Since it is returned to sequence it is shorter then total len of dataframe
        """
        return len(self.data) - self.seq_len

    def load_stock_data(self):
        """
        This method loads data from CSV file and filters it according to start and end date
        It also applier missing data handling by filling NA values.
        It splits data to train, val and test splits

        Returns:
            [type]: [description]
        """
        # build filter dataframe
        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        df = pd.DataFrame(index=dates)

        # read stock data to pandas dataframe , we only need closing values and date as index column.
        stock = pd.read_csv('stocks/{}.csv'.format(self.stock),usecols=['Date','Close'], header=0, index_col="Date", parse_dates=True, na_values=["nan"])
        stock.columns = [self.stock]

        stock = df.join(stock)

        # NaN value handlind
        stock = stock.fillna(method='ffill')
        # incase the first element is null, then we can ues the backfill
        stock = stock.fillna(method='bfill')

        self.scaler = StandardScaler()
        stock[self.stock] = self.scaler.fit_transform(stock[self.stock].values.reshape(-1,1))
        # Convert dataframe to tensor
        stock = torch.tensor(stock[self.stock].values.reshape(-1,1))

        # splitting dataset into training, validation and test by %70, %10, and %20
        number_of_records = stock.shape[0]
        training_size = int(0.7 * number_of_records )
        validation_size = int( 0.1 * number_of_records)
        test_size = number_of_records - ( training_size + validation_size)

        #process the stage
        if self.stage == 'train':
            stock = stock[:training_size,:]
        elif self.stage == 'val':
            stock = stock[training_size:training_size+validation_size,:]
        else:
            stock = stock[-test_size:,:]

        return stock

    def __getitem__(self, idx):
        """
        magic method to return iterated item in training loop.
        """
        X = self.data[idx:idx+self.seq_len].float()
        y = self.data[idx+self.seq_len].float()
        return idx, X, y
