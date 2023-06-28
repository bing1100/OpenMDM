from mdmconfig import MDMConfig
from mdmalgorithms import MdmMatcher
import numpy as np
import logging
from imblearn.over_sampling import SMOTE
from datetime import datetime
import yaml
import recordlinkage
from recordlinkage import datasets
import pandas as pd
from sklearn.model_selection import train_test_split


def block(data, links, blocks):
    """
    Block the dataset.
    :param data: pandas dataframe
    :param links: pandas multiindex
    :param blocks: list of fields to block on
    :return: pandas dataframe
    """
    indexer = recordlinkage.Index()
    for block in blocks:
        indexer.block(block)
    if isinstance(data, tuple):
        pairs = indexer.index(*data)
    else:
        pairs = indexer.index(data)
    return pairs


def translate_fields(data, field_yaml):
    """
    Translate fields in the dataset.
    :param data: pandas dataframe
    :param field_yaml: yaml file containing field translations
    :return: pandas dataframe
    """
    with open(field_yaml, 'r') as stream:
        dct = yaml.safe_load(stream)
    result = data.rename(columns=dct)
    return result


class Dataset:
    """
    Loads data from febrl, krebs, and/or smile csv files.
    Fills missing values with simulated values.
    Does blocking
    Does comparison
    """
    def __init__(self, mdm_json: MDMConfig):
        self.records = None
        self.data = None
        self.links = None
        self.annotations = pd.DataFrame(columns='id1 id2 label sampling_strategy time_stamp'.split())
        self.blocks = mdm_json.blocking_fields
        self.mdm_algorithms = mdm_json.mdm_algorithms
        self.datatype = []

    def use_mdm_matcher(self, data, pairs):
        # convert the two dataframes to a list of tuples of dictionaries based on pairs
        if isinstance(data, tuple):
            left = data[0].loc[pairs.get_level_values(0)]
            right = data[1].loc[pairs.get_level_values(1)]
        else:
            left = data.loc[pairs.get_level_values(0)]
            right = data.loc[pairs.get_level_values(1)]
        records = list(zip(left.to_dict('records'), right.to_dict('records')))

        matcher = MdmMatcher(self.mdm_algorithms)
        features = matcher.predictPairs(records)

        # return the resulting dictionary of lists to a dataframe
        features_df = pd.DataFrame.from_dict(features, orient='columns')
        features_df.index = pairs

        return features_df

    def _save_new_data(self, features_df, links, data):
        """
        Save the new data to the dataset.
        :param features_df: pandas dataframe of features
        :param links: pandas dataframe of links
        :return: None
        """
        if self.records is None:
            self.records = data
        else:
            self.records = pd.concat([self.records, data])

        if self.data is None:
            self.data = features_df
            self.links = links
        else:
            self.data = pd.concat([self.data, features_df])
            self.links = self.links.union(links)

    def get_data(self):
        return self.records, self.data, self.links

    def get_data_np(self, columns=None):
        """
        Get the data as a numpy array.
        :param columns: list of columns to get in order (default: all columns, unspecified order)
        :return: feature names, data, labels
        """
        if columns is not None:
            data = self.data[columns].to_numpy()
        else:
            logging.warning('No columns specified, returning all columns in unspecified order.')
            data = self.data.to_numpy()
        labels = self.data.index.isin(self.links)
        return self.data.columns, data, labels

    def shuffle(self):
        """
        Shuffle the data.
        :return: None
        """
        if self.data is not None:
            idx = np.arange(self.data.shape[0])
            np.random.shuffle(idx)
            self.data = self.data[idx]
            self.links = self.links[idx]

    def add_annotation(self, ids, label, sampling_strategy):
        """
        Add a row to the annotation dataframe with this annotation in it.
        :param ids: index of the row to annotate (will be two ids for a pair)
        :param label: Boolean
        :param sampling_strategy: string, the sampling strategy used
        :return: None
        """
        self.annotations = self.annotations.append(
            {
                'id1': ids[0],
                'id2': ids[1],
                'label': label,
                'sampling_strategy': sampling_strategy,
                'time_stamp': datetime.now()
            },
            ignore_index=True
        )

    def get_annotations(self, active=True):
        """
        Get the annotations from the annotation dataframe.
        :param active: Boolean, if True only return the active annotations
        :return: pandas dataframe of annotations
        """
        if active:
            return self.annotations.sort_values(by="time_stamp").drop_duplicates(subset=["id1", "id2"], keep="last")

        else:
            return self.annotations

    def split_(self, ratio=None, oversample=False, columns=None):
        if ratio is None:
            ratio = [0.1, 0.1]
        features, X, y = self.get_data_np(columns=columns)
        if oversample:
            sm = SMOTE(sampling_strategy=0.5)
            sample, _ = sm.fit_resample([[idx] for idx in range(len(X))], self.datatype)
            X = [X[idx] for [idx] in sample]
            y = [y[idx] for [idx] in sample]
        # Ratio passed as list
        # ratio = [test_ratio, val_ ratio]
        test_ratio = ratio[0]
        val_ratio = ratio[1]
        # First split the test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)

        # Split train again to generate validate data
        val_ratio = val_ratio / (1 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio)

        return features, X_train, X_test, X_val, y_train, y_test, y_val


class FebrlDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_febrl(self, version=1, translate_yaml=None):
        """
        Load the FEBRL1, FEBRL2, or FEBRL3 dataset.
        :param blocks: list of fields to block on
        :param fields: list of fields to compare
        :param version: int, version of the dataset to load (1, 2, or 3)
        :param fill: list of fields to fill with simulated values
        :return: None
        """
        versions = {1: datasets.load_febrl1, 2: datasets.load_febrl2, 3: datasets.load_febrl3}
        data, links = versions[version](True)
        data = data.astype({'street_number': float, 'postcode': float, 'date_of_birth': float, 'soc_sec_id': float})

        if translate_yaml:
            data = translate_fields(data, translate_yaml)

        pairs = block(data, links, self.blocks)

        features_df = self.use_mdm_matcher(data, pairs)

        self._save_new_data(features_df, links, data)
        self.datatype.extend([0] * len(features_df.index))


class KrebsDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_krebs(self):
        """
        Load the Krebs dataset.
        We can't block or fill because the dataset is already blocked and compared.
        :return: None
        """
        features_df, pairs = datasets.load_krebsregister()
        self._save_new_data(features_df, pairs)


class WalmartAmazonDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_walmart_amazon(self, data_path):
        """
        Load the Amazon-Google dataset.
        :param data_path: path to the Amazon-Google dataset
        :param links_path: path to the Amazon-Google links
        :return: None
        """
        data1 = pd.read_csv(data_path + '/tableA.csv', index_col='id')
        data2 = pd.read_csv(data_path + '/tableB.csv', index_col='id')
        links = pd.read_csv(data_path + '/test.csv', index_col=['ltable_id', 'rtable_id'])
        links = pd.concat([links, pd.read_csv(data_path + '/train.csv', index_col=['ltable_id', 'rtable_id'])])
        links = pd.concat([links, pd.read_csv(data_path + '/valid.csv', index_col=['ltable_id', 'rtable_id'])])
        links = links[links.label == 1].index
        pairs = block((data1, data2), links, self.blocks)
        features_df = self.use_mdm_matcher((data1, data2), pairs)
        data1.index = 'amazon/' + data1.index.astype(str)
        data2.index = 'walmart/' + data2.index.astype(str)
        data = pd.concat([data1, data2])
        links = links.set_levels(['amazon/' + links.levels[0].astype(str), 'walmart/' + links.levels[1].astype(str)], level=[0, 1])
        features_df.index = features_df.index.set_levels(['amazon/' + features_df.index.levels[0].astype(str), 'walmart/' + features_df.index.levels[1].astype(str)], level=[0, 1])
        self._save_new_data(features_df, links, data)
        self.datatype.extend([1]*len(features_df.index))


class CustomFebrlDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_custom_febrl(self, data_path, links_path):
        """
        Laod the custom febrl dataset
        :param data_path: path to the febrl dataset
        :param links_path: path to the febrl links
        :return: None
        """
        data = pd.read_csv(data_path, index_col='rec_id', skipinitialspace=True)
        links = pd.read_csv(links_path, index_col=[0, 1], header=None).index
        data = data.astype({'street_number': float, 'postcode': float, 'date_of_birth': float, 'soc_sec_id': float})
        data = translate_fields(data, './dataset_data/febrl_field_names.yaml')
        pairs = block(data, links, self.blocks)
        features_df = self.use_mdm_matcher(data, pairs)
        self._save_new_data(features_df, links)


class CustomDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_smile(self, data_path, links_path):
        """
        Load the SMILE dataset.
        :param data_path: path to the SMILE dataset
        :param links_path: path to the SMILE links
        :return: None
        """
        data = pd.read_csv(data_path, index_col='ID')
        links = pd.read_csv(links_path, index_col=[0, 1], header=None).index
        pairs = block(data, links, self.blocks)
        features_df = self.use_mdm_matcher(data, pairs)
        self._save_new_data(features_df, links, data)
        self.datatype.extend([1]*len(features_df.index))


