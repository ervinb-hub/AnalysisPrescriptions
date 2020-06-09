# Generic imports
import pyspark
import re
import json
import numpy as np
import logging
import os

# Spark mllib imports
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import RegressionMetrics

# User defined imports
from bnf import parse_n_save


def get_logger():
    frmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler('cluster_execution.log')
    fh.setFormatter(frmt)
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    return log


def to_Dicts(rows, cols, sep):
    """
    Reads a csv formated line and provides a dictionary as output.
    @param rows: The row to be processed.
    @param cols: A list with the columns headers previously extracted.
    @param sep: The separator used to divide several fields.
    @return: A dictionary with the header names and their corresponding values.
    """
    res = {}
    rows = rows.split(sep)
    for i, j in zip(cols, rows):
        j = re.sub('[\'"]', '', j.strip())
        res.update({i: j})
    return res


def parse_val(row, fx, name):
    """
    Parses a single named value from the PDPI file. It also removes white spaces and and 
    applies a function to the value.
    @param row: a single row in csv format.
    @param fx: a function to be applied to a value.
    @param name: The (column header) name to identity the field.
    @return: A transformed value having a suitable datatype.
    """
    row[name] = re.sub(' ', '', row[name])
    if (row[name].lower() != name):
        return fx(row[name])
    return fx(0)


def parse_header(row):
    """
    Parses a header file from a csv format and extracts the column names.
    @param row: The header row to be parsed.
    @return: A list containing the 'cleaned' field names.
    """
    res = []
    for col in row.split(','):
        col = col.strip().lower()
        col = re.sub('[ \'"]', '', col)
        res.append(col)
    return res


def season(date):
    """
    Determines the season given the month.
    @param date: The date to be parsed (its month component)
    @return: The season of the year
    """
    if (len(date) != 6):
        return None

    month = int(date[-2:])
    if (month in [12, 1, 2]):
        return 'Winter'
    elif (month in [3, 4, 5]):
        return 'Spring'
    elif (month in [6, 7, 8]):
        return 'Summer'
    elif (month in [9, 10, 11]):
        return 'Autumn'
    else:
        return None


def encode(rdd, idx):
    """
    For each categorical variable encode a category to an ordinal number
    @param rdd: The data rdd
    @param idx: The positional column index
    @return: A map storing the categories for each field
    """
    result = (rdd
              .map(lambda x: x[idx])
              .distinct()
              .zipWithIndex()
              .collectAsMap()
              )
    return result


def get_label(record):
    """
    Simply extraxts the last column which contains the labels
    """
    return float(record[-1])


def get_features_trees(mappings, record):
    """
    Important. Categorical variables need to be represented by numbers to avoid
    ValueError: could not convert string to float: 'abc'. The constructor of LabeledPoint will create
    numpy arrays given an observation in input, the function returns the numeric code for each category.
    @param record: A single observation
    @return: A new observation where the nominal variables are represented by their mapping value
    """
    result = []
    idx = 0
    for i in record[1:7]:
        try:
            result.append(float(i))
        except ValueError:
            result.append(float(mappings[idx][i]))
            idx = idx + 1
    return np.array(result)


def read_prescriptions(spark_context, pattern):
    """
    Reads and prepares a set of csv files into an RDD with the right data types
    @param pattern: Expects a string indicating the path with wildcards e.g: "./data/T*PDPI*.CSV"
    @return: A clean RDD reppresenting PDPI data.
    """
    pdpi_lines = spark_context.textFile(pattern)
    first_row = pdpi_lines.first()
    headers = parse_header(first_row)

    return (pdpi_lines
            .filter(lambda p: p != first_row)
            .map(lambda p: to_Dicts(p, headers, ','))
            .map(lambda p: (p['sha'], p['practice'], p['bnfcode'][:9],
                            parse_val(p, int, 'items'),
                            parse_val(p, float, 'nic'),
                            parse_val(p, float, 'actcost'),
                            parse_val(p, int, 'quantity'),
                            p['period'])
                 )
            )


def read_addresses(spark_context, pattern, col_names):
    """
    Reads and prepares a set of csv files into an RDD with the right data types
    @param pattern: Expects a string indicating the path with wildcards.
    @return: A clean RDD reppresenting ADDRESSES' data.
    """
    addr_lines = spark_context.textFile(pattern)

    return (addr_lines
            .map(lambda a: to_Dicts(a, col_names, ','))
            .filter(lambda a: a['county'] != '')
            .map(lambda a: (a['date'], a['practice_code'], a['county']))
            )


def read_chemicals(spark_context, pattern):
    """
    Reads and prepares a set of csv files into an RDD with the right data types
    @param pattern: Expects a string indicating the path with wildcards.
    @return: A clean RDD reppresenting CHEMICALS' data.
    """
    chem_lines = spark_context.textFile(pattern)
    header = chem_lines.first()

    return (chem_lines
            .filter(lambda x: x != header)
            .map(lambda x: tuple(y.strip() for y in x.split(',')[:-1]))
            .distinct()
            )


class RandomForestModel():
    """
    Class representing a model built with the algorithm RandomForest.

    Attributes
    ----------
    train_data: The dataset used for training
    test_data: The dataset used for testing
    trees: A list containing a few number of trees used to tune the model
    depths: A list containing a few tree depths used to tune the model
    """

    def __init__(self, train_data, test_data, trees, depths):
        self._train_data = train_data
        self._test_data = test_data
        self._trees = trees
        self._depths = depths
        #self._log = logging.getLogger(self.__class__.__name__)
        self._log = get_logger(self.__class__.__name__)

    def find_rf_parameters(self):
        """
        Iterates through a set of numbers corresponding to numTrees and maxDepth to
        search for the optimal hyperparameters.
        @return: The best hyperparameters, that minimise MSE
        """
        min_error = 99999999
        num_trees = self._trees[0]
        depth = self._depths[0]
        for i in self._trees:
            for j in self._depths:
                rf_model = RandomForest.trainRegressor(self._train_data,
                                                       categoricalFeaturesInfo={
                                                           3: 153, 4: 4, 5: 80},
                                                       numTrees=i,
                                                       featureSubsetStrategy="auto",
                                                       impurity="variance",
                                                       maxDepth=j,
                                                       maxBins=54)
                predictions = rf_model.predict(
                    self._train_data.map(lambda x: x.features))
                target_train = self._train_data.map(lambda p: p.label)
                rf_values = target_train.zip(
                    predictions.map(lambda x: float(x)))
                metrics_rf = RegressionMetrics(rf_values)
                mse = metrics_rf.meanSquaredError
                if (mse < min_error):
                    min_error = mse
                    num_trees = i
                    depth = j

        self._log.info('Estimating Parameters for Random Forests:\n=====')
        self._log.info('MSE = {}, trees = {}, depth = {}'.format(
            min_error, num_trees, depth))
        return {'trees': num_trees, 'depth': depth}

    def train(self):
        """
        Trains the Random Forest model with the optimal parameters.
        @return: The trained RF model
        """
        target_test = self._test_data.map(lambda p: p.label)
        hyper_params = self.find_rf_parameters()
        rf_model = RandomForest.trainRegressor(self._train_data, categoricalFeaturesInfo={},
                                               numTrees=hyper_params['trees'],
                                               featureSubsetStrategy="auto",
                                               impurity="variance",
                                               maxDepth=hyper_params['depth'],
                                               maxBins=54)
        return rf_model

    def predict(self, test_data):
        """
        Predicts the outcome of an unseen data record
        @test_data: Test data or more generally speaking -- unseen data
        @return: The predicted values
        """
        rf_model = self.train()
        self._log.info('Number of trees: {}'.format(str(rf_model.numTrees())))
        self._log.info('Number of nodes: {}'.format(
            str(rf_model.totalNumNodes())))
        predictions = rf_model.predict(test_data.map(lambda x: x.features))
        return predictions

    def evaluate_prediction(self, test_data, test_target):
        """
        Evaluates the performance of the prediction model by printing out MSE
        and other model related data.
        @test_data: The test data
        @test_target: The labels of the test data.
        @return: Each predicted value in pair with the true value
        """
        predictions = self.predict(test_data)
        rf_predicted_values = test_target.zip(
            predictions.map(lambda x: float(x)))
        metrics_rf = RegressionMetrics(rf_predicted_values)
        self._log.info('Random Forest predictions: {}'.format(
            str(rf_predicted_values.take(5))))
        self._log.info('TestSet MSE = {}'.format(metrics_rf.meanSquaredError))
        return rf_predicted_values


def main():
    log = get_logger(__name__)
    sc = pyspark.SparkContext('local[*]')

    # Parse and saves BNF sections
    if not os.path.exists('./sections.json'):
        log.info('Downloading BNF data')
        parse_n_save()

    # Reads and parallelizes BNF sections' file
    with open('sections.json', 'r') as file:
        content = file.read()
    sections = json.loads(content)
    sections = [tuple(x.values()) for x in sections]
    bnf = sc.parallelize(sections)

    # Reads the data from their respective files
    pdpi = read_prescriptions(sc, "./data/T*PDPI*.CSV")
    addr_cols = ['date', 'practice_code', 'name',
                 'first_line_addr', 'street', 'city', 'county', 'postcode']
    addrs = read_addresses(sc, "./data/T*ADDR*.CSV", addr_cols)
    chem = read_chemicals(sc, "./data/T*CHEM*.CSV")

    # Example ('Q44', 'Y04937', '0204000R0', 2, 2.01, 2.08, 63, '201611')
    pdpi_join_chem = (pdpi
                      .map(lambda x: x[1:4] + x[5:])
                      .keyBy(lambda x: x[1])
                      .join(chem)
                      .values()
                      .map(lambda x: x[0] + (x[1],))
                      .keyBy(lambda x: (x[0], x[5]))
                      )

    # Example ('201612', 'A81001', 'CLEVELAND')
    addrs = addrs.map(lambda x: ((x[1], x[0]), (x[2])))

    pdpi_addr = (pdpi_join_chem
                 .join(addrs)
                 .mapValues(lambda x: x[0] + (x[1], ))
                 .values()
                 .keyBy(lambda x: x[1][:7].rstrip('0'))
                 )

    # Example ('01', 'Gastro-Intestinal System')
    bnf_codes = bnf.filter(lambda x: len(x[0]) >= 6)

    df_joined = (pdpi_addr
                 .join(bnf_codes)
                 .mapValues(lambda x: x[0] + (x[1], 1))
                 .values()
                 .map(lambda x: x[:5] + x[6:])
                 .keyBy(lambda x: (x[1], x[6]))
                 .map(lambda x: (x[0], x[1][1:]))
                 .reduceByKey(lambda x, y: (x[0], x[1] + y[1],
                                            round(x[2] + y[2], 3), x[3] + y[3], x[4], x[5], x[6], x[7] + y[7]))
                 .sortBy(lambda x: (x[0][1], x[1][7]), ascending=False)
                 .persist()
                 )
    log.info('\nQuery1, top 10 rows:\n=====')
    log.info(df_joined.take(10))

    most_used_chem = (df_joined
                      .values()
                      .map(lambda x: x[0:])
                      .keyBy(lambda x: x[6])
                      .reduceByKey(lambda x, y: (x if x[7] > y[7] else y))
                      )
    log.info('\nQuery2, top 10 rows:\n=====')
    log.info(most_used_chem.take(10))

    chem_per_seas = (pdpi_addr
                     .join(bnf_codes)
                     .mapValues(lambda x: x[0] + (x[1], 1))
                     .values()
                     .map(lambda x: (x[1], x[3], season(x[5]), x[6], x[8], x[9]))
                     .keyBy(lambda x: (x[0], x[2]))
                     .reduceByKey(lambda x, y: (x[0], round(x[1] + y[1], 3), x[2], x[3], x[4], x[5] + y[5]))
                     .sortBy(lambda x: (x[1][2], x[1][5]), ascending=False)
                     )

    log.info('\nQuery3, top 10 rows:\n=====')
    log.info(chem_per_seas.take(10))

    df = (df_joined
          .values()
          .zipWithUniqueId()
          .map(lambda x: ((x[1],) + x[0][1:]))
          )

    original_data = df.first()
    log.info('\nOriginal data example (df_joined) ready for ML:\n=====')
    log.info(original_data)

    mappings = []
    for i in range(4, 7):
        mappings.append(encode(df, i))

    # Split the dataset into train/test set
    train_set, test_set = df.randomSplit([0.8, 0.2], seed=123)

    log.info('TrainSet = {}, TestSet = {}\n'.format(
        train_set.count(), test_set.count()))
    log.info("The dataset has:")
    for i, j in zip(mappings, ['bnf_code', 'county', 'usage']):
        log.info('{}: {} distinct categories'.format(j, len(i)))

    # Extracting features for Decision Trees
    tree_train_data = train_set.map(
        lambda r: LabeledPoint(get_label(r), get_features_trees(mappings, r)))
    tree_test_data = test_set.map(lambda r: LabeledPoint(
        get_label(r), get_features_trees(mappings, r)))

    # Training Random Forests Model
    ntrees = [2, 6, 10, 20]
    depths = [3, 6, 10]

    rfm = RandomForestModel(tree_train_data, tree_test_data, ntrees, depths)
    target_test = tree_test_data.map(lambda p: p.label)
    result = rfm.evaluate_prediction(tree_test_data, target_test)


if __name__ == '__main__':
    main()
