import yaml
import datetime
from training import Training
import pandas as pd

FILENAME_DEFAULT = "experiments/tests.yaml"
FILENAME_CUSTOM = "experiments/tests_custom_loss.yaml"
FILENAME_DEBUG = "experiments/test_debug.yaml"

class Testing():
    def __init__(self):  
        pass        

    def run_tests(self, experiment_file="debug", experiment_parameters=None):
        if experiment_parameters == None:
            if experiment_file == "default":
                filename = FILENAME_DEFAULT
            elif experiment_file == "custom":
                filename = FILENAME_CUSTOM
            elif experiment_file == "debug":
                filename = FILENAME_DEBUG
            else:
                filename = experiment_file

            # create test dictionary
            with open(filename) as file:
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                tests = yaml.safe_load(file)
        else:
            tests = experiment_parameters

        # create a csv file with all test details
        test_dict = tests['Tests']
        df = pd.DataFrame(test_dict)
        df["RMSE"] = ""
        df['MPA'] = ""
        df['MDA'] = ""

        # Run experiments
        for i, test in enumerate(tests['Tests']):
            globals().update(test)
            trainer = Training(model)
            trainer.create_data_loaders(symbol, start_date, end_date, seq_len, batch_size)
            trainer.train(input_dim, hidden_dim, n_layers, output_dim, loss_funct, optimiser, learning_rate, epochs, test_no)
            y_pred, y_test, testScoreRMSE, testScoreMPA, testScoreMDA, = trainer.evaluate()
            df.loc[i,'RMSE'] = round(testScoreRMSE, 3)
            df.loc[i,'MPA'] = round(testScoreMPA, 3)
            df.loc[i,'MDA'] = round(testScoreMDA, 3)
            trainer.generate_results(y_pred, y_test, test_no, test)

        # save to csv file
        filename = 'results/' + 'test_table_2' + loss_funct + '_loss.csv'
        df.to_csv(filename)
        return df

tests = Testing()
df = tests.run_tests("experiments/tests-short.yaml")