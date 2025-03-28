import os

class DataProcessor:
    """
    A class to define the data processor.
    This is a base class that is inherited by the specific data processors: RawDataProcessor and SimDataProcessor.

    Args:
        - process_data_name (str): The name of the processed data.
        - processor_params (dict): The parameters for the data processor.
        - train_test_split (float, optional): Ratio of training to test data. Defaults to 0.8.
        - data_path (str, optional): The path to the raw data. Defaults to '/data'.

    Attributes:
        - process_data_name (str): The name of the processed data.
        - processor_params (dict): The parameters for the data processor.
        - train_test_split (float): Ratio of training to test data.
        - data_path (str): The path to the raw data.
        - processed_data (pd.DataFrame): The processed data.
    """

    def __init__(self,
                 processed_data_name: str,
                 processor_params: dict,
                 train_test_split: float = 0.8,
                 data_path='/data'):

        """
        Initialises the DataProcessor class.

        Args:
            - process_data_name (str): The name of the processed data.
            - processor_params (dict): The parameters for the data processor.
            - train_test_split (float, optional): Ratio of training to test data. Defaults to 0.8.
            - data_path (str, optional): The path to the raw data. Defaults to '/data'.
        """
        self.processed_data_name = processed_data_name
        self.processor_params = processor_params
        self.train_test_split = train_test_split
        self.data_path = os.getcwd() + data_path
        self.processed_data = None

        if (not isinstance(self.train_test_split, float) or
                self.train_test_split < 0 or self.train_test_split > 1):
            raise ValueError('DataProcessors - train_test_split must be a float between 0 and 1')

    def process_data(self):
        """
        Specified by the subclass.
        """
        pass

    def get_construction(self):
        """
        Specified by the subclass.
        """
        pass
