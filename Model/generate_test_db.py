from Model.dataset import *
from common import *
import os

os.chdir('..')

db_type = DBType.Validation

db_config = ConfigParser('./dbs_config.yaml')
config = ConfigParser('./Model/config.yaml')

dataset = Dataset(db_config, config, db_type, True, False, gen_mode=True)
dataset.generate_test_db()
