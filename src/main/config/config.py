########################################
######### Preprocessing Config #########
########################################

# DATASET_DIR = "src/data/"

# TRAINED_MODEL_DIR = "src/trained_models/"

FEATURES_TO_DROP = ['Unnamed: 2',
                    'Unnamed: 3',
                    'Unnamed: 4']

FEATURES_TO_RENAME = {'v1':'labels', 'v2':'data'}

LABEL_MAPPING = {'ham':0, 'spam':1}

NUM_WORDS = 2000

MAXLEN = 189

FEATURES=

TARGET=

########################################
############# Train Config #############
########################################


########################################
############# Predict Config ###########
########################################
