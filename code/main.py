import argparse
import pandas as pd
import logging
import time
from regression import testRBFSVR as RBFSVR
from regression import testLinearSVR as LinearSVR
from regression import testLinear as Linear
from regression import testPOLYSVR as POLYSVR
from regression import testLogistic as Logistic
from regression import testMLP as MLP
# from regression import testCNN as CNN

print('start at', time.ctime())
st = time.time()

parser = argparse.ArgumentParser(description='Specificity Reg')

# parser.add_argument('-train', type=str, default='training', help='Name of training file')
# parser.add_argument('-valid', type=str, default='validation', help='Name of validation file')
# parser.add_argument('-test', type=str, default='testing', help='Name of test file')
parser.add_argument('-all', type=str, default='all', help='Name of the combined file for training, validation and testing')
parser.add_argument('-test_or_not', type=bool, default=False, help='Test or not')
parser.add_argument('-train_target', type=str, default='training_target', help='Name of training_target file')
parser.add_argument('-valid_target', type=str, default='validation_target', help='Name of validation_target file')
parser.add_argument('-test_target', type=str, default='testing_target', help='Name of test_target file')
parser.add_argument('-output', type=str, default='RBF_BEST_EXP_output', help='Name of output file')
parser.add_argument('-model', type=str, default='RBFSVR', help='Regression model [LinearSVR, RBFSVR]')
parser.add_argument('-features', type=str, default='All', help='Group of features to be used [All, Lexical, word, tweet, emotion]')
parser.add_argument('-exclude', type=str, default='DEMO Tweet', help='Features to be excluded')
parser.add_argument('-c', type=str, default=False, help='penalty factor')
parser.add_argument('-gamma', type=str, default=False, help='gamma factor if model is RBF')

args = parser.parse_args()

# ALL = range(5, 233)
ALL = range(3, 232)
# LEXICAL = range(5, 27)
LEXICAL = list(range(3, 25)) + list(range(228, 232)) # range(3, 25) + range(228, 232)
ALL_LEXICAL = range(25, 228)
# TWEET = range(227, 230)
# EMOTION = range(230, 233)
EMOTION = range(225, 228)
ALL_EMOTION = list(range(3, 225)) + list(range(228, 232))
PAR = range(228, 229)
# ALL_PAR = list(range(3, 11)) + list(range(12, 228)) + list(range(229, 232))
ALL_PAR = list(range(3, 228)) + list(range(229, 232))
NON_PAR = range(229, 230)
# ALL_NON_PAR = list(range(3, 11)) + list(range(12, 229)) + list(range(230, 232))
ALL_NON_PAR = list(range(3, 229)) + list(range(230, 232))
DEIXIS = range(230, 232)
ALL_DEIXIS = range(3, 230)
JJ = range(11, 12)
# ALL_JJ = list(range(3, 11)) + list(range(12, 228)) + list(range(230, 232))
ALL_JJ = list(range(3, 11)) + list(range(12, 232))
# DEMO = range(233, 239)
# SIG_DEMO = [234, 235]
# EMBEDDING = range(27, 127)
EMBEDDING = range(25, 125)
# WORDREP = range(27, 227)
WORDREP = range(25, 225)
ALL_WORDREP = list(range(3, 25)) + list(range(225, 232))
# length = [6]
average = [3]
length = [4]
# URL = [227]
# CONCRETE = [226]
CONCRETE = [224]
# USER = [228, 229]
# TEMP = [239, 240, 241]


FACTOR = 1

models = {'LINEARSVR': LinearSVR, 'RBFSVR': RBFSVR, 'LINEAR': Linear, 'LOGISTIC': Logistic, 'POLYSVR': POLYSVR, 'MLP': MLP} # , 'CNN': CNN}
'''
features_group = {'ALL': ALL, 'LEXICAL': LEXICAL, 'WORD': WORDREP, 'TWEET': TWEET, 'EMOTION': EMOTION, 'AVERAGE' : average, 'LENGTH': length,
                  'EMBEDDING': EMBEDDING, 'URL': URL, 'USER': USER, 'DEMO': DEMO, 'CONCRETE': CONCRETE, 'TEMP': TEMP,
                  '': [], 'SIG_DEMO': SIG_DEMO}
'''
features_group = {'ALL': ALL, 'LEXICAL': LEXICAL, 'WORD': WORDREP, 'EMOTION': EMOTION, 'AVERAGE': average, 'LENGTH': length,
                  'EMBEDDING': EMBEDDING, 'CONCRETE': CONCRETE, '': [], 'PAR': PAR, 'NON_PAR': NON_PAR, 'DEIXIS': DEIXIS, 'JJ': JJ, 'ALL_LEXICAL': ALL_LEXICAL,
                  'ALL_EMOTION': ALL_EMOTION, 'ALL_WORDREP': ALL_WORDREP, 'ALL_DEIXIS': ALL_DEIXIS, 'ALL_JJ': ALL_JJ, 'ALL_PAR': ALL_PAR,
                  'ALL_NON_PAR': ALL_NON_PAR}

if args.model.upper() in models:
    logging.basicConfig(filename='.\\logs\\%s logger l4.log' % args.model, level=logging.INFO)
    MODEL = models[args.model.upper()]
else:
    raise KeyError('Please select a valid model [LinearSVR, RBFSVR]')



input_features = args.features.upper().split()
FEATURES = []
# print(input_features)
for feature in input_features:
    if feature in features_group:
        FEATURES += features_group[feature]
# print(FEATURES)
# print(len(FEATURES))
exclusion = []
if args.exclude:
    exclusion = args.exclude.upper().split()
    for exl in exclusion:
        if exl in features_group:
            FEATURES = list(set(FEATURES) - set(features_group[exl]))
print(FEATURES)
# print(len(FEATURES))
# print(df_train.iloc[:, 3:242])
# print(df_valid.iloc[:, 3:242])
training_features = pd.DataFrame()
validation_features = pd.DataFrame()

df_all = pd.read_csv(args.all, sep='\t', skip_blank_lines=True)

df_train = df_all.iloc[:5767, :]
df_valid = df_all.iloc[5767:6267, :]

if args.test_or_not:
    df_test = df_all.iloc[6267:, :]

df_train_target = pd.read_csv(args.train_target, sep='\t', skip_blank_lines=True)
df_valid_target = pd.read_csv(args.valid_target, sep='\t', skip_blank_lines=True)

if 'DEMO' not in exclusion:
    df_train = df_train.dropna()
# print(df_valid.iloc[:, 227:234])

if args.test_or_not:
    df_test_target = pd.read_csv(args.test_target, sep='\t', skip_blank_lines=True)
    testing_features = pd.DataFrame()


for idx, i in enumerate(FEATURES):

    training_features[idx] = df_train.iloc[:, i]
    validation_features[idx] = df_valid.iloc[:, i]
    if args.test_or_not:
        testing_features[idx] = df_test.iloc[:, i]

print(training_features)
print(validation_features)
if args.test_or_not:
    print(testing_features)
training_target = df_train_target.iloc[:, 3]
validation_target = df_valid_target.iloc[:, 3]
if args.test_or_not:
    testing_target = df_test_target.iloc[:, 3]
    with open('.\\result\\ testing_target', 'w') as out_file:
        for item in testing_target:
            out_file.write(str(item))
            out_file.write('\n')

logging.info('Model %s' % args.model)
logging.info('features %s' % args.features)
logging.info('exclude %s' % args.exclude)

# _c = [i/32 + 17.5 for i in range(1, 129)]
# _c = [i/32 + 19.40625 for i in range(1, 33)]
_c = [i/32 + 19.625000 for i in range(1, 33)]
''' RBFSVR
SpearmanrResult(correlation=0.7435706961027945, pvalue=4.2902439420391166e-89)
SpearmanrResult(correlation=0.720022569414593, pvalue=1.5504038103572662e-160)
t = 12 valida_mae = 0.337441 test_MAE = 0.363657 C = 14.406250 gamma = 0.001000
'''
# _c = [14.406250]  # gamma = 0.001000

'''LINEARSVR
SpearmanrResult(correlation=0.7127798473686194, pvalue=9.694194636644628e-79)
SpearmanrResult(correlation=0.637351255116972, pvalue=4.3415466880400465e-115)
t = 22 valida_mae = 0.352965 test_MAE = 0.409402 C = 13.718750 gamma = -1.000000
'''
# _c = [13.718750]


# _c = [15.84375]
# _c = [18.1875]
_c = [20.1875]
if args.model.upper() == 'LINEARSVR' or args.model.upper() == 'Logistic' or args.model.upper() == 'LINEAR':
    _gamma = [-1]
else:
    _gamma_all = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
    _gamma_all_demo = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]
    _gamma_embedding = [0.1, 0.2, 0.4, 0.6, 0.8]
    _gamma_Lexical = [0.0006, 0.0008, 0.001, 0.002, 0.005]
    _gamma_Tweet = [0.00001, 0.00002, 0.00004, 0.00008, 0.0001, 0.0002]
    _gamma = [0.0001, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.4, 0.6, 0.8]
    # _gamma = [0.01]
    _gamma = [0.001]


best_mae = 5
best_predicted = []
best_para = []

for idx, c in enumerate(_c):

    for gamma in _gamma:
        # print(training_features)
        # print(validation_features)
        # print(validation_target)
        valid_MAE, valid_predicted = MODEL(training_features, training_target, validation_features, validation_target, c, gamma)
        if args.test_or_not:
            MAE, predicted = MODEL(training_features, training_target, testing_features, testing_target, c, gamma)
        print('t = %d valida_mae = %f test_MAE = %f C = %f gamma = %f' % (idx, valid_MAE, MAE, c, gamma))
        if valid_MAE < best_mae:
            best_mae = valid_MAE
            best_predicted = predicted
            best_para = [c, gamma]


print(best_mae)
print(best_para)
logging.info(best_mae)
logging.info(best_para)


with open('.\\result\\' + args.output, 'w') as out_file:
    out_file.write('%s best parameter %s' % (args.model, str(best_para)))
    out_file.write('\n')
    out_file.write(str(best_mae))
    out_file.write('\n')
    for line in best_predicted:
        out_file.write(str(line))
        out_file.write('\n')

logging.info('')
logging.info('')
logging.info('')
print('finished at', time.ctime())
print('Time elapsed', time.time() - st)