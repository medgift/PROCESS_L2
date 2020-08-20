## Loading OS libraries to configure server preferences
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import setproctitle
SERVER_NAME = 'evenfaster'
import time
import sys
import shutil
## Adding PROCESS_UC1 utilities
sys.path.append('lib')
from models import *
from functions import *
from util import otsu_thresholding
from extract_xml import *
from functions import *
from mlta import *
import ConfigParser
"""
python train_cnn.py GPU_DEVICE EXPERIMENT_NAME RANDOM_SEED
"""



# DATA PATHS --> THESE WILL HAVE TO BE MODIFIED FOR THE DENSE DATASET
DATA_FILE = r'./data/data.cfg'
configParser = ConfigParser.RawConfigParser()
configParser.read(DATA_FILE)
#
cam16 = hd.File(configParser.get('hdf5', 'cam16'), 'r')
all500 = hd.File(configParser.get('hdf5', 'all500'), 'r')
extra17 = hd.File(configParser.get('hdf5', 'extra17'), 'r')
tumor_extra17=hd.File(configParser.get('hdf5', 'tumor_extra17'),'r')
test2 = hd.File(configParser.get('hdf5', 'test2'),'r')
global data
data={'cam16':cam16,'all500':all500,'extra17':extra17, 'tumor_extra17':tumor_extra17, 'test_data2': test2}
# DATA SPLIT CSVs
train_csv=open(configParser.get('csv', 'train_csv'), 'r')
val_csv=open(configParser.get('csv', 'val_csv'), 'r')
test_csv=open(configParser.get('csv', 'test_csv'), 'r')
train_list=train_csv.readlines()
val_list=val_csv.readlines()
test_list=test_csv.readlines()
test2_csv = open(configParser.get('csv', 'test2_csv'), 'r')
test2_list=test2_csv.readlines()
test2_csv.close()
train_csv.close()
val_csv.close()
test_csv.close()
data_csv=open(configParser.get('csv', 'data_csv'), 'r')
data_list=data_csv.readlines()
data_csv.close()

#SYSTEM CONFIGS
CONFIG_FILE = 'doc/config.cfg'
GPU_DEVICE = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_DEVICE)
EXPERIMENT_TYPE = sys.argv[2]
COLOR = True
global new_folder
new_folder = getFolderName()
new_folder = 'results/'+new_folder
os.mkdir(new_folder)
# creating an INFO.log file to keep track of the model run
llg.basicConfig(filename=os.path.join(new_folder, 'INFO.log'), filemode='w', level=llg.INFO)
shutil.copy2(src=CONFIG_FILE, dst=os.path.join(new_folder, '.'))
# SAVE FOLD
f=open(new_folder+"/seed.txt","w")
seed=int(sys.argv[3])
print seed
f.write(str(seed))
f.close()

# SET PROCESS TITLE
setproctitle.setproctitle('UC1_{}_{}'.format(EXPERIMENT_TYPE, new_folder))

# SET SEED
np.random.seed(seed)
tf.set_random_seed(seed)

# GPU CONFIG
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# STAIN NORMALIZATION
def get_normalizer(patch, save_folder=''):
    normalizer = ReinhardNormalizer()
    normalizer.fit(patch)
    np.save('{}/normalizer'.format(save_folder),normalizer)
    np.save('{}/normalizing_patch'.format(save_folder), patch)
    print('Normalisers saved to disk.')
    return normalizer

def normalize_patch(patch, normalizer):
    return np.float64(normalizer.transform(np.uint8(patch)))

# LOAD DATA NORMALIZER
global normalizer
db_name, entry_path, patch_no = get_keys(data_list[0])
normalization_reference_patch = data[db_name][entry_path][patch_no]
normalizer = get_normalizer(normalization_reference_patch, save_folder=new_folder)

"""
Batch generators:
They load a patch list: a list of file names and paths.
They use the list to create a batch of 32 samples.
"""
# BATCH GENERATORS
def get_batch_data(patch_list, batch_size=32):
    num_samples=len(patch_list)
    while True:
        for offset in range(0,num_samples, batch_size):
            batch_x = []
            batch_y = []
            batch_cm = []
            #print offset
            batch_samples=patch_list[offset:offset+batch_size]
            for line in batch_samples:
                db_name, entry_path, patch_no = get_keys(line)
                patch=data[db_name][entry_path][patch_no]
                patch=normalize_patch(patch, normalizer)
                # patch=keras.applications.inception_v3.preprocess_input(patch) #removed bc of BNorm
                patch=keras.applications.resnet50.preprocess_input(patch)
                label = get_class(line, entry_path)
                batch_x.append(patch)
                batch_y.append(label)
                # BASELINE
                batch_cm.append(1.) # cm is a constant in the baseline
            batch_x = np.asarray(batch_x, dtype=np.float32)
            yield batch_samples, np.asarray(batch_x, dtype=np.float32), np.asarray(batch_y, dtype=np.float32), np.asarray(batch_cm, dtype=np.float32)

def get_test_batch(patch_list, batch_size=32):
    num_samples=len(patch_list)
    while True:
        for offset in range(0,num_samples, batch_size):
            batch_x = []
            batch_y = []
            batch_cm = []
            #print offset
            batch_samples=patch_list[offset:offset+batch_size]
            for line in batch_samples:
                db_name, entry_path, patch_no = get_keys(line)
                patch=data[db_name][entry_path][patch_no]

                patch=normalize_patch(patch, normalizer)
                #patch=keras.applications.inception_v3.preprocess_input(patch)
                patch=keras.applications.resnet50.preprocess_input(patch)
                label = get_test_label(entry_path)
                batch_x.append(patch)
                batch_y.append(label)
                # BASELINE
                batch_cm.append(1.) # cm is a constant in the baseline
            batch_x = np.asarray(batch_x, dtype=np.float32)
            yield batch_samples, np.asarray(batch_x, dtype=np.float32), np.asarray(batch_y, dtype=np.float32), np.asarray(batch_cm, dtype=np.float32)

"""
Building baseline model
"""
#
# MODEL: BASELINE
#base_model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet')
base_model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')

feature_output=base_model.layers[-2].output
feature_output = Dense(2048, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features1')(feature_output)
feature_output = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features2')(feature_output)
feature_output = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features3')(feature_output)
finetuning = Dense(1,name='predictions')(feature_output)
regression_output = keras.layers.Dense(1, activation = keras.layers.Activation('linear'), name='concept_regressor')(feature_output)
model = Model(input=base_model.input, output=[finetuning, regression_output])

# LOG FILE
f=open("{}/{}_log.txt".format(new_folder, EXPERIMENT_TYPE), 'w')
f.write('Trainable layers: ')
for layer in model.layers:
    if layer.trainable==True:
        f.write('{}\n'.format(layer.name))
# parameters early stopping
patience_counter=0
best_val_loss= 100000#metrics[1]
best_test_auc= 0.0 #auc_test
best_test_r2 = 0.0 #test_metrics[-1]
MAX_PATIENCE=2
# batch params for train
BATCH_SIZE=32
MAX_BATCHES= 100#len(train_list)/BATCH_SIZE
# no batches for evaluation
TEST_BATCHES= len(test_list)/32
TEST_2_BATCHES= len(test2_list)/32
VAL_BATCHES= len(val_list)/32
f.write("MAX_PATIENCE: {}\n BATCH_SIZE: {}\n MAX_BATCHES: {}".format(MAX_PATIENCE,BATCH_SIZE,MAX_BATCHES))

model.compile(optimizer=optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
             loss=[classifier_loss, zero_loss],
              loss_weights=[1., 0.], # BASELINE => concept loss weight = 0
               metrics=[my_accuracy, r_square])

starting_time = time.time()

for e in range(20):
    p = np.float(e) / 10
    lr = 1e-4
    K.set_value(model.optimizer.lr, lr)
    f.write('\n[debug] desired lr at epoch {}: {}, lr value: {}\n'.format(e, lr, K.eval(model.optimizer.lr)))
    # parameters for training
    total_loss=0.
    main_task_loss=0.
    concept_loss=0.
    main_task_acc=0.
    concept_regressor_r2=0.
    n_batch=0
    total_n1_in_batch=0
    # data generators
    train_generator=get_batch_data(data_list)
    train_generator2=get_batch_data(data_list)
    test_generator=get_test_batch(test_list)
    test2_generator=get_test_batch(test2_list)
    val_generator=get_test_batch(val_list)
    # parameters for evaluation
    y_pred=np.zeros(32*TEST_BATCHES)
    y_pred2=np.zeros(32*TEST_2_BATCHES)
    y_test_all=np.zeros(32*TEST_BATCHES)
    y_test2_all=np.zeros(32*TEST_2_BATCHES)
    y_pred_val = np.zeros(32*VAL_BATCHES)
    y_val_all = np.zeros(32*VAL_BATCHES)
    n_t_batches=0
    n_t2_batches=0
    n_val_batches=0
    batch_size=32
    len_batch=32
    # evaluation metrics
    #validation
    val_loss_total=0.
    val_loss_main_task =0.
    val_loss_concept=0.
    val_acc=0
    val_r2=0.
    #test
    test_loss_total=0.
    test_loss_main_task=0.
    test_loss_concept=0.
    test_acc=0
    test_r2=0.
    #test2
    test_2_loss_total=0.
    test_2_loss_main_task=0.
    test_2_loss_concept=0.
    test_2_acc=0
    test_2_r2=0.
    #
    loss_total=0.
    loss_main_task=0.
    loss_concept=0.
    acc_main_task=0.
    concept_r2=0.

    # PRE-TRAINING evaluation
    # test
    while n_t_batches<TEST_BATCHES:
        batch_samples, x_test, y_test, cm_test=next(test_generator)
        y_pred[n_t_batches*len_batch:n_t_batches*len_batch+len(y_test)]=model.predict(x_test)[0].ravel()
        y_test_all[n_t_batches*len_batch:n_t_batches*len_batch+len(y_test)]=y_test
        t_score = model.evaluate(x_test, [y_test, cm_test], verbose=0)
        test_loss_total += t_score[0]
        test_loss_main_task += t_score[1]
        test_loss_concept += t_score[2]
        test_acc+=t_score[3]
        test_r2 += t_score[6]
        n_t_batches+=1
        len_batch=len(y_test)
    len_batch=32
    # test2
    while n_t2_batches<TEST_2_BATCHES:
        batch_samples, x_test, y_test, cm_test = next(test2_generator)
        model_output = model.predict(x_test)
        y_pred2[n_t2_batches*len_batch:n_t2_batches*len_batch+len(y_test)]=model_output[0].ravel()
        y_test2_all[n_t2_batches*len_batch:n_t2_batches*len_batch+len(y_test)]=y_test
        t2_score = model.evaluate(x_test, [y_test, cm_test], verbose=0)
        test_2_loss_total += t2_score[0]
        test_2_loss_main_task += t2_score[1]
        test_2_loss_concept += t2_score[2]
        test_2_acc += t2_score[3]
        test_2_r2 += t2_score[6]
        len_batch=len(y_test)
        n_t2_batches+=1
    # validation
    while n_val_batches<VAL_BATCHES:
        batch_samples, x_val, y_val, cm_val = next(val_generator)
        val_score = model.evaluate(x_val, [y_val, cm_val], verbose=0)
        model_output = model.predict(x_val)
        y_pred_val[n_val_batches*len_batch:n_val_batches*len_batch+len(y_val)]=model_output[0].ravel()
        y_val_all[n_val_batches*len_batch:n_val_batches*len_batch+len(y_val)]=y_val
        len_batch=len(y_val)
        val_loss_total += val_score[0]
        val_loss_main_task += val_score[1]
        val_loss_concept += val_score[2]
        val_acc += val_score[3]
        val_r2 += val_score[6]
        n_val_batches+=1
    test_auc=sklearn.metrics.roc_auc_score(y_test_all, y_pred)
    test_2_auc=sklearn.metrics.roc_auc_score(y_test2_all, y_pred2)
    val_auc = sklearn.metrics.roc_auc_score(y_val_all, y_pred_val)

    sign_pred = np.sign(y_pred_val)
    zeros_count = np.count_nonzero(sign_pred == -1)
    ones_count =  np.count_nonzero(sign_pred == 1)
    f.write('\nValidation: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {}, percentage of predicted zeros-ones: {}-{} \n'.format(val_loss_total/VAL_BATCHES,
                                            val_loss_main_task/VAL_BATCHES,
                                            val_loss_concept/VAL_BATCHES,
                                            val_acc/VAL_BATCHES,
                                            val_r2/VAL_BATCHES,
                                            val_auc,
                                            float(zeros_count)/len(y_pred_val),
                                            float(ones_count)/len(y_pred_val)
                                            )
                                            )
    print '\nValidation: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {}, percentage of predicted zeros-ones: {}-{} \n'.format(val_loss_total/VAL_BATCHES,
                                            val_loss_main_task/VAL_BATCHES,
                                            val_loss_concept/VAL_BATCHES,
                                            val_acc/VAL_BATCHES,
                                            val_r2/VAL_BATCHES,
                                            val_auc,
                                            float(zeros_count)/len(y_pred_val),
                                            float(ones_count)/len(y_pred_val)
                                            )

    sign_pred = np.sign(y_pred)
    zeros_count = np.count_nonzero(sign_pred == -1)
    ones_count =  np.count_nonzero(sign_pred == 1)
    f.write('Test: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {}, percentage of predicted zeros-ones: {}-{} \n'.format(test_loss_total/TEST_BATCHES,
                                            test_loss_main_task/TEST_BATCHES,
                                            test_loss_concept/TEST_BATCHES,
                                            test_acc/TEST_BATCHES,
                                            test_r2/TEST_BATCHES,
                                            test_auc,
                                            float(zeros_count)/len(y_pred),
                                            float(ones_count)/len(y_pred)
                                            ))
    print 'Test: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {}, percentage of predicted zeros-ones: {}-{} \n'.format(test_loss_total/TEST_BATCHES,
                                            test_loss_main_task/TEST_BATCHES,
                                            test_loss_concept/TEST_BATCHES,
                                            test_acc/TEST_BATCHES,
                                            test_r2/TEST_BATCHES,
                                            test_auc,
                                            float(zeros_count)/len(y_pred),
                                            float(ones_count)/len(y_pred)
                                            )
    sign_pred = np.sign(y_pred2)
    zeros_count = np.count_nonzero(sign_pred == -1)
    ones_count =  np.count_nonzero(sign_pred == 1)
    f.write('Test2: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {}, percentage of predicted zeros-ones: {}-{} \n'.format(test_2_loss_total/TEST_2_BATCHES,
                                            test_2_loss_main_task/TEST_2_BATCHES,
                                            test_2_loss_concept/TEST_2_BATCHES,
                                            test_2_acc/TEST_2_BATCHES,
                                            test_2_r2/TEST_2_BATCHES,
                                            test_2_auc,
                                            float(zeros_count)/len(y_pred2),
                                            float(ones_count)/len(y_pred2)
                                            )
           )

    print 'Test2: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {}, percentage of predicted zeros-ones: {}-{} \n'.format(test_2_loss_total/TEST_2_BATCHES,
                                            test_2_loss_main_task/TEST_2_BATCHES,
                                            test_2_loss_concept/TEST_2_BATCHES,
                                            test_2_acc/TEST_2_BATCHES,
                                            test_2_r2/TEST_2_BATCHES,
                                            test_2_auc,
                                            float(zeros_count)/len(y_pred2),
                                            float(ones_count)/len(y_pred2)
                                            )
    if e>10:
        if patience_counter==MAX_PATIENCE:
            print "EARLY STOPPING suggested at epoch {}: best_val_loss {}, best_test_auc {}, best_test2_auc {}".format(e, best_val_loss, best_test_auc, best_test2_auc)
            f.write("EARLY STOPPING suggested at epoch {}: best_val_loss {}, best_test_auc {}, best_test2_auc {}".format(e, best_val_loss, best_test_auc, best_test2_auc))
            patience_counter=0
        elif (val_loss_total/VAL_BATCHES)<best_val_loss:
            patience_counter=0
            best_val_loss=(val_loss_total/VAL_BATCHES)
            best_test_auc=test_auc
            best_test2_auc=test_2_auc
        else:
            patience_counter+=1

    loss_total=loss_main_task=acc_main_task=0.
    predicted_ys=[]
    for n_batch in range(MAX_BATCHES):
        paths, x, y, cm = next(train_generator)
        # EVAL on training
        # n1 is used to correctly weight classes
        # and neglect samples that have class -1 in the average accuracy
        n1 = BATCH_SIZE - len(np.argwhere(y==-1))
        total_n1_in_batch += n1
        eval_ = model.evaluate(x, [y, cm], verbose=0)
        predict = model.predict(x)[0]
        loss_total += n1 * eval_[0] #loss
        loss_main_task += n1 * eval_[1] #loss classifier
        loss_concept += eval_[2] # concept MSE
        acc_main_task += n1* eval_[3] # classifier accuracy
        concept_r2 += eval_[6] # concept r2
    #print loss_total, loss_main_task, acc_main_task
    loss_total /=total_n1_in_batch
    loss_main_task/=total_n1_in_batch
    acc_main_task/=total_n1_in_batch
    print '\n Pre Training Epoch {}, total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}'.format(e,
           loss_total,
           loss_main_task,
           loss_concept/MAX_BATCHES,
           acc_main_task,
           concept_r2/MAX_BATCHES
           )
    f.write('\n Pre Training Epoch {}: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {} : concept r2: {} \n'.format(e,
              loss_total,
              loss_main_task,
              loss_concept/MAX_BATCHES,
              acc_main_task,
              concept_r2/MAX_BATCHES
             )
            )
    #
    n1=0
    total_n1_in_batch=0
    total_loss= main_task_loss= main_task_acc=0.
    for n_batch in range(MAX_BATCHES):
        paths, x, y, cm = next(train_generator2)
        # TRAIN -- baseline
        # batch_metrics = model.fit(x, [y, cm],
        #     batch_size=64,
        #      class_weight=[{-1:1., 0:1., 1:1.,}, {-1: 1, 0: 1., 1: 1.}],
        #      epochs=1,
        #     verbose=0
        #     )
        """
        Here I am loading the importance sampling pipeline.
        """
        # TRAIN -- importance sampling
        batch_metrics = ImportanceTraining(model).fit(
                          x, [y, cm],
                          batch_size=BATCH_SIZE,
                          epochs=5,
                          verbose=1,
                          #validation_data=(x_test, y_test)
                        )
while n_t_batches<TEST_BATCHES:
    batch_samples, x_test, y_test, cm_test=next(train_generator)
    y_pred[n_t_batches*len_batch:n_t_batches*len_batch+len(y_test)]=model.predict(x_test)[0].ravel()
    y_test_all[n_t_batches*len_batch:n_t_batches*len_batch+len(y_test)]=y_test
    t_score = model.evaluate(x_test, [y_test, cm_test], verbose=0)
    test_loss_total += t_score[0]
    test_loss_main_task += t_score[1]
    test_loss_concept += t_score[2]
    test_acc+=t_score[3]
    test_r2 += t_score[6]
    n_t_batches+=1
    len_batch=len(y_test)
len_batch=32
while n_t2_batches<TEST_2_BATCHES:
    batch_samples, x_test, y_test, cm_test=next(train_generator)
    model_output = model.predict(x_test)
    y_pred2[n_t2_batches*len_batch:n_t2_batches*len_batch+len(y_test)]=model_output[0].ravel()
    y_test2_all[n_t2_batches*len_batch:n_t2_batches*len_batch+len(y_test)]=y_test
    t2_score = model.evaluate(x_test, [y_test, cm_test], verbose=0)
    test_2_loss_total += t_score[0]
    test_2_loss_main_task += t_score[1]
    test_2_loss_concept += t_score[2]
    test_2_acc += t_score[3]
    test_2_r2 += t2_score[6]
    len_batch=len(y_test)
    n_t2_batches+=1
while n_val_batches<VAL_BATCHES:
    batch_samples, x_val, y_val, cm_val=next(train_generator)
    val_score = model.evaluate(x_val, [y_val, cm_val], verbose=0)
    model_output = model.predict(x_val)
    y_pred_val[n_val_batches*len_batch:n_val_batches*len_batch+len(y_val)]=model_output[0].ravel()
    y_val_all[n_val_batches*len_batch:n_val_batches*len_batch+len(y_val)]=y_val
    len_batch=len(y_val)
    val_loss_total += val_score[0]
    val_loss_main_task += val_score[1]
    val_loss_concept += val_score[2]
    val_acc += val_score[3]
    val_r2 += val_score[6]
    n_val_batches+=1
test_auc=sklearn.metrics.roc_auc_score(y_test_all, y_pred)
test_2_auc=sklearn.metrics.roc_auc_score(y_test2_all, y_pred2)
val_auc = sklearn.metrics.roc_auc_score(y_val_all, y_pred_val)
f.write('Validation: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {}, \n'.format(val_loss_total/VAL_BATCHES,
            val_loss_main_task/VAL_BATCHES,
            val_loss_concept/VAL_BATCHES,
            val_acc/VAL_BATCHES,
            val_r2/VAL_BATCHES,
            val_auc,
            ))
print 'Validation: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {},  \n'.format(val_loss_total/VAL_BATCHES,
            val_loss_main_task/VAL_BATCHES,
            val_loss_concept/VAL_BATCHES,
            val_acc/VAL_BATCHES,
            val_r2/VAL_BATCHES,
            val_auc,
            )
f.write('Test: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {}, \n'.format(test_loss_total/TEST_BATCHES,
            test_loss_main_task/TEST_BATCHES,
            test_loss_concept/TEST_BATCHES,
            test_acc/TEST_BATCHES,
            test_r2/TEST_BATCHES,
            test_auc,
            ))
print 'Test: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {}, \n'.format(test_loss_total/TEST_BATCHES,
                                        test_loss_main_task/TEST_BATCHES,
                                        test_loss_concept/TEST_BATCHES,
                                        test_acc/TEST_BATCHES,
                                        test_r2/TEST_BATCHES,
                                        test_auc,
                                        )
f.write('Test2: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {}, \n'.format(test_2_loss_total/TEST_2_BATCHES,
                                        test_2_loss_main_task/TEST_2_BATCHES,
                                        test_2_loss_concept/TEST_2_BATCHES,
                                        test_2_acc/TEST_2_BATCHES,
                                        test_2_r2/TEST_2_BATCHES,
                                        test_2_auc,
                                        ))
print 'Test2: total loss: {}, main task loss: {}, concept loss: {}, main task accuracy: {}, concept r2: {}, AUC: {},  \n'.format(test_2_loss_total/TEST_2_BATCHES,
                                        test_2_loss_main_task/TEST_2_BATCHES,
                                        test_2_loss_concept/TEST_2_BATCHES,
                                        test_2_acc/TEST_2_BATCHES,
                                        test_2_r2/TEST_2_BATCHES,
                                        test_2_auc,
                                       )
""" END TRANSFER LEARNING """

end_time = time.time()

total_training_time = end_time - starting_time
#
f.write('Time elapsed for model training: {}'.format(total_training_time))
f.close()
# logging and plotting training curves

baseline_log=open('./{}/baseline_log.txt'.format(new_folder), 'r')
read = baseline_log.readlines()
train_loss=[]
val_loss=[]
test_loss=[]
test2_loss=[]
lr=[]
lambda_=[]
train_acc=[]
val_acc=[]
test_acc=[]
test2_acc=[]
for line in read:
    if 'Training ' in line:
        train_loss.append(line.split('main task loss: ')[1].split(', ')[0])
    elif 'Validation' in line:
        val_loss.append(line.split('main task loss: ')[1].split(', ')[0])
    elif 'Test: ' in line:
        test_loss.append(line.split('main task loss: ')[1].split(', ')[0])
    elif 'Test2: ' in line:
        test2_loss.append(line.split('main task loss: ')[1].split(', ')[0])
    elif 'desired lr at epoch' in line:
        lr.append(line.split('value: ')[1].split(', ')[0])
    elif 'desired lambda' in line:
        lambda_.append(line.split('value: ')[1].split(', ')[0])
train_loss=np.asarray(train_loss, dtype=np.float32)
val_loss=np.asarray(val_loss, dtype=np.float32)
test_loss=np.asarray(test_loss, dtype=np.float32)
test2_loss=np.asarray(test2_loss, dtype=np.float32)
lr=np.asarray(lr, dtype=np.float32)
lambda_=np.asarray(lambda_, dtype=np.float32)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_loss)
plt.plot(val_loss)
plt.plot(test_loss)
plt.plot(test2_loss)
plt.title('Main task loss per epoch')
plt.legend(['train', 'validation', 'test', 'test2'])
plt.savefig('./{}/patched_baseline_loss.png'.format(new_folder))
train_acc=[]
val_acc=[]
test_acc=[]
test2_acc=[]
for line in read:
    if 'Training ' in line:
        train_acc.append(line.split('main task accuracy: ')[1].split(': ')[0])
    elif 'Validation' in line:
        val_acc.append(line.split('main task accuracy: ')[1].split(', ')[0])
    elif 'Test: ' in line:
        test_acc.append(line.split('main task accuracy: ')[1].split(', ')[0])
    elif 'Test2: ' in line:
        test2_acc.append(line.split('main task accuracy: ')[1].split(', ')[0])
train_acc=np.asarray(train_acc, dtype=np.float32)
val_acc=np.asarray(val_acc, dtype=np.float32)
test_acc=np.asarray(test_acc, dtype=np.float32)
test2_acc=np.asarray(test2_acc, dtype=np.float32)
plt.figure()
plt.plot(train_acc)
plt.plot(val_acc)
plt.plot(test_acc)
plt.plot(test2_acc)
plt.title('Main task accuracy per epoch')
plt.legend(['train', 'validation', 'test', 'test2'])
plt.savefig('./{}/patched_baseline_acc.png'.format(new_folder))

print "++++++ BASELINE: OK WE'RE DONE OVER HERE ++++++ FOLDER: {} ++++++ GOOD JOB. ".format(new_folder)
