## Loading OS libraries to configure server preferences
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import setproctitle
SERVER_NAME = 'ultrafast'
import time
import sys
import shutil
## Adding PROCESS_UC1 utilities
sys.path.append('lib')
from models import *
from util import otsu_thresholding
from extract_xml import *
from functions import *
from mlta import *
import math
import horovod.keras as hvd
import sklearn.metrics
#
"""
bash hvd_train.sh EXPERIMENT_NAME
"""
#
EXPERIMENT_TYPE=sys.argv[2]
#
# Horovod: initialize Horovod.
hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
keras.backend.set_session(tf.Session(config=config))
#
verbose=1 if hvd.local_rank()==1 else 0
# DATA PATHS
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
COLOR = True
global new_folder
new_folder = getFolderName()
folder_name= EXPERIMENT_TYPE
new_folder = 'results/'+folder_name
# creating an INFO.log file to keep track of the model run
llg.basicConfig(filename=os.path.join(new_folder, 'INFO.log'), filemode='w', level=llg.INFO)
shutil.copy2(src=CONFIG_FILE, dst=os.path.join(new_folder, '.'))
BATCH_SIZE = 32
# SAVE FOLD
f=open(new_folder+"/seed.txt","w")
seed=int(sys.argv[1])
print seed
f.write(str(seed))
f.close()
# SET PROCESS TITLE
setproctitle.setproctitle('UC1_{}'.format(EXPERIMENT_TYPE))
# SET SEED
np.random.seed(seed)
tf.set_random_seed(seed)
#
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
#print data_list[0]
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
    #global batch_size
    while True:
        offset = 0
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
                patch=keras.applications.inception_v3.preprocess_input(patch) #removed bc of BNorm
                #patch=keras.applications.resnet50.preprocess_input(patch)
                label = get_class(line, entry_path) # is there a problem with get_class ?
                #print "append"
                batch_x.append(patch)
                batch_y.append(label)
                # BASELINE
                #batch_cm.append(1.) # cm is a constant in the baseline
            batch_x = np.asarray(batch_x, dtype=np.float32)
            #batch_x /= 255.
            yield np.asarray(batch_x, dtype=np.float32), np.asarray(batch_y, dtype=np.float32)#, np.asarray(batch_cm, dtype=np.float32)

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
                patch=keras.applications.inception_v3.preprocess_input(patch)
                #patch=keras.applications.resnet50.preprocess_input(patch)
                label = get_test_label(entry_path)
                batch_x.append(patch)
                batch_y.append(label)
                # BASELINE
                #batch_cm.append(1.) # cm is a constant in the baseline
            batch_x = np.asarray(batch_x, dtype=np.float32)
            #batch_x /= 255.
            yield np.asarray(batch_x, dtype=np.float32), np.asarray(batch_y, dtype=np.float32)#, np.asarray(batch_cm, dtype=np.float32)

"""
Building baseline model
"""
#
# MODEL: BASELINE
#base_model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet')
input_tensor = keras.layers.Input(shape=(224, 224, 3))
base_model = keras.applications.inception_v3.InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)

# freezing some layers
layers_list=['conv2d_92', 'conv2d_93', 'conv2d_88', 'conv2d_89', 'conv2d_86']
for i in range(len(base_model.layers[:])):
    layer=base_model.layers[i]
    if layer.name in layers_list:
        print layer.name
        layer.trainable=True
    else:
        layer.trainable = False

feature_output_first=base_model.layers[-2].output
# adding dropout as extra-regularization
dropout_layer = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output_first)
feature_output = Dense(2048, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features1')(dropout_layer)
dropout_layer = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
feature_output = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features2')(dropout_layer)
dropout_layer = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
feature_output = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features3')(dropout_layer)

finetuning = Dense(1, name='predictions')(feature_output)
#finetuning = Dense(1, name='predictions', activation ='sigmoid')(feature_output)
model = Model(input=base_model.input, output=finetuning)

# LOG FILE
f=open("{}/{}_log.txt".format(new_folder, EXPERIMENT_TYPE), 'w')
f.write('Trainable layers: ')
for layer in model.layers:
    if layer.trainable==True:
        f.write('{}\n'.format(layer.name))
# HVD optimizer
initial_lr = 1e-4 * hvd.size()
opt = keras.optimizers.SGD(lr=initial_lr, decay=1e-6, momentum=0.9, nesterov=True)
opt = hvd.DistributedOptimizer(opt)
# Model Compile
model.compile(optimizer=opt,
             loss=classifier_loss,
               metrics=[my_accuracy])
# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '{}'.format(new_folder) if hvd.rank() == 0 else None
# HVD callbacks
callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        # Horovod: average metrics among workers at the end of every epoch.        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard, or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5,
                                                 verbose=verbose),
        # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=5, end_epoch=30, multiplier=1.
                                                   ),
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1 ),
        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50),
        #hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2, initial_lr=initial_lr),
        #hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3, initial_lr=initial_lr),
]
#
if hvd.rank()==0:
    callbacks.append(keras.callbacks.ModelCheckpoint('{}/best_model.h5'.format(checkpoint_dir), monitor='val_loss', mode='min', save_best_only=True, verbose=1))
#
train_generator=get_batch_data(data_list, batch_size=BATCH_SIZE)
val_generator=get_test_batch(val_list, batch_size=BATCH_SIZE)
test_generator=get_test_batch(test_list, batch_size=BATCH_SIZE)
before_training_score = hvd.allreduce(model.evaluate_generator(test_generator, (len(test_list)// BATCH_SIZE)//hvd.size(), workers=4))
avg_auc = 0
T_B=len(test_list)//BATCH_SIZE
ys=np.zeros(len(test_list))
preds=np.zeros((len(test_list),1))
#
for i in range(T_B):
    x,y=test_generator.next()
    ys[i*BATCH_SIZE:(i)*BATCH_SIZE+len(y)] = y
    preds[i*BATCH_SIZE:(i)*BATCH_SIZE+len(y)] = model.predict(x)
auc=sklearn.metrics.roc_auc_score(ys,preds)
#
print 'Before training auc: ', auc
f.write('Before training auc: {}\n'.format(auc))
f.write('Before training loss, acc: {}'.format(before_training_score))
# START TRAINING
starting_time = time.time()
history = model.fit_generator(train_generator,
                    steps_per_epoch= (len(data_list)// BATCH_SIZE) // hvd.size(),
                    callbacks=callbacks,
                    epochs=40,
                    verbose=verbose,
                    workers=4,
                    validation_data=val_generator,
                    validation_steps= (len(val_list)// BATCH_SIZE) // hvd.size()
                             )
end_time = time.time()
#
score = hvd.allreduce(model.evaluate_generator(test_generator, (len(test_list)// BATCH_SIZE)//hvd.size(), workers=4))
pred = hvd.allreduce(
    model.predict_generator(test_generator, (len(test_list)// BATCH_SIZE)//hvd.size(), workers=4))
if verbose:
    print ('Before training test score: ', before_training_score[0], before_training_score[1])
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
f.write('Post training loss, acc: {}\n'.format(score))
total_training_time = end_time - starting_time
#
avg_auc = 0
T_B=len(test_list)//BATCH_SIZE
ys=np.zeros(len(test_list))
preds=np.zeros((len(test_list),1))
for i in range(T_B):
    x,y=test_generator.next()
    ys[i*BATCH_SIZE:(i)*BATCH_SIZE+len(y)] = y
    preds[i*BATCH_SIZE:(i)*BATCH_SIZE+len(y)] = model.predict(x)
auc=sklearn.metrics.roc_auc_score(ys,preds)
#
print 'Post training auc: ', auc
f.write('Post training auc: {}\n'.format(auc))
#
print history.history.keys()
f.write('Time elapsed for model training: {}\n'.format(total_training_time))
f.close()
np.save('{}/training_log'.format(new_folder), history.history)
print "++++++ BASELINE: OK WE'RE DONE OVER HERE ++++++ FOLDER: {} ++++++ GOOD JOB. ".format(new_folder)
exit(0)
