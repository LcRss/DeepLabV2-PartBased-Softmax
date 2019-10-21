import argparse
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from Utils import *
from myMetrics import *
from MyCustomCallback import CallbackmIoU
from TensorboardLR import TensorboardLR
from DeeplabV2_resnet101_params import ResNet101

print("Start at" + datetime.now().strftime("%Y%m%d-%H%M%S"))

parser = argparse.ArgumentParser()

parser.add_argument("--input_height", type=int, default=321)
parser.add_argument("--input_width", type=int, default=321)

parser.add_argument("--epochs", type=int, default=14)
parser.add_argument("--batch_size", type=int, default=2)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--maxIter", type=int, default=50000)
parser.add_argument("--baseOnEpochs", type=int, default=1)
parser.add_argument("--poly_learning_rate", type=int, default=0)

parser.add_argument("--use_batchNorm", type=int, default=0)
parser.add_argument("--use_dil_rate", type=int, default=0)
parser.add_argument("--dropout", type=int, default=0)
parser.add_argument("--mult_rate", type=int, default=1)
parser.add_argument("--kernel_init", type=str, default='he_uniform')

parser.add_argument("--pixel_distance", type=int, default=1)
parser.add_argument("--loss", type=str, default='standard')
parser.add_argument("--fit_1_epochs", type=int, default=4)
parser.add_argument("--fit_2_epochs", type=int, default=10)

args = parser.parse_args()

h_img = args.input_height
w_img = args.input_width
epochs_sz = args.epochs
batch_sz = args.batch_size
lr_p = args.lr
pat = args.patience
maxIter = args.maxIter
baseOnEpochs = args.baseOnEpochs
batchNorm = args.use_batchNorm
dropout = args.dropout
mult_rate = args.mult_rate
dil_rate = args.use_dil_rate
kernel_init = args.kernel_init
loss_name = args.loss
poly_lr = args.poly_learning_rate

fit_1_epochs = args.fit_1_epochs
fit_2_epochs = args.fit_2_epochs
lambda_loss = args.lambda_loss
pixel_distance = args.pixel_distance

num_cl = 108
train_sz = 4498
valid_sz = 500
rsize = True

print('Model')

deeplab_model = ResNet101(input_shape_1=(None, None, 3), input_shape_2=(None, None, 21), classes=num_cl,
                          kernel_init=kernel_init, batch_norm=batchNorm, dil_rate_model=dil_rate, mult_rate=mult_rate)

# pathLoadWeights = "Y:/tesisti/rossi/Weights/prova.h5"
pathLoadWeights = "Y:/tesisti/rossi/Weights/common_weights_test.h5"

deeplab_model.load_weights(pathLoadWeights, True)

if loss_name == "standard":
    loss = ["categorical_crossentropy"]
    metric_custom = [tf.keras.metrics.CategoricalAccuracy()]
    loss_name = "standard_loss"

elif loss_name == "custom_loss":
    loss = [custom_loss(deeplab_model.get_layer('logits'))]
    metric_custom = [tf.keras.metrics.CategoricalAccuracy()]
    loss_name = "custom_loss"

elif loss_name == "l1":
    loss = [custom_adj_loss_l1(batch_sz, lambda_loss, pixel_distance)]
    metric_custom = [tf.keras.metrics.CategoricalAccuracy(), metric_adj, metric_categ_cross]
    loss_name = "custom_adj_loss_l1_" + str(pixel_distance) + "_"

elif loss_name == "l2":
    loss = [custom_adj_loss_l2(batch_sz, lambda_loss, pixel_distance)]
    metric_custom = [tf.keras.metrics.CategoricalAccuracy(), metric_adj, metric_categ_cross]
    loss_name = "custom_adj_loss_l2_" + str(pixel_distance) + "_"

elif loss_name == "l1_weighted":
    loss = [custom_adj_loss_l1_weighted(batch_sz, lambda_loss, pixel_distance)]
    metric_custom = [tf.keras.metrics.CategoricalAccuracy(), metric_adj, metric_categ_cross]
    loss_name = "custom_adj_loss_l1_weighted_" + str(pixel_distance) + "_"

elif loss_name == "l2_weighted":
    loss = [custom_adj_loss_l2_weighted(batch_sz, lambda_loss, pixel_distance)]
    metric_custom = [tf.keras.metrics.CategoricalAccuracy(), metric_adj, metric_categ_cross]
    loss_name = "custom_adj_loss_l2_weighted_" + str(pixel_distance) + "_"

elif loss_name == "frobenius":
    loss = [custom_adj_loss_frobenius(batch_sz, lambda_loss, pixel_distance)]
    metric_custom = [tf.keras.metrics.CategoricalAccuracy(), metric_adj, metric_categ_cross]
    loss_name = "custom_adj_loss_frobenius_" + str(pixel_distance) + "_"

elif loss_name == "l2_2batch":
    loss = [custom_adj_loss_l2_different_adj_mat_for_dif_img(batch_sz, lambda_loss, pixel_distance)]
    metric_custom = [tf.keras.metrics.CategoricalAccuracy(), metric_adj, metric_categ_cross]
    loss_name = "2_mat_adj_" + str(pixel_distance) + "_"

else:
    print("Loss Name not value--END CODE")
    exit()

prefix = "Test_softmax_2_Fit_" + loss_name + "_lambda_" + str(lambda_loss) + "_kern_init_" + kernel_init + \
         "_epochs_" + str(epochs_sz) + "_"

# create all directories for checkpoint weights graph..
path, pathTBoard, pathTChPoints, pathWeight = createDirectories(prefix=prefix, lr_p=lr_p, batch_sz=batch_sz,
                                                                h_img=h_img, mult_rate=mult_rate, dil_rate=dil_rate,
                                                                use_BN=batchNorm)
print(path)
# Train
# train_images_path = "Y:/tesisti/rossi/data/train_val_test_png/train_png/"
# train_softmax_path = "Y:/tesisti/rossi/data/segmentation_gray/results_softmax_deeplabV2/train/"
# train_segs_path = "Y:/tesisti/rossi/data/segmentation_part_gray/new_dataset_107/data_part_107part_train/"
#
# # Val
# val_images_path = "Y:/tesisti/rossi/data/train_val_test_png/val_png/"
# val_softmax_path = "Y:/tesisti/rossi/data/segmentation_gray/results_softmax_deeplabV2/val/"
# val_segs_path = "Y:/tesisti/rossi/data/segmentation_part_gray/new_dataset_107/data_part_107part_val/"

# # # Train
train_images_path = "D:/tesisti/Rossi/Data_correct/train_val_test_png/train_png/"
train_softmax_path = "D:/tesisti/Rossi/Data_correct/segmentation_gray/results_softmax_deeplabV2/train/"
train_segs_path = "D:/tesisti/Rossi/Data_correct/segmentation_part_gray/new_dataset_107/data_part_107part_train/"
#
# # Val
val_images_path = "D:/tesisti/Rossi/Data_correct/train_val_test_png/val_png/"
val_softmax_path = "D:/tesisti/Rossi/Data_correct/segmentation_gray/results_softmax_deeplabV2/val/"
val_segs_path = "D:/tesisti/Rossi/Data_correct/segmentation_part_gray/new_dataset_107/data_part_107part_val/"

print('Loader')
G1 = data_loader(dir_img=train_images_path, dir_seg=train_segs_path, dir_softmax=train_softmax_path,
                 batch_size=batch_sz, h=h_img, w=w_img,
                 num_classes=num_cl, resize=rsize)

G2 = data_loader(dir_img=val_images_path, dir_seg=val_segs_path, dir_softmax=val_softmax_path,
                 batch_size=batch_sz, h=h_img, w=w_img,
                 num_classes=num_cl, resize=rsize)

print("FIT_1_LOSS_CATEGORICAL_CROSSENTROPY")

cb_tensorBoard = TensorBoard(log_dir=pathTBoard, histogram_freq=0, write_graph=True,
                             write_grads=False,
                             write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                             embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

cb_earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, min_delta=0,
                                 restore_best_weights=True)

cb_modelCheckPoint = ModelCheckpoint(filepath=pathTChPoints + 'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5',
                                     monitor='val_loss',
                                     save_best_only=False, save_weights_only=True, mode='auto', period=2)

logLr = pathTBoard + "/logLR"
if not os.path.isdir(logLr):
    os.mkdir(logLr)

cb_tensorBoardLR = TensorboardLR(log_dir=logLr)

cb_mIou = CallbackmIoU(path, lr=lr_p, pathRGB=val_images_path, pathSEG=val_segs_path, pathSoft=val_softmax_path,
                       pathGraphs=pathTBoard,
                       lr_base_on_epochs=baseOnEpochs, max_iter=maxIter, poly_lr=poly_lr)

print("FIT_1_LOSS_CATEGORICAL_CROSSENTROPY")

deeplab_model.compile(optimizer=optimizers.SGD(lr=lr_p, momentum=0.9, decay=0, nesterov=True),
                      loss=["categorical_crossentropy"],
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

history = deeplab_model.fit_generator(generator=G1, steps_per_epoch=train_sz // batch_sz, epochs=fit_1_epochs,
                                      verbose=1,
                                      callbacks=[cb_tensorBoard, cb_earlyStopping, cb_modelCheckPoint,
                                                 cb_mIou, cb_tensorBoardLR],
                                      validation_data=G2,
                                      validation_steps=valid_sz // batch_sz)

print("FIT_2_LOSS_ADJ")

checkpoint = glob.glob(pathTChPoints + "/*.hdf5")
checkpoint = checkpoint[len(checkpoint) - 1]

deeplab_model.load_weights(checkpoint)

deeplab_model.compile(optimizer=optimizers.SGD(lr=lr_p, momentum=0.9, decay=0, nesterov=True),
                      loss=loss,
                      metrics=[tf.keras.metrics.CategoricalAccuracy(), metric_adj, metric_categ_cross])

history_2 = deeplab_model.fit_generator(generator=G1, steps_per_epoch=train_sz // batch_sz, epochs=epochs_sz, verbose=1,
                                        callbacks=[cb_tensorBoard, cb_earlyStopping, cb_modelCheckPoint,
                                                   cb_mIou, cb_tensorBoardLR],
                                        validation_data=G2,
                                        validation_steps=valid_sz // batch_sz, initial_epoch=fit_1_epochs)
