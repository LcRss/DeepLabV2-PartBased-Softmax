import argparse
from Utils import *
from myMetrics import *
from DeeplabV2_resnet101_params import ResNet101
from tensorflow.python.keras import optimizers

print("Start at" + datetime.now().strftime("%Y%m%d-%H%M%S"))

parser = argparse.ArgumentParser()

parser.add_argument("--input_height", type=int, default=321)
parser.add_argument("--input_width", type=int, default=321)

parser.add_argument("--epochs", type=int, default=14)
parser.add_argument("--batch_size", type=int, default=2)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--maxIter", type=int, default=50000)
parser.add_argument("--baseOnEpochs", type=int, default=0)
parser.add_argument("--poly_learning_rate", type=int, default=1)

parser.add_argument("--use_batchNorm", type=int, default=0)
parser.add_argument("--use_dil_rate", type=int, default=0)
parser.add_argument("--mult_rate", type=int, default=1)
parser.add_argument("--kernel_init", type=str, default='he_uniform')

parser.add_argument("--lambda_loss", type=float, default=1)
parser.add_argument("--pixel_distance", type=int, default=1)
parser.add_argument("--loss", type=str, default='standard')

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
mult_rate = args.mult_rate
dil_rate = args.use_dil_rate
kernel_init = args.kernel_init
loss_name = args.loss
poly_lr = args.poly_learning_rate
lambda_loss = args.lambda_loss
pixel_distance = args.pixel_distance

num_cl = 108
train_sz = 4498
valid_sz = 500
rsize = True

print('Model')

deeplab_model = ResNet101(input_shape_1=(None, None, 3), input_shape_2=(None, None, 21), classes=num_cl,
                          kernel_init=kernel_init, batch_norm=batchNorm, dil_rate_model=dil_rate, mult_rate=mult_rate)

pathLoadWeights = "Y:/tesisti/rossi/Weights/prova.h5"
deeplab_model.load_weights(pathLoadWeights, True)

if loss_name == "standard":
    loss = ["categorical_crossentropy"]
    loss_name = "standard_loss"

elif loss_name == "custom_loss":
    loss = [custom_loss(deeplab_model.get_layer('logits'))]
    loss_name = "custom_loss"

elif loss_name == "l1":
    loss = [custom_adj_loss_l1(batch_sz, lambda_loss, pixel_distance)]
    loss_name = "custom_adj_loss_l1_" + str(pixel_distance) + "_"

elif loss_name == "l2":
    loss = [custom_adj_loss_l2(batch_sz, lambda_loss, pixel_distance)]
    loss_name = "custom_adj_loss_l2_" + str(pixel_distance) + "_"

elif loss_name == "l1_weighted":
    loss = [custom_adj_loss_l1_weighted(batch_sz, lambda_loss, pixel_distance)]
    loss_name = "custom_adj_loss_l1_weighted_" + str(pixel_distance) + "_"

elif loss_name == "l2_weighted":
    loss = [custom_adj_loss_l2_weighted(batch_sz, lambda_loss, pixel_distance)]
    loss_name = "custom_adj_loss_l2_weighted_" + str(pixel_distance) + "_"

elif loss_name == "frobenius":
    loss = [custom_adj_loss_frobenius(batch_sz, lambda_loss, pixel_distance)]
    loss_name = "custom_adj_loss_frobenius_" + str(pixel_distance) + "_"

elif loss_name == "l2_2batch":
    loss = [custom_adj_loss_l2_different_adj_mat_for_dif_img(batch_sz, lambda_loss, pixel_distance)]
    loss_name = "2_mat_adj_" + str(pixel_distance) + "_"

else:
    print("Loss Name not value--END CODE")
    exit()

prefix = "Test_incremental_softmax_adj_" + loss_name + "_lambda_" + str(lambda_loss) + "_kern_init_" + kernel_init + \
         "_epochs_" + str(epochs_sz) + "_"

# create all directories for checkpoint weights graph..
path, pathTBoard, pathTChPoints, pathWeight = createDirectories(prefix=prefix, lr_p=lr_p, batch_sz=batch_sz,
                                                                h_img=h_img, mult_rate=mult_rate, dil_rate=dil_rate,
                                                                use_BN=batchNorm)

deeplab_model.compile(optimizer=optimizers.SGD(lr=lr_p, momentum=0.9, decay=0, nesterov=True),
                      loss=loss,
                      metrics=[tf.keras.metrics.CategoricalAccuracy(), metric_adj, metric_categ_cross])

deeplab_model.save_weights(os.path.join(path, 'common_weights_test_1.h5'))
