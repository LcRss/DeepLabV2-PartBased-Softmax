from tqdm import tqdm
from Utils import *

dir_img = "H:\Data_correct\segmentation_gray/results_argmax/test/"
inf_path = "H:\Test_Tesi_ordinati\Dati_utili\Ground_Truth_Train_Test_Val\gt_21_right_color/test/"
####

images = glob.glob(dir_img + "*.png")
images.sort()

for k in tqdm(range(len(images))):

    img = cv2.imread(images[k])

    imgNew = img[..., ::-1].copy()
    name = images[k]
    name = name[-15:]
    pathTmp = inf_path + "/" + name
    cv2.imwrite(pathTmp, imgNew)
