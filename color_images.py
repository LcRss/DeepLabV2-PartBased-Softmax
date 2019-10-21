from tqdm import tqdm
from Utils import *

dir_img = "H:\Data_correct\segmentation_gray\gray/val/"
inf_path = "H:\Test_Tesi_ordinati\Dati_utili\Ground_Truth_Train_Test_Val\gt_21_right/val/"

####

images = glob.glob(dir_img + "*.png")
images.sort()

pathCMap = 'Y:/tesisti/rossi/cmap255.mat'
fileMat = loadmat(pathCMap)
cmap = fileMat['cmap']

for k in tqdm(range(len(images))):

    img = cv2.imread(images[k], cv2.IMREAD_GRAYSCALE)
    h_z, w_z = img.shape
    imgNew = np.zeros((h_z, w_z, 3), np.uint8)
    for i in range(1, 21):
        mask = cv2.inRange(img, i, i)
        v = cmap[i]
        # v = v[::-1]
        imgNew[mask > 0] = v

    imgNew = imgNew[..., ::-1]
    name = images[k]
    name = name[-15:]

    # cv2.imshow("2",imgNew)
    # cv2.waitKey()
    pathTmp = inf_path + "/" + name
    cv2.imwrite(pathTmp, imgNew)
