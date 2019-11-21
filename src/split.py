import sys
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def denoize(cv_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cv_img = cv2.morphologyEx(cv_img, cv2.MORPH_OPEN, kernel)
    cv_img = cv2.morphologyEx(cv_img, cv2.MORPH_CLOSE, kernel)
    return cv_img

def draw_kps(cv_img, kps, labels):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    for kp, label in zip(kps, labels):
        cv2.circle(cv_img, tuple(map(int, kp)), 2, colors[label], -1)
    return cv_img

def draw_centers(cv_img, centers):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    for i, center in enumerate(centers):
        cv2.drawMarker(cv_img, tuple(map(int, center)), colors[i],
            markerType=cv2.MARKER_TILTED_CROSS, thickness=3)
    return cv_img

def draw_bbox(cv_img, kclass):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    for i, c in enumerate(kclass):
        cv2.rectangle(cv_img, tuple(map(int, c["min"])),
            tuple(map(int, c["max"])), colors[i], 3)
    return cv_img

def gmplot(gm, hist):
    # plot graph
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    colors = ["r", "g", "b", "y"]
    y = np.linspace(0, len(hist), len(hist))
    for i in range(gm.n_components):
        plt.plot(y, gm.weights_[i] * norm.pdf(y, gm.means_[i], np.sqrt(gm.covariances_[i]))[0],
                 color=colors[i])
    plt.bar(y,hist, width=1, color="k")
    plt.show()

def hist(imagePath, k=3):
    pil_img = Image.open( imagePath )
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    th, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    denoize_img = denoize(binary_img)
    hist = np.array([sum([1 for pixel in row if pixel == 0]) for row in binary_img])
    
    data = []
    for i, n in enumerate(hist):
        data.extend([i]*n)
    gm = GaussianMixture(k)
    gm.fit(np.array(data).reshape(-1, 1))
    prob = gm.predict_proba(np.linspace(0, len(hist), len(hist)).reshape(-1, 1))
    hist_labels = [np.argmax(p) for p in prob]

    #gmplot(gm, hist/sum(hist))

    order = list( dict.fromkeys( hist_labels ) )
    images = []
    top = 0
    for o in order:
        hsize = hist_labels.count( o )
        images.append( Image.fromarray( gray_img[ top : top + hsize ] ) )
        top += hsize
    return images
    #後で消す
    #colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    #hist_img = np.array([
    #    [colors[l] if i < n else (255, 255, 255) for i in range(gray_img.shape[1])]
    #    for n, l in zip(hist, hist_labels)],
    #    dtype=np.uint8)
    
    #result = cv2.hconcat((cv_img, cv2.cvtColor(denoize_img, cv2.COLOR_GRAY2BGR), hist_img))

    #return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #ここまで
def kmeans(imagePath, k=3, dtype="SIFT", binarize=True):
    if dtype == "SIFT":
        detector = cv2.xfeatures2d.SIFT_create()
    elif dtype == "AKAZE":
        detector = cv2.AKAZE_create()

    pil_img = Image.open(imagePath)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    if binarize:
        th, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
        denoize_img = denoize(binary_img)
        kps = [kp.pt for kp in detector.detect(denoize_img)]
    else:
        kps = [kp.pt for kp in detector.detect(cv_img)]

    if len(kps) == 0:
        print("Cannot detect any keypoint.")
        sys.exit()

    kmeans_model = KMeans(n_clusters=k, random_state=10).fit(kps)
    #kmeans_img = draw_kps(cv_img, kps, kmeans_model.labels_)
    #kmeans_img = draw_centers(kmeans_img, kmeans_model.cluster_centers_)
    
    kmeans_class = [{
        "pts": np.array([kp for kp, label in zip(kps, kmeans_model.labels_) if label == idx]).T,
        "center": center
        } for idx, center in enumerate(kmeans_model.cluster_centers_)]
    for i in range(k):
        kmeans_class[i]["min"] = (min(kmeans_class[i]["pts"][0]), min(kmeans_class[i]["pts"][1]))
        kmeans_class[i]["max"] = (max(kmeans_class[i]["pts"][0]), max(kmeans_class[i]["pts"][1]))
    #kmeans_img = draw_bbox(kmeans_img, kmeans_class)
    kmeans_class.sort(key=lambda x:x["center"][1])

    imgs = []
    for i in range(3):
        x1 = kmeans_class[i]['min'][0]
        x2 = kmeans_class[i]['max'][0]
        y1 = kmeans_class[i]['min'][1]
        y2 = kmeans_class[i]['max'][1]
        xwidth = x2 - x1
        ywidth = y2 - y1
        xmin = x1 - 0.2*xwidth
        xmax = x2 + 0.2*xwidth
        ymin = y1 - 0.2*ywidth
        ymax = y2 + 0.2*ywidth
        
        #imgs.append(gray_img[int(xmin):int(xmax), int(ymin):int(ymax)])
        imgs.append(gray_img[int(max(ymin,0)):int(min(ymax,gray_img.shape[0])), int(max(xmin,0)):int(min(xmax,gray_img.shape[1]))])
        #imgs.append(gray_img[int(max(kmeans_class[i]['min'][1]-20,0)):int(kmeans_class[i]['max'][1]+20), int(max(kmeans_class[i]["min"][0]-10,0)):int(kmeans_class[i]["max"][0]+10)])
    
    return [Image.fromarray(i) for i in imgs]

    #black kmeans
def kmeans_black(imagePath, k=3, binarize=False):
##二値化
# 画像の読み込み
    cv_img = cv2.imread(imagePath)

    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    ret, img_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

#黒点を抽出する
    black_points = []

    '''
    for y in range(img_otsu.shape[0]):
        for x in range(img_otsu.shape[1]):
            if img_otsu[y, x] == 0:
                black_points.append((x,y))
    ''' 
    img_otsu_array = np.array(img_otsu).T
    black_points = np.argwhere(img_otsu_array == 0)

    #change to numpy array 
    kps = np.array(black_points)
    
    ##clustering with kmeans
    kmeans_model = KMeans(n_clusters=k, random_state=10).fit(kps)
    kmeans_img = draw_kps(cv_img, kps, kmeans_model.labels_)
    kmeans_img = draw_centers(kmeans_img, kmeans_model.cluster_centers_)
    
    kmeans_class = [{
        "pts": np.array([kp for kp, label in zip(kps, kmeans_model.labels_) if label == idx]).T,
        "center": center
        } for idx, center in enumerate(kmeans_model.cluster_centers_)]
    for i in range(k):
        kmeans_class[i]["min"] = (min(kmeans_class[i]["pts"][0]), min(kmeans_class[i]["pts"][1]))
        kmeans_class[i]["max"] = (max(kmeans_class[i]["pts"][0]), max(kmeans_class[i]["pts"][1]))
    kmeans_img = draw_bbox(kmeans_img, kmeans_class)
    kmeans_class.sort(key=lambda x:x["center"][1])

    imgs = []
    for i in range(3):
        x1 = kmeans_class[i]['min'][0]
        x2 = kmeans_class[i]['max'][0]
        y1 = kmeans_class[i]['min'][1]
        y2 = kmeans_class[i]['max'][1]
        xwidth = x2 - x1
        ywidth = y2 - y1
        xmin = x1 - 0.2*xwidth
        xmax = x2 + 0.2*xwidth
        ymin = y1 - 0.2*ywidth
        ymax = y2 + 0.2*ywidth
        
        #imgs.append(gray_img[int(xmin):int(xmax), int(ymin):int(ymax)])
        imgs.append(gray[int(max(y1,0)):int(min(y2,gray.shape[0])), int(max(x1,0)):int(min(x2,gray.shape[1]))])
    
    #test save 
    #cv2.imwrite(imagePath.replace("jpg","")+"_test.jpg",kmeans_img)
    return [Image.fromarray(i) for i in imgs]
    #return Image.fromarray(cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    img = Image.open(sys.argv[1])
    split_imgs = hist(img)
    # split_img = split_kmeans(img, k=3)
    split_img.show()
    if len(sys.argv) == 3:
        for i, split_img in enumerate(split_imgs):
            split_img.save(str(i)+"_"+sys.argv[2])
