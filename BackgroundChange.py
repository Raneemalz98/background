import matplotlib

matplotlib.use('Agg')
from PIL import Image
import cv2
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from fastai.conv_learner import *
from fastai.dataset import *
from fastai.models.resnet import vgg_resnet50
import numpy as np
import json

PATH = Path(
    'C:/Users/thevirus/Desktop/Background-removal-using-deep-learning-master/Background-removal-using-deep-learning-master')
print(type(PATH))

tr_path = Path('E:/ite/emotion_detection-master/COCOdataset2017/CreatedData/train/CreatedTrain')
msk_path = Path('E:/ite/emotion_detection-master/COCOdataset2017/CreatedData/train/CreatedMaskImages')
train_dir = os.listdir('E:/ite/emotion_detection-master/COCOdataset2017/CreatedData/train/CreatedTrain')
msk_dir = os.listdir('E:/ite/emotion_detection-master/COCOdataset2017/CreatedData/train/CreatedMaskImages')

tr_path_test = Path('E:/ite/emotion_detection-master/COCOdataset2017/CreatedData/val/CreatedTrain')
# msk_path_test = Path('project/test_mask')
train_dir_test = os.listdir('E:/ite/emotion_detection-master/COCOdataset2017/CreatedData/val/CreatedMaskImages')
# msk_dir_test = os.listdir('project/test_mask')

ims = [open_image(tr_path / train_dir[i]) for i in range(16)]
im_masks = [open_image(msk_path / msk_dir[i]) for i in range(16)]

x_names = np.array([tr_path / o for o in train_dir])

y_names = np.array([msk_path / f'{o[:-4]}.jpg' for o in train_dir])

val_idxs = list(range(300))
((val_x, trn_x), (val_y, trn_y)) = split_by_idx(val_idxs, x_names, y_names)
print(len(val_x), len(trn_x))
print(trn_x[0])
aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]
sz = 512
bs = 8
tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y = y
        assert (len(fnames) == len(y))
        super().__init__(fnames, transform, path)

    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))

    def get_c(self): return 0


# def get_base():
#     layers = cut_model(f(True), cut)
#     return nn.Sequential(*layers)

def dice(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred * targs).sum() / (pred + targs).sum()


class StdUpsample(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(nout)

    def forward(self, x): return self.bn(F.relu(self.conv(x)))


class Upsample34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.features = nn.Sequential(
            rn, nn.ReLU(),
            StdUpsample(512, 256),
            StdUpsample(256, 256),
            StdUpsample(256, 256),
            StdUpsample(256, 256),
            nn.ConvTranspose2d(256, 1, 2, stride=2))

    def forward(self, x): return self.features(x)[:, 0]


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:, 0]

    def close(self):
        for sf in self.sfs: sf.remove()


class UnetModel():
    def __init__(self, model, lr_cut, name='unet'):
        self.model, self.name = model, name
        self.lr_cut = lr_cut

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [self.lr_cut]))
        return lgs + [children(self.model)[1:]]

    def dice(pred, targs):
        pred = (pred > 0).float()
        return 2. * (pred * targs).sum() / (pred + targs).sum()

    def show_img(im, figsize=None, ax=None, alpha=None):
        if not ax: fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(im, alpha=alpha)
        ax.set_axis_off()
        return ax

    def OpenFile(self):
        self.name = askopenfilename(initialdir="C:/Users/thevirus/Desktop",
                                    filetypes=(("jpeg files", "*.png"), ("all files", "*.*")),
                                    title="Choose a image."
                                    )
        print(self.name)
        input_img = Image.open(self.name)
        input_img = input_img.resize((512, 512))
        input_img.save(
            r'C:\Users\thevirus\Desktop\Background-removal-using-deep-learning-master\Background-removal-using-deep-learning-master\background images\building\new.png')
        input_img = Image.open(
            r'C:\Users\thevirus\Desktop\Background-removal-using-deep-learning-master\Background-removal-using-deep-learning-master\background images\building\new.png')

    # input_img = PhotoImage(file=self.name)
    # input_img.save(r'C:\Users\thevirus\Desktop\Background-removal-using-deep-learning-master\Background-removal-using-deep-learning-master\background images\building\new.png')
    # input_img = Image.open(r'C:\Users\thevirus\Desktop\Background-removal-using-deep-learning-master\Background-removal-using-deep-learning-master\background images\building\new.png')
    def get_base(self):
        layers = cut_model(self.f(True), self.cut)
        return nn.Sequential(*layers)

    def ChangeFile(self):
        # print(self.name)
        pp = Path(
            r'C:\Users\thevirus\Desktop\Background-removal-using-deep-learning-master\Background-removal-using-deep-learning-master\background images\building\new.png')
        # pp = Path(self.name)
        x_test = np.array([pp])
        #         print(pp)
        #         print(type(pp))
        datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x, trn_y), (val_x, val_y), tfms, test=(x_test, x_test),
                                    path=PATH)
        md = ImageData(PATH, datasets, bs, num_workers=4, classes=None)
        denorm = md.trn_ds.denorm
        self.f = resnet34
        self.cut, self.lr_cut = model_meta[self.f]
        self.m_base = self.get_base(self)
        self.m = to_gpu(Unet34(self.m_base))
        models = UnetModel(self.m, self.lr_cut)
        learn = ConvLearner(md, models)
        learn.opt_fn = optim.Adam
        learn.crit = nn.BCEWithLogitsLoss()
        learn.metrics = [accuracy_thresh(0.5), dice]
        learn.load('total_new_512urn-tmp16')
        x, y = next(iter(md.test_dl))
        self.py = to_np(learn.model(V(x)))
        self.pyy = self.py[0] > 0
        for i in range(10):
            for row_index in range(len(self.pyy)):  # Loops through each row in the array
                for col_index in range(len(self.pyy[row_index])):
                    s = 0  # Loops through each column in the row
                    # Make sure you check for out of bounds
                    if (row_index > 0 and self.pyy[row_index - 1][col_index]):
                        s = s + 1
                    if (row_index < len(self.pyy) - 1 and self.pyy[row_index + 1][col_index]):
                        s = s + 1
                    if (col_index > 0 and self.pyy[row_index][col_index - 1]):
                        s = s + 1
                    if (col_index < len(self.pyy[0]) - 1 and self.pyy[row_index][col_index + 1]):
                        s = s + 1
                    if (s > 2):
                        self.pyy[row_index][col_index] = True
                    if (s < 1):
                        self.pyy[row_index][col_index] = False

                    s = 0  # Loops through each column in the row
                    # Make sure you check for out of bounds
                    if (row_index > 4 and self.pyy[row_index - 5][col_index]):
                        s = s + 1
                    if (row_index < len(self.pyy) - 5 and self.pyy[row_index + 5][col_index]):
                        s = s + 1
                    if (col_index > 4 and self.pyy[row_index][col_index - 5]):
                        s = s + 1
                    if (col_index < len(self.pyy[0]) - 5 and self.pyy[row_index][col_index + 5]):
                        s = s + 1
                    if (s > 2):
                        self.pyy[row_index][col_index] = True
                    if (s < 1):
                        self.pyy[row_index][col_index] = False
                    # print(self.pyy.shape)
        for i in range(2):
            for row_index in range(len(self.pyy)):  # Loops through each row in the array
                for col_index in range(len(self.pyy[row_index])):
                    s = 0  # Loops through each column in the row
                    # Make sure you check for out of bounds
                    if (row_index > 1 and self.pyy[row_index - 2][col_index]):
                        s = s + 1
                    if (row_index < len(self.pyy) - 2 and self.pyy[row_index + 2][col_index]):
                        s = s + 1
                    if (col_index > 1 and self.pyy[row_index][col_index - 2]):
                        s = s + 1
                    if (col_index < len(self.pyy[0]) - 2 and self.pyy[row_index][col_index + 2]):
                        s = s + 1
                    if (s > 2):
                        self.pyy[row_index][col_index] = True
        #           if(s<1):
        #              self.pyy[row_index][col_index] = False
        #   def ChangeBackground(self):
        # هون شغلك يا برهوم
        self.city_img = cv2.imread(
            'C:/Users/thevirus/Desktop/Background-removal-using-deep-learning-master/Background-removal-using-deep-learning-master/background images/6.png')
################## #########
        self.city_img = self.city_img.resize((512, 512))
        self.city_img.save(
            r'C:\Users\thevirus\Desktop\Background-removal-using-deep-learning-master\Background-removal-using-deep-learning-master\background images\building\new.png')
        self.city_img = Image.open(
            r'C:\Users\thevirus\Desktop\Background-removal-using-deep-learning-master\Background-removal-using-deep-learning-master\background images\building\new.png')
####################
        self.imgg = cv2.imread(
            r'C:\Users\thevirus\Desktop\Background-removal-using-deep-learning-master\Background-removal-using-deep-learning-master\background images\building\new.png')
        print(self.imgg.shape)
        # s1 = np.resize(self.imgg, (512,512,3))
        # s = np.resize(self.pyy, (512,512,3))
        # self.pyy = s
        # self.imgg=s1
        print(self.pyy.shape)
        print(self.pyy)
        for i in range(3):
            print(i)
            self.imgg[:, :, i] = self.imgg[:, :, i] * self.pyy
        for i in range(3):
            print(i)
            self.city_img[:, :, i] = self.city_img[:, :, i] * (~self.pyy)
        print('2')
        self.total = self.city_img + self.imgg
        cv2.imwrite('total2.png', self.total)


class UpsampleModel():
    def __init__(self, model, name='upsample'):
        self.model, self.name = model, name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model.features)[1:]]


def Start1():
    x = UnetModel

    x.OpenFile(x)
    x.ChangeFile(x)


#   x.ChangeBackground(x)
Start1()
