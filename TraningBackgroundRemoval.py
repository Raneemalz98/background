from fastai.conv_learner import *
from fastai.dataset import *
from fastai.models.resnet import vgg_resnet50
from fastai.models.unet import *
import json
TRAIN_DN = 'C:/Users/thevirus/Desktop/Background-removal-using-deep-learning-master/Background-removal-using-deep-learning-master/ds'
MASKS_DN = 'C:/Users/thevirus/Desktop/Background-removal-using-deep-learning-master/Background-removal-using-deep-learning-master/ds/d'
PATH = Path('C:/Users/thevirus/Desktop/Background-removal-using-deep-learning-master')
sz = 128
bs = 4
nw = 16
tr_path = Path('C:/Users/thevirus/Desktop/COCO-Semantic-Segmentation-master/CreatedData/train/CreatedTrain')
msk_path = Path('C:/Users/thevirus/Desktop/COCO-Semantic-Segmentation-master/CreatedData/train/CreatedMaskImages')
train_dir = os.listdir('C:/Users/thevirus/Desktop/COCO-Semantic-Segmentation-master/CreatedData/train/CreatedTrain')
msk_dir = os.listdir('C:/Users/thevirus/Desktop/COCO-Semantic-Segmentation-master/CreatedData/train/CreatedMaskImages')
#open_image(TRAIN_DN/train_dir[0])
ims = [open_image(tr_path/train_dir[i]) for i in range(16)]
im_masks = [open_image(msk_path/msk_dir[i]) for i in range(16)]
#print(train_dir[:5])
class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): return 0

x_names = np.array([tr_path / o for o in train_dir])
y_names = np.array([msk_path/f'{o[:-4]}.jpg' for o in train_dir])
val_idxs = list(range(300))
((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names, y_names)
len(val_x),len(trn_x)
print(trn_x[:5])
aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]
tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=16, classes=None)
denorm = md.trn_ds.denorm
f = resnet34
cut,lr_cut = model_meta[f]
def get_base():
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)
def dice(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


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
class UpsampleModel():
    def __init__(self,model,name='upsample'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model.features)[1:]]

m_base = get_base()
m = to_gpu(Upsample34(m_base))
models = UpsampleModel(m)
learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5),dice]
learn.freeze_to(1)
lr=4e-2
wd=1e-7
lrs = np.array([lr/100,lr/10,lr])/2
x,y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))
def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax
class SaveFeatures():
    features=None
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
    def __init__(self,model,name='unet'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]
m_base = get_base()
m = to_gpu(Unet34(m_base))
models = UnetModel(m)
learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5),dice]
learn.summary()
[o.features.size() for o in m.sfs]
learn.freeze_to(1)
class SaveBestModel(LossRecorder):
    def __init__(self, model, lr, name='best_model'):
        super().__init__(model.get_layer_opt(lr, None))
        self.name = name
        self.model = model
        self.best_loss = None
        self.best_acc = None

    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        loss, acc = metrics
        if self.best_acc == None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.model.save(f'{self.name}')
        elif acc == self.best_acc and  loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')
learn.unfreeze()
learn.bn_freeze(True)
m.close()
sz=512
bs=4
tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=4, classes=None)
denorm = md.trn_ds.denorm

m_base = get_base()
m = to_gpu(Unet34(m_base))
models = UnetModel(m)
learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5),dice]
learn.freeze_to(1)
learn.load('E:/ite/emotion_detection-master/total_new_512urn-tmp16')
for i in range(5):
    print('starting with epoch   ')
    print('loading')
    if i > 0 :
        learn.load('total_new_512urn-tmp15')
    print('loaded')
    print('traning')
    learn.fit(lr,1,wds=wd, cycle_len=1,use_clr=(5,5))
    print('traned')
    print('saving')
    learn.save('total_new_512urn-tmp15')
    print('saved')
    print('ending epoch  ')
