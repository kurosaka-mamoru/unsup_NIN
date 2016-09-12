require("paths")
require("xlua")
require("image")
require("unsup")

package.path = package.path..';lib/kmeans-learning-torch/?.lua'
require("kmeans")
require("extract")
require("train-svm")

torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
opt = {
   whiten = true,
}
