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

-- set parameters
local CIFAR_dim = {3, 32, 32}
local trsize = 50000
local tesize = 10000
local kSize = 7
local nkernel1 = 32
local nkernel2 = 32
local fanIn1 = 1
local fanIn2 = 4

print("==> download dataset")
if not paths.dirp('cifar-10-batches-t7') then
   tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar10.t7.tgz'
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

