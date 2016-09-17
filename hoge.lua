require('mobdebug').start()
require("paths")
require("xlua")
require("image")
require("unsup")
require 'nn';
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
local nkernel = 200;

-- define net

local net = nn.Sequential()
net:add(nn.SpatialConvolution(3, nkernel, kSize, kSize))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))

print("==> download dataset")
if not paths.dirp('cifar-10-batches-t7') then
   tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar10.t7.tgz'
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

print("==> load dataset")
local trainData = {
   data = torch.Tensor(50000, CIFAR_dim[1]*CIFAR_dim[2]*CIFAR_dim[3]),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}
for i = 0, 4 do
   local subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]
trainData.data = trainData.data:reshape(trsize,CIFAR_dim[1],CIFAR_dim[2],CIFAR_dim[3])

local subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
local testData = {
   data = subset.data:t():float(),
   labels = subset.labels[1]:float(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

testData.data   = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]
testData.data = testData.data:reshape(tesize,3,32,32)


print("==> extract patches")
local numPatches = 50000
local patches = torch.zeros(numPatches, kSize*kSize*CIFAR_dim[1])
for i = 1,numPatches do
   xlua.progress(i,numPatches)
   local r = torch.random(CIFAR_dim[2] - kSize + 1)
   local c = torch.random(CIFAR_dim[3] - kSize + 1)
   patches[i] = trainData.data[{math.fmod(i-1,trsize)+1,{},{r,r+kSize-1},{c,c+kSize-1}}]
   patches[i] = patches[i]:add(-patches[i]:mean())
   patches[i] = patches[i]:div(math.sqrt(patches[i]:var()+10))
end

if opt.whiten then
   print("==> whiten patches")
   local function zca_whiten(x)
      local dims = x:size()
      local nsamples = dims[1]
      local ndims    = dims[2]
      local M = torch.mean(x, 1)
      local D, V = unsup.pcacov(x)
      x:add(torch.ger(torch.ones(nsamples), M:squeeze()):mul(-1))
      local diag = torch.diag(D:add(0.1):sqrt():pow(-1))
      local P = V * diag * V:t()
      x = x * P
      return x, M, P
   end
   patches, M, P = zca_whiten(patches)
end


print("==> find clusters")
local ncentroids = nkernel
kernels, counts = unsup.kmeans_modified(patches, ncentroids, nil, 0.1, 10, 1000, nil, true)


net.modules[1].weight = kernels:reshape(200,3,7,7)

