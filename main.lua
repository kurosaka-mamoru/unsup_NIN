require 'paths'
require 'xlua'
require 'image'
require 'unsup'
require 'nn'
package.path = package.path..';lib/kmeans-learning-torch/?.lua'
require 'kmeans'
require 'train-svm'

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
local nkernel = 200
local trNINsize = 1000

-- define net

local cheight = CIFAR_dim[2] - kSize + 1
local cwidth = CIFAR_dim[3] - kSize + 1
local pheight = torch.floor(cheight / 2)
local pwidth = torch.floor(cwidth / 2)

local recog_net = nn.Sequential()
recog_net:add(nn.SpatialConvolution(3, nkernel, kSize, kSize))
recog_net:add(nn.ReLU())
recog_net:add(nn.SpatialMaxPooling(pwidth, pheight, pwidth, pheight))

local compose_net = nn.Sequential()
compose_net:add(nn.SpatialConvolution(3, nkernel, kSize, kSize))
compose_net:add(nn.ReLU())
compose_net:add(nn.SpatialAveragePooling(cheight, cwidth, cheight, cwidth))

print('==> download dataset')
if not paths.dirp('cifar-10-batches-t7') then
  tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar10.t7.tgz'
  os.execute('wget ' .. tar)
  os.execute('tar xvf ' .. paths.basename(tar))
end

--[[
print("==> load dataset")
local trainData = {
  data = torch.Tensor(50000, CIFAR_dim[1] * CIFAR_dim[2] * CIFAR_dim[3]),
  labels = torch.Tensor(50000)
}

local testData = {
  data = torch.Tensor(10000, CIFAR_dim[1] * CIFAR_dim[2] * CIFAR_dim[3]),
  labels = torch.Tensor(10000)
}

local hollData = {
  data = torch.Tensor(60000, CIFAR_dim[1], CIFAR_dim[2], CIFAR_dim[3]),
  labels = torch.Tensor(60000)
}

for i = 0, 4 do
  local subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
  hollData.data[{ {i * 10000 + 1, (i+1) * 10000} }] = subset.data:t()
  hollData.labels[{ {i * 10000 + 1, (i+1) * 10000} }] = subset.labels
end

local subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
hollData.data[{ {50001, 60000} }] = subset.data:t()
hollData.labels[{ {50001, 60000} }] = subset.labels

hollData.labels = hollData.labels + 1
--]]

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

print("==> compose feature map (NIN)")
compose_net.modules[1].weight = kernels:reshape(200,3,7,7)

local output = torch.Tensor(trNINsize, nkernel)
for i = 1, trNINsize do
  output[i] = compose_net:forward(trainData.data[i])
end

local sigma = output:transpose(1,2) * output

for i = 1, nkernel do
  sigma[i] = sigma[i] / torch.sum(sigma[i])
end
