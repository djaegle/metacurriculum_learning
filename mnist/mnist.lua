--[[
Modified from example mnist execution code in torchnet.
Copyright (c) 2016-present, Facebook, Inc.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
]]--

torch.setdefaulttensortype('torch.FloatTensor')

--TODO: generalize training procedure to pairs of images for siamese 2AFC training
--TODO: given a batch, figure out which pairs of images to use for the comparison. Basically, the siamese thing only requires
-- across-image comparisons at the very end.
   -- TODO: remove notes for Metacurriculum learning project

-- TODO: Implement metacurriculum learning strategies:
   -- (1) RL algorithms using validation loss
   -- (2) simple deterministic learned robust loss: E(w(L)), with weighting function w(L,class,f(img))
   -- (2a) more sophisticated version using reparameterization trick and ensuring balanced sets
   -- Others?

-- TODO: add network/state save functionality

-- TODO: add general optim support instead of just vanilla SGD

-- TODO: add hooks for better hyperparameter schedules

-- TODO: add hooks for more validation

-- load torchnet:
local tnt = require 'torchnet'
local debugger = require('fb.debugger')

local architectures = require('../utils/architectures')
local weight_init = require('../utils/weight-init')
local trans = require('../utils/transforms')

-- Command line options
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
cmd:option('-perm_invar', false, 'permutation invariant: reshape to 1d')
cmd:option('-lr', 0.01, 'learning rate')
cmd:option('-mu', 0.0, 'SGD Momentum') -- Set to zero for the moment
cmd:option('-maxepoch', 5, 'Maximum number of epochs to run')
cmd:option('-batchsize',128, 'Batch size')
cmd:option('-augMode','','Data augmentation mode')
cmd:option('-normMode','mnistZeroMean','Input data normalization')
cmd:option('-metaMode','','Metacurriculum learning mode')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

-- function that sets of dataset iterator:
local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init    = function() require 'torchnet' end,
      closure = function() -- will repeatedly call dataset:get()

         -- load MNIST dataset:
         local mnist = require 'mnist'
         local dataset = mnist[mode .. 'dataset']()
         local augmentFunction = trans.spatialTransform(config)
         local normFunction = trans.dataNormTransform(config)

         -- Make images 1d: full dataset
         if config.perm_invar then
            dataset.data = dataset.data:reshape(dataset.data:size(1),
               dataset.data:size(2) * dataset.data:size(3)):float()
         else
            dataset.data = dataset.data:float()
         end

         -- Duplicate labels as doubles for regression
         -- TODO: remove for Metacurriculum learning project
         --dataset.intlabel = torch.FloatTensor(dataset.label:size()):copy(dataset.label)

         -- return batches of data:
         return tnt.BatchDataset{
            batchsize = config.batchsize, -- Can get this > 10k with no trouble
            dataset = tnt.TransformDataset { -- apply transform closure
               transform = function(sample)
                  sample = augmentFunction(normFunction(sample))
                  return {
                     input = sample.input,
                     target = sample.target,
                  }
               end, -- closure for transformation
               dataset = tnt.ShuffleDataset { -- Always shuffle w/ replacement each epoch
                  dataset = tnt.ListDataset{  -- replace this by your own dataset
                     list = torch.range(1, dataset.data:size(1)):long(),
                     load = function(idx)

                     local input
                     if config.perm_invar then
                        input = dataset.data[idx]
                     else
                        input = dataset.data[{{idx},{},{}}]
                     end
                        return {
                           input = input,
                           target = torch.LongTensor{dataset.label[idx] + 1},
                        }  -- sample contains input and target
                     end,
                  }
               }
            }
         }
      end,
   }
end

-- set up logistic regressor:
local xDim = 28 -- TODO: make flexible for cropping, as in 24x24 for Wan2013
local nLabels = 10
local nImChans = 1
local net
if config.perm_invar then
   net = architectures.Cir2010_4_ReLU(xDim^2,nLabels)
else
   net = architectures.Wan2013_CNN(nImChans,nLabels,xDim)
end

net = weight_init(net,'kaiming')

local criterion = nn.CrossEntropyCriterion()

-- set up training engine:
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter() -- Low better
local clerr  = tnt.ClassErrorMeter{topk = {1}} -- Low better
engine.hooks.onStartEpoch = function(state)
   -- Add epoch-wise evaluation here.
   meter:reset()
   clerr:reset()
end
engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   if state.training then
      print(string.format('avg. loss: %2.4f; avg. error: %2.4f',
         meter:value(), clerr:value{k = 1}))
   end
end

-- set up GPU training:
if config.usegpu then

   -- copy model to GPU:
   require 'cunn'
   net       = net:cuda()
   criterion = criterion:cuda()

   -- copy sample to GPU buffer:
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end
end

-- train the model:
engine:train{
   network   = net,
   iterator  = getIterator('train'),
   criterion = criterion,
   lr        = config.lr,
   maxepoch  = config.maxepoch,
}

-- measure test loss and error:
meter:reset()
clerr:reset()
engine:test{
   network   = net,
   iterator  = getIterator('test'),
   criterion = criterion,
}
print(string.format('test loss: %2.4f; test error: %2.4f',
   meter:value(), clerr:value{k = 1}))
