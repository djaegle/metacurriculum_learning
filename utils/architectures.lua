-- Standard packages
require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'nnx'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'csvigo'
local debugger = require('fb.debugger')

local function shrink(size,n)
   return math.ceil(size/(2^n))
end

architectures = {}


-- References:
-- Cir2010: Ciresan et al 2010 - Deep Big Simple Neural Nets Excel on Handwritten Digit Recognition
-- Cir2012: Ciresan et al 2012 - Multi-column Deep Neural Networks for Image Classification
-- Wan2013: Wan et al - Regularization of Neural Networks using DropConnect

function architectures.linearDemo(nIn,nOut)
   -- A simple one-layer linear perceptron, used in the original MNIST demo.
   return nn.Sequential():add(nn.Linear(nIn,nOut))
end

function architectures.Cir2010_4(imsize,noutputs,opt)
   -- Permutation invariant architecture.
   -- TODO: finish implementation

   -- The best performing MLP from Ciresan et al 2010 (architecture 4 in table 1)
   -- gives 0.35 validation, 0.32 test error performance.
   -- The whole undeformed training set is used for validation; only deformations used
   -- for training.

   -- TODO: initialize weights with uniform distr. on [-0.05, 0.05]
   -- TODO: input pixels should be mapped to [-1.0,1.0] by dividing by 127.5 and subtracting 1
   -- TODO: use Ciresan deformation augmentation (p. 4)

   -- MLP: 2500, 2000, 1500, 1000, 500, 10
   -- activation function is a*tanh(b*y), a=1.7159, b=0.6666
    model = nn.Sequential()
    model:add(nn.SpatialConvolutionMM(
            nInput,nOutput,filterSize,filterSize,stride,stride,padding,padding))

    model:add(modules.ConvBlock(2, 1024, 2, 3, 1))
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],1), shrink(imsize[2],1),opt))
    return model
end

function architectures.Cir2010_4_ReLU(nIn,nOut)
   -- Permutation invariant architecture.
   -- The same as Cir2010_4 but with ReLU instead of tanh
   -- local nUnits = {2500,2000,1500,1000,500}

   local nUnits = {32,64}

   local model = nn.Sequential()
   local nPrev = nIn
   for i = 1,table.getn(nUnits) do
      model:add(nn.Linear(nPrev,nUnits[i])):add(nn.ReLU(true))
      nPrev = nUnits[i]
   end

   -- Output layer
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(nPrev,nOut))

   return model
end

function architectures.Wan2013_CNN(nFiltersIn, nFiltersOut, xDim)
   -- The basic CNN architecture used in the DropConnect paper (results in table 3). Uses standard ReLU
   -- nonlinearities. Note that in their best results, drop-connect doesn't actually improve over
   -- the no-drop condition (the reported 0.21% error is an ensemble result, and shows up
   -- in no-drop as well as drop-connect conditions).
   -- This is effectively the same architecture as in Ciresan et al 2012, except for the use of ReLU instead
   -- of tanh.

   -- Note: we assume all input images are square for convenience.

   local nUnits = {32,64} -- number of feature planes
   local kWConv = {5,5}
   local kWPool = {2,2}
   local stridesConv = {1,1}
   local stridesPool = {2,2}
   local model = nn.Sequential()
   local nPrev = nFiltersIn

   for i = 1,table.getn(nUnits) do
      local padW = (kWConv[i]-1)/2 -- pad to maintain size
      model:add(nn.SpatialConvolutionMM(nPrev,nUnits[i],kWConv[i],kWConv[i],stridesConv[i],stridesConv[i],padW,padW))
         :add(nn.ReLU(true))
         :add(nn.SpatialMaxPooling(kWPool[i],kWPool[i],stridesPool[i],stridesPool[i],0,0))
      nPrev = nUnits[i]
   end

   -- Reshape output of last conv layer to single vector
   -- Output layer
   local dimOutFlat = shrink(xDim,table.getn(nUnits))^2*nPrev
   model:add(nn.Reshape(dimOutFlat,true))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(dimOutFlat,nFiltersOut))

   return model

   -- TODO: note the difference in preprocessing between Wan et al and Ciresan et al 2012

   -- TODO: initialize weights (N(0,0.1) for FC, N(0,0.01) for conv)
   -- TODO: preprocessing (image mean subtracted)
   -- TODO: implement the schedule used (600-400-200 epochs, lr=0.1,multipliers: 1,0.5,0.1,{0.05,0.01,0.005,0.001})
   --    So: run at lr=0.1 for 600 epochs; run for 400 epochs, decaying by 0.5 each epoch; run for 400 epochs, decaying at 0.1
   --    each epoch; run for 200 epochs decaying at 0.05 each epoch, etc.
   -- TODO: no momentum in training here.

   -- 32 conv (4x4s1) -> 2x2s2 max pool -> 64 (5x5s1) conv -> 3x3s3 max pool -> 150 FC -> 10

end

return architectures
