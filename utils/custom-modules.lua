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

require '../ElemwiseClassNLL.lua'
require '../ElemwiseCrossEntropy.lua'


-- References:
-- [1] He et al 2015, Deep Residual Learning for Image Recognition
-- [2] He et al 2016, Identity Mappings in Deep Residual Networks
-- [3] He et al 2015, Delving Deeper Into Rectifiers

modules = {}

function shrink(size,n)
   return math.ceil(size/(2^n))
end

function modules.MetaRobustLayer(nIn, nOut, weightNet)
   model = nn.Sequential()
   model:add(nn.ParallelTable())
   -- First channel is the ElemwiseClassNLL
   model:add(nn.ElemwiseCrossEntropy())
   model:add(nn.Identity())
   model:add(nn.JoinTable())
   model:add(weightNet)
end

function modules.ConvBlock(nInput, nOutput, stride, filterSize, padding)
    -- Default arguments
    stride = stride or 1
    filterSize = filterSize or 3
    padding = padding or (filterSize-(filterSize%2))/2
    standardActivation = modules.StandardActivation(nOutput)
    return nn.Sequential()
                :add(nn.SpatialConvolutionMM(
                        nInput,nOutput,filterSize,filterSize,stride,stride,padding,padding))
                :add(standardActivation)
end

return modules
