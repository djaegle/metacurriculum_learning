-- Check implementation of ElemwiseClassNLL and ElemwiseCrossEntropy using finite differences

require 'nn'
require 'optim'
require '../ElemwiseClassNLL.lua'
require '../ElemwiseCrossEntropy.lua'
debugger = require('fb.debugger')

local n_classes = 10
local n_ims = 5

-- Correct setting:
local target_class = 3
local target = torch.Tensor(n_ims):fill(target_class)
local input_img = torch.Tensor(n_ims,n_classes):fill(-100)
input_img[{{},{target_class}}] = 100


-- Random guess setting
-- local target = torch.Tensor(n_ims):random(1,n_classes)
-- local input_img = torch.rand(n_ims,n_classes)


local input = {input_img,target}

function forward_est(x,target)
  local module = nn.ElemwiseCrossEntropy()

  input = {x,target}
  local err = module:forward(input)
  -- Put in a setting where this is mimicking acting as the loss:
  -- The dL/doutput is just 1/n_ims - average the imwise losses
  local t = module:backward(input, torch.Tensor(input[2]:size()):fill(1/n_ims))
  return torch.mean(err), t
end

function true_forward_est(x,target)
  local criterion = nn.CrossEntropyCriterion()
  local err = criterion:forward(x,target)
  local t = criterion:backward(x,target)
  return err, t
end

-- Test by comparing results with the built-in ClassNLLCriterion
test_result, test_deriv = forward_est(input_img,target)
true_result, true_deriv = true_forward_est(input_img,target)
print('Forward: Diff between avg of ElemwiseClassNLL and built-in should be small: diff = '
  .. test_result-true_result)
print('Backward: Diff between avg of ElemwiseClassNLL and built-in should be small: diff: ')
print(test_deriv-true_deriv)

-- print(test_result)
-- print(true_result)
