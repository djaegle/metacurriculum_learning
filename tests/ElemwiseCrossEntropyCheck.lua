-- Check implementation of ElemwiseClassNLL and ElemwiseCrossEntropy using finite differences

require 'nn'
require 'optim'
require '../ElemwiseClassNLL.lua'
require '../ElemwiseCrossEntropy.lua'
debugger = require('fb.debugger')

module = nn.ElemwiseCrossEntropy()
--module = nn.ElemwiseClassNLL()

-- input_img = torch.rand(5,10) -- 5 ims, 10 classes
target_class = 3

-- Correct setting
-- input_img = torch.Tensor(5,10):fill(-100)
-- input_img[{{},{target_class}}] = 100

-- Random setting
input_img = torch.rand(5,10)

input = {input_img,target}

function feval(x)
  local target = torch.Tensor(5):fill(target_class)
  -- debugger.enter()
  input = {x,target}
  module:forward(input)
  module:backward(input, torch.ones(input[2]:size()))
  return module.output:sum(), module.gradInput:reshape(input[1]:numel())
end

function forward_est(x)
  local target = torch.Tensor(5):fill(target_class)
  -- debugger.enter()
  input = {x,target}
  module:forward(input)
  module:backward(input, torch.ones(input[2]:size()))
  return module.output
end

function true_forward_est(x)
  local target = torch.Tensor(5):fill(target_class)
  local criterion = nn.ClassNLLCriterion()
  local err = criterion:forward(x,target)
  return err
end

diff,dc,dc_est = optim.checkgrad(feval,input_img)
-- print('Diff must be close to 1e-8: diff = ' .. diff)
-- print(dc)
-- print(dc_est)


-- Try it manually:
eps = 1e-5
input_img2 = torch.Tensor(input_img:size()):copy(input_img)
input_img2[{{1},{1}}] = input_img2[{{1},{1}}] + eps

input_img3 = torch.Tensor(input_img:size()):copy(input_img)
input_img3[{{1},{1}}] = input_img3[{{1},{1}}] - eps
dinput = (forward_est(input_img3)-forward_est(input_img2))/(2*eps)
print(dinput)

-- Test by comparing results with the built-in ClassNLLCriterion
test_result = forward_est(input_img)
true_result = true_forward_est(input_img)
print('Diff between avg of ElemwiseClassNLL and built-in should be small: diff = ' .. torch.mean(test_result)-true_result)
