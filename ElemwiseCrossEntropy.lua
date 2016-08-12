-- Implements the CrossEntropy criterion for logistic regression as a forward layer
-- to allow robust kernel learning as a function of the imagewise CrossEntropy loss.

require 'nn'

local ElemwiseCrossEntropy, parent = torch.class('nn.ElemwiseCrossEntropy','nn.Module')

-- Effectively we've just transmuted CrossEntropyCriterion into a layer, and
-- done the same for ElemwiseClassNLL. The only real differences are in the
-- computation of ElemwiseClassNLL, which is now done without collapsing
-- input into a


function ElemwiseCrossEntropy:__init()
  -- no parameters here: just need to pass forward the loss
  parent.__init(self)
  self.lsm = nn.LogSoftMax()
  self.nll = nn.ElemwiseClassNLL()
  self.output = torch.zeros(1)
end


function ElemwiseCrossEntropy:updateOutput(input)
  -- Input is a table of batch images and batch targets
  target = input[2]
  input_img = input[1]

  input_img = input_img:squeeze() -- NB: only remove singleton dims, so should still be here
  target = type(target) == 'number' and target or target:squeeze()
  self.target = target
  self.output:resizeAs(input_img)
  self.lsm:updateOutput(input_img)
  self.nll:updateOutput({self.lsm.output, target}, self.target)
  self.output = self.nll.output
  return self.output
end

function ElemwiseCrossEntropy:updateGradInput(input, gradOutput)
  local size = input[1]:size()
  self.nll:updateGradInput({self.lsm.output, input[2]},gradOutput)
  self.lsm:updateGradInput(input[1], self.nll.gradInput)
  self.gradInput:view(self.lsm.gradInput, size)
  return self.gradInput
end

return nn.ElemwiseCrossEntropy
