require 'nn'

local MetaRobustCriterion, Criterion = torch.class('nn.MetaRobustCriterion','nn.Criterion')

--TODO: test this criterion

function MetaRobustCriterion:__init(sizeAverage)
  Criterion.__init(self)
  self.abs = nn.AbsCriterion()
  self.sizeAverage = false -- This is handled in the computation of MetaRobustLayer
  self.target = torch.Tensor(input:size()):fill(0)
end

function MetaRobustCriterion:updateOutput(input, target)
  -- Note that we ignore target here - it's encoded in input. We just want
  -- to take the average over inputs, so using AbsCriterion for that.

  self.abs:updateOutput(input, self.target)

  return self.output
end

function MetaRobustCriterion:updateGradInput(input,target)
  target = torch.Tensor(input:size()):fill(0)
  self.abs:updateGradInput(input, self.target)
  return self.gradInput
end

return nn.MetaRobustCriterion
