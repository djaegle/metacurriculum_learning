-- Implements the ClassNLL criterion for logistic regression as a forward layer
-- to allow robust kernel learning as a function of the imagewise CrossEntropy loss.

require 'nn'

local ElemwiseClassNLL, parent = torch.class('nn.ElemwiseClassNLL','nn.Module')

-- Note, we don't need sizeAverage to hook into here: it only matters when the final
-- loss gradients are computed, so we can just defer it to then.

-- TODO: gradient check backward pass
-- TODO: C implementation of forward pass


function ElemwiseClassNLL:__init()
  -- no parameters here: just need to pass forward the loss
  parent.__init(self)
  self.output = torch.zeros(1)
  self.target = torch.zeros(1):long()
end

function ElemwiseClassNLL:updateOutput(input)
  -- Takes input and target (same dim) and computes the imagewise
  -- NLL of the inputs w/r/t the targets.
  input_img = input[1]
  target = input[2]


  if type(target) == 'number' then
    if input_img:type() ~= 'torch.CudaTensor' then
      self.target = self.target:long()
    end
    self.target[1] = target
  elseif target:type() == 'torch.CudaTensor' then
    self.target = target
  else
    self.target = target:long()
  end

  if input_img:dim() == 1 then
    self.output:resize(1)
    self.output = -input_img[target]

  elseif input_img:dim() == 2 then
    self.output:resize(input_img:size(1))

    for im_i = 1,input_img:size(1) do
      self.output[im_i] = -input_img[{im_i,target[im_i]}]
    end

  else
    error('input must be a vector or matrix')
  end

  return self.output

end

function ElemwiseClassNLL:updateGradInput(input, gradOutput)
  -- Change to input_img and target here as well
  input_img = input[1]
  target = input[2]

  self.gradInput:resizeAs(input_img):zero()

  if input_img:dim() == 1 then
    self.gradInput[target[1]] = -1
    self.gradInput = self.gradInput*gradOutput

  elseif input_img:dim() == 2 then

    for im_i = 1,input_img:size(1) do
      self.gradInput[{im_i,self.target[im_i]}] = -1
    end

    -- Matrix-multiply way to do this. Note that the cmul way is cleaner
    -- tmpResult = torch.Tensor(input_img:size(1),input_img:nElement()):fill(0)
    -- -- Compute deriv of output w/r/t input
    -- for im_i = 1,input_img:size(1) do
    --   tmpResult[{im_i,(im_i-1)*input_img:size(2) + self.target[im_i]}] = -1
    -- end
    --
    -- self.gradInput = self.gradInput:view(self.gradInput:nElement()) -- flatten
    -- self.gradInput:mv(tmpResult:t(),gradOutput)
    -- self.gradInput = self.gradInput:view(input_img:size()) -- flatten

    -- From here, only need to broadcast the dLoss w/r/t each z to each row (class) of input
    self.gradInput:cmul(gradOutput:view(gradOutput:size(1),1):expandAs(self.gradInput))
  end

  return self.gradInput
end
