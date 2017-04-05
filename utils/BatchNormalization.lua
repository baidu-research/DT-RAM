local _ = require 'moses'

local BN = nn.BatchNormalization
local THNN = require 'nn.THNN'

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
         self._gradOutput = self._gradOutput or gradOutput.new()
         self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
         gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function BN:updateOutput(input)
   self:checkInputDim(input)
   if self.freeze then
      self.train = false
   end
   input = makeContiguous(self, input)

   self.output:resizeAs(input)
   self.save_mean = self.save_mean or input.new()
   self.save_mean:resizeAs(self.running_mean)
   self.save_std = self.save_std or input.new()
   self.save_std:resizeAs(self.running_var)

   input.THNN.BatchNormalization_updateOutput(
      input:cdata(),
      self.output:cdata(),
      THNN.optionalTensor(self.weight),
      THNN.optionalTensor(self.bias),
      self.running_mean:cdata(),
      self.running_var:cdata(),
      self.save_mean:cdata(),
      self.save_std:cdata(),
      self.train,
      self.momentum,
      self.eps)

   return self.output
end

local function backward(self, input, gradOutput, scale, gradInput, gradWeight, gradBias)
   self:checkInputDim(input)
   self:checkInputDim(gradOutput)
   assert(self.save_mean and self.save_std, 'must call :updateOutput() first')

   if self.freeze then
      self.train = false
   end
   
   input, gradOutput = makeContiguous(self, input, gradOutput)

   scale = scale or 1
   if gradInput then
      gradInput:resizeAs(gradOutput)
   end

   input.THNN.BatchNormalization_backward(
      input:cdata(),
      gradOutput:cdata(),
      THNN.optionalTensor(gradInput),
      THNN.optionalTensor(gradWeight),
      THNN.optionalTensor(gradBias),
      THNN.optionalTensor(self.weight),
      self.running_mean:cdata(),
      self.running_var:cdata(),
      self.save_mean:cdata(),
      self.save_std:cdata(),
      self.train,
      scale,
      self.eps)

   return self.gradInput
end

function BN:backward(input, gradOutput, scale)
   return backward(self, input, gradOutput, scale, self.gradInput, self.gradWeight, self.gradBias)
end

function BN:updateGradInput(input, gradOutput)
   return backward(self, input, gradOutput, 1, self.gradInput)
end

function BN:accGradParameters(input, gradOutput, scale)
   return backward(self, input, gradOutput, scale, nil, self.gradWeight, self.gradBias)
end


