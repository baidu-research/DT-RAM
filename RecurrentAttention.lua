------------------------------------------------------------------------
--[[ RecurrentAttention ]]-- 
-- Ref. A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- module which takes an RNN as argument with other 
-- hyper-parameters such as the maximum number of steps, 
-- action (actions sampling module like ReinforceNormal) and 
------------------------------------------------------------------------
local RecurrentAttention, parent = torch.class("nn.RecurrentAttentionNoShare", "nn.Container")

function RecurrentAttention:__init(init, rnn, action, nStep, hiddenSize)
   parent.__init(self)
   assert(torch.isTypeOf(action, 'nn.Module'))
   assert(torch.type(nStep) == 'number')
   assert(torch.type(hiddenSize) == 'table')
   assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers" )
   self.rnn = {}
   self.rnn[1] = init
   self.rnn[2] = rnn
   for i = 3, nStep do
       self.rnn[i] = rnn:clone()
   end
   self.action = {}
   self.action[1] = action
   self.action[1].name = 1
   for i = 2, nStep do
       self.action[i] = action:clone()
       self.action[i].name = i
   end
   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {}
   for i = 1,#self.rnn do
       table.insert(self.modules, self.rnn[i])
       table.insert(self.modules, self.action[i])
   end
   
   self.output = {} -- rnn output
   self.actions = {} -- action output
   
   self.forwardActions = false
   self.gradHidden = {}
end

function RecurrentAttention:updateOutput(input)

   local nDim = input:dim()
   if train ~= false then
       self.output = {} -- rnn output
       self.actions = {} -- action output
       self.gradHidden = {}
   end

   for step=1, self.nStep do
      
      if step == 1 then
         -- sample an initial starting actions by forwarding zeros through the action
         self._initInput = self._initInput or input.new()
         self._initInput:resize(input:size(1),table.unpack(self.hiddenSize)):zero()
         self.actions[1] = self.action[step]:updateOutput(self._initInput)
      else
         -- sample actions from previous hidden activation (rnn output)
         self.actions[step] = self.action[step]:updateOutput(self.output[step-1])
      end
      -- rnn handles the recurrence internally
      if step == 1 then
         output = self.rnn[step]:updateOutput(input)
      else
         output = self.rnn[step]:updateOutput{{input, self.actions[step], step},self.output[step - 1]}
      end
      self.output[step] = self.forwardActions and {output, self.actions[step]} or output

   end
   self.gradPrevOutput = nil

   return self.output
end

function RecurrentAttention:updateGradInput(input, gradOutput)
   -- assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradOutput_, gradAction_ = gradOutput[step]
      if self.forwardActions then
         gradOutput_, gradAction_ = unpack(gradOutput[step])
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give a zero Tensor instead
         self._gradAction = self._gradAction or self.action[1].output.new()
         if not self._gradAction:isSameSizeAs(self.action[1].output) then
            self._gradAction:resizeAs(self.action[1].output):zero()
         end
         gradAction_ = self._gradAction
      end
      
      if step == self.nStep then
         self.gradHidden[step] = nn.rnn.recursiveCopy(self.gradHidden[step], gradOutput_)
      else
         -- gradHidden = gradOutput + gradAction
         nn.rnn.recursiveAdd(self.gradHidden[step], gradOutput_)
      end
	  
      if self.gradPrevOutput then
         nn.rnn.recursiveAdd(self.gradHidden[step], self.gradPrevOutput)
      end
      
      if step == 1 then
         -- backward through initial starting actions
         self.action[step]:updateGradInput(self._initInput, gradAction_)
      else
         local gradAction = self.action[step]:updateGradInput(self.output[step-1], gradAction_)
         self.gradHidden[step-1] = nn.rnn.recursiveCopy(self.gradHidden[step-1], gradAction)
      end
      
      -- 2. backward through the rnn layer
      if step == 1 then
          gradInput = self.rnn[step]:updateGradInput(input, self.gradHidden[step])
      else
          tmp = self.rnn[step]:updateGradInput({{input, self.actions[step]}, self.output[step - 1]}, self.gradHidden[step])
          gradInput = tmp[1][1]
          self.gradPrevOutput = tmp[2]
      end
      if step == self.nStep then
          self.gradInput:resizeAs(gradInput):copy(gradInput)
      else
          self.gradInput:add(gradInput)
      end
   end

   return self.gradInput
end

function RecurrentAttention:accGradParameters(input, gradOutput, scale)
   -- assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
            
      if step == 1 then
         -- backward through initial starting actions
         self.action[step]:accGradParameters(self._initInput, gradAction_, scale)
      else
         self.action[step]:accGradParameters(self.output[step-1], gradAction_, scale)
      end
      -- 2. backward through the rnn layer
      if step == 1 then
          self.rnn[step]:accGradParameters(input, self.gradHidden[step], scale)
      else
          self.rnn[step]:accGradParameters({{input, self.actions[step]}, self.output[step - 1]}, self.gradHidden[step], scale)
      end
   end
end



function RecurrentAttention:type(type)
   self._input = nil
   self._actions = nil
   self._crop = nil
   self._pad = nil
   self._byte = nil
   return parent.type(self, type)
end

function RecurrentAttention:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'action : ' .. tostring(self.action):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
