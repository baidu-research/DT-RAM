------------------------------------------------------------------------
--[[ RecurrentAttention ]]-- 
-- Ref. A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- module which takes an RNN as argument with other 
-- hyper-parameters such as the maximum number of steps, 
-- action (actions sampling module like ReinforceNormal) and 
------------------------------------------------------------------------
local RecurrentAttention, parent = torch.class("nn.RecurrentAttentionDynamic", "nn.AbstractSequencer")


function recursivecmul(t1, t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursivecmul(t1[key], t2[key])
      end
   elseif torch.isTensor(t1) and torch.isTensor(t2) then
      t1 = torch.cmul(t1, torch.repeatTensor(t2:t(), t1:size()[2],1):t())
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function RecurrentAttention:__init(rnn, action, action2, nStep, hiddenSize, discount, e, greedy, threshold)
   parent.__init(self)
   assert(torch.isTypeOf(action, 'nn.Module'))
   assert(torch.type(nStep) == 'number')
   assert(torch.type(hiddenSize) == 'table')
   assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers" )
   
   self.rnn = rnn
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.rnn = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(rnn) or rnn
   
   -- samples an x,y actions for each example
   self.action =  (not torch.isTypeOf(action, 'nn.AbstractRecurrent')) and nn.Recursor(action) or action 
   self.action2 =  (not torch.isTypeOf(action2, 'nn.AbstractRecurrent')) and nn.Recursor(action2) or action2 


   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {self.rnn, self.action, self.action2}
   
   self.output = {} -- rnn output
   self.actions = {} -- action output
   self.actions2 = {} -- action2 output
   
   self.forwardActions = false
   
   self.gradHidden = {}
   self.mask = {}
   self.mask1 = {}
   self.mask2 = {}
   
   self.edecay = true
   self.discount = discount or 0.8
   self.e = e or 0.2
   self.replace = {}
   self.edecay = true
   self.greedy = greedy or false
   self.threshold = threshold or 0.9
end

function RecurrentAttention:updateOutput(input)
   self.rnn:forget()
   self.action:forget()
   self.action2:forget()
   self.mask = {}
   self.mask2 = {}
   local nDim = input:dim()
   
  if self.train ~= false then
       self.edecay = true
   end
   if self.train == false and self.edecay then
       self.e = self.e * self.discount
       if self.e < 0.1 then
           self.e = 0.1
       end
	   print('e:' .. self.e)
       self.edecay = false
   end
   if not self.greedy then
       self.e = 0
   end

   
   if not self.numOfStep then
      self.flag = true
      self.numOfStep = {} 
      for i = 1,self.nStep  do
          self.numOfStep[i] = 0
      end
   end
   if self.train ~= false and self.flag then
      sum = 20000
      for i = 1, self.nStep - 1 do
          print(i .. ":" .. self.numOfStep[i])
          sum = sum - self.numOfStep[i]
      end
      print(self.nStep .. ":" .. sum)
      self.flag = false
      for i = 1,self.nStep - 1 do
         self.numOfStep[i] = 0
      end
   end
   if self.train == false and not self.flag then
       self.flag = true
   end
   
   for step=1,self.nStep do
      
      if step == 1 then
         -- sample an initial starting actions by forwarding zeros through the action
         self._initInput = self._initInput or input.new()
         self._initInput:resize(input:size(1),table.unpack(self.hiddenSize)):zero()
         self.actions[1] = self.action:updateOutput(self._initInput)
         self.actions2[1] = self.action2:updateOutput(self._initInput)
      else
         -- sample actions from previous hidden activation (rnn output)
         if self.train ~= false then
             self.actions[step] = self.action:updateOutput(self.output[step-1])
             self.actions2[step] = self.action2:updateOutput(self.output[step-1])
         else
             self.actions[step] = self.action:updateOutput(self.output[step-1]):clone()
             self.actions2[step] = self.action2:updateOutput(self.output[step-1]):clone()
         end

      end
      -- rnn handles the recurrence internally
      local output = self.rnn:updateOutput{input, self.actions[step],step}
      -- print(output)
      if self.train ~= false then
          self.output[step] =  output:clone()
      else
          self.output[step] =  output:clone()
      end
   end

   self.zeroMatrix =  nn.rnn.recursiveCopy(self.zeroMatrix, self.actions2)
   for i = 1,self.nStep do
      self.zeroMatrix[i]:fill(0)
   end
   
   if torch.rand(1)[1] < self.e and self.train ~= false then
      
      local randomStep = torch.rand(input:size(1)) * self.nStep
      randomStep:ceil()
      self.replace = nn.rnn.recursiveCopy(self.replace, self.zeroMatrix)
     
      for i = 1, input:size(1) do
        self.replace[randomStep[i]][i] = 1
      end
      self.actions2 = nn.rnn.recursiveCopy(self.actions2, self.replace)
	  for i = 1, self.nStep do
	     local recurrentModule = self.action2:getStepModule(i)
		 recurrentModule:setOutputStep(i)
		 recurrentModule.output = self.replace[i]
      end
	end

   self.mask = nn.rnn.recursiveCopy(self.mask, self.actions2)
   self.mask1 = nn.rnn.recursiveCopy(self.mask1, self.actions2)
   self.mask2 = nn.rnn.recursiveCopy(self.mask2, self.actions2)
   self.mask[1]:fill(1)
   self.mask2[1]:fill(1)
   self.stopStep = self.mask[1].new()  
   self.stopStep:resizeAs(self.mask[1]):fill(self.nStep)

   for i = 1, self.nStep do
       self.mask1[i]:fill(0)
   end
   tmpOutput = self.output[self.nStep]
   
   for i = 1,input:size()[1] do 
        flag = 0
        for j = 1,self.nStep do      
            if flag == 1 then
                self.mask[j][i] = 0
                self.mask2[j][i] = 0
            elseif self.actions2[j][i][1] > self.threshold and j ~= 1 then
                flag = 1
                self.mask[j][i] = 0
                self.mask1[j-1][i] = 1
                self.mask2[j][i] = 1
                self.stopStep[i] = j - 1
                tmpOutput[i] = self.output[j-1][i] -- trick to get element in -1
                if self.train == false then
                    self.numOfStep[j - 1] = self.numOfStep[j - 1] + 1
                end
            else 
                self.mask[j][i] = 1
                self.mask2[j][i] = 1
            end
            
        end
    end   

   recursivecmul(self.output, self.mask)
   self.rnn:maskOutput(self.mask)
   self.output[self.nStep] = tmpOutput
   return {self.output, self.stopStep}
end

function RecurrentAttention:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput[1] == self.nStep, "gradOutput should have nStep elements")
   
   gradOutput = gradOutput[1]
   for i=1,self.nStep do
       gradOutput[i] = gradOutput[self.nStep]
   end
   recursivecmul(gradOutput, self.mask1)
   
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradOutput_, gradAction_ = gradOutput[step]
      if self.forwardActions then
         gradOutput_, gradAction_ = unpack(gradOutput[step])
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give a zero Tensor instead
         self._gradAction = self._gradAction or self.action.output.new()
         self._gradAction2 = self._gradAction2 or self.action2.output.new()
         if not self._gradAction:isSameSizeAs(self.action.output) then
            self._gradAction:resizeAs(self.action.output):zero()
         end
         gradAction_ = self._gradAction
         if not self._gradAction2:isSameSizeAs(self.action2.output) then
            self._gradAction2:resizeAs(self.action2.output):zero()
         end
         gradAction2_ = self._gradAction2
      end
      
      if step == self.nStep then
         self.gradHidden[step] = nn.rnn.recursiveCopy(self.gradHidden[step], gradOutput_)
      else
         -- gradHidden = gradOutput + gradAction
         nn.rnn.recursiveAdd(self.gradHidden[step], gradOutput_)
      end
      
      if step == 1 then
         -- backward through initial starting actions
         self.action:updateGradInput(self._initInput, gradAction_)
         self.action2:updateGradInput(self._initInput, gradAction2_)
      else
         local gradAction = self.action:updateGradInput(self.output[step-1], gradAction_)
         local gradAction2 = self.action2:updateGradInput(self.output[step-1], gradAction2_)  -- i dont know if the gradAction will affect the action's grad
         -- every output of time-step need to mul a mask
         gradAction = torch.cmul(gradAction, torch.repeatTensor(self.mask[step]:t(),gradAction:size()[2],1):t())
         gradAction2 = torch.cmul(gradAction2, torch.repeatTensor(self.mask2[step]:t(),gradAction2:size()[2],1):t()) --youwen ti
         -- print(gradAction2)
         self.gradHidden[step-1] = nn.rnn.recursiveCopy(self.gradHidden[step-1], gradAction)
         nn.rnn.recursiveAdd(self.gradHidden[step-1], gradAction2)
      end
      
      -- 2. backward through the rnn layer
      local gradInput = self.rnn:updateGradInput({input, self.actions[step]}, self.gradHidden[step])[1]
      
      local dim = gradInput:size()
      gradInput = torch.cmul(gradInput:resize(dim[1],dim[2]*dim[3]*dim[4]),torch.repeatTensor(self.mask[step]:t(),dim[2]*dim[3]*dim[4],1):t()):resize(dim[1],dim[2],dim[3],dim[4])  
      
      if step == self.nStep then
         self.gradInput:resizeAs(gradInput):copy(gradInput)
      else
         self.gradInput:add(gradInput)
      end
   end

   return self.gradInput
end

function RecurrentAttention:accGradParameters(input, gradOutput, scale)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput[1] == self.nStep, "gradOutput should have nStep elements")
   
   gradOutput = gradOutput[1]
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
      local gradAction2_ = self.forwardActions and gradOutput[step][2] or self._gradAction2      
	   
	  self.action2:maskOutput(self.mask2, step)
	  self.action:maskOutput(self.mask, step)
	   
      if step == 1 then
         -- backward through initial starting actions
         self.action:accGradParameters(self._initInput, gradAction_, scale)
         -- self.action2:accGradParameters(self._initInput, gradAction2_, scale)
      else
         self.action:accGradParameters(self.output[step-1], gradAction_, scale)
         self.action2:accGradParameters(self.output[step-1], gradAction2_, scale)
      end
      
      -- 2. backward through the rnn layer
      local dim = input:size()
      input_mask = torch.cmul(input:clone():resize(dim[1],dim[2]*dim[3]*dim[4]),torch.repeatTensor(self.mask[step]:t(),dim[2]*dim[3]*dim[4],1):t()):resize(dim[1],dim[2],dim[3],dim[4])
      action_mask = torch.cmul(self.actions[step],torch.repeatTensor(self.mask[step]:t(),self.actions[step]:size()[2],1):t())
      self.rnn:accGradParameters({input_mask, action_mask}, self.gradHidden[step], scale)
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
   str = str .. line .. tab .. 'action2 : ' .. tostring(self.action2):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
