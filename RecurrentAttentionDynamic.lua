------------------------------------------------------------------------
--[[ RecurrentAttention ]]-- 
-- Ref. A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- module which takes an RNN as argument with other 
-- hyper-parameters such as the maximum number of steps, 
-- action (actions sampling module like ReinforceNormal) and 
------------------------------------------------------------------------
local RecurrentAttention, parent = torch.class("nn.RecurrentAttentionNoShareDynamic", "nn.Container")

function recursivecmul(t1, t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursivecmul(t1[key], t2[key])
      end
   elseif torch.isTensor(t1) and torch.isTensor(t2) then
      t1 = torch.cmul(t1,torch.repeatTensor(t2:t(),t1:size()[2],1):t())
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function RecurrentAttention:__init(Ram, action2, nStep, hiddenSize, discount, e, greedy)
   parent.__init(self)
   -- rnn
   self.rnn = {}
   for i = 1, nStep do
       self.rnn[i] = Ram.rnn[i]
   end
   -- locator
   self.action = {}
   for i = 1, nStep do
       self.action[i] = Ram.action[i]
   end
   -- decider
   self.action2 = {}
   for i = 1, nStep do
       self.action2[i] = action2[i]
       self.action2[i].name = i
   end
   
   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {}
   for i = 1,#self.rnn do
       table.insert(self.modules, self.rnn[i])
       table.insert(self.modules, self.action[i])
       table.insert(self.modules, self.action2[i])
   end
   
   self.output = {} -- rnn output
   self.actions = {} -- locator output
   self.actions2 = {} -- decider output
   
   self.forwardActions = false
   -- greedy options
   self.edecay = true
   self.discount = discount or 0.8
   self.e = e or 0.2
   self.replace = {}
   self.edecay = true
   self.greedy = greedy or true
   
   self.gradHidden = {}
end

function RecurrentAttention:updateOutput(input)

   local nDim = input:dim()
   if train ~= false then
       self.output = {} 
       self.actions = {} 
       self.actions2 = {} 
       self.gradHidden = {}
       self.mask = {}
       self.mask2 = {}
   end
    
   if self.train ~= false then
       self.edecay = true
   end
   if self.train == false and self.edecay then
       self.e = self.e * self.discount
       if self.e < 0.1 then
           self.e = 0.1
       end
       self.edecay = false
   end
   if not self.greedy then
       self.e = 0
   end

   for step=1, self.nStep do
      
   if step == 1 then
        -- sample an initial starting actions by forwarding zeros through the action
        self._initInput = self._initInput or input.new()
        self._initInput:resize(input:size(1),table.unpack(self.hiddenSize)):zero()
        self.actions[1] = self.action[step]:updateOutput(self._initInput)
        self.actions2[1] = self.action2[step]:updateOutput(self._initInput)
   else
         -- sample actions from previous hidden activation (rnn output)
        self.actions[step] = self.action[step]:updateOutput(self.output[step-1])
        self.actions2[step] = self.action2[step]:updateOutput(self.output[step-1])
   end

      -- rnn handles the recurrence internally
   if step == 1 then
       output = self.rnn[step]:updateOutput(input)
   else
       output = self.rnn[step]:updateOutput{{input, self.actions[step], step},self.output[step - 1]}
   end
       self.output[step] = self.forwardActions and {output, self.actions[step]} or output
   end
   
   
   self.zeroMatrix =  nn.rnn.recursiveCopy(self.zeroMatrix, self.actions2)
   for i = 1,self.nStep do
       self.zeroMatrix[i]:fill(0)
   end

   if torch.rand(1)[1]<self.e and self.train ~= false then     
       local randomStep = torch.rand(input:size(1))*self.nStep
       randomStep:ceil()
       self.replace = nn.rnn.recursiveCopy(self.replace, self.zeroMatrix)
     
       for i = 1,input:size(1) do
            self.replace[randomStep[i]][i] = 1
       end
       self.actions2 = nn.rnn.recursiveCopy(self.actions2, self.replace)
   end
   self.mask = nn.rnn.recursiveCopy(self.mask, self.actions2)
   self.mask2 = nn.rnn.recursiveCopy(self.mask2, self.actions2)
   self.mask[1]:fill(1)
   self.mask2[1]:fill(1)
   for i = 1, input:size()[1] do 
       flag = 0
       for j = 1, self.nStep do      
           if flag == 1 then
               self.mask[j][i] = 0
               self.mask2[j][i] = 0
           elseif self.actions2[j][i][1] >= 0.5 and j ~= 1 then
               flag = 1
               self.mask[j][i] = 0
               self.mask2[j][i] = 1
           else 
               self.mask[j][i] = 1
               self.mask2[j][i] = 1
           end
        end
   end   

   recursivecmul(self.output, self.mask)
   self.gradPrevOutput = nil

   return self.output
end

function RecurrentAttention:updateGradInput(input, gradOutput)
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    

   recursivecmul(gradOutput, self.mask)
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradOutput_, gradAction_ = gradOutput[step]
      if self.forwardActions then
         gradOutput_, gradAction_ = unpack(gradOutput[step])
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give a zero Tensor instead
         self._gradAction = self._gradAction or self.action[1].output.new()
         self._gradAction2 = self._gradAction2 or self.action2[1].output.new()
      if not self._gradAction:isSameSizeAs(self.action[1].output) then
         self._gradAction:resizeAs(self.action[1].output):zero()
      end
      if not self._gradAction2:isSameSizeAs(self.action2[1].output) then
         self._gradAction2:resizeAs(self.action2[1].output):zero()
      end
         gradAction_ = self._gradAction
         gradAction2_ = self._gradAction2
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
         self.action2[step]:updateGradInput(self._initInput, gradAction2_)
      else
         local gradAction = self.action[step]:updateGradInput(self.output[step-1], gradAction_)
         local gradAction2 = self.action2[step]:updateGradInput(self.output[step-1], gradAction2_)  -- i dont know if the gradAction will affect the action's grad
         -- every output of time-step need to mul a mask
         gradAction = torch.cmul(gradAction, torch.repeatTensor(self.mask[step]:t(), gradAction:size()[2],1):t())
         gradAction2 = torch.cmul(gradAction2, torch.repeatTensor(self.mask2[step]:t(), gradAction2:size()[2],1):t()) --youwen ti
		 
         self.gradHidden[step-1] = nn.rnn.recursiveCopy(self.gradHidden[step-1], gradAction)
         nn.rnn.recursiveAdd(self.gradHidden[step-1], gradAction2)
      end
      
      -- 2. backward through the rnn layer
      if step == 1 then
         gradInput = self.rnn[step]:updateGradInput(input, self.gradHidden[step])
      else
         gradInputTable = self.rnn[step]:updateGradInput({{input, self.actions[step]}, self.output[step - 1]}, self.gradHidden[step])
         gradInput = gradInputTable[1][1]
         self.gradPrevOutput = gradInputTable[2]
      end

      local dim = gradInput:size()
      gradInput = torch.cmul(gradInput:resize(dim[1],dim[2]*dim[3]*dim[4]),torch.repeatTensor(self.mask[step]:t(),dim[2]*dim[3]*dim[4],1):t()):resize(dim[1],dim[2],dim[3],dim[4])  

      self.gradPrevOutput = torch.cmul(self.gradPrevOutput, torch.repeatTensor(self.mask[step]:t(), self.gradPrevOutput:size()[2],1):t())

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
      for j, m in ipairs(self.action2[step].modules) do
         m.output = torch.cmul(m.output, torch.repeatTensor(self.mask2[step]:t(), m.output:size()[2],1):t())
      end
      for j, m in ipairs(self.action[step].modules) do
         m.output = torch.cmul(m.output, torch.repeatTensor(self.mask[step]:t(), m.output:size()[2],1):t())
      end
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
      local gradAction2_ = self.forwardActions and gradOutput[step][2] or self._gradAction2      
      if step == 1 then
         -- backward through initial starting actions
         self.action[step]:accGradParameters(self._initInput, gradAction_, scale)
      else
         self.action[step]:accGradParameters(self.output[step-1], gradAction_, scale)
		 self.action2[step]:accGradParameters(self.output[step-1], gradAction2_, scale)
	  end
      -- 2. backward through the rnn layer
      local dim = input:size()
      input_mask = torch.cmul(input:clone():resize(dim[1],dim[2]*dim[3]*dim[4]),torch.repeatTensor(self.mask[step]:t(),dim[2]*dim[3]*dim[4],1):t()):resize(dim[1],dim[2],dim[3],dim[4])
      action_mask = torch.cmul(self.actions[step], torch.repeatTensor(self.mask[step]:t(), self.actions[step]:size()[2],1):t())
	  
      if step == 1 then
         self.rnn[step]:accGradParameters(input_mask, self.gradHidden[step], scale)
      else
         self.rnn[step]:accGradParameters({{input_mask, action_mask}, self.output[step - 1]}, self.gradHidden[step], scale)
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
   str = str .. line .. tab .. 'action2 : ' .. tostring(self.action2):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
