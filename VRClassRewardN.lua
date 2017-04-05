------------------------------------------------------------------------
--[[ VRClassReward ]]--
-- Variance reduced classification reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRClassReward, nn.SelectTable(-1))
------------------------------------------------------------------------
local VRClassReward, parent = torch.class("nn.VRClassRewardN", "nn.Criterion")

function VRClassReward:__init(module, scale, criterion, name, dynamic, rho)
   parent.__init(self)
   self.module = module -- so it can call module:reinforce(reward)
   self.scale = scale or 1 -- scale of reward
   self.name = name
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.dynamic = dynamic or false
   self.rho = rho or 1
   if self.dynamic then
       self.mode = 0
   else
       self.mode = 1
   end
   self.gradInput = {torch.Tensor()}
end

function VRClassReward:updateOutput(inputTable, target)
   assert(torch.type(inputTable) == 'table')
   if self.dynamic then 
       input = {}
       for i = 1, self.rho do
           input[i]= self:toBatch(inputTable[i], 1)
       end
	
 	   tmpInput = input[1]
	   local scale = input[1].new()
       scale:resize(input[1]:size(1)):fill(1 / (self.rho - 1))
       tag = scale[1]
       for i = 3, self.rho do
           tmpAction = self.module:getAction(i)
           for j = 1, tmpAction:size(1) do
               if tmpAction[j][1] > 0.5 and scale[j] == tag then
                   scale[j] = 1 / (i - 2)                
               end
           end
       end
       scale = torch.repeatTensor(scale, input[1]:size(2),1):t()
       local sumOutput = input[1].new()
       sumOutput:resizeAs(input[1]):fill(0)
       for i = 2, self.rho  do
           sumOutput = sumOutput + input[i]
       end
       sumOutput:cmul(scale)
       tmpInput = tmpInput + sumOutput
       input = tmpInput
   else
       input = self:toBatch(inputTable[1], 1)
   end
   self._maxVal = self._maxVal or input.new()
   self._maxIdx = self._maxIdx or torch.type(input) == 'torch.CudaTensor' and torch.CudaLongTensor() or torch.LongTensor()
   self._maxVal,self._maxIdx = torch.max(input,2)
   -- reward = scale when correctly classified
   local maxIdx = self._maxIdx
   if torch.type(self._maxIdx) == 'torch.CudaLongTensor' then
      self.__maxIdx = self.__maxIdx or torch.CudaTensor()
      self.__maxIdx:resize(maxIdx:size()):copy(maxIdx)
      maxIdx = self.__maxIdx
   end
   
   if torch.type(maxIdx) ~= torch.type(target) then
      self._target = self._target or maxIdx.new()
      self._target:resize(target:size()):copy(target)
      target = self._target
   end
   
   self._reward_ = self._reward_ or maxIdx.new()
   self._reward = self._reward or maxIdx.new()
   self._reward_:eq(maxIdx, target)
   -- self._reward:eq(self._reward_, actions)

   self.reward = self.reward or input.new()
   self.reward:resize(self._reward_:size(1)):copy(self._reward_)
   self.reward:mul(self.scale)
   -- loss = -sum(reward)
   self.output = -self.reward:sum()
   if self.sizeAverage then
      self.output = self.output/input:size(1)
   end
   return self.output
end

function VRClassReward:updateGradInput(inputTable, target)
   local input = self:toBatch(inputTable[1], 1)
   if self.dynamic then 
       baseline = self:toBatch(inputTable[self.rho + 1], 1)
   else
       baseline = self:toBatch(inputTable[2], 1)
   end
   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   self.vrReward:add(-1, baseline)
   if self.sizeAverage then
      self.vrReward:div(input:size(1))
   end

   self.module:reinforce(self.vrReward, self.name, self.mode)  


   -- zero gradInput (this criterion has no gradInput for class pred)
   if self.dynamic then 
      for i = 1, self.rho do
          self.gradInput[i] = input.new()
          self.gradInput[i]:resizeAs(input):zero()
          self.gradInput[i] = self:fromBatch(self.gradInput[1], 1)
      end
      -- learn the baseline reward
      self.criterion:forward(baseline, self.reward)
      self.gradInput[self.rho + 1] = self.criterion:backward(baseline, self.reward)
      self.gradInput[self.rho + 1] = self:fromBatch(self.gradInput[self.rho + 1], 1)
   else
      self.gradInput[1]:resizeAs(input):zero()
      self.gradInput[1] = self:fromBatch(self.gradInput[1], 1)
      -- learn the baseline reward
      self.criterion:forward(baseline, self.reward)
      self.gradInput[2] = self.criterion:backward(baseline, self.reward)
      self.gradInput[2] = self:fromBatch(self.gradInput[2], 1)
   end

   return self.gradInput
end

function VRClassReward:type(type)
   self._maxVal = nil
   self._maxIdx = nil
   self.__maxIdx = nil
   self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
