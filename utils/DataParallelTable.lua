local _ = require 'moses'

local DataParallelTable = nn.DataParallelTable

local function _hasData(input)
   if torch.isTensor(input) then
      return input:numel() ~= 0
   else
      assert(type(input) == 'table')
      for i = 1, #input do
         if _hasData(input[i]) then
            return true
         end
      end
      return false
   end
end

function DataParallelTable:getAction(idx)
   local prevGpuid = cutorch.getDevice()
   if not self.ActionGpu then 
       self.ActionGpu = {}
   end
   -- update output for each module
   self.ActionGpu = self.impl:exec(function(m, i)
      nodes = m:findName(idx)

      return nodes[2].output
   end)

   -- concatenate the outputs to the base GPU
   self.Action = self:_concat(self.Action, self.ActionGpu)

   cutorch.setDevice(prevGpuid)

   return self.Action
end

function DataParallelTable:reinforce(reward, idx, mode)
   local prevGpuid = cutorch.getDevice()
   if not self.rewardGpu then 
        self.rewardGpu = {}
   end
   -- distribute the input to GPUs
   self:_distribute(self.rewardGpu, reward)

   -- update output for each module
   local rewardGpu = self.rewardGpu
   self.impl:exec(function(m, i)
      if _hasData(rewardGpu[i]) then
         if mode == 1 then
             nodes = m:findName(idx)
             for j = 1,#nodes do
                 nodes[j]:reinforce(rewardGpu[i])
             end
             if idx == 2 then
                 nodes = m:findName(1)
                 for j = 1,#nodes do
                     nodes[j]:reinforce(rewardGpu[i])
                 end
             end
         else
             m:reinforce(rewardGpu[i])
         end
      end
   end)

   cutorch.setDevice(prevGpuid)
end

