local _ = require 'moses'

local Recursor = nn.Recursor


function Recursor:maskOutput(mask, step)
   -- recursivecmul(self.outputs,mask)
   local recurrentModule = self:getStepModule(step)
   recurrentModule:setOutputStep(step)
   -- recurrentModule:get(4).output = mask[i]
   for j, m in ipairs(recurrentModule.modules) do
       m.output = torch.cmul(m.output, torch.repeatTensor(mask[step]:t(), m.output:size()[2],1):t())
   end
end
