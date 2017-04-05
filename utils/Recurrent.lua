local _ = require 'moses'

local Recurrent = nn.Recurrent


local function recursivecmul(t1, t2)
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

function Recurrent:maskOutput(mask)
   recursivecmul(self.outputs, mask)
end

