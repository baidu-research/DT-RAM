
require 'SpatialGlimpseCrop'
require 'dp'
require 'rnn'
require 'RecurrentAttention'
require 'RecurrentAttentionDynamic'

local function createModel(opt, modelPreStep)
   
   if not opt.dynamic then
       -- train Ram step by step
       RAM =  modelPreStep:get(1)
       RAM.nStep = opt.rho
       if not RAM.action[RAM.nStep]  then
          RAM.action[RAM.nStep] = RAM.action[RAM.nStep - 1]:clone()
          RAM.action[RAM.nStep].name = RAM.nStep
          table.insert(RAM.modules, RAM.action[RAM.nStep])
       end
       if not RAM.rnn[RAM.nStep] then
          RAM.rnn[RAM.nStep] = RAM.rnn[RAM.nStep - 1]:clone()
          table.insert(RAM.modules,RAM.rnn[RAM.nStep])
       end
       if opt.freezeParam then
          for i = 1, RAM.nStep - 1 do
             RAM.action[i].parameters = function() return nil end
             RAM.action[i].accGradParameters = function(self) end
             RAM.rnn[i].parameters = function() return nil end
             RAM.rnn[i].accGradParameters = function(self) end
             modelPreStep:get(2):get(i).accGradParameters = function(self) end
             modelPreStep:get(2):get(i).parameters = function() return nil end
          end
       end
       
       if opt.uniform > 0 then
          for k,param in ipairs(RAM.action[RAM.nStep]:parameters()) do
             param:uniform(-opt.uniform, opt.uniform)
          end
       end
   
       modelPreStep:get(2):add(nn.Linear(opt.modelOutputSize, opt.nClasses))
   
       concat2 =  nn.ConcatTable()
       for i = 1,opt.rho do
           concat2:add(nn.SelectTable(i))
       end

       for i = 2,opt.rho do
           seq = nn.Sequential()
           seq:add(nn.SelectTable(i))
           seq:add(nn.Constant(0,1))
           seq:add(nn.Add(1))
           concat = nn.ConcatTable():add(nn.SelectTable(i)):add(seq)
           concat2:add(concat)
       end
   else
       -- train dynamic action
       RAM =  modelPreStep:get(1) 
       if opt.freezeParam then   
          for i = 1, RAM.nStep do
             RAM.action[i].parameters = function() return nil end
             RAM.action[i].accGradParameters = function(self) end
             RAM.rnn[i].parameters = function() return nil end
             RAM.rnn[i].accGradParameters = function(self) end
          end
       end
       
       linear = {}
       decider = {}
       for i = 1,opt.rho do
           linear[i] = modelPreStep:get(2):get(i)
           linear[i].accGradParameters = function(self) end
           linear[i].parameters = function() return nil end
           decider[i] = nn.Sequential()
           decider[i]:add(linear[i])
           decider[i]:add(nn.Tanh())
           action = nn.Sequential()
           action:add(nn.Linear(opt.nClasses, 256))
           action:add(nn.Tanh())
           action:add(nn.Linear(256, 1))
           action:add(nn.HardTanh(-10,10))
           action:add(nn.Sigmoid()) -- bounds mean between -1 and 1
           action:add(nn.ReinforceBernoulli(opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
           decider[i]:add(action)
       end
       
       for i = 1, opt.rho do
           local method = 'xavier_caffe'
           decider[i].modules[3] = require('weight-init')(decider[i].modules[3], method)
       end
   
       attention = nn.RecurrentAttentionNoShareDynamic(RAM, decider, opt.rho, {opt.modelOutputSize})
       modelPreStep:remove(1)
       modelPreStep:insert(attention, 1)
   
       concat2 =  nn.ConcatTable()
       for i = 1,opt.rho do
           concat2:add(nn.SelectTable(i))
       end
   
       seq = nn.Sequential()
       seq:add(nn.SelectTable(1))
       seq:add(nn.Constant(0,1))
       seq:add(nn.Add(1))
       concat = nn.ConcatTable()
       for i = 1,opt.rho do
           concat:add(nn.SelectTable(i))
       end
       concat:add(seq)
       concat2:add(concat)  
   end
   
   modelPreStep:remove(3)
   modelPreStep:add(concat2)
   modelPreStep:cuda()
   return modelPreStep
end

return createModel
