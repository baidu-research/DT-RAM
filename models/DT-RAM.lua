
require 'SpatialGlimpseCrop'
require 'dp'
require 'rnn'
require 'RecurrentAttention'

local function createModel(opt, retrainModel, modelNextStep)

   modelOutputSize = opt.modelOutputSize
   
   if opt.imageSize == 512 then
       resize = false
   else 
       resize = true
   end       
   
   glimpseSensor = nn.Sequential()
   glimpseSensor:add(nn.SpatialGlimpseCrop(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale, resize, opt.cropSize):float())
   glimpseSensor:add(modelNextStep)
    
   inputModule = glimpseSensor
   feedbackModule = nn.Linear(modelOutputSize, modelOutputSize)
   
   parallelModule = nn.ParallelTable()
   parallelModule:add(inputModule)
   parallelModule:add(feedbackModule)
   
   recurrentModule = nn.Sequential()
   recurrentModule:add(parallelModule)
   recurrentModule:add(nn.CAddTable())
   recurrentModule:add(nn.ReLU())
    
   -- actions (locator)
   locator = nn.Sequential()
   locator:add(nn.Linear(modelOutputSize, modelOutputSize / 2))
   locator:add(nn.Tanh())
   locator:add(nn.Linear(modelOutputSize / 2, 2))
   locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
   locator:add(nn.ReinforceNormal(2 * opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
   locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
   locator:add(nn.MulConstant(opt.unitPixels / opt.cropSize))
    
   if opt.uniform > 0 then
       for k,param in ipairs(locator:parameters()) do
           param:uniform(-opt.uniform, opt.uniform)
       end
   end
   
   for i, m in ipairs(retrainModel.modules) do
       if torch.type(m):find('SpatialBatchNormalization') then
           m.freeze = true
       end
   end

   attention = nn.RecurrentAttentionNoShare(retrainModel, recurrentModule, locator, opt.rho, {modelOutputSize})

   Parallel = nn.ParallelTable()
   for i = 1,opt.rho do
        Parallel:add(nn.Linear(modelOutputSize, opt.nClasses))
   end
   
   agent = nn.Sequential()
   agent:add(attention)

   -- classifier :
   agent:add(Parallel)


   -- add the baseline reward predictor
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
   
   agent:add(concat2)
	
   agent:cuda()
    -- print(agent)
   return agent
end

return createModel
