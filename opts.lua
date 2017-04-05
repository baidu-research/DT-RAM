
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text()
   cmd:text('Options:')
   ----------------RAM Options--------------------
   cmd:option('-gama',            '1',    'gama for reward')
   cmd:option('-minLR',           0.00001,'minimum learning rate')
   cmd:option('-saturateEpoch',   100,    'epoch at which linear decayed LR will reach minLR')
   cmd:option('-transfer',        'ReLU', 'activation function')
   --[[ reinforce ]]--
   cmd:option('-rewardScale',     1,      "scale of positive reward (negative is 0)")
   cmd:option('-unitPixels',      13,     "the locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)")
   cmd:option('-locatorStd',      0.12,   'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
   cmd:option('-stochastic',      false,  'Reinforce modules forward inputs stochastically during evaluation')
   cmd:option('-uniform',         0.1,    'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
   --[[ glimpse layer ]]--
   cmd:option('-glimpseHiddenSize',128,   'size of glimpse hidden layer')
   cmd:option('-glimpsePatchSize', 8,     'size of glimpse patch at highest res (height = width)')
   cmd:option('-glimpseScale',     2,     'scale of successive patches w.r.t. original input image')
   cmd:option('-glimpseDepth',     1,     'number of concatenated downscaled patches')
   cmd:option('-locatorHiddenSize',128,   'size of locator hidden layer')
   cmd:option('-imageHiddenSize',  256,   'size of hidden layer combining glimpse and locator hiddens')
   --[[ recurrent layer ]]--
   cmd:option('-rho',              1,     'back-propagate through time (BPTT) for rho time-steps')
   cmd:option('-hiddenSize',       256,   'number of hidden units used in Simple RNN.')
   cmd:option('-FastLSTM',         false, 'use LSTM instead of linear layer')
   cmd:option('-finetune',         false, 'use LSTM instead of linear layer')
   -----------------------------------------------
   ------------ General options --------------------
   cmd:option('-data',             '',       'Path to dataset')
   cmd:option('-train_list',       'train.list',   'Path to train list')
   cmd:option('-val_list',         'val.list',     'Path to val list')
   cmd:option('-dataset',          'imagenet',     'Options: imagenet | cifar10')
   cmd:option('-manualSeed',       0,        'Manually set RNG seed')
   cmd:option('-nGPU',             4,        'Number of GPUs to use by default')
   cmd:option('-backend',          'cudnn',  'Options: cudnn | cunn')
   cmd:option('-cudnn',            'fastest','Options: fastest | default | deterministic')
   cmd:option('-gen',              'gen',    'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',         4,        'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',          90,       'Number of total epochs to run')
   cmd:option('-epochNumber',      1,        'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',        96,       'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',         false,    'Run on validation set only')
   cmd:option('-tenCrop',          false,    'Ten-crop testing')
   cmd:option('-resume',           'none',   'Path to directory containing checkpoint')
   ---------- Optimization options ----------------------
   cmd:option('-LR',               0.1,      'initial learning rate')
   cmd:option('-momentum',         0.9,      'momentum')
   cmd:option('-dropout',          false,    'momentum')
   cmd:option('-weightDecay',      1e-4,     'weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-netType',          'DT-RAM', 'Options: resnet | preresnet')
   cmd:option('-retrain',          'none',   'Path to model to retrain with')
   cmd:option('-optimState',       'none',   'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',   false,    'Share gradInput tensors to reduce memory usage')
   cmd:option('-resetClassifier',  false,    'Reset the fully connected layer for fine-tuning')
   cmd:option('-learnStep',        false,    'train step by step')
   cmd:option('-nClasses',         200,      'Number of classes in the dataset')
   cmd:option('-imageSize',        256,      'Size of input image')
   cmd:option('-cropSize',         224,      'Size of crop')
   cmd:option('-modelNextStep',    '',       'Path to model to train next step')
   cmd:option('-modelOutputSize',  2048,     'Dim of output size of model')
   cmd:option('-dynamic',          false,    'train dyanamic action')
   cmd:option('-freezeParam',      false,    'freeze the param')
   cmd:option('-RandomSizeCrop',   false,    'use RandomSizeCrop in training')
   cmd:text()

   local opt = cmd:parse(arg or {})

   
   if opt.dataset == 'imagenet' then
      -- Handle the most common case of missing -data flag
      local trainDir = paths.concat(opt.data, 'train')
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
   elseif opt.dataset == 'cifar10' then
      -- Default shortcutType=A and nEpochs=164
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end

   return opt
end

return M
