require 'nn'
require 'cunn'
require 'cudnn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
	local columns = 4
	local join_block = nn.CAddTable
	local function fractal_layer_preactive_unit(nInputPlane, nOutputPlane)
		local b1 = nn.Sequential()
		b1:add(Convolution(nInputPlane, nOutputPlane, 3, 3, 1, 1,1, 1))
		b1:add(SBatchNorm(nOutputPlane))
		b1:add(ReLU(true))
		b1:add(Convolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1,1, 1))
		return nn.Sequential()
				:add(nn.ConcatTable()
					:add(b1)
					:add(Convolution(nInputPlane, nOutputPlane, 3, 3, 1, 1,1, 1)))
				:add(join_block(true))
	end

	local function fractal_layer_preactive(nInputPlane, nOutputPlane, cols)
		--if cols == 1:
		--	return nn.Sequential()
		--		:add(Convolution(nInputPlane, nOutputPlane, 3, 3, 1, 1,1, 1))
		local ret = nn.Sequential()
		ret:add(SBatchNorm(nInputPlane))
		ret:add(ReLU(true))

		local s = nn.ConcatTable()
		s:add(Convolution(nInputPlane, nOutputPlane, 3, 3, 1, 1,1, 1))

		local s1a = nn.Sequential()
		s1a:add(fractal_layer_preactive_unit(nInputPlane, nOutputPlane))
		s1a:add(SBatchNorm(nOutputPlane))
		s1a:add(ReLU(true))
		s1a:add(fractal_layer_preactive_unit(nOutputPlane, nOutputPlane))

		local s1 = nn.Sequential()
					:add(nn.ConcatTable()
						:add(Convolution(nInputPlane, nOutputPlane, 3, 3, 1, 1,1, 1))
						:add(s1a))
					:add(join_block(true))

		s1:add(SBatchNorm(nOutputPlane))
		s1:add(ReLU(true))
        
        local s1b = nn.Sequential()
        s1b:add(fractal_layer_preactive_unit(nOutputPlane, nOutputPlane))
        s1b:add(SBatchNorm(nOutputPlane))
        s1b:add(ReLU(true))
        s1b:add(fractal_layer_preactive_unit(nOutputPlane, nOutputPlane))

		s1:add(nn.ConcatTable()
					:add(Convolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1,1, 1))
					:add(s1b))
				:add(join_block(true))

		s:add(s1)
		--s:add(join_block(true))

		ret:add(s)
		ret:add(join_block(true))
        return ret
	end

    local nFeatures = 1024

	local model = nn.Sequential()
	model:add(Convolution(3,64,7,7,2,2,3,3))
	model:add(SBatchNorm(64))
	model:add(ReLU(true))
	model:add(Max(3,3,2,2,1,1))

	model:add(fractal_layer_preactive(64, 128, 4))
	model:add(Max(3,3,2,2,1,1))
	model:add(fractal_layer_preactive(128, 256, 4))
	model:add(Max(3,3,2,2,1,1))
	model:add(fractal_layer_preactive(256, 512, 4))
	model:add(Max(3,3,2,2,1,1))
	model:add(fractal_layer_preactive(512, 1024, 4))
	model:add(SBatchNorm(1024))
	model:add(ReLU(true))
	model:add(Avg(7, 7, 1, 1))
	model:add(nn.View(nFeatures):setNumInputDims(3))
	model:add(nn.Linear(nFeatures, 100))

	local function ConvInit(name)
		for k,v in pairs(model:findModules(name)) do
			local n = v.kW*v.kH*v.nOutputPlane
			v.weight:normal(0,math.sqrt(2/n))
			if cudnn.version >= 4000 then
				v.bias = nil
				v.gradBias = nil
			else
				v.bias:zero()
			end
		end
	end
	local function BNInit(name)
		for k,v in pairs(model:findModules(name)) do
			v.weight:fill(1)
			v.bias:zero()
		end
	end
	
	ConvInit('cudnn.SpatialConvolution')
	ConvInit('nn.SpatialConvolution')
	BNInit('fbnn.SpatialBatchNormalization')
	BNInit('cudnn.SpatialBatchNormalization')
	BNInit('nn.SpatialBatchNormalization')

	for k,v in pairs(model:findModules('nn.Linear')) do
		v.bias:zero() 
	end
	model:cuda()

	if opt.cudnn == 'deterministic' then
		model:apply(function(m)
			if m.setMode then m:setMode(1,1,1) end
		end)
	end

	model:get(1).gradInput = nil

	return model
end

return createModel


	

