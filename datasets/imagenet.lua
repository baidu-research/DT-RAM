--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--

require 'torch'
local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local ImagenetDataset = torch.class('resnet.ImagenetDataset', M)

function ImagenetDataset:__init(list, opt, split)
   self.opt = opt
   self.imagePath, self.imageClass = self:LoadList(list)
   self.split = split
end

function ImagenetDataset:LoadList(infile)
    local list = {}
    local imageClasses = {}
    local fd=torch.DiskFile(infile, 'r')
    fd:quiet()
    print('loading file ' .. infile)
    local idx = 1
    while 1 do
        local d=fd:readString("*l");
        if fd:hasError() then
            break
        end
        local dd=string.split(d, '\t');
        if #dd > 1 then
            table.insert(list, dd[1])
            table.insert(imageClasses, dd[2]+0)
        end
        idx = idx + 1 
    end
    local nImages = #list
    local imagePath = torch.CharTensor(nImages, 1024):zero()
    for i, path in ipairs(list) do
        ffi.copy(imagePath[i]:data(), path)
    end
    local imageClass = torch.LongTensor(imageClasses)
    fd:close()
    list = nil
    imageClasses = nil
    collectgarbage("collect")
    return imagePath, imageClass
end

function ImagenetDataset:get(i)
   local path = ffi.string(self.imagePath[i]:data()) --ffi.string(self.imageInfo.imagePath[i]:data())

   local image = self:_loadImage(path)
   local class = self.imageClass[i] --self.imageInfo.imageClass[i]

   --print('read data:' .. path .. '\t' .. class)
   return {
      input = image,
      target = class,
   }
end

function ImagenetDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function ImagenetDataset:size()
   return self.imageClass:size(1)
   --return #(self.list)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
    mean = { 0.486, 0.501, 0.432 }, -- bird
    std = { 0.232, 0.228, 0.267 }, -- bird
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function ImagenetDataset:preprocess()
   if self.split == 'train' then
      local Crop = self.opt.RandomSizeCrop and t.RandomSizedCrop or t.RandomCrop
      local imageSize = self.opt.imageSize
      if self.opt.RandomSizeCrop then
         imageSize = self.opt.cropSize
      end 
      return t.Compose{
         -- t.RandomSizedCrop(224),
         t.Scale(imageSize),
         Crop(self.opt.cropSize),
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, pca.eigval, pca.eigvec),
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(self.opt.imageSize),
         t.ColorNormalize(meanstd),
         Crop(self.opt.cropSize),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.ImagenetDataset
