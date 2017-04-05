local _ = require 'moses'

local Module = nn.Module

function Module:findName(name, container)
  container = container or self
  local nodes = {}
  local containers = {}
  local current_name = self.name
  if current_name == name then
    nodes[#nodes+1] = self
    containers[#containers+1] = container
  end
  -- Recurse on nodes with 'modules'
  if (self.modules ~= nil) then
    if (torch.type(self.modules) == 'table') then
      for i = 1, #self.modules do
        local child = self.modules[i]
        local cur_nodes, cur_containers =
        child:findName(name, self)
        assert(#cur_nodes == #cur_containers,
          'Internal error: incorrect return length')  -- This shouldn't happen
        -- add the list items from our child to our list (ie return a
        -- flattened table of the return nodes).
        for j = 1, #cur_nodes do
          nodes[#nodes+1] = cur_nodes[j]
          containers[#containers+1] = cur_containers[j]
        end
      end
    end
  end
  return nodes, containers
end

function Module:getParameters()
   -- get parameters
   local parameters,gradParameters = self:parameters()
   local p, g = Module.flatten(parameters), Module.flatten(gradParameters)
   assert(p:nElement() == g:nElement(),
      'check that you are sharing parameters and gradParameters')
   if parameters then
      for i=1,#parameters do
         assert(parameters[i]:storageOffset() == gradParameters[i]:storageOffset(),
            'misaligned parameter at ' .. tostring(i))
      end
   end
   return p, g
end
