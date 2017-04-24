------------------------------------------------------------------------
--[[ TreeRecursor ]]--
-- Decorates module to be used within an AbstractTree.
-- It does this by making the decorated module conform to the
-- AbstractTree interface (which is inherited by RecRecTree classes)
------------------------------------------------------------------------
local TreeRecursor, parent = torch.class('tree2tree.TreeRecursor', 'tree2tree.AbstractTree')

function TreeRecursor:__init(module, rho)
   parent.__init(self, rho or 9999999)
   self.recurrentModule = module
   self.module = module
   self.modules = {module}
   self.sharedClones[1] = self.recurrentModule
end


function TreeRecursor:updateOutput(input)
   local output
   if self.train ~= false then -- if self.train or self.train == nil then
      -- set/save the output states
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      output = recurrentModule:updateOutput(input)
   else
      output = self.recurrentModule:updateOutput(input)
   end


   self.outputs[self.step] = output
   self.output = output
   self.step = self.step + 1
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   return self.output
end

function TreeRecursor:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)

   local recurrentModule = self:getStepModule(step)
   recurrentModule:setOutputStep(step)
   local gradInput = recurrentModule:updateGradInput(input, gradOutput)

   return gradInput
end

function TreeRecursor:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)

   local recurrentModule = self:getStepModule(step)
   recurrentModule:setOutputStep(step)
   recurrentModule:accGradParameters(input, gradOutput, scale)
end

function TreeRecursor:includingSharedClones(f)
   local modules = self.modules
   self.modules = {}
   local sharedClones = self.sharedClones
   self.sharedClones = nil
   for i,modules in ipairs{modules, sharedClones} do
      for j, module in pairs(modules) do
         table.insert(self.modules, module)
      end
   end
   local r = {f()}
   self.modules = modules
   self.sharedClones = sharedClones
   return unpack(r)
end

function TreeRecursor:forget(offset)
   parent.forget(self, offset)
   nn.Module.forget(self)
   return self
end

function TreeRecursor:maxBPTTstep(rho)
   self.rho = rho
   nn.Module.maxBPTTstep(self, rho)
end



TreeRecursor.__tostring__ = nn.Decorator.__tostring__
