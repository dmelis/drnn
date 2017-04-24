local _ = require 'moses'

local AbstractTree, parent = torch.class('tree2tree.AbstractTree', 'nn.Container')

AbstractTree.dpnn_stepclone = true

function AbstractTree:__init(rho)
   parent.__init(self)

   self.rho = rho or 99999 --the maximum number of time steps to BPTT

   self.outputs = {}
   self.gradInputs = {}
   self._gradOutputs = {}

   self.step = 1

   -- stores internal states of Modules at different time-steps
   self.sharedClones = {}

   self:reset()
end

function AbstractTree:getStepModule(step)
   local _ = require 'moses'
   assert(step, "expecting step at arg 1")
   local recurrentModule = self.sharedClones[step]
   if not recurrentModule then
      recurrentModule = self.recurrentModule:stepClone()
      self.sharedClones[step] = recurrentModule
      self.nSharedClone = _.size(self.sharedClones)
   end
   return recurrentModule
end


-- DONT TOUCH THESE. These are used by TreeRecursor too!
function AbstractTree:updateGradInput(input, gradOutput)
   -- updateGradInput should be called in reverse order of time
   self.updateGradInputStep = self.updateGradInputStep or self.step
   -- BPTT for one time-step
   self.gradInput = self:_updateGradInput(input, gradOutput)
   self.updateGradInputStep = self.updateGradInputStep - 1
   self.gradInputs[self.updateGradInputStep] = self.gradInput
   return self.gradInput
end

function AbstractTree:accGradParameters(input, gradOutput, scale)
   -- accGradParameters should be called in reverse order of time
   assert(self.updateGradInputStep < self.step, "Missing updateGradInput")
   self.accGradParametersStep = self.accGradParametersStep or self.step
   -- BPTT for one time-step
   self:_accGradParameters(input, gradOutput, scale)
   self.accGradParametersStep = self.accGradParametersStep - 1
end

-- function AbstractTree:updateGradInput(input, gradOutput)
--   print('Hello from abstracttree updateGRad')
--    -- updateGradInput should be called in reverse order of time
--    self.updateGradInputStep = self.updateGradInputStep or self.step
--    -- BPTT for one time-step
--    --print(#input)
--    print(torch.type(input))
--    bk()
--    self.gradInput = self:_updateGradInput(input, gradOutput)
--    self.updateGradInputStep = self.updateGradInputStep - 1
--    self.gradInputs[self.updateGradInputStep] = self.gradInput
--    print(torch.type(self.gradInput), #self.gradInput, self.gradInput:norm(1))
--    return self.gradInput
-- end

-- function AbstractTree:accGradParameters(input, gradOutput, scale)
--    -- accGradParameters should be called in reverse order of time
--    assert(self.updateGradInputStep < self.step, "Missing updateGradInput")
--    self.accGradParametersStep = self.accGradParametersStep or self.step
--    -- BPTT for one time-step
--    self:_accGradParameters(input, gradOutput, scale)
--    self.accGradParametersStep = self.accGradParametersStep - 1
-- end

-- goes hand in hand with the next method : forget()
-- this methods brings the oldest memory to the current step
function AbstractTree:recycle(offset)
   -- offset can be used to skip initialModule (if any)
   local _ = require 'moses' 
   offset = offset or 0
   self.nSharedClone = self.nSharedClone or _.size(self.sharedClones)
   local rho = math.max(self.rho + 1, self.nSharedClone)
   if self.sharedClones[self.step] == nil then
      self.sharedClones[self.step] = self.sharedClones[self.step-rho]
      self.sharedClones[self.step-rho] = nil
      self._gradOutputs[self.step] = self._gradOutputs[self.step-rho]
      self._gradOutputs[self.step-rho] = nil
   end
   self.outputs[self.step-rho-1] = nil
   self.gradInputs[self.step-rho-1] = nil
   return self
end

function AbstractTree:clearState()
   nn.utils.clear(self, '_input', '_gradOutput', '_gradOutputs', 'sharedClones', 'gradPrevOutput', 'cell', 'cells', 'gradCells')
   self.nSharedClone = 0
   return parent.clearState(self)
end

function AbstractTree:forget()
   -- the recurrentModule may contain an AbstractTree instance (issue 107)
   parent.forget(self)

    -- bring all states back to the start of the sequence buffers
   if self.train ~= false then
      self.outputs = {}
      self.gradInputs = {}
      if self.sharedClones['L'] then -- lr tree
        self.sharedClones['L'] = _.compact(self.sharedClones)
        self.sharedClones['R'] = _.compact(self.sharedClones)
      else
        self.sharedClones = _.compact(self.sharedClones)
      end
      self._gradOutputs = _.compact(self._gradOutputs)
   end

   -- forget the past inputs; restart from first step
   self.step = 1


  if not self.rmInSharedClones then
      -- Asserts that issue 129 is solved. In forget as it is often called.
      -- Asserts that self.recurrentModule is part of the sharedClones.
      -- Since its used for evaluation, it should be used for training.
      local nClone, maxIdx = 0, 1
      for k,v in pairs(self.sharedClones) do -- to prevent odd bugs
         if torch.pointer(v) == torch.pointer(self.recurrentModule) then
            self.rmInSharedClones = true
            maxIdx = math.max(k, maxIdx)
         end
         nClone = nClone + 1
      end
      if nClone > 1 then
         if not self.rmInSharedClones then
            print"WARNING : recurrentModule should be added to sharedClones in constructor."
            print"Adding it for you."
            assert(torch.type(self.sharedClones[maxIdx]) == torch.type(self.recurrentModule))
            self.recurrentModule = self.sharedClones[maxIdx]
            self.rmInSharedClones = true
         end
      end
   end
   return self
end


-- TODO: Function clear states


function AbstractTree:includingSharedClones(f)
   local modules = self.modules
   local sharedClones = self.sharedClones
   self.sharedClones = nil
   self.modules = {}
   for i,modules in ipairs{modules, sharedClones} do
      for j, module in pairs(modules or {}) do
         table.insert(self.modules, module)
      end
   end
   local r = {f()}
   self.modules = modules
   self.sharedClones = sharedClones
   return unpack(r)
end


function AbstractTree:type(type, tensorcache)
   return self:includingSharedClones(function()
      return parent.type(self, type, tensorcache)
   end)
end

function AbstractTree:training()
   return self:includingSharedClones(function()
      return parent.training(self)
   end)
end

function AbstractTree:evaluate()
   return self:includingSharedClones(function()
      return parent.evaluate(self)
   end)
end


-- used by Recursor() after calling stepClone.
-- this solves a very annoying bug...
function AbstractTree:setOutputStep(step)
   self.output = self.outputs[step] --or self:getStepModule(step).output
   assert(self.output, "no output for step "..step)
   self.gradInput = self.gradInputs[step]
end

-- function ChildSumTreeLSTM:clean(tree)
--   self:free_module(tree, 'composer')
--   self:free_module(tree, 'output_module')
--   tree.state = nil
--   tree.output = nil
--   for i = 1, tree.num_children do
--     self:clean(tree.children[i])
--   end
-- end
