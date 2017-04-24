-----------------------------------------------------------------------
--[[ TreeCriterion ]]--
-- (taken from https://github.com/Element-Research/rnn/blob/master/TreeCriterion.lua)
-- Applies a criterion to each of the inputs and targets in the
-- corresponding input and target Tables.
-- Useful for nn.Repeater and nn.Tree.
-- WARNING : assumes that the decorated criterion is stateless, i.e.
-- the backward doesn't need to be preceded by a commensurate forward.
------------------------------------------------------------------------
local TreeCriterion, parent = torch.class('tree2tree.TreeCriterion', 'nn.Criterion')

function TreeCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   if torch.isTypeOf(criterion, 'nn.ModuleCriterion') then
      error("TreeCriterion shouldn't decorate a ModuleCriterion. "..
         "Instead, try the other way around : "..
         "ModuleCriterion decorates a TreeCriterion. "..
         "Its modules can also be similarly decorated with a Tree.")
   end
   self.clones = {}
   self.gradInput = {}
end

function TreeCriterion:getStepCriterion(step)
   assert(step, "expecting step at arg 1")
   local criterion = self.clones[step]
   if not criterion then
      criterion = self.criterion:clone()
      self.clones[step] = criterion
   end
   return criterion
end

-- TODO: updateOutput should check that the input and target trees have exactly the same topology before proceeding
function TreeCriterion:updateOutput(input, target)
  -- Can only take two arguments since it extends Criterion class.
  self.output = 0
  local targetSent, targetTree  = target.sent, target.tree
  local inputTree, inputVectors = input.tree, input.sent
  local input_nodes = inputTree:getTraversal()
  local target_nodes = targetTree:getTraversal()
  assert(#input_nodes == #target_nodes,"Error: Dont know how to do criterion on traversals of different length")
  for step_input, step_target in seq.zip(input_nodes, target_nodes) do
    local innode = step_input[1]
    local targnode = step_target[1]
    local criterion = self:getStepCriterion(targnode.index)
    self.output = self.output + criterion:forward(inputVectors[innode.index],
                                  targetSent[targnode.idx])
    debugutils.debug_criterion(2,innode.index,innode.idx,inputVectors[innode.index],targetSent[targnode.idx])
    --print(inputVectors[innode.index]:norm())
  end
  return self.output
end

-- Older tree format version of updateOuptut
-- function TreeCriterion:updateOutput(input, target)
--   -- Can only take two arguments since it extends Criterion class.
--    self.output = 0
--    local targetSent = target.sent
--    local targetTree = target.tree
--    --print('gerge')
--    --print(targetSent)
--    --print(targetTree:size())
--    --assert(inputTree:size() == targetTree:size(),"Error: Dont know how to do do criterion on trees of dif size")
--    local input_nodes  = inputTree:depth_first_preorder()
--    local target_nodes = targetTree:depth_first_preorder()
--    --print(#input_nodes, #target_nodes)
--    for innode, targnode in seq.zip(input_nodes, target_nodes) do
--       local index = targnode.index
--       local criterion = self:getStepCriterion(index)
--       --print(innode.value:size(), targnode.value:size())
--       self.output = self.output + criterion:forward(innode.value, targetSent[targnode.idx])
--    end
--    return self.output
-- end


-- function TreeCriterion:_updateGradInput(input, target, grad)
--    local input_nodes  = inputTree:depth_first_preorder()
--    local target_nodes = targetTree:depth_first_preorder()
--    local i = 1
--    for innode, targnode in seq.zip(input_nodes, target_nodes) do
--       local criterion = self:getStepCriterion(i)
--       self.gradInput[i] = criterion:forward(input.value, targetSent[targnode.idx])
--
--
--    for i,input in ipairs(inputTable) do
--       local criterion = self:getStepCriterion(i)
--    end
--
--    return self.gradInput
-- end


--[[
  Note that target is a table with keys sent and tree.
]]--


function TreeCriterion:updateGradInput(input, target)
  self.gradInput = {}
  local targetSent, targetTree  = target.sent, target.tree
  local inputTree, inputVectors = input.tree, input.sent
  local input_nodes  = inputTree:getTraversal()
  local target_nodes = targetTree:getTraversal()
  assert(#input_nodes == #target_nodes,"Error: Dont know how to do criterion on traversals of different length")
  for step_input, step_target in seq.zip(input_nodes, target_nodes) do
    local innode = step_input[1]
    local targnode = step_target[1]
    local criterion = self:getStepCriterion(targnode.index)
    --print(inputVectors[innode.index]:norm(), targetSent[targnode.idx])
    self.gradInput[targnode.index] = criterion:backward(inputVectors[innode.index], targetSent[targnode.idx])
    --print(self.gradInput[targnode.index]:size(),self.gradInput[targnode.index]:norm() )
  end
  return self.gradInput
end




-- Older, tree shaped version:
-- function TreeCriterion:updateGradInput(inputTree, target)
--   local to_tree = true
--   self.gradInput = {}
--   local targetSent = target.sent
--   local targetTree = target.tree
--   local input_nodes  = inputTree:depth_first_preorder()
--   local target_nodes = targetTree:depth_first_preorder()
--   for innode, targnode in seq.zip(input_nodes, target_nodes) do
--     local index = targnode.index
--     local criterion = self:getStepCriterion(index)
--     self.gradInput[index] = criterion:backward(innode.value, targetSent[targnode.idx])
--   end
--   -- If want gradients as a tree:
--   if to_tree then
--     local gradInputTree = tree2tree.Tree():copy(inputTree,"skeleton")
--     local grad_nodes = gradInputTree:depth_first_preorder()
--     for k,node in ipairs(grad_nodes) do
--       node.value = self.gradInput[node.index]
--     end
--     self.gradInput = gradInputTree
--   end
--   return self.gradInput
-- end



-- function TreeCriterion:forward(tree, inputTable, targetTable)
--   -- top to bottom error computation
--   self.output = 0   -- collects error
--   for i = 1, tree.num_children do
--     local criterion = self:getStepCriterion(i)
--     self.output = self.output + criterion:forward(input, targetTable[i])
--
--
--     local _, child_loss = self:forward(tree.children[i], inputs)
--     loss = loss + child_loss
--   end
--   local child_c, child_h = self:get_child_states(tree)
--   self:allocate_module(tree, 'composer')
--   tree.state = tree.composer:forward{inputs[tree.idx], child_c, child_h}
--
--   if self.output_module ~= nil then
--     self:allocate_module(tree, 'output_module')
--     tree.output = tree.output_module:forward(tree.state[2])
--     if self.train and tree.gold_label ~= nil then
--       loss = loss + self.criterion:forward(tree.output, tree.gold_label)
--     end
--   end
--   return tree.state, loss
-- end
