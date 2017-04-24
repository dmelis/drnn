------------------------------------------------------------------------
--[[ Treequencer ]]--
-- Encapsulates a Module.
-- Input is a sequence (a table) of tensors.
-- Output is a sequence (a table) of tensors of the same length.
-- Applies the module to each element in the sequence.
-- Handles both recurrent modules and non-recurrent modules.
-- The sequences in a batch must have the same size.
-- But the sequence length of each batch can vary.
------------------------------------------------------------------------
--assert(not nn.Treequencer, "update nnx package : luarocks install nnx")
local Treequencer, parent = torch.class('tree2tree.Treequencer', 'nn.Container')

function Treequencer:__init(module, pass_topo_info, collect_loss, lr_tree)
   parent.__init(self)
   if not torch.isTypeOf(module, 'nn.Module') then
      error"Treequencer: expecting nn.Module instance at arg 1"
   end

   self.lr_tree = lr_tree or false
   self.pass_topo_info = pass_topo_info or false
   self.collect_loss = collect_loss or false

   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.module = (not torch.isTypeOf(module, 'tree2tree.AbstractTree')) and tree2tree.TreeRecursor(module) or module

   -- backprop through time (BPTT) will be done online (in reverse order of forward)
   self.modules = {self.module}

   self.output = {}

   -- table of buffers used for evaluation
   self._output = {}
   -- so that these buffers aren't serialized :
   local _ = require 'moses'
   self.dpnn_mediumEmpty = _.clone(self.dpnn_mediumEmpty)
   table.insert(self.dpnn_mediumEmpty, '_output')
   -- default is to forget previous inputs before each forward()
   self._remember = 'neither'
end

function Treequencer:updateOutput(input)
   --local tree = input[1]
   --local values = input[2]
   self.fwd_traversal = nil  -- In case they leaked from previous examples
   self.bwd_traversal = nil
   if self.lr_tree then
     assert(torch.type(self.tree) == 'tree2tree.LRTree', "expecting an input tree")
   else
     assert(torch.type(self.tree) == 'tree2tree.Tree', "expecting an input tree")
   end

   local total_loss = (self.collect_loss) and 0 or nil

   --assert(torch.type(inputTree) == 'table', "expecting input table")
   --print(values)
   --print(tree)
   -- Note that the Treequencer hijacks the rho attribute of the rnn
   --TODO: self.module:maxBPTTstep(#inputTable)
   if self.train ~= false or self.train == false then -- training  NOTE: OR Evaluating. Have no reason to do it different for now.
      if not (self._remember == 'train' or self._remember == 'both') then
         self.module:forget()
      end
      self.output = {}
      self.logprobs = {}
      local node_traversal,k = self:getTraversal('forward') -- wil depend on type of tree, should already remove padding and root, etc.

      debugutils.debug_topo_pred_header(Debug,self.lr_tree)
      for i, node_step in ipairs(node_traversal) do
        local node, prev_frat_node, next_frat_node, side = unpack(node_step) -- side will be nil for non LR tree
        if self.pass_topo_info then -- only done at train time
          self.module.module.has_brother  = next_frat_node and true or false
          if self.lr_tree then
            self.module.module.has_lchildren = (node.num_lchildren > 0) and true or false
            self.module.module.has_rchildren = (node.num_rchildren > 0) and true or false
          else
            self.module.module.has_children = (node.num_children > 0) and true or false
          end
        end
        if self.collect_loss then -- There will be two outputs, output and loss
          debugutils.dprintf(1,'%4i %4i |', node.index, node.idx)
          local output, loss = unpack(self.module:updateOutput(input[node.index]))
          -- Get argmax of word softmax
          local maxLogprobs, wt = torch.max(output, 1)
          wt = wt:view(-1):long()
          -- FIXME: Might need to do something similar to sampleLogprobs to make this work in batch mode too
          local correct_w = (wt[1] == node.label) and '' or 'X'
          debugutils.dprintf(1,'| %6i %4.4f %6s %2s |\n',wt[1], math.exp(maxLogprobs[1]),node.label, correct_w)
          self.output[node.index] = output
          self.logprobs[node.index] = maxLogprobs
          -- print(total_loss)
          -- print(loss)
          total_loss = total_loss + loss
        else
          self.output[node.index] = self.module:updateOutput(input[node.index])
        end
      end
      debugutils.debug_topo_pred_header(Debug,self.lr_tree,true)
   else -- evaluation  -- TODO: Fix this.
      if not (self._remember == 'eval' or self._remember == 'both') then
         self.module:forget()
      end
      local node_traversal,k = self:getTraversal('forward') -- wil depend on type of tree, should already remove padding and root, etc.
      bk()
      -- remove extra tensors





      -- during evaluation, recurrent modules reuse memory (i.e. outputs)
      -- so we need to copy each output into our own table
      --self.output = tree2tree:Tree():copy(inputTree,"skeleton")
      -- for innode, outnode in seq.zip(inputTree:depth_first_preorder(),self.output:depth_first_preorder()) do
      -- --for step, input in ipairs(inputTable) do
      --    outnode.value = self.module:updateOutput(innode.value)
      -- end

      -- TODO: Sequencer had this for evaluation. Do I need it?
      -- for step, input in ipairs(inputTable) do
      --    self.output[step] = nn.rnn.recursiveCopy(
      --       self.output[step] or table.remove(self._output, 1),
      --       self.module:updateOutput(input)
      --    )
      -- end
      -- -- remove extra output tensors (save for later)
      -- for i=#inputTable+1,#self.output do
      --    table.insert(self._output, self.output[i])
      --    self.output[i] = nil
      -- end
   end
   --print('treequencer, input type: ', torch.type(input),'output type: ', torch.type(self.output))
   local outputs = (self.collect_loss) and {self.output, total_loss} or self.output
   return outputs
end

function Treequencer:updateGradInput(input, gradOutput)
  if self.lr_tree then
    assert(torch.type(self.tree) == 'tree2tree.LRTree', "expecting an input tree")
  else
    assert(torch.type(self.tree) == 'tree2tree.Tree', "expecting an input tree")
  end
  self.gradInput = {}
  local node_traversal = self:getTraversal('backward') -- wil depend on type of tree, should already remove padding and root, etc.

  debugutils.debug_topo_grad_header(Debug,self.lr_tree)
  for i, node_step in ipairs(node_traversal) do
    local node, prev_frat_node, next_frat_node, side = unpack(node_step)
    if self.pass_topo_info then
      self.module.module.has_brother  = next_frat_node and true or false
      if self.lr_tree then
        self.module.module.has_lchildren = (node.num_lchildren > 0) and true or false
        self.module.module.has_rchildren = (node.num_rchildren > 0) and true or false
      else
        self.module.module.has_children = (node.num_children > 0) and true or false
      end
    end
    debugutils.dprintf(1,'%4i %4i | ', node.index, node.idx)
    self.gradInput[node.index] = self.module:updateGradInput(input[node.index],gradOutput[node.index])
    --print(  self.gradInput[node.index]); bk()
    --print('Here6:',torch.type(self.module),input[node.index]:norm(1),gradOutput[node.index]:norm(1),self.gradInput[node.index]:norm())
  end
  debugutils.debug_topo_grad_header(Debug,self.lr_tree, true)
  -- self.module.module.num_children = nil -- TODO: Add this to clean somewhere. Can't do it here because accGrad needs it still.
  -- self.module.module.has_brother  = nil
  return self.gradInput
end

function Treequencer:accGradParameters(input, gradOutput, scale)
  if self.lr_tree then
    assert(torch.type(self.tree) == 'tree2tree.LRTree', "expecting an input tree")
  else
    assert(torch.type(self.tree) == 'tree2tree.Tree', "expecting an input tree")
  end
  --assert(#gradOutput == #input, "gradOutput should have as many elements as input")
  --assert(#gradOutput == self.tree:size(), "gradOutput should have as many elements as input")
  --self.gradInput = tree2tree:Tree():copy(inputTree,"skeleton")
  local node_traversal = self:getTraversal('backward') -- wil depend on type of tree, should already remove padding and root, etc.
  for i, node_step in ipairs(node_traversal) do
    local node, prev_frat_node, next_frat_node, side = unpack(node_step)
    if self.pass_topo_info then
      self.module.module.has_brother  = next_frat_node and true or false
      if self.lr_tree then
        self.module.module.has_lchildren = (node.num_lchildren > 0) and true or false
        self.module.module.has_rchildren = (node.num_rchildren > 0) and true or false
      else
        self.module.module.has_children = (node.num_children > 0) and true or false
      end
    end
    self.module:accGradParameters(input[node.index],gradOutput[node.index], scale)
  end
end

function Treequencer:accUpdateGradParameters(input, gradOutput, lr)
  assert(torch.type(self.tree) == 'tree2tree.Tree', "expecting an input tree")
  assert(#gradOutput == #input, "gradOutput should have as many elements as input")
  assert(#gradOutput == self.tree:size(), "gradOutput should have as many elements as input")
  --self.gradInput = tree2tree:Tree():copy(inputTree,"skeleton")
  local node_traversal = self:getTraversal('backward') -- wil depend on type of tree, should already remove padding and root, etc.
  for i, node_step in ipairs(node_traversal) do
    local node, prev_frat_node, next_frat_node, side = unpack(node_step)
     self.module:accUpdateGradParameters(input[node.index],gradOutput[node.index], lr)
  end
end

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
-- Essentially, forget() isn't called on rnn module when remember is on
function Treequencer:remember(remember)
   self._remember = (remember == nil) and 'both' or remember
   local _ = require 'moses'
   assert(_.contains({'both','eval','train','neither'}, self._remember),
      "Treequencer : unrecognized value for remember : "..self._remember)
   return self
end

function Treequencer:training()
   if self.train == false then
      -- empty output table (tensor mem was managed by seq)
      for i,output in ipairs(self.output) do
         table.insert(self._output, output)
         self.output[i] = nil
      end
      -- forget at the start of each training
      self:forget()
   end
   parent.training(self)
end

function Treequencer:evaluate()
   if self.train ~= false then
     -- forget at the start of each evaluation
     self:forget()
      -- empty output table (tensor mem was managed by rnn)
      for i,output in ipairs(self.output) do
         table.insert(self._output, output)
         self.output[i] = nil
      end
   end
   -- NOTE: Ugly patch to fix bug here.
   --parent.evaluate(self) -- This was causing the bug too! I added the command below instead
   self.train = false
   assert(self.train == false)
end

-- These two came from AbstractSequencer
function Treequencer:sharedClone(shareParams, shareGradParams, clones, pointers, stepClone)
   -- stepClone is ignored (always false, i.e. uses sharedClone)
   return parent.sharedClone(self, shareParams, shareGradParams, clones, pointers)
end

-- AbstractSequence handles its own rho internally (dynamically)
function Treequencer:maxBPTTstep(rho)
end


function Treequencer:getTraversal(direction)
  -- should porbably only do this once at the beginning and save
  --local direction = direction or 'forward'
  assert(self.tree  ~= nil, "Error: no learning tree")
  assert(direction == 'forward' or direction == 'backward', "Wrong direction")
  local all_nodes, k
  local nodes = {}
  if (direction == 'forward') then
    if self.fwd_traversal == nil then
      nodes,k = self.tree:getTraversal()
      self.fwd_traversal = nodes
    else
      nodes = self.fwd_traversal
      k = #self.fwd_traversal
    end
  else -- backward
    if self.bwd_traversal == nil then
      assert(self.fwd_traversal, "Error: no previous forward traversal found.")
      for i,node_fwd in pairs(self.fwd_traversal) do
        nodes[#self.fwd_traversal -i + 1] = node_fwd
      end
      self.bwd_traversal = nodes
    else
      nodes = self.bwd_traversal
      k = #self.bwd_traversal
    end
  end
  return nodes, k
end



Treequencer.__tostring__ = nn.Decorator.__tostring__
