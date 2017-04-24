--[[

      Doubly-recurrent neural network



    Important observations:
      [1] We share gradients (in addition to weights) because we update via :getParameters.
          See https://github.com/torch/nn/blob/master/doc/overview.md#nn.overview.sharedparams for details.
      [2]
--]]

--flatten = false


-- Variable meaning:
-- hA - hidden ancestral state
-- hF - hidden fraternal state

-- TODO: assert at some point that type of tree provided is consistent with type of
-- dynamics (e.g. left right)


local DRNN, parent = torch.class('tree2tree.DRNN', 'tree2tree.AbstractTree')

function DRNN:__init(config)
  parent.__init(self)

  self.in_dim     = config.in_dim or 300
  self.mem_dim    = config.mem_dim or 150
  self.out_dim    = config.vocab_dim
  self.in_zeros   = torch.zeros(self.in_dim)
  self.mem_zeros  = torch.zeros(self.mem_dim)

  self.model      = config.model or 'ntp'   -- {ntp,stp,dtp}
  self.recurrence = config.recurrence or 'gru'
  self.DM_type    = self.recurrence
  self.WM_type    = self.recurrence
  self._remember  = 'neither'

  self.output = {}
  self.tree = nil      -- guiding tree
  self.lr_tree = config.lr_tree or false -- use l-r tree
  --self.step = 1

  -- stores internal states of Modules at different time-steps
  --self.sharedClones = {}
  self.modules = {}

  -- if self.lr_tree then
  --   self.recurrentModuleL, self.recurrentModuleR = self:buildModel()
  -- else
  --   self.recurrentModule = self:buildModel()
  -- end
  self.recurrentModule = self:buildModel()

-- make it work with nn.Container

  if not self.lr_tree then
    self.modules[1] = self.recurrentModule
    --self.sharedClones[1] = self.recurrentModule
  else
    self.modules[1] = self.recurrentModule['L']
    self.modules[2] = self.recurrentModule['R']
    self.sharedClones = {L = {}, R= {}}
    --self.sharedClones['L'][1] = self.recurrentModule['L']

    --self.sharedClones['R'][1] = self.recurrentModule['R']
  end
  self.stateFrat        = {}
  self.stateAncest      = {}
  self.gradFrat         = {}
  self.gradAncest       = {}
  self.gradInput        = {}
  if (self.model == 'stp') or (self.model == 'dtp') then
    self.logprob_ancest  = {}
  end
  if self.model == 'dtp' then
    self.logprob_frat       = {}
  end
  --self.userPrevAncestral = nil

  self:reset()
end
---

function DRNN:buildModel()
  local d_mod       = factory.new_recurrent_module(self.DM_type,self.in_dim,self.mem_dim)
  local w_mod       = factory.new_recurrent_module(self.WM_type,self.in_dim,self.mem_dim)
  local state_comb  = factory.new_combination_module(self.mem_dim)

  -- Model selector
  -- TODO: add lr tree models to ntp and stp
  local cell_builder -- this will be a function!
  if self.model == 'ntp' then
    cell_builder = factory.model_ntp_composer
  elseif self.model == 'stp' and not self.lr_tree then
    cell_builder = factory.model_stp_composer
  elseif self.model == 'stp' and self.lr_tree then
    cell_builder = factory.model_stp_lr_composer
  elseif self.model == 'dtp' and not self.lr_tree then
    cell_builder = factory.model_dtp_composer        -- dtp
  elseif self.model == 'dtp' and self.lr_tree then
    cell_builder = factory.model_dtp_lr_composer        -- dtp + L/R
  else
    print("ERROR in DRNN: unrecognized model. Aborting")
    os.exit()
  end

  local cell = cell_builder(d_mod, w_mod, state_comb, self.in_dim, self.mem_dim, self.out_dim)

  if not self.lr_tree then
    -- We're done, return
    return cell
  else
    -- Create another type of cell for left children
    local cell_R = cell
    local d_mod_L      = factory.new_recurrent_module(self.DM_type,self.in_dim,self.mem_dim)
    local w_mod_L      = factory.new_recurrent_module(self.WM_type,self.in_dim,self.mem_dim)
    local state_comb_L = factory.new_combination_module(self.mem_dim)

    d_mod_L:share(d_mod,'weight','bias','gradWeight','gradBias') -- See obs [1]
    --share_params(depth_module_left,depth_module) -- TODO: Which of the two share syntaxes use?
    state_comb_L:share(state_comb, 'weight','bias','gradWeight','gradBias')

    cell_L = cell_builder(d_mod_L, w_mod_L, state_comb_L, self.in_dim, self.mem_dim, self.out_dim)
    local cells_LR = {L = cell_L, R = cell_R}
    return cells_LR
  end
end



-----
-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional DRNNs).
-- Returns the final hidden state of the DRNN.
function DRNN:updateOutput(input)
  self._input_size = #input -- for debugging

  --flesh_out_sequential(self.recurrentModule['L'])
  local output_states, output_tree
  if (not self.predict) then -- Training or evaluation
    if (self.train ~= false) and not (self._remember == 'train' or self._remember == 'both') then
        --self:forget() -- clears out any persisten memory in modules.
         self:forget()
    else -- eval
      if not (self._remember == 'eval' or self._remember == 'both') then
         self:forget()
      end
    end
    assert(next(self.gradAncest) == nil and next(self.gradFrat) == nil, "Error, previous grads found")
    assert(self.tree ~= nil, "expecting a teaching tree")
    debugutils.dprintf(2,'%8s %8s %8s %8s %8s %8s %8s %8s\n','Index','Idx','P.Index','P.Idx','B.Index','B.Idx','||h_fr||','||h_an||')

    local node_traversal = self:getTraversal('forward') -- wil depend on type of tree, should already remove padding and root, etc.

    debugutils.debugtraversal(2,node_traversal)

    for i, node_step in ipairs(node_traversal) do
      local node, prev_frat_node, _, side = unpack(node_step) -- side will be nil for non LR tree
      local inA, hA_prev, inF, hF_prev = self:get_node_inputs(input,node, prev_frat_node, side)

      -- self:recycle() needed?
      local recurrentModule
      if self.train ~= false then
        recurrentModule = (side) and self:getStepModule(node.index,side) or self:getStepModule(node.index)
      else
        -- FIXME: Im doing getStepModule to avoid the bug, this is not very efficient.
        -- A more efficient, analogoys to the recylicling that happens with sequences is
        -- to get rid of node once father, brother have used it.
        recurrentModule = (side) and self:getStepModule(node.index,side) or self:getStepModule(node.index)
        --recurrentModule = (side) and self.recurrentModule[side] or self.recurrentModule
      end

      --debugutils.dprintf(2,'Norms before entering cell fwd: %8.2f %8.2f %8.2f %8.2f\n',
      --inF:norm(), hF_prev:norm(), inA:norm(), hA_prev:norm())

      local cell_output_table = recurrentModule:updateOutput{inA, hA_prev, inF, hF_prev}

      -- Need to unpack according to the type of model. Some have additional outputs.
      local hA, hF, hOut, pA, pA_L, pA_R, pF     -- some will be nil for some models

      if self.model == 'stp' then
        hA, hF, hOut = unpack(cell_output_table)
      elseif self.model == 'stp' and not self.lr_tree then
        hA, hF, hOut, pA = unpack(cell_output_table)
      elseif self.model == 'stp' and self.lr_tree then
        hA, hF, hOut, pA_L, pA_R = unpack(cell_output_table)
      elseif self.model == 'dtp' and not self.lr_tree then
        hA, hF, hOut, pA, pF = unpack(cell_output_table)
      elseif self.model == 'dtp' and self.lr_tree then
        hA, hF, hOut, pA_L, pA_R, pF = unpack(cell_output_table)
      else
        print("ERROR in DRNN: unrecognized model. Aborting")
        os.exit()
      end

      debugutils.debug_DRNN_forward(Debug,self.model,self.lr_tree,#node_traversal,i,
      node,prev_frat_node,inA,hA_prev,inF,hF_prev,hA,hF,hOut,pA,pA_L,pA_R,pF)
      --if node.index == 2 then print(pA,pF) end

      -- Every model has these:
      self.stateAncest[node.index]   = hA
      self.stateFrat[node.index]     = hF

      local output -- this will be the "true" output, returned by this module

      -- TODO: I don't think I use the logprob_ancest values anywhere. Check and delete if not.
      if (self.model == 'ntp') then
        output = hOut
      elseif (self.model == 'stp') and not self.lr_tree then
        self.logprob_ancest[node.index] = pA
        output = {hOut, pA}
      elseif (self.model == 'stp') and self.lr_tree then
        self.logprob_ancest[node.index] = {L = torch.log(pA_L), R = torch.log(pA_R)}
        output = {hOut, pA_L, pA_R}
      elseif (self.model == 'dtp') and not self.lr_tree then
        self.logprob_ancest[node.index] = torch.log(pA)
        self.logprob_frat[node.index]   = torch.log(pF)
        output = {hOut, pA, pF}
      elseif (self.model == 'dtp') and self.lr_tree then
        self.logprob_ancest[node.index] = {L = torch.log(pA_L), R = torch.log(pA_R)}
        self.logprob_frat[node.index]      = torch.log(pF)
        output = {hOut, pA_L, pA_R, pF}
      end
      self.output[node.index]          = output
      self._output = output -- Use underscore not to override "true" output of module, a table
    end -- end for over nodes

    -- For trainig we simply use the tree topology that we used for forwarding, prune
    -- output_tree =  tree2tree.Tree():copy(self.tree.children[1])
    -- output_tree:prune_leftmost_leaves()
    output_tree = self.tree:prune_padding_leaves('SOS')
    self.outputTree = output_tree -- passed by TreeSequential to next module
  else -- i.e. not training NOR evaluation. Here we should deal with prediction setting.
    debugutils.dprint(1,'Will start tree prediction now.')
    assert(not (next(self.stateFrat) or next(self.stateAncest)), "Previous states found. Aborting")
    assert(self.userPrevAncestral and self.userPrevAncestral:norm() >0, "Missing or null encoded priming state")
    --assert(not self.tree, "Previous tree found. Aborting")
    assert(self.topologyModule, "Topology module not passed on correctly") -- Has to be passed from outside!
    if self.lr_tree then
      self.tree.lchildren[1].depth = 1
      self.predictionQueue = {self.tree.lchildren[1]} --  nodes yet to predict. Will work as queue (stack) if breadth_first (depth-first) prediction
    else
      self.tree.children[1].depth = 1
      self.predictionQueue = {self.tree.children[1]} --  nodes yet to predict. Will work as queue (stack) if breadth_first (depth-first) prediction
    end
    -- In prediction setting, logprobs contains both topo and word probabilities, since they are all computed in-house
    wordIds, logprobs = self:forward_predict_nodewise(self.predictionQueue, input)
    self.logprobs = logprobs
    -- Overall logprob is sum of all collected probabilities
    local overall_logprob = rnn.recursiveSum(logprobs)


    --output_tree = self.tree:prune_padding_leaves('SOS')
    output_tree = self.tree -- TODO: Revert back to above after debug, get rid of remove padding in Tree2Tree
    self.outputTree = output_tree -- passed by TreeSequential to next module
    debugutils.dprint(1,'Done predicting. Tree size is :',self.outputTree:size())
    return {self.outputTree, wordIds, overall_logprob}
  end -- end if train/eval/pred

  if Debug > 1 then
    local nonempty_indices = {}
    for i,p in pairs(self.output) do if p then table.insert(nonempty_indices,i) end end
    --debugutils.dprintf(1,"Leaving DRNN:updateOutput. %5i elems in output, \
    --(%5i nonempty), of size %5i, norm of root node: %8.2f\n",
    --#self.output,#nonempty_indices,self.output[2]:size(1),self.output[2]:norm(1))
  end

  return self.output
end

-- -- Requires that :forward() was run before
-- function DRNN:output_loglikelihood()
--   local log_prob
--   for i=1,#self.output do
--     local node_logprob
--     if (self.model == 'ntp') then
--       hOut = self.output[i]
--       node_logprob = math.log(hOut)
--     elseif (self.model == 'stp') and not self.lr_tree then
--       hOut, pA = unpack(self.output[i])
--     elseif (self.model == 'stp') and self.lr_tree then
--       hOut, pA_L, pA_R = unpack(self.output[i])
--     elseif (self.model == 'dtp') and not self.lr_tree then
--       hOut, pA, pF = unpack(self.output[i])
--     elseif (self.model == 'dtp') and self.lr_tree then
--       hOut, pA_L, pA_R, pF = unpack(self.output[i])
--     end
--     log_prob = log_prob + node_logprob
--   end
--   return log_prob
-- end
--

MAX_TREE_DEPTH = 5
MAX_OFFSPRING = 4 -- not including padding

-- Recursive prediction function
function DRNN:forward_predict_nodewise(to_process, inputs)
  local wordIds, logprobs = {}, {}
  local depth
  local string_chronol_words  = '' -- Debug
  debugutils.dprint(1,'Hello from prediction nodewise')
  --if not to_process[1].idx then to_process[1].idx = to_process[1].index + 2

  --ocal next_idx   = next_index + 1
  assert(to_process[1].index and to_process[1].idx, "Priming tree root missing index or idx")
  local next_index = to_process[1].index + 1
  local next_idx   = to_process[1].idx + 1

  local total_steps = 0
  while next(to_process) and (total_steps < 1000) do
    local stop_fraternity = false
    local stop_fertility = (self.lr_tree) and {L = false, R = false} or false
    total_steps = total_steps +1
    local node = table.remove(to_process)  -- Process top of the queue
    local child_no = (node.index ==2) and 1
            or (self.lr_tree and node.side == 'L') and node.parent.num_lchildren
            or (self.lr_tree and node.side == 'R') and node.parent.num_rchildren
            or node.parent.num_children

    --node.idx = (node.index == 2) and 3 or next_idx
    local prev_frat_node = (self.lr_tree and node.side == 'L') and node.right_brother or node.left_brother
    local inA, hA_prev, inF, hF_prev = self:get_node_inputs(inputs,node,prev_frat_node,node.side)

    -- debugutils.dprintf(1,'%5i %5i %5i %5i %5i %5s %5i %5i\n',total_steps,self.tree:size(true),#to_process,node.idx, node.index,node.side, node.depth, node.parent.index)
    -- debugutils.dprintf(1,'%8s %8s %8s %8s %8s %8s %8s %8s\n','Index','Idx','P.Index','P.Idx','B.Index','B.Idx','||h_fr||','||h_an||')

    --print(hA_prev:nDimension(), hF_prev:nDimension())
    --print(hA_prev:norm())



    -- Need to unpack according to the type of model. Some have additional outputs.
    local cell_output_table
    if self.lr_tree then
      cell_output_table = self.recurrentModule[node.side]:updateOutput{inA, hA_prev, inF, hF_prev}
      self.recurrentModule[node.side]:updateOutput{inA, hA_prev, inF, hF_prev}
      cell_output_table = {}
      cell_output_table = rnn.recursiveCopy(cell_output_table,self.recurrentModule[node.side].output)
    else
      --local cell_output_table = self.recurrentModule[node.side]:updateOutput{inA, hA_prev, inF, hF_prev}
      self.recurrentModule:updateOutput{inA, hA_prev, inF, hF_prev}
      cell_output_table = {}
      cell_output_table = rnn.recursiveCopy(cell_output_table,self.recurrentModule.output)
    end


    -- self.tgt_emb_module:forward(priming_sent)
    --local emb_sent = torch.Tensor(self.tgt_emb_module.output:size()):copy(self.tgt_emb_module.output))


    local hA, hF, hOut, pA, pA_L, pA_R, pF     -- some will be nil for some models
    local input_to_topology
    if self.model == 'ntp' then
      hA, hF, hOut = unpack(cell_output_table)
      input_to_topology = {hOut}
    elseif self.model == 'stp' and not self.lr_tree then
      hA, hF, hOut, pA = unpack(cell_output_table)
      input_to_topology = {hOut,pA}
    elseif self.model == 'stp' and self.lr_tree then
      hA, hF, hOut, pA_L, pA_R = unpack(cell_output_table)
      input_to_topology = {hOut,pA_L,pA_R}
    elseif self.model == 'dtp' and not self.lr_tree then
      hA, hF, hOut, pA, pF = unpack(cell_output_table)
      input_to_topology = {hOut,pA,pF}
    elseif self.model == 'dtp' and self.lr_tree then
      hA, hF, hOut, pA_L, pA_R, pF = unpack(cell_output_table)
      input_to_topology = {hOut,pA_L,pA_R,pF}
    else
      print("ERROR in DRNN: unrecognized model. Aborting")
      os.exit()
    end

    --print(hA_prev:nDimension(), hF_prev:nDimension(), hA:nDimension(),hF:nDimension())

    assert(hA:size(1) == self.mem_dim and hF:size(1) == self.mem_dim, "Wrong hidden size" .. hA:size(1) .. ', '.. hF:size(1))
    -- Every model has these:
    self.stateAncest[node.index]   = hA
    self.stateFrat[node.index]     = hF

    local pW, indA, indF = unpack(self.topologyModule:forward(input_to_topology))

    --pW,indA,indF = unpack(self.topologyModule:forward({hOut,pA,pF})) -- TODO: Pass neagitves if exceeded max's here?

    --debugutils.dprint(2,pW:norm(),pA[1],pF and pF[1] or nil,indA,indF or nil)

    --print(self.userPrevAncestral:norm())
    --print(inA:norm(),inputs[1]:norm(), inputs[2]:norm())

    -- Sample word
    local word_id, word_logprob = self.sampler:tree_sample_node(pW)

    -- Need to get vector with deep copy of embedder output, otherwise in next
    -- call of embedder this input will change!!
    self.embedder:forward(word_id)
    local word_vec = torch.Tensor(self.embedder.output:size(2)):copy(self.embedder.output)

    --
    -- print(inA:norm(),inputs[1]:norm(), inputs[2]:norm())
    --
    -- local word_vec = torch.Tensor():resizeAs(inA)
    -- word_vec:copy(self.embedder:forward(word_id):select(1,1))

    --local word_id = torch.Tensor(1):fill(202)
    --local word_vec = torch.Tensor(300):copy(self.embedder:forward(word_id):select(1,1))


    --print(inA:norm(),inputs[1]:norm(), inputs[2]:norm())

    --print(hA:size())

    --debugutils.dprintf(1,'%5i %5i %5i %5i %5i %5s %5i %5i\n',

    -- debugutils.debug_DRNN_forward_predict(1,
    -- self.model,self.lr_tree,total_steps,self.tree,to_process,
    -- node,prev_frat_node,inA,hA_prev,inF,hF_prev,hA,hF,hOut,pA,pA_L,pA_R,pF,pW,indA,indF,word_vec)

    -- Debug
    string_chronol_words = string_chronol_words .. ' ' .. tostring(word_id[1])


    -- Check whether we should stop production in either direction
    if not self.lr_tree then
      stop_fertility  = (indA == 0)
                     or (self.model ~= 'dtp' and word_id[1] == self.end_token_ind) -- EOS -> No children
                     or (node.depth > MAX_TREE_DEPTH - 1)
    else
      stop_fertility['L']  = (indA['L'] == 0)
                     or (self.model ~= 'dtp' and word_id[1] == self.end_token_ind) -- EOS -> No children
                     or (node.depth > MAX_TREE_DEPTH - 1)
      stop_fertility['R']  = (indA['R'] == 0)
                     or (self.model ~= 'dtp' and word_id[1] == self.end_token_ind) -- EOS -> No children
                     or (node.depth > MAX_TREE_DEPTH - 1)
    end

    stop_fraternity = (node.index == 2)                    -- Root -> No siblings
                   or (self.model == 'dtp' and indF == 0)
                   or (self.model ~= 'dtp' and word_id[1] == self.end_token_ind)
                   or (child_no > MAX_OFFSPRING)           -- Max width exceeded


    -- Save the sampled token and its prob
    assert(wordIds[node.idx] == nil, "Trying to overwrite: " .. node.idx)
    inputs[node.idx] = word_vec
    wordIds[node.idx] = word_id[1]
    iF = word_vec

    -- Save all probabilities involved in node sampling: topo and content
    if self.model == 'ntp' then
      logprobs[node.idx] = {word_logprob}
    elseif self.model == 'stp' and not self.lr_tree then
      logprobs[node.idx] = {word_logprob, torch.log(pA)}
    elseif self.model == 'stp' and self.lr_tree then
      logprobs[node.idx] = {word_logprob, torch.log(pA_L), torch.log(pA_R)}
    elseif self.model == 'dtp' and not self.lr_tree then
      logprobs[node.idx] = {word_logprob, torch.log(pA), torch.log(pF)}
    elseif self.model == 'dtp' and self.lr_tree then
      logprobs[node.idx] = {word_logprob, torch.log(pA_L), torch.log(pA_R), torch.log(pF)}
    else
      print("ERROR in DRNN: unrecognized model. Aborting")
      os.exit()
    end


    --next_idx = next_idx + 1

    if (not self.lr_tree) and (not stop_fertility) then
      local c,pad
      pad = tree2tree.Tree({idx=2, index=next_index,bdy='SOS', depth=node.depth + 1})
      next_index = next_index + 1
      node:add_child(pad,'start')
      c = tree2tree.Tree({index=next_index, idx=next_idx,depth=node.depth + 1})
      node:add_child(c,'end')
      next_index = next_index + 1
      next_idx = next_idx + 1
      table.insert(to_process,1,c) -- pushes to queue
    elseif self.lr_tree then
      if (not stop_fertility['L']) then
        local c, pad
        pad = tree2tree.LRTree({index=next_index,idx=2,bdy='SOS', depth=node.depth + 1})
        next_index = next_index + 1
        node:add_left_child(pad,'start')
        c = tree2tree.LRTree({index=next_index,idx=next_idx,depth=node.depth + 1})
        node:add_left_child(c,'end')
        next_index = next_index + 1
        next_idx = next_idx + 1
        table.insert(to_process,1,c) -- pushes to queue
      end
      if (not stop_fertility['R']) then
        local c, pad
        pad = tree2tree.LRTree({index=next_index, idx=2, bdy='SOS', depth=node.depth + 1})
        next_index = next_index + 1
        node:add_right_child(pad,'start')
        c = tree2tree.LRTree({index=next_index, idx=next_idx, depth=node.depth + 1})
        node:add_right_child(c, 'end')
        next_index = next_index + 1
        next_idx = next_idx + 1
        table.insert(to_process,1,c) -- pushes to queue
      end
    end -- end if fertility

    if (not stop_fraternity) then
      debugutils.dprint(2,"Adding brother")
      -- add brother node
      local b = (self.lr_tree) and tree2tree.LRTree({index=next_index, idx = next_idx,depth=node.depth})
                                or tree2tree.Tree({index=next_index,idx = next_idx,depth=node.depth})
      if node.side == 'L' then node.parent:add_left_child(b,'end')
      elseif node.side == 'R' then node.parent:add_right_child(b, 'end')
      else node.parent:add_child(b,'end') end
      -- -- Check
      --
      -- assert(node.left_brother == b, "error")

      next_index = next_index + 1
      next_idx   = next_idx + 1
      table.insert(to_process,b) -- pushes to front of queue
    end

    debugutils.debug_DRNN_forward_predict(Debug,
        self.model,self.lr_tree,total_steps,self.tree,to_process,
        node,prev_frat_node,inA,hA_prev,inF,hF_prev,hA,hF,hOut,pA,pA_L,pA_R,pF,pW,indA,indF,word_vec,word_id[1])
  end -- end while



  --print(string_chronol_words .. '\n') -- NOTE: This was on. Add debug flag?

  return wordIds, logprobs
end



function DRNN:updateGradInput(input, gradOutput)
  --local tree, sent = input[1], input[2]
  assert(self.tree ~= nil, "expecting a teaching tree")
  assert(self.fwd_traversal ~= nil, "no previous forward traversal found")
  assert((#self.gradInput == 0) and (#self.gradFrat == 0) and (#self.gradAncest == 0),
  "gradInput should be empty when entering updateGradInput")
  debugutils.dprintf(1,'~ Entering DRNN:updateGradInput.\n\t- Input type/size: %s/%i\
  \t- gradOutput type/size: %s/%i\n',torch.type(input),#input,torch.type(gradOutput),#gradOutput)

  --debugutils.print_tensor_info(0,gradOutput)

  -- local max = 0
  -- for k,u in pairs(gradOutput) do
  --   for l,v in pairs(u) do
  --     --print(v:size())
  --     local norm = v:norm()
  --     max = (norm > max) and norm or max
  --   end
  -- end
  -- print(max)

  -- Traversal will depend on type of tree, should already remove padding and root, etc.
  local node_traversal = self:getTraversal('backward')

  debugutils.debugtraversal(2,node_traversal)

  for i, node_step in ipairs(node_traversal) do
    debugutils.dprintf(2,' %i ',i,node_step[2],node_step[3],node_step[4])
    local node, prev_frat_node, next_frat_node, side = unpack(node_step) -- side will be nil for non LR tree
    local inA, hA_prev, inF, hF_prev = self:get_node_inputs(input,node, prev_frat_node, side)
    local grad_hA_children, grad_hF_next = self:get_node_gradients(node,next_frat_node)
    local inputTable  = {inA, hA_prev, inF, hF_prev}
    local gradOutputTable = TableConcat({grad_hA_children, grad_hF_next},gradOutput[node.index])


    debugutils.norm_halt('Bwd:NodeInputs',{inA, hA_prev, inF, hF_prev})
    debugutils.norm_halt('Bwd:NodeGrads',gradOutputTable)

    if (self.model == 'dtp' and node.bdy ~= 'EOS' and next_frat_node == nil
     and ((self.lr_tree and node.num_lchildren + node.num_rchildren ==0 ) or
      (not self.lr_tree and node.num_children ==0))) then
      -- If using dtp model without EOS padding, the embedding of this node's input word will not
      -- get a gradient update, because there's nothing downstream of it.
      -- So we patch this by pre-emptively assining a zero gradient to its input
      -- There might be a more correct way of dealing with this.
      self.gradInput[node.idx] = self.in_zeros
    end

    assert((self.lr_tree and self.sharedClones[side][node.index]) or
    self.sharedClones[node.index], "oops. Shared clone for bwd not found")

    local recurrentModule = (side) and self:getStepModule(node.index,side) or self:getStepModule(node.index)

    local grad_inputs = recurrentModule:backward(inputTable,gradOutputTable)
    local grad_inA, grad_hA, grad_inF, grad_hF = unpack(grad_inputs)

    debugutils.debug_DRNN_backward(1,self.model,self.lr_tree,
    #node_traversal,i,node,next_frat_node,inputTable, gradOutputTable,grad_inputs)

    self.gradAncest[node.index] = grad_hA
    self.gradFrat[node.index]   = grad_hF
    -- In order to decide where to send grad of inputs, need to know sibling and parent's

    if node.index ~= 2 then -- 2 corresponds to root node, which has no siblings -> no input from brother
      --print('Here',node.index,prev_frat_node.idx)
      self.gradInput[prev_frat_node.idx] = ((prev_frat_node ~= nil) and
            self.gradInput[prev_frat_node.idx] or self.in_zeros) + grad_inF
    end
    -- but everybody has parent:
    self.gradInput[node.parent.idx] =
            (self.gradInput[node.parent.idx] or self.in_zeros) + grad_inA


    local this_clone = self.lr_tree and self.sharedClones[side][node.index]or
    self.sharedClones[node.index]
    local params, gparams =this_clone:parameters()
    local halt_string = (self.lr_tree) and ('Side: ' .. side .. ', index: '.. node.index) or  'Index: '.. node.index
    --debugutils.norm_halt('Bwd:gradParameters ' .. halt_string,gparams)
    debugutils.norm_halt('Bwd:gradInput. Side: ' .. halt_string,self.gradInput)

  end -- traversal
  self.userGradPrevAncestral = self.gradAncest[2] -- The root nodes

  --debugutils.print_tensor_info(1,self.gradInput)
  debugutils.check_zeros{self.in_zeros, self.mem_zeros}

  --print(self.gradInput[1]:size())
  assert(#self.gradInput == self._input_size, "Error: input and gradInput of \
  different sizes: " .. self._input_size .."~="..  #self.gradInput)
  debugutils.dprintf(1,
  "~ Leaving DRNN:updateGradInput. %5i elems in gradInput of size %i, norm of root node: %.2f\n",
  #self.gradInput,self.gradInput[1]:size(1),self.gradInput[2]:norm())
  --print(self.gradInput)
  return self.gradInput
end


function DRNN:accGradParameters(input, gradOutput, scale)
  assert(self.tree ~= nil, "expecting a teaching tree")
  local node_traversal = self:getTraversal('backward') -- wil depend on type of tree, should already remove padding and root, etc.
  debugutils.dprint(2,'Hello DRNN:accGradParameters')
  for i, node_step in ipairs(node_traversal) do
    local node, prev_frat_node, next_frat_node, side = unpack(node_step) -- side will be nil for non LR tree
    local inA, hA_prev, inF, hF_prev = self:get_node_inputs(input,node, prev_frat_node, side)
    local grad_hA_children, grad_hF_next = self:get_node_gradients(node,next_frat_node)
    local inputTable  = {inA, hA_prev, inF, hF_prev}
    local gradOutputTable = {gradOutput[node.index], grad_hA_children, grad_hF_next}
    local recurrentModule = (side) and self:getStepModule(node.index,side) or self:getStepModule(node.index)
    recurrentModule:accGradParameters(inputTable,gradOutputTable, scale)
  end
  debugutils.dprint(2,'Bye bye from DRNN:accGradParameters')
  -- self:clean()
  -- self:clearState()
end


function DRNN:share(gru, ...)
  if self.in_dim ~= gru.in_dim then error("DRNN input dimension mismatch") end
  if self.mem_dim ~= gru.mem_dim then error("DRNN memory dimension mismatch") end
  if self.num_layers ~= gru.num_layers then error("DRNN layer count mismatch") end
  if self.gate_output ~= gru.gate_output then error("DRNN output gating mismatch") end
  share_params(self.master_cell, gru.master_cell, ...)
end

-- function DRNN:zeroGradParameters()
--   self.master_cell:zeroGradParameters()
-- end

-- function DRNN:parameters()
-- -- TODO: add fertility module parameters here.
--   return self.master_cell:parameters()
-- end



function DRNN:getTraversal(direction)
  -- should porbably only do this once at the beginning and save
  --local direction = direction or 'forward'
  assert(self.tree  ~= nil, "Error: no learning tree")
  assert(direction == 'forward' or direction == 'backward', "Wrong direction")
  local all_nodes, k
  local nodes = {}
  if (direction == 'forward') then
      nodes,k = self.tree:getTraversal()
      self.fwd_traversal = nodes
    --else
    --   nodes = self.fwd_traversal
    --   k = #self.fwd_traversal
    -- end
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

-- Convenience function to get poitners to the fraternal/ancestral states and inputs
-- Haven;t decided yet how to pass parent, brother. Indices or whole node? In table?
function DRNN:get_node_inputs(inputs,node, prev_frat, side)
  --local prev_idx, prev_index     = (prev_frat) and prev_frat.idx, prev_frat.index or nil,nil
  --print('Hello from get node')
  local parent_idx, parent_index = node.parent.idx, node.parent.index
  -- Get inputs from external inputs with idx
  local inF   = (prev_frat) and inputs[prev_frat.idx] or self.in_zeros   --TODO: I dont like this.
  local inA = inputs[parent_idx]

  -- ... and get states with index from interal containers
  local hA_prev

  if (node.index == 2) then -- this is the root
  -- FIXME: Maybe change to if (node.parent.bdy == 'SOT') then
    hA_prev = self.userPrevAncestral or self.mem_zeros
    --print(hA_prev:norm());
  else
    hA_prev = self.stateAncest[parent_index]
  end
  --print(node.index)
  assert(hA_prev:norm() > 0, "Warning: Decoder stated without initial ancestral state")

  local hF_prev = (prev_frat) and self.stateFrat[prev_frat.index] or torch.zeros(self.mem_dim)

  --assert(hF_prev:norm() > 0 or (prev_frat == nil), "Zero hF in non-boundary node: "..node.index)
  -- Had this before. SHouldnt be needed if getTraversal does it job correctly:
  -- (prev_fraternal and prev_fraternal.idx ~= 2)
  -- TODO: Check that this is ok. If not, can query (isboundary).
  return inA, hA_prev, inF, hF_prev
end

function DRNN:get_node_gradients(node, next_frat)
  -- Get gradient w.r.t fraternal hidden states.
  local grad_hF = (next_frat) and self.gradFrat[next_frat.index]
                                            or torch.zeros(self.mem_dim)
  local grad_hA = self.mem_zeros
  -- local all_children = (not self.lr_tree) and node.children or
  if (self.lr_tree and (node.num_lchildren + node.num_rchildren > 0)) then
    for t = 1, node.num_rchildren do -- We skip leftmost child, has no grads.
      if (self.gradAncest[node.rchildren[t].index]) then -- this should only skip SOS nodes. Check.
        grad_hA = grad_hA + self.gradAncest[node.rchildren[t].index]
      end
    end
    for t = 1, node.num_lchildren do -- We skip leftmost child, has no grads.
      if (self.gradAncest[node.lchildren[t].index]) then -- this should only skip SOS nodes. Check.
        grad_hA = grad_hA + self.gradAncest[node.lchildren[t].index]
      end
    end
    grad_hA:div(node.num_lchildren + node.num_rchildren) -- Average over n children. Not sure if this helps.
  elseif (not self.lr_tree and node.num_children > 0) then -- Non-lr Tree
    for t = 1, node.num_children do -- We skip leftmost child, has no grads.
      if (self.gradAncest[node.children[t].index]) then -- this should only skip SOS nodes. Check.
        grad_hA = grad_hA + self.gradAncest[node.children[t].index]
      end
    end
    grad_hA:div(node.num_children) -- Average over n children. Not sure if this helps.
  end
  return grad_hA, grad_hF
end


-- Overrides abstract tree's version, to account for possibly L/R tree
function DRNN:getStepModule(step, side)
  local _ = require 'moses'
  --local n,ta = get_keys(self.sharedClones['L'])
  assert(step, "expecting step at arg 1")
  assert(side or not self.lr_tree, "Cant ask for side module in non lr tree")
  debugutils.dprint(3,'Getting step module', step, side)
  local recurrentModule  = (self.lr_tree) and self.sharedClones[side][step] or self.sharedClones[step]
  if not recurrentModule then
    debugutils.dprint(3,'Step module not found, cloning')
    if not self.lr_tree then
      recurrentModule = self.recurrentModule:stepClone()
      self.sharedClones[step] = recurrentModule
      self.nSharedClone = _.size(self.sharedClones)
    else
      assert(self.sharedClones[2] == nil, 'Clones assigned to wrong level of sharedClones before')
      recurrentModule = self.recurrentModule[side]:stepClone()
      self.sharedClones[side][step] = recurrentModule
      self.nSharedClone = _.size(self.sharedClones['L']) + _.size(self.sharedClones['R'])
      assert(self.sharedClones[2] == nil, 'Clones assigned to wrong level of sharedClones')
    end
  end
  return recurrentModule
end

function DRNN:includingSharedClones(f)
  local r
  local modules = self.modules
  self.modules = {}

  --print(self.sharedClones)
  --  print(torch.type(sharedClones))
  --  for k,v in pairs(sharedClones) do print(k) end
  if (self.lr_tree) then
    local sharedClones = {}
    sharedClones['L'] = self.sharedClones['L']
    sharedClones['R'] = self.sharedClones['R']
    self.sharedClones['L'] = nil
    self.sharedClones['R'] = nil
    for i,modules in ipairs({modules, sharedClones['L'],sharedClones['R']}) do
      for j, module in pairs(modules or {}) do
    --print(torch.type(module))
        table.insert(self.modules, module)
      end
    end
    r = {f()}
    self.sharedClones['L'] = sharedClones['L']
    self.sharedClones['R'] = sharedClones['R']
  else
    local sharedClones = self.sharedClones
    self.sharedClones = nil
    for i,modules in ipairs{modules, sharedClones} do
      for j, module in pairs(modules or {}) do
        table.insert(self.modules, module)
      end
    end
    r = {f()}
    self.sharedClones = sharedClones
  end
  self.modules = modules
  -- print(torch.type(self.sharedClones))
  -- for k,v in pairs(self.sharedClones) do print(k) end
  return unpack(r)
end

function DRNN:predicting()
  --  return self:includingSharedClones(function()
  --     return parent.predicting(self)
  --  end)
  self.train = false
  self.predict = true
  self:forget()
end

function DRNN:training()
  self.train   = true
  self.predict = false
  -- for i=1,#self.modules do
  --   self.modules[i]:training()
  -- end
  self:forget()
   return self:includingSharedClones(function()
      return parent.training(self)
   end)
end

function DRNN:evaluate()
  self.train   = false
  self.predict = false
  -- for i=1,#self.modules do
  --   self.modules[i]:evaluate()
  -- end
  self:forget()
   return self:includingSharedClones(function()
      return parent.evaluate(self)
   end)
end

function DRNN:zeroGradParameters()
  --print('Here')
  local _,gradParams = self:parameters()
  --print(#gradParams)
  if gradParams then
    for i=1,#gradParams do
      gradParams[i]:zero()
      --print(gradParams[i]:size(1))
    end
  end
  for i=1,#self.modules do   -- Will be either just one module (Tree) or 2 (LRTree)
    self.modules[i]:zeroGradParameters()
  end
  self.gradInput         = {}
  --bk()
end

function DRNN:clean()
  --self:free_module(tree, 'composer')
  --self:free_module(tree, 'output_module')
  --self.gradAncest       = {}
  --self.gradFrat         = {}
  --self.gradInput         = {}
  --self.stateFrat        = {}
  --self.stateAncest      = {}
  self.fwd_traversal = nil  -- In case they leaked from previous examples
  self.bwd_traversal = nil
  nn.utils.clear(self,'userGradPrevAncestral','stateFrat','stateAncest','logprob_ancest',
  'logprob_frat','gradFrat','gradAncest','output','_output','gradInput')

  --
  -- tree.output = nil
  -- for i = 1, tree.num_children do
  --   self:clean(tree.children[i])
  -- end
end

-- Full cleanign
function DRNN:clearState()
  --self.gradAncest       = {}
  --self.gradFrat         = {}
  --self.gradInput         = {}
  --self.stateFrat        = {}
  --self.stateAncest      = {}
  nn.utils.clear(self,'userGradPrevAncestral','stateFrat','stateAncest','logprob_ancest',
  'logprob_frat','gradFrat','gradAncest','output','_output','gradInput')
  --nn.utils.clear(self.sharedClones,{'L','R'})
  self.sharedClones['L'] = {}
  self.sharedClones['R'] = {}
  --print(self.sharedClones['L'])
  -- nn.utils.clear(self,'userGradPrevAncestral','stateFrat','stateAncest','logprob_ancest',
  -- 'logprob_frat','gradFrat','gradAncest','output','_output','gradInput','sharedClones.L')
  assert(next(self.sharedClones['L']) == nil, "Shared clones not cleared properly")
  assert(next(self.sharedClones['R']) == nil, "Shared clones not cleared properly")
  self.nSharedClone = 0
  self.fwd_traversal = nil  -- In case they leaked from previous examples
  self.bwd_traversal = nil
  self.modules[1]:clearState()
  self.modules[2]:clearState()
end

-- Clear saved gradients and states. Less agressive that clearState(): doesnt
-- eliminate the sharedClones
function DRNN:forget()
  -- self.depth = 0
  -- for k, gtable in pairs{self.gradInput, self.gradAncest, self.gradFrat} do
  --   for i, g in pairs(gtable) do
  --     local gradInput = g
  --     if type(gradInput) == 'table' then
  --       for _, t in pairs(gradInput) do t:zero() end
  --     else
  --       gradInput:zero()
  --     end
  --   end
  -- end
  self.fwd_traversal = nil  -- In case they leaked from previous examples
  self.bwd_traversal = nil
  nn.utils.clear(self,
  'stateFrat','stateAncest','logprob_ancest','logprob_frat',
  'gradFrat','gradAncest','userGradPrevAncestral','gradInput','outputTree')
  -- 'userPrevAncestral' not this one!!! We call forget at the beginning of each forward
  -- so we don't want to get rid of the state pass by encoder.

end
