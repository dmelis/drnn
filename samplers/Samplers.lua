
--[[
  Modes:
      - Argmax "sampling" at each node
      - Node-wise sampling
]]--

local Sampler, parent = torch.class('tree2tree.Sampler', 'nn.Module')

function Sampler:__init(config)
  parent.__init(self)
  self.sample_mode     = config.sample_mode or 'max'
  self.temperature     = config.temperature or 1
  self.topology        = config.topology
  self.max_sent_length = config.max_sent_length or 15
end

-- MAX_SEQUENCE_LENGTH = 15


function Sampler:tree_sample_node(pW) -- Pass states?
  -- This should in theory also work in batchmode as is. But would need to force non batch to batch form.
  -- t=1,self.seq_length+2 do
  --print(torch.type(pW))
  --print_tensor_info(pW)
  local sampleLogprobs, wt
  if self.sample_mode == 'max' then -- use argmax "sampling"
    sampleLogprobs, wt = torch.max(pW, 1)  -- TODO: Check that what enters this are logprobs. Also, change this 1 to 2 fo rbatch processing.
    wt = wt:view(-1):long()
  elseif self.sample_mode == 'temp' then
    -- sample from the distribution of previous predictions
    local prob_prev
    if self.temperature == 1.0 then
      prob_prev = torch.exp(pW) -- fetch prev distribution: shape Nx(M+1)
    else
      -- scale logprobs by temperature
      prob_prev = torch.exp(torch.div(pW, self.temperature))
    end
    wt = torch.multinomial(prob_prev, 1)
    sampleLogprobs = pW:gather(1, wt) -- gather the logprobs at sampled positions
    wt = wt:view(-1):long() -- and flatten indices for downstream processing
  end
  -- Save sampled value
  --inputs[node.index] = it
  -- and probs for  bookkeeping
  --logProbs[node.index] =  sampleLogprobs:view(-1):float()
  return wt, sampleLogprobs
end


function Sampler:sequence_sample_node(pW) -- Pass states?
  -- This should in theory also work in batchmode as is. But would need to force non batch to batch form.
  -- t=1,self.seq_length+2 do
  --print(torch.type(pW))
  --print_tensor_info(pW)
  local sampleLogprobs, wt
  if self.sample_mode == 'max' then -- use argmax "sampling"
    sampleLogprobs, wt = torch.max(pW, 1)  -- TODO: Check that what enters this are logprobs. Also, change this 1 to 2 fo rbatch processing.
    wt = wt:view(-1):long()
  elseif self.sample_mode == 'temp' then
    -- sample from the distribution of previous predictions
    local prob_prev
    if self.temperature == 1.0 then
      prob_prev = torch.exp(pW) -- fetch prev distribution: shape Nx(M+1)
    else
      -- scale logprobs by temperature
      prob_prev = torch.exp(torch.div(pW, self.temperature))
    end
    wt = torch.multinomial(prob_prev, 1)
    sampleLogprobs = pW:gather(1, wt) -- gather the logprobs at sampled positions
    wt = wt:view(-1):long() -- and flatten indices for downstream processing
  end
  -- Save sampled value
  --inputs[node.index] = it
  -- and probs for  bookkeeping
  --logProbs[node.index] =  sampleLogprobs:view(-1):float()
  return wt, sampleLogprobs
end

function Sampler:sequence_sampler(embedder,sequencer,predicter,priming,stop_token_index) -- Pass states?
  local sequence = {}
  local wt = embedder:forward(priming)[1]
  local stop = false
  while not stop do
    local h                = sequencer:forward(wt)
    local prob             = predicter:forward(h)
    local it               = self:sequence_sample_node(prob)
    sequence[#sequence + 1]=it[1]
    if (it[1] == stop_token_index) or (#sequence == self.max_sent_length) then
      stop = true
    else
      wt = embedder:forward(it)[1]
    end
  end
  return sequence
end



--
--
-- function TreeSampler:sample_node2(node, inputs, probs) -- Pass states?
--   -- t=1,self.seq_length+2 do
--   local xt, it, sampleLogprobs
--   local iA, hA, iF, hF = self:get_node_inputs(inputs,node,side) -- takes care of initial, boundary, etc
--   hA, hF, hOut, pA, pF = unpack(self.DRNN_cell:forward{iA, hA, iF, hF})
--   pW,indA,indF = self.topo_module:forward(hOut,pA,pF)
--
--   if self.sample_mode == 'max' then -- use argmax "sampling"
--     sampleLogprobs, wt = torch.max(pW, 2)  -- TODO: Check that what enters this are logprobs
--     wt = wt:view(-1):long()
--   elseif self.sample_mode == 'temp' then
--     -- sample from the distribution of previous predictions
--     local prob_prev
--     if temperature == 1.0 then
--       prob_prev = torch.exp(pW) -- fetch prev distribution: shape Nx(M+1)
--     else
--       -- scale logprobs by temperature
--       prob_prev = torch.exp(torch.div(pW, temperature))
--     end
--     it = torch.multinomial(prob_prev, 1)
--     sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
--     it = it:view(-1):long() -- and flatten indices for downstream processing
--   end
--   -- Save sampled value
--   inputs[node.index] = it
--   -- and probs for  bookkeeping
--   logProbs[node.index] =  sampleLogprobs:view(-1):float()
--   return it, sampleLogprobs
-- end
--
--
--
--
--
--
-- function layer:sample(initial_state, opt)
--   local sample_max = utils.getopt(opt, 'sample_max', 1)
--   local beam_size = utils.getopt(opt, 'beam_size', 1)
--   local temperature = utils.getopt(opt, 'temperature', 1.0)
--   if sample_max == 1 and beam_size > 1 then return self:sample_beam(imgs, opt) end -- indirection for beam search
--
--   -- local batch_size = imgs:size(1)
--   -- self:_createInitState(batch_size)
--   -- local state = self.init_state
--
--   -- we will write output predictions into tensor seq
--   local seq = torch.LongTensor(self.seq_length, batch_size):zero()
--   local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
--   local logprobs -- logprobs predicted in last time step
--
--
--
--
--
--
--   -- return the tree inputs and its proabilities
--   return seq, seqLogprobs
-- end
--
--
--
--     if node.index == 1 then
--       -- feed in the images
--       xt = imgs
--     elseif t == 2 then
--       -- feed in the start tokens
--       it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
--       xt = self.lookup_table:forward(it)
--     else
--       -- take predictions from previous time step and feed them in
--       if sample_max == 1 then
--
--
--       else
--         -- sample from the distribution of previous predictions
--         local prob_prev
--         if temperature == 1.0 then
--           prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
--         else
--           -- scale logprobs by temperature
--           prob_prev = torch.exp(torch.div(logprobs, temperature))
--         end
--         it = torch.multinomial(prob_prev, 1)
--         sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
--         it = it:view(-1):long() -- and flatten indices for downstream processing
--       end
--       xt = self.lookup_table:forward(it)
--     end
--
--     if t >= 3 then
--       seq[t-2] = it -- record the samples
--       seqLogprobs[t-2] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
--     end
--
--     local inputs = {xt,unpack(state)}
--     local out = self.core:forward(inputs)
--     logprobs = out[self.num_state+1] -- last element is the output vector
--     state = {}
--     for i=1,self.num_state do table.insert(state, out[i]) end
--   end
--
--   -- return the samples and their log likelihoods
--   return seq, seqLogprobs
-- end
--
-- function getKBest(tree, order, max_width, max_depth, max_size)
--
--   local rooted_tree = tree2tree:Tree() -- deal only with subtree, don't care about parents.
--   words = getKBest(tree,0) -- gets just k best words for this node.
--
--
--
--
--
--
--
-- end
