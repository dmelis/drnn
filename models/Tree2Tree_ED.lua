--[[

  Tree to tree encoder/decoder wrapper.
  Processes arguments, instantiaties encoder and decoder, loss objective
  "Stitches" encoder and decoder.
  Provides training, testing modules.
  Should be called from main.

--]]

local Tree2Tree_ED = torch.class('tree2tree.Tree2Tree_ED')

function Tree2Tree_ED:__init(config)
  self.debug             = config.debug
  self.mem_dim           = config.mem_dim or 150
  self.mem_zeros         = torch.zeros(self.mem_dim)
  self.learning_rate     = config.learning_rate or 0.06
  self.emb_learning_rate = config.emb_learning_rate or 0.05   --TODO: Add funcitonality to freeze embedding weights
  self.batch_size        = config.batch_size or 25
  self.reg               = config.reg           or 1e-4
  self.grad_clip         = config.grad_clip or 0
  self.sim_nhidden       = config.sim_nhidden or 150
  self.lr_tree           = config.lr_tree or false
  self.pad_eos           = config.pad_eos
  -- self.source_vocab_dim  = config.emb_vecs.source:size(1)
  -- self.target_vocab_dim  = config.emb_vecs.target:size(1)
  -- self.source_vocab      = config.source_vocab
  -- self.target_vocab      = config.target_vocab
  -- self.source_emb_dim    = config.emb_vecs.source:size(2)
  -- self.target_emb_dim    = config.emb_vecs.target:size(2)
  self.source_vocab      = config.source_vocab or 1000
  self.target_vocab      = config.target_vocab or 1000
  self.source_vocab_dim  = config.vocab_dim and config.vocab_dim['source']
  self.target_vocab_dim  = config.vocab_dim and config.vocab_dim['target']
  self.source_emb_dim    = config.source_emb_dim or 150
  self.target_emb_dim    = config.target_emb_dim or 150


  self.zeroTensor        = torch.Tensor(2):zero()

  -- If vectors provided, override previous vocab dim and emv dim
  if config.emb_vecs then
    self.source_vocab_dim = config.emb_vecs.source:size(1)
    self.target_vocab_dim = config.emb_vecs.target:size(1)
    self.source_emb_dim   = config.emb_vecs.source:size(2)
    self.target_emb_dim   = config.emb_vecs.target:size(2)
  end

  -- Parse our composite args.
  -- Encoder.
  for i,v in ipairs(string.split(string.lower(config.encoder_type),",")) do
    if string.lower(v) == 'tree' then self.encoder_topology    = 'tree' end
    if string.lower(v) == 'lstm' then self.encoder_recurrence  = 'lstm' end
    if string.lower(v) == 'gru' then self.encoder_recurrence  = 'gru' end
    if string.lower(v) == 'seqlstm' then self.encoder_recurrence  = 'seqlstm' end
    if string.lower(v) == 'seqgru' then self.encoder_recurrence  = 'seqgru' end
    if string.lower(v) == 'bilstm'then self.encoder_recurrence = 'bilstm' end
    if string.lower(v) == 'bigru' then self.encoder_recurrence = 'bigru' end
    if string.find(v,'cs') then  self.encoder_model = 'ChildSum' end
  end

  -- Decoder
  for i,v in ipairs(string.split(string.lower(config.decoder_type),",")) do
    if string.lower(v) == 'tree' then self.decoder_topology   = 'tree' end
    if string.lower(v) == 'lstm' then self.decoder_recurrence = 'lstm' end
    if string.match(v,'tp') then self.decoder_model = v end
  end

  -- Set defaults
  self.encoder_topology      = self.encoder_topology or 'seq'
  self.encoder_recurrence    = self.encoder_recurrence or 'gru'
  self.decoder_topology      = self.decoder_topology or 'seq'
  self.decoder_recurrence    = self.decoder_recurrence or 'gru'
  --self.decoder_model         = self.decoder_model or 'seq'

  assert(self.encoder_topology and self.encoder_recurrence, "Missing encoder info")
  assert(self.decoder_topology and self.decoder_recurrence, "Missing decoder info")

  -- initialize encoder and decoder
  local encoder_config = {
    in_dim             = self.source_emb_dim,
    mem_dim            = self.mem_dim,
    gate_output        = false,
    vocab_dim          = self.source_vocab_dim,
    lr_tree            = self.lr_tree,
    recurrence         = self.encoder_recurrence,
    model              = self.encoder_model
  }

  local decoder_config = {
    in_dim             = self.target_emb_dim,
    mem_dim            = self.mem_dim,
    gate_output        = true,
    vocab_dim          = self.target_vocab_dim,
    lr_tree            = self.lr_tree,
    recurrence         = self.decoder_recurrence,
    model              = self.decoder_model
  }

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- Instantiate Encoder
  local encoder_name
  if self.encoder_topology == 'seq' then
    local er = self.encoder_recurrence
    local core =        (er == 'lstm')   and nn.LSTM(self.source_emb_dim, self.mem_dim)
                    or  (er == 'gru')    and nn.GRU(self.source_emb_dim, self.mem_dim)
                    or  (er == 'seqlstm')and nn.SeqLSTM(self.source_emb_dim, self.mem_dim)
                    or  (er == 'seqgru') and nn.SeqGRU(self.source_emb_dim, self.mem_dim)
    self.encoder = self:new_encoder_sequence_module(core,config.emb_vecs and config.emb_vecs.source)
    encoder_name = 'seq' .. '-' .. er
  else -- tree encoder
    if self.encoder_model == 'ChildSum' then
      encoder_name = 'CS-TREE-LSTM'
      self.encoder = tree2tree.tree_encoder(encoder_config,config.emb_vecs and config.emb_vecs.source)
    else
      print("unrecognized encoder model")
      os.exit()
    end
  end

  -- Instantiate Decoder
  local decoder_name
  if self.decoder_topology == 'seq' then
    local dt = self.encoder_recurrence
    local core = (dt == 'lstm')   and nn.SeqLSTM(self.target_emb_dim, self.mem_dim)
                    or  (dt == 'gru')    and nn.SeqGRU(self.target_emb_dim, self.mem_dim)
    self.criterion_type = 'seq'
    self.decoder = self:new_decoder_sequence_module(core,config.emb_vecs and config.emb_vecs.target)
    decoder_name = 'seq' .. '-' .. dt
  else
    local core = tree2tree.DRNN(decoder_config)
    self.criterion_type = 'tree'
    self.decoder = self:new_decoder_tree_module(core,config.emb_vecs and config.emb_vecs.target)
    self.decoder.lr_tree = self.lr_tree
    decoder_name = 'tree' .. '-' .. self.decoder_recurrence .. '-' .. self.decoder_model
  end

  -- Instantiate Criterion
  self.criterion = (self.decoder_topology == 'seq') and
            nn.SequencerCriterion(nn.ClassNLLCriterion()) or
            tree2tree.TreeCriterion(nn.ClassNLLCriterion())

  local model_name = encoder_name .. ':'.. decoder_name

  -- Get function an gradient parameters of encoder and decoder.
  -- TODO: Move this as a getParameters function of an encoder-decoder class?
  -- self.params, self.grad_params = nil, nil
  -- Currently, I only use these to print the number of parameters in encoder and decoder.
  -- I don't use these handles to modify anything.
  self.enc_params, self.enc_grad_params = self.encoder:getParameters()
  self.dec_params, self.dec_grad_params = self.decoder:getParameters()

  consistent_module = nn.Sequential():add(self.encoder):add(self.decoder)
  self.params, self.grad_params = consistent_module:getParameters()
  self.config = config -- For easy saving in :save
end


function Tree2Tree_ED:new_encoder_sequence_module(core,vecs)
  local sequence_module = nn.Sequential()
  self.source_emb_module = nn.LookupTable(self.source_vocab_dim, self.source_emb_dim)
  if vecs then
    self.source_emb_module.weight:copy(vecs)
  end
  sequence_module:add(self.source_emb_module)
  sequence_module.core = core
  sequence_module:add(nn.SplitTable(1, 2))
  sequence_module:add(nn.Sequencer(sequence_module.core))
  sequence_module:add(nn.SelectTable(-1))
  return sequence_module
end

function Tree2Tree_ED:new_decoder_sequence_module(core,vecs)
  local sequence_module = nn.Sequential()
  self.tgt_emb_module = nn.LookupTable(self.target_vocab_dim, self.target_emb_dim)
  if vecs then
    self.tgt_emb_module.weight:copy(vecs)
  end
  sequence_module:add(self.tgt_emb_module)
  sequence_module:add(nn.SplitTable(1, 2))
  sequence_module:add(nn.Sequencer(self.decoder.core))
  self.decoder.prediction_module = nn.Sequential()
                        :add(nn.Linear(self.mem_dim, self.target_vocab_dim))
                        :add(nn.LogSoftMax())
  sequence_module:add(nn.Sequencer(self.decoder.prediction_module))
  return sequence_module
end

function Tree2Tree_ED:new_decoder_tree_module(core,vecs)
  local tree_module = tree2tree.TreeSequential()
  self.tgt_emb_module = nn.LookupTable(self.target_vocab_dim, self.target_emb_dim)
  if vecs then
    self.tgt_emb_module.weight:copy(vecs)
  end
  tree_module:add(self.tgt_emb_module)
  tree_module:add(nn.SplitTable(1, 2)) -- table to tensor, works for both online and mini-batch mode
  tree_module.core = core
  tree_module:add(tree_module.core)
  tree_module.prediction_module = tree2tree.TreePredictionLayer(self.decoder_model, self.target_vocab_dim, self.lr_tree)
  if self.decoder_model == 'ntp' then
    tree_module:add(tree2tree.Treequencer(tree_module.prediction_module, false,false, self.lr_tree))
  else
    tree_module:add(tree2tree.Treequencer(tree_module.prediction_module, true, true, self.lr_tree))
  end -- Pass topo info true
  return tree_module
end

--[[ Forward coupling: Copy encoder cell and output to decoder ]]--
function Tree2Tree_ED:forwardConnect(encoder_core,decoder_core,inputSeqLen)
  local eTo, eTy = self.encoder_topology, self.encoder_recurrence
  local dTo, dTy = self.decoder_topology, self.decoder_recurrence
  local enc_has_cell, dec_has_cell = false, false

  local enc_state -- Dont one-lineize.
  if eTo == 'seq' and eTy == 'gru' then
    enc_state = encoder_core.outputs[inputSeqLen]
  elseif eTo == 'seq' and eTy == 'lstm' then
    enc_state, enc_cell = encoder_core.outputs[inputSeqLen], encoder_core.cells[inputSeqLen]
    enc_has_cell = true
  elseif eTo == 'tree' and eTy =='lstm' then
    enc_state, enc_cell = self.encoder.processer.output[1], self.encoder.processer.output[2]
    enc_has_cell = true
end

  local dec_state
  if dTo == 'seq' and dTy == 'gru' then
    decoder_core.userPrevOutput = self.mem_zeros   -- Hack to initialize and allow reference copy below
    dec_state = decoder_core.userPrevOutput
  elseif dTo == 'seq' and dTy == 'lstm' then
    decoder_core.userPrevOutput,decoder_core.userPrevCell = self.mem_zeros,self.mem_zeros
    dec_state, dec_cell = decoder_core.userPrevOutput, decoder_core.userPrevCell
    dec_has_cell = true
  elseif dTo == 'tree' then
    decoder_core.userPrevAncestral = self.mem_zeros
    dec_state = decoder_core.userPrevAncestral
  end

  -- Copy hidden state  (a.k.a output for lstm)
  dec_state = nn.rnn.recursiveCopy(dec_state, enc_state)
  assert(dec_state:norm() == enc_state:norm(), "Something went wrong forward-copying state.")

  -- Copy memory cell
  if (enc_has_cell and dec_has_cell) then -- only if both have memory cell
    dec_cell = nn.rnn.recursiveCopy(dec_cell, enc_cell)
    assert(dec_cell:norm() == enc_cell:norm(), "Something went wrong forward-copying state.")
  end

end

--[[ Backward coupling: Copy decoder gradients to encoder ]]--
function Tree2Tree_ED:backwardConnect(encoder_core,decoder_core)
  -- TODO: Change syntax of tree decoder processer -> core.
  local eTo, eTy = self.encoder_topology, self.encoder_recurrence
  local dTo, dTy = self.decoder_topology, self.decoder_recurrence

  local enc_state -- Dont one-lineize.
  local enc_has_cell = false
  if eTo == 'seq' and eTy == 'gru' then
    encoder_core.gradPrevOutput = self.mem_zeros -- Hack
    enc_state = encoder_core.gradPrevOutput
  elseif eTo == 'seq' and eTy == 'lstm' then
    encoder_core.gradPrevOutput, encoder_core.userNextGradCell = self.mem_zeros, self.mem_zeros
    enc_state, enc_cell = encoder_core.gradPrevOutput, encoder_core.userNextGradCell
    enc_has_cell = true
  elseif eTo == 'tree' and eTy =='cs-tree-lstm' then
    self.encoder.processer.gradPrevOutput   = self.mem_zeros
    self.encoder.processer.userNextGradCell = self.mem_zeros
    enc_state, enc_cell = self.encoder.processer.gradPrevOutput, self.encoder.processer.userNextGradCell
    enc_has_cell = true
end

  local dec_state
  local dec_has_cell = false
  if dTo == 'seq' and dTy == 'gru' then
    dec_state = decoder_core.userGradPrevOutput
  elseif dTo == 'seq' and dTy == 'lstm' then
    dec_state, dec_cell = decoder_core.userGradPrevOutput, decoder_core.userGradPrevCell
    enc_has_cell = true
  elseif dTo == 'tree' then
    dec_state = decoder_core.userGradPrevAncestral
  end

  -- Copy hidden state  (a.k.a output for lstm)
  local enc_state = nn.rnn.recursiveCopy(enc_state, dec_state)
  assert(dec_state:norm() == enc_state:norm(), "Something went wrong backward-copying state.")

  -- Copy memory cell
  if (enc_has_cell and dec_has_cell) then -- only if both have memory cell
    enc_cell = nn.rnn.recursiveCopy(enc_cell, dec_cell)
    assert(dec_cell:norm() == enc_cell:norm(), "Something went wrong backward-copying state.")
  end
end

-- forward needs to have been run already
-- So far, this is only tested for tree decoders
function Tree2Tree_ED:compute_output_LL(dec_output)
  assert(not self.decoder.predict, "TO compute LL during prediction, don't use this. forward already resturns it")
  assert(self.decoder.modules[4].logprobs, "Need to run forward before")
  -- Get word probs from Treequencer layer
  local word_logprobs = self.decoder.modules[4].logprobs
  -- Get topo probs from DRNN layer
  local topo_a_logprobs = self.decoder.modules[3].logprob_ancest
  local topo_f_logprobs = self.decoder.modules[3].logprob_frat
  local nA, kA = get_keys(topo_a_logprobs)
  local nF, kF = get_keys(topo_f_logprobs)
  assert(nA == nF, 'Logprob tables of different sizes!')
  debugutils.dprintf(2,'Word Logprob: %8.2f, Topo A LogProb: %8.2f, Topo F LogProb: %8.2f',
  rnn.recursiveSum(word_logprobs),rnn.recursiveSum(topo_a_logprobs),rnn.recursiveSum(topo_f_logprobs))
  local output_logprob = rnn.recursiveSum(word_logprobs) + rnn.recursiveSum(topo_a_logprobs) + rnn.recursiveSum(topo_f_logprobs)
  return output_logprob
end



function Tree2Tree_ED:forward(enc_input,dec_input)
  local enc_output = nil
  local dec_output = {}
  if self.encoder_topology == 'seq' then
    enc_output = self.encoder:forward(enc_input.sent)
  end

  debugutils.checkpoint(self,'F:E')
  local input_size = (enc_input.sent) and enc_input.sent:size(1) or enc_input:size(1)
  self:forwardConnect(self.encoder.core,self.decoder.core,input_size)
  debugutils.checkpoint(self,'F:FC')
  if Debug > 0 then
    -- Add word index info to tree, for debug printing
    dec_input.tree:labelNodes(dec_input.sent)
  end
  if self.decoder_topology == 'tree' then
    self.decoder:setInitialTree(dec_input.tree) -- might be changed after decoder core forward
  end
  dec_output = self.decoder:forward(dec_input.sent)
  debugutils.checkpoint(self,'F:D')
  return dec_output
end

function Tree2Tree_ED:backward(enc_input,dec_input,grad_output)
  self.decoder:backward(dec_input.sent,grad_output)
  debugutils.checkpoint(self,'B:D')
  self:backwardConnect(self.encoder.core,self.decoder.core)
  debugutils.checkpoint(self,'B:BC')
  local zeroTensor = torch.Tensor(self.encoder.output):zero()
  local zeroTensorGrad = (self.encoder_topology== 'seq')
                        and zeroTensor
                        or {zeroTensor, zeroTensor}
  self.encoder:backward(enc_input.sent,zeroTensorGrad)
  debugutils.checkpoint(self,'B:E')
end


--[[
  Function to prepare inputs before feeding them to the network. Handles differently
  depending on the topology of encoder and decoder. Note that from here on, inputs
  will be either:
    - a table corresponding to the sentence (for sequence topologies)
    - a "dictionary" of sentence (table) and tree (Tree)
]]--
function Tree2Tree_ED:prepare_inputs(source, target) -- target is optional
  local enc_input, dec_input, dec_gold
  local et,dt = self.encoder_topology, self.decoder_topology

  -- Encoder is easy, no changes
  enc_input = (et == "seq") and {sent=source.sent} or {sent=source.sent, tree=source.tree}
  if not target then -- Only asked for encoder part
    return enc_input
  end
  -- Decoder input depends on padding, and gold is different from input
  local dec_tree, dec_sent = target.tree, target.sent

  -- If there is </s> at the end, get rid of it, don't need its embedding.
  dec_sent_in = (self.pad_eos) and dec_sent:sub(1, -2) or dec_sent
  dec_input = (dt == 'seq') and {sent=dec_sent_in} or {tree=dec_tree, sent=dec_sent_in}
  if dt == 'seq' then
      dec_gold  = {sent = nn.SplitTable(1, 1):forward(dec_sent:sub(2, -1))}
  else -- tree decoder
      gold_tree = target.tree:prune_padding_leaves('SOS')
      dec_gold  = {tree = gold_tree, sent = dec_sent}
  end
  return enc_input, dec_input, dec_gold
end

function Tree2Tree_ED:prepare_prediction_inputs(tree, sent)
  local enc_input
  local et,dt = self.encoder_topology
  -- Encoder is easy, no changes
  enc_input = (et == "seq") and {sent=sent} or {sent=sent, tree=tree}
  return enc_input
end

function Tree2Tree_ED:prepare_inputs_criterion(dec_output,dec_gold)
  local et,dt = self.encoder_topology, self.decoder_topology
  local input_to_criterion = (dt == 'tree') and
  {sent = dec_output, tree =self.decoder.core.outputTree} or dec_output
  local gold_to_criterion = (dt == 'tree') and dec_gold or dec_gold.sent
  return input_to_criterion, gold_to_criterion
end

function Tree2Tree_ED:train(dataset)
  self.encoder:training()
  self.decoder:training()
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    -- A function to give to adagrad as input.
    local feval = function(x)
      -- Use either:
      self.grad_params:zero()
      assert(self.grad_params:norm() == 0, "Non zero gradients")

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]

        local enc_input, dec_input, dec_gold =
        self:prepare_inputs(dataset.source[idx],dataset.target[idx])

        debugutils.dprintf(1,'\n\n\nBatch: %i, Example Index: %i, Example True Index: %i\n',i,j,idx)
        debugutils.debug_inputs(1,enc_input, dec_input, dec_gold,
        self.encoder_topology, self.decoder_topology)

        --  FORWARD PASS --
        -- loss will be nil unless decoder is tree-(s|d)tp model
        local dec_output = self:forward(enc_input, dec_input)
        local example_topo_loss
        if #dec_output == 2 and torch.type(dec_output[2]) == 'number' then
          dec_output, example_topo_loss = unpack(dec_output)
        else
          example_topo_loss = nil
        end

        debugutils.debug_outputs(1,dec_input, dec_gold,dec_output,self)

        local criterion_input, criterion_target = self:prepare_inputs_criterion(dec_output,dec_gold)

        debugutils.debug_criterion_size(1,criterion_input, criterion_target,self.decoder_topology)

        local example_loss = self.criterion:forward(criterion_input, criterion_target)
        self.example_loss = example_loss
        debugutils.checkpoint(self,'F:C')
        if not example_topo_loss then
          debugutils.dprintf(1,"Example loss: %8.2f\n", example_loss)
        else
          debugutils.dprintf(1,"Example loss: word =%8.2f,  topo =%8.2f, joint =%8.2f\n", example_loss, example_topo_loss, example_loss+ example_topo_loss)
          example_loss = example_loss + example_topo_loss
        end
        loss = loss + example_loss

        -- BACKWARD PASS --
        local grad_output = self.criterion:backward(criterion_input, criterion_target)
        debugutils.checkpoint(self,'B:C')
        debugutils.dprint(1,"~ Checkpoint ED: backward criterion.")
        self:backward(enc_input, dec_input, grad_output)
        debugutils.dprint(1,"~ Checkpoint ED: backward encoder-decoder.")
      end -- iter over examples in batch

      loss = loss / batch_size
      self.grad_params:div(batch_size) -- Divide collected grad info by batch size

      if self.grad_clip > 0 then
        self.grad_params:clamp(-self.grad_clip, self.grad_clip)
      end

      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end -- feval

    if Debug > 0 then
      debugutils.debug_optim_step(feval,self.params,self.grad_params,self.optim_state)
    else
      optim.adagrad(feval, self.params, self.optim_state)
    end
    self:forget()
    if i % 10 == 0 then
      -- Clean up every few iters - trees seem to leak memory
      collectgarbage()  --
      collectgarbage()  --
    end
  end -- batch iterator
  xlua.progress(dataset.size, dataset.size)
end -- train


--[[ Given a source sentence (and tree) predict target sentence
    TODO: This function should probably live somwhere else. It will very likely
    depend on the decoder topology, so it should either be an attribute of the
    decoder itself, or the decoder class.
]]--
local MAX_SEQUENCE_LENGTH = 10
local MAX_TREE_DEPTH      = 4
local MAX_TREE_BREADTH    = 5

--  Without teacher forcing, free length (up to MAX_OUTPUT_SIZE)
function Tree2Tree_ED:predict_target(enc_input,vocab_tgt,debug)
  -- TODO: Add option to use beam search. Check from karpathy

  self.encoder:forward(enc_input)
  local input_size = (enc_input.sent ) and enc_input.sent:size(1) or enc_input:size(1)
  self:forwardConnect(input_size)

  dec_output = self.decoder:forward(dec_input)

  if self.decoder_topology == 'seq' then
    local predicted, probabilities = {}, {}
    local output = vocab_tgt.start_index
    for i = 1, MAX_OUTPUT_SIZE do
       local prediction = self.decoder:forward(torch.Tensor{output})[1]
         -- prediction contains the probabilities for each word IDs.
         -- The index of the probability is the word ID.
       local prob, wordIds = prediction:sort(1, true)
       -- First one is the most likely.
       output = wordIds[1]
       table.insert(predicted, wordIds[1])
       table.insert(probabilities, prob[1])
     -- Terminate on EOS token
       if output == vocab_tgt.end_index then
         break
       end
    end
  else
     local output = vocab_tgt.root_index
  end
  self:forget()
  -- self.decoder.core:forget()
  return predicted, probabilities
end

--[[ Compute loss for an example source-target pair (same criterion used as in trainin,
    i.e. With teacher forcing. ]]--
function Tree2Tree_ED:compute_example_loss(enc_input,dec_input,dec_gold, do_topoloss)
  local do_topoloss = do_topoloss or false
  --local example_loss = self.criterion:forward(dec_output.sent,dec_gold.sent)
  local dec_output_table = self:forward(enc_input, dec_input)
  local dec_output, example_topo_loss
  if not do_topoloss then
    dec_output, example_topo_loss = dec_output_table, nil
  else
    dec_output, example_topo_loss = unpack(dec_output_table)
  end

  local criterion_input, criterion_target = self:prepare_inputs_criterion(dec_output,dec_gold)
  local example_loss = self.criterion:forward(criterion_input, criterion_target)
  self:forget()
  return example_loss, example_topo_loss
end

function Tree2Tree_ED:compute_loss(dataset)
  self.encoder:evaluate()
  self.decoder:evaluate()
  local do_topoloss = (self.decoder_model == 'dtp' or self.decoder_model == 'stp')
  local loss = 0
  local topo_loss = (do_topoloss) and 0 or nil
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local enc_input, dec_input, dec_gold =
                          self:prepare_inputs(dataset.source[i],dataset.target[i])
    local example_loss, example_topo_loss
    example_loss, example_topo_loss = self:compute_example_loss(
    enc_input, dec_input, dec_gold, do_topoloss)
    if example_topo_loss then debugutils.dprint(1,example_loss, example_topo_loss)
    else debugutils.dprint(1,example_loss) end
    loss = loss + example_loss
    if example_topo_loss then topo_loss = topo_loss + example_topo_loss end
    collectgarbage()
  end
  loss = loss / dataset.size  -- average to get mean example loss
  if do_topoloss then
    topo_loss = topo_loss / dataset.size  -- average to get mean example loss
  end
  return loss, topo_loss
end


function Tree2Tree_ED:print_config()
  local num_params = self.params:size(1)
  local dec_top_string = self.decoder_topology
  local enc_top_string = self.encoder_topology

  if self.lr_tree then dec_top_string = dec_top_string .. ' (left-right)' end
  printf('%-25s = %d\n','num params', num_params)
  printf('%-25s = %d\n','    - encoder   ', self.enc_params:size(1))
  printf('%-25s = %d\n','    - decoder   ', self.dec_params:size(1))
  printf('%-25s\n', 'encoder:')
  printf('%-25s = %s\n','    - topology  ', self.encoder_topology)
  printf('%-25s = %s\n','    - recurrence', self.encoder_recurrence)
  printf('%-25s = %s\n','    - model     ', self.encoder_model)
  printf('%-25s\n', 'decoder:')
  printf('%-25s = %s\n','    - topology  ', dec_top_string)
  printf('%-25s = %s\n','    - recurrence', self.decoder_recurrence)
  printf('%-25s = %s\n','    - model     ', self.decoder_model)

  printf('%-25s = %d\n',   'source word vector dim', self.source_emb_dim)
  printf('%-25s = %d\n',   'target word vector dim', self.target_emb_dim)
  printf('%-25s = %d\n',   'memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %.2e\n', 'gradient clipping', self.grad_clip)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)

end

--
-- Serialization
--

function Tree2Tree_ED:save(path)
  local config = self.config
  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function Tree2Tree_ED.load(path)
  local state = torch.load(path)
  local model = tree2tree.Tree2Tree_ED.new(state.config)
  model.params:copy(state.params)
  return model, state.config
end


function Tree2Tree_ED:predict_dataset(format, dataset, maxpred)
  local predictions = {}
  local to_predict = maxpred or #dataset
  self:predicting()
  for i = 1, to_predict do
    xlua.progress(i, to_predict)
    local enc_input = self:prepare_inputs(dataset[i])
    --local enc_input  = self:prepare_prediction_inputs(enc_input)
    local pred_tree, pred_sent = self:tree_sampling(format,enc_input)
    if equals_any(format,{'tree','IFTTT','synth'}) then
      table.insert(predictions,{tree = pred_tree, sent = pred_sent})
    elseif format == 'BABYMT' then
      table.insert(predictions,{tree = pred_tree, sent = pred_sent})
    elseif format == 'atis' then
      local str_pred = ntTree2logical(pred_tree,true,pred_sent,self.target_vocab)
      table.insert(predictions,str_pred)
    end
    collectgarbage()
  end
  self:training()
  return predictions
end

-- Priming Sent: <T> <S> for tree decoder, <S> for seq decoder
function Tree2Tree_ED:newPrimingSent()
  local len = (self.decoder_topology == 'tree') and 2 or 1
  local priming_sent = torch.Tensor(len)
  if self.decoder_topology == 'tree' then
    priming_sent[1] = self.target_vocab.root_index
    priming_sent[2] = self.target_vocab.start_index
  else
    priming_sent[1] = self.target_vocab.start_index
  end
  return priming_sent
end

-- Priming Tree:  <S> <-(left) <T> for LR tree, <T> -> <S> for Tree.
-- Indices: <T> = 1, <S> = 2,
-- Position of correponding token in sentence (idx):
--        <T>: 1
--        <S>: 3??? NOTE CHECK
function Tree2Tree_ED:newPrimingTree()
  local primingTree
  if not self.primingTree then
    local primingTree = (self.lr_tree) and tree2tree.LRTree() or tree2tree.Tree()
    primingTree.index = 1
    primingTree.idx = 1
    primingTree.bdy = 'SOT'
    if self.lr_tree then
      c = tree2tree.LRTree()
      c.index = 2
      c.idx   = 3  -- Had 3 here before
      primingTree:add_left_child(c)
    else
      c = tree2tree.Tree()
      c.index = 2
      c.idx   = 3  -- Had 3 here
      primingTree:add_child(c)
    end
    self.primingTree = primingTree
  end
  local primingTree = (self.lr_tree) and tree2tree:LRTree():copy(self.primingTree,"full")
                                      or tree2tree:Tree():copy(self.primingTree,"full")
  return primingTree
end

-- Priming Tree:  <S> <-(left) <T> for LR tree, <T> -> <S> for Tree.
function Tree2Tree_ED:newIFTTTPrimingTree()
  local primingTree
  if not self.primingTree then
    local primingTree = (self.lr_tree) and tree2tree.LRTree() or tree2tree.Tree()
    primingTree.index = 1
    primingTree.idx = 1
    primingTree.bdy = 'SOT'
    if self.lr_tree then
      c = tree2tree.LRTree()
      c.index = 2
      c.idx   = 3  -- Had 3 here before
      primingTree:add_left_child(c)
    else
      c = tree2tree.Tree()
      c.index = 2
      c.idx   = 3  -- Had 3 here
      primingTree:add_child(c)
    end
    self.primingTree = primingTree
  end
  local primingTree = (self.lr_tree) and tree2tree:LRTree():copy(self.primingTree,"full")
                                      or tree2tree:Tree():copy(self.primingTree,"full")
  return primingTree
end

-- Priming Sent: <T> <S> for tree decoder, <S> for seq decoder
function Tree2Tree_ED:newIFTTTPrimingSent()
  --print(self.target_vocab.size)
  local len = 3
  local priming_sent = torch.Tensor(len)
  if self.decoder_topology == 'tree' then
    priming_sent[1] = self.target_vocab.root_index
    priming_sent[2] = self.target_vocab.start_index
    priming_sent[3] = self.target_vocab:index('ROOT')
  else
    priming_sent[1] = self.target_vocab.start_index
  end
  return priming_sent
end


function Tree2Tree_ED:test_sampling(format, enc_in,  dec_gold)
  print("******")
  print("Sampling sentence...")
  if self.decoder_topology == 'seq' then
    self:sequence_sampling(format, enc_in,dec_gold, 1)
  else
    self:tree_sampling(format, enc_in,dec_gold, 1)
  end
  print("******")
end

function Tree2Tree_ED:tree_sampling(format, enc_input, dec_gold, verbose)
  -- dec_gold is optional, for debugging purposes
  local verbose = verbose or 0
  local src_table  = (enc_input.sent) and torch.totable(enc_input.sent) or torch.totable(enc_input)
  local gold_table
  if dec_gold then
    gold_table = (dec_gold.sent) and torch.totable(dec_gold.sent) or torch.totable(dec_gold)
  end
  if verbose > 0  then
    printf('%-35s\n\t%s\n','Source sentence string:',table.concat(self.source_vocab:reverse_map(src_table,true),' '))
    printf('\t%s\n',table.concat(src_table,' '))
  end

  -- Need to give "super powers" to DRNN
  self.decoder.core.embedder = self.tgt_emb_module
  self.decoder.core.topologyModule = self.decoder.prediction_module
  if not self.decoder.core.sampler then
    self.decoder.core.sampler = tree2tree.Sampler({sample_mode = 'max', temperature=1})
  end
  self.decoder.core.end_token_ind = self.target_vocab.end_index  -- Not needed for DTP

  local priming_sent

  if format == 'IFTTT' then
    priming_sent = self:newIFTTTPrimingSent()
    if self.decoder_topology == 'tree' then
      self.decoder.core.tree = self:newIFTTTPrimingTree()
    end
  else
    priming_sent = self:newPrimingSent()
    if self.decoder_topology == 'tree' then
      self.decoder.core.tree = self:newPrimingTree()
    end
  end
  enc_out = self.encoder:forward(enc_input.sent) -- We pass encoder input as is
  local input_size = (enc_input.sent) and enc_input.sent:size(1) or enc_in:size(1)
  self:forwardConnect(self.encoder.core,self.decoder.core,input_size)

  self.tgt_emb_module:forward(priming_sent)
  local emb_sent = nn.SplitTable(1, 2)
  :forward(torch.Tensor(self.tgt_emb_module.output:size()):copy(self.tgt_emb_module.output))

  local dec_out = self.decoder.core:forward(emb_sent)
  local dec_out_tree = dec_out[1]
  local dec_out_sent = dec_out[2]
  local dec_output_logprob = dec_out[3]

  if verbose > 0 then
    printf('%-35s \t %8.2f\n','Predicted output logprob:',dec_output_logprob)
    printf('%-35s \n\t','Predicted tree:')
    dec_out_tree = dec_out_tree:prune_padding_leaves('SOS')
    dec_out_tree:print_preorder(dec_out_sent,self.target_vocab)
    dec_out_tree:print_offspring_layers(self.target_vocab:reverse_map(dec_out_sent))

    printf('%-35s \n\t','Predicted (linearized) sentence:')
    dec_out_tree:print_lexical_order(self.target_vocab:reverse_map(dec_out_sent))
    if print_tokenids then
      printf('\t')
      dec_out_tree:print_lexical_order(dec_out_sent)
    end
    printf('%-35s \n\t','Gold tree:')
    dec_gold.tree:print_offspring_layers(self.target_vocab:reverse_map(gold_table))
    printf('%-35s \n\t','Gold (linearized) sentence:')
    dec_gold.tree:print_lexical_order(self.target_vocab:reverse_map(gold_table))
    if print_tokenids then
      printf('\t')
      dec_gold.tree:print_lexical_order(dec_gold.sent)
    end
  end
  self:forget()
  return dec_out_tree, dec_out_sent, dec_output_logprob
end


function Tree2Tree_ED:sequence_sampling(enc_input, dec_input, dec_gold)
  -- testing sampling
  print("Sampling sentence...")

  local src_table = (enc_input.sent) and torch.totable(enc_input.sent) or torch.totable(enc_in)
  printf('%-35s %s\n\t','Source sentence string:',table.concat(self.source_vocab:reverse_map(src_table,true),' '))

  self.decoder.core:evaluate()
  self.decoder.prediction_module:evaluate()

  -- Need to give "super powers" to DRNN
  self.sampler = tree2tree.Sampler({sample_mode = 'temp', temperature=1})

  local priming_sent = self:newPrimingSent()


  enc_out = self.encoder:forward(enc_in) -- We pass encoder input as is
  local input_size = (enc_input.sent) and enc_input.sent:size(1) or enc_input:size(1)
  self:forwardConnect(self.encoder.core,self.decoder.core,input_size)

  dec_out_sent, dec_out_pobs = self.sampler:sequence_sampler(self.tgt_emb_module,self.decoder.core,
    self.decoder.prediction_module,priming_sent,self.target_vocab.end_index)

  printf('%-35s ','Predicted sentence:  ')
  print(table.concat(self.target_vocab:reverse_map(dec_out_sent),' '))

  printf('%-35s ','Gold sentence:  ')
  print(table.concat(self.target_vocab:reverse_map(dec_gold.sent),' '))


  self:forget()
  self.decoder.prediction_module:forget()

  self.decoder.core:training()
  self.decoder.prediction_module:training()

  assert(self.decoder.core.train and self.decoder.prediction_module.train, "Not cleared predict")
end

function Tree2Tree_ED:evaluate()
  self.encoder:evaluate()
  self.decoder:evaluate()
end

function Tree2Tree_ED:training()
  self.encoder:training()
  self.decoder:training()
end

function Tree2Tree_ED:forget()
  -- -- TODO: Make cleaning syntax homogenous
  if self.encoder_topology == 'tree' then
    self.encoder:clean(tree)
  else
    self.encoder:forget()
  end
  self.decoder.core:forget()
  self.decoder.prediction_module:forget()
end

function Tree2Tree_ED:predicting()
  if self.encoder_topology == 'seq' then
    self.encoder:evaluate()
  end
  if self.decoder_topology == 'tree' then
    self.decoder.core:predicting()
    self.decoder.prediction_module:predicting()
  end
end
