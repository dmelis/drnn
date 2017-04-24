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


function Sampler:tree_sample_node(pW)
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
  return wt, sampleLogprobs
end


function Sampler:sequence_sample_node(pW)
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
