
factory = {}
--[[
  Requirements:
    - 'DTP' (model 3) can work with or withoutc EOS padding.
    - 'STP' requires EOS padding but not leaf padding.
    - 'NTP' requires EOS padding *and* leaf padding.

]]


function factory.new_recurrent_module(type,in_dim,mem_dim)
  assert(type == 'gru' or type == 'lstm', "DRNN recurrence has to be one of LSTM or GRU")
  local wrapper_module, core_module
  if type == 'gru' then
    wrapper_module = nn.GRU(in_dim,mem_dim)
  elseif type == 'lstm' then
    wrapper_module = nn.LSTM(in_dim,mem_dim)
  end
  local core_module = wrapper_module.recurrentModule
  return core_module
end

--[[ Sequential version ]]--
function factory.new_combination_module(mem_dim)
  local module = nn.Sequential()
  local p = nn.ParallelTable():add(nn.Linear(mem_dim, mem_dim))
                              :add(nn.Linear(mem_dim, mem_dim))
  module:add(p)
  module:add(nn.CAddTable())
  module:add(nn.Tanh())
  return module
end


--[[
    Model 1: Stopping prediction made purely with vocabularty tokens,
    i.e. no additional modules for widtrh/depth stopping prediction.
    TODO: Have the pre-softmax linear layer here (instead of outside)?
]]--
function factory.model_ntp_composer(DM,WM,SC,in_dim,mem_dim,out_dim)
  local cell = nn.Sequential()
  -- Level 1: Split
  local p = nn.ConcatTable()
  p:add(nn.NarrowTable(1,2))
  p:add(nn.NarrowTable(3,2))
  cell:add(p)
  -- Level 2: Send to width/depth modules
  local p2 = nn.ParallelTable():add(DM):add(WM)
  cell:add(p2)
  -- Level 3: Duplicate output from previous layer, send one to state combination.
  local conc = nn.ConcatTable()
           :add(nn.Identity())
           :add(SC)
  cell:add(conc)
  local map_to_vocab = nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.Linear(mem_dim,out_dim))     -- Word prediction
  cell:add(map_to_vocab)
  -- Level 4: Flattem the three resulting vectors into size 3 table.
  cell:add(nn.FlattenTable())
  -- output = {h_a, h_f, h_out}
  return cell
end


--[[
    Model 2: Single-Topology prediction module.
    Both depth stopping prediction made with additional independent
    module, but depth is controlled with stop token.
    TODO: Sharing for all sub-modules!!!
]]--
function factory.model_stp_composer(DM,WM,SC,in_dim,mem_dim,out_dim)
  local cell = nn.Sequential()
  -- Level 1: Split
  local p = nn.ConcatTable()
  p:add(nn.NarrowTable(1,2))
  p:add(nn.NarrowTable(3,2))
  cell:add(p)
  -- Level 2: Send to width/depth modules
  local p2 = nn.ParallelTable():add(DM):add(WM)
  cell:add(p2)
  -- Level 3: Duplicate output from previous layer, send one to state combination.
  -- 3.1: we do state combination
  local combine_predict = nn.Sequential()
        :add(SC)
  -- 3.2: two predictors based on state combination
  local l3_splitter = nn.ConcatTable()
        :add(nn.Linear(mem_dim,out_dim))     -- Word prediction
        :add(nn.Linear(mem_dim,1))          -- Depth prediction
  local sigm = nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.Sigmoid())
  combine_predict:add(l3_splitter):add(sigm)
  -- Merge this branch with previous level. Use indentity on the other side.
  local conc = nn.ConcatTable()
           :add(nn.Identity())
           :add(combine_predict)
  cell:add(conc)
  -- Level 4: Flatten the three resulting vectors into a size 5 table.
  cell:add(nn.FlattenTable())
  -- output = {h_a, h_f, h_out, p_a}
  return cell
end


--[[
    Model 3: Dual-Topology prediction.
    Both width/depth stopping prediction made with additional independent
    modules.
    TODO: Sharing for all sub-modules!!!
]]--
function factory.model_dtp_composer(DM,WM,SC,in_dim,mem_dim,out_dim)
  local cell = nn.Sequential()
  -- Level 1: Split
  local p = nn.ConcatTable()
  p:add(nn.NarrowTable(1,2))
  p:add(nn.NarrowTable(3,2))
  cell:add(p)
  -- Level 2: Send to width/depth modules
  local p2 = nn.ParallelTable():add(DM):add(WM)
  cell:add(p2)
  -- Level 3: Duplicate output from previous layer, send one to state combination.
  -- 3.1: we do state combination
  local combine_predict = nn.Sequential()
        :add(SC)
  --3.2: three predictors based on state combination
  local l3_splitter = nn.ConcatTable()
        :add(nn.Linear(mem_dim,out_dim))     -- Word prediction
        :add(nn.Linear(mem_dim,1))          -- Depth prediction
        :add(nn.Linear(mem_dim,1))          -- Width prediction
  local sigm = nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.Sigmoid())
        :add(nn.Sigmoid()) -- If this is ever changed to LogSigmoid, need to change sampling accordingly
  combine_predict:add(l3_splitter):add(sigm)
  -- Merge this branch with previous level. Use indentity on the other side.
  local conc = nn.ConcatTable()
           :add(nn.Identity())
           :add(combine_predict)
  cell:add(conc)
  -- Level 4: Flatten the three resulting vectors into a size 5 table.
  cell:add(nn.FlattenTable())
  -- output = {h_a, h_f, h_out, p_a, p_f}
  return cell
end

--[[=========================================================================
  THE following three are Left-right versions of the above for trees that
  distinguish left from right offspring. Everything else is the same.
============================================================================]]--


--- BIG TODO: Models 4 and 5

--[[
    Model 6: Dual-Topology prediction + distinct L/R depth prediction
    Both width/depth stopping prediction made with additional independent
    modules.
    TODO: Sharing for all sub-modules!!!
]]--
function factory.model_dtp_lr_composer(DM,WM,SC,in_dim,mem_dim,out_dim)
  local cell = nn.Sequential()
  -- Level 1: Split
  local p = nn.ConcatTable()
  p:add(nn.NarrowTable(1,2))
  p:add(nn.NarrowTable(3,2))
  cell:add(p)
  -- Level 2: Send to width/depth modules
  local p2 = nn.ParallelTable():add(DM):add(WM)
  cell:add(p2)
  -- Level 3: Duplicate output from previous layer, send one to state combination.
  -- 3.1: we do state combination
  local combine_predict = nn.Sequential()
        :add(SC)
  -- 3.2: three predictors based on state combination
  local l3_splitter = nn.ConcatTable()
        :add(nn.Linear(mem_dim,out_dim))    -- Word prediction
        :add(nn.Linear(mem_dim,1))          -- Depth prediction: Left Side
        :add(nn.Linear(mem_dim,1))          -- Depth prediction: Right Side
        :add(nn.Linear(mem_dim,1))          -- Width prediction
  local sigm = nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.Sigmoid())
        :add(nn.Sigmoid())
        :add(nn.Sigmoid())
  combine_predict:add(l3_splitter):add(sigm)
  -- Merge this branch with previous level. Use identity on the other side.
  local conc = nn.ConcatTable()
           :add(nn.Identity())
           :add(combine_predict)
  cell:add(conc)
  -- Level 4: Flatten the three resulting vectors into a size 6 table.
  cell:add(nn.FlattenTable())
  -- output = {h_a, h_f, h_out, p_a_l, p_a_r, p_f}
  return cell
end





return factory

--[[ Graph version of combination module]]--
-- function DRNN:new_combination_module()
--   local input_frat       = nn.Identity()()
--   local input_ancest     = nn.Identity()()
--   local module = nn.Sequential(
--       nn.Tanh()(
--       nn.CAddTable(){
--       nn.Linear(mem_dim, 2*mem_dim)(input_frat),
--       nn.Linear(mem_dim, 2*mem_dim)(input_ancest)
--     }))
--   return module
-- end
