
local debugutils = {}

debugutils.tensor = {}

-- Set debuggin levels
debug_level_pred      = 1  -- Affects debug_topo_pred, debug_topo_grad
debug_level_DRNN_pred = 1 -- Affects debug_DRNN_forward_predict
debug_level_DRNN_fwd  = 1 -- Affects debug_DRNN_forward
debug_level_DRNN_bwd  = 1 -- Affects debug_DRNN_backward

--------------------------------------------------------------------------------
-----------------------------    Debugging utils  ------------------------------
--------------------------------------------------------------------------------

function get_norm_clone(x)
  local var = x:clone()
  return var:norm()
end

--[[
    Function to check parameter/gradient norms before and after optimization step.
]]--
function debugutils.debug_optim_step(feval,params,grad_params,optim_state)
  local norm_param_prev     = get_norm_clone(params)
  local norm_grad_prev      = get_norm_clone(grad_params)
  --local norm_enc_param_prev = get_norm_clone(self.enc_params)
  --local norm_dec_param_prev = get_norm_clone(self.dec_params)

  x_star, fx = optim.adagrad(feval, params, optim_state)

  print('~ Optim step:')
  -- Print values before optimization step...
  printf('%10s %10s %10s\n','','||params||','||g params||')
  printf('%10s %10.4f %10.4f\n','  Before:',norm_param_prev,norm_grad_prev)
  -- ... and after
  printf('%10s %10.4f %10.4f\n','  After:',params:norm(),grad_params:norm())
end

function debugutils.debug_output_tree(predicted, target)
  local targettree, targetSent = target.tree, target.sent
  local input_nodes  = predicted:depth_first_preorder()
  local target_nodes = targettree:depth_first_preorder()
  printf('%4s %4s %10s %10s\n','ind','idx','pred','target')
  printf('-------------------------------\n')
  for innode, targnode in seq.zip(input_nodes, target_nodes) do
    local pred_word = torch.range(1,innode.value:nElement())[innode.value:eq(torch.max(innode.value))]
    local targ_word = targetSent[targnode.idx]
    printf('%4i %4i %10i %10i\n',innode.index,innode.idx,pred_word[1],targ_word)
end
printf('-------------------------------\n')
end

-- Utility function for conditional printing when Debug is on.
function debugutils.dprint(level,...)
  local args = {...}
  if (Debug ~= nil and level <= Debug) then
    print(unpack(args))
    --if object ~= nil then print(object) end
  end
end

function debugutils.dprintf(level,formatter,...)
  local args = {...}
  if (Debug ~= nil and level <= Debug) then
    printf(formatter,unpack(args))
  end
end


--[[
    Checkpoints for encoder decoder script.
]]--
function debugutils.checkpoint(ED,id)
  local et,dt = ED.encoder_topology, ED.decoder_topology
  local e,d = ED.encoder, ED.decoder

  local vars = {}
  local label = ""
  if id == 1 then
  elseif id == 'F:E' then
    label = "forward - encoder"
    var = (et == 'seq') and {output = ED.encoder.core.outputs[#ED.encoder.core.outputs]} or {output = e.processer.output[1]}

  elseif id == 'F:FC' then
    label = "forward connect"
    var = (dt == 'seq') and {userPrevOutput = ED.decoder.core.userPrevOutput} or {userPrevAncestral = ED.decoder.core.userPrevAncestral}

  elseif id == 'F:D' then
    label = "forward - decoder"
    var = (dt == 'seq') and {output = ED.decoder.core.outputs[#ED.decoder.core.outputs]} or {output = d.output.value} -- head

  elseif id == 'F:C' then
    label = "forward - criterion"
    var = {loss = ED.example_loss}
    --var = (dt == 'seq') and {output = ED.decoder_core.outputs[#ED.decoder_core.outputs]} or {output = d.output.value} -- head

  elseif id == 'B:C' then
    label = "backward - criterion"
    var = {}
    --var = (dt == 'seq') and {output = ED.decoder_core.outputs[#ED.decoder_core.outputs]} or {output = d.output.value} -- head

  elseif id == 'B:D' then
    label = "backward - decoder"
    var = (dt == 'seq') and {userGradPrevOutput = ED.decoder.core.userGradPrevOutput, gradInput = d.gradInput}
                         or {userGradPrevAncestral = ED.decoder.core.userGradPrevAncestral}

  elseif id == 'B:BC' then
    label = "backward connect"
    var = (et == 'seq') and {gradPrevOutput = ED.encoder.core.gradPrevOutput} or {gradPrevOutput = e.processer.gradPrevOutput}
    --if (dt == 'seq') then var["userGradPrevOutput"] = ED.decoder_core.userGradPrevOutput else var["gradPrevAncestral"] = ED.decoder_core.userGradPrevAncestral end

  elseif id == 'B:E' then
    label = "backward - encoder"
    var = (et == 'seq') and {gradInput = e.gradInput} or {gradInput = e.gradInput}
    --print('Here')
    --print(e.gradInput)
  end
  local formatter = "~ Checkpoint ED: passed %s. "
  local vals = {label}
  for k,v in pairs(var) do
    local vtype = (torch.type(v) == "torch.DoubleTensor") and 't'
              or ((torch.type(v) == "torch.IntTensor") and 'it' or 'n')
    local suffix = (vtype == 't') and " ||%s|| = %8.2f" or "%s = %8.2f "
    formatter = formatter .. suffix ..",  "
    vals[#vals + 1] = k
    vals[#vals + 1] = (vtype == 't') and v:norm() or
    (vtype == 'it' and v:type('torch.DoubleTensor'):norm() or v)
  end
  formatter = formatter .. "\n"
  debugutils.dprintf(1,formatter, unpack(vals))
end


function debugutils.debugtraversal(level,node_traversal)
  if (Debug ~= nil and level <= Debug) then
    printf('\nTraversal plan:\n')
    for i,k in pairs(node_traversal) do printf(' (%i,%i) ', k[1].index,k[1].idx) end
    printf('\n')
  end
end

function debugutils.debug_DRNN_forward(level,model,lr_tree,total_steps,step,node,prev_node,
  inA,hA_prev,inF,hF_prev,hA,hF,hOut,pA,pA_L,pA_R,pF)
  if (level ~= nil and debug_level_DRNN_fwd > level) then
    return
  end
  local meta_labels        = {'NODE','PARENT / PREV BRO','INPUTS       ','OUTPUTS        '}
  local id_labels          = {'ind','idx'}
  local relative_id_labels = {'ind','idx','ind','idx'}
  local input_labels       = {'inA','hA','inF','hF'}
  local output_labels      = not lr_tree and {'hA\'','hF\'','hOut','pA','pF'}
                                        or {'hA\'','hF\'','hOut','pAL','pAR','pF'}
  local input_tensors      = {inA,hA_prev,inF,hF_prev}
  local output_tensors     = lr_tree and {hA,hF,hOut,pA_L,pA_R,pF} or {hA,hF,hOut,pA,pF}
  local labels             = {id_labels, relative_id_labels, input_labels, output_labels}
  local node_id = {node.index,node.idx}
  local relative_ids =  {node.parent.index,
                         node.parent.bdy or node.parent.idx,
                        (prev_node) and prev_node.index or '--',
                        (prev_node==nil) and '--' or prev_node.bdy or prev_node.idx,
                      }
  local tensors = {node_id, relative_ids, input_tensors, output_tensors}
  if(step == 1) then
    print('DRNN forward steps:')
    pretty_print_tensors(labels, true, false, meta_labels)
  end
  pretty_print_tensors(tensors, false, step == total_steps)
end

function debugutils.debug_DRNN_forward_predict(level,model,lr_tree,total_steps,tree,to_process,
  node,prev_node,inA,hA_prev,inF,hF_prev,hA,hF,hOut,pA,pA_L,pA_R,pF,pW,indA,indF, word_vec,word_id)

  -- debugutils.dprintf(1,total_steps,self.tree:size(true),#to_process,node.idx, node.index,node.side, node.depth, node.parent.index)
  if (level ~= nil and debug_level_DRNN_pred > level) then
    return
  end
  -- if node.index == 2 then
  --   debugutils.dprintf(1,'%5s %5s %5s %5s %5s %5s %5s %5s\n','Step','TSize','#queque','idx','index','side','depth','p.index')
  -- end

  local meta_labels        = {'TREE STATE','NODE','PARENT PREV BRO','INPUTS','OUTPUTS '}
  local tree_state_labels  = {'step','|T|','|q|','dpth'}
  local id_labels          = lr_tree and {'ind','idx','s'} or {'ind','idx'}
  local relative_id_labels = {'ind','idx','ind','idx'}
  local input_labels       = {'inA','hA','inF','hF'}
  local output_labels      = not lr_tree and {'hA\'','hF\'','hOut','pA','pF','wvec','word'}
                                        or {'hA\'','hF\'','hOut','pAL','pAR','pF','wvec','word'}
  local input_tensors      = {inA,hA_prev,inF,hF_prev}
  local output_tensors     = lr_tree and {hA,hF,hOut,pA_L,pA_R,pF,word_vec,word_id} or {hA,hF,hOut,pA,pF,word_vec,word_id}
  local labels             = {tree_state_labels,id_labels, relative_id_labels, input_labels, output_labels}
  local tree_state         = {total_steps,tree:size(true),#to_process,node.depth}
  local node_id = {node.index,node.idx, node.side}
  local relative_ids =  {node.parent.index,
                         node.parent.bdy or node.parent.idx,
                        (prev_node) and prev_node.index or '--',
                        (prev_node==nil) and '--' or prev_node.bdy or prev_node.idx,
                      }
  local tensors = {tree_state,node_id, relative_ids, input_tensors, output_tensors}
  if(node.index == 2) then pretty_print_tensors(labels, true, false, meta_labels) end
  pretty_print_tensors(tensors, false, not(next(to_process) and (total_steps < 1000)))
end

function debugutils.debug_DRNN_backward(level,model,lr_tree,total_steps,step,node,prev_node,
  inputTable, gradOutputTable,gradInputs)
  if (not Debug or level > Debug) then
    return
  else
    local meta_labels        = {'NODE','PARENT / PREV BRO','INPUTS        ','GRAD OUTPUTS         ','GRAD INPUTS     '}
    local id_labels          = {'ind','idx'}
    local relative_id_labels = {'ind','idx','ind','idx'}
    local input_labels       = {'inA','hA','inF','->hF'}
    local output_grad_labels = {'hA', 'hF<-','hOut'}
    local input_grad_labels  = {'inA','hA^','inF','->hF'}
    local node_id = {node.index,node.idx}
    local relative_ids =  {node.parent.index,
                           node.parent.bdy or node.parent.idx,
                          (prev_node) and prev_node.index or '--',
                          (prev_node==nil) and '--' or prev_node.bdy or prev_node.idx,
                        }
    if model ~= 'ntp' then
      if not lr_tree then
        table.insert(output_grad_labels,'pA')
      else
        table.insert(output_grad_labels,'pAL')
        table.insert(output_grad_labels,'pAR')
      end
    end
    if model == 'dtp' then
      table.insert(output_grad_labels,'gpF')
    end
    local labels  = {id_labels, relative_id_labels,input_labels, output_grad_labels, input_grad_labels}
    local tensors = {node_id, relative_ids, inputTable, gradOutputTable, gradInputs}
    if(step == 1) then pretty_print_tensors(labels,true,false,meta_labels) end
    pretty_print_tensors(tensors,false,step == total_steps)
  end
end


function hsep()
  print(string.rep('-', 80))
end

function string_tensor_table_size(table)
  local size = '{'
  for k,v in pairs(table) do
    if torch.type(v) == 'table' then
      key_size = string_tensor_table_size(v)
    else
      key_size = tostring(v:size(1))
    end
    size = size .. key_size
    if k < #table then size = size .. ',' end
  end
  size = size .. '}'
  return size
end

function string_tensor_table_norm(table)
  local norm = '{'
  for k,v in pairs(table) do
    if torch.type(v) == 'table' then
      key_norm = string_tensor_table_norm(v)
    else
      key_norm = string.format('%.2f',v:norm())
    end
    norm = norm .. key_norm
    if k < #table then norm = norm .. ',' end
  end
  norm = norm .. '}'
  return norm
end

-- Can only be run after a pass of forward on the module
function debugutils.flesh_out_sequential(level,seqmodule,mode)
  if (not Debug or level > Debug) then
    return
  end

  local mode = mode or 'fwd'
  local meta_labels
  if mode == 'fwd' then
    meta_labels        = {'MODULE','OUTPUTS'}
  else
    meta_labels        = {'MODULE','GRADINPUT'}
  end
  local module_labels = {'No','Type'}
  local value_labels  = {'Type','Size','Norm'}

  --local module_values      = {module_i, module_type}
  -- local value_tensors     = lr_tree and {hA,hF,hOut,pA_L,pA_R,pF,word_vec} or {hA,hF,hOut,pA,pF,word_vec}
  hsep()

  printf('%28s | %70s\n',unpack(meta_labels))
  printf('%6s %21s | %21s %20s %35s\n','No','Type','Type','Size','Norm')

  hsep()
  for i=1,#seqmodule.modules do
    local out
    if mode == 'fwd' then
      out = seqmodule.modules[i].output
    else
      out = seqmodule.modules[i].gradInput
    end
    local type, type_out = torch.type(seqmodule.modules[i]), torch.type(out)
    local n,keyset, size_elem, size_out, keys, num_elem
    if string.find(type,'equencer') or type == 'tree2tree.DRNN' or type == 'nn.SelectTable' then
      -- All elements are the same, take a representative for the size_out
      local example = out[1] or out[2] -- In case first is empty, as in Tree
      if (type == 'nn.SelectTable') then example = out[#out] end
      local size_one_elem = (torch.type(example) == 'table') and string_tensor_table_size(example) or tensor_size_to_string(example)
      local norm_one_elem = (torch.type(example) == 'table') and string_tensor_table_norm(example) or string.format('%.2f',example:norm())
      size_elem = '{' .. tostring(#out) .. 'x' .. size_one_elem ..'}'
      norm_elem = '{' .. norm_one_elem ..', ...}'
    elseif type_out == 'table' then
      size_elem = string_tensor_table_size(out)
      norm_elem = string_tensor_table_norm(out)
      keys = ""
      size_out = -1
      num_elem = -1
    elseif torch.isTensor(out) then
      size_out = (out:nDimension() >0) and out:size(1) or nil
      size_elem = tensor_size_to_string(out)
      --size_elem = out and out:size(1) or nil
      --print(out:size())
      --if size_elem == 1 then size_elem = out end
      size_out = tostring(size_out)
      --size_elem = tostring(size_elem)
      norm_elem = (type_out == 'torch.IntTensor') and string.format('%.2f',out[1]) or string.format('%.2f',out:norm())
      keys = ""
    end
    printf('%6i %21s | %21s %20s %35s\n',i,type,type_out,size_elem, norm_elem)
  end
  hsep()
  print('')
end



function tensor_size_to_string(t)
  local nD = t:nDimension()
  local str = '('
  for i=1,nD do
    str = str .. tostring(t:size(i)) .. (i<nD and 'x' or '')
  end
  str = str .. ')'
  return str
end






function debugutils.debug_inputs(level,enc_input, dec_input, dec_gold,enc_topo,dec_topo)
  if (not Debug or level > Debug) then
    return
  else
    print('Encoder input:',tensor_to_string(enc_input.sent))
    print('Decoder input:',tensor_to_string(dec_input.sent))
    print('Decoder gold:',tensor_to_string(dec_gold.sent))

    if dec_topo == 'tree' then
      print('Decoder input tree:')
      dec_input.tree:print_offspring_layers(torch.totable(dec_input.sent))
      if torch.type(dec_input.tree) == 'LRTree' then
        dec_input.tree:print_preorder(torch.totable(dec_input.sent),'outward')
      else
        dec_input.tree:print_preorder(torch.totable(dec_input.sent))
      end
    end
  end
end

function debugutils.debug_criterion_size(level,criterion_input, criterion_target,ED)
  if (not Debug or level > Debug) then
    return
  else
    print('Criterion input size:', #criterion_input.sent)
    print('Criterion target size:',criterion_target.sent:size(1))
  end
end

-- This one is called from within TreeCriterion
function debugutils.debug_criterion(level,index,idx,criterion_input,criterion_target)
  if (not Debug or level > Debug) then
    return
  else
    if index == 2 then
      debugutils.dprint(level,string.rep('-',28))
      printf('%3s %3s | %5s %5s %2s |\n','ind','idx','Pred','True','')
      debugutils.dprint(level,string.rep('-',28))
    end
    local sampleLogprobs, wt = torch.max(criterion_input, 1)
    wt = wt:view(-1):long()
    local correct = (wt[1] == criterion_target) and '' or 'X'
    printf('%3i %3i | %5i %5i %2s |\n',index,idx,wt[1],criterion_target,correct)
  end
end


function debugutils.debug_outputs(level,dec_input, dec_gold,dec_output,dec_topo, ED)
  if (not Debug or level > Debug) then
    return
  else
    local dec_output_sent_size = #dec_output
    local dec_output_tree_size = (dec_topo ==  'tree')
                    and ED.decoder_core.outputTree:size() or nil

    local dec_gold_sent_size = dec_gold.sent:size(1)
    local dec_gold_tree_size = (dec_topo ==  'tree')
                    and dec_gold.tree:size() or nil

    local dec_input_size = (dec_topo == 'seq')
        and dec_input.sent:size(1) or dec_input.tree:size()

    --print(self.decoder_core.outputTree:size(), dec_gold.tree:size())

    print('\t - decoder input/output/gold sentence sizes:',
    dec_input_size, dec_output_sent_size,dec_gold_sent_size)

    --assert(dec_output_sent_size ==  dec_gold_sent_size,  "Error: Output and target w/ different sent sizes:" .. tostring(dec_output_sent_size) .. "~=" .. tostring(dec_gold_sent_size))
    assert(dec_output_tree_size ==  dec_gold_tree_size,
    "Error: Output and target w/ different tree sizes:" ..
    tostring(dec_output_tree_size) .. "~=" .. tostring(dec_gold_tree_size))
  end
end




function pretty_print_tensors(tensors,header,last,meta)
  local last = last or false
  local header = header or false
  local funit = {string='s',int='i',number='i',tensor='.2f'}--,small_string='%5s'}
  local line_formatter, meta_formatter = "",""
  local values = {}
  local global_width = 0
  for i,t in ipairs(tensors) do
    local len = #t
    local section_width = 0
    for j,k in ipairs(t) do
      local type = torch.isTensor(k) and 'tensor' or torch.type(k)
      local val = (type == 'string' or type == 'number') and k or k:norm()
      if type == 'number' then
        type = (torch.round(val) == val) and 'int' or 'float' -- Any other wat to do this???
      end
      values[#values+1] = val
      if val == 's' or val == 'L' or val == 'R' then width = 3
      elseif (type == 'int' and (val <1000)) or ((type == 'string') and (string.find(val,'ind')
                                                     or string.find(val,'idx')
                                                     or string.find(val,'step')
                                                     or string.find(val,'idx')
                                                     or val == 'dpth'
                                                     or val == '--'
                                                     or string.find(val,'SO')
                                                     or string.find(val,'|')
            )) then width = 5
      --or string.find(val,'SO') or string.find(val,'OS') or val == 'step'
      --elseif (type == 'number' and val < 100) then width = 5
      --elseif (type == 'tensor') then width = 7
      else width = 7 end
      section_width = section_width + width
      line_formatter = line_formatter ..'%'.. tostring(width) .. funit[type]
    end
    if meta then
      meta_formatter = meta_formatter .. '%' .. tostring(section_width) ..'s'
    end
    if i < #tensors then
      line_formatter = line_formatter .. ' | '
      if meta then meta_formatter = meta_formatter .. ' | ' end
    end
    global_width = global_width + section_width
  end
  line_formatter = line_formatter .. ' |\n'
  meta_formatter = meta_formatter .. ' |\n'
  if header then print(string.rep('-',global_width + 3*#tensors -1)) end -- extra 3 for each section division, except last, only 2
  if meta then printf(meta_formatter,unpack(meta)) end
  printf(line_formatter, unpack(values))
  if header or last then print(string.rep('-',global_width + 3*#tensors -1))  end
end



-- function print_table_tensors(tables)
--   local line_formatter = ""
--   local norms = {}
--   for i,t in ipairs(tables) do
--     local len = #t
--     line_formatter = line_formatter .. '|' .. string.rep('%8.2f ',len)
--     for j,k in pairs(labels[i]) do
--       norms[i*j + j] = t[k]:norm()
--     end
--   end
--   printf(line_formatter, unpack(norms))
-- end




-- function debugutils.debug_DRNN_backward(level,model,node,inA,hA_prev,inF,hF_prev,
--   grad_hA_children,grad_hF_next,gradOutput)
--   local line_width = 60
--   if (not Debug or level > Debug) then
--     return
--   else
--     local ins_and_outs = {inA=inA,hA=hA_prev,inF=inF, hF= hF_prev,
--                           grad_hA=grad_hA_children,
--                           grad_hF=grad_hF_next}
--     local states = {'inA','inF','hA','hF'}
--     local grads  = {'grad_hA', 'grad_hF','grad_Out'}
--
--     if model == 'ntp' then
--       ins_and_outs.grad_Out = gradOutput[node.index]
--     end
--     if model ~= 'ntp' then
--       ins_and_outs.grad_Out = gradOutput[node.index][1]
--       ins_and_outs.grad_pA     = gradOutput[node.index][2]
--       table.insert(grads,'grad_pA')
--     end
--     if model == 'dtp' then
--       ins_and_outs.grad_pF = gradOutput[node.index][3]
--       table.insert(grads,'grad_pF')
--     end
--
--     printf('\nInputs to cell module backward at node (%i/%i):\n', node.index,node.idx)
--     print(string.rep('-',line_width))
--     printf('%10s %20s %5s %8s\n','name','type','size','norm')
--     --printf('%5i %20s %20s %5i %8.2f\n',node.index,k,torch.type(v),v:size(1),v:norm(1))
--     print(string.rep('-',line_width))
--
--     for i,list in pairs{states, grads} do
--       for j,k in ipairs(list) do
--         local v = ins_and_outs[k]
--         printf('%10s %20s %5i %8.2f\n',
--         k,torch.type(v),v:size(1),v:norm(1))
--       end
--       print(string.rep('-',line_width))
--     end
--     print(string.rep('-',line_width))
--   end
-- end





MAX_ALLOWED_NORM = 10000
function debugutils.norm_halt(id,t)
  local max_norm = get_max_norm(t)
  --print('Norms: ' .. id .. ' ' .. string_norms_table(t))
  if max_norm > MAX_ALLOWED_NORM then
    print("Halting because of large norm at ",id)
    print('Norms: ' .. id .. ' ' .. string_norms_table(t))
    tensor_table_clamp(t,-10,10) -- FIXME: Do this in a cleaner way
    --bk()
  end
end

function get_max_norm(t)
  local max_norm = 0
  if torch.isTensor(t) then
    max_norm = t:norm()
  elseif torch.type(t) == 'table' then
    for k,v in pairs(t) do
      if torch.isTensor(v) then
        this_norm = v:norm()
      elseif torch.type(v) == 'table' then
        this_norm = get_max_norm(v)
      end
      max_norm = (this_norm > max_norm) and this_norm or max_norm
    end
  end
  return max_norm
end


function debugutils.check_zeros(t)
  local n = get_max_norm(t)
  if n > 0 then
    print("Halting because of non zero zero vectors")
    bk()
  end
end


function debugutils.print_tensor_info(level,t)
  if (Debug ~= nil and level <= Debug) then
    if torch.isTensor(t) then
      print(v:size(1), v:norm())
    elseif torch.type(t) == 'table' then
      for k,v in pairs(t) do
        if torch.isTensor(v) then
          print(k,v:size(1), v:norm())
        elseif torch.type(v) == 'table' then
          debugutils.print_tensor_info(level,v)
        end
      end
    end
  end
end

-- TODO: Fix format for all others beyond *
function debugutils.debug_topo_pred_header(level,lr_tree,bottom)
  if (level ~= nil and debug_level_pred > level) then
    return
  end
  local bottom = bottom or false
  local width = lr_tree and 82 or 67
  if not bottom then
    debugutils.dprint(1,'Node prediction layer forward:')
    debugutils.dprint(1,string.rep('-',width))
    if not lr_tree then
      debugutils.dprintf(1,'%8s  |  %s   |   %s  |         %s            |\n', 'NODE', 'ANCESTRAL','FRATERNAL','WORD')
      debugutils.dprintf(1,'%4s %4s |  %4s  %5s |  %4s %5s  | %6s %5s  %6s    |\n', 'ind', 'idx', 'Pred','True','Pred','True','Pred','Prob','True')
    else -- *
      debugutils.dprintf(1,'%8s  |  %s |  %s |   %s  |         %s            |\n', 'NODE', 'ANCEST. (L)','ANCEST. (R)','FRATERNAL','WORD')
      debugutils.dprintf(1,'%4s %4s |  %4s  %5s |  %4s  %5s |  %4s  %5s | %6s %5s  %6s    |\n', 'ind', 'idx', 'Pred','True','Pred','True','Pred','True','Pred','Prob','True')
    end
  end
  debugutils.dprint(1,string.rep('-',width))
end



function debugutils.debug_topo_grad_header(level,lr_tree,bottom)
  if (level ~= nil and debug_level_pred > level) then
    return
  end
  local bottom = bottom or false
  local width = lr_tree and 72 or 50
  if not bottom then
    debugutils.dprint(1,'Node prediction layer backward:')
    debugutils.dprint(1,string.rep('-',width))
    if not lr_tree then
      debugutils.dprintf(1,' %8s  |     %s     |      %s   |\n', 'NODE', 'ANCESTRAL','FRATERNAL')
      debugutils.dprintf(1,' %4s %4s | %4s %5s %6s | %4s %5s %6s |\n', 'ind', 'idx', 'Pred','True','Grad','Pred','True','Grad')
    else
      debugutils.dprintf(1,'%8s  |   %s   |   %s   |      %s     |\n', 'NODE', 'ANCESTRAL (L)','ANCESTRAL (R)','FRATERNAL')
      debugutils.dprintf(1,'%4s %4s | %4s %5s %6s | %4s %5s %6s |  %4s %5s %6s |\n', 'ind', 'idx', 'Pred','True','Grad','Pred','True','Grad','Pred','True','Grad')
    end
  end
  debugutils.dprint(1,string.rep('-',width))
end




function debugutils.debug_topo_pred(level,model,lr_tree,prob_a,tau_a,gold_a_bin,prob_f,tau_f,gold_f_bin)
  --print(torch.gt(prob_a-0.2,0.5)); bk()
  if (level ~= nil and debug_level_pred > level) then
    return
  end
  local pred_a_L, pred_a_R, pred_a, correct_a_L, correct_a_R, correct_a
  if lr_tree then
    pred_a_L = (prob_a['L'][1] > tau_a)  and 1 or 0 -- bool
    pred_a_R = (prob_a['R'][1] > tau_a)  and 1 or 0
    correct_a_L = (pred_a_L == gold_a_bin['L'][1]) and '' or 'X'
    correct_a_R = (pred_a_R == gold_a_bin['R'][1]) and '' or 'X'
  else
    pred_a = (prob_a[1] > tau_a)  and torch.Tensor(1):fill(1) or torch.zeros(1)-- bool
    correct_a = (pred_a[1] == gold_a_bin[1]) and '' or 'X'
  end
  local pred_f = (prob_f[1] > tau_f)  and 1 or 0  -- bool
  local correct_f = (pred_f == gold_f_bin[1]) and '' or 'X'
  if lr_tree then
    printf(' %4.4f %2i %2s | %4.4f %2i %2s | %4.4f %2i %2s ',
    prob_a['L'][1], gold_a_bin['L'][1], correct_a_L,
    prob_a['R'][1], gold_a_bin['R'][1], correct_a_R,
    prob_f[1], gold_f_bin[1], correct_f)
  else
    printf(' %4.4f %2i %2s | %4.4f %2i %2s ',
    prob_a[1], gold_a_bin[1], correct_a, prob_f[1], gold_f_bin[1], correct_f)
  end
end

function debugutils.debug_topo_grad(level,model,lr_tree,prob_a,gold_a,grad_topo_a,prob_f,gold_f,grad_topo_f)
  --print(torch.gt(prob_a-0.2,0.5)); bk()
  if (level ~= nil and debug_level_pred > level) then
    return
  end
  if not lr_tree then
    printf('%4.4f %2i % 4.4f | %4.4f %2i % 4.4f |\n',
    prob_a[1],gold_a[1],grad_topo_a[1],prob_f[1], gold_f[1],grad_topo_f[1])
  else
    printf('%4.4f %2i % 4.4f | %4.4f %2i % 4.4f |  %4.4f %2i % 4.4f |\n',
    prob_a['L'][1],gold_a['L'][1],grad_topo_a['L'][1],
    prob_a['R'][1],gold_a['R'][1],grad_topo_a['R'][1],
    prob_f[1], gold_f[1],grad_topo_f[1])
  end
end

function debugutils.tensor.debug_topo_grad(level,model,lr_tree,inputs)
  --print(torch.gt(prob_a-0.2,0.5)); bk()
  -- for k,v in pairs(inputs) do
  --   if torch.type(v) == 'table' then
  --     print(k,v)
  --   else
  --     print(k,v:size())
  --   end
  -- end
  local i = 1 -- item in batch to display

  prob_a,gold_a,grad_topo_a,prob_f,gold_f,grad_topo_f =
  unpack(recursiveSelect(inputs,1,i))
  --
  -- local step_inputs = recursiveSelect(real_input,2,i)
  if (Debug ~= nil and level <= Debug) then
    if not lr_tree then
      printf('%4.4f %2i % 4.4f | %4.4f %2i % 4.4f\n',
      prob_a[1],gold_a[1],grad_topo_a[1],prob_f[1], gold_f[1],grad_topo_f[1])
    else
      printf('%4.4f %2i % 4.4f | %4.4f %2i % 4.4f |  %4.4f %2i % 4.4f\n',
      prob_a['L'][1],gold_a['L'][1],grad_topo_a['L'][1],
      prob_a['R'][1],gold_a['R'][1],grad_topo_a['R'][1],
      prob_f[1], gold_f[1],grad_topo_f[1])
    end
  end
end


function debugutils.tensor.debug_DRNN_forward(level,model,lr_tree,total_steps,step,
  inputs)

  local  self_idx,
    parent_idx, parent_ind,
    bro_idx, bro_ind,
    inA,hA_prev,hA,
    inF,hF_prev,hF,
    hOut,pA,pA_L,pA_R,pF = unpack(recursiveSelect(inputs,1,1))

  if (not Debug or level > Debug) then
    return
  else
    local meta_labels        = {'NODE','PARENT PREV BRO','INPUTS ','OUTPUTS '}
    local id_labels          = {'step','idx'}
    local relative_id_labels = {'ind','idx','ind','idx'}
    local input_labels       = {'inA','hA','inF','hF'}
    local output_labels      = not lr_tree and {'hA\'','hF\'','hOut','pA','pF'}
                                          or {'hA\'','hF\'','hOut','pAL','pAR','pF'}
    local input_tensors      = {inA,hA_prev,inF,hF_prev}
    local output_tensors     = lr_tree and {hA,hF,hOut,pA_L,pA_R,pF} or {hA,hF,hOut,pA,pF}
    local labels             = {id_labels, relative_id_labels, input_labels, output_labels}
    local node_id = {step,self_idx}
    local relative_ids =  {parent_ind,parent_idx,bro_ind,bro_idx}
    local tensors = {node_id, relative_ids, input_tensors, output_tensors}
    if(step == 1) then pretty_print_tensors(labels, true, false, meta_labels) end
    pretty_print_tensors(tensors, false, step == total_steps)
  end
end

function debugutils.tensor.debug_topo_pred(level,model,lr_tree,tau_a,tau_f,inputs)
  --inputs = (prob_a,tau_a,gold_a_bin,prob_f,tau_f,gold_f_bin)
  --print(torch.gt(prob_a-0.2,0.5)); bk()

  -- for k,v in ipairs(inputs) do
  --   local val = (torch.type(v) == 'table' and v['L']:size()) or ((torch.isTensor(v)) and v:size() or v)
  --   print(torch.type(v), val)
  -- end

  local  prob_a,gold_a_bin,prob_f,gold_f_bin
   = unpack(recursiveSelect(inputs,1,1))

  if (Debug ~= nil and level <= Debug) then
    local pred_a_L, pred_a_R, pred_a, correct_a_L, correct_a_R, correct_a
    if lr_tree then
      pred_a_L = (prob_a['L'][1] > tau_a)  and 1 or 0 -- bool
      pred_a_R = (prob_a['R'][1] > tau_a)  and 1 or 0
      correct_a_L = (pred_a_L == gold_a_bin['L'][1]) and '' or 'X'
      correct_a_R = (pred_a_R == gold_a_bin['R'][1]) and '' or 'X'
    else
      pred_a = (prob_a[1] > tau_a)  and torch.Tensor(1):fill(1) or torch.zeros(1)-- bool
      correct_a = (pred_a[1] == gold_a_bin[1]) and '' or 'X'
    end
    local pred_f = (prob_f[1] > tau_f)  and 1 or 0  -- bool
    local correct_f = (pred_f == gold_f_bin[1]) and '' or 'X'
    if lr_tree then
      printf(' %4.4f %2i %2s | %4.4f %2i %2s | %4.4f %2i %2s ',
      prob_a['L'][1], gold_a_bin['L'][1], correct_a_L,
      prob_a['R'][1], gold_a_bin['R'][1], correct_a_R,
      prob_f[1], gold_f_bin[1], correct_f)
    else
      printf(' %4.4f %2i %2s | %4.4f %2i %2s ',
      prob_a[1], gold_a_bin[1], correct_a, prob_f[1], gold_f_bin[1], correct_f)
    end
  end
end


--[[
    Checkpoints for encoder decoder script.
]]--
function debugutils.tensor.checkpoint(ED,id)
  local et,dt = ED.encoder_topology, ED.decoder_topology
  local e,d = ED.encoder, ED.decoder

  local vars = {}
  local label = ""
  if id == 1 then
  elseif id == 'F:E' then
    label = "encoder forward."
    var = (et == 'seq') and {output = ED.encoder.core.outputs[#ED.encoder.core.outputs]} or {output = e.processer.output[1]}

  elseif id == 'F:FC' then
    label = "forward connect."
    var = (dt == 'seq') and {userPrevOutput = ED.decoder.core.userPrevOutput} or {userPrevAncestral = ED.decoder.core.userPrevAncestral}

  elseif id == 'F:D' then
    label = "decoder forward."
    var = (dt == 'seq') and {output = ED.decoder.core.outputs[#ED.decoder.core.outputs]} or {output = d.output.value} -- head

  elseif id == 'F:C' then
    label = "criterion forward."
    var = {loss = ED.example_loss}
    --var = (dt == 'seq') and {output = ED.decoder_core.outputs[#ED.decoder_core.outputs]} or {output = d.output.value} -- head

  elseif id == 'B:C' then
    label = "criterion backward."
    var = {}
    --var = (dt == 'seq') and {output = ED.decoder_core.outputs[#ED.decoder_core.outputs]} or {output = d.output.value} -- head

  elseif id == 'B:D' then
    label = "decoder backward."
    var = (dt == 'seq') and {userGradPrevOutput = ED.decoder.core.userGradPrevOutput, gradInput = d.gradInput}
                         or {userGradPrevAncestral = ED.decoder.core.userGradPrevAncestral}

  elseif id == 'B:BC' then
    label = "backward connect."
    var = (et == 'seq') and {gradPrevOutput = ED.encoder.core.gradPrevOutput} or {gradPrevOutput = e.processer.gradPrevOutput}
    --if (dt == 'seq') then var["userGradPrevOutput"] = ED.decoder_core.userGradPrevOutput else var["gradPrevAncestral"] = ED.decoder_core.userGradPrevAncestral end

  elseif id == 'B:E' then
    label = "encoder backward"
    var = (et == 'seq') and {gradInput = e.gradInput} or {gradInput = e.gradInput}
    print('Here')
    print(e.gradInput)

  end
  local formatter = "~ Checkpoint ED: passed %s"
  local vals = {label}
  for k,v in pairs(var) do
    local vtype = (torch.type(v) == "torch.DoubleTensor") and 't'
              or ((torch.type(v) == "torch.IntTensor") and 'it' or 'n')
    local suffix = (vtype == 't') and " ||%s|| = %8.2f" or "%s = %8.2f "
    formatter = formatter .. suffix ..",  "
    vals[#vals + 1] = k
    vals[#vals + 1] = (vtype == 't') and v:norm() or
    (vtype == 'it' and v:type('torch.DoubleTensor'):norm() or v)
  end
  formatter = formatter .. "\n"
  debugutils.dprintf(1,formatter, unpack(vals))
end



-- IFTTT stuff

function debugutils.debug_IFTTT_matching(match_debug_level,extracted_cand, extracted_ref, vocab)
  debugutils.dprint(match_debug_level+1,'channel',extracted_cand['channel'])
  debugutils.dprint(match_debug_level+1,'func',extracted_cand['func'])
  debugutils.dprint(match_debug_level+1,'channel',extracted_ref['channel'])
  debugutils.dprint(match_debug_level+1,'func',extracted_ref['func'])
  local width = 73
  debugutils.dprint(match_debug_level,string.rep('-',width))
  debugutils.dprintf(match_debug_level,'%10s | %6s | %25s %25s\n','Level','Branch','Candidate','Reference')
  debugutils.dprint(match_debug_level,string.rep('-',width))
  for i,level in ipairs({'channel','func'}) do
    for j, branch in ipairs({'IF', 'THEN'}) do
      debugutils.dprintf(match_debug_level,'%10s | %6s | %25s %25s\n',level,branch,
                extracted_cand[level][branch] and vocab:token(extracted_cand[level][branch]) or 'NA',
                extracted_ref[level][branch]  and vocab:token(extracted_ref[level][branch]) or 'NA')
    end
  end
  debugutils.dprint(match_debug_level,string.rep('-',width))
end






return debugutils
