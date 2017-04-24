
function display_tree_example(tree,sent,vocab,padded,prune)
  local prune = prune or false
  local display_tree, recons_tree
  local sent = (torch.type(sent) == 'table') and sent or torch.totable(sent)
  if padded then
    if torch.type(tree) == 'tree2tree.LRTree' then
      --recons_tree = tree2tree.LRTree():copy(tree)
      recons_tree = tree2tree.LRTree():copy(tree, "full", true)
    else
      recons_tree = tree2tree.Tree():copy(tree, "full", true)
      --recons_tree = tree:prune_padding_leaves()
      -- TODO: Copy the style of TreeLR copy to accept removepad arg
    end
    display_tree = (prune) and recons_tree or tree
  else
    display_tree = tree
    recons_tree  = tree
  end
  print('Sentence string:"' ,table.concat(vocab:reverse_map(sent),' '),'"')
  if lr_tree then
    print('Tree break-down:')
    display_tree:print_preorder(vocab:reverse_map(sent))
    print('Reconstructed sentence:') -- Always with pruned
    recons_tree:print_lexical_order(vocab:reverse_map(sent))
  else
    print('Tree break-down:')
    display_tree:print_preorder(vocab:reverse_map(sent))
    print('Reconstructed sentence:') -- Always with pruned d
    -- TODO
  end
  print('Branch-sentences:')
  display_tree:print_offspring_layers(vocab:reverse_map(sent))
end



-- Expects table of tables. If len not specified, pads to lingest in the batch
function zero_pad_sequence_batch(seqs, len)
  local outTensor
  local max_len = 0
  if len == nil then
    for i=1,#seqs do
      max_len = (#seq[i] > max_len) and #seq[i] or max_len
    end
    len = max_len
  end
  outTensor = torch.Tensor(#seqs,len):zero()
  for k, seq in ipairs(seqs) do
    --print(k, #sent)
    local z = outTensor:sub(k,k)
    z = torch.Tensor(seqs)
  end
  return outTensor
end



function prepare_inputs(trees, sents, batch)
  local encoder_input, decoder_input, decoder_gold
  local batch_mode = batch or false
  if batch_mode then
    local batch_size = sents.source
  end
  --for k=1,batch_size do -- TODO
    encoder_input = sents.source
    if dec_type == 'seq' and criterion_type ~= 'tree' then
        decoder_input = sents.target:sub(2, -2)
        decoder_gold  = nn.SplitTable(1, 1):forward(sents.target:sub(2, -1))
    elseif dec_type == 'seq' and criterion_type == 'tree' then
      decoder_input = sents.target:sub(2, -2)
      local gold_tree = tree2tree.Tree():copy(trees.target.children[1])
      gold_tree:prune_leftmost_leaves()
      decoder_gold  = {tree = gold_tree, sent = sents.target}
    else -- tree decoder
        decoder_input = {tree = trees.target,sent = sents.target:sub(1, -2)}
        local gold_tree = tree2tree.Tree():copy(trees.target.children[1])
        gold_tree:prune_leftmost_leaves()
        decoder_gold  = {tree = gold_tree, sent = sents.target}
    end
  --end
  return encoder_input, decoder_input, decoder_gold
end

torch.manualSeed(123)
function generate_random_dataset(vocab_size, batch_size, max_len)
  local max_branching = 4
  local max_depth     = 4
  local min_sent_len  = 6

  local random_branch
  random_branch = function(depth)
    local tree = tree2tree.Tree()
    local num_children = (depth < max_depth) and torch.random(0,max_branching) or 0
    for i=1,num_children do
      local c = random_branch(depth+1)
      tree:add_child(c)
    end
    return tree
  end
  -- local len = #
  -- local sent = torch.IntTensor(len)
  -- for i = 1, len do
  --   local token = tokens[i]
  --   sent[i] = vocab:index(token)
  -- end

  local assign_labels = function(tree)
    local nodes = tree:depth_first_preorder()
    --sent[1] = 1
    --sent[2] = 2
    local sent = {}
    local max_idx = 2
    for _,node in pairs(nodes) do
      if node.parent == nil then
        node.idx = 1
        sent[node.idx] = 1
      elseif node.left_brother == nil then
        node.idx = 2
        sent[node.idx] = 2
      elseif node.right_brother == nil then
        node.idx = nil
      else
        max_idx = max_idx + 1
        node.idx = max_idx
        sent[node.idx] = torch.random(3,vocab_size+3) -- SOS/SOT offset
      end
    end
    sent[max_idx + 1] = vocab_size + 3
    for _,node in pairs(nodes) do node.idx = node.idx or max_idx +1 end
    print(sent)
    sent = torch.Tensor(sent)
    return sent
  end

  local create_pair = function()
    --local sents = torch.Tensor(batch_size,max_len):zero() -- zero padding
    local sents = {}
    local trees = {}
    for k=1,batch_size do
      local sent, tree
      while (not sent) or sent:size(1) < min_sent_len do
        tree = random_branch(0)
        tree.idx = 1
        tree:set_preorder_index()
        sent = assign_labels(tree)
      end
      --print(k, #sent)
      --local z = sents:sub(k,k,1,#sent)
      --z = torch.Tensor(sent)
      sents[k] = sent
      trees[k] = tree
    end
    return trees,sents
  end

  local src_t, src_s = create_pair()
  local tgt_t, tgt_s = create_pair()
  sents = {source = src_s, target = tgt_s}
  trees = {source = src_t, target = tgt_t}
  return trees, sents
end

function make_toy_example(batch_size)
  local sents = torch.Tensor()
  local sent = torch.Tensor({1,2,5,6,11,3,12})
  local a, b, _b, b_ = tree2tree.Tree(), tree2tree.Tree(), tree2tree.Tree(), tree2tree.Tree()
  a.idx, b.idx, _b.idx, b_.idx = 6, 5, 2, 7
  b:add_child(_b)
  b:add_child(a)
  b:add_child(b_)
  local c, d, _d, d_ = tree2tree.Tree(), tree2tree.Tree(), tree2tree.Tree(), tree2tree.Tree()
  c.idx, d.idx, _d.idx, d_.idx = 3, 4, 2, 7
  d:add_child(_d)
  d:add_child(c)
  d:add_child(b)
  d:add_child(d_)
  local tree = tree2tree.Tree()
  tree.idx = 1
  tree:add_child(d)
  tree:set_preorder_index()
  local sents = {source = {sent}, target = {sent}}
  local trees = {source = {tree}, target = {tree}}
  return trees, sents
end

function hsep()
  print(string.rep('-', 80))
end



-- Can only be run after a pass of forward on the module
function flesh_out_sequential(seqmodule)
  print('')
  printf('%6s %25s %25s %35s\n', 'Module','Module Type','Output Type','Output Size')
  hsep()
  for i=1,#seqmodule.modules do
    local out = seqmodule.modules[i].output
    local type, type_out = torch.type(seqmodule.modules[i]), torch.type(seqmodule.modules[i].output)
    local n,keyset, size_elem, size_out, keys, num_elem
    if type_out == 'table' then
      size_elem = string_tensor_table(out)
      keys = ""
      size_out = -1
      num_elem = -1
    else
      size_out = (out:nDimension() >0) and out:size(1) or nil
      size_elem = out and out:size(1) or nil
      --print(out:size())
      --if size_elem == 1 then size_elem = out end
      size_out = tostring(size_out)
      size_elem = tostring(size_elem)
      keys = ""
    end
    printf('%6i %25s %25s %35s\n',i,type,type_out,size_elem)
  end
  hsep()
  print('')
end



-- function treeify_table(tree, table)
--   local out = {}
--   local nodes = tree:depth_first_preorder()
--   for _,node in pairs(nodes) do
--     if (node.idx > 2) then
--       out[node.idx-2] = table[node.index]
--     end
--   end
--   return out
-- end


function flatten_tree_table(tree, table)
  local out = {}
  local nodes = tree:depth_first_preorder()
  for _,node in pairs(nodes) do
    if (node.idx > 2) then
      out[node.idx-2] = table[node.index]
    end
  end
  return out
end

function reverse_flatten_tree_table(tree, table)
  local grad = {}
  local nodes = tree:depth_first_preorder()
  for _,node in pairs(nodes) do
    if (node.idx > 2) then
      grad[node.index] = table[node.idx-2]
    end
  end
  return grad
end
