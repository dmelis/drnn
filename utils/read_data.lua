--[[

  Functions for loading/writing data from/to disk, e.g.:
    - Loading datasets
    - Getting paths
    - Loading word embeddings
    - Padding
    - Writing predictions

    IMPORTANT: IF these indices are changed, must make sure that prepare_inputs
    agrees with this. We need to eliminate the one corresponding to EOS before
    forwarding to newtork.
]]--

map_to_tds = true

function tree2tree.read_embedding(vocab_path, emb_path)
  local vocab = tree2tree.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function tree2tree.prune_unused_vectors(emb_vecs,emb_vocab,vocab_keep)
  -- use only vectors in data (not necessary, but gives faster training), get rid of everything else
  -- for words not in embeddig vocabulary , assign random
  -- returns tensor of size |vocab_keep| x vec_dim
  --
  local num_embedding_complement = 0
  local emb_dim = emb_vecs:size(2)
  local vecs = torch.Tensor(vocab_keep.size, emb_dim)
  for i = 1, vocab_keep.size do
    local w = string.gsub(vocab_keep:token(i), '\\', '') -- remove escape characters
    if emb_vocab:contains(w) then
      vecs[i] = emb_vecs[emb_vocab:index(w)]
    else
      num_embedding_complement = num_embedding_complement + 1
      vecs[i]:uniform(-0.05, 0.05)
    end
  end
  --print('        unk count = ' .. num_embedding_complement)
  emb_vocab = nil
  emb_vecs = nil
  collectgarbage()
  collectgarbage()
  return vocab_keep,vecs,num_embedding_complement
end


function tree2tree.read_default_embedding(slang,tlang,vocab_keep_src,vocab_keep_tgt,disjoint,initialize_composed)
  -- second, third argument is optional
  -- disjoint: even if src and tgt language is the same, process them individually (e.g. when domains are different)
  -- initalize_composed: if True, tokens like President_Obama will be initalized to the sum of the parts
  local disjoint = disjoint or false
  local initialize_composed = initialize_composed == nil and true or initialize_composed
  local tlang = tlang or nil
  local emb_dir = tree2tree.data_dir .. '/embeddings/' .. slang .. '/'
  local emb_prefix = emb_dir .. ((slang=='en') and 'glove.840B' or 'glove.Wiki')
  local emb_vocab, emb_vecs = {}, {}
  local rand_init_src, rand_init_tgt

  --------------------------
  --- Start with source side
  local emb_vocab_src, emb_vecs_src = tree2tree.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
  -- Expand
  if initialize_composed then
    local composed_vocab, composed_vecs = tree2tree.initialize_composed_embeddings(emb_vocab_src, emb_vecs_src, vocab_keep_src)
    if #composed_vocab > 0 then
      emb_vocab_src, emb_vecs_src = tree2tree.expand_embeddings({emb_vocab_src,emb_vecs_src},{composed_vocab, composed_vecs})
    end
  end
  -- Prune
  if vocab_keep_src ~= nil then
    emb_vocab.source, emb_vecs.source, rand_init_src = tree2tree.prune_unused_vectors(emb_vecs_src,emb_vocab_src,vocab_keep_src)
  else
    emb_vocab.source, emb_vecs.source = emb_vocab_src, emb_vecs_src
  end


  ----------------------------
  -- Now target side
  local emb_vocab_tgt, emb_vecs_tgt
  if tlang == nil then
    emb_vocab, emb_vecs = emb_vocab_src, emb_vecs_src
  elseif tlang == slang and (not disjoint) then
    emb_vocab.target, emb_vecs.target = emb_vocab_src, emb_vecs_src
    rand_init_tgt = rand_init_src
    -- TODO: Not very efficient. Maybe in the future pass the src=tgt check further down the code to avoid storage duplication
  elseif tlang == slang and (disjoint) then
    -- Means same language on src and tgt, but we want to decouple the embeddings
    local composed_vocab_tgt, composed_vecs_tgt
    if initialize_composed then
      composed_vocab_tgt, composed_vecs_tgt = tree2tree.initialize_composed_embeddings(emb_vocab_src, emb_vecs_src, vocab_keep_tgt)
      if #composed_vocab_tgt > 0 then
        emb_vocab_tgt, emb_vecs_tgt = tree2tree.expand_embeddings({emb_vocab_src,emb_vecs_src},{composed_vocab_tgt, composed_vecs_tgt})
      end
    else
      emb_vocab_tgt, emb_vecs_tgt = emb_vocab_src, emb_vecs_src -- NOTE: Dangerous. Make sure this doe not affect stc vocab./vecs
    end
    emb_vocab.target, emb_vecs.target, rand_init_tgt = tree2tree.prune_unused_vectors(emb_vecs_tgt,emb_vocab_tgt,vocab_keep_tgt)
  else -- means different target than source
    local tgt_emb_dir = tree2tree.data_dir .. '/embeddings/' .. tlang .. '/'
    local tgt_emb_prefix = tgt_emb_dir .. ((tlang=='en') and 'glove.840B' or 'glove.Wiki')
    local emb_vocab_tgt, emb_vecs_tgt = tree2tree.read_embedding(tgt_emb_prefix .. '.vocab', tgt_emb_prefix .. '.300d.th')
    if vocab_keep_tgt ~= nil then
      emb_vocab_tgt, emb_vecs_tgt,rand_init_tgt = tree2tree.prune_unused_vectors(emb_vecs_tgt,emb_vocab_tgt,vocab_keep_tgt)
    end
    --emb_vocab.source, emb_vocab.target = emb_vocab_src, emb_vocab_tgt
    emb_vocab.target,  emb_vecs.target  = emb_vocab_tgt,  emb_vecs_tgt
  end
  printf('%8s %8s %8s %8s %8s\n','','Vocab','Dim','Used',' Rand Init (i.e. not in Emb)')
  printf('%8s %8d %8d %8d %8d\n','source',emb_vocab.source.size,emb_vecs.source:size(2),vocab_keep_src.size,rand_init_src)
  printf('%8s %8d %8d %8d %8d\n','target',emb_vocab.target.size,emb_vecs.source:size(2),vocab_keep_tgt.size,rand_init_tgt)
  collectgarbage()
  collectgarbage() -- Why twice? Read this: http://stackoverflow.com/questions/28320213/why-do-we-need-to-call-luas-collectgarbage-twice
  return emb_vocab, emb_vecs
end

-- One string per line, no processing
function tree2tree.read_list(path, maxk)
  local strings = {}
  local maxk = maxk or -1
  local file = io.open(path, 'r')
  local line
  while true and (maxk == -1 or (#strings < maxk)) do
    line = file:read()
    if line == nil then break end
    strings[#strings + 1] = line
  end
  file:close()
  return strings
end

-- Converts string to tensor of indices corresponding to words in vocab
-- Pads as requested
function tree2tree.string_to_sentence(str, vocab, lowercase, pad_sos, pad_eos, pad_sot)
  local tokens = str:split(' ') -- Before usinf import() on stringx: stringx.split(line)
  if pad_sot then
    table.insert(tokens,SOT_IDX,vocab.root_token)
  end
  local last_spot = #tokens
  if pad_sos then
    table.insert(tokens,SOS_IDX,vocab.start_token)
  end
  if pad_eos then
    table.insert(tokens,vocab.end_token) -- At the end
  end
  local len = #tokens
  local sent = torch.IntTensor(len)
  for i = 1, len do
    local token = tokens[i]
    if lowercase then
      token = string.lower(token)
    end
    sent[i] = vocab:index(token)
  end
  return sent
end


-- TODO: After rebuttal. Remove inner function, replace with string_to_setence above
function tree2tree.read_sentences(path,vocab,args)
  pad_sos = args.pad_sos or false   -- pads start of sentence
  pad_eos = args.pad_eos or false   -- pads end of sentence
  pad_sot = args.pad_sot or false   -- pads beginning of sentence with tree "root" symbol. Only useful when padding trees too.
  local maxk    = args.maxk               -- If provided, reads only first k examples
  local lowercase = args.lowercase

  local sentences = {}
  local file = io.open(path, 'r')
  local line
  local count = 0
  while true and (not maxk or count<maxk) do
    line = file:read()
    if line == nil then break end
    local tokens = line:split(' ') -- Before usinf import() on stringx: stringx.split(line)
    if pad_sot then
      table.insert(tokens,SOT_IDX,vocab.root_token)
    end
    local last_spot = #tokens
    if pad_sos then
      table.insert(tokens,SOS_IDX,vocab.start_token)
    end
    if pad_eos then
      table.insert(tokens,vocab.end_token) -- At the end
    end
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      if lowercase then
        token = string.lower(token)
      end
      sent[i] = vocab:index(token)
    end
    count = count + 1
    sentences[count] = sent
  end
  file:close()
  if map_to_tds then
    sentences_tds = tds.Hash(sentences)
    sentences = nil
    collectgarbage()  --clean data variable
    return sentences_tds
  end
  return sentences
end

function tree2tree.read_sentences_to_tensor(path,vocab,args)
  pad_sos = args.pad_sos or false   -- pads start of sentence
  pad_eos = args.pad_eos or false   -- pads end of sentence
  pad_sot = args.pad_sot or false   -- pads beginning of sentence with tree "root" symbol. Only useful when padding trees too.
  max_len = args.max_len
  maxk    = args.maxk               -- If provided, reads only first k examples
  pad_from = args.pad_from or 'R'

  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = line:split(' ')
    if pad_sot then
      table.insert(tokens,SOT_IDX,vocab.root_token)
    end
    local last_spot = #tokens
    if pad_sos then
      table.insert(tokens,SOS_IDX,vocab.start_token)
    end
    if pad_eos then
      table.insert(tokens,vocab.end_token) -- At the end
    end
    local len = #tokens
    local sent = torch.IntTensor(1,max_len):fill(0)
    assert(max_len >= len, "Error: Sentence exceeds max len" .. tostring(max_len) .. '<' .. tostring(len))
    local true_len = math.min(len,max_len)
    for i = 1, true_len do
      local token = tokens[i]
      if pad_from == 'L' then
        sent[{1,max_len - true_len + i }] = vocab:index(token)
      else
        sent[{1,i}] = vocab:index(token)
      end
    end
    sentences[#sentences + 1] = sent
  end
  file:close()
  local sentences_tensor = nn.JoinTable(1):forward(sentences)
  return sentences_tensor
end

function tree2tree.read_trees(args)
  local lr_tree = args.lr_tree
  local parent_file = io.open(args.parent_path, 'r')
  local label_file
  if args.label_path ~= nil then label_file = io.open(label_path, 'r') end
  local count = 0
  local trees = {}
  local maxk    = args.maxk               -- If provided, reads only first k examples

  while true and (not maxk or count<maxk) do
    local parents = parent_file:read()
    if parents == nil then break end
    parents = parents:split(' ')
    for i, p in ipairs(parents) do
      parents[i] = tonumber(p)
    end

    local labels
    if label_file ~= nil then
      labels = label_file:read():split(' ') -- Before had:  stringx.split(label_file:read()
      for i, l in ipairs(labels) do
        -- ignore unlabeled nodes
        if l == '#' then
          labels[i] = nil
        else
          labels[i] = tonumber(l)
        end
      end
    end
    count = count + 1
    trees[count] = tree2tree.tree_from_parents(parents, labels, lr_tree, args.multi_root)
    if lr_tree then
      trees[count]:set_outward_preorder_index()
    else
      trees[count]:set_preorder_index()
    end
    -- trees[count] = this_tree
    -- this_tree = nil -- NEW. Check if it breaks things.
    -- print(count, collectgarbage('count'))
  end
  parent_file:close()
  --
  -- Doesnt' work:
  -- if map_to_tds then
  --   trees_tds = tds.Hash(trees)
  --   trees = nil
  --   collectgarbage()  --clean data variable
  --   return trees_tds
  -- end
  return trees
end

function tree2tree.read_trees_tensor(sents,parent_path,lr_tree,seq_len,pad_args,label_path)
  local lr_tree = lr_tree or false
  local parent_file = io.open(parent_path, 'r')
  local label_file
  if label_path ~= nil then label_file = io.open(label_path, 'r') end
  --if path ~= nil then label_file = io.open(label_path, 'r') end

  local count = 0
  local trees = {}
  local trees_topo = {}
  local trees_off  = {}

  while true do
    local parents = parent_file:read()
    if parents == nil then break end
    parents = parents:split(' ')
    for i, p in ipairs(parents) do
      parents[i] = tonumber(p)
    end

    local labels
    if label_file ~= nil then
      labels = label_file:read():split(' ')
      for i, l in ipairs(labels) do
        -- ignore unlabeled nodes
        if l == '#' then
          labels[i] = nil
        else
          labels[i] = tonumber(l)
        end
      end
    end
    count = count + 1
    local this_tree = tree2tree.read_tree(parents, labels, lr_tree)
    if lr_tree then
      this_tree:set_outward_preorder_index()
    else
      this_tree:set_preorder_index()
    end

    -- Pad
    local n = sents:size(1)

    local sent = sents[{count,{}}]

    local sent_eos_idx = torch.range(1,sent:nElement())[sent:eq(0)][1] - 1 -- First element that is zero, previous one is EOS

    local final_tree
    if pad_args then
      final_tree = (lr_tree)
                   and tree2tree.pad_lr_tree(this_tree,sent_eos_idx, vocab, pad_args.pad_sot,pad_args.pad_eos,pad_args.pad_leaves) -- in place
                   or  tree2tree.pad_tree(this_tree,sent_eos_idx, vocab, pad_args.pad_sot,pad_args.pad_eos,pad_args.pad_leaves)
    else
      final_tree = this_tree
    end
    -- Tensorize
    local max_tree_size = seq_len
    local n_dims_tree = (lr_tree) and 7 or 6
    local n_dims_topo = (lr_tree) and 3 or 2
    local treeT, offT,topoT  = final_tree:convert_to_tensor(max_tree_size)
    trees[count] = treeT:view(1,max_tree_size,n_dims_tree)
    trees_off[count] = offT:view(1,max_tree_size,max_tree_size)
    trees_topo[count] = topoT:view(1,max_tree_size,n_dims_topo)
  end
  parent_file:close()
  local trees_tensor     = nn.JoinTable(1):forward(trees)
  local offspring_tensor = nn.JoinTable(1):forward(trees_off)
  local topo_tensor      = nn.JoinTable(1):forward(trees_topo)
  return {trees_tensor, offspring_tensor,  topo_tensor}
end


function tree2tree.tree_from_parents(parents, labels, lr_tree, multi_root)
  local multi_root = multi_root or false
  local lr_tree = lr_tree or false -- left-right tree, default is normal tree
  local size = #parents
  local trees = {}
  if labels == nil then labels = {} end
  local root
  if multi_root then
    root = (lr_tree) and tree2tree.LRTree() or tree2tree.Tree()
    --root.idx = 1
    root.index = 1
  end
  for i = 1, size do
    if not trees[i] and parents[i] ~= -1 then
      local idx = i
      local prev = nil
      while true do
        local parent = parents[idx]
        if parent == -1 then
          break
        end
        local tree = (lr_tree) and tree2tree.LRTree() or tree2tree.Tree()
        if prev ~= nil then
          if lr_tree then
            if prev.idx > idx then
              tree:add_right_child(prev)
            else
              tree:add_left_child(prev)
            end
          else
            tree:add_child(prev)
          end
        end
        trees[idx] = tree
        tree.idx = idx
        tree.gold_label = labels[idx]
        if trees[parent] ~= nil then
          if lr_tree and (idx > trees[parent].idx) then
            trees[parent]:add_right_child(tree)
          elseif lr_tree and (idx < trees[parent].idx) then
            trees[parent]:add_left_child(tree)
          elseif (not lr_tree) then
            trees[parent]:add_child(tree)
          else
            print('Error. Shouldnt be here!!!')
          end
          break
        elseif parent == 0 then
          if multi_root then
            root:add_child(tree)
          else
            root = tree
          end
          break
        else
          prev = tree
          idx = parent
        end
      end
    end
  end

  -- index leaves (only meaningful for constituency trees)
  local leaf_idx = 1
  for i = 1, size do
    local tree = trees[i]
    if tree ~= nil and tree.num_children == 0 then
      tree.leaf_idx = leaf_idx
      leaf_idx = leaf_idx + 1
    end
  end
  return root
end

--[[

  Semantic Relatedness

--]]

function tree2tree.read_relatedness_dataset(dir, vocab, constituency)
  local dataset = {}
  dataset.vocab = vocab
  if constituency then
    dataset.ltrees = tree2tree.read_trees(dir .. 'a.cparents')
    dataset.rtrees = tree2tree.read_trees(dir .. 'b.cparents')
  else
    dataset.ltrees = tree2tree.read_trees(dir .. 'a.parents')
    dataset.rtrees = tree2tree.read_trees(dir .. 'b.parents')
  end
  dataset.lsents = tree2tree.read_sentences(dir .. 'a.toks', vocab)
  dataset.rsents = tree2tree.read_sentences(dir .. 'b.toks', vocab)
  dataset.size = #dataset.ltrees
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    dataset.ids[i] = id_file:readInt()
    dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1)
  end
  id_file:close()
  sim_file:close()
  return dataset
end


--[[
  MS COCO dataset.
   Assumes captions have already been parsed, and
]] --
function tree2tree.read_coco_dataset(dir, vocab, fine_grained, dependency)
  local dataset = {}
  dataset.vocab = vocab
  dataset.fine_grained = fine_grained
  local trees
  if dependency then
    trees = tree2tree.read_trees(dir .. 'dparents.txt', dir .. 'dlabels.txt')
  else
    trees = tree2tree.read_trees(dir .. 'parents.txt', dir .. 'labels.txt')
    for _, tree in ipairs(trees) do
      set_spans(tree)
    end
  end

  local sents = tree2tree.read_sentences(dir .. 'sents.txt', vocab)
  if not fine_grained then
    dataset.trees = {}
    dataset.sents = {}
    for i = 1, #trees do
      if trees[i].gold_label ~= 0 then
        table.insert(dataset.trees, trees[i])
        table.insert(dataset.sents, sents[i])
      end
    end
  else
    dataset.trees = trees
    dataset.sents = sents
  end

  dataset.size = #dataset.trees
  dataset.labels = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    remap_labels(dataset.trees[i], fine_grained)
    dataset.labels[i] = dataset.trees[i].gold_label
  end
  return dataset
end

function tree2tree.pad_tree(tree,eos_index,vocab, pad_sot, pad_eos, pad_leaves, multi_root)
  --[[
    Idx codes:
       1: start of tree
       2: start of offspring branch
       |sent|+1: end of offspring branch

    Multi_root: Whether native tree has multiple words stemming from root (e.g. IFTTT) or only one (eg dep parse)
  ]]--
  --local sent_size = sent:size(1)
  local pad_leaves = pad_leaves or 'half'   -- 'No', 'Half', 'Yes'. 'Half means dont pad L/R side if no child on that side'
  local pad_offspring
  pad_offspring = function(srctree, isroot)
    local padt = tree2tree.Tree()
    padt.idx = srctree.idx + OFFSET
    if srctree.num_children == 0 then
      return padt
    end
    for i=1, srctree.num_children do
      local child = pad_offspring(srctree.children[i], false)
      if child then
        child.idx = srctree.children[i].idx + OFFSET
        padt:add_child(child)
      end
    end
    if isroot == false then
      if ((pad_leaves == 'half' and srctree.num_children > 0) or pad_leaves == 'yes') then
        local left_pad  = tree2tree.Tree()
        local right_pad = tree2tree.Tree()
        left_pad.idx  = SOS_IDX
        left_pad.bdy  = 'SOS'
        padt:add_child(left_pad,'start')
        if pad_eos then
          right_pad.idx = sent_size
          right_pad.bdy = 'EOS'
          padt:add_child(right_pad,'end')
        end
      end
    end
    return padt
  end

  local padded_tree
  if pad_sot == true then
    if not multi_root then
      padded_tree = tree2tree.Tree() -- root will have a start of tree symbol
      padded_tree.idx = SOT_IDX
      padded_tree.bdy = 'SOT'
      padded_tree:add_child(pad_offspring(tree, false))
    else
      tree.idx = SOT_IDX
      padded_tree = pad_offspring(tree, false)
      padded_tree.idx = SOT_IDX
      padded_tree.bdy = 'SOT'
    end
  else
    padded_tree = pad_offspring(tree, true)
  end
  padded_tree:set_preorder_index()
  return padded_tree
end

function tree2tree.pad_lr_tree(tree,eos_idx,vocab, pad_sot, pad_eos,pad_leaves)
  --[[
    Idx codes:
       1: start of tree
       2: start of offspring branch
       |sent|+1: end of offspring branch
  ]]--
  --local sent_size = sent:size(1)
  local pad_leaves = pad_leaves or 'half'   -- 'No', 'Half', 'Yes'. 'Half means dont pad L/R side if no child on that side'

  local pad_offspring
  pad_offspring = function(srctree, isroot)
    local padt = tree2tree.LRTree()
    padt.idx = srctree.idx + OFFSET
    if srctree.num_lchildren == 0 and srctree.num_rchildren == 0 then  -- TODO: Must change this for NTP model - need to stop by token.
      return padt
    end
    for i=1, srctree.num_lchildren do
      local child = pad_offspring(srctree.lchildren[i], false)
      if child then
        child.idx = srctree.lchildren[i].idx + OFFSET
        padt:add_left_child(child)
      end
    end
    for i=1, srctree.num_rchildren do
      local child = pad_offspring(srctree.rchildren[i], false)
      if child then
        child.idx = srctree.rchildren[i].idx + OFFSET
        padt:add_right_child(child)
      end
    end
    if isroot == false then --terminal leaves didn't make it here, so ok to pad
      if ((pad_leaves == 'half' and srctree.num_lchildren > 0) or pad_leaves == 'yes') then
        local right_pad_leftbranch = tree2tree.LRTree()
        local left_pad_leftbranch = tree2tree.LRTree()
        -- We always pad SOS
        right_pad_leftbranch.idx = SOS_IDX
        right_pad_leftbranch.bdy = 'SOS'
        padt:add_left_child(right_pad_leftbranch,'start')
        -- But whether or not we pad EOS depends on model
        if pad_eos then
          left_pad_leftbranch.idx  = eos_idx
          left_pad_leftbranch.bdy  = 'EOS'
          padt:add_left_child(left_pad_leftbranch,'end')
        end
      end
      if ((pad_leaves == 'half' and srctree.num_rchildren > 0) or pad_leaves == 'yes') then
        local left_pad_rightbranch = tree2tree.LRTree()
        local right_pad_rightbranch = tree2tree.LRTree()
        left_pad_rightbranch.idx  = SOS_IDX
        left_pad_rightbranch.bdy  = 'SOS'
        padt:add_right_child(left_pad_rightbranch,'start')
        if pad_eos then
          right_pad_rightbranch.idx = eos_idx
          right_pad_rightbranch.bdy = 'EOS'
          padt:add_right_child(right_pad_rightbranch,'end')
        end
      end
    end
    return padt
  end

  local padded_tree
  if pad_sot == true then
    padded_tree = tree2tree.LRTree() -- root will have a start of tree symbol
    padded_tree.idx = SOT_IDX
    padded_tree.bdy = 'SOT'
    padded_tree:add_left_child(pad_offspring(tree, false))
  else
    padded_tree = pad_offspring(tree, true)
  end
  padded_tree:set_outward_preorder_index()
  return padded_tree
end

function pad_trees(trees, sents, vocab,args)
  local lr_tree = args.lr_tree or false
  local n = #sents
  for i=1,n do
    trees[i] = (lr_tree)
              and tree2tree.pad_lr_tree(trees[i],sents[i]:size(1), vocab, args.pad_sot,args.pad_eos,args.pad_leaves,args.multi_root) -- in place
              or  tree2tree.pad_tree(trees[i],sents[i]:size(1), vocab, args.pad_sot,args.pad_eos,args.pad_leaves, args.multi_root)
  end
end

-- This function is a for of read_parallel_dataset, adapted to read three sources instead of two
function tree2tree.read_perturbation_dataset(dir, vocab_src, vocab_tgt, args)
  local pad_sot = args.pad_sot
  local pad_eos = args.pad_eos
  local lr_tree = args.lr_tree
  --local seq_len = args.seq_len -- Seems like we shoudlt need this if there is not batching
  local max_examples = args.maxexamples
  local constituency = args.constituency
  local use_source_trees = args.use_source_trees
  local use_target_trees = args.use_target_trees
  local read_native_source = args.read_native_source
  local read_native_target = args.read_native_target
  local multi_root = args.multi_root
  local read_ids = args.read_ids or false
  local shuffle = args.shuffle
  local lowercase = args.lowercase

  -- These are global on purpose!!!
  SOT_IDX = pad_sot and 1 or nil
  SOS_IDX = pad_sot and 2 or 1
  -- EOS_IDX = will be sent_size + 1, depends on sent. Much easier to put it at the end, for logistic reasons.
  OFFSET  = pad_sot and 2 or 1

  if shuffle and max_examples then
    max_examples = nil
  end
  local dataset = {}
  local source_dataset = {}
  local target_dataset = {}
  local perturb_dataset = {}
  dataset.vocab_source = vocab_src
  dataset.vocab_target = vocab_tgt
  if use_source_trees then
    source_dataset.trees = tree2tree.read_trees{parent_path=path.join(dir,'s.parents'),lr_tree=lr_tree,pad_eos=pad_eos, maxk=max_examples,multi_root=multi_root}
  end
  target_dataset.trees = tree2tree.read_trees{parent_path=path.join(dir,'t_o.parents'),lr_tree=lr_tree,pad_eos=pad_eos, maxk=max_examples, multi_root=multi_root}
  perturb_dataset.trees = tree2tree.read_trees{parent_path=path.join(dir,'t_p.parents'),lr_tree=lr_tree,pad_eos=pad_eos, maxk=max_examples, multi_root=multi_root}


  --print(perturb_dataset.trees[10].idx)

  source_dataset.sents = tree2tree.read_sentences(path.join(dir,'s.toks'),vocab_src,{pad_sos=false,pad_eos=false,pad_sot=false,maxk=max_examples,lowercase=lowercase}) -- Maybe not necessary to pad source
  target_dataset.sents = tree2tree.read_sentences(path.join(dir,'t_o.toks'),vocab_tgt,{pad_sos=true,pad_eos=pad_eos, pad_sot=pad_sot,maxk=max_examples,lowercase=lowercase})
  perturb_dataset.sents = tree2tree.read_sentences(path.join(dir,'t_p.toks'),vocab_tgt,{pad_sos=true,pad_eos=pad_eos, pad_sot=pad_sot,maxk=max_examples,lowercase=lowercase})

  local args_treepad = {pad_eos=pad_eos,pad_sot=true,lr_tree=lr_tree,pad_leaves='half',multi_root=multi_root}
  pad_trees(target_dataset.trees, target_dataset.sents, vocab_tgt,args_treepad)
  pad_trees(perturb_dataset.trees, perturb_dataset.sents, vocab_tgt,args_treepad)

  dataset.native = {}
  if read_native_source then
    dataset.native.source = tree2tree.read_list(path.join(dir,'s.toks'),max_examples)
  end
  if read_native_target then
    dataset.native.target_original = tree2tree.read_list(path.join(dir,'t_o.toks'),max_examples)
    dataset.native.target_perturbed = tree2tree.read_list(path.join(dir,'t_p.toks'),max_examples)
  end
  if args.native_to_table then
    for k,natives in pairs(dataset.native) do
      for i=1,#natives do
        local native_str = natives[i]
        natives[i] = native_str:split(' ')
      end
    end
  end
  dataset.size = #source_dataset.sents
  dataset.source, dataset.target_original, dataset.target_perturbed = {}, {}, {}
  local map_ids_to_tds = false -- Tds causes problems when writing with csvigo
  local ids = {}
  if read_ids then
    dataset.ids = tree2tree.read_list(path.join(dir,'ids.txt'),max_examples)
    if map_ids_to_tds then
      dataset.ids = tds.Hash(dataset.ids)
    end
  end

  -- Interlace so that we can access examples by ind easily
  --if not shuffle then
  for i = 1, dataset.size do
    local src_tree = use_source_trees and source_dataset.trees[i] or nil
    local tgt_orig_tree = use_target_trees and target_dataset.trees[i] or nil
    local tgt_perturb_tree = use_target_trees and perturb_dataset.trees[i] or nil
    table.insert(dataset.source,{tree = src_tree, sent = source_dataset.sents[i]})
    table.insert(dataset.target_original,{tree = tgt_orig_tree, sent = target_dataset.sents[i]})
    table.insert(dataset.target_perturbed,{tree = tgt_perturb_tree, sent = perturb_dataset.sents[i]})
  end

  -- else
  --   if read_ids then
  --     all_ids = dataset.ids
  --   --dataset.ids = {}
  --     print('all ids', all_ids); bk()
  --   end
  --   dataset['ids'] = {}
  --   local indices = torch.randperm(dataset.size)
  --   local nexamples = args.maxexamples and math.min(args.maxexamples,dataset.size) or dataset.size
  --   for j = 1, nexamples do
  --       local i = indices[j]
  --       local src_tree = use_source_trees and source_dataset.trees[i] or nil
  --       local tgt_tree = use_target_trees and target_dataset.trees[i] or nil
  --       local perturb_tree = use_target_trees and perturb_dataset.trees[i] or nil
  --       dataset.source[j] = {tree = src_tree, sent = source_dataset.sents[i]}
  --       dataset.target_original[j] = {tree = tgt_tree, sent = target_dataset.sents[i]}
  --       dataset.target_perturbed[j] = {tree = tgt_tree, sent = perturb_dataset.sents[i]}
  --       dataset.ids[j] = all_ids and all_ids[i] or i
  --   end
  --   dataset.size = nexamples
  -- end
  source_dataset = nil
  target_dataset = nil
  perturb_dataset = nil
  return dataset
end




function tree2tree.read_parallel_tree_dataset(dir, vocab_src, vocab_tgt, args)
  local pad_sot = args.pad_sot
  local pad_eos = args.pad_eos
  local lr_tree = args.lr_tree
  --local seq_len = args.seq_len -- Seems like we shoudlt need this if there is not batching
  local max_examples = args.maxexamples
  local constituency = args.constituency
  local use_source_trees = args.use_source_trees
  local use_target_trees = args.use_target_trees
  local read_native_source = args.read_native_source
  local read_native_target = args.read_native_target
  local multi_root = args.multi_root
  local read_ids = args.read_ids or false
  local shuffle = args.shuffle
  local lowercase = args.lowercase

  -- These are global on purpose!!!
  SOT_IDX = pad_sot and 1 or nil
  SOS_IDX = pad_sot and 2 or 1
  -- EOS_IDX = will be sent_size + 1, depends on sent. Much easier to put it at the end, for logistic reasons.
  OFFSET  = pad_sot and 2 or 1

  if shuffle and max_examples then
    max_examples = nil
  end
  local dataset = {}
  local source_dataset = {}
  local target_dataset = {}
  dataset.vocab_source = vocab_src
  dataset.vocab_target = vocab_tgt
  collectgarbage() -- To free up space before readin trees
  if TRACKMEM then print('Mem usage before trees: ' ..collectgarbage('count')) end
  if constituency then
    source_dataset.trees = tree2tree.read_trees{parent_path=path.join(dir,'s.cparents'),lr_tree=lr_tree,pad_eos=pad_eos}
    target_dataset.trees = tree2tree.read_trees{parent_path=path.join(dir,'t.cparents'),lr_tree=lr_tree,pad_eos=pad_eos}
  else
    if use_source_trees then
      source_dataset.trees = tree2tree.read_trees{parent_path=path.join(dir,'s.parents'),lr_tree=lr_tree,pad_eos=pad_eos, maxk=max_examples,multi_root=multi_root}
    end
    if use_target_trees then
      target_dataset.trees = tree2tree.read_trees{parent_path=path.join(dir,'t.parents'),lr_tree=lr_tree,pad_eos=pad_eos, maxk=max_examples, multi_root=multi_root}
    end
  end
  if TRACKMEM then print('Mem usage after trees: ' ..collectgarbage('count')) end
  source_dataset.sents = tree2tree.read_sentences(path.join(dir,'s.toks'),vocab_src,{pad_sos=false,pad_eos=false,pad_sot=false,maxk=max_examples,lowercase=lowercase}) -- Maybe not necessary to pad source
  target_dataset.sents = tree2tree.read_sentences(path.join(dir,'t.toks'),vocab_tgt,{pad_sos=true,pad_eos=pad_eos, pad_sot=pad_sot,maxk=max_examples,lowercase=lowercase})
  if TRACKMEM then print('Mem usage after sentences: ' ..collectgarbage('count')) end

  if use_target_trees then
    local args_treepad = {pad_eos=pad_eos,pad_sot=true,lr_tree=lr_tree,pad_leaves='half',multi_root=multi_root}
    pad_trees(target_dataset.trees, target_dataset.sents, vocab_tgt,args_treepad)
  end

  dataset.native = {}
  if read_native_target then
    if path.exists(path.join(dir,'s.txt')) then -- Try native .txt file
      dataset.native.source = tree2tree.read_list(path.join(dir,'s.txt'),max_examples)
    else -- settle for token file
      dataset.native.source = tree2tree.read_list(path.join(dir,'s.toks'),max_examples)
    end
  end
  if read_native_target then
    if path.exists(path.join(dir,'t.txt')) then -- Try native .txt file
      dataset.native.target = tree2tree.read_list(path.join(dir,'t.txt'),max_examples)
    else -- settle for token file
      dataset.native.target = tree2tree.read_list(path.join(dir,'t.toks'),max_examples)
    end
  end
  if args.native_to_table then -- Return native sentence as tables instead of strings
    for k,natives in pairs(dataset.native) do
      for i=1,#natives do
        local native_str = natives[i]
        natives[i] = native_str:split(' ')
      end
    end
  end

  dataset.size = #source_dataset.sents
  dataset.source, dataset.target = {}, {}
  -- for i = 1, dataset.size do
  --   local src_tree = use_source_trees and source_dataset.trees[i] or nil
  --   local tgt_tree = use_target_trees and target_dataset.trees[i] or nil
  --   dataset.trees[i] = {source = src_tree, target = tgt_tree}
  --   dataset.sents[i] = {source = source_dataset.sents[i], target = target_dataset.sents[i]}
  -- end
  local map_ids_to_tds = false -- Tds causes problems when writing with csvigo
  local ids = {}
  if read_ids then
    dataset.ids = tree2tree.read_list(path.join(dir,'ids.txt'),max_examples)
    if map_ids_to_tds then
      dataset.ids = tds.Hash(dataset.ids)
    end
  end

  -- Interlace so that we can access examples by ind easily
  if not shuffle then
    for i = 1, dataset.size do
      local src_tree = use_source_trees and source_dataset.trees[i] or nil
      local tgt_tree = use_target_trees and target_dataset.trees[i] or nil
      dataset.source[i] = {tree = src_tree, sent = source_dataset.sents[i]}
      dataset.target[i] = {tree = tgt_tree, sent = target_dataset.sents[i]}
    end
  else
    if read_ids then
      all_ids = dataset.ids
    --dataset.ids = {}
      print('all ids', all_ids); bk()
    end
    dataset['ids'] = {}
    local indices = torch.randperm(dataset.size)
    local nexamples = args.maxexamples and math.min(args.maxexamples,dataset.size) or dataset.size
    for j = 1, nexamples do
        local i = indices[j]
        local src_tree = use_source_trees and source_dataset.trees[i] or nil
        local tgt_tree = use_target_trees and target_dataset.trees[i] or nil
        dataset.source[j] = {tree = src_tree, sent = source_dataset.sents[i]}
        dataset.target[j] = {tree = tgt_tree, sent = target_dataset.sents[i]}
        dataset.ids[j] = all_ids and all_ids[i] or i
    end
    dataset.size = nexamples
  end
  source_dataset = nil
  target_dataset = nil
  collectgarbage()
  return dataset
end


function tree2tree.read_parallel_tree_dataset_tensor(dir, vocab_src, vocab_tgt, pad_sot, pad_eos,lr_tree,seq_len,constituency)
  -- These are global on purpose!!!
  SOT_IDX = pad_sot and 1 or nil
  SOS_IDX = pad_sot and 2 or 1
  -- EOS_IDX = will be sent_size + 1, depends on sent. Much easier to put it at the end, for logistic reasons.
  OFFSET  = pad_sot and 2 or 1

  local max_len = seq_len  -- TODO: Do a first pass and check max length?

  local dataset = {}
  local source_dataset = {}
  local target_dataset = {}
  dataset.vocab_source = vocab_src
  dataset.vocab_target = vocab_tgt

  source_dataset.sents = tree2tree.read_sentences_to_tensor(dir .. 's.toks',
  vocab_src,{pad_sos=false,pad_eos=false,pad_sot=false,max_len=max_len,pad_from='L'}) -- Maybe not necessary to pad source
  target_dataset.sents = tree2tree.read_sentences_to_tensor(dir .. 't.toks',
  vocab_tgt,{pad_sos=true,pad_eos=pad_eos, pad_sot=pad_sot,max_len=max_len})

  local args_treepad = {pad_eos=pad_eos,pad_sot=true,lr_tree=lr_tree,pad_leaves='half'}

  args_treepad.max_len = max_len

  if constituency then
    source_dataset.trees = tree2tree.read_trees_tensor(source_dataset.sents,dir .. 's.cparents',lr_tree,seq_len,nil)
    target_dataset.trees = tree2tree.read_trees_tensor(target_dataset.sents,dir .. 't.cparents',lr_tree,seq_len,args_treepad)
  else
    source_dataset.trees = tree2tree.read_trees_tensor(source_dataset.sents,dir .. 's.parents',lr_tree,seq_len,nil)
    target_dataset.trees = tree2tree.read_trees_tensor(target_dataset.sents,dir .. 't.parents',lr_tree,seq_len,args_treepad)
  end

  dataset.size = source_dataset.sents:size(1)
  dataset.source = {
    source_dataset.trees,
    source_dataset.sents
  }

  dataset.target = {
    target_dataset.trees,
    target_dataset.sents
  }

  debugutils.dprint(1,'Sentence tensor size: ' .. torch_dims_string(source_dataset.sents))
  debugutils.dprint(1,'Tree tensor size: ' .. torch_dims_string(source_dataset.trees[1]))
  debugutils.dprint(1,'Topo tensor size: ' .. torch_dims_string(source_dataset.trees[2]))
  debugutils.dprint(1,'Off  tensor size: ' .. torch_dims_string(source_dataset.trees[3]))

  return dataset
end

function tree2tree.write_sentences(sentences, file)
  fd = io.open(file, 'w')
  for i=1,#sentences do
    fd:write(sentences[i] ..'\n')
  end
  fd:close()
end


function tree2tree.write_predictions(dataset,summary_file,raw_file,ids,extracted,stats,correct)
  if dataset == 'IFTTT' then
    tree2tree.write_IFTTT_predictions(summary_file,raw_file,ids,extracted,stats,correct)
  else
    local data_table = {
        --QId = actual_attempted,
        id              = ids,
        pred_tokens     = extracted.pred.tokens,
        pred_parents    = extracted.pred.parents,
        pred_linearized = extracted.pred.linearized,
        gold_tokens     = extracted.gold.tokens,
        gold_parents    = extracted.gold.parents,
        gold_linearized = extracted.gold.linearized,
        node_rec        = stats['node_rec'],
        edge_rec        = stats['edge_rec'],
        node_prec       = stats['node_prec'],
        edge_prec       = stats['edge_prec'],
        node_f1         = stats['node_f1'],
        edge_f1         = stats['edge_f1']
      }
    local order={'id','pred_tokens','pred_parents','pred_linearized',
                      'gold_tokens','gold_parents','gold_linearized',
                      'node_prec','node_rec','node_f1',
                      'edge_prec','edge_rec','edge_f1'}
    csvigo.save{path=summary_file,data=data_table,separator='\t',column_order = order}
  end
end

function tree2tree.write_IFTTT_predictions(summary_file,raw_file,ids,extracted,stats,correct)--predictions, gold,vocab,max_pred)
  local pred_extracted = extracted['pred']
  local gold_extracted = extracted['gold']
  local column_order, data_table
  local data_table = {
      --QId = actual_attempted,
      id                   = ids,
      pred_trigger_channel = pred_extracted['if_channel'],
      pred_action_channel  = pred_extracted['then_channel'],
      pred_trigger_fun     = pred_extracted['if_function'],
      pred_action_fun      = pred_extracted['then_function'],
      gold_trigger_channel = gold_extracted['if_channel'],
      gold_action_channel  = gold_extracted['then_channel'],
      gold_trigger_fun     = gold_extracted['if_function'],
      gold_action_fun      = gold_extracted['then_function'],
      correct_channel      = correct['channel'],
      correct_function     = correct['func'],
      args_rec             = stats['args_rec'],
      node_rec             = stats['node_rec'],
      edge_rec             = stats['edge_rec'],
      args_prec            = stats['args_prec'],
      node_prec            = stats['node_prec'],
      edge_prec            = stats['edge_prec']
    }

  local order={'id','pred_trigger_channel','pred_trigger_fun','pred_action_channel','pred_action_fun',
          'gold_trigger_channel','gold_trigger_fun','gold_action_channel','gold_action_fun',
         'correct_channel','correct_function','args_prec','args_rec','node_prec','node_rec','edge_prec','edge_rec'}

  csvigo.save{path=summary_file, data=data_table, separator='\t', column_order=order} --,column_order = order

  -- Also save raw recipes
  local recipes_table = {
    id = ids,
    predicted_recipe = pred_extracted['recipe']
  }
  local order = {'id','predicted_recipe'}
  csvigo.save{path=raw_file, data=recipes_table, separator='\t',column_order=order}
end

--[[
    Returns table of paths for saving models and predictions
]]--


function tree2tree.get_save_paths(opt)
  save_paths = {}
  -- This are declared in init
  local pred_dir       = tree2tree.predictions_dir .. '/' .. opt.dataset .. '/'
  local model_dir      = tree2tree.models_dir .. '/' .. opt.dataset .. '/'

  for i,pth in ipairs{pred_dir, model_dir} do
    if not paths.dirp(pth) then
      recursive_mkdir(pth)
    end
  end

  local encoder_string = opt.encoder:replace(',','-')
  local decoder_string = opt.decoder:replace(',','-')

  -- get model name for final model
  local file_idx = 1
  local string_id, model_save_path
  while true do
    string_id = string.format('%s_%s%s_%dD_%dL_%.2fLR_%.4eR_%dB.%d',
    encoder_string,opt.lr and 'LR' or '',  decoder_string, opt.memdim,
    opt.layers,opt.learning_rate, opt.reg, opt.batch, file_idx)
    model_save_path = path.join(model_dir, string_id .. '.th')
    if lfs.attributes(model_save_path .. '.model') == nil then
      break
    end
    file_idx = file_idx + 1
  end

  -- Collect all paths
  save_paths['models'] = model_save_path
  save_paths['checkpoints'] = path.join(model_dir, string_id)

  -- Prediction summary files have summary features and stats about predicted tree
  save_paths['preds'] = {} -- {dev = nil, test = nil}
  save_paths['preds']['dev'] = path.join(pred_dir,'dev',string_id .. '.pred_summary')
  save_paths['preds']['test'] = path.join(pred_dir,'test',string_id .. '.pred_summary')

  -- Raw prediction files only have predicted tree, no stats nor gold values
  save_paths['raw_preds'] = {}
  save_paths['raw_preds']['dev'] = path.join(pred_dir,'dev',string_id .. '.raw_preds')
  save_paths['raw_preds']['test'] = path.join(pred_dir,'test',string_id .. '.raw_preds')

  dirs_mk = {path.join(pred_dir,'dev'),path.join(pred_dir,'test')}

  for i,pth in ipairs(dirs_mk) do
    if not paths.dirp(pth) then
      recursive_mkdir(pth)
    end
  end
  return save_paths
end
