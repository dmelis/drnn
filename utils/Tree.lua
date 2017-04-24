--[[

  A basic tree structure.

--]]

local Tree = torch.class('tree2tree.Tree')

function Tree:__init(args)
  self.parent = nil
  self.num_children = 0
  self.children = {}
  self.left_brother = nil
  self.right_brother = nil
  if args then
    for k,v in pairs(args) do
      self[k] = v
    end
  end
end

-- function Tree:add_child(c,pos)
--   c.parent = self
--   c.left_brother = self.children[self.num_children]
--   if self.num_children > 0 then
--     self.children[self.num_children].right_brother = c
--   end
--   self.num_children = self.num_children + 1
--   self.children[self.num_children] = c
-- end

function Tree:add_child(c, pos)
  pos = pos or 'idx' --default is lexical behavior
  -- Want to have outward order, so need to check where to put them.
  c.parent = self
  local i = 1
  if pos == 'start' or (pos == 'idx' and c.bdy == 'SOS') then
    i = 1
  elseif pos == 'end' or (pos == 'idx' and c.bdy == 'EOS' )then
    i = self.num_children + 1
  else
    if (self.num_children > 0 and self.children[1].bdy == 'SOS') then i = 2 end -- go past start boundary if any
    while (i<= self.num_children) and (self.children[i].bdy ~= 'EOS') and (c.idx > self.children[i].idx) do
      i = i + 1
    end
  end
  table.insert(self.children, i, c)
  local left_bro = (i>1) and self.children[i-1] or nil
  if pos =='idx' then
    while (i<= self.num_children) and (c.idx < self.children[i].idx) do
      i = i + 1
    end
  elseif pos == 'start' then
    i = 1
  else --end
    i = self.num_children + 1
  end
  c.left_brother = left_bro
  if left_bro then left_bro.right_brother = c end
  local right_bro = (i<=self.num_children) and self.children[i+1] or nil
  c.right_brother = right_bro
  if right_bro then right_bro.left_brother = c end
  self.num_children = self.num_children + 1
  self._size = self._size and self._size + 1 or nil
end


function Tree:remove_child(k)
  local removed
  assert(self.num_children >= k, 'Error: removing nonexistent right child')
  removed = table.remove(self.children, k)
  if k > 1 then
    self.children[k-1].right_brother =  self.children[k]
    if k < self.num_children then
      self.children[k].left_brother =  self.children[k-1]
    end
  end
  self.num_children = self.num_children - 1
  if self._size then self._size = self._size - 1 end
  -- TODO: Must also remove it from self.children
  return removed.idx
end



-- function Tree:remove_child(k)
--   assert(self.num_children >= k, 'Error: removing nonexistent child')
--   table.remove(self.children, k)
--   if k >1 then
--     self.children[k-1].right_brother =  self.children[k]
--     if k < self.num_children then
--       self.children[k].left_brother =  self.children[k-1]
--     end
--   end
--   self.num_children = self.num_children - 1
--   if self._size then self._size = self._size - 1 end
-- end

function Tree:get_leftmost_brother()
  local brother
  if self.parent == nil then
    brother = nil
  elseif self.parent.children[1] == self then
    brother = nil
  else
    brother =  self.parent.children[1]
  end
  return brother
end


local function set_preorder_index(tree,ind)
  -- variable ind keeps track of last assigned index
  local total_add = 0
  if tree == nil then
    return 0
  end
  ind = ind + 1
  tree.index = ind
  for i = 1, tree.num_children do
    ind = set_preorder_index(tree.children[i],ind)
  end
  return ind
end

function Tree:set_preorder_index()
  set_preorder_index(self,0)
end

function Tree:size()
  if self._size ~= nil then return self._size end
  local size = 1
  for i = 1, self.num_children do
    size = size + self.children[i]:size()
  end
  self._size = size
  return size
end

function Tree:size(force)
  local force = force or false -- if true forces recomputing sizes: ignores previous sizings
  if (not force and self._size ~= nil) then return self._size end
  local size = 1
  for i = 1, self.num_children do
    size = size + self.children[i]:size()
  end
  self._size = size
  return size
end

function Tree:depth()
  local depth = 0
  if self.num_children > 0 then
    for i = 1, self.num_children do
      local child_depth = self.children[i]:depth()
      if child_depth > depth then
        depth = child_depth
      end
    end
    depth = depth + 1
  end
  return depth
end


function table.contains(table, element)
  for _, value in pairs(table) do
    if value == element then
      return true
    end
  end
  return false
end


function Tree:copy(tree, copytype, removepad)
  local copytype = copytype or "full"
  local removepad = removepad or false
  local new_tree = tree2tree.Tree()
  local attributes_nocopy   = {"parent", "left_brother", "right_brother", "num_children", "children"}
  local attributes_skeleton = {"index", "idx", "bdy","label"}

  local tree = (tree.bdy == 'SOT' and removepad) and tree.children[1] or tree

  for k, v in pairs(tree) do
    if ((copytype == "full") and (not table.contains(attributes_nocopy,k))) or
       ((copytype == "skeleton") and (table.contains(attributes_skeleton,k))) then
      new_tree[k] = v
    end
  end
  if tree.num_children > 0 then
    for i = 1, tree.num_children do
      if (not removepad) or (not tree.children[i].bdy) then
        local c = tree2tree.Tree():copy(tree.children[i], copytype, removepad)
        new_tree:add_child(c)
      end
    end
  end
  return new_tree
end

local function breadth_first(tree, nodes)
  if tree == nil then
    return
  end
  if tree.parent == nil then
    table.insert(nodes, tree)
  end
  for i = 1, tree.num_children do
    table.insert(nodes,tree.children[i])
  end
  for i = 1, tree.num_children do
    breadth_first(tree.children[i], nodes)
  end
end

function Tree:breadth_first()
  local nodes = {}
  breadth_first(self, nodes)
  return nodes
end

-- Lexical order is depth first pre_order (between left and right sets of children)

local function lexical_order(tree, nodes)
  if tree == nil then
    return
  end
  if (tree.index == 1 and tree.bdy == 'SOT') then    -- This must be SOT root for padded tree
    lexical_order(tree.children[1], nodes)
    table.insert(nodes, tree)
  else
    table.insert(nodes, tree)
    for i = 1,tree.num_children do
      lexical_order(tree.children[i], nodes)
    end
  end
end

function Tree:lexical_order()
  local nodes = {}
  lexical_order(self, nodes)
  return nodes
end


-- TODO: Consolidate the two below with additional arg to print or return?
function Tree:print_lexical_order(node_labels, prune_padding)
  local prune_padding = prune_padding or true
  local nodes = self:lexical_order()
  local lstring = ""
  for i=1,#nodes do
    -- if (not is_padded) or (nodes[i].idx > 2) and (nodes[i].idx < #node_labels) then
    local symbol = (node_labels) and node_labels[nodes[i].idx] or nodes[i].idx
    lstring = lstring .. symbol .. " "
    --end
  end
  print(lstring)
end

function Tree:totable_lexical_order(node_labels, tokens_to_ignore)
  assert(not tokens_to_ignore or (tokens_to_ignore and (node_labels ~= nil)),
  "Nodes to table. Must provide node labels if providing tokens to ignote")
  local prune_padding = prune_padding or true
  local nodes = self:lexical_order()
  local t = {}
  --print('To ignore: ',tokens_to_ignore)
  for i=1,#nodes do
    -- if (not is_padded) or (nodes[i].idx > 2) and (nodes[i].idx < #node_labels) then
    local node = nodes[i]
    local symbol = (node_labels) and node_labels[node.idx] or node.idx
    if (tokens_to_ignore ~= nil) and equals_any(symbol,tokens_to_ignore) then
      --print('Skipping node' .. node_labels[nodes[i].idx])
    else
      t[#t+1]=symbol
    end
  end
  return t
end


-- Converts tree to original parents/tokens format
function Tree:convert_to_original(node_labels)
  --local prune_padding = prune_padding or true
  local nodes = self:lexical_order()
  local p = {}
  local t = {}
  local idxs = {}
  for i=1,#nodes do
    -- if (not is_padded) or (nodes[i].idx > 2) and (nodes[i].idx < #node_labels) then
    -- if (tokens_to_ignore) and equals_any(node_labels[nodes[i].idx],tokens_to_ignore) then
      --print('Skipping node' .. node_labels[nodes[i].idx])
    if (nodes[i].bdy == nil)  then
      idxs[nodes[i].index] = #t + 1
      p[#t+1]= nodes[i].parent and idxs[nodes[i].parent.index] or 0 -- Assign index zero to root's parent
      t[#t+1]= node_labels[nodes[i].idx]
    end
  end
  return t,p
end

-- NOTE: Only counts actual edges (parent-children). Counting both parent-child,
-- brother-brother seems like doulbe counting
-- If tokens_to_ignore is provided, so must be node_lables
function Tree:edges_table_lexical_order(node_labels, tokens_to_ignore)
  local prune_padding = prune_padding or true
  local nodes = self:lexical_order()
  assert(not tokens_to_ignore or (tokens_to_ignore and (node_labels ~= nil)),
  "Edges to table. Must provide node labels if providing tokens to ignote")
  local edges = {}
  for i=1,#nodes do
    -- if (not is_padded) or (nodes[i].idx > 2) and (nodes[i].idx < #node_labels) then
    local node = nodes[i]
    for j=1,node.num_children do
      local src_valid, dest_valid = true, true
      local symbol_p = (node_labels) and node_labels[node.idx] or node.idx
      local symbol_b = (node_labels) and node_labels[node.children[j].idx] or node.children[j].idx
      if (node.bdy) or ((tokens_to_ignore ~= nil) and (equals_any(symbol_p,tokens_to_ignore) or
      equals_any(symbol_b,tokens_to_ignore))) then
        --print('Skipping edge',symbol_p, symbol_b)
      else
        edges[#edges+1] = symbol_p .. "->" -- .. symbol_b   -- TODO: Decide whether to match full (n,e,n) or just (n,e)
      end
    end
  end
  return edges
end


local function depth_first_preorder(tree, nodes)
  if tree == nil then
    return
  end
  table.insert(nodes, tree)
  for i = 1, tree.num_children do
    depth_first_preorder(tree.children[i], nodes)
  end
end

function Tree:depth_first_preorder()
  local nodes = {}
  depth_first_preorder(self, nodes)
  return nodes
end

function Tree:prune_leftmost_leaves()
  -- Decided not to do the sift idx thing.
  --local shiftidx = shiftidx or true -- shift the idx of all nodes to compensate for <s> missing now.
  local nodes = {}
  self._size = nil  -- If not, memory about size from prevous sizing is kept, no good.
  depth_first_preorder(self, nodes)
  for i =1, #nodes do
    if nodes[i].num_children > 1 then
      nodes[i]:remove_child(1)
    end
    nodes[i]._size = nil
  end
  --self:set_preorder_index()  --NO! Want to keep original index
end

function Tree:prune_padding_leaves(type,labels)
  local type = type or 'both' -- options: {SOS,EOS,both}
  local labels = labels or nil
  self._size = nil  -- If not, memory about size from prevous sizing is kept, no good.
  -- if self.idx == 1 then --there's root padding, remove
  --   lexical_order(self.lchildren[1], all_nodes)
  -- else
  local unpadded_tree = (self.bdy == 'SOT') and self:copy(self.children[1],"full") or self:copy(self,"full")-- get rid of top
  local nodes = unpadded_tree:depth_first_preorder()
  for i = 1, #nodes do
    local ls,le,rs,re, prev_ls
    if (not nodes[i].bdy and nodes[i].num_children > 1) then
      prev_ls = nodes[i].children[1].idx
      if type == 'SOS' or type =='both' then
        ls = nodes[i]:remove_child(1)
      end
      if type == 'EOS' or type =='both' then
        le = nodes[i]:remove_child(nodes[i].num_children)
      end
    end
    if labels then -- Im dbugging
      print(nodes[i].index,labels[nodes[i].idx],'L',ls,le,'. R:', rs, re, prev_ls)
    end
    nodes[i]._size = nil
  end
  return unpadded_tree
end

function Tree:print_preorder(node_labels,vocab)
  -- Node labels must be in same order as .idx indices of tree.
  local node_labels = node_labels or nil
  local nodes = {}
  printf('%4s %4s %4s %15s %15s %15s %15s\n','ind','idx','bdy','word','parent','left_bro','right_bro')
  printf('-----------------------------------------------\n')
  depth_first_preorder(self,nodes)
  for i=1,#nodes do
    local node = nodes[i]
    local parent_word
    if node.parent == nil then
      parent_word = 'METAROOT'
    else
      parent_word = (node_labels) and node_labels[node.parent.idx] or '--'
    end
    --print(node_labels[node.idx])
    local this_word = (node_labels) and node_labels[node.idx] or '--'
    local left_word = (node_labels and node.left_brother) and node_labels[node.left_brother.idx] or '--'
    local right_word = (node_labels and node.right_brother) and node_labels[node.right_brother.idx] or '--'
    if node_labels and vocab ~= nil then
      this_word = (this_word ~= '--') and vocab:token(this_word) or this_word
      left_word = (left_word ~= '--') and vocab:token(left_word) or left_word
      right_word = (right_word ~= '--') and vocab:token(right_word) or right_word
      parent_word = (parent_word ~= '--' and parent_word ~= 'METAROOT') and vocab:token(parent_word) or parent_word
    end
    local idx = tostring(node.idx) or '--'
    local index = tostring(node.index) or '--'
    local bdy = node.bdy and tostring(node.bdy) or '--'
    printf('%4s %4s %4s %15s %15s %15s %15s\n',index,idx,bdy,this_word, parent_word, left_word, right_word)
  end
  printf('-----------------------------------------------\n')
end

function Tree:print_offspring_layers(node_labels)
  -- Node labels must be in same order as .idx indices of tree.
  local nodes = {}
  printf('%4s %4s %10s %10s\n','ind','idx','word','children')
  printf('-------------------------------\n')
  breadth_first(self,nodes)
  for i=1,#nodes do
    local node = nodes[i]
    if (node.bdy) then goto continue end
    local word = node_labels[node.idx]
    local children_tokens = {}
    for i = 1, node.num_children do
      children_tokens[i] = node_labels[node.children[i].idx]
    end
    local children_string = table.concat(children_tokens,' ')
    printf('%4d %4d %10s  ->  %s\n',node.index,node.idx,word,children_string)
    ::continue::
  end
  printf('-------------------------------\n')
end

-- This as a property of the tree, so that we don't have to recompute
-- it for subsequent modules. Upon new calls of this method, simply retrieve fwd_traversal
function Tree:getTraversal()
  -- NOTE: This was giving me the wrong order in traversal. Not sure why yet.
  -- local all_nodes, nodes  =  {}, {}
  -- breadth_first(self,all_nodes)
  local all_nodes, nodes  =  self:breadth_first(), {}
  local k = 1
  for i,node in pairs(all_nodes) do
    if (node.index ~= 1) and (node.bdy ~= 'SOS') then
      local prev, next
      if (node.parent ~= nil) then
        prev = node.left_brother
        next = node.right_brother
      elseif node.index == 2 then
        -- This is the root, has no side, so prev and next are nil
        prev, next = nil, nil
      else
        print("Error. I should not be here!. Node index/idx: ", node.index, node.idx)
      end
      local prev_indices = (prev) and {index = prev.index, idx =  prev.idx,bdy=prev.bdy} or nil
      local next_indices= (next) and {index = next.index, idx =  next.idx, bdy=next.bdy} or nil
      local tuple = {node, prev_indices, next_indices, node.side}
      nodes[k] = tuple
      k = k + 1
    end
  end
  self.fwd_traversal = nodes
  return nodes, k
end

-- Attaches provided labels to nodes. Labels must be either a tensor or a table,
-- indexed by the node.idx values. Thus it has to be the same size as nodes in the tree.
function Tree:labelNodes(labels)
  local nodes = {}
  depth_first_preorder(self,nodes)
  for i, node in ipairs(nodes) do
    if node.idx then
      if labels:size(1) < node.idx then
        print(labels,node.idx)
        bk('Label size does not match tree size')
      end
      node.label = labels[node.idx]
      --print(labels[node.idx])
    end
  end
end


function Tree:get_node_byidx(idx)
  local nodes = {}
  local interest_node
  depth_first_preorder(self,nodes)
  for i, node in ipairs(nodes) do
    --print(node.idx)
    if node.idx and node.idx == idx then
      if interest_node ~= nil then
        print('More than one node with that idx!! will return first one found')
      else
        interest_node = node
      end
    end
  end
  if interest_node == nil then
    print('No node found with that idx!')
  end
  return interest_node
end


-- The following two come from lang2logic's Tree function. Convert tree to
-- lambda-calculus expression
-- function Tree:to_lambda_string()
--   local r_list = {}
--   for i = 1, self.num_children do
--     if class.istype(self.children[i], 'tree2tree.Tree') then
--       table.insert(r_list, '( '..self.children[i]:to_lambda_string()..' )')
--     else
--       table.insert(r_list, tostring(self.children[i]))
--     end
--   end
--   return (' '):join(r_list)
-- end

--
-- function Tree:to_lambda_string()
--   local r_list = {}
--   table.insert(r_list, self.idx)
--   for i = 1, self.num_children do
--     if class.istype(self.children[i], 'tree2tree.Tree') then
--       table.insert(r_list, '( '..self.children[i]:to_lambda_string()..' )')
--     else
--       table.insert(r_list, tostring(self.children[i]))
--     end
--   end
--   return (' '):join(r_list)
-- end
--
-- function Tree:to_lambda_list(form_manager)
--   local r_list = {}
--   for i = 1, self.num_children do
--     if class.istype(self.children[i], 'tree2tree.Tree') then
--       table.insert(r_list, form_manager:get_symbol_idx('('))
--       local cl = self.children[i]:to_lambda_list(form_manager)
--       for k = 1, #cl do table.insert(r_list, cl[k]) end
--       table.insert(r_list, form_manager:get_symbol_idx(')'))
--     else
--       table.insert(r_list, self.children[i])
--     end
--   end
--   return r_list
-- end
