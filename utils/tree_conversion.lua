--[[
    Types of trees:
      -- ntTree Tree with non-terminals

]]--

-- Tree with non-terminals to logical form

function ntTree2logical(tree,tostring,node_labels, vocab)
  local tostring = tostring or false
  local node_labels = node_labels or nil  -- Only needed in top most
  local vocab = vocab or nil              -- Only needed in top most
  local parens_root = false               -- Whether to include parethesis around children of root
  local r_list = {}
  if tree.num_children > 0 then
    if tree.idx  ~= 3 or parens_root then -- Don't put parethesis around top
      table.insert(r_list, '(')
    end
    for i = 1, tree.num_children do
      table.insert(r_list, ntTree2logical(tree.children[i],tostring,node_labels,vocab))
    end
    if tree.idx  ~= 3 or parens_root then -- Don't put parethesis around top
      table.insert(r_list, ')')
    end
  else
    if node_labels and vocab then
      local symbol = vocab:token(node_labels[tree.idx])
      if symbol ~= '<n>' then
        table.insert(r_list, symbol)
      end
    elseif node_labels then
      table.insert(r_list, node_labels[tree.idx])
    else
      table.insert(r_list, tree.idx)
    end
  end
  if tostring then
    return (' '):join(r_list)
  end
  return r_list
end
