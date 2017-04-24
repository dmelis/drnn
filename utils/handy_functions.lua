
function script_path()
   local str = debug.getinfo(2, "S").source:sub(2)
   return str:match("(.*/)")
end

function get_parent_dir(levels)
  pth = script_path()
  for i=1,levels do
    pth = path.dirname(pth)
  end
  return pth
end


-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

-- Pearson correlation
function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

function compute_accuracy(pred,gold)
  -- Pred, gold should be Int Tensors
  local answered = pred:size(1)
  local asked = gold:size(1)
  if (answered < asked) then
    print('Warning in compute accuracy: not all questions answered')
  end
  local total = pred:size(1)
  local correct = torch.eq(pred,gold:narrow(1,1,total)):sum()
  return correct/total, correct, total
end


-- Hacky checkpoint function

function bk(message)
  if message then print(message) end
  os.exit()
end


function prompt_continue()
  local ans = ''
  while ans ~= 'y' and ans ~= 'n' do
    print("Continue? (y/n)")
    local re = io.read()
    ans =re:lower()
    if ans == "y" then
      return 1
    elseif ans == "n" then
        bk("Exiting.")
    end
  end
end

-- Compatibility: Lua-5.0
-- function string.split(str, delim, maxNb)
--     -- Eliminate bad cases...
--     if string.find(str, delim) == nil then
--         return { str }
--     end
--     if maxNb == nil or maxNb < 1 then
--         maxNb = 0    -- No limit
--     end
--     local result = {}
--     local pat = "(.-)" .. delim .. "()"
--     local nb = 0
--     local lastPos
--     for part, pos in string.gfind(str, pat) do
--         nb = nb + 1
--         result[nb] = part
--         lastPos = pos
--         if nb == maxNb then break end
--     end
--     -- Handle the last field
--     if nb ~= maxNb then
--         result[nb + 1] = string.sub(str, lastPos)
--     end
--     return result
-- end

-- function split(pString, pPattern)
--    local Table = {}  -- NOTE: use {n = 0} in Lua-5.0
--    local fpat = "(.-)" .. pPattern
--    local last_end = 1
--    local s, e, cap = pString:find(fpat, 1)
--    while s do
--       if s ~= 1 or cap ~= "" then
--      table.insert(Table,cap)
--       end
--       last_end = e+1
--       s, e, cap = pString:find(fpat, last_end)
--    end
--    if last_end <= #pString then
--       cap = pString:sub(last_end)
--       table.insert(Table, cap)
--    end
--    return Table
-- end

function equals_any(s,t)
  bool_any = false
  for k,v in ipairs(t) do
    bool_any = bool_any or (s == v)
  end
  return bool_any
end

function get_keys(table)
  local keyset={}
  local n=0
  for k,v in pairs(table) do
    n=n+1
    keyset[n]=k
  end
  return n,keyset
end

function table.clone(org)
  return {table.unpack(org)}
end


function table.slice(tbl, first, last, step)
  local sliced = {}
  for i = first or 1, last or #tbl, step or 1 do
    sliced[#sliced+1] = tbl[i]
  end
  return sliced
end

function recursiveCuda(table)
  for k,v in ipairs(table) do
    if torch.type(v) == 'table' then
      recursiveCuda(v)
    elseif hasattr(v, "cuda") then
      v:cuda()
    -- end
    end
  end
end

function recursiveAssign(table)
  for k,v in ipairs(table) do
    if torch.type(v) == 'table' then
      recursiveCuda(v)
    elseif hasattr(v, "cuda") then
      v:cuda()
    -- end
    end
  end
end



function gennorm_print(input, label)
  if torch.type(input) == 'table' then
    printf('{ ')
    for k,v in ipairs(input) do
      printf('|| %s || = %8.2f, ', label,v:norm())
    end
    printf(' }\n')
  elseif torch.isTensor(input) then
    printf('|| %s || = %8.2f\n', label, input:norm())
  else
    printf('|| %s || = %s\n', label,'???')
  end
end



function gensize_print(input, label)
  if torch.type(input) == 'table' then
    printf('{ ')
    for k,v in ipairs(input) do
      printf('dim(%s) = %8s, ', label,torch_dims_string(v))
    end
    printf(' }\n')
  elseif torch.isTensor(input) then
    printf('dim(%s) = %8s\n', label, torch_dims_string(input))
  else
    printf('dim(%s) = %s\n', label,'???')
  end
end

-- Needs require 'paths'
-- This will fail if basename has any dots. !!
function recursive_mkdir(filepath)
  local ext = paths.extname(filepath)
  local top_dir = ext and paths.dirname(filepath) or filepath
  local dir = top_dir
  local parts = {}
  local max_depth = 10
  local depth = 0
  while not paths.dirp(dir) and (depth < max_depth) do
    print(dir)
    sub_dir = paths.dirname(dir)
    part = paths.basename(dir)
    table.insert(parts, part)
    dir = sub_dir
    depth = depth + 1
  end
  for i=#parts,1,-1 do
    super_dir = paths.concat(dir, parts[i])
    paths.mkdir(super_dir)
    dir = super_dir
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end



function TableConcat(t1,t2)
  local t1 = (torch.type(t1) == 'table') and t1 or {t1}
  local t2 = (torch.type(t2) == 'table') and t2 or {t2}
  for i=1,#t2 do
    t1[#t1+1] = t2[i]
  end
  return t1
end


function recursiveSelect(t,dim,ind)
  local output = {}
  for k,v in pairs(t) do
    if torch.isTensor(v) then
      output[k] = v:select(dim,ind)
    elseif torch.type(v) == 'table' then
      output[k] = recursiveSelect(v,dim,ind)
    else
      print("error, unexpected type")
      bk()
    end
  end
  return output
end

function table_to_string(t)
  local out = ""
  for k,v in pairs(t) do
    out = out .. tostring(v)
    out = (k == n) and out or (out .. ", ")
  end
  return out
end

function tensor_to_string(T)
  local n = T:size(1)
  local t = nn.SplitTable(1,1):forward(T)
  local out = ""
  for k,v in pairs(t) do
    out = out .. tostring(v)
    out = (k == n) and out or (out .. ", ")
  end
  return out
end

function torch_dims_string(t)
  local size_str = t:size(1)
  for k = 2,t:nDimension() do
    size_str = size_str .. 'x' .. t:size(k)
  end
  return size_str
end

function string_tensor_table(table)
  local size = '{'
  for k,v in pairs(table) do
    if torch.type(v) == 'table' then
      key_size = string_tensor_table(v)
    else
      --key_size = tostring(v:size(1))
      key_size = torch_dims_string(v)
    end
    size = size .. key_size
    if k < #table then size = size .. ',' end
  end
  size = size .. '}'
  return size
end

function string_norms_table(table)
  if table == nil then return 'NIL' end
  local size = '{'
  for k,v in pairs(table) do
    if torch.type(v) == 'table' then
      key_size = string_norms_table(v)
    else
      --key_size = tostring(v:size(1))
      key_size = v:norm()
    end
    size = size .. key_size
    if k < #table then size = size .. ',' end
  end
  size = size .. '}'
  return size
end

function tensor_table_clamp(t,a,b)
  for k,v in pairs(t) do
    v:clamp(a,b)
  end
end
