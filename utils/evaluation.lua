
match_debug_level = 1

-- This two come from lang2logical repo code
function is_all_same(c1,c2)
  if #c1 == #c2 then
    local all_same = true
    for j = 1, #c1 do
      if c1[j] ~= c2[j] then
        all_same = false
        break
      end
    end
    return all_same
  else
    return false
  end
end

function display_accuracy(dataset, acc_type, stats)
  if dataset == 'IFTTT' then
    printf('Accuracy: %4.2f channel, %4.2f function, %4.2f chan+func\n',
      100*stats['channel-acc'],100*stats['func-acc'],100*stats['chan-func-acc'])
    printf('Args accuracy: %4.2f prec, %4.2f rec, %4.2f Macro-F1\n', 100*stats['args_prec'],100*stats['args_rec'],100*stats['args-macro-f1'])
  elseif dataset == 'BABYMT' and acc_type == 'BLEU' then
    printf('BLEU score: %4.2f\n', stats['BLEU'])
  end
  if acc_type == 'tree' or dataset == 'IFTTT' then
    printf('Tree node accuracy: %4.2f prec, %4.2f rec, %4.2f Macro-F1\n', 100*stats['node_prec'],100*stats['node_rec'],100*stats['node_macro_f1'])
    printf('Tree edge accuracy: %4.2f prec, %4.2f rec, %4.2f Macro-F1\n', 100*stats['edge_prec'],100*stats['edge_rec'],100*stats['edge_macro_f1'])
  end
end


function os.capture(cmd, raw)
  local f = assert(io.popen(cmd, 'r'))
  local s = assert(f:read('*a'))
  f:close()
  if raw then return s end
  s = string.gsub(s, '^%s+', '')
  s = string.gsub(s, '%s+$', '')
  s = string.gsub(s, '[\n\r]+', ' ')
  return s
end

function compute_BLEU_score(predicteds, refs, type)
  local type = type or 'macro'
  -- Write data
  local ref_file = '/tmp/ref_wmt'
  local pred_file = '/tmp/predicted_wmt'
  tree2tree.write_sentences(predicteds,pred_file)
  tree2tree.write_sentences(refs,ref_file)
  -- Call shell script to python
  local res_file = '/tmp/results.txt'
  local cmd = 'perl utils/multi-bleu.perl ' .. ref_file .. ' < ' .. pred_file
  local out_str = os.capture(cmd)
  local out_list = out_str:split(',')
  local bleu = tonumber(out_list[1]:split(' = ')[2])
  return bleu
end


function compute_BLEU_accuracy(predicteds, golds, vocab_tgt, explicit_cues, explicit_root)
  -- Will compute exact tree accuracy
  if #predicteds ~= #golds then
    print(string.format("LENGTH: #predicted_list(%d) ~= #golds(%d)", #predicteds,#golds))
  end
  local all_extracted = {gold = {},pred = {}}
  local len = math.min(#predicteds, #golds)
  local c = 0
  local all_stats, globals = {}, {}
  for i = 1, len do
    local pred_sent_table =  torch.type(predicteds[i].sent) == 'table' and predicteds[i].sent or predicteds[i].sent:totable()
    local gold_sent_table =  torch.type(golds[i].sent) == 'table' and golds[i].sent or golds[i].sent:totable()
    local mapped_pred_sent = vocab_tgt:reverse_map(pred_sent_table)
    local mapped_gold_sent = vocab_tgt:reverse_map(gold_sent_table)
    local pred_tree, gold_tree
    if has_SOS(predicteds[i].tree) then
      pred_tree = predicteds[i].tree:prune_padding_leaves('SOS')
    else
      pred_tree = predicteds[i].tree
    end
    if has_SOS(golds[i].tree) then
    --if gold_AST.children[1].bdy == 'SOS' or gold_AST.children[1].children[1].bdy == 'SOS' then
      gold_tree = golds[i].tree:prune_padding_leaves('SOS')
    else
      gold_tree = golds[i].tree
    end
    local pred_list = pred_tree:totable_lexical_order(mapped_pred_sent)
    local gold_list = gold_tree:totable_lexical_order(mapped_gold_sent)
    local pred_lin_string = (' '):join(pred_list)
    local gold_lin_string = (' '):join(gold_list)
    table.insert(all_extracted['pred'],pred_lin_string)
    table.insert(all_extracted['gold'],gold_lin_string)
  end
  local BLEU = compute_BLEU_score(all_extracted['pred'],all_extracted['gold'], type)

  -- Compute globals
  local globals = {BLEU = BLEU}
  debugutils.dprint(match_debug_level,'BLEU Score: ' .. globals['BLEU'])
  return globals, all_extracted
end

-- Centralized accuracy computation function.
function compute_accuracy(dataset,type,predicteds, golds, vocab_tgt, explicit_cues, explicit_root)
  if dataset == 'IFTTT' then
    return compute_IFTTT_accuracy(predicteds, golds, vocab_tgt, explicit_cues, explicit_root)
  elseif dataset == 'BABYMT' and type == 'BLEU' then
    return compute_BLEU_accuracy(predicteds, golds, vocab_tgt, explicit_cues, explicit_root)
  elseif dataset == 'synth' then
    return compute_tree_accuracy(dataset,predicteds, golds, vocab_tgt, explicit_cues, explicit_root)
  else
    bk('Unrecognized dataset/ accuracy type')
  end
end

-- Returns three dictionaries:
-- stats:  values for various accuracy statistics
-- pred_extracted: extracted values of predicted tree (e.g. linearized sentence, or channel and func for iFTTT)
-- gold_extracted: same for gold

function compute_tree_accuracy(dataset,predicteds, golds, vocab_tgt, explicit_cues, explicit_root)
  -- Will compute exact tree accuracy
  if #predicteds ~= #golds then
    print(string.format("LENGTH: #predicted_list(%d) ~= #golds(%d)", #predicteds,#golds))
  end
  local all_extracted = {gold = {linearized={}, tokens={}, parents={}},
                         pred = {linearized={}, tokens={}, parents={}}}
  local len = math.min(#predicteds, #golds)
  local c = 0
  local all_stats, globals = {}, {}
  local per_example_stats = {node_prec = {},  node_rec= {}, node_f1 = {}, edge_prec = {}, edge_rec = {}, edge_f1={}}
  for i = 1, len do
    local pred_sent_table =  torch.type(predicteds[i].sent) == 'table' and predicteds[i].sent or predicteds[i].sent:totable()
    local gold_sent_table =  torch.type(golds[i].sent) == 'table' and golds[i].sent or golds[i].sent:totable()
    local mapped_pred_sent = vocab_tgt:reverse_map(pred_sent_table)
    local mapped_gold_sent = vocab_tgt:reverse_map(gold_sent_table)

    -- Prepare trees
    local pred_tree, gold_tree
    if has_SOS(predicteds[i].tree) then
      pred_tree = predicteds[i].tree:prune_padding_leaves('SOS')
    else
      pred_tree = predicteds[i].tree
    end
    if has_SOS(golds[i].tree) then
      gold_tree = golds[i].tree:prune_padding_leaves('SOS')
    else
      gold_tree = golds[i].tree
    end

    -- Trees are already pruned, so there shouldnt be need to pass ignore tokens to totable methods
    local pred_list = pred_tree:totable_lexical_order(mapped_pred_sent)
    local gold_list = gold_tree:totable_lexical_order(mapped_gold_sent)
    local pred_lin_string = (' '):join(pred_list)
    local gold_lin_string = (' '):join(gold_list)

    local pred_tokens, pred_parents = pred_tree:convert_to_original(mapped_pred_sent)
    local pred_tokens_str = (' '):join(pred_tokens)
    local pred_parents_str = (' '):join(pred_parents)

    local gold_tokens, gold_parents = gold_tree:convert_to_original(mapped_gold_sent)
    local gold_tokens_str = (' '):join(gold_tokens)
    local gold_parents_str = (' '):join(gold_parents)
    table.insert(all_extracted['pred']['linearized'],pred_lin_string)
    table.insert(all_extracted['gold']['linearized'],gold_lin_string)
    table.insert(all_extracted['pred']['tokens'],pred_tokens_str)
    table.insert(all_extracted['gold']['tokens'],gold_tokens_str)
    table.insert(all_extracted['pred']['parents'],pred_parents_str)
    table.insert(all_extracted['gold']['parents'],gold_parents_str)

    -- Soft matching
    local tokens_ignore = {vocab_tgt.unk_index,vocab_tgt.start_index, vocab_tgt.root_index} -- TODO: If adding root index here, is it still necessary to prune padding leaves above?

    local node_stats, edge_stats =
    tree_precision_recall({tree = pred_tree, sent = pred_sent_table},
                          {tree = gold_tree, sent = gold_sent_table}, tokens_ignore)
    local stats = {}
    stats['node_prec'] = node_stats[1] --tree_prec
    stats['node_rec']  = node_stats[2] -- tree_rec
    stats['node_f1']   = node_stats[3] -- tree_f1
    stats['edge_prec'] = edge_stats[1]
    stats['edge_rec']  = edge_stats[2]
    stats['edge_f1']   = edge_stats[3]

    for k,v in pairs(stats) do
      table.insert(per_example_stats[k],stats[k])
    end

    -- Hard matching
    if is_all_same(pred_list, gold_list) then
      c = c + 1
      debugutils.dprintf(1,'Correct:   %s || %s\n', pred_lin_string, gold_lin_string)
    else
      debugutils.dprintf(1,'Incorrect: %s || %s\n',pred_lin_string, gold_lin_string)
      debugutils.dprint(2,'Gold')
      debugutils.dprint(2,'Token Indexes: ' .. table_to_string(gold_sent_table))
      debugutils.dprint(2,'idxs:' .. table_to_string(golds[i].tree:totable_lexical_order()))
      debugutils.dprint(2,'Pred')
      debugutils.dprint(2,'Token indexes: ' .. table_to_string(predicteds[i].sent))
      debugutils.dprint(2,'Idxs:' .. table_to_string(predicteds[i].tree:totable_lexical_order()))
      debugutils.dprint(2,'---')
    end

    for k,v in pairs(stats) do
      if all_stats[k] == nil then
        all_stats[k] = 0
      end
      all_stats[k] = all_stats[k] + v
    end
  end

  -- Compute globals
  local globals = {}
  for k,v in pairs(all_stats) do
    globals[k] = v/len
  end
  globals['node_macro_f1'] = (globals['node_prec']+globals['node_rec'] >0 ) and
    2*(globals['node_prec']*globals['node_rec'])/(globals['node_prec']+globals['node_rec'])
    or 0
  globals['edge_macro_f1'] = (globals['edge_prec']+globals['edge_rec'] >0 ) and
    2*(globals['edge_prec']*globals['edge_rec'])/(globals['edge_prec']+globals['edge_rec'])
    or 0
  debugutils.dprint(match_debug_level,'Mean Precision tree (nodes): ' .. globals['node_prec'])
  debugutils.dprint(match_debug_level,'Mean Recall tree (nodes): ' .. globals['node_rec'])
  debugutils.dprint(match_debug_level,'Macro-Av F1 tree (nodes): ' .. globals['node_macro_f1'])
  debugutils.dprint(match_debug_level,'Mean Precision tree (edges): ' .. globals['edge_prec'])
  debugutils.dprint(match_debug_level,'Mean Recall tree (edges): ' .. globals['edge_rec'])
  debugutils.dprint(match_debug_level,'Macro-Av F1 tree (edges): ' .. globals['edge_macro_f1'])
  return globals, all_extracted, bool_correct, per_example_stats
end

function compute_list_accuracy(predicted_list, gold_list)
  if #predicted_list ~= #gold_list then
    print(string.format("LENGTH: #predicted_list(%d) ~= #gold_list(%d)", #predicted_list, #gold_list))
  end

  local len = math.min(#predicted_list, #gold_list)
  print(len)
  local c = 0
  for i = 1, len do
    if is_all_same(predicted_list[i], gold_list[i]) then
      c = c + 1
      debugutils.dprintf(2,'Correct:   %s || %s\n',predicted_list[i], gold_list[i])
    else
      debugutils.dprintf(2,'Incorrect: %s || %s\n',predicted_list[i], gold_list[i])
    end
  end

  return c / len
end


--#############################################################################
--#########################      IFTTT Functions      #########################
--#############################################################################

function has_SOS(tree)
  if torch.type(tree) == 'tree2tree.Tree' then
    if tree.children[1] and tree.children[1].bdy == 'SOS' then
      return true
    elseif tree.children[1] and tree.children[1].num_children > 0 and
       tree.children[1].children[1].bdy == 'SOS' then
         return true
    end
  elseif torch.type(tree) == 'tree2tree.LRTree' then
    -- FIXME: Might need to consider all four cases in elseif (L/R,L/L/, etc)
    -- First check where root child is
    local first_non_root = tree.lchildren[1] or tree.rchildren[1]
    if (first_non_root and first_non_root.bdy == 'SOS') then
      return true
    elseif ((first_non_root and first_non_root.num_rchildren > 0) and
       (first_non_root.rchildren[1].bdy == 'SOS')) then
         return true
    end
  end
  return false
end

-- Assumes tree is SOS pad-free!
-- Extracts: channel, function, args for each branch
function extract_recipe_nocues(AST,sentence, explicit_root)
  local nodes  =  {channel = {IF = nil, THEN = nil}, func = {IF = nil, THEN= nil }, args = {IF = {}, THEN= {}}, params = {IF = {}, THEN= {}}}
  local token_idxs   = {channel = {IF = nil, THEN = nil}, func = {IF = nil, THEN= nil }, args = {IF = {}, THEN= {}}, params = {IF = {}, THEN= {}}}
  local tokens = {channel = {IF = nil, THEN = nil}, func = {IF = nil, THEN= nil }, args = {IF = {}, THEN= {}}, params = {IF = {}, THEN= {}}}
  local root_node = AST --(explicit_root) and AST.children[1] or AST
  local branches = {'IF','THEN'}
  if root_node.num_children > 0 then -- First node is <t>, second is ROOT when having explicit roots
    for i=1,math.min(root_node.num_children,2) do
      if root_node.children[i].bdy then
        print('Shoult not have SOS padding at this point')
        bk('Aborting in extract_recipe_nocues')
      end
      nodes['channel'][branches[i]] = root_node.children[i]
    end
    for j,branch in ipairs(branches) do
      if nodes['channel'][branch] then
        token_idxs['channel'][branch]  = nodes['channel'][branch].idx
        tokens['channel'][branch] = sentence[token_idxs['channel'][branch]]
        if nodes['channel'][branch].num_children > 0 then
          nodes['func'][branch] = nodes['channel'][branch].children[1]
          token_idxs['func'][branch] = nodes['func'][branch].idx
          tokens['func'][branch] = sentence[token_idxs['func'][branch]]
          for i,n in ipairs(nodes['func'][branch].children) do
            nodes['args'][branch][i] = nodes['func'][branch].children[i]
            token_idxs['args'][branch][i] = nodes['args'][branch][i].idx
            tokens['args'][branch][i] = sentence[token_idxs['args'][branch][i]]
            if nodes['args'][branch][i].num_children > 0 then -- Means this arg has param
              nodes['params'][branch][i] = nodes['args'][branch][i].children[1]
              token_idxs['params'][branch][i] = nodes['params'][branch][i].idx
              tokens['params'][branch][i] = sentence[token_idxs['params'][branch][i]]
            end
          end
        end
      else
        debugutils.dprint(match_debug_level+1, 'No node found for' .. branch)
      end
    end
  end
  return tokens
end

-------------------------------------
-- Converts an IFTTT tree to a recipe in the syntax of Quirk et al.
-- @AST object of class tree
-- @sentence table of tokens associated to nodes
-- @vocab Vocab object corresponding to output (target domain)
-- TODO: Move this to tree_coversion ?
-------------------------------------
function tree_to_IFTTT_recipe(AST, sentence, vocab)
  local extracted = extract_recipe_nocues(AST,sentence, true)
  local token_strings = {}
  local branch_strings = {}
  for i,b in ipairs({'IF','THEN'}) do
    token_strings = {}
    for j,l in ipairs({'channel','func'}) do
      token_strings[l] = extracted[l][b] and vocab:token(extracted[l][b]) or 'NONE'
    end
    local args_params_string = ''
    for k,v in pairs(extracted['args'][b]) do
      local arg_string = v and vocab:token(v) or 'NONE'
      local param_string = 'NONE'
      if extracted['params'][b][k] then
        param_string = v and vocab:token(extracted['params'][b][k]) or 'NONE'
      end
      args_params_string = args_params_string .. ' (' .. arg_string .. ' ('  .. param_string .. ')' .. ')'
    end
    token_strings['params'] = args_params_string
    local prefix = (b == 'IF') and '(IF) (TRIGGER (' or '(THEN) (ACTION ('
    branch_strings[b] = prefix .. token_strings['channel'] .. ') (FUNC (' ..
    token_strings['func'] .. ') (PARAMS' .. token_strings['params'] .. ')))'
  end
  recipe_string = '(ROOT ' .. branch_strings['IF'] .. ' ' .. branch_strings['THEN'] .. ')'
  return recipe_string
end


function append_to_extracted(base,extra)
  table.insert(base['if_channel'],extra['channel']['IF'])
  table.insert(base['then_channel'],extra['channel']['THEN'])
  table.insert(base['if_function'],extra['func']['IF'])
  table.insert(base['then_function'],extra['func']['THEN'])
end

-- Optionally returns extracted channel/function tokens for both prediction and gold
function compute_IFTTT_accuracy(predicteds, golds, vocab, explicit_root, explicit_cues)
  -- NOTE: explicit_root not used yet
  if #predicteds ~= #golds then
    print(string.format("LENGTH: #predicted_list(%d) ~= #gold_list(%d)", #predicteds, #golds))
  end
  local all_extracted = {
    gold = {if_channel = {}, then_channel = {}, if_function = {}, then_function={}, recipe = {}},
    pred = {if_channel = {}, then_channel = {}, if_function = {}, then_function={}, recipe = {}},
  }
  local len = math.min(#predicteds, #golds)
  local bool_correct = {channel = {}, func = {}}
  local per_example_stats = {args_prec = {} , args_rec = {}, node_prec = {},  node_rec= {}, edge_prec = {}, edge_rec = {}}
  local total_sums = {}
  for i = 1, len do
    local stats, bool_correct_example, extracted =
    IFTTT_AST_match(predicteds[i].tree, predicteds[i].sent,
                    golds[i].tree, golds[i].sent,
                    vocab, explicit_root,explicit_cues)
    append_to_extracted(all_extracted['pred'],extracted['pred'])
    append_to_extracted(all_extracted['gold'],extracted['gold'])

    table.insert(bool_correct['channel'], bool_correct_example['channel'] and 1 or 0)
    table.insert(bool_correct['func'], bool_correct_example['func'] and 1 or 0)
    table.insert(bool_correct['func'], bool_correct_example['func'] and 1 or 0)
    for k,v in pairs(per_example_stats) do
      table.insert(per_example_stats[k],stats[k])
    end

    local pruned_pred_tree = predicteds[i].tree:prune_padding_leaves('SOS')
    local pred_recipe = tree_to_IFTTT_recipe(pruned_pred_tree, predicteds[i].sent, vocab)
    table.insert(all_extracted['pred']['recipe'],pred_recipe)

    for k,v in pairs(stats) do
      if total_sums[k] == nil then
        total_sums[k] = 0
      end
      if equals_any(k, {'channel-acc','func-acc','chan-func-acc'}) then
        v = v and 1 or 0
      end
      total_sums[k] = total_sums[k] + v
    end
  end
  -- Compute globals
  local globals = {}
  for k,v in pairs(total_sums) do
    globals[k] = v/len
  end

  globals['args-macro-f1'] = (globals['args_prec']+globals['args_rec'] >0) and
    2*(globals['args_prec']*globals['args_rec'])/(globals['args_prec']+globals['args_rec'])
    or 0
  globals['node_macro_f1'] = (globals['node_prec']+globals['node_rec'] >0 ) and
    2*(globals['node_prec']*globals['node_rec'])/(globals['node_prec']+globals['node_rec'])
    or 0
  globals['edge_macro_f1'] = (globals['edge_prec']+globals['edge_rec'] >0 ) and
    2*(globals['edge_prec']*globals['edge_rec'])/(globals['edge_prec']+globals['edge_rec'])
    or 0

  debugutils.dprint(match_debug_level,'Accuracy channel: '.. 100*globals['channel-acc'] .. '%')
  debugutils.dprint(match_debug_level,'Accuracy function: ' .. 100*globals['func-acc'] .. '%')
  debugutils.dprint(match_debug_level,'Accuracy channel + function: ' .. 100*globals['chan-func-acc'] .. '%')
  debugutils.dprint(match_debug_level,'Mean Precision args: ' .. globals['args_prec'])
  debugutils.dprint(match_debug_level,'Mean Recall args: ' .. globals['args_rec'])
  debugutils.dprint(match_debug_level,'Macro-Av F1 args: ' .. globals['args-macro-f1'])
  debugutils.dprint(match_debug_level,'Mean Precision tree: ' .. globals['node_prec'])
  debugutils.dprint(match_debug_level,'Mean Recall nodes: ' .. globals['node_rec'])
  debugutils.dprint(match_debug_level,'Macro-Av F1 nodes: ' .. globals['node_macro_f1'])
  debugutils.dprint(match_debug_level,'Mean Precision edges: ' .. globals['edge_prec'])
  debugutils.dprint(match_debug_level,'Mean Recall edges: ' .. globals['edge_rec'])
  debugutils.dprint(match_debug_level,'Macro-Av F1 edges: ' .. globals['edge_macro_f1'])

  return globals, all_extracted, bool_correct, per_example_stats
end

function IFTTT_AST_match(predicted_AST, predicted_sent, gold_AST, gold_sent,vocab,explicit_root,explicit_cues)
  if explicit_cues then
    return IFTTT_AST_match_cues(predicted_AST, predicted_sent, gold_AST, gold_sent, explicit_root, vocab)
  else
    return IFTTT_AST_match_nocues(predicted_AST, predicted_sent, gold_AST, gold_sent, explicit_root, vocab)
  end
end

function IFTTT_AST_match_cues(predicted_AST, predicted_sent, gold_AST, gold_sent, vocab, explicit_root)
  --print(predicted_AST, predicted_sent, gold_AST, gold_sent); bk()
  local pred_values, gold_values = {},{}
  -- Check that TRIGGER channel is correct
  local vocab_indexes = {}
  local pred_token_idxs, gold_token_idxs = {}, {}
  local pred_nodes, gold_nodes = {}, {}
  local pred_channel_idx, gold_channel_idx = {}, {}
  local bool_agreement = {channel = {},  func = {}}
  local level_names = {
      channel = {IF = 'TRIGGER', THEN = 'ACTION'},
      func = {IF = 'FUNC', THEN = 'FUNC'},
    }
  -- Find indices of cue tokens in vocab
  for i,level in ipairs({'channel','func'}) do
    vocab_indexes[level] = {}
    for j, branch in ipairs({'IF', 'THEN'}) do
      local vocab_index = vocab:index(level_names[level][branch])
      vocab_indexes[level][branch] = vocab_index
      --print(level,branch,level_names[level][branch], vocab_index)
    end
  end
  -- Find cue tokens indices in sentences
    -- Find idx in predicted_sent
  if torch.isTensor(predicted_sent) then
    predicted_sent = nn.SplitTable(1):forward(predicted_sent)
  end
  local pred_token_idxs = {channel = {IF = nil, THEN = nil}, func = {IF = nil, THEN= nil }}
  for l, sent_idx in pairs(predicted_sent) do
    local assigned = false
    for i,level in ipairs({'channel','func'}) do
      for j, branch in ipairs({'IF', 'THEN'}) do
        if sent_idx == vocab_indexes[level][branch] and (not assigned) and (pred_token_idxs[level][branch] == nil) then
           -- The second condition is is a hack, to void getting 2 matches for function.
           -- This way, we rely on the fact that we know IF FUNC comes before THEN FUNC
          pred_token_idxs[level][branch] = l
          assigned = true
        end
      end
    end
  end
  debugutils.dprint(match_debug_level+1,'channel',pred_token_idxs['channel'])
  debugutils.dprint(match_debug_level+1,'func',pred_token_idxs['func'])


  local gold_token_idxs = {channel = {IF = nil, THEN = nil}, func = {IF = nil, THEN= nil }}
  for idx=1,gold_sent:size(1) do
    local sent_idx = gold_sent[idx]
    local assigned = false
    for i,level in ipairs({'channel','func'}) do
      for j, branch in ipairs({'IF', 'THEN'}) do
        if sent_idx == vocab_indexes[level][branch] and (not assigned) and (gold_token_idxs[level][branch] == nil) then
           -- The second condition is is a hack, to void getting 2 matches for function.
           -- This way, we rely on the fact that we know IF FUNC comes before THEN FUNC
          gold_token_idxs[level][branch] = idx
          assigned = true
        end
      end
    end
  end
  debugutils.dprint(match_debug_level+1,'channel',gold_token_idxs['channel'])
  debugutils.dprint(match_debug_level+1,'func',gold_token_idxs['func'])

  -- Get tokens
  local pred_tokens = {channel = {IF = nil, THEN = nil}, func = {IF = nil, THEN= nil }}
  local gold_tokens = {channel = {IF = nil, THEN = nil}, func = {IF = nil, THEN= nil }}

  debugutils.dprint(match_debug_level,'--------------------------------------------------')
  debugutils.dprintf(match_debug_level,'%10s %5s %25s %25s\n','Level','Branch','Candidate','Reference')
  for i,level in ipairs({'channel','func'}) do
    for j, branch in ipairs({'IF', 'THEN'}) do
      local pred_branch_root_node, gold_branch_root_node
      if pred_token_idxs[level][branch] ~= nil then
        pred_branch_root_node = predicted_AST:get_node_byidx(pred_token_idxs[level][branch])
        if pred_branch_root_node.children[1] then
          pred_tokens[level][branch]  = predicted_sent[pred_branch_root_node.children[1].idx]
        end
      end
      if gold_token_idxs[level][branch] ~= nil then
        gold_branch_root_node  = gold_AST:get_node_byidx(gold_token_idxs[level][branch])
        gold_tokens[level][branch]   = gold_sent[gold_branch_root_node.children[1].idx]
      end
      debugutils.dprintf(match_debug_level,'%10s %5s %25s %25s\n',level,branch,
                pred_tokens[level][branch] and vocab:token(pred_tokens[level][branch]) or 'NA',
                gold_tokens[level][branch]  and vocab:token(gold_tokens[level][branch]) or 'NA')
    end
  end
  debugutils.dprint(match_debug_level,'--------------------------------------------------')


  local bool_agreement = {}
  for i,k in ipairs({'channel','func'}) do
    bool_agreement[k] = (pred_tokens[k]['IF'] ~= nil) and
                                (pred_tokens[k]['THEN'] ~= nil) and
                                (pred_tokens[k]['IF'] == gold_tokens[k]['IF']) and
                                (pred_tokens[k]['THEN'] == gold_tokens[k]['THEN'])
  end
  debugutils.dprint(match_debug_level+1,bool_agreement)
  return bool_agreement
end


-- For trees. Optional argument remove is a list of token indices to remvoe
function tree_precision_recall(pred,gold, tokens_to_ignore)
  local tokens_pred = pred.tree:totable_lexical_order(pred.sent, tokens_to_ignore)
  local tokens_gold = gold.tree:totable_lexical_order(gold.sent, tokens_to_ignore)

  local edges_pred = pred.tree:edges_table_lexical_order(pred.sent, tokens_to_ignore)
  local edges_gold = gold.tree:edges_table_lexical_order(gold.sent, tokens_to_ignore)

  local node_stats = precision_recall(tokens_pred, tokens_gold)
  local edge_stats = precision_recall(edges_pred, edges_gold)
  return node_stats, edge_stats
end

-- For sets
-- Note that when there is repetition of tokens, need to decide what to do.
-- For now, I'm crossing off elements from pred when matched, so as to not
-- count a retrieved element as relevant for more than one token in gold.
-- An alternative would be to uniq both lists
function precision_recall(pred, gold)
  local rel, ret, relevant_pred = #pred, #gold, 0
  local remain_pred = table.clone(pred)
  for i,a in ipairs(gold) do
    --rel = rel + 1
    matched = false
    for j,b in ipairs(remain_pred) do
      if (not matched) and (a == b) then
        matched = true
        table.remove(remain_pred,j)
      end
    end
    if matched then
      relevant_pred = relevant_pred + 1
    end
  end

  if ret == 0 and rel == 0 then
    return {1,1,1}
  elseif ret == 0 then -- Pred is empty, gold is not. Full precision, zero recall.
    return {1,0,0}
  elseif rel == 0 then -- Gold is empty, prediction is not. Zero precision, full recall
    return {0,1,0}
  end
  -- If we made it here, neither ret nor rel are emtpy
  local precision = relevant_pred/ret
  local recall = relevant_pred/rel
  --assert(precision <= 1, 'precision >1' ..
  if precision > 1 then
    print(relevant_pred, ret,remain_pred, pred, gold)
    bk('Prec >1')
  elseif recall >1 then
    bk('Recall >1')
  end
  local F1 = (precision + recall > 0) and 2*(precision*recall)/(precision+recall) or 0
  return {precision, recall, F1}
end


function IFTTT_AST_match_nocues(predicted_AST, predicted_sent, gold_AST, gold_sent, explicit_root, vocab)
  --print(predicted_AST, predicted_sent, gold_AST, gold_sent); bk()
  local bool_agreement = {channel = {},  func = {}}
  if torch.isTensor(predicted_sent) then
    predicted_sent = nn.SplitTable(1):forward(predicted_sent)
  end
  --predicted_AST:print_preorder(predicted_sent, vocab)
  if has_SOS(predicted_AST) then
    predicted_AST = predicted_AST:prune_padding_leaves('SOS')
  end
  if gold_AST.children[1].bdy == 'SOS' or gold_AST.children[1].children[1].bdy == 'SOS' then
    gold_AST = gold_AST:prune_padding_leaves('SOS')
  end
  extracted_pred  = extract_recipe_nocues(predicted_AST,predicted_sent, explicit_root)
  extracted_gold  = extract_recipe_nocues(gold_AST,gold_sent, explicit_root)
  debugutils.debug_IFTTT_matching(match_debug_level, extracted_pred, extracted_gold, vocab)

  local bool_agreement = {}  -- For columns in output file
  local stats = {}
  for i,k in ipairs({'channel','func'}) do
    stats[k..'-acc'] = (extracted_pred[k]['IF'] ~= nil) and
                                (extracted_pred[k]['THEN'] ~= nil) and
                                (extracted_pred[k]['IF'] == extracted_gold[k]['IF']) and
                                (extracted_pred[k]['THEN'] == extracted_gold[k]['THEN'])
    bool_agreement[k] = stats[k..'-acc']
  end
  stats['chan-func-acc'] = (stats['channel-acc'] and stats['func-acc'])

  -- Agreement is boolan for channel and function, but it's prec/rec for multi children layers, e.g. args
  -- NEW: stats is a dictionary of stats
  local args_stats = precision_recall(extracted_pred['args']['IF'],extracted_gold['args']['IF'])
  stats['args_prec'] = args_stats[1]
  stats['args_rec']  = args_stats[2]
  stats['args-f1']   = args_stats[3]

  -- Full tree score
  --local tokens_ignore = {vocab:index('ROOT'),vocab.unk_index,vocab.start_index}
  local tokens_ignore = {vocab.unk_index,vocab.start_index, vocab.root_index} -- TODO: If adding root index here, is it still necessary to prune padding leaves above?

  local node_stats, edge_stats =
  tree_precision_recall({tree = predicted_AST, sent = predicted_sent},
                        {tree = gold_AST, sent = gold_sent}, tokens_ignore)
  stats['node_prec'] = node_stats[1] --tree_prec
  stats['node_rec']  = node_stats[2] -- tree_rec
  stats['tree-f1']   = node_stats[3] -- tree_f1
  stats['edge_prec'] = edge_stats[1]
  stats['edge_rec']  = edge_stats[2]
  stats['edges-f1']   = edge_stats[3]

  debugutils.dprint(match_debug_level+1,stats)

  -- Convert to words before returning
  if vocab then
    for i,k in ipairs({'channel','func'}) do
      for j,l in ipairs({'IF','THEN'}) do
        extracted_pred[k][l] = extracted_pred[k][l] and vocab:token(extracted_pred[k][l]) or 'NA'
        extracted_gold[k][l] = extracted_gold[k][l] and vocab:token(extracted_gold[k][l]) or 'NA'
      end
    end
  end
  return stats, bool_agreement, {pred = extracted_pred, gold = extracted_gold}
end

-- ============ WRAPPER FUNCTIONS FOR PREDICTION =============  --
-- Used in the evaluation Notebooks

-- These two are modified from onmt/translate.lua
local function reportScore_onmt(name, scoreTotal, wordsTotal)
    printf("%s AVG SCORE: %.4f, %s PPL: %.4f\n\n",
            name,scoreTotal / wordsTotal,
            name,math.exp(-scoreTotal/wordsTotal))
end

-- If prediction from table input, entries of tables must be table of tokens
function onmt_predict(source,target,opt)
    local from_file, withGoldScore
    local srcReader, tgtReader -- Only used for file reading
    if torch.type(source) == 'string' and path.exists(source) then
        print("Predicting from file.." .. source)
        srcReader = onmt.utils.FileReader.new(source)
        withGoldScore = (target and target:len() > 0)
        if withGoldScore then
            tgtReader = onmt.utils.FileReader.new(target)
        end
        from_file = true
    else
        print("Predicting from table.")
        from_file = false
        withGoldScore = (target and #target > 0)
    end
    local srcBatch = {}
    local srcWordsBatch = {}
    local srcFeaturesBatch = {}
    if withGoldScore then
        tgtBatch = {}
        tgtWordsBatch = {}
        tgtFeaturesBatch = {}
    end
    local sentId = 1
    local batchId = 1
    local predScoreTotal = 0
    local predWordsTotal = 0
    local goldScoreTotal = 0
    local goldWordsTotal = 0

    local translator = onmt.translate.Translator.new(opt)
    local predictions = {}
    while true do
        local srcTokens, tgtTokens
        if from_file then
            srcTokens = srcReader:next()
            if withGoldScore then
              tgtTokens = tgtReader:next()
            end
        else
            srcTokens = source[sentId]
            if withGoldScore then
                tgtTokens = target[sentId]
            end
        end
        if srcTokens ~= nil then
          sentId = sentId + 1
          local srcWords, _ = onmt.utils.Features.extract(srcTokens) -- Second output is feature, not using features for now
          table.insert(srcBatch, srcTokens)
          table.insert(srcWordsBatch, srcWords)
          if withGoldScore then
            local tgtWords, _ = onmt.utils.Features.extract(tgtTokens)
            table.insert(tgtBatch, tgtTokens)
            table.insert(tgtWordsBatch, tgtWords)
          end
        elseif #srcBatch == 0 then
          break
        end
        if srcTokens == nil or #srcBatch == opt.batch_size then
            local predBatch, info = translator:translate(srcWordsBatch, srcFeaturesBatch,
                                                       tgtWordsBatch, tgtFeaturesBatch)
            for b = 1, #predBatch do
                local pred_summary   = {}
                pred_summary['src']  = table.concat(srcBatch[b], " ")
                pred_summary['pred'] = table.concat(predBatch[b], " ")
                --print(pred_summary['src'],pred_summary['pred'])
                predScoreTotal = predScoreTotal + info[b].score
                predWordsTotal = predWordsTotal + #predBatch[b]
                pred_summary['score'] = info[b].score
                pred_summary['goldScore'] = info[b].goldScore
                if withGoldScore then
                    goldScoreTotal = goldScoreTotal + info[b].goldScore
                    goldWordsTotal = goldWordsTotal + #tgtBatch[b]
                    pred_summary['tgt'] = table.concat(tgtBatch[b], " ")
                end
                table.insert(predictions,pred_summary)
            end
            if srcTokens == nil then
                break
            end
            batchId = batchId + 1
            srcBatch = {}
            srcWordsBatch = {}
            srcFeaturesBatch = {}
            if withGoldScore then
                tgtBatch = {}
                tgtWordsBatch = {}
                tgtFeaturesBatch = {}
            end
            collectgarbage()
        end -- process batch
    end -- while
    reportScore_onmt('PRED', predScoreTotal, predWordsTotal)
    if withGoldScore then
        reportScore_onmt('GOLD', goldScoreTotal, goldWordsTotal)
    end
    translator = nil
    return predictions
end


-- Directly from file not supported
function drnn_predict(dataset,opt, compute_score)
    local from_file, withGoldScore
    local srcReader, tgtReader
    local compute_score = compute_score or false

    local srcBatch = {}
    local srcWordsBatch = {}
    local srcFeaturesBatch = {}
    if withGoldScore then
        tgtBatch = {}
        tgtWordsBatch = {}
        tgtFeaturesBatch = {}
    end
    local sentId = 1

    -- These are global in DRNN. TODO: Add them as self.attributes in DRNN.
    MAX_OFFSPRING  = opt.max_offspring or 4
    MAX_TREE_DEPTH = opt.max_depth     or 5

    -- First load Seq2Tree DRNN to get its vocab
    model, config = tree2tree.Tree2Tree_ED.load(opt.model)
    model:predicting()

    if opt.tau_a then
      model.decoder.prediction_module.tau_a = opt.tau_a
    end
    if opt.tau_f then
      model.decoder.prediction_module.tau_f = opt.tau_f
    end

    -- Possible sampling regimes:
    --   - both topo and label deterministic (max for both)
    --   -
    local deterministic_topo = opt.deterministic_topo or false
    model.decoder.prediction_module.sample_mode_prediction = deterministic_topo and 'max' or 'sample'

    local deterministic_label =  opt.deterministic_label or false
    -- Hack: Adding Sampler now will prevent model from creating its own    FIXME: Pass as opt to tree_sampling?
    if deterministic_label then
      model.decoder.core.sampler = tree2tree.Sampler({sample_mode = 'max'})
    else
      local temp = opt.temperature or 1
      model.decoder.core.sampler = tree2tree.Sampler({sample_mode = 'temp', temperature=temp})
    end
    local num_samples = opt.num_samples or 10

    local vocab_src = config.source_vocab
    local vocab_tgt = config.target_vocab
    local predictions = {}
    for i=1,dataset.size do
      print('Predicting example number..'..i)
      local enc_in, dec_in, dec_gold = model:prepare_inputs(dataset.source[i],dataset.target[i])
      local best_logprob = - math.huge
      local pred_tree, pred_sent, pred_logprob
      for k=1,num_samples do
        local temp_pred_tree, temp_pred_sent, temp_pred_logprob = model:tree_sampling('BABYMT',enc_in, dec_gold, 0)
        debugutils.dprintf(2,'Sample number %i, score %8.2f',k,temp_pred_logprob)
        if temp_pred_logprob > best_logprob then
          pred_tree = temp_pred_tree
          pred_sent = temp_pred_sent
          pred_logprob = temp_pred_logprob
          best_logprob = temp_pred_logprob
          debugutils.dprint(2,'keeping sample, best logprob is now' .. temp_pred_logprob)
        end
        model:forget()
      end
      collectgarbage()
      pred_tree = pred_tree:prune_padding_leaves('SOS')
      local pred_str = table.concat(pred_tree:totable_lexical_order(vocab_tgt:reverse_map(pred_sent)), " ")
      local pred_summary   = {}
      pred_summary['src']  = table.concat(dataset.native.source[i]," ") --table.concat(vocab_src:reverse_map(torch.totable(dataset.source[i].sent)), " ")
      pred_summary['pred'] = pred_str --vocab_tgt:reverse_map(table.concat(pred_sent, " ")
      --predScoreTotal = predScoreTotal + info[b].score
      --predWordsTotal = predWordsTotal + #predBatch[b]
      if compute_score then
          pred_summary['score'] =  model:compute_output_LL(a)
          pred_summary['goldScore'] = nil
      end
      pred_summary['tgt'] = table.concat(dataset.native.target[i]," ")
      table.insert(predictions,pred_summary)
    end
    collectgarbage()
    model = nil
    return predictions
end
