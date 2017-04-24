--[[

  Training script for trainin and testing encoder-decoder tree method.

  Parses args.
  Loads word embeddings
  Calls train on every epoch.
  Keeps track of progress, early stopping.

  Example usage:

    th main.lua -e gru -d tree,gru,DTP -lr

    th main.lua -s fr -t en -e gru -d tree,gru,DTP -lr -clip 5 -n 10 -cropv 1000

    th main.lua -source fr -target en -encoder gru -decoder tree,gru,DTP -lr
    -clip 5 -epochs 10 -cropv 1000 -batch 3 -embedding rand

--]]

require 'init'

-------------------------------------------------------------------------------
--------------------   Input arguments and options  ---------------------------
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training script for question/relation scoring')
cmd:text()
cmd:text('Options')

-- Global settings
cmd:option('-verbose', true, 'maximum length of relation chain to consider')
cmd:option('-tensor_mode',false,'use tensorized version')
cmd:option('-dataset','NA','code of dataset to use')
cmd:option('-source', 'en', 'source language')
cmd:option('-target', 'en', 'target language')
cmd:option('-maxexamples',-1, 'Use only k examples')
cmd:option('-maxpred',-1, 'Use only k examples in predicition dataset')
cmd:option('-prediction',false,'if true compute prediction accuracy in addition to training loss')
cmd:option('-prediction_dataset','','dataset to compute prediction scores on')
cmd:option('-prediction_folds','dev','folds to compute prediction scores on, separated by comma')

-- Data representation
cmd:option('-shuffle',false,'randomize order of examples') -- Mostly for selecting different subsets when maxexamples is set
cmd:option('-seq_len',50,'sequennce length')
cmd:option('-pad_eos',false,'Pad end of string')
cmd:option('-cropv', -1, 'Crops vocab to keep only top [VALUE] most frequent words')

-- Model
cmd:option('-encoder','lstm','Encoder type: [gru,lstm,cs-tree-lstm,tree-gru]')
cmd:option('-decoder','lstm','Decoder type: [gru,lstm,cs-tree-lstm,tree-gru]')
cmd:option('-criterion','seq','Loss function criterion: [sentence-xent,tree-...]')
cmd:option('-reg',1e-4,'Regularization strength')
cmd:option('-clip', 5, 'Clip gradients at specified value')
cmd:option('-learning_rate',0.05,'Learning rate')
cmd:option('-layers',1,'Number of layers (ignored for Tree-LSTM)')
cmd:option('-memdim',150,'Hidden memory dimension')
cmd:option('-embedding','glove','pretrained embeddings to use [glove,rand]')
cmd:option('-lr', false, 'distinguish between left and right children')

-- Training
cmd:option('-dev_loss_every',1,'Compute dev loss every k iterations')
cmd:option('-dev_pred_every',1,'Compute dev prediction every k iterations')
cmd:option('-select_criterion','prediction','Criterion to select best model (prediction or loss)')
cmd:option('-batch', 10, 'Batch size')
cmd:option('-epochs', 20, 'number of training epochs')
cmd:option('-early_stop',-1,'Number of epochs without progress after which training is stopped. Default: -1 (no early stopping)')
cmd:option('-train_scores',false,'compute training scores')

-- Options for IFTTT
cmd:option('-explicit_cues', false, 'Use explicit nodes for cues in IFTTT (e.g. TRIGGER and ACTION) ')
cmd:option('-explicit_root', true, 'Use explicit root for iFTTT (e.g. TRIGGER and ACTION) ')

-- Saving
cmd:option('-nosave', false, 'Wont save model')
cmd:option('-append_results', false, 'Append prediction results to combined file')
cmd:option('-logfile', 'NA', 'Append prediction results to combined file')
cmd:option('-predictions_file', 'NA', 'File name for saving predictions')
cmd:option('-model_file', 'NA', 'File name for saving trained model')
cmd:option('-checkpoint_every', 1, 'Save model every k epochs')

-- Debugging
cmd:option('-debug',0,'debugging level (0 for no debugging)')
cmd:option('-debugdata',false,'use debug dataset')
cmd:option('-test_generation', false, 'Generate example every epoch')
cmd:option('-noprompt',false, 'do not prompt for continue')
cmd:option('-seed',-1, 'do not prompt for continue')

cmd:text()


local opt = cmd:parse(arg)

Debug = opt.debug -- Make it global, easier than passing around.

header('Tree to tree encoder-decoder model for '..opt.dataset)

if opt.seed ~= -1 then
  torch.manualSeed(opt.seed)
end
opt.cropv = opt.cropv ~=  -1 and opt.cropv or nil
local verbose = opt.verbose

-- Implicit arguments
opt.maxpred = opt.maxpred ~= -1 and opt.maxpred or nil
opt.early_stop = opt.early_stop ~= -1 and opt.early_stop or nil
opt.maxexamples = (opt.maxexamples > 0) and opt.maxexamples or nil

if opt.prediction and (opt.prediction_dataset == '') then
  opt.prediction_dataset = opt.dataset -- ie self
elseif (opt.prediction_dataset ~= '') then
  opt.prediction = true
  opt.prediction_dataset = opt.prediction_dataset == 'self' and opt.dataset or opt.prediction_dataset
-- elseif opt.prediction_folds ~= '' then
--   opt.prediction = true
--   opt.prediction_dataset = opt.prediction_dataset == 'self' and opt.dataset or opt.prediction_dataset
-- end
end
if opt.prediction then
  opt.prediction_folds = opt.prediction_folds:split(',')
end

if opt.select_criterion == 'prediction' and not opt.prediction then
  bk("Error! Can't do select criterion prediction without computing prediction!")
end

local dataset, data_dir, vocab_file_src, vocab_file_tgt, source_lang, target_lang
local initialize_composed, acc_type
if opt.dataset ~= 'NA' then
  dataset = opt.dataset
  data_dir = tree2tree.data_dir .. '/' .. dataset .. '/'
  vocab_file_src = path.join(data_dir,'vocab_source.txt')
  vocab_file_tgt = path.join(data_dir,'vocab_target.txt')
  source_lang = 'en'
  target_lang = 'en'
  read_native_target = (opt.dataset == 'IFTTT')
  disjoint_vocabs = true
  initialize_composed = (opt.dataset == 'IFTTT')
  acc_type = 'tree'
  lowercase = false
elseif opt.source ~= 'NA' and opt.target ~= 'NA' then
  source_lang, target_lang = opt.source, opt.target
  local langpair = source_lang ..'-'.. target_lang
  --opt.dataset = langpair
  data_dir = tree2tree.data_dir .. '/baby_wmt/' .. source_lang .. '-' .. target_lang .. '/'
  vocab_file_src = (langpair == 'en-en') and data_dir .. 'vocab-cased.txt'
                or data_dir .. source_lang .. '_vocab.txt'
  vocab_file_tgt = (langpair == 'en-en') and data_dir .. 'vocab-cased.txt'
                or data_dir .. target_lang .. '_vocab.txt'
  initialize_composed = false
  opt.dataset = 'BABYMT'
  acc_type = 'BLEU'
  lowercase = true
else
  print('Have to specifiy either dataset or lang pair!')
  bk()
end
-- directory containing dataset files
local lr_tree  = opt.lr
local pad_sot = (string.find(opt.decoder,'tree') ~= nil)

 -- Override EOS choice - for seq decoders or NTP/STP we always need it
local pad_eos = (string.find(opt.decoder,'DTP') == nil) and true or opt.pad_eos
local use_source_trees = (string.find(opt.encoder,'tree') ~= nil)
local use_target_trees = (string.find(opt.decoder,'tree') ~= nil)

-- load vocab
print('loading vocab...')
local vocab_src = tree2tree.Vocab(vocab_file_src, opt.cropv)
local vocab_tgt = tree2tree.Vocab(vocab_file_tgt, opt.cropv)

if opt.dataset == 'IFTTT' then
  -- Reagardless of vocabulary cropping, these should always be in vocab for IFTTT
  vocab_tgt:add('ROOT')
  vocab_tgt:add('FUNC')
  vocab_tgt:add('IF')
  vocab_tgt:add('THEN')
  vocab_tgt:add('PARAMS')
  vocab_tgt:add('TRIGGER')
elseif opt.dataset == 'synth' then
  vocab_tgt:add('ROOT')
end
for k,v in pairs{vocab_src, vocab_tgt} do
  v:add_unk_token()
  v:add_start_token()
  v:add_end_token()
  v:add_root_token()
end

if verbose then
  printf('%8s %10s %8s %10s %8s %10s %8s\n',
    '','SOT Tok','SOS Ind','SOS Tok','SOS Ind','EOS Tok','EOS Ind')
  printf('%8s %10s %8d %10s %8d %10s %8d\n',
    'source',vocab_src.root_token,vocab_src.root_index,
    vocab_src.start_token,vocab_src.start_index,
    vocab_src.end_token,vocab_src.end_index)
  printf('%8s %10s %8d %10s %8d %10s %8d\n',
    'target',vocab_tgt.root_token,vocab_tgt.root_index,
    vocab_tgt.start_token,vocab_tgt.start_index,
    vocab_tgt.end_token,vocab_tgt.end_index)
end

-- load embeddings
print('loading word embeddings...')
local emb_vocab, emb_vecs, vocab_dim
if opt.embedding == 'glove' then
  emb_vocab, emb_vecs =  tree2tree.read_default_embedding(
  source_lang,target_lang,vocab_src,vocab_tgt,disjoint_vocabs,initialize_composed)
elseif opt.embedding == 'rand' then
  vocab_dim = {source = vocab_src.size, target = vocab_tgt.size}
end

-- load datasets
print('loading datasets...')
local train_dir = data_dir .. 'train/'
local dev_dir   = data_dir .. 'dev/'
local test_dir  = data_dir .. 'test/'
local debug_dir = data_dir .. 'debug/'
if opt.debugdata then
  print('Will use debug dataset')
  train_dir = debug_dir
  dev_dir   = debug_dir
  test_dir  = debug_dir
end

local read_args = {
  seq_len = opt.seq_len, maxexamples = opt.maxexamples, shuffle = opt.shuffle,
  use_source_trees = use_source_trees, use_target_trees = use_target_trees,
  lr_tree = opt.lr, pad_eos = pad_eos, pad_sot = pad_sot, multi_root =false,
  read_native_target= read_native_target, lowercase = lowercase
}
local train_dataset, dev_dataset
if opt.tensor_mode then
  train_dataset = tree2tree.read_parallel_tree_dataset_tensor(
  train_dir,vocab_src,vocab_tgt,pad_sot,pad_eos,lr_tree,opt.seq_len)
  dev_dataset   = tree2tree.read_parallel_tree_dataset_tensor(
  dev_dir,vocab_src,vocab_tgt,pad_sot,pad_eos,lr_tree,opt.seq_len)
else
  train_dataset = tree2tree.read_parallel_tree_dataset(
  train_dir,vocab_src,vocab_tgt,read_args)
  read_args.shuffle=false
  dev_dataset   = tree2tree.read_parallel_tree_dataset(
  dev_dir,vocab_src,vocab_tgt,read_args)
end

printf('\tnum train = %d\n', train_dataset.size)
printf('\tnum dev   = %d\n', dev_dataset.size)
--printf('\tnum test  = %d\n', test_dataset.size)


--If test sentence generation
--local example_tree = train_dataset.source[example_k].tree
--local example_sent = train_dataset.source[example_k].sent
local source_example  = {}
if opt.test_generation then
  local example_k = 4
  local example_tree, example_sent
  if opt.tensor_mode then
    example_tree, example_sent = dev_dataset.source[1][1]:select(1,8), dev_dataset.source[2]:select(1,8)
  else
    --print(dev_dataset.sents[8].source)
    --print(dev_dataset.trees[8].source)
    example_tree, example_sent = dev_dataset.source[8].tree, dev_dataset.source[8].sent
  end
  source_example = {tree = example_tree, sent = example_sent}
end

--print(vocab_tgt:reverse_map({1406, 2181}))
-- local exno = 2
-- if use_source_trees then
--   display_tree_example(train_dataset.source[exno].tree, train_dataset.source[exno].sent, vocab_src,false, false)
-- end
-- if use_target_trees then
--   display_tree_example(train_dataset.target[exno].tree, train_dataset.target[exno].sent, vocab_tgt,true, false)
-- end



display_tree_example(train_dataset.target[1].tree, train_dataset.target[1].sent, vocab_tgt,true, false)

-- print(train_dataset[exno].tree.target:to_lambda_string())
--

-- local nodes = train_dataset[exno].tree.target:breadth_first()
-- for i,k in pairs(nodes) do printf(' (%i,%i) ', k.index,k.idx) end
-- printf('\n')
--
-- local nodes = train_dataset[exno].tree.target:getTraversal('forward')
--
-- for i,k in pairs(nodes) do printf(' (%i,%i) ', k[1].index,k[1].idx) end
-- printf('\n')
--
-- bk()

-- initialize model
local config = {
  emb_vecs     = emb_vecs,
  vocab_dim    = vocab_dim,
  num_layers   = opt.layers,
  mem_dim      = opt.memdim,
  batch_size   = opt.batch,
  encoder_type = opt.encoder,
  decoder_type = opt.decoder,
  criterion    = opt.criterion,
  reg          = opt.reg,
  lr_tree      = opt.lr,
  target_vocab = vocab_tgt,
  source_vocab = vocab_src,
  grad_clip    = opt.clip,
  pad_eos      = pad_eos,
  seq_len      = opt.seq_len,
  debug        = opt.debug,
}

-- Convenient general caller for model
local model_class
if opt.tensor_mode then
  model_class = tree2tree.Tree2Tree_ED_tensor
else
  model_class = tree2tree.Tree2Tree_ED
end

local model = model_class(config)

-- number of epochs to train
local num_epochs = opt.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- Get paths for outputs
local save_paths = tree2tree.get_save_paths(opt)
local model_path
if opt.model_file ~= 'NA' then
  save_paths['models'] = opt.model_file
end
if not opt.nosave then
  printf('%-25s = %s\n', 'model save path', save_paths['models'])
  if opt.prediction then
    printf('%-25s = %s\n', 'pretty predsave path', save_paths['preds']['test'])
    printf('%-25s = %s\n', 'raw pred save path', save_paths['raw_preds']['test'])
  end
else
  print('Will NOT save models nor predictions')
end

if Debug > 2 then
  print(vocab_tgt:index('ROOT'))
  dev_dataset.target[1].tree:print_preorder(dev_dataset.target[1].sent,vocab_tgt)
  pred_acc = compute_accuracy(
        opt.dataset,acc_type,
        dev_dataset.target,
        dev_dataset.target,
        vocab_tgt, opt.explicit_root,opt.explicit_cues)
  print(pred_acc)
end

-- Last opportunity to bail before computations begin
if not opt.noprompt then
  prompt_continue()
end


-- model:predicting()
-- torch.manualSeed(10)
-- enc_in, dec_in, dec_gold = model:prepare_inputs(train_dataset.trees[15],train_dataset.sents[15])
-- model:test_sampling(enc_in, dec_gold)
-- model:training()
-- bk()

-- train
local train_start = sys.clock()
local best_dev_loss = 1e309
local best_dev_acc   = 0
local best_dev_score = (opt.select_criterion == 'loss') and best_dev_loss or best_dev_acc
local best_dev_model = model
local best_epoch = 0
header('Training model')
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  model:train(train_dataset)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)
  -- uncomment to compute train scores
  if opt.train_scores then
    local train_loss = model:compute_loss(train_dataset)
    printf('-- train loss: %.4f\n', train_loss)
  end

  if opt.test_generation then
    model:predicting()
    torch.manualSeed(10)
    enc_in, dec_in, dec_gold = model:prepare_inputs(train_dataset.source[5],train_dataset.target[5])
    model:test_sampling(opt.dataset,enc_in, dec_gold)
    torch.manualSeed(10)
    enc_in, dec_in, dec_gold = model:prepare_inputs(dev_dataset.source[8],dev_dataset.target[8])
    model:test_sampling(opt.dataset, enc_in, dec_gold)
    model:training()
  end

  -- Compute teacher forced-loss (i.e. average next-step error)
  local dev_loss, dev_topo_loss
  if i % opt.dev_loss_every == 0 then
    dev_loss, dev_topo_loss  = model:compute_loss(dev_dataset)
    if dev_topo_loss then
      printf('-- dev loss: %.4f  (topo: %.4f)\n', dev_loss, dev_topo_loss)
    else
      printf('-- dev loss: %.4f\n', dev_loss)
    end
  end

  -- Compute prediction accuracy
  local dev_pred_acc, dev_predictions
  if opt.prediction and equals_any('dev',opt.prediction_folds) and i % opt.dev_pred_every == 0 then
    dev_predictions = model:predict_dataset(opt.dataset,dev_dataset.source,opt.maxpred)
    model:training() -- NOTE: MOVE THIS TO END OF THIS IF, TO MAKE CLEAR THIS IS FOR NEXT ROUND
    dev_pred_stats = compute_accuracy(
      opt.dataset,acc_type,dev_predictions, dev_dataset.target,vocab_tgt,
      opt.explicit_root,opt.explicit_cues)
    display_accuracy(opt.dataset, acc_type, dev_pred_stats)
    if opt.dataset == 'IFTTT' then
      dev_pred_acc = dev_pred_stats['channel-acc']
    elseif opt.dataset == 'BABYMT' then
      dev_pred_acc = dev_pred_stats['BLEU']
    else
      dev_pred_acc = dev_pred_stats['node_macro_f1']
    end
  end

  if i % opt.checkpoint_every then
    local checkpoint_path = save_paths['checkpoints'] .. '_ckp' .. tostring(i) .. '.t7'
    print('writing model to ' .. checkpoint_path)
    model:save(checkpoint_path)
    --torch.save(checkpoint_path .. '.dump', model)
  end

  if opt.select_criterion == 'loss' and (dev_loss < best_dev_loss) then
    update_best_model = true
    best_dev_score = dev_loss
    best_dev_loss  = dev_loss
  elseif opt.select_criterion == 'prediction' and  (dev_pred_acc > best_dev_acc) then
    -- if (dataset == 'IFTTT' and (dev_pred_acc['func'] > best_dev_acc['func'])) or -- Change to F1?
    --    (dataset ~= 'IFTTT' and (dev_pred_acc > best_dev_acc)) then
    update_best_model = true
    best_dev_acc   = dev_pred_acc
    best_dev_score = dev_pred_acc
  else
    update_best_model = false
  end

  -- if opt.early_stopping then
  --   prev_dev_score = (opt.select_criterion == 'prediction') and dev_pred_acc or dev_loss
  --   early_stop =
  -- end

  if update_best_model then
    print('Updating best dev model!')
    best_dev_model = model_class(config)
    best_dev_model_preds = dev_predictions
    best_dev_model.params:copy(model.params)
    best_epoch     = i
  end
  collectgarbage() -- Why not
end
printf('finished training in %.2fs\n', sys.clock() - train_start)

-- evaluate
local train_dataset, dev_dataset
read_args.read_ids = true
if opt.tensor_mode then
  test_dataset = tree2tree.read_parallel_tree_dataset_tensor(test_dir,vocab_src,vocab_tgt,pad_sot,pad_eos,lr_tree,opt.seq_len)
else
  test_dataset = tree2tree.read_parallel_tree_dataset(test_dir,vocab_src,vocab_tgt,read_args)
end
printf('\tnum test  = %d\n', test_dataset.size)

header('Evaluating on test set')
printf('-- using model with dev %s = %.4f\n',
  opt.select_criterion == 'loss' and 'loss' or 'accuracy', best_dev_score)

local test_loss, topo_loss = best_dev_model:compute_loss(test_dataset)
if topo_loss then
  printf('-- test loss: %.4f  (topo: %.4f)\n', test_loss, topo_loss)
else
  printf('-- test loss: %.4f\n', dev_loss)
end

local test_predictions = model:predict_dataset(opt.dataset,test_dataset.source, opt.maxpred)

local test_pred_stats, test_extracted, test_bool_correct, test_example_stats = compute_accuracy(
  opt.dataset,acc_type,test_predictions, test_dataset.target,vocab_tgt,
  opt.explicit_root,opt.explicit_cues)


display_accuracy(opt.dataset, acc_type,test_pred_stats)

-- Save models and predictions
local model_path = save_paths['models']
local pred_path

if not opt.nosave then
  -- model
  print('writing model to ' .. model_path)
  best_dev_model:save(model_path)
  torch.save(model_path .. '.dump', best_dev_model)

  -- predictions
  if opt.predictions_file == 'NA' then
    summary_pred_path = save_paths['preds']['test']
    raw_pred_path = save_paths['raw_preds']['test']
  else
    summary_pred_path = opt.predictions_path .. '.pred_summary'
    raw_pred_path = opt.predictions_path .. '.raw_preds'
  end
  print('writing predictions to ' .. summary_pred_path)
  tree2tree.write_predictions(opt.dataset,summary_pred_path,raw_pred_path,test_dataset.ids,
  test_extracted, test_example_stats,test_bool_correct, vocab_tgt)
end



-- if opt.dataset == 'IFTTT' then
--   tree2tree.write_IFTTT_predictions(save_paths['preds']['test'],test_extracted, test_correct)
-- else
--   tree2tree.write_predictions(save_paths['preds']['test'],test_pred_linearized, test_gold_linearized)
-- end

-- -- write predictions to disk
-- local predictions_file = torch.DiskFile(predictions_save_path, 'w')
-- print('writing predictions to ' .. predictions_save_path)
-- for i = 1, test_predictions:size(1) do
--   predictions_file:writeFloat(test_predictions[i])
-- end
-- predictions_file:close()

-- local test_pred_size = opt.paradigm == 'scoring' and test_score_predictions:size(1) or test_score_predictions[1]:size(1)
-- local score_predictions_file = torch.DiskFile(save_paths['score_preds']['test'], 'w')
-- for i = 1, test_pred_size do
--   if opt.paradigm == 'scoring' then
--     score_predictions_file:writeFloat(test_score_predictions[i])
--   elseif opt.paradigm == 'ranking' then
--     -- THere are two scores in this case: for positive and negative. We dump the
--     -- coding of ranking: +1 means first one is better rankged, -1 the opposite.
--     local outval = test_score_predictions[1][i] > test_score_predictions[2][i] and 1 or -1
--     score_predictions_file:writeInt(outval)
--   end
-- end
-- score_predictions_file:close()

-- write dev preds
-- if opt.prediction and equals_any('dev',opt.prediction_folds) and best_dev_model_preds then
--   print('writing relation predictions on dev set to ' .. save_paths['preds']['dev'])
--
--   if opt.dataset == 'IFTTT' then
--     tree2tree.write_IFTTT_predictions(save_paths['preds']['dev'],dev_extracted, dev_correct)
--   else
--     tree2tree.write_predictions(save_paths['preds']['dev'],dev_predictions)
--   end
-- end


if opt.append_results then
  local global_results_file
  if opt.logfile == 'NA' then
    global_results_file = path.join('result_logs',opt.dataset,'log.tsv')
  else
    global_results_file = opt.logfile
  end
  print('appending results to ' .. global_results_file)
  local global_results_dfile = io.open(global_results_file, 'a')
  local etc = ""
  local row
  if opt.dataset ~= 'IFTTT' then
    row = string.format("%s\t%d\t%s\t%s\t%d\t%f\t%f\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
    model_path,opt.maxexamples or -1,opt.encoder,opt.decoder,opt.memdim,opt.reg,opt.learning_rate,
    opt.batch,best_epoch,best_dev_score,test_loss,
    100*test_pred_stats['node_prec'],100*test_pred_stats['node_rec'],100*test_pred_stats['node_macro_f1'],
    100*test_pred_stats['edge_prec'],100*test_pred_stats['edge_rec'],100*test_pred_stats['edge_macro_f1']
  )
  else
    row = string.format("%s\t%s\t%s\t%s\t%d\t%f\t%f\t%d\t%d\t%f\t%f\t%f\t%f\n",
    model_path,model_name,opt.encoder,opt.decoder,opt.memdim,opt.reg,opt.learning_rate,
    opt.batch,best_epoch,best_dev_score,test_loss,100*test_pred_stats['channel-acc'],100*test_pred_stats['channel-func-acc'])
  end
  global_results_dfile:write(row)
  global_results_dfile:close()
end


-- to load a saved model
-- local loaded = model_class.load(model_save_path)


--[[

      TRASH

]]--


-- Horrible patch. FIXME
-- local dev_prediction_dataset = {source = {}, target = {}}
-- if opt.prediction then
--   local num_pred_examples = opt.maxpred and opt.maxpred or #dev_dataset.sents
--   for i=1,num_pred_examples do -- CHANGE to dev!
--     local enc_input, dec_input, dec_gold = model:prepare_inputs(dev_dataset.trees[i], dev_dataset.sents[i])
--     dev_prediction_dataset['source'][i] = enc_input
--     dev_prediction_dataset['target'][i] = dec_gold
--   end
-- end
