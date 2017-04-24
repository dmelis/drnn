local TreePredictionLayer, parent = torch.class('tree2tree.TreePredictionLayer', 'nn.Module')

--[[
  The last layer in a DRNN cell. Used mostly for handling prediction of label and topology.
  Behavior changes:
    - during training, it computes a loss of input topology probablities and replaces them
      with true topologial features, passes these to SoftMax
    - during prediction, it first predicts binary topo features using probs and passes these
      to softmaxe module
]]--

function TreePredictionLayer:__init(model,target_dim, lr_tree)
  parent.__init(self)
  self.model = model
  self.target_dim = target_dim
  self.lr_tree = lr_tree or false

  self.output = {}
  self.tau_a = 0.5
  self.tau_f = 0.5
  self.sample_mode_prediction = 'max' -- ['max','sample']. For binary decisions in topology modules

  self.predict = false -- turn on to trigger prediction behavior

  local softmax
  if self.model == 'ntp' then
    softmax = self:model1_softmax()
  elseif self.model == 'stp' and not self.lr_tree then
    softmax = self:model_stp_softmax()
    self.topo_criterion_depth = nn.BCECriterion()
  elseif self.model == 'stp' and self.lr_tree then
    softmax = self:model_stp_lr_softmax()
    self.topo_criterion_depth = {L = nn.BCECriterion(), R = nn.BCECriterion()}
  elseif self.model == 'dtp' and not self.lr_tree then
    softmax = self:model_dtp_softmax()
    self.topo_criterion_depth = nn.BCECriterion()
    self.topo_criterion_width = nn.BCECriterion()  -- They have to be instantiated independenty, or cloned
  elseif self.model == 'dtp' and self.lr_tree then
    softmax = self:model_dtp_lr_softmax()
    self.topo_criterion_depth = {L = nn.BCECriterion(), R = nn.BCECriterion()}
    self.topo_criterion_width = nn.BCECriterion()  -- They have to be instantiated independenty, or cloned
  else
    print("Error: unrecognized model")
  end
  self.softmax = softmax

  self.modules = {self.softmax}

end

function TreePredictionLayer:model_ntp_softmax()
  -- Node inputs are: {h_Out, pA, pF}
  local softmax = nn.LogSoftMax()
  return softmax
end

function TreePredictionLayer:model_stp_softmax()
  -- Node inputs are: {h_Out, pA}
  local p1 = nn.ParallelTable()
              :add(nn.Identity())
              :add(nn.Linear(1,self.target_dim))
  local softmax = nn.Sequential()
            :add(p1)
            :add(nn.CAddTable())
            :add(nn.LogSoftMax())
  return softmax
end

function TreePredictionLayer:model_dtp_softmax()
  -- Node inputs are: {h_Out, pA, pF}
  local p1 = nn.ParallelTable()
              :add(nn.Identity())
              :add(nn.Linear(1,self.target_dim))
              :add(nn.Linear(1,self.target_dim))
              -- :add(nn.LookupTable(2,self.target_dim))
              -- :add(nn.LookupTable(2,self.target_dim))
  local softmax = nn.Sequential()
            :add(p1)
            :add(nn.CAddTable())
            :add(nn.LogSoftMax())
  return softmax
end

-- No model_ntp_lr because it doesn't have offspring prediction anyways,
-- so softmax is the same. We thus use the same model_ntp in that case.

-- STP + LR
function TreePredictionLayer:model_stp_lr_softmax()
  -- Node inputs are: {h_Out, pA_L, pA_R}
  local p1 = nn.ParallelTable()
              :add(nn.Identity())
              :add(nn.Linear(1,self.target_dim))
              :add(nn.Linear(1,self.target_dim))
  local softmax = nn.Sequential()
            :add(p1)
            :add(nn.CAddTable())
            :add(nn.LogSoftMax())
  return softmax
end

-- DTP + LR
function TreePredictionLayer:model_dtp_lr_softmax()
  -- Node inputs are: {h_Out, pA_L, pA_R, pF}
  local p1 = nn.ParallelTable()
              :add(nn.Identity())
              :add(nn.Linear(1,self.target_dim))
              :add(nn.Linear(1,self.target_dim))
              :add(nn.Linear(1,self.target_dim))
  local softmax = nn.Sequential()
            :add(p1)
            :add(nn.CAddTable())
            :add(nn.LogSoftMax())
  return softmax
end

-- Reusable function to unpack input, check it has the correct number of elements, etc
function TreePredictionLayer:process_inputs(input)
  local prob_a, prob_f
  local word_output = (self.model == 'ntp') and input or input[1]
  if self.model ~= 'ntp' and not self.lr_tree then
    assert((self.model == 'stp' and #input == 2) or (self.model == 'dtp' and #input == 3),
    "Input to TreePredictionLayer with wrong number of elements")
    prob_a  = input[2]
  elseif self.model ~= 'ntp' then
    assert((self.model == 'stp' and #input == 3) or (self.model == 'dtp' and #input == 4),
    "Input to TreePredictionLayer with wrong number of elements")
    prob_a = {L = input[2], R = input[3]}
  end
  local prob_f      = (self.model == 'dtp' and not self.lr_tree) and input[3] or
                      (self.model == 'dtp' and self.lr_tree) and input[4] or nil
  return word_output, prob_a, prob_f
end


function TreePredictionLayer:process_softmax_train(word_output, prob_a, prob_f, mode)
  --local return_loss = return_loss or false -- True for updateOutput in STP/DTP
  --local return_grad = return_grad or false -- True for updateGradInput in STP/DTP
  local mode = mode or 'forward'  -- Options: forward, backward, accgrad

  local input_to_softmax
  local gold_a_bin, loss_topo_a, grad_topo_a
  local gold_f_bin, loss_topo_f, grad_topo_f
  if self.model == 'ntp' then
    input_to_softmax = word_output
    return input_to_softmax -- All we need in this case, so let's get out
  end
  if self.lr_tree then
    assert(self.has_lchildren ~= nil and self.has_rchildren ~=nil,
    "Depth learning arguments not passed correctly")
    gold_a_bin = {
                   L = (self.has_lchildren) and torch.Tensor(1):fill(1) or torch.zeros(1),
                   R = (self.has_rchildren) and torch.Tensor(1):fill(1) or torch.zeros(1)
                 }
    if mode == 'forward' then
      local loss_topo_a_L = self.topo_criterion_depth['L']:forward(prob_a['L'],gold_a_bin['L'])
      local loss_topo_a_R = self.topo_criterion_depth['R']:forward(prob_a['R'],gold_a_bin['R'])
      loss_topo_a = loss_topo_a_L + loss_topo_a_R
    elseif mode == 'backward' then
      local grad_topo_a_L = self.topo_criterion_depth['L']:backward(prob_a['L'],gold_a_bin['L'])
      local grad_topo_a_R = self.topo_criterion_depth['R']:backward(prob_a['R'],gold_a_bin['R'])
      grad_topo_a = {L = grad_topo_a_L, R = grad_topo_a_R}
    end
  else -- Not LR tree
    assert(self.has_children ~= nil,
    "Depth learning argument not passed correctly")
    gold_a_bin = (self.has_children) and torch.Tensor(1):fill(1) or torch.zeros(1)
    if mode == 'forward' then
      loss_topo_a = self.topo_criterion_depth:forward(prob_a,gold_a_bin)
    elseif mode == 'backward' then
      grad_topo_a = self.topo_criterion_depth:backward(prob_a,gold_a_bin)
    end
  end
  if (self.model == 'dtp') then
    assert(self.has_brother ~= nil, "Width learning arguments not passed correctly")
    gold_f_bin = (self.has_brother) and torch.Tensor(1):fill(1) or torch.zeros(1)
    if mode == 'forward' then
      loss_topo_f = self.topo_criterion_width:forward(prob_f,gold_f_bin)
    elseif mode == 'backward' then
      grad_topo_f = self.topo_criterion_width:backward(prob_f,gold_f_bin)
    end
  end
  if mode == 'forward' then
    debugutils.debug_topo_pred(Debug,self.model,self.lr_tree,prob_a,self.tau_a,
    gold_a_bin,prob_f,self.tau_f,gold_f_bin)
  end

  --local loss_topo_a = (self.model ~= 'ntp') and self.topo_criterion_depth:forward(prob_a,gold_a_bin)
  if self.model == 'stp' and not self.lr_tree then
    input_to_softmax = {word_output, gold_a_bin}  -- Need to shift to {1,2} for Lookuptable
  elseif self.model == 'stp' and self.lr_tree then
    input_to_softmax = {word_output, gold_a_bin['L'], gold_a_bin['R']}
  elseif self.model == 'dtp' and not self.lr_tree then
    input_to_softmax = {word_output, gold_a_bin, gold_f_bin}  -- Need to shift to {1,2} for Lookuptable
  elseif self.model == 'dtp' and self.lr_tree then
    input_to_softmax = {word_output, gold_a_bin['L'], gold_a_bin['R'], gold_f_bin}
  else
    print("Error: unrecognized model")
  end
  if mode == 'forward' then
    return input_to_softmax, loss_topo_a, loss_topo_f
  elseif mode == 'backward' then
    return input_to_softmax, grad_topo_a, grad_topo_f, gold_a_bin, gold_f_bin
  end
  return input_to_softmax
end

function TreePredictionLayer:process_softmax_predict(word_output, prob_a, prob_f)
  local input_to_softmax, pred_a_tensor, pred_f_tensor, pred_a, pred_f
  -- The last two are scalar version, which we return
  if self.model == 'ntp' then
    input_to_softmax = input
  else
    if not self.lr_tree then
      if self.sample_mode_prediction == 'sample' then
        pred_a_tensor = torch.Tensor(1):fill(torch.bernoulli(prob_a[1]))
      elseif self.sample_mode_prediction == 'max' then
        pred_a_tensor = (prob_a[1] > self.tau_a) and torch.Tensor(1):fill(1) or torch.zeros(1)
      else
        bk('Unrecognized sample mode for prediction in Tree Prediction Layer')
      end
      input_to_softmax =  {word_output, pred_a_tensor}
      pred_a = pred_a_tensor[1] -- Tensor to value
    else -- lr tree
      if self.sample_mode_prediction == 'sample' then
        pred_a_tensor = {L  = torch.Tensor(1):fill(torch.bernoulli(prob_a['L'][1])),
                         R  = torch.Tensor(1):fill(torch.bernoulli(prob_a['R'][1]))}
      elseif self.sample_mode_prediction == 'max' then
        pred_a_tensor = {L  = (prob_a['L'][1] > self.tau_a) and torch.Tensor(1):fill(1) or torch.zeros(1),
                         R  = (prob_a['R'][1] > self.tau_a) and torch.Tensor(1):fill(1) or torch.zeros(1)}
      else
        bk('Unrecognized sample mode for prediction in Tree Prediction Layer')
      end
      input_to_softmax =  {word_output, pred_a_tensor['L'], pred_a_tensor['R']}
      pred_a = {L = pred_a_tensor['L'][1], R = pred_a_tensor['R'][1]}
    end
    if self.model == 'dtp' then
      if self.sample_mode_prediction == 'sample' then
        pred_f_tensor = torch.Tensor(1):fill(torch.bernoulli(prob_f[1]))
      elseif self.sample_mode_prediction == 'max' then
        pred_f_tensor = (prob_f[1] > self.tau_f) and torch.Tensor(1):fill(1) or torch.zeros(1)
      else
        bk('Unrecognized sample mode for prediction in Tree Prediction Layer')
      end
      --pred_f_tensor = (prob_f[1] > self.tau_f)  and torch.Tensor(1):fill(1) or torch.zeros(1)
      local ind = self.lr_tree and 4 or 3
      input_to_softmax[ind] = pred_f_tensor
      pred_f = pred_f_tensor[1]
    end
  end
  return input_to_softmax, pred_a, pred_f
end

--[[
      Outputs:
      -- In train mode: tensor
      -- In predict mode: {tensor, tensor, tensor}
]]--
function TreePredictionLayer:updateOutput(input)
  -- Unpack inputs according to model.
  local output, combined_loss, table_output, input_to_softmax
  local word_output, prob_a, prob_f = self:process_inputs(input)
  if self.predict ~= true then -- train mode
    if self.model == 'ntp' then
      input_to_softmax = self:process_softmax_train(word_output)
      output = {self.softmax:forward(input_to_softmax)} -- TODO: With or without brackets?
    else -- must track topo loss for stp and dtp
      input_to_softmax, loss_topo_a, loss_topo_f = self:process_softmax_train(word_output, prob_a, prob_f, 'forward')
      combined_loss = loss_topo_a + ((loss_topo_f) or 0)
      output = {self.softmax:forward(input_to_softmax), combined_loss}
    end
  else -- predict mode
    -- In this case we need to return the predictions too.
    local input_to_softmax, pred_a, pred_f = self:process_softmax_predict(word_output, prob_a, prob_f)
    --print(pred_a, pred_f, prob_a['L'][1], prob_a['R'][1], prob_f)
    outW = self.softmax:forward(input_to_softmax)
    output = (self.model == 'ntp') and outW
          or (self.model == 'stp' and not self.lr_tree) and {outW,pred_a}
          or (self.model == 'stp' and self.lr_tree)     and {outW,pred_a}
          or (self.model == 'dtp' and not self.lr_tree) and {outW,pred_a,pred_f}
          or (self.model == 'dtp' and self.lr_tree)     and {outW,pred_a,pred_f}
  end
  self.output = output
  return self.output
end

function TreePredictionLayer:updateGradInput(input, gradOutput)
  local gradInput
  local word_output, prob_a, prob_f = self:process_inputs(input)
  --local input_to_softmax = process_softmax_train(word_output, prob_a, prob_f)

  if self.model == 'ntp' then
    local input_to_softmax = input -- Same as doing process_softmax_train(word_output)
    gradInput = self.softmax:backward(input_to_softmax, gradOutput)
  else -- STP/DTP: Will use topological arguments
    -- The last two are only for debugging. Can remove eventually.
    local input_to_softmax, grad_topo_a, grad_topo_f, gold_a_bin, gold_f_bin = self:process_softmax_train(word_output, prob_a, prob_f,'backward')
    --output = {self.softmax:forward(input_to_softmax), combined_loss}
    grad_word = self.softmax:backward(input_to_softmax, gradOutput)[1]
    debugutils.debug_topo_grad(Debug,self.model, self.lr_tree,
    prob_a,gold_a_bin,grad_topo_a,prob_f,gold_f_bin,grad_topo_f)

    if self.model == 'stp' and not self.lr_tree then
      gradInput = {grad_word, grad_topo_a}
    elseif self.model == 'stp' and self.lr_tree then
      gradInput = {grad_word, grad_topo_a['L'], grad_topo_a['R']}
    elseif self.model == 'dtp' and not self.lr_tree then
      gradInput = {grad_word, grad_topo_a, grad_topo_f}
    elseif self.model == 'dtp' and self.lr_tree then
      gradInput = {grad_word, grad_topo_a['L'], grad_topo_a['R'], grad_topo_f}
    else
      print("Error: unrecognized model")
    end
  end
  self.gradInput = gradInput
  return self.gradInput
end

function TreePredictionLayer:accGradParameters(input, gradOutput, scale)
  local word_output, prob_a, prob_f = self:process_inputs(input)
  local input_to_softmax = self:process_softmax_train(word_output, prob_a, prob_f,'accGrad')
  self.softmax:accGradParameters(input_to_softmax, gradOutput, scale)
end

function TreePredictionLayer:training()
  self.train   = true
  self.predict = false
  for i=1,#self.modules do
    self.modules[i]:training()
  end
  self:forget()
end

function TreePredictionLayer:evaluate()
  self.train   = false
  self.predict = false
  for i=1,#self.modules do
    self.modules[i]:evaluate()
  end
  self:forget()
end

function TreePredictionLayer:predicting()
  self.train   = false
  self.predict = true
  self:forget()
end

function TreePredictionLayer:forget()
  self.output = {}
end
