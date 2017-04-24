-- Based on elementresearch rnn's AbstractSequencer module
local AbstractTreequencer, parent = torch.class("nn.AbstractTreequencer", "nn.Container")

function AbstractTreequencer:getStepModule(step)
   error"DEPRECATED 27 Oct 2015. Wrap your internal modules into a Recursor instead"
end

function AbstractTreequencer:sharedClone(shareParams, shareGradParams, clones, pointers, stepClone)
   -- stepClone is ignored (always false, i.e. uses sharedClone)
   return parent.sharedClone(self, shareParams, shareGradParams, clones, pointers)
end

-- AbstractSequence handles its own rho internally (dynamically)
function AbstractTreequencer:maxBPTTstep(rho)
end
