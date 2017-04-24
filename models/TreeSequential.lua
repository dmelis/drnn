local TreeSequential, _ = torch.class('tree2tree.TreeSequential', 'nn.Sequential')

-- This overrides Sequential's updateOutput. Only difference is that this passess
-- on a tree from module to module
function TreeSequential:updateOutput(input)
   local currentOutput = input
   for i=1,#self.modules do
      if torch.isTypeOf(self.modules[i], 'nn.ParallelTable') then -- To acccount for models with parallel outputs
        for j=1,#self.modules[i].modules do
          self.modules[i].modules[j].tree = (i==1) and self.initialTree or
                            (self.modules[i-1].outputTree or self.modules[i-1].tree)
          self.modules[i].modules[j].lr_tree = self.lr_tree
        end
      end
      -- Do this also for parallel modules, so that the next one can grab it without another exception
      self.modules[i].tree = (i==1) and self.initialTree or
                        (self.modules[i-1].outputTree or self.modules[i-1].tree)
      self.modules[i].lr_tree = self.lr_tree
      currentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', currentOutput)
   end
   self.output = currentOutput
   return currentOutput
end

function TreeSequential:setInitialTree(tree)
  local lr_tree = (torch.type(tree) == "tree2tree.LRTree")
  local initialTree = (lr_tree) and tree2tree.LRTree():copy(tree, "skeleton")
                                or  tree2tree.Tree():copy(tree, "skeleton")
  self.initialTree = initialTree
  self.lr_tree = lr_tree
end
