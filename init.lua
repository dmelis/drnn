require('torch')
require('nn')
require('nngraph')
require('optim')
require('sys')

require('rnn')
require('xlua')
require('lfs')
require('csvigo')

dl = require 'dataload'
tds = require('tds')

torch.setheaptracking(true)

-- Penlight stuff
path = require 'pl.path'
pretty = require 'pl.pretty'
require('pl.stringx').import()  -- The import makes them available as string methods, e.g. (' '):join({'foo','bar'})
lapp = require 'pl.lapp'
lapp.slack = true
plutils = require 'pl.utils'
printf = plutils.printf

-- My stuff
tree2tree = {}


-- Utils. TODO: put a init.lua file in utils and require that instead
require 'utils.read_data'
require 'utils.Vocab'
require 'utils.Tree'
require 'utils.test_utils'
require 'utils.handy_functions'
require 'utils.evaluation'
require 'utils.tree_conversion'
debugutils = require 'utils.debugutils'

-- Third Party
require 'utils.CRowAddTable'
-- require 'models.thirdparty.TreeLSTM'
-- require 'models.thirdparty.ChildSumTreeLSTM'

factory    = require 'models.module_factory'
require 'models.module_factory'
require 'models.AbstractTree'
require 'models.TreeSequential'
require 'models.TreeRecursor'
require 'models.Tree2Tree_ED'
require 'models.DRNN'
require 'models.TreePredictionLayer'
require 'models.Treequencer'

-- Criteria
require 'criteria.TreeCriterion'
require 'samplers.Samplers'


-- Globals (modify if desired)
tree2tree.data_dir        = 'data'    -- TODO: Move all data locally
tree2tree.models_dir      = 'trained_models'
tree2tree.predictions_dir = 'predictions'

tree2tree.goToken  = '<'
tree2tree.eosToken = '>'
