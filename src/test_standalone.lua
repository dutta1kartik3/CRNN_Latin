require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

require('libcrnn')
require('utilities')
require('inference')
require('CtcCriterion')
require('DatasetLmdb')
require('LstmLayer')
require('BiRnnJoin')
require('SharedParallelTable')


cutorch.setDevice(1)
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')

print('Loading model...')
local modelDir = '../model/stnd-residual/'
paths.dofile(paths.concat(modelDir, 'config.lua'))
local modelLoadPath = paths.concat(modelDir, 'snapshot_100200.t7')
gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0
local model, criterion = createModel(gConfig)
local snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)
model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))

--local file = io.open("", "r");
--local lexicons = {}
--for line in file:lines() do
	--local temp = string.gsub(line, "\n", "")
	--table.insert (lexicons, temp);
--end
--file:close()

local line = arg[1]; --The file containing the path to test images

local imagePath = string.gsub(line, "\n", "")
--local addn = '/ssd_scratch/cvit/' -- Add the parent path to the test images folder
local addn = ''
imagePath = addn .. imagePath
local img = loadAndResizeImage(imagePath)
local text = recognizeImageLexiconFree(model, img) --Change based on whether lexicon free or lexicon based output is required
--local text = recognizeImageWithLexicion(model, img, lexicons)
print(string.format('Recognized text: %s', text))
