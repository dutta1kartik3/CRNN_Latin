function getConfig()
    local config = {
        nClasses         = 37,
        maxT             = 32,
        displayInterval  = 100,
        testInterval     = 100,
        nTestDisplay     = 15,
        trainBatchSize   = 64, --64 original
        valBatchSize     = 32,--32, --32 original
        snapshotInterval = 100,
        maxIterations    = 2000000,
        optimMethod      = optim.adadelta,
        optimConfig      = {},
        trainSetPath     = '/home/kartik/iam-train/data.mdb',
	--trainSetPath     = '/ssd_scratch/cvit/train-elastic-lmdb/data.mdb',
	--trainSetPath     = '/ssd_scratch/cvit/iam-train/data.mdb',
        valSetPath       = '/home/kartik/iam-val/data.mdb',
        --valSetPath       = '/ssd_scratch/cvit/cal-elastic-lmdb/data.mdb',
	savePath         = '/ssd_scratch/cvit/'
    }
    return config
end

function createModel(config)
    local nc = config.nClasses
    local nl = nc + 1
    local nt = config.maxT

    local ks = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}
    local ps = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}
    local ss = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    local nm = {64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512}
    local nh = {256, 256}

    function convRelu(i, batchNormalization)
        batchNormalization = batchNormalization or false
        local nIn = nm[i-1] or 1
        local nOut = nm[i]
        local subModel = nn.Sequential()
        local conv = cudnn.SpatialConvolution(nIn, nOut, ks[i], ks[i], ss[i], ss[i], ps[i], ps[i])
        subModel:add(conv)
        if batchNormalization then
            subModel:add(nn.SpatialBatchNormalization(nOut))
        end
        subModel:add(nn.ReLU(true))
        return subModel
    end

    function residual(i,subsample)
	local concat = nil
        local block = nn.Sequential()
        local nIn = nm[i-1] or 1
	local subsample_val = nIn
        local nOut = nm[i]
        local subModel = nn.Sequential()
        subModel:add(nn.SpatialBatchNormalization(nIn))
        subModel:add(nn.ReLU(true))
	local conv = cudnn.SpatialConvolution(nIn, nOut, ks[i], ks[i], ss[i], ss[i], ps[i], ps[i])
        subModel:add(conv)
        local i=i+1
        local nIn = nm[i-1]
        local nOut = nm[i]
        subModel:add(nn.SpatialBatchNormalization(nIn))
        subModel:add(nn.ReLU(true))
        local conv = cudnn.SpatialConvolution(nIn, nOut, ks[i], ks[i], ss[i], ss[i], ps[i], ps[i])
        subModel:add(conv)
        if subsample then
		local shortcut = cudnn.SpatialConvolution(subsample_val, nOut, 1,1, 1,1, 0,0)
    		concat = nn.ConcatTable():add(subModel):add(shortcut)

	else
        	concat = nn.ConcatTable():add(subModel):add(nn.Identity())
	end
        block:add(concat)
        block:add(nn.CAddTable())
        return block
    end

    function bidirectionalLSTM(nIn, nHidden, nOut, maxT)
        local fwdLstm = nn.LstmLayer(nIn, nHidden, maxT, 0.2, false)
        local bwdLstm = nn.LstmLayer(nIn, nHidden, maxT, 0.2, true)
        local ct = nn.ConcatTable():add(fwdLstm):add(bwdLstm)
        local blstm = nn.Sequential():add(ct):add(nn.BiRnnJoin(nHidden, nOut, maxT))
        return blstm
    end

    local model = nn.Sequential()
    model:add(nn.Copy('torch.ByteTensor', 'torch.CudaTensor', false, true))
    model:add(nn.AddConstant(-128.0))
    model:add(nn.MulConstant(1.0 / 128))
    -- local networks = require 'networks'    
    require 'nn'
    require 'stn'
    
    ------
    -- Prepare your localization network
    local lk_nw =  nn.Sequential()
    lk_nw:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) 
    lk_nw:add(cudnn.SpatialConvolution(1, 64, 3, 3, 1, 1, 1,1))
    lk_nw:add(cudnn.ReLU(true))
    lk_nw:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) 
    lk_nw:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1,1))
    lk_nw:add(cudnn.ReLU(true))
    lk_nw:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
    lk_nw:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1,1))
    lk_nw:add(cudnn.ReLU(true))
    lk_nw:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) 
    
    lk_nw:add(nn.View(-1):setNumInputDims(3))
    lk_nw:add(nn.Linear(12288, 30))
    lk_nw:add(cudnn.ReLU(true))
    
    local classifier = nn.Sequential()
    classifier:add(nn.Linear(30, 6))
    classifier:get(1).weight:zero()
    classifier:get(1).bias = torch.Tensor({1,0,0,0,1,0})
    lk_nw:add(classifier)
    ------
    -- prepare both branches of the st
    local ct = nn.ConcatTable()
    -- This branch does not modify the input, just change the data layout to bhwd
    local branch1 = nn.Transpose({3,4},{2,4})
    -- This branch will compute the parameters and generate the grid
    local branch2 = nn.Sequential()
    branch2:add(lk_nw)
    -- Here you can restrict the possible transformation with the "use_*" boolean variables
    branch2:add(nn.AffineTransformMatrixGenerator(True, True, True))
    branch2:add(nn.AffineGridGeneratorBHWD(96, 256))
    ct:add(branch1)
    ct:add(branch2)
    --print(ct)
    ------
    -- Wrap the st in one module
    local st_module = nn.Sequential()
    st_module:add(ct)
    st_module:add(nn.BilinearSamplerBHWD())
    -- go back to the bdhw layout (used by all default torch modules)
    st_module:add(nn.Transpose({2,4},{3,4}))
    
    model:add(st_module)
    model:add(convRelu(1,true))
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))       -- 64x48x128
    --
    model:add(residual(2,false))
    model:add(residual(4,false))
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))       -- 64x24x64
    model:add(residual(6,true))
    model:add(residual(8,false))
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))       -- 128x12x32
    model:add(residual(10,true))
    model:add(residual(12,false))
    model:add(cudnn.SpatialMaxPooling(2, 2, 1, 2, 1, 0)) -- 256x6x33
    model:add(residual(14,true))
    model:add(residual(16,false))
    model:add(cudnn.SpatialMaxPooling(2, 2, 1, 2, 1, 0)) -- 512x3x34
    --
    model:add(convRelu(18, true))                         -- 512x1x32
    model:add(nn.View(512, -1):setNumInputDims(3))       -- 512x32
    model:add(nn.Transpose({2, 3}))                      -- 32x512
    model:add(nn.SplitTable(2, 3))
    model:add(bidirectionalLSTM(512, 256, 256, nt))
    model:add(bidirectionalLSTM(256, 256,  nl, nt))
    model:add(nn.SharedParallelTable(nn.LogSoftMax(), nt))
    model:add(nn.JoinTable(1, 1))
    model:add(nn.View(-1, nl):setNumInputDims(1))
    model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', false, true))
    model:cuda()
    local criterion = nn.CtcCriterion()

    return model, criterion
end   
