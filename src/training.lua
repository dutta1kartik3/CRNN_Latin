require("levenshtein_algorithm")
function trainModel(model, criterion, trainSet, testSet)
    -- get model parameters
    local params, gradParams = model:getParameters()
    local optimMethod = gConfig.optimMethod
    local optimState = gConfig.optimConfig

    function trainBatch(inputBatch, targetBatch)
        --[[ One step of SGD training
        ARGS:
          - `inputBatch`  : batch of inputs (images)
          - `targetBatch` : batch of targets (groundtruth labels)
        ]]
        model:training()
	-- nFrame is the #samples in the current batch
        local nFrame = inputBatch:size(1)
        local feval = function(p)
            if p ~= params then
                params:copy(x)
            end
            gradParams:zero()
	    --print("Before Model Forward")
            local outputBatch = model:forward(inputBatch)
	    --print("Before Crit Forward")
	    --[[for i=1,targetBatch:size(1) do
	    	for j=1, targetBatch:size(2) do
			io.write(targetBatch[i][j] .. ' ')
		end
		io.write('\n')
	    end
	    print("~~~~~~~~~~~~~~~~~~~~~")--]]
            local f = criterion:forward(outputBatch, targetBatch)
	    --print("Before Model+CRIT Backward")
            model:backward(inputBatch, criterion:backward(outputBatch, targetBatch))
	    --print("Before Updating Params")
            gradParams:div(nFrame)
            f = f / nFrame
            return f, gradParams
        end
        local _, loss = optimMethod(feval, params, optimState); loss = loss[1]
        return loss
    end

    function validation(input, target)
        --[[ Do validation
        ARGS:
          - `input`  : validation inputs
          - `target` : validation targets
        ]]
        logging('Validating...')
        model:evaluate()

        -- batch feed forward
        local batchSize = gConfig.valBatchSize
        local nFrame = input:size(1)
        local output = torch.Tensor(nFrame, gConfig.maxT, gConfig.nClasses+1)
        for i = 1, nFrame, batchSize do
	    --print(i)
            local actualBatchSize = math.min(batchSize, nFrame-i+1)
	    --print ('actualbatchsize')
	    --print (actualBatchSize)
            local inputBatch = input:narrow(1,i,actualBatchSize)
            local outputBatch = model:forward(inputBatch)
	    --print('inputBatch sie')
	    --print(inputBatch:size())
	    --print ('outputBath size')
	    --print(outputBatch:size())
	    --print('output narrow size')

	    --print (output:narrow(1,i,actualBatchSize):size())
            
	    output:narrow(1,i,actualBatchSize):copy(outputBatch)
        end

        -- compute loss
        local loss = criterion:forward(output, target, true) / nFrame
	--print (output)

        -- decoding
        local pred, rawPred = naiveDecoding(output)
	-- rawPred is a nil value
	--print (pred)
	--print ('################')
	--print (rawpred)







        local predStr = label2str(pred)
	-- predStr is a list of predicted strings ; the predicted sequences of each of the input image in val set
        -- compute recognition metrics
        local gtStr = label2str(target)
        local nWrong = 0
	local totalEditDistance = 0
	local totalGTLength = 0
	for i=1, nFrame do
            --print(i)
	    local outputSeq={}
	    local gtSeq={}
	    --pred[i] will be 1*26 seq of op classes trailing empty ones are zeros same is target[i]
	    for  j=1, gConfig.maxT do

	    	if pred[i][j] == 0 then
	    		break
		end
	    	gtSeq[j]=target[i][j]


	    end

	    for  j=1, gConfig.maxT do

		if pred[i][j]==0 then
			break
		end	
		outputSeq[j]=pred[i][j]

	    end	
									            
	    local outputSeqLength = #outputSeq
	    local gtSeqLength = #gtSeq

	    if outputSeqLength ~= gtSeqLength then
	    	nWrong = nWrong + 1
	    else
		for k=1, outputSeqLength do
			
			if outputSeq[k] ~= gtSeq[k] then
				nWrong=nWrong+1
				break
			end
				
		end
		
	    end

		
	    editDistance=levenshtein(outputSeq,gtSeq)
	    
	    totalEditDistance=totalEditDistance + editDistance
	    totalGTLength= totalGTLength + outputSeqLength 
        end
        local wordError = (nWrong) *100/ nFrame
	local characterError= totalEditDistance * 100/ totalGTLength
        logging(string.format('Test loss = %f, CER = %f, WER=%f', loss, characterError, wordError))
        --logging(string.format('Test loss = %f,  WER=%f', loss, wordError))

        -- show prediction examples
	-- minesh : commenting printing examples part since it was anyway junk with unicode labels
        local rawPredStr = label2str(rawPred, true)
        for i = 1, math.min(nFrame, gConfig.nTestDisplay) do
            local idx = math.floor(math.random(1, nFrame))
            logging(string.format('%25s  =>  %-25s  (GT:%-20s)',
                rawPredStr[idx], predStr[idx], gtStr[idx]))
        end
    end

    function validation_lexicon(input, target)
        --[[ Do validation
        ARGS:
          - `input`  : validation inputs
          - `target` : validation targets
        ]]
        logging('Validating...')
        model:evaluate()

        -- batch feed forward
        local batchSize = gConfig.valBatchSize
        local nFrame = input:size(1)
        local output = torch.Tensor(nFrame, gConfig.maxT, gConfig.nClasses+1)
        for i = 1, nFrame, batchSize do
	    --print(i)
            local actualBatchSize = math.min(batchSize, nFrame-i+1)
	    --print ('actualbatchsize')
	    --print (actualBatchSize)
            local inputBatch = input:narrow(1,i,actualBatchSize)
            local outputBatch = model:forward(inputBatch)
	    --print('inputBatch sie')
	    --print(inputBatch:size())
	    --print ('outputBath size')
	    --print(outputBatch:size())
	    --print('output narrow size')

	    --print (output:narrow(1,i,actualBatchSize):size())
            
	    output:narrow(1,i,actualBatchSize):copy(outputBatch)
        end

        -- compute loss
        local loss = criterion:forward(output, target, true) / nFrame
	--print (output)

        -- decoding
        local file = io.open("lexicontf.txt", "r");

	local lexicon = {}
	for line in file:lines() do
		local temp = string.gsub(line, "\n", "")
		table.insert(lexicon, temp);
	end
	file:close()
        local pred, rawPred = decodingWithLexicon(output, lexicon)
	-- rawPred is a nil value
	--print (pred)
	--print ('################')
	--print (rawpred)







        local predStr = label2str(pred)
	-- predStr is a list of predicted strings ; the predicted sequences of each of the input image in val set
        -- compute recognition metrics
        local gtStr = label2str(target)
        local nWrong = 0
	local totalEditDistance = 0
	local totalGTLength = 0
        print(nFrame)
	for i=1, nFrame do
            print(i)
	    local outputSeq={}
	    local gtSeq={}
	    --pred[i] will be 1*26 seq of op classes trailing empty ones are zeros same is target[i]
	    for  j=1, gConfig.maxT do

	    	if pred[i][j] == 0 then
	    		break
		end
	    	gtSeq[j]=target[i][j]


	    end

	    for  j=1, gConfig.maxT do

		if pred[i][j]==0 then
			break
		end	
		outputSeq[j]=pred[i][j]

	    end	
									            
	    local outputSeqLength = #outputSeq
	    local gtSeqLength = #gtSeq

	    if outputSeqLength ~= gtSeqLength then
	    	nWrong = nWrong + 1
	    else
		for k=1, outputSeqLength do
			
			if outputSeq[k] ~= gtSeq[k] then
				nWrong=nWrong+1
				break
			end
				
		end
		
	    end

		
	    editDistance=levenshtein(outputSeq,gtSeq)
	    
	    totalEditDistance=totalEditDistance + editDistance
	    totalGTLength= totalGTLength + outputSeqLength 
        end
        local wordError = (nWrong) *100/ nFrame
	local characterError= totalEditDistance * 100/ totalGTLength
        logging(string.format('Test loss = %f, CER = %f, WER=%f', loss, characterError, wordError))
        --logging(string.format('Test loss = %f,  WER=%f', loss, wordError))

        -- show prediction examples
	-- minesh : commenting printing examples part since it was anyway junk with unicode labels
        local rawPredStr = label2str(rawPred, true)
        for i = 1, math.min(nFrame, gConfig.nTestDisplay) do
            local idx = math.floor(math.random(1, nFrame))
            logging(string.format('%25s  =>  %-25s  (GT:%-20s)',
                rawPredStr[idx], predStr[idx], gtStr[idx]))
        end
    end

    -- train loop
    local iterations = 0
    local loss = 0
    while true do
        -- validation
        if iterations == 0 or iterations % gConfig.testInterval == 0 then
            local valInput, valTarget = testSet:allImageLabel(5000)
            validation(valInput, valTarget)
            collectgarbage()
        end

        -- train batch
        local input, target = trainSet:nextBatch()
        assert(input:nDimension() == 4)
        loss = loss + trainBatch(input, target)
        iterations = iterations + 1

        -- display
        if iterations % gConfig.displayInterval == 0 then
            loss = loss / gConfig.displayInterval
            logging(string.format('Iteration %d - train loss = %f', iterations, loss))
            diagnoseGradients(model:parameters())
            loss = 0
            collectgarbage()
        end

        -- save snapshot
        if iterations > 0 and iterations % gConfig.snapshotInterval == 0 then
            local savePath = paths.concat(gConfig.savePath, string.format('snapshot_%d.t7', iterations))
            torch.save(savePath, modelState(model))
            logging(string.format('Snapshot saved to %s', savePath))
            collectgarbage()
        end

        -- terminate
        if iterations > gConfig.maxIterations then
            logging('Maximum iterations reached, terminating ...')
            break
        end
    end
end
