-- Returns the Levenshtein distance between the two given strings
function levenshtein(str1, str2)
	-- the input arguments are already tables
	local individualLabelsOutput = str1 
	local individualLabelsGT = str2
	--for w in str1:gmatch("%S+") do table.insert(individualLabelsOutput,w) end 
	--for w in str2:gmatch("%S+") do table.insert(individualLabelsGT,w) end

	local len1 = #individualLabelsOutput
	local len2 = #individualLabelsGT
	local matrix = {}
	local cost = 0
	
        -- quick cut-offs to save time
	if (len1 == 0) then
		return len2
	elseif (len2 == 0) then
		return len1
	elseif (str1 == str2) then
		return 0
	end
	
        -- initialise the base matrix values
	for i = 0, len1, 1 do
		matrix[i] = {}
		matrix[i][0] = i
	end
	for j = 0, len2, 1 do
		matrix[0][j] = j
	end
	
        -- actual Levenshtein algorithm
	for i = 1, len1, 1 do
		for j = 1, len2, 1 do
			if (individualLabelsOutput[i] == individualLabelsGT[j]) then
				cost = 0
			else
				cost = 1
			end
			
			matrix[i][j] = math.min(matrix[i-1][j] + 1, matrix[i][j-1] + 1, matrix[i-1][j-1] + cost)
		end
	end
	
        -- return the last value - this is the Levenshtein distance
	return matrix[len1][len2]
end
