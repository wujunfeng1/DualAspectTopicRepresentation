function isHit(phraseWords::Vector{SubString{String}}, words::Vector{SubString{String}})::Bool
    if length(phraseWords) > length(words)
		return false
	end
	m = length(phraseWords)
	n = length(words) - length(phraseWords) + 1
	for idxWord in 1:n 
		if words[idxWord] != phraseWords[1]
			continue
        end
		matched = true
		for i in 1:m - 1 
			if words[idxWord + i] != phraseWords[1 + i]
				matched = false
				break
            end
		end
		if matched
			return true
        end
    end
	return false
end

function countMAGHits(phrasePairs::Vector{Tuple{String,String}})::Vector{Dict{Int,Int}}
    numPairs = length(phrasePairs)
    pairHits = [Dict{Int,Int}() for i in 1:numPairs]
    pairTokens = [(split(pair[1], "-"), split(pair[2], "-")) for pair in phrasePairs]

    numCPUs = length(Sys.cpu_info())
    jobs = Channel{Vector{Tuple{Int,String}}}(numCPUs)
    jobOutputs = Channel{Vector{Dict{Int,Int}}}(numCPUs)

    function runJob(iCPU::Int)
        myPairHits = [Dict{Int,Int}() for i in 1:numPairs]
        for job in jobs
            for yearTitle in job
                year = yearTitle[1]
                title = yearTitle[2]
                titleTokens = split(title)
                for pairID in 1:numPairs
                    phraseWords1 = pairTokens[pairID][1]
                    phraseWords2 = pairTokens[pairID][2]
                    isHit1 = isHit(phraseWords1, titleTokens)
                    isHit2 = isHit(phraseWords2, titleTokens)
                    if isHit1 && isHit2
                        myPairHits[pairID][year] = get(myPairHits[pairID], year, 0) + 1
                    end
                end
            end
        end
        put!(jobOutputs, myPairHits)
    end

    function makeJobs(batchSize::Int)
        for magID in 0:106
            fileName = "TokenizedMAGTitles/MAGTitles.$magID.txt"
            file = open(fileName)
            job = Vector{Tuple{Int,String}}()
            for line in readlines(file)
                fields = split(line, ". ")
                if length(fields) == 2
                    push!(job, (parse(Int, fields[1]), fields[2]))
                    if length(job) == batchSize
                        put!(jobs, job)
                        job = Vector{Tuple{Int,String}}()
                    end
                end
            end
            if length(job) > 0
                put!(jobs, job)
            end
            close(file)
        end
    end

    bind(jobs, @async makeJobs(1000))
    for iCPU in 1:numCPUs
        Threads.@spawn runJob(iCPU)
    end
    for iCPU in 1:numCPUs
        localPairHits = take!(jobOutputs)
        for pairID in 1:numPairs
            for (year, hits) in localPairHits[pairID]
                pairHits[pairID][year] = get(pairHits[pairID], year, 0) + hits
            end
        end
    end
    
    pairHits
end

function loadPhrasePairs(fileName::String)::Vector{Tuple{String,String}}
    phrasePairs = Vector{Tuple{String,String}}()
    file = open(fileName)
    for line in readlines(file)
        fields = split(line)
        if length(fields) == 8
            push!(phrasePairs, (fields[3], fields[4]))
            push!(phrasePairs, (fields[5], fields[6]))
            push!(phrasePairs, (fields[7], fields[8]))
        end
    end
    close(file)
    phrasePairs
end

function saveMAGHits(fileName::String, phrasePairs::Vector{Tuple{String,String}}, magHits::Vector{Dict{Int,Int}})
    numPairs = length(phrasePairs)
    file = open(fileName, "w")
    for pairID in 1:numPairs
        phrase1 = phrasePairs[pairID][1]
        phrase2 = phrasePairs[pairID][2]
        print(file, "$phrase1 $phrase2")
        pairYearHits = sort([yearHits for yearHits in magHits[pairID]])
        for yearHits in pairYearHits
            year = yearHits[1]
            hits = yearHits[2]
            print(file, " $year:$hits")
        end
        print(file, "\n")
    end
    close(file)
end

phrasePairs = loadPhrasePairs("DATR/workspace/NIPS.Top100Topics.txt")
magHits = countMAGHits(phrasePairs)
saveMAGHits("DATR/workspace/NIPS.Top100Topics.MAGHits.txt", phrasePairs, magHits)