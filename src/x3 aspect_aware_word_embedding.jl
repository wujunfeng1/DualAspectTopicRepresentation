using LinearAlgebra

function loadTitlePhrases(fileNameIn::String)::Tuple{Vector{Vector{String}}, Dict{String,Int}}
    fileIn = open(fileNameIn)
    phraseGroups = Vector{Vector{String}}()
    phraseFreqs = Dict{String,Int}()
    for line in readlines(fileIn)
        phrases = split(line)
        if length(phrases) > 0
            push!(phraseGroups, phrases)
            for phrase in phrases
                phraseFreqs[phrase] = get(phraseFreqs, phrase, 0) + 1
            end
        end  
    end
    close(fileIn)
    (phraseGroups, phraseFreqs)
end

function getSubPhrasePosition(words1::Vector{SubString{String}},words2::Vector{SubString{String}})::Tuple{Float64, Float64}
    n1 = length(words1)
    n2 = length(words2)
    if n1 <= n2
        return (-1.0,-1.0)
    end
    for i in 1:n1-n2+1
        matched = true
        for j in 1:n2
            if words1[i+j-1] != words2[j]
                matched = false
                break
            end
        end
        if matched
            return (i/n1,(n1-i-n2+1)/n1)
        end
    end
    return (-1.0,-1.0)
end

SubPhrasePositions = Dict{String,Tuple{Float64,Float64}}
function findSubPhrases(phraseFreqs::Dict{String,Int})::Tuple{Dict{String,SubPhrasePositions}, Dict{String, Int}}
    phrases = [phrase for (phrase, freq) in phraseFreqs]
    numPhrases = length(phrases)
    phraseWords = [split(phrases[i], "-") for i in 1:numPhrases]

    numCPUs = length(Sys.cpu_info())
    jobs = Channel{Tuple{Int,Int}}(numCPUs)
    jobOutputs = Channel{Dict{String, SubPhrasePositions}}(numCPUs)
    function runJob(iCPU::Int)
        myOutputs = Dict{String, SubPhrasePositions}()
        for job in jobs
            for phraseID1 in job[1]:job[2]
                phrase1 = phrases[phraseID1]
                words1 = phraseWords[phraseID1]
                subPhrases1 = SubPhrasePositions()
                for phraseID2 in 1:numPhrases
                    words2 = phraseWords[phraseID2]
                    subPhrasePosition = getSubPhrasePosition(words1, words2)
                    if subPhrasePosition[1] >= 0.0
                        phrase2 = phrases[phraseID2]
                        subPhrases1[phrase2] = subPhrasePosition
                    end
                end
                myOutputs[phrase1] = subPhrases1 
            end
        end
        put!(jobOutputs, myOutputs)
    end
    function makeJobs(batchSize::Int)
        for i in 1:batchSize:numPhrases
            put!(jobs, (i, min(i + batchSize - 1, numPhrases)))
        end
    end
    bind(jobs, @async makeJobs(100))
    for iCPU in 1:numCPUs
        Threads.@spawn runJob(iCPU)
        # runJob(iCPU)
    end
    
    subPhrases = Dict{String, SubPhrasePositions}()
    for iCPU in 1:numCPUs
        localSubPhrases = take!(jobOutputs)
        for (phrase,subPhrasePositions) in localSubPhrases
            subPhrases[phrase] = subPhrasePositions
        end
    end

    subPhraseFreqs = copy(phraseFreqs)
    for (phrase,subPhrasePositions) in subPhrases
        freq = phraseFreqs[phrase]
        for (subPhrase, position) in subPhrasePositions
            subPhraseFreqs[subPhrase] += freq
        end
    end

    (subPhrases, subPhraseFreqs)
end

function getScores(fileNameIn::String, minCount::Int)::Tuple{Array{Float64,2},Array{String,1},Dict{String,Int64}}
    phraseGroups, phraseFreqs = loadTitlePhrases(fileNameIn)
    subPhrases, subPhraseFreqs = findSubPhrases(phraseFreqs)

    phrases = Vector{String}()
    phraseIDs = Dict{String,Int}()
    for (phrase, freq) in subPhraseFreqs
        if freq >= minCount
            push!(phrases, phrase)
            phraseIDs[phrase] = length(phrases)
        end
    end

    numPhrases = length(phrases)
    scores = fill(0.0, (numPhrases, numPhrases))
    for titlePhrases in phraseGroups
        highFreqs = Vector{Bool}(undef, length(titlePhrases))
        subHighFreqs = Vector{Bool}(undef, length(titlePhrases))
        for (j, phraseJ) in enumerate(titlePhrases)
            highFreqs[j] = haskey(phraseIDs, phraseJ)
            subHighFreqs[j] = false
            for (subPhrase, position) in subPhrases[phraseJ]
                if haskey(phraseIDs, subPhrase)
                    subHighFreqs[j] = true
                end
            end
        end
        for (j, phraseJ) in enumerate(titlePhrases)
            if subHighFreqs[j]
                for (subPhrase1, position1) in subPhrases[phraseJ]
                    if haskey(phraseIDs, subPhrase1)
                        id1 = phraseIDs[subPhrase1]
                        for (subPhrase2, position2) in subPhrases[phraseJ]
                            if subPhrase1 == subPhrase2
                                continue
                            end
                            if haskey(phraseIDs, subPhrase2)
                                id2 = phraseIDs[subPhrase2]
                                ldist = abs(position1[1] - position2[1])
                                rdist = abs(position1[2] - position2[2])
                                dist = max(ldist, rdist)
                                @assert dist > 0.0
                                score = 1.0 / dist
                                # make sure the following update is one sided
                                scores[id1, id2] += score
                            end
                        end
                        if highFreqs[j]
                            id2 = phraseIDs[phraseJ]
                            dist = max(position1[1], position1[2])
                            @assert dist > 0.0
                            score = 1.0 / dist
                            scores[id1, id2] += score
                            scores[id2, id1] += score
                        end
                    end
                end
            end
            for k in j+1:length(titlePhrases)
                phraseK = titlePhrases[k]
                if highFreqs[j]
                    idJ = phraseIDs[phraseJ]
                    if highFreqs[k]
                        idK = phraseIDs[phraseK]
                        score = 1.0 / (k - j)
                        scores[idJ, idK] += score
                        scores[idK, idJ] += score
                    end
                    if subHighFreqs[k]
                        for (subPhraseK, positionK) in subPhrases[phraseK]
                            if haskey(phraseIDs, subPhraseK)
                                idK = phraseIDs[subPhraseK]
                                score = 1.0 / (k - j + positionK[1])
                                scores[idJ, idK] += score
                                scores[idK, idJ] += score
                            end
                        end
                    end
                end
                if subHighFreqs[j]
                    for (subPhraseJ, positionJ) in subPhrases[phraseJ]
                        if haskey(phraseIDs, subPhraseJ)
                            idJ = phraseIDs[subPhraseJ]
                            if highFreqs[k]
                                idK = phraseIDs[phraseK]
                                score = 1.0 / (k - j + positionJ[2])
                                scores[idJ, idK] += score
                                scores[idK, idJ] += score
                            end
                            if subHighFreqs[k]
                                for (subPhraseK, positionK) in subPhrases[phraseK]
                                    if haskey(phraseIDs, subPhraseK)
                                        idK = phraseIDs[subPhraseK]
                                        score = 1.0 / (k - j + positionJ[2] + positionK[1])
                                        scores[idJ, idK] += score
                                        scores[idK, idJ] += score
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    A = fill(0.0, (numPhrases, numPhrases))
    y = fill(0.0, numPhrases)
    for i in 1:numPhrases
        for j in 1:numPhrases
            if j == i
                continue
            end
            if scores[i, j] > 0.0
                A[i,i] += 1.0
                A[i,j] += 1.0
                y[i] += log(scores[i,j])
            end 
        end
    end
    bias = A \ y
    for i in 1:numPhrases
        scores[i,i] = 1.0
        for j in 1:numPhrases
            if j == i
                continue
            end
            if scores[i, j] > 0.0
                unbiasedScore = log(scores[i,j]) - bias[i] - bias[j]
                scores[i,i] += abs(unbiasedScore)
                scores[i,j] = unbiasedScore
            end
        end
    end

    # this enforce symmetry to eliminate numerical corruption of symmetry
    scores = 0.5(scores + scores')

    (scores, phrases, phraseIDs)
end

function trainWordEmbedding(fileNameIn::String, fileNameOut::String, numDims::Int, minCount::Int) 
    scores, phrases, phraseIDs = getScores(fileNameIn, minCount)

    eigScores = eigen(Symmetric(scores))
    eigOrder = sortperm(eigScores.values, rev=true)

    # note: eigScores.vectors * diagm(eigScores.values) * eigScores.vectors' == scores
    numPhrases = length(phrases)
    w = fill(0.0, min(numDims, numPhrases))
    for k in 1:min(numDims, numPhrases)
        w[k] = sqrt(eigScores.values[eigOrder[k]])
    end
    fileOut = open(fileNameOut, "w")
    println(fileOut, "$numPhrases $numDims")
    for i in 1:numPhrases
        phrase = phrases[i]
        print(fileOut, phrase)
        for k in 1:min(numDims, numPhrases)
            value = eigScores.vectors[i,eigOrder[k]] * w[k]
            print(fileOut, " $value")
        end
        println(fileOut)
    end
    close(fileOut)
end
