using Word2Vec
using LinearAlgebra

const stopWords = Set{String}([
    "a",
	"an",
	"and",
	"as",
	"based",
	"by",
	"for",
	"from",
	"in",
	"on",
	"of",
    "over",
	"that",
	"the",
	"this",
	"to",
	"via",
	"with",
	"without",
	"is",
	"are",
	"be",
	"you",
	"we",
	"they",
	"it",
	"your",
	"our",
	"their",
	"its",
	"what",
	"when",
	"where",
	"how",
	"do",
	"use",
	"here",
	"there",
	"using",
    "than",
	"、",
	",",
	"，",
	":",
	"：",
	".",
	"。",
	"‧",
	"!",
	"！",
	"?",
	"？",
	";",
	"；",
	"(",
	"（",
	")",
	"）",
	"'",
	"‘",
	"’",
	"\"",
	"「",
	"」",
	"“",
	"”",
	"`",
	"…",
    "</s>",
])

function getNonStopWords(wordVectors::WordVectors)::Tuple{Vector{String},Dict{String,Int}}
    nonStopWords = Vector{String}()
    nonStopWordIDs = Dict{String,Int}()
    for (wordID, word) in enumerate(wordVectors.vocab)
        if !(word in stopWords)
            push!(nonStopWords, word)
            nonStopWordIDs[word] = wordID
        end
    end
    (nonStopWords, nonStopWordIDs)
end

function getOverlap(words1::Vector{SubString{String}}, words2::Vector{SubString{String}})::Float64
    n1 = length(words1)
    n2 = length(words2)
    if n1 == 1
        if n2 == 1
            if words1[1] == words2[1]
                return 1.0
            else
                return 0.0
            end
        elseif n2 > 1
            for j in 1:n2
                if words2[j] == words1[1]
                    return 1 / n2
                end
            end
            return 0.0
        end
    elseif n1 > 1
        if n2 == 1
            for i in 1:n1
                if words1[i] == words2[1]
                    return 1 / n1
                end
            end
            return 0.0
        elseif n2 > 1
            wordIDs = Dict{String,Vector{Int}}()
            for (j, word) in enumerate(words2)
                if haskey(wordIDs, word)
                    push!(wordIDs[word], j)
                else
                    wordIDs[word] = [j]
                end
            end
            wordOverlaps1 = Dict{Int,Int}()
            wordOverlaps2 = Dict{Int,Int}()
            for (i, word) in enumerate(words1)
                if haskey(wordIDs, word)
                    js = wordIDs[word]
                    for j in js
                        if !haskey(wordOverlaps2, j)
                            wordOverlaps1[i] = j
                            wordOverlaps2[j] = i
                            break
                        end
                    end
                end
            end
            m = length(wordOverlaps1)
            return m / (n1 + n2 - m)
        else
            return 0.0
        end
    else
        return 0.0
    end
end

function getPhraseOverlaps(nonStopWords::Vector{String})::Vector{Dict{Int,Float64}}
    numPhrases = length(nonStopWords)
    phraseOverlaps = [Dict{Int,Float64}() for i in 1:numPhrases]
    for (phraseID1, phrase1) in enumerate(nonStopWords)
        phrase1Words = split(phrase1, "-")
        for (phraseID2, phrase2) in enumerate(nonStopWords)
            if phraseID1 == phraseID2
                continue
            end
            phrase2Words = split(phrase2, "-")
            overlap = getOverlap(phrase1Words, phrase2Words)
            if overlap > 0.0
                phraseOverlaps[phraseID1][phraseID2] = overlap
            end
        end
    end
    phraseOverlaps
end

function loadPhraseIDGroups(fileName::String, nonStopWordIDs::Dict{String,Int})::Vector{Vector{Int}}
    file = open(fileName)
    phraseIDGroups = Vector{Vector{Int}}()
    for line in readlines(file)
        if length(line) == 0 
            continue
        end
        phrases = split(line)
        idGroup = Vector{Int}()
        for phrase in phrases
            if haskey(nonStopWordIDs, phrase)
                push!(idGroup, nonStopWordIDs[phrase])
            end
        end
        push!(phraseIDGroups, idGroup)
    end
    close(file)
    phraseIDGroups
end

function getCardinalities(nonStopWords::Vector{String}, nonStopWordIDs::Dict{String,Int}, phraseIDGroups::Vector{Vector{Int}})::Vector{Int}
    numPhrases = length(nonStopWords)
    idMap = Dict{Int,Int}()
    for (newID, phrase) in enumerate(nonStopWords)
        idMap[nonStopWordIDs[phrase]] = newID
    end

    phraseFreqs = fill(0, numPhrases)
    for idGroup in phraseIDGroups
        for id in idGroup
            phraseFreqs[idMap[id]] += 1
        end
    end

    phraseFreqs
end

function normalizeVectors(wordVectors::WordVectors, nonStopWords::Vector{String})::Array{Float64,2}
    numDims, numWords = size(wordVectors)
    numVectors = length(nonStopWords)
    result = Array{Float64,2}(undef, (numDims, numVectors))
    for i in 1:numVectors
        vector = get_vector(wordVectors, nonStopWords[i])
        w = 1.0 / norm(vector)
        result[:,i] = vector * w
    end
    result
end

function getSimilarityMatrix(vectors::Array{Float64,2})::Array{Float64,2}
    0.5 .+ 0.5 .* (vectors' * vectors)
end

function getConnectionMatrix(simMat::Array{Float64,2}, cardinalities::Vector{Int})::Array{Float64,2}
    simMat .* (cardinalities * cardinalities')
end

function deltaModularity(connMat::Array{Float64,2}, sumConn::Vector{Float64}, totalConn::Float64, 
    pairFreqs::Dict{Tuple{Int,Int},Int}, phraseOverlaps::Vector{Dict{Int,Float64}}, cardinalities::Vector{Int},
    u::Int, oldCu::Set{Int}, newCu::Set{Int})
    result = 0.0
    for v in oldCu
        if v != u
            wuv = sumConn[u] * sumConn[v] / totalConn
            result -= connMat[u,v] - wuv
            if haskey(pairFreqs, (u, v))
                result += pairFreqs[(u, v)] * wuv
            elseif haskey(pairFreqs, (v, u))
                result += pairFreqs[(v, u)] * wuv
            end
            if haskey(phraseOverlaps[u], v)
                result -= phraseOverlaps[u][v] * cardinalities[u] * cardinalities[v]
            end
        end
    end
    for v in newCu
        wuv = sumConn[u] * sumConn[v] / totalConn
        result += connMat[u,v] - wuv
        if haskey(pairFreqs, (u, v))
            result -= pairFreqs[(u, v)] * wuv
        elseif haskey(pairFreqs, (v, u))
            result -= pairFreqs[(v, u)] * wuv
        end
        if haskey(phraseOverlaps[u], v)
            result += phraseOverlaps[u][v] * cardinalities[u] * cardinalities[v]
        end
    end
    result
end

function findAspects(connMat::Array{Float64,2}, pairFreqs::Dict{Tuple{Int,Int},Int}, phraseOverlaps::Vector{Dict{Int,Float64}}, 
    cardinalities::Vector{Int}, maxIter::Int)::Tuple{Vector{Int},Vector{Set{Int}}}
    numPoints = size(connMat, 1)
    clusters = collect(Set{Int}(i) for i in 1:numPoints)
    clusterIDs = collect(1:numPoints)
    numClusters = numPoints
    sumConn = sum(connMat, dims=2)[:]
    totalConn = sum(sumConn)

    numCPUs = length(Sys.cpu_info())
    for iter in 1:maxIter
        ###########################################################################################
        # step 1: randomly pick a destination with positive gain for each point
        jobs = Channel{Tuple{Int,Int}}(numCPUs)
        Segments = Vector{Tuple{Int,Int}}
        WeightedDests = Vector{Tuple{Float64,Int}}
        jobOutputs = Channel{Tuple{Segments,Vector{WeightedDests}}}(numCPUs)

        function makeJobs(batchSize::Int)
            for i in 1:batchSize:numPoints
                put!(jobs, (i, min(i + batchSize - 1, numPoints)))
            end
        end

        function runJob(iCPU::Int)
            mySegments = Segments()
            myWeightedDests = Vector{WeightedDests}()
            for job in jobs
                push!(mySegments, job)
                jobWeightedDests = WeightedDests(undef, job[2] - job[1] + 1)
                # println("cpu $iCPU: computing weighted dests for $job")
                for u in job[1]:job[2]
                    # println("cpu $iCPU: computing for $u")
                    oldCuID = clusterIDs[u]
                    oldCu = clusters[oldCuID]
                    attractions = fill(0.0, numClusters)
                    for (newCuID, newCu) in enumerate(clusters)
                        if newCuID != oldCuID 
                            attractions[newCuID] = max(0.0, deltaModularity(connMat, sumConn, totalConn, pairFreqs, phraseOverlaps, cardinalities, u, oldCu, newCu))
                        end
                    end
                    # println("cpu $iCPU: attractions of $u is $attractions")
                    sumAttractions = sum(attractions)
                    thres = rand() * sumAttractions
                    partialSumAttr = 0.0
                    weightedDest = (0.0, oldCuID)
                    for destCID in 1:numClusters
                        attr = attractions[destCID]
                        partialSumAttr += attr
                        if attr > 0.0 && partialSumAttr >= thres
                            weightedDest = (attr, destCID)
                            break
                        end
                    end
                    jobWeightedDests[u - job[1] + 1] = weightedDest 
                end # u
                push!(myWeightedDests, jobWeightedDests)
                # println("cpu $iCPU: finish computing weighted dests for $job") 
            end # job
            put!(jobOutputs, (mySegments, myWeightedDests))
        end # runJob

        bind(jobs, @async makeJobs(100))
        for iCPU in 1:numCPUs
            Threads.@spawn runJob(iCPU)
        end
        weightedDests = WeightedDests(undef, numPoints)
        for iCPU in 1:numCPUs
            segmentedWeightedDests = take!(jobOutputs)
            for (segID, segment) in enumerate(segmentedWeightedDests[1])
                weightedDests[segment[1]:segment[2]] = segmentedWeightedDests[2][segID]
            end
        end

        ###########################################################################################
        # step 2: merge point-movement requests
        sortedPoints = sortperm(weightedDests, rev=true)
        clusterChanged = fill(false, numClusters)
        mergedRequests = Vector{Tuple{Int,Int}}()
        mergedGain = 0.0
        for u in sortedPoints
            gain = weightedDests[u][1]
            if gain <= 0.0
                break
            end
            oldCuID = clusterIDs[u]
            newCuID = weightedDests[u][2]
            if clusterChanged[oldCuID] || clusterChanged[newCuID]
                continue
            end
            push!(mergedRequests, (u, newCuID))
            mergedGain += gain
            clusterChanged[oldCuID] = true
            clusterChanged[newCuID] = true
        end

        numMerged = length(mergedRequests)
        if numMerged == 0
            break
        end

        ###########################################################################################
        # step 3: move points according to the merged requests
        for request in mergedRequests
            u = request[1]
            newCuID = request[2]
            oldCuID = clusterIDs[u]
            delete!(clusters[oldCuID], u)
            push!(clusters[newCuID], u)
            clusterIDs[u] = newCuID
        end

        ###########################################################################################
        # step 4: remove empty clusters
        iLeft = 1
        iRight = numClusters
        while length(clusters[iRight]) == 0 && iRight > iLeft
            iRight -= 1
        end
        while iLeft < iRight
            if length(clusters[iLeft]) == 0
                temp = clusters[iLeft]
                clusters[iLeft] = clusters[iRight]
                clusters[iRight] = temp
                iRight -= 1
                while length(clusters[iRight]) == 0 && iRight > iLeft
                    iRight -= 1
                end
                for u in clusters[iLeft]
                    clusterIDs[u] = iLeft
                end
            end
            iLeft += 1
        end
        if length(clusters[iLeft]) == 0
            numClusters = iLeft - 1
        else
            numClusters = iLeft
        end

        clusters = clusters[1:numClusters]
        println("Aspect Iter $iter: moved $numMerged of $numPoints points to gain $mergedGain, $numClusters clusters remain")
    end # iter

    (clusterIDs, clusters)
end

function getPhrasePairFreqs(nonStopWords::Vector{String}, nonStopWordIDs::Dict{String,Int}, phraseIDGroups::Vector{Vector{Int}})::Dict{Tuple{Int,Int},Int}
    numPhrases = length(nonStopWords)
    idMap = Dict{Int,Int}()
    for (newID, phrase) in enumerate(nonStopWords)
        idMap[nonStopWordIDs[phrase]] = newID
    end

    pairFreqs = Dict{Tuple{Int,Int},Int}()
    for idGroup in phraseIDGroups
        numIDsInGroup = length(idGroup)
        convertedIDs = [idMap[id] for id in idGroup]
        for (j1, id1) in enumerate(convertedIDs)
            for j2 in j1 + 1:numIDsInGroup
                id2 = convertedIDs[j2]
                pair = if id1 < id2
                    (id1, id2)
                else
                    (id2, id1)
                end
                pairFreqs[pair] = get(pairFreqs, pair, 0) + 1 
            end
        end
    end

    pairFreqs
end

function saveNonStopWords(fileName::String, nonStopWords::Vector{String})
    file = open(fileName, "w")
    for phrase in nonStopWords
        println(file, phrase)
    end
    close(file)
end

function saveAspects(fileName::String, aspects::Vector{Set{Int}}, nonStopWords::Vector{String}, cardinalities::Vector{Int})
    file = open(fileName, "w")
    for (aspectID, aspect) in enumerate(aspects)
        for phraseID in aspect
            phrase = nonStopWords[phraseID]
            card = cardinalities[phraseID]
            println(file, "$aspectID $card $phrase")
        end
    end
    close(file)
end

function saveAspects(fileName::String, aspects1::Vector{Set{Int}}, aspects2::Vector{Set{Int}}, nonStopWords::Vector{String}, cardinalities::Vector{Int})
    file = open(fileName, "w")
    for (aspectID2, aspect2) in enumerate(aspects2)
        for aspectID1 in aspect2
            aspect1 = aspects1[aspectID1]
            for phraseID in aspect1
                phrase = nonStopWords[phraseID]
                card = cardinalities[phraseID]
                println(file, "$aspectID2 $card $phrase")
            end
        end
    end
    close(file)
end

function saveAspects(fileName::String, aspects1::Vector{Set{Int}}, aspects2::Vector{Set{Int}}, aspects3::Vector{Set{Int}}, nonStopWords::Vector{String}, cardinalities::Vector{Int})
    file = open(fileName, "w")
    for (aspectID3, aspect3) in enumerate(aspects3)
        for aspectID2 in aspect3
            aspect2 = aspects2[aspectID2]
            for aspectID1 in aspect2
                aspect1 = aspects1[aspectID1]
                for phraseID in aspect1
                    phrase = nonStopWords[phraseID]
                    card = cardinalities[phraseID]
                    println(file, "$aspectID3 $card $phrase")
                end
            end
        end
    end
    close(file)
end

function mergeAspects(aspects::Vector{Set{Int}}, connMat::Array{Float64,2}, pairFreqs::Dict{Tuple{Int,Int},Int})::Tuple{Array{Float64,2},Dict{Tuple{Int,Int},Int}}
    numAspects = length(aspects)
    newConnMat = Array{Float64,2}(undef, (numAspects, numAspects))
    newPairFreqs = Dict{Tuple{Int,Int},Int}()
    for i in 1:numAspects
        for j in 1:numAspects
            if i != j
                conn = 0.0
                freq = 0
                for ui in aspects[i]
                    for uj in aspects[j]
                        conn += connMat[ui,uj]
                        freq += max(get(pairFreqs, (ui, uj), 0), get(pairFreqs, (uj, ui), 0)) 
                    end
                end
                newConnMat[i,j] = conn
                newPairFreqs[(i, j)] = freq ÷ (length(aspects[i]) + length(aspects[j]))
            else
                newConnMat[i,j] = 0.0
            end
        end
    end
    (newConnMat, newPairFreqs)
end

maxIter = 3000
numSingleSideNeighbors = 100
wordVectors = wordvectors("DATR/workspace/NIPS.w2v.model.txt")
nonStopWords, nonStopWordIDs = getNonStopWords(wordVectors)
saveNonStopWords("DATR/workspace/NIPS.nonStopWords.txt", nonStopWords)
phraseOverlaps = getPhraseOverlaps(nonStopWords)
phraseIDGroups = loadPhraseIDGroups("DATR/workspace/NIPS.w2v.training.txt", nonStopWordIDs)
cardinalities = getCardinalities(nonStopWords, nonStopWordIDs, phraseIDGroups)
vectors = normalizeVectors(wordVectors, nonStopWords)
simMat = getSimilarityMatrix(vectors)
connMat1 = getConnectionMatrix(simMat, cardinalities)
pairFreqs1 = getPhrasePairFreqs(nonStopWords, nonStopWordIDs, phraseIDGroups)
aspectIDs1, aspects1 = findAspects(connMat1, pairFreqs1, phraseOverlaps, cardinalities, maxIter)
saveAspects("DATR/workspace/NIPS.aspects.1.txt", aspects1, nonStopWords, cardinalities)
connMat2, pairFreqs2 = mergeAspects(aspects1, connMat1, pairFreqs1)
aspectIDs2, aspects2 = findAspects(connMat2, pairFreqs2, [Dict{Int,Float64}() for i in 1:length(aspectIDs1)], [1 for i in 1:length(aspectIDs1)], maxIter)
saveAspects("DATR/workspace/NIPS.aspects.2.txt", aspects1, aspects2, nonStopWords, cardinalities)
connMat3, pairFreqs3 = mergeAspects(aspects2, connMat2, pairFreqs2)
aspectIDs3, aspects3 = findAspects(connMat3, pairFreqs3, [Dict{Int,Float64}() for i in 1:length(aspectIDs2)], [1 for i in 1:length(aspectIDs2)], maxIter)
saveAspects("DATR/workspace/NIPS.aspects.3.txt", aspects1, aspects2, aspects3, nonStopWords, cardinalities)
