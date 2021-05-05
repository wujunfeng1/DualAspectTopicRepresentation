using Word2Vec
using LinearAlgebra

function loadNonStopWords(fileName::String)::Tuple{Vector{String}, Dict{String, Int}}
    file = open(fileName)
    nonStopWords = Vector{String}()
    nonStopWordIDs = Dict{String, Int}()
    for phrase in readlines(file)
        if phrase != ""
            push!(nonStopWords, phrase)
            nonStopWordIDs[phrase] = length(nonStopWords)
        end
    end
    close(file)
    (nonStopWords, nonStopWordIDs)
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

function loadAspects(fileName::String, nonStopWordIDs::Dict{String,Int})::Tuple{Vector{Set{Int}}, Vector{Int}}
    file = open(fileName)
    aspects = Vector{Set{Int}}()
    numPhrases = length(nonStopWordIDs)
    aspectIDs = Vector{Int}(undef, numPhrases)
    for line in readlines(file)
        fields = split(line)
        if length(fields) != 3
            continue
        end
        aspectID = parse(Int, fields[1])
        card = parse(Int, fields[2])
        phrase = fields[3]
        phraseID = nonStopWordIDs[phrase]
        if aspectID > length(aspects)
            @assert aspectID == length(aspects) + 1
            push!(aspects, Set{Int}())
        end
        push!(aspects[aspectID], phraseID)
        aspectIDs[phraseID] = aspectID
    end
    close(file)
    (aspects, aspectIDs)
end

function findDualAspectPairFreqs(pairFreqs::Dict{Tuple{Int,Int},Int}, aspectIDs::Vector{Int})::Vector{Tuple{Tuple{Int,Int},Int}}
    dualAspectPairFreqs = Vector{Tuple{Tuple{Int,Int},Int}}()
    for (pair, freq) in pairFreqs
        aspectID1 = aspectIDs[pair[1]]
        aspectID2 = aspectIDs[pair[2]]
        if aspectID1 != aspectID2
            push!(dualAspectPairFreqs, (pair, freq)) 
        end
    end
    dualAspectPairFreqs
end

function getDualAspectConn(simMat::Array{Float64}, dualAspectPairFreqs::Vector{Tuple{Tuple{Int,Int},Int}}, pairID1::Int, pairID2::Int)::Float64
    if pairID1 != pairID2
        pairFreq1 = dualAspectPairFreqs[pairID1]
        pair1 = pairFreq1[1]
        freq1 = pairFreq1[2]
        pairFreq2 = dualAspectPairFreqs[pairID2]
        pair2 = pairFreq2[1]
        freq2 = pairFreq2[2]
        sim1 = simMat[pair1[1], pair2[1]] * simMat[pair1[2], pair2[2]]
        sim2 = simMat[pair1[1], pair2[2]] * simMat[pair1[2], pair2[1]]
        return min(sim1, sim2) * freq1 * freq2
    else
        return 0.0
    end
end

function sparsifyDualAspectConn(simMat::Array{Float64}, dualAspectPairFreqs::Vector{Tuple{Tuple{Int,Int},Int}}, numSingleSideNeighbors::Int)::Vector{Dict{Int,Float64}}
    numPoints = length(dualAspectPairFreqs)
    singleSideConns = Vector{Dict{Int,Float64}}(undef, numPoints)
    numCPUs = length(Sys.cpu_info())
    jobs = Channel{Tuple{Int,Int}}(numCPUs)
    Segments = Vector{Tuple{Int,Int}}
    jobOutputs = Channel{Tuple{Segments,Vector{Vector{Dict{Int,Float64}}}}}(numCPUs)

    function makeJobs(batchSize::Int)
        for i in 1:batchSize:numPoints
            put!(jobs, (i, min(i + batchSize - 1, numPoints)))
        end
    end

    function runJob(iCPU::Int)
        mySegments = Segments()
        myOutputs = Vector{Vector{Dict{Int,Float64}}}()
        conn1s = Vector{Float64}(undef, numPoints)
        for job in jobs
            push!(mySegments, job)
            segOutput = Vector{Dict{Int,Float64}}(undef, job[2] - job[1] + 1)
            # println("cpu $iCPU: computing sumConn for $job")
            for pairID1 in job[1]:job[2]
                sumConn1 = 0.0
                for pairID2 in 1:numPoints
                    conn1s[pairID2] = getDualAspectConn(simMat, dualAspectPairFreqs, pairID1, pairID2)
                end
                perm = sortperm(conn1s, rev=true)
                neighbors = Dict{Int, Float64}()
                for i in 1:numSingleSideNeighbors
                    neighbors[perm[i]] = conn1s[perm[i]]
                end
                segOutput[pairID1 - job[1] + 1] = neighbors
                if pairID1 % 1000 == 0
                    println("cpu $iCPU: $pairID1 of $numPoints have been processed")
                end 
            end
            push!(myOutputs, segOutput)
            # println("cpu $iCPU: finish computing sumConn for $job") 
        end # job
        put!(jobOutputs, (mySegments, myOutputs))
    end # runJob

    bind(jobs, @async makeJobs(100))
    for iCPU in 1:numCPUs
        Threads.@spawn runJob(iCPU)
    end

    for iCPU in 1:numCPUs
        segmentedOutputs = take!(jobOutputs)
        for (segID, segment) in enumerate(segmentedOutputs[1])
            singleSideConns[segment[1]:segment[2]] = segmentedOutputs[2][segID]
        end
    end

    doubleSideConns = [Dict{Int,Float64}() for i in 1:numPoints]
    for i in 1: numPoints
        for (j, conn) in singleSideConns[i]
            doubleSideConns[i][j] = conn
            doubleSideConns[j][i] = conn
        end
    end
    doubleSideConns
end

function saveDualAspectPairFreqs(fileName::String, dualAspectPairFreqs::Vector{Tuple{Tuple{Int,Int},Int}})
    file = open(fileName, "w")
    for pairFreq in dualAspectPairFreqs
        pair = pairFreq[1]
        freq = pairFreq[2]
        i = pair[1]
        j = pair[2]
        println(file, "$i $j $freq")
    end
    close(file)
end

function saveConns(fileName::String, sparseConns::Vector{Dict{Int,Float64}})
    file = open(fileName, "w")
    for (i, connsI) in enumerate(sparseConns)
        for (j, conn) in connsI
            println(file, "$i $j $conn")
        end
    end
    close(file)
end

numSingleSideNeighbors = 100
wordVectors = wordvectors("DualAspectTopicRepresentation/workspace/NIPS.awe.model.txt")
nonStopWords, nonStopWordIDs = loadNonStopWords("DualAspectTopicRepresentation/workspace/NIPS.nonStopWords.txt")
phraseIDGroups = loadPhraseIDGroups("DualAspectTopicRepresentation/workspace/NIPS.awe.training.txt", nonStopWordIDs)
pairFreqs = getPhrasePairFreqs(nonStopWords, nonStopWordIDs, phraseIDGroups)
vectors = normalizeVectors(wordVectors, nonStopWords)
simMat = getSimilarityMatrix(vectors)
aspects, aspectIDs = loadAspects("DualAspectTopicRepresentation/workspace/NIPS.aspects.1.txt", nonStopWordIDs)
dualAspectPairFreqs = findDualAspectPairFreqs(pairFreqs, aspectIDs)
saveDualAspectPairFreqs("DualAspectTopicRepresentation/workspace/NIPS.DAPairFreq.txt", dualAspectPairFreqs)
println("sparsifying connections ...")
sparseConns = sparsifyDualAspectConn(simMat, dualAspectPairFreqs, numSingleSideNeighbors)
saveConns("DualAspectTopicRepresentation/workspace/NIPS.DAPairConns.txt", sparseConns)

