using Word2Vec
using LinearAlgebra
include("x1 io.jl")
include("x2 preprocessing.jl")

function getSimilarityMatrix(vectors::Array{Float64,2})::Array{Float64,2}
    0.5 .+ 0.5 .* (vectors' * vectors)
end

function getConnectionMatrix(simMat::Array{Float64,2}, cardinalities::Vector{Int})::Array{Float64,2}
    simMat .* (cardinalities * cardinalities')
end

function deltaModularity(connMat::Array{Float64,2}, sumConn::Vector{Float64}, totalConn::Float64, 
    u::Int, oldCu::Set{Int}, newCu::Set{Int})
    result = 0.0
    for v in oldCu
        if v != u
            wuv = sumConn[u] * sumConn[v] / totalConn
            result -= connMat[u,v] - wuv
        end
    end
    for v in newCu
        wuv = sumConn[u] * sumConn[v] / totalConn
        result += connMat[u,v] - wuv
    end
    result
end

function findAspects(connMat::Array{Float64,2}, maxIter::Int)::Tuple{Vector{Int},Vector{Set{Int}}}
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
                            attractions[newCuID] = max(0.0, deltaModularity(connMat, sumConn, totalConn, u, oldCu, newCu))
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

maxIter = 3000
numSingleSideNeighbors = 100
wordVectors = wordvectors("DualAspectTopicRepresentation/workspace/NIPS.awe.model.txt")
nonStopWords, nonStopWordIDs = getNonStopWords(wordVectors)
saveNonStopWords("DualAspectTopicRepresentation/workspace/NIPS.nonStopWords.txt", nonStopWords)
phraseIDGroups = loadPhraseIDGroups("DualAspectTopicRepresentation/workspace/NIPS.awe.training.txt", nonStopWordIDs)
cardinalities = getCardinalities(nonStopWords, nonStopWordIDs, phraseIDGroups)
#vectors = normalizeVectors(wordVectors, nonStopWords)
vectors = getVectors(wordVectors, nonStopWords)
simMat = getSimilarityMatrix(vectors)
connMat1 = getConnectionMatrix(simMat, cardinalities)
pairFreqs1 = getPhrasePairFreqs(nonStopWords, nonStopWordIDs, phraseIDGroups)
aspectIDs1, aspects1 = findAspects(connMat1, maxIter)
saveAspects("DualAspectTopicRepresentation/workspace/NIPS.aspects.4.txt", aspects1, nonStopWords, cardinalities)
connMat2 = mergeAspects(aspects1, connMat1)
aspectIDs2, aspects2 = findAspects(connMat2, maxIter)
saveAspects("DualAspectTopicRepresentation/workspace/NIPS.aspects.5.txt", aspects1, aspects2, nonStopWords, cardinalities)
connMat3 = mergeAspects(aspects2, connMat2)
aspectIDs3, aspects3 = findAspects(connMat3, maxIter)
saveAspects("DualAspectTopicRepresentation/workspace/NIPS.aspects.6.txt", aspects1, aspects2, aspects3, nonStopWords, cardinalities)
