function loadNonStopWords(fileName::String)::Tuple{Vector{String},Dict{String,Int}}
    file = open(fileName)
    nonStopWords = Vector{String}()
    nonStopWordIDs = Dict{String,Int}()
    for phrase in readlines(file)
        if phrase != ""
            push!(nonStopWords, phrase)
            nonStopWordIDs[phrase] = length(nonStopWords)
        end
    end
    close(file)
    (nonStopWords, nonStopWordIDs)
end

function loadDualAspectPairFreqs(fileName::String)::Vector{Tuple{Tuple{Int,Int},Int}}
    file = open(fileName)
    dualAspectPairFreqs = Vector{Tuple{Tuple{Int,Int},Int}}()
    for line in readlines(file)
        if line == ""
            continue
        end
        fields = split(line)
        @assert length(fields) == 3
        i = parse(Int, fields[1])
        j = parse(Int, fields[2])
        freq = parse(Int, fields[3])
        push!(dualAspectPairFreqs, ((i, j), freq))
    end
    close(file)
    dualAspectPairFreqs
end

function loadConns(fileName::String, n::Int)::Vector{Dict{Int,Float64}}
    sparseConns = [Dict{Int,Float64}() for i in 1:n]
    file = open(fileName)
    for line in readlines(file)
        if line == ""
            continue
        end
        fields = split(line)
        @assert length(fields) == 3
        i = parse(Int, fields[1])
        j = parse(Int, fields[2])
        conn = parse(Float64, fields[3])
        sparseConns[i][j] = conn
    end
    close(file)
    sparseConns
end

function deltaModularity(sparseConns::Vector{Dict{Int,Float64}}, sumConn::Vector{Float64}, totalConn::Float64, u::Int, oldCu::Set{Int}, newCu::Set{Int})
    result = 0.0
    for v in oldCu
        if v != u
            wuv = sumConn[u] * sumConn[v] / totalConn
            result -= get(sparseConns[u], v, 0.0) - wuv
        end
    end
    for v in newCu
        wuv = sumConn[u] * sumConn[v] / totalConn
        result += get(sparseConns[u], v, 0.0) - wuv
    end
    result
end

function sumDualAspectConn(sparseConns::Vector{Dict{Int,Float64}})::Vector{Float64}
    numPoints = length(sparseConns)
    sumConn = Vector{Float64}(undef, numPoints)
    for i in 1:numPoints
        sumI = 0.0
        for (j, conn) in sparseConns[i]
            sumI += conn
        end
        sumConn[i] = sumI
    end
    sumConn
end

function findDualAspectTopics(sparseConns::Vector{Dict{Int,Float64}}, maxIter::Int)::Tuple{Vector{Int},Vector{Set{Int}}}
    sumConn = sumDualAspectConn(sparseConns)
    totalConn = sum(sumConn)
    numPoints = length(sumConn)
    clusters = collect(Set{Int}(i) for i in 1:numPoints)
    clusterIDs = collect(1:numPoints)
    numClusters = numPoints

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
                    neighborClusterIDs = Set{Int}()
                    for (neighbor, conn) in sparseConns[u]
                        push!(neighborClusterIDs, clusterIDs[neighbor])
                    end
                    neighborClusterIDs = [clusterID for clusterID in neighborClusterIDs]
                    numNeighborClusters = length(neighborClusterIDs)
                    attractions = fill(0.0, numNeighborClusters)

                    for (i, newCuID) in enumerate(neighborClusterIDs)
                        if newCuID != oldCuID
                            newCu = clusters[newCuID] 
                            attractions[i] = max(0.0, deltaModularity(sparseConns, sumConn, totalConn, u, oldCu, newCu))
                        end
                    end
                    # println("cpu $iCPU: attractions of $u is $attractions")
                    sumAttractions = sum(attractions)
                    thres = rand() * sumAttractions
                    partialSumAttr = 0.0
                    weightedDest = (0.0, oldCuID)
                    for dest in 1:numNeighborClusters
                        attr = attractions[dest]
                        partialSumAttr += attr
                        if attr > 0.0 && partialSumAttr >= thres
                            weightedDest = (attr, neighborClusterIDs[dest])
                            break
                        end
                    end
                    jobWeightedDests[u - job[1] + 1] = weightedDest
                    # if u % 10000 == 0
                    #     println("cpu $iCPU: $u of $numPoints have been processed")
                    # end 
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
        #println("Topic Iter $iter: moved $numMerged of $numPoints points to gain $mergedGain, $numClusters clusters remain")
    end # iter

    (clusterIDs, clusters)
end

function loadAspects(fileName::String, nonStopWordIDs::Dict{String,Int})::Tuple{Vector{Set{Int}},Vector{Int}}
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

function saveTopics(fileName::String, topics::Vector{Set{Int}}, nonStopWords::Vector{String}, dualAspectPairFreqs::Vector{Tuple{Tuple{Int,Int},Int}})
    file = open(fileName, "w")
    for (topicID, topic) in enumerate(topics)
        for pairID in topic
            pairFreq = dualAspectPairFreqs[pairID]
            pair = pairFreq[1]
            freq = pairFreq[2]
            phrase1 = nonStopWords[pair[1]]
            phrase2 = nonStopWords[pair[2]]
            println(file, "$topicID $freq $phrase1 $phrase2")
        end
    end
    close(file)
end

function divideGroups(dualAspectPairFreqs::Vector{Tuple{Tuple{Int,Int},Int}}, regions::Vector{Vector{Int}}, aspectIDs::Vector{Int}, maxPairsInGroup::Int)::Vector{Vector{Int}}
    groups = Vector{Vector{Int}}()
    for region in regions
        if length(region) < maxPairsInGroup
            push!(groups, region)
            continue
        end

        subRegions = Dict{Tuple{Int,Int},Vector{Int}}()
        for pairID in region
            pairFreq = dualAspectPairFreqs[pairID]
            pair = pairFreq[1]
            i = pair[1]
            j = pair[2]
            aspectI = aspectIDs[i]
            aspectJ = aspectIDs[j]
            if haskey(subRegions, (aspectI, aspectJ))
                push!(subRegions[(aspectI, aspectJ)], pairID)
            else
                subRegions[(aspectI, aspectJ)] = [pairID]
            end
        end
        
        for (aspectPair, group) in subRegions
            push!(groups, group)
        end
    end
    groups
end

function getRegionalConns(sparseConns::Vector{Dict{Int,Float64}}, region::Vector{Int})::Vector{Dict{Int,Float64}}
    regionalConns = [Dict{Int,Float64}() for i in region]
    idMap = Dict{Int,Int}()
    for (localID, globalID) in enumerate(region)
        idMap[globalID] = localID
    end
    for (localID, globalID) in enumerate(region)
        for (globalNeighbor, conn) in sparseConns[globalID]
            if haskey(idMap, globalNeighbor)
                regionalConns[localID][idMap[globalNeighbor]] = conn
            end
        end 
    end
    regionalConns
end

maxIter = 1000
maxPairsInGroup = 10000
nonStopWords, nonStopWordIDs = loadNonStopWords("DATR/workspace/NIPS.nonStopWords.txt")
dualAspectPairFreqs = loadDualAspectPairFreqs("DATR/workspace/NIPS.DAPairFreq.txt")
sparseConns = loadConns("DATR/workspace/NIPS.DAPairConns.txt", length(dualAspectPairFreqs))
aspect1, aspectIDs1 = loadAspects("DATR/workspace/NIPS.aspects.1.txt", nonStopWordIDs)
aspect2, aspectIDs2 = loadAspects("DATR/workspace/NIPS.aspects.2.txt", nonStopWordIDs)
aspect3, aspectIDs3 = loadAspects("DATR/workspace/NIPS.aspects.3.txt", nonStopWordIDs)
regions0 = [[i for i in 1:length(dualAspectPairFreqs)]]
regions1 = divideGroups(dualAspectPairFreqs, regions0, aspectIDs3, maxPairsInGroup)
regions2 = divideGroups(dualAspectPairFreqs, regions1, aspectIDs2, maxPairsInGroup)
regions3 = divideGroups(dualAspectPairFreqs, regions2, aspectIDs1, maxPairsInGroup)
topics = Vector{Set{Int}}()
numRegions = length(regions3)
for (iRegion, region) in enumerate(regions3)
    println("finding topics in region $iRegion of $numRegions ...")
    regionalConns = getRegionalConns(sparseConns, region)
    regionalTopicIDs, regionalTopics = findDualAspectTopics(regionalConns, maxIter)
    for regionalTopic in regionalTopics
        topic = Set{Int}()
        for regionalID in regionalTopic 
            push!(topic, region[regionalID])
        end
        push!(topics, topic)
    end
end
saveTopics("DATR/workspace/NIPS.topics.txt", topics, nonStopWords, dualAspectPairFreqs)
