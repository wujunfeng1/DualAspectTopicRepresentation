function loadYears(fileName::String)::Vector{Int}
    file = open(fileName)
    years = Vector{Int}()
    for line in readlines(file)
        if line == ""
            continue
        end
        fields = split(line, ". ")
        try 
            year = parse(Int, fields[1])
            push!(years, year)
        catch
            println(line)
        end
    end
    close(file)
    years
end

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

function loadTopics(fileName::String, nonStopWordIDs::Dict{String,Int}, dualAspectPairFreqs::Vector{Tuple{Tuple{Int,Int},Int}})::Tuple{Vector{Int},Vector{Set{Int}}}
    pairIDs = Dict{Tuple{Int,Int},Int}()
    for (pairID, pairFreq) in enumerate(dualAspectPairFreqs)
        pair = pairFreq[1]
        pairIDs[pair] = pairID
    end
    numPairs = length(dualAspectPairFreqs)
    topicIDs = Vector{Int}(undef, numPairs)
    topics = Vector{Set{Int}}()
    file = open(fileName)
    for line in readlines(file)
        if line == ""
            continue
        end
        fields = split(line)
        @assert length(fields) == 4
        topicID = parse(Int, fields[1])
        freq = parse(Int, fields[2])
        phrase1 = fields[3]
        phrase2 = fields[4]
        phraseID1 = nonStopWordIDs[phrase1]
        phraseID2 = nonStopWordIDs[phrase2]
        pairID = pairIDs[(phraseID1, phraseID2)]
        @assert freq == dualAspectPairFreqs[pairID][2]
        topicIDs[pairID] = topicID
        if topicID > length(topics)
            @assert topicID == length(topics) + 1
            push!(topics, Set{Int}())
        end
        push!(topics[topicID], pairID)
    end
    close(file)

    (topicIDs, topics)
end

function getPairYearDistributions(dualAspectPairFreqs::Vector{Tuple{Tuple{Int,Int},Int}}, phraseIDGroups::Vector{Vector{Int}}, years::Vector{Int})::Vector{Dict{Int,Int}}
    numPairs = length(dualAspectPairFreqs)
    pairIDs = Dict{Tuple{Int,Int},Int}()
    for (pairID, pairFreq) in enumerate(dualAspectPairFreqs)
        pair = pairFreq[1]
        pairIDs[pair] = pairID
    end

    pairYearDistributions = [Dict{Int,Int}() for i in 1:numPairs]
    for (groupID, phraseIDGroup) in enumerate(phraseIDGroups)
        year = years[groupID]
        groupSize = length(phraseIDGroup)
        groupPairIDs = Set{Int}()
        for i in 1:groupSize
            for j in 1:groupSize
                if i == j
                    continue
                end
                pair = (phraseIDGroup[i], phraseIDGroup[j])
                if haskey(pairIDs, pair)
                    pairID = pairIDs[pair]
                    push!(groupPairIDs, pairID)
                end
            end
        end
        for pairID in groupPairIDs
            pairYearDistributions[pairID][year] = get(pairYearDistributions[pairID], year, 0) + 1
        end
    end

    pairYearDistributions
end

function getTopicYearDistributions(topicIDs::Vector{Int}, topics::Vector{Set{Int}}, pairYearDistributions::Vector{Dict{Int,Int}})::Vector{Dict{Int,Int}}
    numTopics = length(topics)
    topicYearDistributions = [Dict{Int,Int}() for i in 1:numTopics]
    for (pairID, yearDistributions) in enumerate(pairYearDistributions)
        topicID = topicIDs[pairID]
        for (year, freq) in yearDistributions
            topicYearDistributions[topicID][year] = get(topicYearDistributions[topicID], year, 0) + freq
        end
    end
    topicYearDistributions
end

function getFiveYearSlopeSums(topicYearDistributions::Vector{Dict{Int,Int}}, toYear::Int)::Vector{Float64}
    numTopics = length(topicYearDistributions)
    slopeSums = Vector{Float64}(undef, numTopics)
    for (topicID, yearDistributions) in enumerate(topicYearDistributions)
        topicSlopeSum = 0.0
        for year in toYear - 4:toYear
            slope = 0.3 * (get(yearDistributions, year, 0) - get(yearDistributions, year - 3, 0)) + 0.1 *  (get(yearDistributions, year - 1, 0) - get(yearDistributions, year - 2, 0))
            topicSlopeSum += slope
        end
        slopeSums[topicID] = topicSlopeSum
    end
    slopeSums
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

function findTop3Representations(topics::Vector{Set{Int}}, topicID::Int, dualAspectPairFreqs::Vector{Tuple{Tuple{Int,Int},Int}}, sparseConns::Vector{Dict{Int,Float64}})::NTuple{3, Int}
    topic = topics[topicID]
    numPairsInTopic = length(topic)
    topicPairIDs = [pairID for pairID in topic]
    pairScores = Vector{Float64}(undef, numPairsInTopic)
    for (i, pairID) in enumerate(topicPairIDs)
        freq = dualAspectPairFreqs[pairID][2]
        pairConns = sparseConns[pairID]
        sumPairConns = 0.0
        for (neighbor, conn) in pairConns
            sumPairConns += conn
        end
        pairScores[i] = freq^2 + 0.5 * sumPairConns
    end
    pairRanks = sortperm(pairScores, rev=true)
    (topicPairIDs[pairRanks[1]], topicPairIDs[pairRanks[2]], topicPairIDs[pairRanks[3]])
end

function saveTopKTopics(fileName::String, K::Int, topics::Vector{Set{Int}}, topicSlopeSums::Vector{Float64}, dualAspectPairFreqs::Vector{Tuple{Tuple{Int,Int},Int}}, sparseConns::Vector{Dict{Int,Float64}}, nonStopWords::Vector{String})
    file = open(fileName, "w")
    topicRanks = sortperm(topicSlopeSums, rev=true)
    for k = 1:K
        topicID = topicRanks[k]
        score = topicSlopeSums[topicID]
        pairID1, pairID2, pairID3 = findTop3Representations(topics, topicID, dualAspectPairFreqs, sparseConns)
        pair1 = dualAspectPairFreqs[pairID1][1]
        phrase11 = nonStopWords[pair1[1]]
        phrase12 = nonStopWords[pair1[2]]
        pair2 = dualAspectPairFreqs[pairID2][1]
        phrase21 = nonStopWords[pair2[1]]
        phrase22 = nonStopWords[pair2[2]]
        pair3 = dualAspectPairFreqs[pairID3][1]
        phrase31 = nonStopWords[pair3[1]]
        phrase32 = nonStopWords[pair3[2]]
        println(file, "$topicID $score $phrase11 $phrase12 $phrase21 $phrase22 $phrase31 $phrase32")
    end
    close(file)
end

years = loadYears("DATR/data/NIPS.txt")
nonStopWords, nonStopWordIDs = loadNonStopWords("DATR/workspace/NIPS.nonStopWords.txt")
dualAspectPairFreqs = loadDualAspectPairFreqs("DATR/workspace/NIPS.DAPairFreq.txt")
phraseIDGroups = loadPhraseIDGroups("DATR/workspace/NIPS.w2v.training.txt", nonStopWordIDs)
topicIDs, topics = loadTopics("DATR/workspace/NIPS.topics.txt", nonStopWordIDs, dualAspectPairFreqs)
pairYearDistributions = getPairYearDistributions(dualAspectPairFreqs, phraseIDGroups, years)
topicYearDistributions = getTopicYearDistributions(topicIDs, topics, pairYearDistributions)
topicSlopeSums = getFiveYearSlopeSums(topicYearDistributions, 2020)
sparseConns = loadConns("DATR/workspace/NIPS.DAPairConns.txt", length(dualAspectPairFreqs))
saveTopKTopics("DATR/workspace/NIPS.Top100Topics.txt", 100, topics, topicSlopeSums, dualAspectPairFreqs, sparseConns, nonStopWords)