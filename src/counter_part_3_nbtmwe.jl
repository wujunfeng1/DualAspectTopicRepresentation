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

function getNonStopWords(wordVectors::WordVectors)::Tuple{Vector{String}, Dict{String,Int}}
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

struct IDPair
    i::Int
    j::Int
end

mutable struct Biterm
    pair::IDPair
    topicID::Int
    gpuFlag::Bool
end

struct NBTMWEParameters
    numTopics::Int
    maxIter::Int 
    simThres::Float64
    alpha::Float64
    beta::Float64
    phrases::Vector{String}
    phraseIDs::Dict{String,Int}
    docs::Vector{Vector{Int}}
end

mutable struct NBTMWE
    parameters::NBTMWEParameters
    biterms::Vector{Biterm}
    similarPhraseIDs::Dict{IDPair, Vector{Int}}
    numBitermsOfTopic::Vector{Int}
    numRelatedOfTopic::Vector{Int}
    phraseFreqsOfTopic::Array{Float64,2} 
end

function saveNBTMWE(fileName::String, btm::NBTMWE)
    file = open(fileName, "w")
    param = btm.parameters
    numPhrases = length(param.phrases)
    numDocs = length(param.docs)
    numBiterms = length(btm.biterms)
    numSimilar = length(btm.similarPhraseIDs)
    println(file, "$(param.numTopics) $(param.maxIter) $(param.simThres) $(param.alpha) $(param.beta) $numPhrases $numDocs $numBiterms $numSimilar")
    for phraseID in 1:numPhrases
        println(file, param.phrases[phraseID])
    end
    for docID in 1:numDocs
        doc = param.docs[docID]
        docSize = length(doc)
        for i in 1:docSize
            if i > 1
                print(file, " ")
            end
            print(file, doc[i])
        end
        print(file, "\n")
    end
    for bitermID in 1:numBiterms
        biterm = btm.biterms[bitermID]
        i = biterm.pair.i
        j = biterm.pair.j
        topicID = biterm.topicID
        gpuFlag = biterm.gpuFlag
        println(file, "$i $j $topicID $gpuFlag")
    end
    for (pair, relatedIDs) in btm.similarPhraseIDs
        print(file, "$(pair.i) $(pair.j)")
        for id in relatedIDs
            print(file, " $id")
        end
        print(file, "\n")
    end
    for topicID in 1:param.numTopics
        if topicID > 1
            print(file, " ")
        end
        print(file, btm.numBitermsOfTopic[topicID])
    end
    print(file, "\n")
    for topicID in 1:param.numTopics
        if topicID > 1
            print(file, " ")
        end
        print(file, btm.numRelatedOfTopic[topicID])
    end
    print(file, "\n")
    for topicID in 1:param.numTopics
        for phraseID in 1:numPhrases
            if phraseID > 1
                print(file, " ")
            end
            print(file, btm.phraseFreqsOfTopic[topicID,phraseID])
        end
        print(file, "\n")
    end
    close(file)
end

function newNBTMWE(fileName::String, nonStopWords::Vector{String}, vectors::Array{Float64,2}, numTopics::Int, maxIter::Int, simThres::Float64, alpha::Float64, beta::Float64)::NBTMWE
    phrases = nonStopWords
    phraseIDs = Dict{String,Int}()
    for (phraseID, phrase) in enumerate(phrases)
        phraseIDs[phrase] = phraseID
    end

    docs = Vector{Vector{Int}}()
    file = open(fileName)
    for line in readlines(file)
        phrasesAtLine = split(line)
        doc = Vector{Int}()
        for phrase in phrasesAtLine
            if haskey(phraseIDs, phrase)
                push!(doc, phraseIDs[phrase])
            end
        end
        if length(doc) > 0
            push!(docs, doc)
        end
    end

    numPhrases = length(phrases)
    numBitermsOfTopic = fill(0, numTopics)
    numRelatedOfTopic = fill(0, numTopics)
    phraseFreqsOfTopic = fill(0.0, (numTopics, numPhrases))
    biterms = Vector{Biterm}()
    for doc in docs
        docSize = length(doc)
        for i in 1:docSize
            for j in 1:docSize
                if i != j
                    topicID = rand(1:numTopics)
                    push!(biterms, Biterm(IDPair(doc[i], doc[j]), topicID, false))
                    numBitermsOfTopic[topicID] += 1
                    phraseFreqsOfTopic[topicID, doc[i]] += 1.0
                    phraseFreqsOfTopic[topicID, doc[j]] += 1.0 
                end
            end
        end
    end

    similarPhraseIDs = Dict{IDPair, Vector{Int}}()
    numBiterms = length(biterms)
    println("building similar phrases list...")
    simMat = 0.5 .+ 0.5 .* (vectors' * vectors)
    for (bitermID, biterm) in enumerate(biterms)
        if haskey(similarPhraseIDs, biterm.pair)
            continue
        end
        relatedIDs = Vector{Int}()
        for phraseID in 1:numPhrases
            if phraseID == biterm.pair.i || phraseID == biterm.pair.j
                continue
            end
            sim1 = simMat[biterm.pair.i, phraseID]
            sim2 = simMat[biterm.pair.j, phraseID]
            if sim1 >= simThres && sim2 >= simThres
                push!(relatedIDs, phraseID)
            end
        end
        similarPhraseIDs[biterm.pair] = relatedIDs
        if bitermID % 10000 == 0
            println("similar phrase list $bitermID of $numBiterms")
        end
    end
    println("finish building similar phrases list")

    NBTMWE(NBTMWEParameters(numTopics, maxIter, simThres, alpha, beta, phrases, phraseIDs, docs), biterms, similarPhraseIDs, numBitermsOfTopic, numRelatedOfTopic, phraseFreqsOfTopic)
end

function loadNBTMWE(fileName::String)::NBTMWE
    file = open(fileName)
    firstLine = readline(file)
    fieldsOfFirstLine = split(firstLine)
    @assert length(fieldsOfFirstLine) == 9
    numTopics = parse(Int, fieldsOfFirstLine[1])
    maxIter = parse(Int, fieldsOfFirstLine[2])
    simThres = parse(Float64, fieldsOfFirstLine[3])
    alpha = parse(Float64, fieldsOfFirstLine[4])
    beta = parse(Float64, fieldsOfFirstLine[5])
    numPhrases = parse(Int, fieldsOfFirstLine[6])
    numDocs = parse(Int, fieldsOfFirstLine[7])
    numBiterms = parse(Int, fieldsOfFirstLine[8])
    numSimilar = parse(Int, fieldsOfFirstLine[9])
    phrases = Vector{String}()
    phraseIDs = Dict{String,Int}()
    for phraseID in 1:numPhrases
        phrase = readline(file)
        push!(phrases, phrase)
        phraseIDs[phrase] = phraseID
    end
    docs = Vector{Vector{Int}}()
    for docID in 1:numDocs
        doc = Vector{Int}()
        lineOfDoc = readline(file)
        fieldsOfDoc = split(lineOfDoc)
        docSize = length(fieldsOfDoc)
        for i in 1:docSize
            phraseID = parse(Int, fieldsOfDoc[i])
            push!(doc, phraseID)
        end
    end
    biterms = Vector{Biterm}(undef, numBiterms)
    for bitermID in 1:numBiterms
        lineOfBiterm = readline(file)
        fieldsOfBiterm = split(lineOfBiterm)
        @assert length(fieldsOfBiterm) == 4
        i = parse(Int, fieldsOfBiterm[1])
        j = parse(Int, fieldsOfBiterm[2])
        topicID = parse(Int, fieldsOfBiterm[3])
        gpuFlag = parse(Bool, fieldsOfBiterm[4])
        biterm = Biterm(IDPair(i, j), topicID, gpuFlag)
        biterms[bitermID] = biterm
    end
    similarPhraseIDs = Dict{IDPair, Vector{Int}}()
    for iSimilar in 1:numSimilar
        lineOfSimilar = readline(file)
        fieldsOfSimilar = split(lineOfSimilar)
        @assert length(fieldsOfSimilar) >= 2
        i = parse(Int, fieldsOfSimilar[1])
        j = parse(Int, fieldsOfSimilar[2])
        relatedIDs = Vector{Int}()
        for k in 3:length(fieldsOfSimilar)
            id = parse(Int, fieldsOfSimilar[k])
            push!(relatedIDs, id)
        end
        similarPhraseIDs[IDPair(i,j)] = relatedIDs
    end
    numBitermsOfTopic = Vector{Int}(undef, numTopics)
    lineOfNumBiterms = readline(file)
    fieldsOfNumBiterms = split(lineOfNumBiterms)
    @assert length(fieldsOfNumBiterms) == numTopics
    for topicID in 1:numTopics
        numBitermsOfTopic[topicID] = parse(Int, fieldsOfNumBiterms[topicID])
    end
    numRelatedOfTopic = Vector{Int}(undef, numTopics)
    lineOfNumRelated = readline(file)
    fieldsOfNumRelated = split(lineOfNumRelated)
    @assert length(fieldsOfNumRelated) == numTopics
    for topicID in 1:numTopics
        numRelatedOfTopic[topicID] = parse(Int, fieldsOfNumRelated[topicID])
    end
    phraseFreqsOfTopic = Array{Float64,2}(undef, (numTopics, numPhrases))
    for topicID in 1:numTopics
        lineOfTopic = readline(file)
        fieldsOfTopic = split(lineOfTopic)
        @assert length(fieldsOfTopic) == numPhrases
        for phraseID in 1:numPhrases
            freq = parse(Float64, fieldsOfTopic[phraseID])
            phraseFreqsOfTopic[topicID, phraseID] = freq
        end
    end
    close(file)

    NBTMWE(NBTMWEParameters(numTopics, maxIter, simThres, alpha, beta, phrases, phraseIDs, docs), biterms, phraseFreqsOfTopic, numBitermsOfTopic, numRelatedOfTopic, phraseFreqsOfTopic)
end

function computeProbTopicBiterm(btm::NBTMWE, biterm::Biterm)::Vector{Float64}
    numTopics = btm.parameters.numTopics
    numPhrases = length(btm.parameters.phrases)
    numBiterms = length(btm.biterms)
    numTotalRelated = sum(btm.numRelatedOfTopic)
    numBitermRelated = if biterm.gpuFlag
        length(btm.similarPhraseIDs[biterm.pair])
    else
        0
    end
    alpha = btm.parameters.alpha
    beta = btm.parameters.beta
    prob = Vector{Float64}(undef, numTopics)
    for topicID in 1:numTopics
        wRelated = 0.5 * btm.parameters.simThres * (numTotalRelated - numBitermRelated)
        if topicID != biterm.topicID
            w = 1.0 / (2 * btm.numBitermsOfTopic[topicID] + numPhrases * beta)
            prob1 = (btm.phraseFreqsOfTopic[topicID, biterm.pair.i] + beta) * w
            prob2 = (btm.phraseFreqsOfTopic[topicID, biterm.pair.j] + beta) * w
            wTopicRelated = 0.5 * btm.parameters.simThres * btm.numRelatedOfTopic[topicID]
            probTopic = (btm.numBitermsOfTopic[topicID] + wTopicRelated + alpha) / (numBiterms - 1 + wRelated + numTopics * alpha)
            prob[topicID] = prob1 * prob2 * probTopic
        else
            w = 1.0 / (2 * btm.numBitermsOfTopic[topicID] - 2 + numPhrases * btm.parameters.beta)
            prob1 = (btm.phraseFreqsOfTopic[topicID, biterm.pair.i] - 1 + btm.parameters.beta) * w 
            prob2 = (btm.phraseFreqsOfTopic[topicID, biterm.pair.j] - 1 + btm.parameters.beta) * w
            wTopicRelated = 0.5 * btm.parameters.simThres * (btm.numRelatedOfTopic[topicID] - numBitermRelated)
            probTopic = (btm.numBitermsOfTopic[topicID] - 1 + wTopicRelated + alpha) / (numBiterms - 1 + wRelated + numTopics * alpha)
            prob[topicID] = prob1 * prob2 * probTopic
        end
    end
    c = 1.0 / sum(prob)
    prob .*= c
    prob
end

function sampleProb(prob::Vector{Float64})::Int
    x = rand()
    partialSum = 0.0
    for i in 1:length(prob)
        partialSum += prob[i]
        if partialSum >= x
            return i
        end
    end
    length(prob)
end

function sampleGPUFlag(prob::Vector{Float64}, topicID::Int)::Bool
    maxProb = max(prob...)
    ratio = prob[topicID] / maxProb
    x = rand()
    ratio > x
end

function trainNBTMWE!(btm::NBTMWE, fileName::String)
    numBiterms = length(btm.biterms)
    numCPUs = length(Sys.cpu_info())
    Segments = Vector{Tuple{Int,Int}}
    TopFlag = Tuple{Int, Bool}
    for iter in 1:btm.parameters.maxIter
        jobs = Channel{Tuple{Int,Int}}(numCPUs)
        jobOutputs = Channel{Tuple{Segments,Vector{Vector{TopFlag}}}}(numCPUs)

        function makeJobs(batchSize::Int)
            for i in 1:batchSize:numBiterms
                put!(jobs, (i, min(i + batchSize - 1, numBiterms)))
            end
        end

        function runJob(iCPU::Int)
            mySegments = Segments()
            myOutputs = Vector{Vector{TopFlag}}()
            for job in jobs
                push!(mySegments, job)
                jobOutput = Vector{TopFlag}(undef, job[2] - job[1] + 1)
                for bitermID in job[1]:job[2]
                    prob = computeProbTopicBiterm(btm, btm.biterms[bitermID])
                    newTopID = sampleProb(prob)
                    newFlag = sampleGPUFlag(prob, newTopID)
                    jobOutput[bitermID - job[1] + 1] = (newTopID,newFlag)
                end
                push!(myOutputs, jobOutput)
            end
            put!(jobOutputs, (mySegments, myOutputs))
        end

        bind(jobs, @async makeJobs(100))
        for iCPU in 1:numCPUs
            Threads.@spawn runJob(iCPU)
        end

        prevPhraseFreqsOfTopic = copy(btm.phraseFreqsOfTopic)
        simThres = btm.parameters.simThres
        for iCPU in 1:numCPUs
            segmentedOutputs = take!(jobOutputs)
            for (segID, segment) in enumerate(segmentedOutputs[1])
                segOutput = segmentedOutputs[2][segID]
                for bitermID in segment[1]:segment[2]
                    biterm = btm.biterms[bitermID]
                    topicFlag = segOutput[bitermID - segment[1] + 1]
                    newTopicID = topicFlag[1]
                    newFlag = topicFlag[2]
                    if newTopicID == biterm.topicID && newFlag == biterm.gpuFlag
                        continue
                    end
                    btm.numBitermsOfTopic[biterm.topicID] -= 1
                    btm.phraseFreqsOfTopic[biterm.topicID, biterm.pair.i] -= 1
                    btm.phraseFreqsOfTopic[biterm.topicID, biterm.pair.j] -= 1
                    if biterm.gpuFlag
                        relatedIDs = btm.similarPhraseIDs[biterm.pair]
                        btm.numRelatedOfTopic[biterm.topicID] -= length(relatedIDs)
                        for id in relatedIDs
                            btm.phraseFreqsOfTopic[biterm.topicID, id] -= simThres
                        end
                    end
                    btm.numBitermsOfTopic[newTopicID] += 1
                    btm.phraseFreqsOfTopic[newTopicID, biterm.pair.i] += 1
                    btm.phraseFreqsOfTopic[newTopicID, biterm.pair.j] += 1
                    btm.biterms[bitermID].topicID = newTopicID
                    if newFlag
                        relatedIDs = btm.similarPhraseIDs[biterm.pair]
                        btm.numRelatedOfTopic[newTopicID] += length(relatedIDs)
                        for id in relatedIDs
                            btm.phraseFreqsOfTopic[newTopicID, id] += simThres
                        end
                    end
                end
            end
        end
        deltaFreq = sum(abs.(btm.phraseFreqsOfTopic - prevPhraseFreqsOfTopic))
        println("iter $iter: deltaFreq = $deltaFreq")

        if iter % 100 == 0
            saveNBTMWE(fileName, btm)
            println("iter $iter: output saved")
        end
    end
    if maxIter % 100 != 0
        saveNBTMWE(fileName, btm)
    end
    println("final output saved")
end

numTopics = 100
maxIter = 300
simThres = 0.9
wordVectors = wordvectors("DATR/workspace/NIPS.w2v.model.txt")
nonStopWords, nonStopWordIDs = getNonStopWords(wordVectors)
vectors = normalizeVectors(wordVectors, nonStopWords)
btm = newNBTMWE("DATR/workspace/NIPS.w2v.training.txt", nonStopWords, vectors, numTopics, maxIter, simThres, 2.0, 0.005)
#saveBTM("DATR/workspace/NIPS.btm.txt", btm)
trainNBTMWE!(btm, "DATR/workspace/NIPS.nbtmwe.txt")