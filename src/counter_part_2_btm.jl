struct IDPair
    i::Int
    j::Int
end

mutable struct Biterm
    pair::IDPair
    topicID::Int
end

struct BTMParameters
    numTopics::Int
    maxIter::Int 
    alpha::Float64
    beta::Float64
    phrases::Vector{String}
    phraseIDs::Dict{String,Int}
    docs::Vector{Vector{Int}}
end

mutable struct BTM
    parameters::BTMParameters
    biterms::Vector{Biterm}
    numBitermsOfTopic::Vector{Int}
    phraseFreqsOfTopic::Array{Int,2} 
end

function saveBTM(fileName::String, btm::BTM)
    file = open(fileName, "w")
    param = btm.parameters
    numPhrases = length(param.phrases)
    numDocs = length(param.docs)
    numBiterms = length(btm.biterms)
    println(file, "$(param.numTopics) $(param.maxIter) $(param.alpha) $(param.beta) $numPhrases $numDocs $numBiterms")
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
        println(file, "$i $j $topicID")
    end
    for topicID in 1:param.numTopics
        if topicID > 1
            print(file, " ")
        end
        print(file, btm.numBitermsOfTopic[topicID])
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

function newBTM(fileName::String, numTopics::Int, maxIter::Int, alpha::Float64, beta::Float64)::BTM
    phrases = Vector{String}()
    phraseIDs = Dict{String,Int}()
    docs = Vector{Vector{Int}}()
    file = open(fileName)
    for line in readlines(file)
        phrasesAtLine = split(line)
        doc = Vector{Int}()
        for phrase in phrasesAtLine
            if phrase != ""
                if haskey(phraseIDs, phrase)
                    push!(doc, phraseIDs[phrase])
                else
                    push!(phrases, phrase)
                    phraseID = length(phrases)
                    phraseIDs[phrase] = phraseID
                    push!(doc, phraseID)
                end 
            end
        end
        if length(doc) > 0
            push!(docs, doc)
        end
    end

    numPhrases = length(phrases)
    numBitermsOfTopic = fill(0, numTopics)
    phraseFreqsOfTopic = fill(0, (numTopics, numPhrases))
    biterms = Vector{Biterm}()
    for doc in docs
        docSize = length(doc)
        for i in 1:docSize
            for j in 1:docSize
                if i != j
                    topicID = rand(1:numTopics)
                    push!(biterms, Biterm(IDPair(doc[i], doc[j]), topicID))
                    numBitermsOfTopic[topicID] += 1
                    phraseFreqsOfTopic[topicID, doc[i]] += 1
                    phraseFreqsOfTopic[topicID, doc[j]] += 1 
                end
            end
        end
    end

    BTM(BTMParameters(numTopics, maxIter, alpha, beta, phrases, phraseIDs, docs), biterms, numBitermsOfTopic, phraseFreqsOfTopic)
end

function loadBTM(fileName::String)::BTM
    file = open(fileName)
    firstLine = readline(file)
    fieldsOfFirstLine = split(firstLine)
    @assert length(fieldsOfFirstLine) == 7
    numTopics = parse(Int, fieldsOfFirstLine[1])
    maxIter = parse(Int, fieldsOfFirstLine[2])
    alpha = parse(Float64, fieldsOfFirstLine[3])
    beta = parse(Float64, fieldsOfFirstLine[4])
    numPhrases = parse(Int, fieldsOfFirstLine[5])
    numDocs = parse(Int, fieldsOfFirstLine[6])
    numBiterms = parse(Int, fieldsOfFirstLine[7])
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
        @assert length(fieldsOfBiterm) == 3
        i = parse(Int, fieldsOfBiterm[1])
        j = parse(Int, fieldsOfBiterm[2])
        topicID = parse(Int, fieldsOfBiterm[3])
        biterm = Biterm(IDPair(i, j), topicID)
        biterms[bitermID] = biterm
    end
    numBitermsOfTopic = Vector{Int}(undef, numTopics)
    lineOfNumBiterms = readline(file)
    fieldsOfNumBiterms = split(lineOfNumBiterms)
    @assert length(fieldsOfNumBiterms) == numTopics
    for topicID in 1:numTopics
        numBitermsOfTopic[topicID] = parse(Int, fieldsOfNumBiterms[topicID])
    end
    phraseFreqsOfTopic = Array{Int,2}(undef, (numTopics, numPhrases))
    for topicID in 1:numTopics
        lineOfTopic = readline(file)
        fieldsOfTopic = split(lineOfTopic)
        @assert length(fieldsOfTopic) == numPhrases
        for phraseID in 1:numPhrases
            freq = parse(Int, fieldsOfTopic[phraseID])
            phraseFreqsOfTopic[topicID, phraseID] = freq
        end
    end
    close(file)

    BTM(BTMParameters(numTopics, maxIter, alpha, beta, phrases, phraseIDs, docs), biterms, numBitermsOfTopic, phraseFreqsOfTopic)
end

function computeProbTopicBiterm(btm::BTM, biterm::Biterm)::Vector{Float64}
    numTopics = btm.parameters.numTopics
    numPhrases = length(btm.parameters.phrases)
    numBiterms = length(btm.biterms)
    alpha = btm.parameters.alpha
    beta = btm.parameters.beta
    prob = Vector{Float64}(undef, numTopics)
    for topicID in 1:numTopics
        if topicID != biterm.topicID
            w = 1.0 / (2 * btm.numBitermsOfTopic[topicID] + numPhrases * beta)
            prob1 = (btm.phraseFreqsOfTopic[topicID, biterm.pair.i] + beta) * w
            prob2 = (btm.phraseFreqsOfTopic[topicID, biterm.pair.j] + beta) * w
            probTopic = (btm.numBitermsOfTopic[topicID] + alpha) / (numBiterms - 1 + numTopics * alpha)
            prob[topicID] = prob1 * prob2 * probTopic
        else
            w = 1.0 / (2 * btm.numBitermsOfTopic[topicID] - 2 + numPhrases * btm.parameters.beta)
            prob1 = (btm.phraseFreqsOfTopic[topicID, biterm.pair.i] - 1 + btm.parameters.beta) * w 
            prob2 = (btm.phraseFreqsOfTopic[topicID, biterm.pair.j] - 1 + btm.parameters.beta) * w
            probTopic = (btm.numBitermsOfTopic[topicID] - 1 + alpha) / (numBiterms - 1 + numTopics * alpha)
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

function trainBTM!(btm::BTM, fileName::String)
    numBiterms = length(btm.biterms)
    numCPUs = length(Sys.cpu_info())
    Segments = Vector{Tuple{Int,Int}}
    for iter in 1:btm.parameters.maxIter
        jobs = Channel{Tuple{Int,Int}}(numCPUs)
        jobOutputs = Channel{Tuple{Segments,Vector{Vector{Int}}}}(numCPUs)

        function makeJobs(batchSize::Int)
            for i in 1:batchSize:numBiterms
                put!(jobs, (i, min(i + batchSize - 1, numBiterms)))
            end
        end

        function runJob(iCPU::Int)
            mySegments = Segments()
            myOutputs = Vector{Vector{Int}}()
            for job in jobs
                push!(mySegments, job)
                jobOutput = Vector{Int}(undef, job[2] - job[1] + 1)
                for bitermID in job[1]:job[2]
                    prob = computeProbTopicBiterm(btm, btm.biterms[bitermID])
                    jobOutput[bitermID - job[1] + 1] = sampleProb(prob)
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
        for iCPU in 1:numCPUs
            segmentedOutputs = take!(jobOutputs)
            for (segID, segment) in enumerate(segmentedOutputs[1])
                segOutput = segmentedOutputs[2][segID]
                for bitermID in segment[1]:segment[2]
                    biterm = btm.biterms[bitermID]
                    newTopicID = segOutput[bitermID - segment[1] + 1]
                    if newTopicID == biterm.topicID
                        continue
                    end
                    btm.numBitermsOfTopic[biterm.topicID] -= 1
                    btm.phraseFreqsOfTopic[biterm.topicID, biterm.pair.i] -= 1
                    btm.phraseFreqsOfTopic[biterm.topicID, biterm.pair.j] -= 1
                    btm.numBitermsOfTopic[newTopicID] += 1
                    btm.phraseFreqsOfTopic[newTopicID, biterm.pair.i] += 1
                    btm.phraseFreqsOfTopic[newTopicID, biterm.pair.j] += 1
                    btm.biterms[bitermID].topicID = newTopicID
                end
            end
        end
        deltaFreq = sum(abs.(btm.phraseFreqsOfTopic - prevPhraseFreqsOfTopic))
        println("iter $iter: deltaFreq = $deltaFreq")

        if iter % 100 == 0
            saveBTM(fileName, btm)
            println("iter $iter: output saved")
        end
    end
    if maxIter % 100 != 0
        saveBTM(fileName, btm)
    end
    println("final output saved")
end

numTopics = 100
maxIter = 30
btm = newBTM("DATR/workspace/NIPS.w2v.training.txt", numTopics, maxIter, 2.0, 0.005)
#saveBTM("DATR/workspace/NIPS.btm.txt", btm)
trainBTM!(btm, "DATR/workspace/NIPS.btm.txt")